# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression tests for diffusion-step understanding KV reuse."""

import inspect

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.attention.natten import NATTEN_SUPPORTED
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, dispatch_attention_with_memory
from cosmos_framework.model.generator.mot.unified_mot import (
    Nemotron3DenseVLTextForCausalLM,
    PackedAttentionMoT,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLTextForCausalLM,
    Qwen3VLTextModel,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.model.generator.utils.memory import DiffusionKVCacheMemoryState


def _tiny_text_model(device: torch.device) -> Qwen3VLTextModel:
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        attention_bias=False,
    )
    model = Qwen3VLTextModel(
        config,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=False,
    ).to(device=device)
    # RoPE inv_freq is required to stay fp32; production materialization also
    # keeps this buffer fp32 while the learned weights use bf16.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.eval()
    for module in model.modules():
        if isinstance(module, PackedAttentionMoT):
            module.dispatch_attention_fn = dispatch_attention_with_memory
    return model


def _make_pack(hidden: torch.Tensor):
    device = hidden.device
    pack, attention_meta, natten_metadata = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=["causal", "full"],
        split_lens=[3, 5],
        sample_lens=[8],
        packed_und_token_indexes=torch.tensor([0, 1, 2], device=device, dtype=torch.long),
        packed_gen_token_indexes=torch.tensor([3, 4, 5, 6, 7], device=device, dtype=torch.long),
        num_heads=4,
        head_dim=64,
        num_layers=3,
    )
    return pack, attention_meta, natten_metadata


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available() or not NATTEN_SUPPORTED, reason="CUDA attention backend required.")
def test_multilayer_diffusion_kv_cache_matches_standard_next_step(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    model = _tiny_text_model(device)

    und = torch.randn(3, 256, device=device, dtype=torch.bfloat16)
    gen_prefill = torch.randn(5, 256, device=device, dtype=torch.bfloat16)
    gen_next = torch.randn(5, 256, device=device, dtype=torch.bfloat16)
    gen_third = torch.randn(5, 256, device=device, dtype=torch.bfloat16)
    prefill_pack, attention_meta, natten_metadata = _make_pack(torch.cat([und, gen_prefill], dim=0))
    next_pack, _, _ = _make_pack(torch.cat([und, gen_next], dim=0))
    third_pack, _, _ = _make_pack(torch.cat([und, gen_third], dim=0))
    position_ids = torch.arange(8, device=device, dtype=torch.long)
    joint_memory = DiffusionKVCacheMemoryState(num_layers=3)
    split_memory = DiffusionKVCacheMemoryState(num_layers=3)

    with torch.no_grad():
        joint_prefill, _ = model(
            prefill_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=joint_memory,
        )
        standard, _ = model(
            next_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=None,
        )
        standard_third, _ = model(
            third_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=None,
        )

        # Production split prefill starts with both fused layer pathways on
        # CPU, then streams only tensors without the ``_moe_gen`` suffix.
        monkeypatch.setenv("COSMOS3_REASONER_OFFLOAD", "1")
        for layer in model.layers:
            layer.cpu()
        und_prefill, _ = model(
            prefill_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=split_memory,
            und_only=True,
        )
        assert get_gen_seq(und_prefill).numel() == 0
        for layer in model.layers:
            for name, parameter in layer.named_parameters():
                assert parameter.device.type == "cpu", name

        # The phase boundary loads GEN exactly once; UND stays on CPU from
        # step 0 onward.
        for layer in model.layers:
            for name, parameter in layer.named_parameters():
                if "_moe_gen" in name:
                    parameter.data = parameter.data.cuda()
        model.norm.cpu()
        model.embed_tokens.cpu()
        split_step0, _ = model(
            prefill_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=split_memory,
        )
        cached, _ = model(
            next_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=split_memory,
        )
        cached_third, _ = model(
            third_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=split_memory,
        )

        # A later policy chunk keeps GEN resident on CUDA while streaming the
        # same UND parameters again.  The new cache must match and prefill must
        # not move any generator tensor.
        next_chunk_memory = DiffusionKVCacheMemoryState(num_layers=3)
        next_chunk_prefill, _ = model(
            next_pack,
            attention_meta,
            position_ids,
            natten_metadata_list=natten_metadata,
            memory=next_chunk_memory,
            und_only=True,
        )
        assert get_gen_seq(next_chunk_prefill).numel() == 0
        for layer in model.layers:
            for name, parameter in layer.named_parameters():
                expected_device = "cuda" if "_moe_gen" in name else "cpu"
                assert parameter.device.type == expected_device, name

    torch.testing.assert_close(get_gen_seq(split_step0), get_gen_seq(joint_prefill), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(get_gen_seq(cached), get_gen_seq(standard), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(get_gen_seq(cached_third), get_gen_seq(standard_third), atol=2e-2, rtol=2e-2)
    joint_k = [value for value in joint_memory._und_k if value is not None]
    split_k = [value for value in split_memory._und_k if value is not None]
    assert len(joint_k) == len(split_k) == 3
    for joint_value, split_value in zip(joint_k, split_k):
        torch.testing.assert_close(split_value, joint_value, atol=2e-2, rtol=2e-2)
    next_chunk_k = [value for value in next_chunk_memory._und_k if value is not None]
    assert len(next_chunk_k) == 3
    for joint_value, next_chunk_value in zip(joint_k, next_chunk_k):
        torch.testing.assert_close(next_chunk_value, joint_value, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
def test_reasoner_generator_parameter_placement_classification():
    classify = OmniMoTModel._reasoner_generator_parameter_target
    assert classify("language_model.model.layers.0.self_attn.q_proj.weight") == "cpu"
    assert classify("language_model.model.layers.0.self_attn.q_proj_moe_gen.weight") == "cuda"
    assert classify("language_model.model.layers.0.mlp.down_proj.weight") == "cpu"
    assert classify("language_model.model.layers.0.mlp_moe_gen.down_proj.weight") == "cuda"
    assert classify("language_model.model.embed_tokens.weight") == "cpu"
    assert classify("language_model.model.norm.weight") == "cpu"
    # The final gen norm is a tiny bridge required by the full prefill pass;
    # unlike the large per-layer generator tower, it stays resident.
    assert classify("language_model.model.norm_moe_gen.weight") is None
    assert classify("language_model.lm_head.weight") == "cpu"
    assert classify("language_model.model.rotary_emb.inv_freq") is None
    assert classify("action2llm.linears.0.weight") is None


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for initial tower placement checks.")
def test_initial_reasoner_offload_places_generator_on_cuda(monkeypatch: pytest.MonkeyPatch):
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_proj_moe_gen = torch.nn.Linear(4, 4, bias=False)

    net = torch.nn.Module()
    net.language_model = torch.nn.Module()
    net.language_model.model = torch.nn.Module()
    net.language_model.model.layers = torch.nn.ModuleList([TinyLayer()])
    net.language_model.model.embed_tokens = torch.nn.Embedding(8, 4)
    net.language_model.model.norm = torch.nn.LayerNorm(4)
    net.language_model.model.norm_moe_gen = torch.nn.LayerNorm(4)
    net.language_model.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = OmniMoTModel.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    monkeypatch.setenv("COSMOS3_REASONER_OFFLOAD", "1")

    def fail_if_storage_is_copied(*args, **kwargs):
        raise AssertionError("allocate_empty=True must not copy cold-start tensor storage with Tensor.to()")

    monkeypatch.setattr(torch.Tensor, "to", fail_if_storage_is_copied)
    model._place_initial_offload_towers(net, allocate_empty=True, log_result=False)

    for name, parameter in net.named_parameters():
        placement = model._reasoner_generator_parameter_target(name)
        if placement == "cuda":
            assert parameter.device.type == "cuda", name
        elif placement == "cpu":
            assert parameter.device.type == "cpu", name
        else:
            assert parameter.device.type == "cuda", name


@pytest.mark.L0
def test_causal_lm_wrappers_accept_und_only():
    for wrapper in (Qwen3VLTextForCausalLM, Qwen3VLMoeTextForCausalLM, Nemotron3DenseVLTextForCausalLM):
        assert "und_only" in inspect.signature(wrapper.forward).parameters


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tower placement checks.")
def test_policy_chunk_switches_generator_and_reasoner_placement(monkeypatch: pytest.MonkeyPatch):
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_proj_moe_gen = torch.nn.Linear(4, 4, bias=False)

    net = torch.nn.Module()
    net.language_model = torch.nn.Module()
    net.language_model.model = torch.nn.Module()
    net.language_model.model.layers = torch.nn.ModuleList([TinyLayer()])
    net.language_model.model.embed_tokens = torch.nn.Embedding(8, 4)
    net.language_model.model.norm = torch.nn.LayerNorm(4)
    net.language_model.model.norm_moe_gen = torch.nn.LayerNorm(4)
    net.language_model.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = OmniMoTModel.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._reasoner_offload_complete = True
    monkeypatch.setenv("COSMOS3_REASONER_OFFLOAD", "1")

    # Recreate the end-of-chunk state: all UND tensors are on CPU while GEN
    # and unclassified non-layer generation helpers remain on CUDA.
    for name, parameter in net.named_parameters():
        placement = model._reasoner_generator_parameter_target(name)
        if placement == "cuda" or placement is None:
            parameter.data = parameter.data.cuda()

    model._prepare_reasoner_prefill(net)
    assert not model._reasoner_offload_complete
    for name, parameter in net.named_parameters():
        placement = model._reasoner_generator_parameter_target(name)
        if placement == "cuda":
            assert parameter.device.type == "cuda", name
        elif placement == "cpu" and not name.startswith("language_model.model.layers."):
            assert parameter.device.type == "cuda", name

    model._offload_reasoner_and_pin_generator(net)
    assert model._reasoner_offload_complete
    for name, parameter in net.named_parameters():
        placement = model._reasoner_generator_parameter_target(name)
        if placement == "cuda":
            assert parameter.device.type == "cuda", name
        elif placement == "cpu":
            assert parameter.device.type == "cpu", name


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to allocate pinned host memory.")
def test_reasoner_offload_can_pin_und_cpu_master_tensors(monkeypatch: pytest.MonkeyPatch):
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_proj_moe_gen = torch.nn.Linear(4, 4, bias=False)

    net = torch.nn.Module()
    net.language_model = torch.nn.Module()
    net.language_model.model = torch.nn.Module()
    net.language_model.model.layers = torch.nn.ModuleList([TinyLayer()])
    net.language_model.model.embed_tokens = torch.nn.Embedding(8, 4)
    net.language_model.model.norm = torch.nn.LayerNorm(4)
    net.language_model.model.norm_moe_gen = torch.nn.LayerNorm(4)
    net.language_model.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = OmniMoTModel.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    monkeypatch.setenv("COSMOS3_REASONER_OFFLOAD", "1")
    monkeypatch.setenv("COSMOS3_REASONER_PIN_CPU_MASTER", "1")

    net.language_model.model.layers[0].q_proj_moe_gen.weight.data = (
        net.language_model.model.layers[0].q_proj_moe_gen.weight.data.cuda()
    )
    net.language_model.model.embed_tokens.weight.data = net.language_model.model.embed_tokens.weight.data.cuda()

    model._maybe_pin_reasoner_cpu_master_tensors(net)

    assert net.language_model.model.layers[0].q_proj.weight.device.type == "cpu"
    assert net.language_model.model.layers[0].q_proj.weight.is_pinned()
    assert net.language_model.model.layers[0].q_proj_moe_gen.weight.device.type == "cuda"
    assert net.language_model.model.embed_tokens.weight.device.type == "cuda"
