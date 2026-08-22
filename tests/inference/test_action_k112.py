# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.inference.action_k112 import (
    ACTION_HORIZON_WEIGHTS,
    NANO_BLOCK_COUNT,
    NANO_HIDDEN_SIZE,
    NANO_KV_HEADS,
    NANO_QUERY_HEADS,
    TOKEN_BUDGET,
    NanoK112Controller,
    build_shared_motion_mask,
    expand_token_mask,
    selected_gen_positions,
)


class FakeAttention:
    def __init__(self, *, hidden_size: int = NANO_HIDDEN_SIZE) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = NANO_QUERY_HEADS
        self.num_key_value_heads = NANO_KV_HEADS
        self._nano_k112_attention_capture_callback = None


class FakeDecoder:
    def __init__(self, *, hidden_size: int = NANO_HIDDEN_SIZE) -> None:
        self.self_attn = FakeAttention(hidden_size=hidden_size)
        self.calls = 0

    def __call__(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return hidden_states, {}, None


def make_net(*, blocks: int = NANO_BLOCK_COUNT, hidden_size: int = NANO_HIDDEN_SIZE):
    layers = [FakeDecoder(hidden_size=hidden_size) for _ in range(blocks)]
    model = SimpleNamespace(layers=layers)
    language_model = SimpleNamespace(model=model)
    net = SimpleNamespace(language_model=language_model)
    return net, layers


def test_nano_k112_constants() -> None:
    assert TOKEN_BUDGET == 112
    assert NANO_BLOCK_COUNT == 36
    assert ACTION_HORIZON_WEIGHTS == pytest.approx((1 / 6, 1 / 3, 1 / 3, 1 / 6))


def test_mask_profiles_all_36_nano_blocks_and_uses_six_of_eight_support() -> None:
    spatial = 12
    stable_blocks = (3, 11, 24, 34)
    records = []
    for block in range(NANO_BLOCK_COUNT):
        if block not in stable_blocks:
            profiles = torch.full((8, spatial), 1.0 / spatial)
        else:
            profiles = torch.full((8, spatial), 1e-4)
            profiles[:, 0] = 0.50
            profiles[:, 1] = 0.30
            profiles[:, 2] = 0.20
            profiles[:6, 3] = 0.15
            profiles[:5, 4] = 0.40
        records.append({"branch": "conditional", "block": block, "profiles": profiles})

    plan = build_shared_motion_mask(
        profile_records=records,
        expected_blocks=range(NANO_BLOCK_COUNT),
        token_budget=4,
        profile_block_count=4,
    )

    assert set(plan["selected_blocks"].tolist()) == set(stable_blocks)
    assert torch.nonzero(plan["shared_mask"], as_tuple=False).flatten().tolist() == [0, 1, 2, 3]
    assert float(plan["shared_score"][3]) > float(plan["shared_score"][4])


def test_shared_mask_keeps_l0_actions_and_same_future_coordinates() -> None:
    spatial = 5
    cursor = 0
    latent_positions = {}
    for latent in range(9):
        latent_positions[f"L{latent}"] = list(range(cursor, cursor + spatial))
        cursor += spatial
    action_positions = list(range(cursor, cursor + 3))
    layout = {"latent_positions": latent_positions, "action_positions": action_positions}
    mask = torch.tensor([False, True, False, True, False])

    selected = selected_gen_positions(layout, mask, torch.device("cpu"))
    selected_set = set(selected.tolist())

    assert selected[:spatial].tolist() == latent_positions["L0"]
    assert selected[-3:].tolist() == action_positions
    for latent in range(1, 9):
        frame = latent_positions[f"L{latent}"]
        assert frame[1] in selected_set
        assert frame[3] in selected_set
        assert frame[0] not in selected_set


def test_velocity_mask_expands_patch_grid() -> None:
    mask = torch.tensor([True, False, False, True])
    expanded = expand_token_mask(mask, token_grid=(2, 2), output_grid=(3, 3))
    assert expanded.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_controller_rejects_edge_geometry_and_wrong_layer_count() -> None:
    edge_net, _ = make_net(blocks=28, hidden_size=2048)
    with pytest.raises(ValueError, match="expects 36 blocks"):
        NanoK112Controller(net=edge_net, guidance=3.0, num_steps=4)

    wrong_hidden_net, _ = make_net(hidden_size=2048)
    with pytest.raises(ValueError, match="expected geometry"):
        NanoK112Controller(net=wrong_hidden_net, guidance=3.0, num_steps=4)


def test_sparse_stack_packs_once_runs_36_blocks_and_restores() -> None:
    net, decoders = make_net()
    controller = NanoK112Controller(net=net, guidance=3.0, num_steps=4)

    spatial = 114
    cursor = 0
    latent_positions = {}
    for latent in range(9):
        latent_positions[f"L{latent}"] = list(range(cursor, cursor + spatial))
        cursor += spatial
    action_positions = list(range(cursor, cursor + 5))
    cursor += len(action_positions)
    controller._layout = {
        "num_gen_tokens": cursor,
        "latent_positions": latent_positions,
        "action_positions": action_positions,
    }
    controller.shared_mask = torch.zeros(spatial, dtype=torch.bool)
    controller.shared_mask[:TOKEN_BUDGET] = True
    controller._current = {"step": 1, "branch": "conditional", "timestep": 833.0}

    from cosmos_framework.inference.action_k112 import _make_sequence_pack

    hidden = _make_sequence_pack(und_seq=torch.zeros(3, 4), gen_seq=torch.zeros(cursor, 4))
    rope = (
        _make_sequence_pack(und_seq=torch.zeros(3, 4), gen_seq=torch.zeros(cursor, 4)),
        _make_sequence_pack(und_seq=torch.zeros(3, 4), gen_seq=torch.zeros(cursor, 4)),
    )
    controller.begin_stack(
        hidden_states=hidden,
        position_embeddings=rope,
        memory_gen_only=False,
        natten_metadata_list=None,
    )
    current = hidden
    expected_sparse = spatial + 8 * TOKEN_BUDGET + len(action_positions)
    for block, decoder in enumerate(decoders):
        current, _, _ = controller.run_layer(
            block=block,
            decoder_layer=decoder,
            hidden_states=current,
            attention_mask=None,
            memory_value=None,
            gen_only=False,
        )
        assert int(get_gen_seq(current).shape[0]) == expected_sparse
    restored = controller.end_stack(current)

    assert sum(decoder.calls for decoder in decoders) == NANO_BLOCK_COUNT
    assert controller._pack_count == 1
    assert controller._restore_count == 1
    assert int(get_gen_seq(restored).shape[0]) == cursor
