# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.utils.memory import DiffusionKVCacheMemoryState


@pytest.mark.L0
def test_diffusion_kv_cache_memory_state_prefill_then_gen_only():
    state = DiffusionKVCacheMemoryState(num_layers=2)
    state.init({}, torch.device("cpu"))

    assert not state.is_gen_only()
    assert state.read_for_layer(0).und_k is None
    assert state.read_for_layer(0).frame_idx == 0

    kv0 = tuple(torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(4))
    kv1 = tuple(torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(4))
    state.write_for_layer(0, kv0)
    assert not state.is_gen_only()
    state.write_for_layer(1, kv1)

    assert state.is_gen_only()
    value = state.read_for_layer(0)
    assert value.frame_idx == 1
    assert value.und_k is not None
    assert value.und_v is not None
    assert not value.und_k.requires_grad
    assert not value.und_v.requires_grad

    state.reset()
    assert not state.is_gen_only()
