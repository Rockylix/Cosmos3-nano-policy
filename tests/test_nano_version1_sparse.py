from __future__ import annotations

import torch

from piper_cosmos.cosmos3.nano_version1_sparse import (
    CORE_TOKEN_BUDGET,
    STABLE_TOKEN_BUDGET,
    TOKEN_BUDGET,
    build_core_stable_plan,
    selected_gen_positions,
)


def test_core_stable_plan_has_exact_disjoint_budget() -> None:
    records = []
    for block in range(36):
        profiles = torch.arange(1, 341, dtype=torch.float32).repeat(8, 1)
        profiles = profiles.roll(block, dims=-1) * float(block + 1)
        records.append({"branch": "conditional", "block": block, "profiles": profiles})
    plan = build_core_stable_plan(torch=torch, profile_records=records)
    execution = plan["execution_mask"]
    assert tuple(execution.shape) == (8, 340)
    assert execution.sum(dim=-1).tolist() == [TOKEN_BUDGET] * 8
    assert plan["core_masks"].sum(dim=-1).tolist() == [CORE_TOKEN_BUDGET] * 8
    assert int(plan["stable_mask"].sum()) == STABLE_TOKEN_BUDGET
    assert not bool((plan["core_masks"].any(dim=0) & plan["stable_mask"]).any())


def test_selected_positions_keep_l0_action_and_k184_per_future() -> None:
    spatial = 340
    layout = {
        "latent_positions": {
            f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
        },
        "action_positions": list(range(9 * spatial, 9 * spatial + 33)),
    }
    mask = torch.zeros((8, spatial), dtype=torch.bool)
    for frame in range(8):
        mask[frame, frame : frame + TOKEN_BUDGET] = True
    selected = selected_gen_positions(torch, layout, mask, torch.device("cpu"))
    assert selected[:spatial].tolist() == layout["latent_positions"]["L0"]
    assert selected[-33:].tolist() == layout["action_positions"]
    assert int(selected.numel()) == spatial + 8 * TOKEN_BUDGET + 33
