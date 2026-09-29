import torch
import pytest

from piper_cosmos.cosmos3.nano_cache_strategies import (
    fresh_indices,
    nano_concat_view_regions,
    summarize_asi_mask_history,
    summarize_execution_masks_by_view,
)
from piper_cosmos.cosmos3.nano_version1_sparse import build_core_stable_plan, selected_gen_positions


def test_asi_core80_stable104_real_nano_grid():
    torch.manual_seed(5)
    records = [dict(block=b, branch="conditional", profiles=torch.rand(8, 360)) for b in range(36)]
    plan = build_core_stable_plan(torch=torch, profile_records=records, core_token_budget=80, stable_token_budget=104)
    assert plan["core_masks"].sum(-1).tolist() == [80] * 8
    assert plan["stable_mask"].sum() == 104
    assert plan["execution_mask"].sum(-1).tolist() == [184] * 8
    assert not (plan["core_masks"] & plan["stable_mask"]).any()
    layout = dict(latent_positions={f"L{f}": list(range(f*360, (f+1)*360)) for f in range(9)},
                  action_positions=list(range(3240, 3273)))
    ids = selected_gen_positions(torch, layout, plan["execution_mask"], "cpu")
    assert ids.numel() == 1865
    assert ids[:360].tolist() == layout["latent_positions"]["L0"]
    assert ids[-33:].tolist() == layout["action_positions"]
    assert torch.all(ids[1:] > ids[:-1])


def test_asi_all_layer_core_changes_only_aggregation_source():
    torch.manual_seed(17)
    records = [dict(block=b, branch="conditional", profiles=torch.rand(8, 360)) for b in range(36)]
    informative = build_core_stable_plan(
        torch=torch,
        profile_records=records,
        core_token_budget=80,
        stable_token_budget=104,
        core_layer_mode="informative",
    )
    all_layers = build_core_stable_plan(
        torch=torch,
        profile_records=records,
        core_token_budget=80,
        stable_token_budget=104,
        core_layer_mode="all",
    )
    assert informative["core_layer_mode"] == "informative"
    assert informative["core_blocks"].numel() == 6
    assert all_layers["core_layer_mode"] == "all"
    assert all_layers["core_blocks"].tolist() == list(range(36))
    assert torch.equal(informative["stable_scores"], all_layers["stable_scores"])
    assert all_layers["execution_mask"].sum(-1).tolist() == [184] * 8


def test_step_refresh_mask_similarity_reports_per_frame_overlap():
    first_core = torch.zeros(8, 12, dtype=torch.bool)
    first_core[:, :4] = True
    second_core = torch.zeros_like(first_core)
    second_core[:, 2:6] = True
    first_stable = torch.zeros(12, dtype=torch.bool)
    first_stable[6:10] = True
    second_stable = first_stable.clone()
    history = [
        {
            "step": 0,
            "core_masks": first_core,
            "stable_mask": first_stable,
            "execution_mask": first_core | first_stable,
        },
        {
            "step": 1,
            "core_masks": second_core,
            "stable_mask": second_stable,
            "execution_mask": second_core | second_stable,
        },
    ]
    report = summarize_asi_mask_history(history)
    pair = report["consecutive_pairs"][0]
    anchor_pair = report["step0_anchor_pairs"][0]
    assert pair["step_from"] == 0 and pair["step_to"] == 1
    assert pair["core"]["intersection"] == [2] * 8
    assert pair["core"]["jaccard_mean"] == pytest.approx(2 / 6)
    assert pair["core"]["retention_mean"] == pytest.approx(0.5)
    assert pair["stable"]["jaccard_mean"] == 1.0
    assert pair["execution"]["intersection"] == [6] * 8
    assert pair["execution"]["retention_mean"] == pytest.approx(0.75)
    assert anchor_pair == pair
    assert report["consecutive_aggregate"]["execution"]["retention_mean"] == pytest.approx(0.75)
    assert report["step0_anchor_aggregate"]["execution"]["retention_mean"] == pytest.approx(0.75)


def test_nano_three_view_partition_and_directional_retention():
    regions = nano_concat_view_regions([9, 20, 18])
    assert {name: int(mask.sum()) for name, mask in regions.items()} == {
        "cam_high": 234,
        "cam_left_wrist": 63,
        "cam_right_wrist": 63,
    }
    first = torch.zeros(8, 360, dtype=torch.bool)
    second = torch.zeros_like(first)
    first[:, :10] = True
    second[:, 5:15] = True
    history = [
        {"step": 0, "execution_mask": first},
        {"step": 1, "execution_mask": second},
    ]
    report = summarize_execution_masks_by_view(history, [9, 20, 18])
    high = report["consecutive_pairs"][0]["views"]["cam_high"]
    assert high["selected_from_mean"] == 10
    assert high["intersection_mean"] == 5
    assert high["retention_from_mean"] == pytest.approx(0.5)


def test_k184_ablation_plans_have_identical_execution_budget():
    torch.manual_seed(7)
    records = [dict(block=b, branch="conditional", profiles=torch.rand(8, 360)) for b in range(36)]
    core = build_core_stable_plan(
        torch=torch, profile_records=records, core_token_budget=184, stable_token_budget=0
    )
    stable = build_core_stable_plan(
        torch=torch, profile_records=records, core_token_budget=0, stable_token_budget=184
    )
    assert core["core_masks"].sum(-1).tolist() == [184] * 8
    assert int(core["stable_mask"].sum()) == 0
    assert stable["core_masks"].sum(-1).tolist() == [0] * 8
    assert int(stable["stable_mask"].sum()) == 184
    assert core["execution_mask"].sum(-1).tolist() == [184] * 8
    assert stable["execution_mask"].sum(-1).tolist() == [184] * 8
    assert torch.equal(stable["execution_mask"][0], stable["execution_mask"][7])


def test_toca_dynamic_depth_and_independent_ages():
    score = torch.arange(2880).float()
    age = torch.zeros_like(score)
    first, next_age = fresh_indices(score, age, 0, 36, 2880)
    last, _ = fresh_indices(score, age, 35, 36, 2880)
    assert first.numel() == 1080
    assert last.numel() == 360
    assert torch.all(next_age[first] == 0)
    assert torch.all(age == 0)
    assert next_age.sum() == 1800


def test_worldcache_patchify_roundtrip_and_history():
    from piper_cosmos.cosmos3.nano_cache_vendor.worldcache import patchify, unpatchify, curvature_and_slopes
    x = torch.randn(4, 8, 6, 8)
    assert torch.equal(unpatchify(patchify(x, 2), x.shape, 2), x)
    z = torch.randn(16, 8)
    c, curr, prev = curvature_and_slopes([z, z+1, z+2], [0, 1, 2], 1e-8)
    torch.testing.assert_close(curr, prev)
    assert c.max() < 1e-5
