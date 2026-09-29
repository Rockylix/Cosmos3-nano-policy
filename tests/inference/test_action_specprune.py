from __future__ import annotations

import torch

from cosmos_framework.inference.action_specprune import (
    ASI_K184_AVERAGE_FUTURE_RETENTION,
    NANO_SPATIAL_TOKENS,
    ExitHidden,
    NanoSpecPruneConfig,
    ObservationPlan,
    observation_grid_patches,
    resolve_instruction_indices,
    top_mask,
)


def test_budget_and_scaled_block_schedule() -> None:
    config = NanoSpecPruneConfig()
    assert config.local_k == 58
    assert config.global_k == 72
    assert config.min_observation_tokens == 184
    assert config.global_blocks == (17, 35)
    assert config.update_blocks == (18, 25, 31)
    assert config.prune_blocks == (13, 19, 26, 32)
    assert abs(ASI_K184_AVERAGE_FUTURE_RETENTION - 0.5722222222222222) < 1e-12


def test_top_mask_is_stable_and_candidate_bounded() -> None:
    score = torch.tensor([1.0, 1.0, 0.5, 2.0])
    candidates = torch.tensor([True, True, False, True])
    mask = top_mask(score, 2, candidates)
    assert mask.tolist() == [True, False, False, True]


def test_instruction_indices_fall_back_for_slow_tokenizer() -> None:
    class SlowTokenizer:
        all_special_ids = [99]

        def decode(self, token_ids, **_kwargs):
            mapping = {
                (): '',
                (10,): 'prefix ',
                (10, 20): 'prefix Assemble ',
                (10, 20, 30): 'prefix Assemble battery.',
                (10, 20, 30, 40): 'prefix Assemble battery. suffix',
            }
            return mapping[tuple(token_ids)]

        def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
            assert not add_special_tokens
            if return_offsets_mapping:
                raise NotImplementedError
            mapping = {
                'Assemble battery.': [20, 30],
                ' Assemble battery.': [20, 30],
            }
            return {'input_ids': mapping[text]}

    assert resolve_instruction_indices(SlowTokenizer(), [10, 20, 30, 40], 'Assemble battery.') == [1, 2]


def test_observation_grid_descriptors_cover_all_positions() -> None:
    rgb = torch.arange(3 * 40 * 36, dtype=torch.float32).reshape(3, 40, 36)
    patches = observation_grid_patches(rgb, (20, 18))
    assert tuple(patches.shape) == (NANO_SPATIAL_TOKENS, 48)
    assert torch.isfinite(patches).all()


def test_plan_masks_are_nested_and_history_is_observation_based() -> None:
    config = NanoSpecPruneConfig()
    plan = ObservationPlan(config)
    observation = torch.arange(360 * 4, dtype=torch.float32).reshape(360, 4) + 1
    plan.begin(observation, nonvisual_tokens=120)
    score0 = torch.linspace(0, 1, 360)
    score1 = torch.linspace(1, 0, 360)
    plan.local(0, score0)
    plan.local(1, score1)
    plan.prune(13)
    plan.update(18, torch.rand(32, 360))
    plan.prune(19)
    plan.update(25, torch.rand(32, 360))
    plan.prune(26)
    plan.update(31, torch.rand(32, 360))
    plan.prune(32)
    previous = None
    for mask in plan.masks.values():
        if previous is not None:
            assert not bool((mask & ~previous).any())
        previous = mask
    plan.global_scores = {17: score0.clone(), 35: score1.clone()}
    plan.finish()

    changed = observation.clone()
    changed[7].zero_()
    plan.begin(changed, nonvisual_tokens=120)
    assert plan.similarity is not None
    assert bool(plan.dynamic_mask[7])
    assert int(plan.global_mask.sum()) > 0
    plan.abort()


def test_exit_hidden_restores_each_row_once() -> None:
    reference = torch.zeros(8, 3)
    stash = ExitHidden(reference)
    first = torch.tensor([0, 2, 4, 6])
    second = torch.tensor([1, 3, 5, 7])
    stash.put(first, torch.ones(4, 3))
    stash.put(second, torch.full((4, 3), 2.0))
    restored = stash.finish()
    assert torch.equal(restored[first], torch.ones(4, 3))
    assert torch.equal(restored[second], torch.full((4, 3), 2.0))


def test_exit_hidden_rejects_double_write() -> None:
    stash = ExitHidden(torch.zeros(2, 1))
    stash.put(torch.tensor([0]), torch.ones(1, 1))
    try:
        stash.put(torch.tensor([0]), torch.ones(1, 1))
    except AssertionError:
        pass
    else:
        raise AssertionError("ExitHidden accepted a duplicate row")
