"""Core64 + Stable120 sparse GEN inference for the offloaded Cosmos3 Nano.

The controller profiles the dense step-0 conditional GEN stack, builds one
fixed per-future-frame K184 mask, and shortens the remaining seven CFG stacks.
UND/reasoner prefill and its cached K/V remain owned by Cosmos Framework's
existing ``COSMOS3_REASONER_OFFLOAD`` implementation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from piper_cosmos.cosmos3.k112_sparse import (
    ACTION_HORIZON_WEIGHTS,
    K112Controller,
    _from_und_gen_splits,
    _get_gen_seq,
    _get_und_seq,
)

STRATEGY_VERSION = "nano-version1-core64-stable120-k184-live-l0-no-velocity-cache"
NANO_BLOCK_COUNT = 36
CORE_TOKEN_BUDGET = 64
STABLE_TOKEN_BUDGET = 120
TOKEN_BUDGET = CORE_TOKEN_BUDGET + STABLE_TOKEN_BUDGET
CORE_BLOCK_COUNT = 6
STABLE_CV_PENALTY = 1.0


def _select_mask(torch: Any, scores: Any, count: int, allowed: Any | None = None) -> Any:
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got {tuple(scores.shape)}")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    candidates = (
        torch.arange(scores.numel(), device=scores.device)
        if allowed is None
        else torch.nonzero(allowed, as_tuple=False).flatten()
    )
    if count < 0 or count > int(candidates.numel()):
        raise ValueError(f"Cannot select {count} tokens from {int(candidates.numel())} candidates")
    if count:
        chosen = torch.topk(scores.index_select(0, candidates), count, sorted=False).indices
        mask[candidates.index_select(0, chosen)] = True
    return mask


def build_core_stable_plan(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
    expected_blocks: Sequence[int] = tuple(range(NANO_BLOCK_COUNT)),
    core_block_count: int = CORE_BLOCK_COUNT,
    core_token_budget: int = CORE_TOKEN_BUDGET,
    stable_token_budget: int = STABLE_TOKEN_BUDGET,
    stable_cv_penalty: float = STABLE_CV_PENALTY,
    core_layer_mode: str = "informative",
) -> dict[str, Any]:
    """Build exact per-frame Core64 plus cross-frame Stable120 masks."""

    blocks = tuple(int(block) for block in expected_blocks)
    lookup: dict[int, Any] = {}
    for record in profile_records:
        if str(record.get("branch")) != "conditional":
            raise ValueError("Nano Version1 accepts conditional profiles only")
        block = int(record["block"])
        if block in lookup:
            raise ValueError(f"Duplicate profile for B{block}")
        lookup[block] = record["profiles"]
    missing = [block for block in blocks if block not in lookup]
    extra = sorted(set(lookup) - set(blocks))
    if missing or extra:
        raise ValueError(f"Profile mismatch: missing={missing}, extra={extra}")

    raw = torch.stack([lookup[block] for block in blocks])
    if raw.ndim != 3 or int(raw.shape[1]) != 8:
        raise ValueError(f"Expected [blocks,8,spatial], got {tuple(raw.shape)}")
    block_count, frame_count, spatial = map(int, raw.shape)
    if not 0 < core_block_count <= block_count:
        raise ValueError("core_block_count must fit the block set")
    if core_token_budget + stable_token_budget > spatial:
        raise ValueError("Core and Stable budgets exceed the spatial grid")
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise RuntimeError("Profile contains NaN/Inf or negative values")

    eps = torch.finfo(raw.dtype).eps
    frame_mass = raw.sum(dim=-1)
    probability = raw / frame_mass.unsqueeze(-1).clamp_min(eps)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=-1) / math.log(spatial)
    entropy = torch.where(frame_mass > eps, entropy, torch.ones_like(entropy))
    block_mass = frame_mass.mean(dim=1)
    block_entropy = entropy.mean(dim=1)
    block_quality = block_mass * (1.0 - block_entropy).clamp_min(0)

    if core_layer_mode == "informative":
        core_local = torch.topk(block_quality, core_block_count).indices
    elif core_layer_mode == "all":
        core_local = torch.arange(block_count, device=raw.device)
    else:
        raise ValueError(f"Unsupported Core layer mode: {core_layer_mode}")
    core_quality = block_quality.index_select(0, core_local)
    core_weights = (
        torch.full_like(core_quality, 1.0 / int(core_local.numel()))
        if float(core_quality.sum()) <= float(eps)
        else core_quality / core_quality.sum()
    )
    core_scores = (raw.index_select(0, core_local) * core_weights[:, None, None]).sum(dim=0)

    global_weights = block_quality
    if float(global_weights.sum()) <= float(eps):
        global_weights = torch.ones_like(global_weights)
    global_weights = global_weights / global_weights.sum()
    global_scores = (raw * global_weights[:, None, None]).sum(dim=0)
    stable_mean = global_scores.mean(dim=0)
    stable_std = global_scores.std(dim=0, unbiased=False)
    stable_cv = stable_std / stable_mean.clamp_min(eps)
    stable_scores = stable_mean / (1.0 + float(stable_cv_penalty) * stable_cv)

    # Stable is selected outside the union of frame-local Core coordinates.
    core_pool = _select_mask(torch, core_scores.amax(dim=0), spatial - stable_token_budget)
    core_masks = torch.stack(
        [_select_mask(torch, core_scores[frame], core_token_budget, core_pool) for frame in range(frame_count)]
    )
    core_union = core_masks.any(dim=0)
    stable_mask = _select_mask(torch, stable_scores, stable_token_budget, ~core_union)
    execution_mask = core_masks | stable_mask.unsqueeze(0)
    if execution_mask.sum(dim=-1).tolist() != [TOKEN_BUDGET] * frame_count:
        raise RuntimeError("Version1 did not produce exact K184 masks")

    block_tensor = torch.tensor(blocks, dtype=torch.long, device=raw.device)
    return {
        "raw_profiles": raw,
        "frame_mass": frame_mass,
        "frame_entropy": entropy,
        "block_mass": block_mass,
        "block_entropy": block_entropy,
        "block_quality": block_quality,
        "core_layer_mode": core_layer_mode,
        "core_blocks": block_tensor.index_select(0, core_local),
        "core_weights": core_weights,
        "core_scores": core_scores,
        "core_masks": core_masks,
        "stable_mean": stable_mean,
        "stable_std": stable_std,
        "stable_cv": stable_cv,
        "stable_scores": stable_scores,
        "stable_mask": stable_mask,
        "execution_mask": execution_mask,
    }


def selected_gen_positions(torch: Any, token_layout: Mapping[str, Any], execution_mask: Any, device: Any) -> Any:
    spatial = len(token_layout["latent_positions"]["L1"])
    if tuple(execution_mask.shape) != (8, spatial):
        raise ValueError(f"Expected execution mask [8,{spatial}], got {tuple(execution_mask.shape)}")
    parts = [torch.tensor(token_layout["latent_positions"]["L0"], dtype=torch.long, device=device)]
    for latent in range(1, 9):
        frame = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=device)
        chosen = torch.nonzero(execution_mask[latent - 1].to(device), as_tuple=False).flatten()
        parts.append(frame.index_select(0, chosen))
    parts.append(torch.tensor(token_layout["action_positions"], dtype=torch.long, device=device))
    return torch.cat(parts).sort().values


class NanoVersion1Controller(K112Controller):
    """One dense profile stack followed by seven fixed-mask sparse stacks."""

    def __init__(self, *, torch: Any, net: Any, guidance: float, num_steps: int, output_dir: Path | None = None) -> None:
        super().__init__(
            torch=torch,
            net=net,
            guidance=guidance,
            num_steps=num_steps,
            output_dir=output_dir,
            expected_blocks=range(NANO_BLOCK_COUNT),
        )
        self.execution_mask: Any | None = None

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        del module, args, kwargs, output
        try:
            if self.stack_active:
                raise RuntimeError("Nano Version1 Transformer stack was not restored")
        finally:
            self._current = None

    def _prepare_selected_positions(self, device: Any) -> Any:
        if self._selected_positions is not None:
            if device != self._selected_device:
                raise RuntimeError("GEN token device changed during a request")
            return self._selected_positions
        if self._layout is None or self.execution_mask is None:
            raise RuntimeError("Version1 execution mask is not finalized")
        self._selected_positions = selected_gen_positions(self.torch, self._layout, self.execution_mask, device)
        self._selected_device = device
        return self._selected_positions

    def end_stack(self, hidden_states: Any) -> Any:
        if not self.stack_active or self._original_pack is None or self._current is None:
            raise RuntimeError("Nano Version1 end_stack has no active stack")
        dense_profile = int(self._current["step"]) == 0 and str(self._current["branch"]) == "conditional"
        if dense_profile:
            restored = hidden_states
            self.plan = build_core_stable_plan(
                torch=self.torch,
                profile_records=self._profile_records,
                expected_blocks=self.expected_blocks,
            )
            self.execution_mask = self.plan["execution_mask"]
            self.shared_mask = self.execution_mask
        else:
            if self._side_buffer is None or self._active_positions is None:
                raise RuntimeError("Nano Version1 sparse restoration state is missing")
            restored_gen = self._side_buffer.clone()
            restored_gen.index_copy_(0, self._active_positions, _get_gen_seq(hidden_states))
            restored = _from_und_gen_splits(_get_und_seq(hidden_states), restored_gen, self._original_pack)
            self._restore_count += 1
        self._completed_stacks += 1
        self.abort_stack()
        return restored

    def finish(self) -> dict[str, Any]:
        if self.stack_active or self.plan is None or self.execution_mask is None or self._layout is None:
            raise RuntimeError("Nano Version1 request did not finish cleanly")
        expected_stacks = self.num_steps * 2
        expected_blocks = expected_stacks * len(self.layers)
        if self._completed_stacks != expected_stacks or self._block_calls != expected_blocks:
            raise RuntimeError("Nano Version1 stack/block accounting is incomplete")
        if self._pack_count != expected_stacks - 1 or self._restore_count != expected_stacks - 1:
            raise RuntimeError("Nano Version1 pack/restore accounting is incomplete")

        full_gen = int(self._layout["num_gen_tokens"])
        spatial = int(self.execution_mask.shape[-1])
        sparse_gen = full_gen - 8 * spatial + int(self.execution_mask.sum())
        summary = {
            "schema_version": 1,
            "strategy_version": STRATEGY_VERSION,
            "profile": f"step0 conditional B0-B{len(self.layers) - 1}",
            "action_horizon_weights": list(ACTION_HORIZON_WEIGHTS),
            "block_score": "R_l * (1 - H_l)",
            "selected_core_blocks": [int(value) for value in self.plan["core_blocks"].cpu().tolist()],
            "core_tokens_per_future": CORE_TOKEN_BUDGET,
            "stable_tokens_per_future": STABLE_TOKEN_BUDGET,
            "future_tokens_per_frame": TOKEN_BUDGET,
            "full_gen_tokens": full_gen,
            "sparse_gen_tokens": sparse_gen,
            "saved_gen_tokens_per_sparse_block": full_gen - sparse_gen,
            "sparse_gen_retained_ratio": sparse_gen / full_gen,
            "dense_stack_count": 1,
            "sparse_stack_count": expected_stacks - 1,
            "velocity_cache_enabled": False,
            "reasoner_offload_owned_by_framework": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact = {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in self.plan.items()}
            self.torch.save(artifact, self.output_dir / "nano_version1_profile.pt")
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary
