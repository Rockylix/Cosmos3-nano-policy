"""Shared-Motion K112 sparse inference for Cosmos3 Edge policy serving.

The implementation is request-local and default-off.  It profiles the dense
step-0 conditional Transformer stack, derives one shared 112-token spatial
mask, and runs the remaining seven CFG stacks with that same mask packed once
before B0 and restored once after B27.

The matching Cosmos Framework hook patch must be applied before this module is
enabled.  See ``patches/cosmos-framework/0002-add-k112-runtime-hooks.patch``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STRATEGY_VERSION = "shared-motion-k112-v1"
TOKEN_BUDGET = 112
PROFILE_BLOCK_COUNT = 4
EXPECTED_BLOCKS = tuple(range(28))
ACTION_HORIZON_WEIGHTS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


def _make_sequence_pack(*, und_seq: Any, gen_seq: Any) -> Any:
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        init_sequence_pack,
    )

    und_len = int(und_seq.shape[0])
    gen_len = int(gen_seq.shape[0])
    metadata = init_sequence_pack(
        sample_lens=[und_len + gen_len],
        split_lens=[und_len, gen_len],
        attn_modes=["causal", "full"],
        device=gen_seq.device,
    )
    return {
        **metadata,
        "max_num_tokens": und_len + gen_len,
        "causal_seq": und_seq,
        "full_only_seq": gen_seq,
        "is_sharded": False,
    }


def _get_gen_seq(pack: Any) -> Any:
    from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq

    return get_gen_seq(pack)


def _get_und_seq(pack: Any) -> Any:
    from cosmos_framework.data.generator.sequence_packing.runtime import get_und_seq

    return get_und_seq(pack)


def _from_und_gen_splits(und_seq: Any, gen_seq: Any, template: Any) -> Any:
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        from_und_gen_splits,
    )

    return from_und_gen_splits(und_seq, gen_seq, template)


def action_query_layout(torch: Any, packed_sequence: Any) -> dict[str, Any]:
    """Resolve L0..L8 and action rows in the live full-token ordering."""

    full_indexes: list[int] = []
    offset = 0
    for mode, split_len in zip(
        packed_sequence.attn_modes,
        packed_sequence.split_lens,
        strict=True,
    ):
        split_len = int(split_len)
        if mode == "full":
            full_indexes.extend(range(offset, offset + split_len))
        offset += split_len
    full_position = {
        original: position for position, original in enumerate(full_indexes)
    }

    vision = getattr(packed_sequence, "vision", None)
    action = getattr(packed_sequence, "action", None)
    vision_indexes = getattr(vision, "sequence_indexes", None)
    action_indexes = getattr(action, "sequence_indexes", None)
    if vision is None or action is None or not torch.is_tensor(vision_indexes):
        raise RuntimeError("K112 requires packed vision and action tokens")
    if not torch.is_tensor(action_indexes):
        raise RuntimeError("K112 requires packed action sequence indexes")
    try:
        vision_positions = [
            full_position[int(value)]
            for value in vision_indexes.detach().cpu().tolist()
        ]
        action_positions = [
            full_position[int(value)]
            for value in action_indexes.detach().cpu().tolist()
        ]
    except KeyError as exc:
        raise RuntimeError(
            f"Modality token {int(exc.args[0])} is outside the GEN sequence"
        ) from exc

    vision_shapes = getattr(vision, "token_shapes", None)
    if not vision_shapes or len(vision_shapes) != 1:
        raise RuntimeError(f"Expected one vision token shape, got {vision_shapes}")
    latent_shape = tuple(int(value) for value in vision_shapes[0])
    if len(latent_shape) != 3 or math.prod(latent_shape) != len(vision_positions):
        raise RuntimeError(
            f"Vision shape {latent_shape} does not match {len(vision_positions)} tokens"
        )
    num_latents, height, width = latent_shape
    if num_latents != 9:
        raise RuntimeError(f"Expected L0..L8, got {num_latents} vision latents")
    spatial = height * width
    latent_positions = {
        f"L{latent}": vision_positions[
            latent * spatial : (latent + 1) * spatial
        ]
        for latent in range(9)
    }
    covered = [
        position for values in latent_positions.values() for position in values
    ] + action_positions
    if len(covered) != len(full_indexes) or sorted(covered) != list(
        range(len(full_indexes))
    ):
        raise RuntimeError("L0..L8 and action tokens do not partition GEN")

    action_shapes = getattr(action, "token_shapes", None)
    if not action_shapes or len(action_shapes) != 1:
        raise RuntimeError(f"Expected one action token shape, got {action_shapes}")
    action_shape = tuple(int(value) for value in action_shapes[0])
    if math.prod(action_shape) != len(action_positions):
        raise RuntimeError("Action token shape does not match the packed positions")
    condition_mask = getattr(action, "condition_mask", None)
    if isinstance(condition_mask, (list, tuple)):
        if len(condition_mask) != 1:
            raise RuntimeError("K112 supports batch-size-one policy inference only")
        condition_mask = condition_mask[0]
    if not torch.is_tensor(condition_mask) or int(condition_mask.numel()) != len(
        action_positions
    ):
        raise RuntimeError("Action condition mask does not match packed actions")
    flags = [
        bool(value)
        for value in condition_mask.detach().reshape(-1).cpu().tolist()
    ]
    predicted_horizon = 0
    queries: list[dict[str, Any]] = []
    for action_index, (gen_position, is_condition) in enumerate(
        zip(action_positions, flags, strict=True)
    ):
        horizon = -1 if is_condition else predicted_horizon
        if not is_condition:
            predicted_horizon += 1
        queries.append(
            {
                "query_action_index": action_index,
                "gen_position": gen_position,
                "query_role": "condition" if is_condition else "predicted",
                "action_horizon": horizon,
            }
        )
    if predicted_horizon != 32:
        raise RuntimeError(f"Expected 32 predicted action rows, got {predicted_horizon}")
    return {
        "num_gen_tokens": len(full_indexes),
        "latent_shape_thw": list(latent_shape),
        "latent_positions": latent_positions,
        "action_positions": action_positions,
        "action_queries": queries,
    }


def action_aligned_future_profiles(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    horizon_weights: Sequence[float] = ACTION_HORIZON_WEIGHTS,
) -> Any:
    """Return raw action-aligned future attention with shape ``[8, spatial]``."""

    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise ValueError("K112 expects Q/K tensors [tokens, heads, head_dim]")
    if int(q_gen.shape[0]) != int(token_layout["num_gen_tokens"]):
        raise RuntimeError("K112 profile requires the complete GEN sequence")
    q_heads = int(q_gen.shape[1])
    kv_heads = int(k_gen.shape[1])
    if q_heads % kv_heads:
        raise RuntimeError(f"Invalid GQA geometry Hq={q_heads}, Hkv={kv_heads}")

    predicted = [
        query
        for query in token_layout["action_queries"]
        if query["query_role"] == "predicted"
    ]
    horizon_to_position = {
        int(query["action_horizon"]): int(query["gen_position"])
        for query in predicted
    }
    if sorted(horizon_to_position) != list(range(32)):
        raise RuntimeError("Expected predicted action horizons 0..31")
    action_index = torch.tensor(
        [horizon_to_position[horizon] for horizon in range(32)],
        dtype=torch.long,
        device=q_gen.device,
    )
    q = (
        q_gen.index_select(0, action_index)
        .detach()
        .float()
        .permute(1, 0, 2)
        .contiguous()
    )
    k = torch.cat((k_ar, k_gen), dim=0).detach().float()
    k = (
        k.repeat_interleave(q_heads // kv_heads, dim=1)
        .permute(1, 0, 2)
        .contiguous()
    )
    probabilities = torch.softmax(
        torch.matmul(q * float(scaling), k.transpose(1, 2)), dim=-1
    )
    head_mean = probabilities.mean(dim=0)
    if not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("K112 action attention contains NaN/Inf")

    weights = torch.tensor(
        horizon_weights, dtype=head_mean.dtype, device=head_mean.device
    )
    if tuple(weights.shape) != (4,) or bool((weights < 0).any()):
        raise ValueError("K112 horizon weights must be four non-negative values")
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    num_ar = int(k_ar.shape[0])
    profiles = []
    for latent in range(1, 9):
        horizons = list(range(4 * (latent - 1), 4 * latent))
        positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=q_gen.device,
        )
        aligned = head_mean[horizons].index_select(-1, positions + num_ar)
        profiles.append((aligned * weights[:, None]).sum(dim=0))
    result = torch.stack(profiles)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise RuntimeError("K112 profile contains NaN/Inf or negative values")
    return result


def build_shared_motion_mask(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
    token_budget: int = TOKEN_BUDGET,
    profile_block_count: int = PROFILE_BLOCK_COUNT,
    expected_blocks: Sequence[int] = EXPECTED_BLOCKS,
) -> dict[str, Any]:
    """Select Top-K from four stable, concentrated action-attention blocks."""

    blocks = tuple(int(value) for value in expected_blocks)
    lookup: dict[int, Any] = {}
    for record in profile_records:
        if str(record.get("branch")) != "conditional":
            raise ValueError("K112 accepts conditional profiles only")
        block = int(record["block"])
        if block in lookup:
            raise ValueError(f"Duplicate K112 profile for B{block}")
        lookup[block] = record["profiles"]
    missing = [block for block in blocks if block not in lookup]
    extra = sorted(set(lookup) - set(blocks))
    if missing or extra:
        raise ValueError(f"K112 profile mismatch: missing={missing}, extra={extra}")
    raw = torch.stack([lookup[block] for block in blocks])
    if raw.ndim != 3 or int(raw.shape[1]) != 8:
        raise ValueError(f"Expected [blocks,8,spatial], got {tuple(raw.shape)}")
    block_count, frame_count, spatial = map(int, raw.shape)
    if not 0 < profile_block_count <= block_count:
        raise ValueError("profile_block_count must fit the block set")
    if not 0 < token_budget <= spatial:
        raise ValueError("token_budget must fit the spatial grid")
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise RuntimeError("K112 profile contains invalid values")

    eps = torch.finfo(raw.dtype).eps
    frame_mass = raw.sum(dim=-1)
    probability = raw / frame_mass.unsqueeze(-1).clamp_min(eps)
    frame_entropy = -(
        probability * probability.clamp_min(eps).log()
    ).sum(dim=-1) / math.log(spatial)
    frame_entropy = torch.where(
        frame_mass > eps, frame_entropy, torch.ones_like(frame_entropy)
    )
    block_mass = frame_mass.mean(dim=-1)
    block_entropy = frame_entropy.mean(dim=-1)
    block_quality = block_mass * (1.0 - block_entropy).clamp_min(0)

    unit_maps = raw / torch.linalg.vector_norm(
        raw, dim=-1, keepdim=True
    ).clamp_min(eps)
    pairwise = torch.matmul(unit_maps, unit_maps.transpose(-1, -2))
    pair_mask = torch.triu(
        torch.ones(
            (frame_count, frame_count), dtype=torch.bool, device=raw.device
        ),
        diagonal=1,
    )
    temporal_stability = pairwise[:, pair_mask].mean(dim=-1)
    block_score = block_quality * temporal_stability.clamp_min(0)
    selected_local = torch.topk(block_score, profile_block_count).indices
    selected_score = block_score.index_select(0, selected_local)
    if float(selected_score.sum()) <= float(eps):
        selected_weights = torch.full_like(
            selected_score, 1.0 / profile_block_count
        )
    else:
        selected_weights = selected_score / selected_score.sum()
    selected_profiles = raw.index_select(0, selected_local)
    frame_scores = (
        selected_profiles * selected_weights[:, None, None]
    ).sum(dim=0)

    # Third-smallest requires a token to remain strong in at least 6/8 frames.
    shared_score = frame_scores.sort(dim=0).values[2]
    chosen = torch.topk(shared_score, token_budget, sorted=False).indices
    shared_mask = torch.zeros(spatial, dtype=torch.bool, device=raw.device)
    shared_mask[chosen] = True
    if int(shared_mask.sum()) != token_budget or not bool(
        torch.isfinite(shared_score).all()
    ):
        raise RuntimeError("K112 mask construction failed validation")
    block_tensor = torch.tensor(blocks, dtype=torch.long, device=raw.device)
    return {
        "raw_profiles": raw,
        "frame_mass": frame_mass,
        "frame_entropy": frame_entropy,
        "block_mass": block_mass,
        "block_entropy": block_entropy,
        "block_quality": block_quality,
        "temporal_stability": temporal_stability,
        "block_score": block_score,
        "selected_blocks": block_tensor.index_select(0, selected_local),
        "selected_block_weights": selected_weights,
        "selected_frame_scores": frame_scores,
        "shared_score": shared_score,
        "shared_mask": shared_mask,
    }


def selected_gen_positions(
    torch: Any, token_layout: Mapping[str, Any], shared_mask: Any, device: Any
) -> Any:
    """Keep L0, all actions, and identical K112 coordinates in L1..L8."""

    spatial = len(token_layout["latent_positions"]["L1"])
    if tuple(shared_mask.shape) != (spatial,):
        raise ValueError(f"Expected K112 mask [{spatial}], got {tuple(shared_mask.shape)}")
    selected_spatial = torch.nonzero(
        shared_mask.to(device=device), as_tuple=False
    ).flatten()
    parts = [
        torch.tensor(
            token_layout["latent_positions"]["L0"],
            dtype=torch.long,
            device=device,
        )
    ]
    for latent in range(1, 9):
        frame = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=device,
        )
        parts.append(frame.index_select(0, selected_spatial))
    parts.append(
        torch.tensor(
            token_layout["action_positions"], dtype=torch.long, device=device
        )
    )
    return torch.cat(parts).sort().values


def expand_token_mask(
    *, torch: Any, mask: Any, token_grid: tuple[int, int], output_grid: tuple[int, int]
) -> Any:
    token_h, token_w = map(int, token_grid)
    output_h, output_w = map(int, output_grid)
    if tuple(mask.shape) != (token_h * token_w,):
        raise ValueError("K112 token mask does not match the latent token grid")
    scale_h = (output_h + token_h - 1) // token_h
    scale_w = (output_w + token_w - 1) // token_w
    expanded = (
        mask.reshape(token_h, token_w)
        .repeat_interleave(scale_h, 0)
        .repeat_interleave(scale_w, 1)
    )
    return expanded[:output_h, :output_w].contiguous()


class K112Controller:
    """One dense profile stack followed by seven pack-once sparse stacks."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        output_dir: Path | None = None,
        expected_blocks: Sequence[int] = EXPECTED_BLOCKS,
    ) -> None:
        if int(num_steps) != 4:
            raise ValueError("K112 is validated only for four denoise steps")
        if float(guidance) == 1.0:
            raise ValueError("K112 requires conditional/unconditional CFG")
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.expected_blocks = tuple(int(block) for block in expected_blocks)
        if not self.expected_blocks or self.expected_blocks != tuple(range(len(self.expected_blocks))):
            raise ValueError("K112 expected_blocks must be a contiguous zero-based range")
        self.model = net.language_model.model
        self.layers = list(self.model.layers)
        if len(self.layers) != len(self.expected_blocks):
            raise ValueError(f"K112 expects {len(self.expected_blocks)} blocks, got {len(self.layers)}")
        self._handles: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._layout: dict[str, Any] | None = None
        self._forward_index = 0
        self._original_pack: Any | None = None
        self._position_embeddings: Any | None = None
        self._active_positions: Any | None = None
        self._side_buffer: Any | None = None
        self._selected_positions: Any | None = None
        self._selected_device: Any | None = None
        self._profile_records: list[dict[str, Any]] = []
        self._profile_callback_block: int | None = None
        self.plan: dict[str, Any] | None = None
        self.shared_mask: Any | None = None
        self.vision_shape: tuple[int, ...] | None = None
        self._step0_conditional_vision: list[Any] = []
        self._velocity_mask: Any | None = None
        self._completed_stacks = 0
        self._pack_count = 0
        self._restore_count = 0
        self._block_calls = 0
        self._cfg_zero_fill_elements = 0
        self.stack_active = False

    def __enter__(self) -> "K112Controller":
        if hasattr(self.model, "_k112_sparse_controller"):
            raise RuntimeError("A K112 controller is already installed")
        for layer in self.layers:
            attention = layer.self_attn
            if not hasattr(attention, "_k112_attention_capture_callback"):
                raise RuntimeError(
                    "Cosmos Framework K112 hooks are missing; apply the bundled patch"
                )
            if attention._k112_attention_capture_callback is not None:
                raise RuntimeError("A K112 attention callback is already installed")
        setattr(self.model, "_k112_sparse_controller", self)
        self._handles.append(
            self.net.register_forward_pre_hook(
                self._network_pre_hook, with_kwargs=True
            )
        )
        self._handles.append(
            self.net.register_forward_hook(
                self._network_post_hook, with_kwargs=True, always_call=True
            )
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.abort_stack()
        for layer in self.layers:
            layer.self_attn._k112_attention_capture_callback = None
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if getattr(self.model, "_k112_sparse_controller", None) is self:
            delattr(self.model, "_k112_sparse_controller")
        self._current = None

    def _network_pre_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        del module
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return
        packed_sequence = args[0] if args else kwargs.get("packed_seq")
        step = self._forward_index // 2
        branch = "conditional" if self._forward_index % 2 == 0 else "unconditional"
        self._forward_index += 1
        if self._layout is None:
            self._layout = action_query_layout(self.torch, packed_sequence)
        timesteps = getattr(getattr(packed_sequence, "vision", None), "timesteps", None)
        timestep = float("nan")
        if self.torch.is_tensor(timesteps) and timesteps.numel():
            timestep = float(timesteps.reshape(-1)[0].detach().float().cpu())
        self._current = {"step": step, "branch": branch, "timestep": timestep}

    def _network_post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Any:
        del module, args, kwargs
        current = None if self._current is None else dict(self._current)
        try:
            if self.stack_active:
                raise RuntimeError("K112 Transformer stack was not restored")
            if not isinstance(output, dict) or current is None:
                return None
            predictions = output.get("preds_vision")
            if not predictions:
                return None
            shape = tuple(int(value) for value in predictions[0].shape)
            if self.vision_shape is None:
                self.vision_shape = shape
            elif shape != self.vision_shape:
                raise RuntimeError(
                    f"K112 vision shape changed: {self.vision_shape} -> {shape}"
                )
            if int(current["step"]) != 0:
                return None
            if str(current["branch"]) == "conditional":
                self._step0_conditional_vision = [
                    value.detach().clone() for value in predictions
                ]
                return None
            if len(predictions) != len(self._step0_conditional_vision):
                raise RuntimeError("K112 conditional/unconditional batch differs")
            if self.shared_mask is None or self._layout is None:
                raise RuntimeError("K112 mask was not ready for step-0 CFG")
            replaced = []
            for conditional, unconditional in zip(
                self._step0_conditional_vision, predictions, strict=True
            ):
                temporal_dim = unconditional.ndim - 3
                if int(unconditional.shape[temporal_dim]) != 9:
                    raise RuntimeError("K112 expects L0..L8 vision predictions")
                height, width = int(unconditional.shape[-2]), int(
                    unconditional.shape[-1]
                )
                if self._velocity_mask is None:
                    _, token_h, token_w = self._layout["latent_shape_thw"]
                    one = expand_token_mask(
                        torch=self.torch,
                        mask=self.shared_mask,
                        token_grid=(int(token_h), int(token_w)),
                        output_grid=(height, width),
                    )
                    self._velocity_mask = one.unsqueeze(0).expand(8, -1, -1)
                conditional_future = conditional.narrow(temporal_dim, 1, 8)
                unconditional_future = unconditional.narrow(temporal_dim, 1, 8)
                view_shape = [1] * (unconditional_future.ndim - 3) + [
                    8,
                    height,
                    width,
                ]
                background = (~self._velocity_mask.to(unconditional.device)).reshape(
                    view_shape
                ).expand_as(unconditional_future)
                value = unconditional.clone()
                value.narrow(temporal_dim, 1, 8).copy_(
                    self.torch.where(
                        background,
                        conditional_future.to(unconditional),
                        unconditional_future,
                    )
                )
                self._cfg_zero_fill_elements += int(background.sum())
                replaced.append(value)
            modified = dict(output)
            modified["preds_vision"] = replaced
            return modified
        finally:
            self._current = None

    def begin_stack(
        self,
        *,
        hidden_states: Any,
        position_embeddings: Any,
        memory_gen_only: bool,
        natten_metadata_list: list | None,
    ) -> None:
        del memory_gen_only
        if self._current is None:
            return
        if self.stack_active:
            raise RuntimeError("Nested K112 Transformer stacks are unsupported")
        if natten_metadata_list is not None:
            raise RuntimeError("K112 requires dense/FlashAttention, not NATTEN")
        if self._layout is None:
            raise RuntimeError("K112 has no token layout")
        num_gen = int(hidden_states["_num_full_tokens"])
        if num_gen != int(self._layout["num_gen_tokens"]):
            raise RuntimeError("K112 stack did not start from the full GEN layout")
        if bool(hidden_states.get("is_sharded", False)):
            raise RuntimeError("K112 does not support context-parallel SequencePacks")
        self._original_pack = hidden_states
        self._position_embeddings = position_embeddings
        self._active_positions = self.torch.arange(
            num_gen,
            dtype=self.torch.long,
            device=_get_gen_seq(hidden_states).device,
        )
        self._side_buffer = _get_gen_seq(hidden_states).detach().clone()
        self.stack_active = True

    def _capture_profile(
        self,
        *,
        layer_index: int,
        q_gen: Any,
        k_ar: Any,
        k_gen: Any,
        scaling: float,
    ) -> None:
        if self._current is None or self._layout is None:
            raise RuntimeError("K112 profile callback has no request context")
        if layer_index != self._profile_callback_block:
            raise RuntimeError("K112 profile callback block mismatch")
        profile = action_aligned_future_profiles(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=scaling,
            token_layout=self._layout,
        ).detach()
        self._profile_records.append(
            {
                "branch": str(self._current["branch"]),
                "block": int(layer_index),
                "profiles": profile,
            }
        )

    def _slice_pack_and_rope(self, hidden_states: Any, selected: Any) -> Any:
        if self._position_embeddings is None:
            raise RuntimeError("K112 RoPE state is missing")
        sparse_pack = _make_sequence_pack(
            und_seq=_get_und_seq(hidden_states),
            gen_seq=_get_gen_seq(hidden_states).index_select(0, selected),
        )
        cos_pack, sin_pack = self._position_embeddings
        sparse_rope = (
            _make_sequence_pack(
                und_seq=_get_und_seq(cos_pack),
                gen_seq=_get_gen_seq(cos_pack).index_select(0, selected),
            ),
            _make_sequence_pack(
                und_seq=_get_und_seq(sin_pack),
                gen_seq=_get_gen_seq(sin_pack).index_select(0, selected),
            ),
        )
        return sparse_pack, sparse_rope

    def _prepare_selected_positions(self, device: Any) -> Any:
        if self._selected_positions is not None:
            if device != self._selected_device:
                raise RuntimeError("K112 GEN token device changed during one request")
            return self._selected_positions
        if self._layout is None or self.shared_mask is None:
            raise RuntimeError("K112 mask is not finalized")
        self._selected_positions = selected_gen_positions(
            self.torch, self._layout, self.shared_mask, device
        )
        self._selected_device = device
        return self._selected_positions

    def run_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: Any,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if not self.stack_active or self._current is None:
            raise RuntimeError("K112 run_layer has no active stack")
        step = int(self._current["step"])
        branch = str(self._current["branch"])
        dense_profile = step == 0 and branch == "conditional"
        attention = decoder_layer.self_attn
        if dense_profile:
            if attention._k112_attention_capture_callback is not None:
                raise RuntimeError(f"K112 callback is occupied at B{block}")
            self._profile_callback_block = block
            attention._k112_attention_capture_callback = self._capture_profile
            try:
                output = decoder_layer(
                    hidden_states,
                    attention_mask,
                    self._position_embeddings,
                    natten_metadata=None,
                    memory_value=memory_value,
                    gen_only=gen_only,
                )
            finally:
                attention._k112_attention_capture_callback = None
                self._profile_callback_block = None
            self._block_calls += 1
            return output

        if self.shared_mask is None:
            raise RuntimeError("K112 sparse stack started before mask finalization")
        if block == 0:
            selected = self._prepare_selected_positions(
                _get_gen_seq(hidden_states).device
            )
            hidden_states, self._position_embeddings = self._slice_pack_and_rope(
                hidden_states, selected
            )
            self._active_positions = selected
            self._pack_count += 1
        output = decoder_layer(
            hidden_states,
            attention_mask,
            self._position_embeddings,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        if int(_get_gen_seq(output[0]).shape[0]) != int(
            self._active_positions.numel()
        ):
            raise RuntimeError("K112 sparse block changed the packed GEN length")
        self._block_calls += 1
        return output

    def end_stack(self, hidden_states: Any) -> Any:
        if not self.stack_active or self._original_pack is None:
            raise RuntimeError("K112 end_stack has no active stack")
        if self._current is None or self._active_positions is None:
            raise RuntimeError("K112 stack state is incomplete")
        dense_profile = int(self._current["step"]) == 0 and str(
            self._current["branch"]
        ) == "conditional"
        if dense_profile:
            restored = hidden_states
            self.plan = build_shared_motion_mask(
                torch=self.torch,
                profile_records=self._profile_records,
                expected_blocks=self.expected_blocks,
            )
            self.shared_mask = self.plan["shared_mask"]
        else:
            if self._side_buffer is None:
                raise RuntimeError("K112 side buffer is missing")
            restored_gen = self._side_buffer.clone()
            restored_gen.index_copy_(
                0, self._active_positions, _get_gen_seq(hidden_states)
            )
            restored = _from_und_gen_splits(
                _get_und_seq(hidden_states), restored_gen, self._original_pack
            )
            self._restore_count += 1
        self._completed_stacks += 1
        self.abort_stack()
        return restored

    def abort_stack(self) -> None:
        self.stack_active = False
        self._original_pack = None
        self._position_embeddings = None
        self._active_positions = None
        self._side_buffer = None
        self._profile_callback_block = None

    def finish(self) -> dict[str, Any]:
        if self.stack_active or self.plan is None or self.shared_mask is None:
            raise RuntimeError("K112 request did not finish cleanly")
        expected_stacks = self.num_steps * 2
        expected_blocks = expected_stacks * len(self.layers)
        if self._completed_stacks != expected_stacks:
            raise RuntimeError(
                f"K112 completed {self._completed_stacks}/{expected_stacks} stacks"
            )
        if self._block_calls != expected_blocks:
            raise RuntimeError(
                f"K112 completed {self._block_calls}/{expected_blocks} blocks"
            )
        if self._pack_count != expected_stacks - 1 or self._restore_count != expected_stacks - 1:
            raise RuntimeError("K112 pack/restore counts are incomplete")
        if self._cfg_zero_fill_elements <= 0 or not self._step0_conditional_vision:
            raise RuntimeError("K112 step-0 CFG background handling was not observed")
        if self._layout is None:
            raise RuntimeError("K112 final token layout is missing")
        full_gen = int(self._layout["num_gen_tokens"])
        spatial = int(self.shared_mask.numel())
        sparse_gen = full_gen - 8 * (spatial - TOKEN_BUDGET)
        selected_blocks = [
            int(value) for value in self.plan["selected_blocks"].detach().cpu().tolist()
        ]
        summary = {
            "schema_version": 1,
            "strategy_version": STRATEGY_VERSION,
            "profile": "step0 conditional B0-B27; one action-row QK softmax per block",
            "action_horizon_weights": list(ACTION_HORIZON_WEIGHTS),
            "block_score": "R_l * (1 - H_l) * max(T_l, 0)",
            "selected_profile_blocks": selected_blocks,
            "shared_motion_tokens_per_future": TOKEN_BUDGET,
            "selected_spatial_positions": self.torch.nonzero(
                self.shared_mask, as_tuple=False
            ).flatten().detach().cpu().tolist(),
            "future_frames_share_spatial_mask": True,
            "cfg_branches_share_spatial_mask": True,
            "all_blocks_share_spatial_mask": True,
            "full_gen_tokens": full_gen,
            "sparse_gen_tokens": sparse_gen,
            "saved_gen_tokens_per_sparse_block": full_gen - sparse_gen,
            "sparse_gen_retained_ratio": sparse_gen / full_gen,
            "dense_stack_count": 1,
            "sparse_stack_count": expected_stacks - 1,
            "pack_count": self._pack_count,
            "restore_count": self._restore_count,
            "step0_background_cfg_delta": "zero outside K112",
            "background_velocity_after_step0": "cached step0 guided velocity",
            "all_finite": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                key: value.detach().cpu() if hasattr(value, "detach") else value
                for key, value in self.plan.items()
            }
            self.torch.save(artifact, self.output_dir / "shared_motion_k112_profile.pt")
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return summary


class K112VelocityCacheSampler:
    """Reuse the step-0 guided future background velocity after step 0."""

    def __init__(self, inner: Any, controller: K112Controller) -> None:
        self.inner = inner
        self.controller = controller
        self.cached_step0_vision: list[Any] = []
        self.completed_steps = 0

    def __call__(self, velocity_fn: Any, initial_noise: Any, **kwargs: Any) -> Any:
        self.cached_step0_vision.clear()
        self.completed_steps = 0

        def cached(noise_x: Any, timestep: Any) -> Any:
            current = velocity_fn(noise_x, timestep)
            if self.controller.vision_shape is None:
                raise RuntimeError("K112 did not observe a vision prediction shape")
            if self.controller._velocity_mask is None:
                raise RuntimeError("K112 velocity ROI was not initialized")
            vision_numel = math.prod(self.controller.vision_shape)
            output = []
            for sample, flat_velocity in enumerate(current):
                vision = flat_velocity[:vision_numel].reshape(
                    self.controller.vision_shape
                )
                if self.completed_steps == 0:
                    self.cached_step0_vision.append(vision.detach().clone())
                    merged = vision
                else:
                    cached_vision = self.cached_step0_vision[sample].to(
                        device=vision.device, dtype=vision.dtype
                    )
                    temporal_dim = vision.ndim - 3
                    current_future = vision.narrow(temporal_dim, 1, 8)
                    cached_future = cached_vision.narrow(temporal_dim, 1, 8)
                    height, width = int(vision.shape[-2]), int(vision.shape[-1])
                    view_shape = [1] * (current_future.ndim - 3) + [
                        8,
                        height,
                        width,
                    ]
                    background = (~self.controller._velocity_mask.to(vision.device)).reshape(
                        view_shape
                    ).expand_as(current_future)
                    merged = vision.clone()
                    merged.narrow(temporal_dim, 1, 8).copy_(
                        self.controller.torch.where(
                            background, cached_future, current_future
                        )
                    )
                output.append(
                    self.controller.torch.cat(
                        (merged.reshape(-1), flat_velocity[vision_numel:]), dim=0
                    )
                )
            self.completed_steps += 1
            return output

        result = self.inner(cached, initial_noise, **kwargs)
        expected = int(kwargs.get("num_steps", self.controller.num_steps))
        if self.completed_steps != expected:
            raise RuntimeError(
                f"K112 sampler completed {self.completed_steps}/{expected} steps"
            )
        return result
