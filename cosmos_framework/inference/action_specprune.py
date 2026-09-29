# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Observation-driven SpecPrune-VLA transfer for Cosmos3-Nano policies.

Selection is performed on the observed frame.  The selected spatial positions
are then applied, unchanged, to every future latent L1..L8.  Masks are nested
across decoder depth.  Rows that exit are kept at their exit hidden state and
the complete GEN layout is reconstructed before final normalization/heads.

This module is opt-in and deliberately does not use ASI scores, velocity
caching, future-noise differences, training, or an auxiliary model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.inference.action_k112 import (
    NANO_BLOCK_COUNT,
    NanoK112Controller,
)

STRATEGY_VERSION = "nano-specprune-vla-observation-asi-k184-budget-v1"
NANO_SPATIAL_TOKENS = 360
ASI_K184_AVERAGE_FUTURE_RETENTION = (NANO_SPATIAL_TOKENS + 7 * 184) / (8 * NANO_SPATIAL_TOKENS)


@dataclass(frozen=True)
class NanoSpecPruneConfig:
    """28-to-36-layer scaled version of the frozen Edge configuration."""

    local_k: int = 58
    global_k: int = 72
    global_blocks: tuple[int, int] = (17, 35)
    update_blocks: tuple[int, int, int] = (18, 25, 31)
    prune_blocks: tuple[int, int, int, int] = (13, 19, 26, 32)
    low_change_cap: int = 331
    dynamic_threshold: float = 0.986
    keep_ratio: float = 0.9
    ema_beta: float = 0.2
    min_observation_tokens: int = 184

    def __post_init__(self) -> None:
        if not 1 <= self.local_k <= NANO_SPATIAL_TOKENS:
            raise ValueError("local_k must fit the Nano spatial grid")
        if not 0 <= self.global_k <= NANO_SPATIAL_TOKENS:
            raise ValueError("global_k must fit the Nano spatial grid")
        if not 1 <= self.min_observation_tokens <= NANO_SPATIAL_TOKENS:
            raise ValueError("min_observation_tokens must fit the Nano spatial grid")
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0,1]")
        blocks = (*self.global_blocks, *self.update_blocks, *self.prune_blocks)
        if any(block < 0 or block >= NANO_BLOCK_COUNT for block in blocks):
            raise ValueError("SpecPrune block schedule does not fit the 36-layer Nano")


def top_mask(score: torch.Tensor, k: int, candidates: torch.Tensor | None = None) -> torch.Tensor:
    if score.ndim != 1 or not bool(torch.isfinite(score).all()):
        raise ValueError("Expected one-dimensional finite scores")
    if candidates is None:
        candidates = torch.ones_like(score, dtype=torch.bool)
    if candidates.shape != score.shape or candidates.dtype != torch.bool:
        raise ValueError("Candidate mask does not match score geometry")
    ids = torch.where(candidates)[0]
    count = min(max(int(k), 0), int(ids.numel()))
    result = torch.zeros_like(candidates)
    if count:
        order = torch.argsort(score.index_select(0, ids), descending=True, stable=True)
        result[ids.index_select(0, order[:count])] = True
    return result


def observation_grid_patches(rgb: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Return fixed-size same-coordinate RGB descriptors for Dynamic selection.

    The already transformed condition frame is area-resampled to a 4x4 RGB
    descriptor per live latent-grid position.  No model inference is involved.
    """

    while rgb.ndim == 4 and int(rgb.shape[0]) == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or int(rgb.shape[0]) != 3:
        raise ValueError(f"Expected condition RGB [3,H,W], got {tuple(rgb.shape)}")
    gh, gw = map(int, grid_hw)
    if gh * gw != NANO_SPATIAL_TOKENS:
        raise ValueError(f"Expected 360 Nano positions, got grid {grid_hw}")
    value = rgb.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Observation RGB contains NaN/Inf")
    resized = F.interpolate(value.unsqueeze(0), size=(gh * 4, gw * 4), mode="area")[0]
    return resized.reshape(3, gh, 4, gw, 4).permute(1, 3, 0, 2, 4).reshape(gh * gw, 3 * 4 * 4).contiguous()


def condition_rgb(data_batch: Mapping[str, Any], key: str = "video") -> torch.Tensor:
    value: Any = data_batch[key]
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("SpecPrune supports one observation item")
        value = value[0]
    if value.ndim == 5 and int(value.shape[0]) == 1:
        value = value[0]
    if value.ndim != 4 or int(value.shape[0]) != 3:
        raise ValueError(f"Expected transformed video [3,T,H,W], got {tuple(value.shape)}")
    return value[:, 0].detach().clone()


def resolve_instruction_indices(tokenizer: Any, text_ids: list[int], instruction: str) -> list[int]:
    """Resolve the unique instruction character span in the actual packed text."""

    rendered = tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not instruction or rendered.count(instruction) != 1:
        raise ValueError("Instruction must occur exactly once in the packed prompt")
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        # Qwen's local Nano deployment uses a Python (slow) tokenizer, which
        # cannot return character offsets.  Recover exact offsets from decoded
        # packed-token prefixes.  This is robust to a BPE token absorbing the
        # JSON whitespace before the instruction or punctuation after it.
        boundaries = [0]
        for right in range(1, len(text_ids) + 1):
            prefix = tokenizer.decode(
                text_ids[:right], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if not rendered.startswith(prefix):
                raise ValueError("Slow-tokenizer prefix decode is inconsistent with packed text")
            boundaries.append(len(prefix))
        start = rendered.index(instruction)
        end = start + len(instruction)
        result = [
            i for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])) if right > start and left < end
        ]
    else:
        if list(encoded["input_ids"]) != list(text_ids):
            raise ValueError("Instruction offset mapping changed the packed token IDs")
        start = rendered.index(instruction)
        end = start + len(instruction)
        result = [i for i, (left, right) in enumerate(encoded["offset_mapping"]) if right > start and left < end]
    if not result or any(text_ids[i] in tokenizer.all_special_ids for i in result):
        raise ValueError("Instruction span is empty or includes a special token")
    return result


class ObservationPlan:
    """Chunk-persistent observation history and request-local nested masks."""

    def __init__(self, config: NanoSpecPruneConfig = NanoSpecPruneConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.previous_rgb: torch.Tensor | None = None
        self.previous_global: dict[int, torch.Tensor] = {}
        self.confidence: dict[int, torch.Tensor] = {}
        self.pending_rgb: torch.Tensor | None = None

    def abort(self) -> None:
        self.pending_rgb = None

    def begin(self, rgb: torch.Tensor, nonvisual_tokens: int) -> None:
        if self.pending_rgb is not None:
            raise RuntimeError("Previous SpecPrune request was not finished/reset")
        if rgb.ndim != 2 or int(rgb.shape[0]) != NANO_SPATIAL_TOKENS:
            raise ValueError("Expected observation descriptors [360,features]")
        if not bool(torch.isfinite(rgb).all()):
            raise ValueError("Observation descriptors contain NaN/Inf")
        self.pending_rgb = rgb.detach().clone()
        self.nonvisual_tokens = int(nonvisual_tokens)
        self.active = torch.ones(NANO_SPATIAL_TOKENS, dtype=torch.bool, device=rgb.device)
        self.global_mask = torch.zeros_like(self.active)
        self.dynamic_mask = torch.zeros_like(self.active)
        self.similarity: torch.Tensor | None = None
        if self.previous_rgb is not None:
            previous = self.previous_rgb.to(device=rgb.device, dtype=rgb.dtype)
            if previous.shape != rgb.shape:
                raise ValueError("Observation descriptor geometry changed; reset task history")
            self.similarity = F.cosine_similarity(rgb, previous, dim=-1, eps=1e-8)
            stable = top_mask(
                self.similarity,
                self.config.low_change_cap,
                self.similarity >= self.config.dynamic_threshold,
            )
            self.dynamic_mask = ~stable
            for block in self.config.global_blocks:
                if block not in self.previous_global:
                    raise RuntimeError(f"Missing historical Global score B{block}")
                self.global_mask |= top_mask(self.previous_global[block].to(rgb.device), self.config.global_k)
        self.importance = torch.zeros(NANO_SPATIAL_TOKENS, device=rgb.device)
        self.dynamic_enabled = True
        self.masks: dict[int, torch.Tensor] = {}
        self.rows: list[dict[str, Any]] = []
        self.local_scores: dict[int, torch.Tensor] = {}
        self.global_scores: dict[int, torch.Tensor] = {}
        self.action_scores: dict[int, torch.Tensor] = {}

    def local(self, block: int, score: torch.Tensor) -> None:
        self.local_scores[block] = score.detach().clone()
        if block == 0:
            self.local0_top = top_mask(score, self.config.local_k)
            self.active = top_mask(score, 2 * self.config.local_k) | self.global_mask | self.dynamic_mask
        elif block == 1:
            keep = top_mask(score, self.config.local_k, self.active) | self.local0_top
            self.active &= keep | self.global_mask | self.dynamic_mask
        else:
            raise ValueError("Local selection is defined only at B0/B1")
        self.save(block, "local")

    def update(self, block: int, action_to_observation: torch.Tensor) -> None:
        expected = (32, NANO_SPATIAL_TOKENS)
        if tuple(action_to_observation.shape) != expected:
            raise ValueError(f"Expected action-to-observation scores {expected}")
        if not bool(torch.isfinite(action_to_observation).all()) or bool((action_to_observation < 0).any()):
            raise ValueError("Action-to-observation scores are invalid")
        self.action_scores[block] = action_to_observation.detach().clone()
        if not self.dynamic_enabled:
            return
        ids = torch.where(self.active)[0]
        attn = action_to_observation.index_select(1, ids).float()
        order = torch.argsort(attn.mean(0), descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(ids), device=ids.device)
        weight = torch.sigmoid(-rank.float())
        weight /= weight.sum().clamp_min(1e-8)
        if block not in self.confidence:
            probability = attn + 1e-8
            probability /= probability.sum(-1, keepdim=True)
            entropy = -(probability * probability.log()).sum(-1).mean()
            entropy /= torch.tensor(max(len(ids), 2), device=ids.device).log()
            self.confidence[block] = (1 / entropy.clamp_min(1e-8)).detach()
        beta = self.config.ema_beta
        self.importance[ids] = (1 - beta) * self.importance[ids] + beta * weight * self.confidence[block]

    def prune(self, block: int) -> None:
        count = int(self.active.sum())
        retain_total = max(
            int(self.config.keep_ratio * (count + self.nonvisual_tokens)),
            self.nonvisual_tokens + self.config.min_observation_tokens,
        )
        retain = min(count, retain_total - self.nonvisual_tokens)
        if retain >= count:
            self.dynamic_enabled = False
        self.active = top_mask(self.importance, retain, self.active)
        self.save(block, "dynamic")

    def save(self, block: int, kind: str) -> None:
        if self.masks and bool((self.active & ~next(reversed(self.masks.values()))).any()):
            raise AssertionError("SpecPrune reintroduced an exited position")
        self.masks[block] = self.active.detach().clone()
        self.rows.append(
            {
                "block": int(block),
                "kind": kind,
                "keep": int(self.active.sum()),
                "scores_nonzero": int((self.importance != 0).sum()),
            }
        )

    def finish(self) -> None:
        if set(self.global_scores) != set(self.config.global_blocks):
            raise ValueError("Incomplete historical Global observation attention")
        if self.pending_rgb is None:
            raise RuntimeError("SpecPrune request has no pending observation")
        self.previous_rgb = self.pending_rgb.detach().clone()
        self.previous_global = {block: score.detach().clone() for block, score in self.global_scores.items()}
        self.pending_rgb = None


class ExitHidden:
    """Store every full GEN row exactly once at its layer of exit."""

    def __init__(self, reference: torch.Tensor) -> None:
        self.values = torch.empty_like(reference)
        self.written = torch.zeros(len(reference), dtype=torch.bool, device=reference.device)

    def put(self, ids: torch.Tensor, hidden: torch.Tensor) -> None:
        if ids.ndim != 1 or len(ids) != len(hidden) or int(ids.unique().numel()) != len(ids):
            raise ValueError("Invalid exit coordinates")
        if bool(self.written.index_select(0, ids).any()):
            raise AssertionError("A GEN hidden row was restored twice")
        self.values.index_copy_(0, ids, hidden)
        self.written[ids] = True

    def finish(self) -> torch.Tensor:
        if not bool(self.written.all()) or not bool(torch.isfinite(self.values).all()):
            raise AssertionError("Full GEN hidden reconstruction is incomplete or nonfinite")
        return self.values


class NanoSpecPruneController(NanoK112Controller):
    """Build one observation-selected layer plan and replay it across CFG/steps."""

    def __init__(
        self,
        *,
        net: Any,
        guidance: float,
        num_steps: int,
        observation_rgb: torch.Tensor,
        instruction: str,
        tokenizer: Any,
        plan: ObservationPlan,
        output_dir: Path | None = None,
        compiled: bool = False,
        cuda_graphs: bool = False,
    ) -> None:
        super().__init__(net=net, guidance=guidance, num_steps=num_steps, output_dir=output_dir)
        self.observation_rgb = observation_rgb
        self.instruction = instruction
        self.tokenizer = tokenizer
        self.observation_plan = plan
        self._instruction_indices: torch.Tensor | None = None
        self._text_token_count: int | None = None
        self._exit_hidden: ExitHidden | None = None
        self._request_started = False
        self._request_finished = False
        self._token_rows: list[dict[str, Any]] = []
        self._exit_rows: list[dict[str, Any]] = []
        self._capture_block: int | None = None
        if cuda_graphs and not compiled:
            raise ValueError("CUDA Graph requires compiled SpecPrune layers")
        self.compiled = bool(compiled)
        self.cuda_graphs = bool(cuda_graphs)
        self._und_tokens: dict[str, int] = {}
        self._compiled_layers: list[Any | None] = [None] * len(self.layers)
        if self.compiled:
            cache_key = ("specprune", self.cuda_graphs)
            cache = getattr(self.model, "_nano_specprune_compiled_layers", None)
            if cache is None:
                cache = {}
                self.model._nano_specprune_compiled_layers = cache
            if cache_key not in cache:
                capture_blocks = {
                    0,
                    1,
                    *self.observation_plan.config.global_blocks,
                    *self.observation_plan.config.update_blocks,
                }
                cache[cache_key] = [
                    None
                    if block in capture_blocks
                    else torch.compile(
                        layer,
                        fullgraph=True,
                        dynamic=True,
                        mode="reduce-overhead" if self.cuda_graphs else "default",
                    )
                    for block, layer in enumerate(self.layers)
                ]
            self._compiled_layers = cache[cache_key]

    def _network_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return
        super()._network_pre_hook(module, args, kwargs)
        if self._current is None or self._layout is None or self._instruction_indices is not None:
            return
        packed = args[0] if args else kwargs["packed_seq"]
        text_ids = packed.text_ids.detach().cpu().tolist()
        indices = resolve_instruction_indices(self.tokenizer, text_ids, self.instruction)
        self._instruction_indices = torch.tensor(indices, dtype=torch.long)
        self._text_token_count = len(text_ids)

    def _network_post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs, output
        try:
            if self.stack_active:
                raise RuntimeError("Nano SpecPrune Transformer stack was not restored")
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
        if self._current is None:
            return
        if not memory_gen_only:
            raise RuntimeError("Nano SpecPrune requires completed Reasoner prefill/offload")
        super().begin_stack(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            memory_gen_only=memory_gen_only,
            natten_metadata_list=natten_metadata_list,
        )
        if not self.stack_active or self._layout is None or self._current is None:
            return
        full = get_gen_seq(hidden_states)
        self._exit_hidden = ExitHidden(full)
        building = int(self._current["step"]) == 0 and str(self._current["branch"]) == "conditional"
        if building:
            if self._request_started:
                raise RuntimeError("SpecPrune plan may be built only once per request")
            _, gh, gw = map(int, self._layout["latent_shape_thw"])
            descriptors = observation_grid_patches(self.observation_rgb, (gh, gw)).to(
                device=full.device, dtype=torch.float32
            )
            if self._text_token_count is None:
                raise RuntimeError("Packed text-token count was not captured")
            action_count = len(self._layout["action_positions"])
            self.observation_plan.begin(descriptors, self._text_token_count + action_count)
            self._request_started = True

    def _relative(self, originals: list[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
        if self._active_positions is None or self._layout is None:
            raise RuntimeError("SpecPrune active GEN mapping is missing")
        source = torch.as_tensor(originals, dtype=torch.long, device=device)
        inverse = torch.full((int(self._layout["num_gen_tokens"]),), -1, dtype=torch.long, device=device)
        inverse[self._active_positions] = torch.arange(len(self._active_positions), dtype=torch.long, device=device)
        result = inverse.index_select(0, source)
        if bool((result < 0).any()):
            raise AssertionError("A required L0/action row exited SpecPrune")
        return result

    @staticmethod
    def _probabilities(q: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
        q_heads = int(q.shape[1])
        kv_heads = int(k.shape[1])
        if q_heads % kv_heads:
            raise RuntimeError("Invalid Nano GQA head mapping")
        queries = q.detach().float().permute(1, 0, 2)
        keys = k.detach().float().repeat_interleave(q_heads // kv_heads, dim=1).permute(1, 0, 2)
        return torch.softmax(torch.matmul(queries * float(scaling), keys.transpose(1, 2)), dim=-1)

    def _capture_scores(
        self,
        *,
        layer_index: int,
        q_gen: torch.Tensor,
        k_ar: torch.Tensor,
        k_gen: torch.Tensor,
        scaling: float,
    ) -> None:
        if layer_index != self._capture_block or self._layout is None:
            raise RuntimeError("Nano SpecPrune attention callback block mismatch")
        self._und_tokens.setdefault(str(self._current["branch"]), int(k_ar.shape[0]))
        device = q_gen.device
        keys = torch.cat((k_ar, k_gen), dim=0)
        if layer_index in (0, 1, *self.observation_plan.config.global_blocks):
            l0 = self._relative(self._layout["latent_positions"]["L0"], device)
            probabilities = self._probabilities(q_gen.index_select(0, l0), keys, scaling)
            if self._instruction_indices is None:
                raise RuntimeError("Instruction token coordinates were not resolved")
            instruction = self._instruction_indices.to(device=device)
            if int(instruction.max()) >= len(k_ar):
                raise RuntimeError("Instruction index is outside cached UND K/V")
            score = probabilities.index_select(-1, instruction).sum(-1).mean(0)
            if tuple(score.shape) != (NANO_SPATIAL_TOKENS,):
                raise RuntimeError("Local/Global observation score geometry changed")
            if layer_index in (0, 1):
                self.observation_plan.local_scores[layer_index] = score.detach().clone()
            if layer_index in self.observation_plan.config.global_blocks:
                self.observation_plan.global_scores[layer_index] = score.detach().clone()
            return

        predicted = [row["gen_position"] for row in self._layout["action_queries"] if row["query_role"] == "predicted"]
        actions = self._relative(predicted, device)
        l0 = self._relative(self._layout["latent_positions"]["L0"], device)
        probabilities = self._probabilities(q_gen.index_select(0, actions), keys, scaling)
        observation_keys = l0 + len(k_ar)
        action_to_observation = probabilities.index_select(-1, observation_keys).mean(0)
        self.observation_plan.update(layer_index, action_to_observation)

    def _positions_for_mask(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self._layout is None or tuple(mask.shape) != (NANO_SPATIAL_TOKENS,):
            raise RuntimeError("SpecPrune mask/layout mismatch")
        selected = torch.where(mask.to(device=device))[0]
        parts = [torch.tensor(self._layout["latent_positions"]["L0"], dtype=torch.long, device=device)]
        for latent in range(1, 9):
            frame = torch.tensor(self._layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=device)
            parts.append(frame.index_select(0, selected))
        parts.append(torch.tensor(self._layout["action_positions"], dtype=torch.long, device=device))
        return torch.cat(parts).sort().values

    def _shrink_after_block(self, hidden_states: Any, mask: torch.Tensor, block: int) -> Any:
        if self._active_positions is None or self._exit_hidden is None:
            raise RuntimeError("SpecPrune shrink state is missing")
        current = get_gen_seq(hidden_states)
        selected_absolute = self._positions_for_mask(mask, current.device)
        inverse = torch.full((int(self._layout["num_gen_tokens"]),), -1, dtype=torch.long, device=current.device)
        inverse[self._active_positions] = torch.arange(
            len(self._active_positions), dtype=torch.long, device=current.device
        )
        relative = inverse.index_select(0, selected_absolute)
        if bool((relative < 0).any()):
            raise AssertionError("SpecPrune layer mask is not nested")
        retained = torch.zeros(len(self._active_positions), dtype=torch.bool, device=current.device)
        retained[relative] = True
        exited_absolute = self._active_positions[~retained]
        if len(exited_absolute):
            self._exit_hidden.put(exited_absolute, current[~retained])
        self._exit_rows.append(
            {
                "step": int(self._current["step"]),
                "branch": str(self._current["branch"]),
                "block": int(block),
                "dropped": int((~retained).sum()),
                "future_keep_per_frame": int(mask.sum()),
            }
        )
        if len(relative) == len(self._active_positions):
            if not torch.equal(relative, torch.arange(len(relative), device=relative.device)):
                raise AssertionError("Equal-size SpecPrune mask reordered GEN rows")
            return hidden_states
        hidden_states, self._position_embeddings = self._slice_pack_and_rope(hidden_states, relative)
        self._active_positions = selected_absolute
        self._pack_count += 1
        return hidden_states

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
        if not self.stack_active or self._current is None or self._active_positions is None:
            raise RuntimeError("Nano SpecPrune layer has no active stack")
        cached_und = getattr(memory_value, "und_k", None)
        if cached_und is None:
            cached_und = getattr(memory_value, "und_k_cached", None)
        if cached_und is not None:
            und_axis = 1 if cached_und.ndim == 4 else 0
            self._und_tokens.setdefault(str(self._current["branch"]), int(cached_und.shape[und_axis]))
        building = int(self._current["step"]) == 0 and str(self._current["branch"]) == "conditional"
        capture = building and block in (
            0,
            1,
            *self.observation_plan.config.global_blocks,
            *self.observation_plan.config.update_blocks,
        )
        attention = decoder_layer.self_attn
        if capture:
            if attention._nano_k112_attention_capture_callback is not None:
                raise RuntimeError(f"Nano SpecPrune callback is occupied at B{block}")
            self._capture_block = block
            attention._nano_k112_attention_capture_callback = self._capture_scores
        input_tokens = int(get_gen_seq(hidden_states).shape[0])
        try:
            compiled_layer = self._compiled_layers[block]
            execution_layer = compiled_layer if compiled_layer is not None else decoder_layer
            output = execution_layer(
                hidden_states,
                attention_mask,
                self._position_embeddings,
                natten_metadata=None,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        finally:
            if capture:
                attention._nano_k112_attention_capture_callback = None
                self._capture_block = None
        self._block_calls += 1
        self._token_rows.append(
            {
                "step": int(self._current["step"]),
                "branch": str(self._current["branch"]),
                "block": int(block),
                "gen_tokens": input_tokens,
                "full_gen_tokens": int(self._layout["num_gen_tokens"]),
            }
        )
        hidden = output[0]
        if building:
            if block in (0, 1):
                self.observation_plan.local(block, self.observation_plan.local_scores[block])
            elif block in self.observation_plan.config.prune_blocks:
                self.observation_plan.prune(block)
        mask = self.observation_plan.masks.get(block)
        if mask is not None:
            hidden = self._shrink_after_block(hidden, mask, block)
        return hidden, output[1], output[2]

    def end_stack(self, hidden_states: Any) -> Any:
        if (
            not self.stack_active
            or self._original_pack is None
            or self._active_positions is None
            or self._exit_hidden is None
        ):
            raise RuntimeError("Nano SpecPrune end_stack state is incomplete")
        self._exit_hidden.put(self._active_positions, get_gen_seq(hidden_states))
        restored_gen = self._exit_hidden.finish()
        restored = from_und_gen_splits(get_und_seq(hidden_states), restored_gen, self._original_pack)
        self._restore_count += 1
        self._completed_stacks += 1
        self.abort_stack()
        return restored

    def finish(self) -> dict[str, Any]:
        if self.stack_active or not self._request_started or self._request_finished:
            raise RuntimeError("Nano SpecPrune request did not finish cleanly")
        expected_stacks = self.num_steps * 2
        expected_calls = expected_stacks * len(self.layers)
        if self._completed_stacks != expected_stacks or self._block_calls != expected_calls:
            raise RuntimeError("Nano SpecPrune stack/block accounting is incomplete")
        if self._restore_count != expected_stacks:
            raise RuntimeError("Nano SpecPrune full-hidden restoration is incomplete")
        self.observation_plan.finish()
        self._request_finished = True
        spatial = NANO_SPATIAL_TOKENS
        action_count = len(self._layout["action_positions"])
        future_ratios = [(row["gen_tokens"] - spatial - action_count) / (8 * spatial) for row in self._token_rows]
        mean_retention = sum(future_ratios) / len(future_ratios)
        summary = {
            "schema_version": 1,
            "strategy_version": STRATEGY_VERSION,
            "config": asdict(self.observation_plan.config),
            "selection_source": "observation",
            "future_frames_share_spatial_mask_per_layer": True,
            "masks_are_nested": True,
            "reasoner_prefill_and_offload_preserved": True,
            "velocity_cache_enabled": False,
            "compile": self.compiled,
            "cuda_graphs_requested": self.cuda_graphs,
            "und_tokens": self._und_tokens,
            "full_gen_tokens": int(self._layout["num_gen_tokens"]),
            "transformer_calls": len(self._token_rows),
            "mean_future_token_retention": mean_retention,
            "asi_k184_average_future_retention_target": ASI_K184_AVERAGE_FUTURE_RETENTION,
            "retention_gap_percentage_points": 100 * (mean_retention - ASI_K184_AVERAGE_FUTURE_RETENTION),
            "layer_plan": self.observation_plan.rows,
            "token_rows": self._token_rows,
            "exit_rows": self._exit_rows,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "local_scores": {
                        block: value.detach().cpu() for block, value in self.observation_plan.local_scores.items()
                    },
                    "global_scores": {
                        block: value.detach().cpu() for block, value in self.observation_plan.global_scores.items()
                    },
                    "action_scores": {
                        block: value.detach().cpu() for block, value in self.observation_plan.action_scores.items()
                    },
                    "masks": {block: value.detach().cpu() for block, value in self.observation_plan.masks.items()},
                    "dynamic_mask": self.observation_plan.dynamic_mask.detach().cpu(),
                },
                self.output_dir / "specprune_profile.pt",
            )
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None and self.observation_plan.pending_rgb is not None:
            self.observation_plan.abort()
        super().__exit__(exc_type, exc, traceback)
