"""Offload-safe GEN-only adapters for the frozen Edge comparison configurations.

No sampler changes. The native framework owns instruction prefill and final heads.
Controllers are request-local; only C3ache's explicit state spans chunks.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq, from_und_gen_splits
from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.unified_mot import _run_mlp
from piper_cosmos.cosmos3.k112_sparse import K112Controller
from piper_cosmos.cosmos3.nano_version1_sparse import build_core_stable_plan, selected_gen_positions


ASI_SELECTION_MODES = {
    "asi": "core80_stable104",
    "asi_step_refresh": "core80_stable104_step_refresh",
    "asi_all_layers": "core80_stable104_all_layers",
    "asi_core184": "core_only184",
    "asi_stable184": "stable_only184",
    "asi_random184": "random_only184",
}


def _mask_pair_metrics(first, second):
    """Return per-row overlap metrics for two equally shaped boolean masks."""

    if first.shape != second.shape:
        raise ValueError(f"Mask shape changed: {tuple(first.shape)} -> {tuple(second.shape)}")
    if first.dtype != torch.bool or second.dtype != torch.bool:
        raise ValueError("Mask similarity expects boolean masks")
    if first.ndim == 1:
        first = first.unsqueeze(0)
        second = second.unsqueeze(0)
    if first.ndim != 2:
        raise ValueError(f"Mask similarity expects [rows,tokens], got {tuple(first.shape)}")
    intersection = (first & second).sum(dim=-1)
    union = (first | second).sum(dim=-1)
    first_count = first.sum(dim=-1)
    second_count = second.sum(dim=-1)
    denominator = torch.minimum(first_count, second_count).clamp_min(1)
    jaccard = intersection.float() / union.clamp_min(1).float()
    retention = intersection.float() / denominator.float()
    changed = (first ^ second).sum(dim=-1)
    return {
        "intersection": intersection.tolist(),
        "union": union.tolist(),
        "jaccard": jaccard.tolist(),
        "retention": retention.tolist(),
        "changed": changed.tolist(),
        "jaccard_mean": float(jaccard.mean()),
        "retention_mean": float(retention.mean()),
        "changed_mean": float(changed.float().mean()),
    }


def summarize_asi_mask_history(history):
    """Compare consecutive denoise-step Core, Stable, and execution masks."""

    def compare(first, second):
        return {
            "step_from": int(first["step"]),
            "step_to": int(second["step"]),
            "core": _mask_pair_metrics(first["core_masks"], second["core_masks"]),
            "stable": _mask_pair_metrics(first["stable_mask"], second["stable_mask"]),
            "execution": _mask_pair_metrics(first["execution_mask"], second["execution_mask"]),
        }

    consecutive = [compare(first, second) for first, second in zip(history, history[1:])]
    anchor = [compare(history[0], item) for item in history[1:]] if history else []

    def aggregate(pairs):
        return {
            kind: {
                metric: (sum(pair[kind][metric] for pair in pairs) / len(pairs) if pairs else None)
                for metric in ("jaccard_mean", "retention_mean", "changed_mean")
            }
            for kind in ("core", "stable", "execution")
        }

    return {
        "consecutive_pairs": consecutive,
        "consecutive_aggregate": aggregate(consecutive),
        "step0_anchor_pairs": anchor,
        "step0_anchor_aggregate": aggregate(anchor),
    }


def nano_concat_view_regions(token_grid, device=None):
    """Assign each spatial patch to high/left-wrist/right-wrist by patch center."""

    if len(token_grid) != 3:
        raise ValueError(f"Expected [time,height,width] token grid, got {token_grid}")
    _, height, width = map(int, token_grid)
    rows = (torch.arange(height, device=device).float() + 0.5) / height
    cols = (torch.arange(width, device=device).float() + 0.5) / width
    bottom = rows[:, None] >= (2.0 / 3.0)
    right = cols[None, :] >= 0.5
    regions = {
        "cam_high": (~bottom).expand(height, width),
        "cam_left_wrist": bottom.expand(height, width) & (~right).expand(height, width),
        "cam_right_wrist": bottom.expand(height, width) & right.expand(height, width),
    }
    coverage = torch.stack(list(regions.values())).sum(dim=0)
    if not bool((coverage == 1).all()):
        raise RuntimeError("Nano concat-view regions do not partition the token grid")
    return regions


def summarize_execution_masks_by_view(history, token_grid):
    """Report directional mask retention for each Nano concat-view region."""

    if not history:
        return {"assignment": "patch_center", "regions": {}, "consecutive_pairs": []}
    regions = nano_concat_view_regions(token_grid, history[0]["execution_mask"].device)

    def selected_counts(item):
        mask = item["execution_mask"].reshape(item["execution_mask"].shape[0], *map(int, token_grid[1:]))
        return {name: mask[:, region].sum(dim=-1).tolist() for name, region in regions.items()}

    pairs = []
    for first, second in zip(history, history[1:]):
        first_mask = first["execution_mask"].reshape(first["execution_mask"].shape[0], *map(int, token_grid[1:]))
        second_mask = second["execution_mask"].reshape(second["execution_mask"].shape[0], *map(int, token_grid[1:]))
        views = {}
        for name, region in regions.items():
            a, b = first_mask[:, region], second_mask[:, region]
            intersection = (a & b).sum(dim=-1)
            union = (a | b).sum(dim=-1)
            count_from = a.sum(dim=-1)
            count_to = b.sum(dim=-1)
            retention_from = intersection.float() / count_from.clamp_min(1).float()
            retention_to = intersection.float() / count_to.clamp_min(1).float()
            jaccard = intersection.float() / union.clamp_min(1).float()
            views[name] = {
                "selected_from": count_from.tolist(),
                "selected_to": count_to.tolist(),
                "intersection": intersection.tolist(),
                "retention_from": retention_from.tolist(),
                "retention_to": retention_to.tolist(),
                "jaccard": jaccard.tolist(),
                "selected_from_mean": float(count_from.float().mean()),
                "selected_to_mean": float(count_to.float().mean()),
                "intersection_mean": float(intersection.float().mean()),
                "retention_from_mean": float(retention_from.mean()),
                "retention_to_mean": float(retention_to.mean()),
                "jaccard_mean": float(jaccard.mean()),
            }
        pairs.append({"step_from": int(first["step"]), "step_to": int(second["step"]), "views": views})
    return {
        "assignment": "patch_center",
        "layout": "cam_high=top_2/3,left_wrist=bottom_left,right_wrist=bottom_right",
        "regions": {name: int(region.sum()) for name, region in regions.items()},
        "selected_tokens_by_step": [
            {"step": int(item["step"]), "views": selected_counts(item)} for item in history
        ],
        "consecutive_pairs": pairs,
    }


def qkv(layer, x, cos, sin, query_ids=None):
    a = layer.self_attn
    x = layer.input_layernorm_moe_gen(x)
    qx = x if query_ids is None else x.index_select(0, query_ids)
    q = a.q_norm_moe_gen(a.q_proj_moe_gen(qx).view(-1, a.num_attention_heads, a.head_dim))
    k = a.k_norm_moe_gen(a.k_proj_moe_gen(x).view(-1, a.num_key_value_heads, a.head_dim))
    v = a.v_proj_moe_gen(x).view(-1, a.num_key_value_heads, a.head_dim)
    if query_ids is None:
        q, k = a._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    else:
        q, _ = a._apply_rotary_pos_emb(q, q[:, :0], cos[query_ids], sin[query_ids], unsqueeze_dim=1)
        _, k = a._apply_rotary_pos_emb(k[:, :0], k, cos, sin, unsqueeze_dim=1)
    return q, k, v


def layer_full(layer, x, cos, sin, ku, vu, future, actions, kind):
    q, k, v = qkv(layer, x, cos, sin)
    a = layer.self_attn
    if kind == "toca":
        from piper_cosmos.cosmos3.nano_cache_vendor.toca_joint_attention import joint_attention_score
        o, score = joint_attention_score(q, ku[0], k, vu[0], v, future, a.scaling)
    else:
        o = attention(query=q[None], key=torch.cat((ku, k[None]), 1),
                      value=torch.cat((vu, v[None]), 1), is_causal=False)[0]
        score = x.new_empty((0,), dtype=torch.float32)
    ar = a.o_proj_moe_gen(o.flatten(-2, -1))
    z = x + ar
    mr, _ = _run_mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(z))
    y = z + mr
    if kind == "profile":
        qa = q.index_select(0, actions)
        _, lse = attention(query=qa[None], key=torch.cat((ku, k[None]), 1),
                           value=torch.cat((vu, v[None]), 1), scale=a.scaling, return_lse=True)
        heads, width = qa.shape[1:]
        spatial = future.numel() // 8
        qs = qa.float().reshape(8, 4, heads, width).permute(0, 2, 1, 3).contiguous()
        ks = k.index_select(0, future).float().repeat_interleave(heads // k.shape[1], 1)
        ks = ks.reshape(8, spatial, heads, width).permute(0, 2, 3, 1).contiguous()
        logits = torch.matmul(qs * a.scaling, ks)
        p = torch.exp(logits - lse.reshape(8, 4, heads).permute(0, 2, 1).unsqueeze(-1).float()).mean(1)
        weights = p.new_tensor([1/6, 1/3, 1/3, 1/6])
        score = (p * weights[None, :, None]).sum(1)
    if kind == "toca":
        return y, ar.index_select(0, future), mr.index_select(0, future), score
    return y, score


def layer_cached(layer, x, cos, sin, ku, vu, future, protected, fresh, old_a, old_m):
    q, k, v = qkv(layer, x, cos, sin, protected)
    o = attention(query=q[None], key=torch.cat((ku, k[None]), 1),
                  value=torch.cat((vu, v[None]), 1), is_causal=False)[0]
    a = layer.self_attn.o_proj_moe_gen(o.flatten(-2, -1))
    ar = torch.empty_like(x).index_copy(0, protected, a).index_copy(0, future, old_a)
    z = x + ar
    ids = torch.cat((protected, future.index_select(0, fresh)))
    m, _ = _run_mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(z.index_select(0, ids)))
    next_m = old_m.index_copy(0, fresh, m[len(protected):])
    mr = torch.empty_like(x).index_copy(0, protected, m[:len(protected)]).index_copy(0, future, next_m)
    return z + mr, next_m


def fresh_indices(scores, age, block, blocks, future_count):
    count = int(.25 * (1.5 - block / max(blocks - 1, 1)) * future_count)
    importance = torch.nn.functional.normalize(scores.float(), dim=0) + .25 * age / 2
    ids = importance.argsort(descending=True, stable=True)[:count]
    return ids, (age + 1).index_fill(0, ids, 0)


class NanoCacheController(K112Controller):
    def __init__(self, net, strategy, *, compiled=False, graphs=False, c3_state=None, chunk=0):
        if strategy not in ("dense_adapter", "toca", "worldcache", "c3ache", *ASI_SELECTION_MODES):
            raise ValueError(strategy)
        if graphs and not compiled:
            raise ValueError("Graph requires compile")
        super().__init__(torch=torch, net=net, guidance=3., num_steps=4,
                         expected_blocks=range(len(net.language_model.model.layers)))
        self.strategy, self.compiled, self.graphs = strategy, compiled, graphs
        self.asi_selection = ASI_SELECTION_MODES.get(strategy)
        self.asi_step_refresh = strategy == "asi_step_refresh"
        self.asi_plan_history = []
        self.cache, self.profiles, self.events = {}, [], []
        self.c3_state = {} if c3_state is None else c3_state
        self.refresh = chunk % 2 == 0 or not self.c3_state
        self.chunk = chunk
        self._kernel_cache = getattr(self.model, "_nano_cache_kernels", None)
        if self._kernel_cache is None:
            self._kernel_cache = {}
            self.model._nano_cache_kernels = self._kernel_cache
        self.positions = None
        self.world = None

    def __enter__(self):
        super().__enter__()
        if self.strategy == "worldcache":
            from piper_cosmos.cosmos3.nano_cache_vendor.worldcache import WorldCacheRequest, WorldCacheConfig
            self.world = WorldCacheRequest(WorldCacheConfig())
            self._saved_forward = self.net.forward
            self._had_forward = "forward" in self.net.__dict__
            self.net.forward = self.world_forward
        return self

    def __exit__(self, *exc):
        if self.world is not None:
            if self._had_forward:
                self.net.forward = self._saved_forward
            else:
                del self.net.forward
        if exc[0] is not None:
            self.c3_state.clear()
        super().__exit__(*exc)

    def world_forward(self, *args, **kwargs):
        if self._current is None:
            return self._saved_forward(*args, **kwargs)
        branch = self._current["branch"]
        if branch == "conditional":
            self.world.begin_step()
        packed = args[0] if args else kwargs["packed_seq"]
        return self.world.evaluate(branch, lambda: self._saved_forward(*args, **kwargs), packed,
                                   self.net.latent_patch_size, 14, self.net.llm2vae)

    def _network_pre_hook(self, module, args, kwargs):
        super()._network_pre_hook(module, args, kwargs)
        if self._current is None:
            return
        if self._current["step"] >= 4:
            raise RuntimeError("Unexpected extra denoising forward")
        # Fail closed: no controller may bring UND weights back to CUDA.
        memory = kwargs.get("memory", args[1] if len(args) > 1 else None)
        if memory is None or not memory.is_gen_only():
            raise RuntimeError("Nano adapters require completed Reasoner prefill and populated K/V")
        if self.strategy == "c3ache":
            signature = (id(self.model), id(memory), repr(self._layout))
            key = ("signature", self._current["branch"])
            previous = self.c3_state.get(key)
            if previous is not None and previous != signature:
                self.c3_state.clear()
                self.refresh = True
            self.c3_state[key] = signature

    def _network_post_hook(self, module, args, kwargs, output):
        if self.stack_active:
            raise RuntimeError("GEN stack was not restored before final heads")
        self._current = None

    def begin_stack(self, **kwargs):
        if self._current is None:
            return
        super().begin_stack(**kwargs)
        if self.positions is None:
            device = get_gen_seq(kwargs["hidden_states"]).device
            future = sum((self._layout["latent_positions"][f"L{f}"] for f in range(1, 9)), [])
            protected = sorted(self._layout["latent_positions"]["L0"] + self._layout["action_positions"])
            actions = self._layout["action_positions"][1:]
            if len(actions) != 32:
                raise ValueError("Expected condition q0 plus 32 predicted action tokens")
            self.positions = [torch.tensor(ids, device=device, dtype=torch.long) for ids in (future, protected, actions)]
        if self.strategy == "c3ache":
            key = self._current["branch"], self._current["step"]
            x = get_gen_seq(kwargs["hidden_states"])
            # Each benchmark state is bound to one instruction, sample layout and seed.
            self.c3_hit = not self.refresh and key[1] < 2
            if self.c3_hit:
                r = self.c3_state[key]
                if r.shape != x.shape or r.dtype != x.dtype or r.device != x.device:
                    raise RuntimeError("C3ache residual signature changed; reset the episode state")
                self.c3_output = from_und_gen_splits(get_und_seq(kwargs["hidden_states"]), x + r, kwargs["hidden_states"])

    def kernel(self, block, kind):
        key = block, kind, self.compiled, self.graphs
        if key not in self._kernel_cache:
            fn = partial(layer_cached, self.layers[block]) if kind == "cached" else partial(layer_full, self.layers[block], kind=kind)
            if self.compiled:
                fn = torch.compile(fn, fullgraph=True, dynamic=False, mode="reduce-overhead" if self.graphs else "default")
            self._kernel_cache[key] = fn
        return self._kernel_cache[key]

    def run_layer(self, *, block, decoder_layer, hidden_states, attention_mask, memory_value, gen_only):
        if not gen_only or memory_value.und_k is None:
            raise RuntimeError("GEN adapter must not execute Reasoner weights")
        step, branch = self._current["step"], self._current["branch"]
        future, protected, actions = self.positions
        dense_anchor = self.asi_selection is not None and branch == "conditional" and (
            step == 0 or self.asi_step_refresh
        )
        profile = dense_anchor and self.asi_selection != "random_only184"
        if self.strategy == "c3ache" and self.c3_hit:
            return self.c3_output, {}, None
        if self.asi_selection is not None and not dense_anchor and block == 0:
            hidden_states, self._position_embeddings = self._slice_pack_and_rope(hidden_states, self._selected_positions)
            self._active_positions = self._selected_positions
            self._pack_count += 1
        x = get_gen_seq(hidden_states)
        if len(x) != hidden_states["_num_full_tokens"]:
            raise RuntimeError("Nano tensor kernels require real unpadded GEN rows")
        if getattr(attention_mask, "is_three_way", False) or getattr(attention_mask, "control_stream_token_ranges", None) is not None:
            raise ValueError("Only batch-one full GEN attention is supported")
        cos, sin = [get_gen_seq(p) for p in self._position_embeddings]
        key = branch, block
        if self.strategy == "toca" and step not in (0, 2):
            entry = self.cache[key]
            ids, entry["age"] = fresh_indices(entry["score"], entry["age"], block, len(self.layers), len(future))
            y, m = self.kernel(block, "cached")(x, cos, sin, memory_value.und_k, memory_value.und_v,
                                              future, protected, ids, entry["a"], entry["m"])
            entry["m"] = m.detach().clone()
            mode = "cached"
        else:
            kind = "profile" if profile else "toca" if self.strategy == "toca" else "dense"
            result = self.kernel(block, kind)(x, cos, sin, memory_value.und_k, memory_value.und_v, future, actions)
            y = result[0]
            if profile:
                self.profiles.append({"block": block, "branch": branch, "profiles": result[1].detach().clone()})
            if kind == "toca":
                self.cache[key] = dict(a=result[1].detach().clone(), m=result[2].detach().clone(),
                                       score=result[3].detach().clone(), age=torch.zeros(len(future), device=x.device))
            mode = "sparse" if self.asi_selection is not None and not dense_anchor else "full"
        self.events.append(dict(step=step, branch=branch, block=block, mode=mode, gen_tokens=len(x)))
        self._block_calls += 1
        return from_und_gen_splits(get_und_seq(hidden_states), y, hidden_states), {}, None

    def end_stack(self, hidden_states):
        step, branch = self._current["step"], self._current["branch"]
        if self.asi_selection is not None:
            dense_anchor = branch == "conditional" and (step == 0 or self.asi_step_refresh)
            if dense_anchor:
                if self.asi_selection == "random_only184":
                    spatial = len(self._layout["latent_positions"]["L1"])
                    generator = torch.Generator(device=get_gen_seq(hidden_states).device)
                    generator.manual_seed(0)
                    random_scores = torch.rand(
                        (8, spatial), device=get_gen_seq(hidden_states).device, generator=generator
                    )
                    random_ids = random_scores.topk(184, dim=-1, sorted=False).indices
                    execution_mask = torch.zeros_like(random_scores, dtype=torch.bool)
                    execution_mask.scatter_(1, random_ids, True)
                    self.plan = {
                        "execution_mask": execution_mask,
                        "core_blocks": torch.empty(0, dtype=torch.long, device=random_scores.device),
                    }
                else:
                    core_budget, stable_budget = {
                        "core80_stable104": (80, 104),
                        "core80_stable104_step_refresh": (80, 104),
                        "core80_stable104_all_layers": (80, 104),
                        "core_only184": (184, 0),
                        "stable_only184": (0, 184),
                    }[self.asi_selection]
                    self.plan = build_core_stable_plan(
                        torch=torch,
                        profile_records=self.profiles,
                        expected_blocks=self.expected_blocks,
                        core_token_budget=core_budget,
                        stable_token_budget=stable_budget,
                        core_layer_mode=(
                            "all" if self.asi_selection == "core80_stable104_all_layers" else "informative"
                        ),
                    )
                if self.asi_selection != "random_only184":
                    self.asi_plan_history.append(
                        {
                            "step": step,
                            "core_masks": self.plan["core_masks"].detach().clone(),
                            "stable_mask": self.plan["stable_mask"].detach().clone(),
                            "execution_mask": self.plan["execution_mask"].detach().clone(),
                            "core_blocks": self.plan["core_blocks"].detach().clone(),
                        }
                    )
                self.profiles.clear()
                self._selected_positions = selected_gen_positions(
                    torch,
                    self._layout,
                    self.plan["execution_mask"],
                    get_gen_seq(hidden_states).device,
                )
                self._selected_device = get_gen_seq(hidden_states).device
            else:
                y = self._side_buffer.index_copy(0, self._active_positions, get_gen_seq(hidden_states))
                hidden_states = from_und_gen_splits(get_und_seq(hidden_states), y, self._original_pack)
                self._restore_count += 1
        if self.strategy == "c3ache" and step < 2 and not self.c3_hit:
            self.c3_state[branch, step] = (get_gen_seq(hidden_states) - self._side_buffer).detach().clone()
        self._completed_stacks += 1
        self.abort_stack()
        return hidden_states

    def finish(self):
        if self._forward_index != 8:
            raise RuntimeError(f"Expected 8 CFG forwards, got {self._forward_index}")
        expected = 6 if self.strategy == "worldcache" else 8
        if self._completed_stacks != expected:
            raise RuntimeError(f"Incomplete stack count: {self._completed_stacks}")
        result = dict(strategy=self.strategy, compile=self.compiled, graphs_requested=self.graphs,
                      actual_layer_calls=self._block_calls, layer_count=len(self.layers),
                      gen_tokens=self._layout["num_gen_tokens"], token_grid=self._layout["latent_shape_thw"],
                      events=self.events, reasoner_cache_required=True)
        if self.asi_selection is not None:
            core_budget, stable_budget = {
                "core80_stable104": (80, 104),
                "core80_stable104_step_refresh": (80, 104),
                "core80_stable104_all_layers": (80, 104),
                "core_only184": (184, 0),
                "stable_only184": (0, 184),
                "random_only184": (0, 0),
            }[self.asi_selection]
            result.update(core_blocks=self.plan["core_blocks"].tolist(), core=core_budget, stable=stable_budget,
                          core_layer_mode=self.plan.get("core_layer_mode", "random"),
                          asi_selection=self.asi_selection, fixed_tokens_per_future=184,
                          sparse_gen_tokens=len(self._selected_positions), sparse_stacks=self._restore_count,
                          dense_stacks=8 - self._restore_count,
                          mask_refresh_steps=[int(item["step"]) for item in self.asi_plan_history],
                          core_blocks_by_step=[item["core_blocks"].tolist() for item in self.asi_plan_history],
                          mask_similarity=summarize_asi_mask_history(self.asi_plan_history),
                          view_mask_similarity=summarize_execution_masks_by_view(
                              self.asi_plan_history, self._layout["latent_shape_thw"]
                          ))
        if self.world:
            result["worldcache"] = self.world.finish()
        if self.strategy == "c3ache":
            result.update(refresh=self.refresh, chunk=self.chunk, cache_hits=0 if self.refresh else 4)
        return result
