"""Request-local WorldCache D/D/D/C for joint Edge video/action prediction.

Algorithm reference: FofGofx/WorldCache b921368f7dfbd7ca5d7cfcd0276fec1d1cfd7d91.
This four-step adaptation lowers FULL warmup to three and removes final-FULL
protection. It is NOT the official scheduler, not token pruning, and not CAS:
with one final CACHE there is no later step on which CAS could trigger a FULL.
No model or training imports: the adapter operates on packed metadata by contract.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class WorldCacheConfig:
    percentile_stable: float = 0.30
    percentile_chaotic: float = 0.70
    n_max: int = 6
    eps: float = 1e-8
    check_finite: bool = True
    compile_compatible: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.percentile_stable < self.percentile_chaotic <= 1:
            raise ValueError("WorldCache requires 0 <= stable < chaotic <= 1")
        if self.n_max < 1 or not 0 < self.eps < float("inf"):
            raise ValueError("WorldCache requires n_max >= 1 and finite eps > 0")


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """[C,F,H,W] -> [F*H/p*W/p, p*p*C], same order as Cosmos3VFMNetwork."""
    c, f, h, w = x.shape
    if p < 1 or h % p or w % p:
        raise ValueError("WorldCache requires an unpadded grid divisible by latent_patch_size")
    return x.reshape(c, f, h // p, p, w // p, p).permute(1, 2, 4, 3, 5, 0).reshape(-1, p * p * c)


def unpatchify(x: torch.Tensor, shape: tuple[int, ...], p: int) -> torch.Tensor:
    c, f, h, w = shape
    return x.reshape(f, h // p, w // p, p, p, c).permute(5, 0, 1, 3, 2, 4).reshape(c, f, h, w)


def _finite(x: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"WorldCache NaN/Inf: {name}")


def curvature_and_slopes(
    history: list[torch.Tensor], steps: list[int], eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(history) != 3 or len(steps) != 3 or not steps[0] < steps[1] < steps[2]:
        raise ValueError("WorldCache requires three distinct ordered FULL histories")
    # FP32 avoids BF16 norm/squared-norm overflow and preserves eps.
    y0, y1, y2 = (y.float() for y in history)
    prev = (y1 - y0) / (steps[1] - steps[0])
    curr = (y2 - y1) / (steps[2] - steps[1])
    accel = (curr - prev) / (steps[2] - steps[1])
    curvature = torch.linalg.vector_norm(accel, dim=-1) / (curr.square().sum(-1) + eps)
    return curvature, curr, prev


def heterogeneous_predict(
    latest: torch.Tensor,
    curr: torch.Tensor,
    prev: torch.Tensor,
    curvature: torch.Tensor,
    thresholds: torch.Tensor,
    config: WorldCacheConfig,
    k: int = 1,
) -> torch.Tensor:
    stable = curvature < thresholds[0]
    chaotic = curvature >= thresholds[1]
    linear = ~(stable | chaotic)
    x = min(k / config.n_max, 1.0)
    alpha = 3 * x * x - 2 * x * x * x
    base = latest.float()
    pred = torch.where(linear[:, None], base + k * curr, base)
    return torch.where(chaotic[:, None], base + k * ((1 - alpha) * curr + alpha * prev), pred)


@dataclass
class OutputLayout:
    vision_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    future_ids: torch.Tensor
    action_ids: torch.Tensor
    action_dim: int
    patch_size: int
    vision_dtype: torch.dtype
    action_dtype: torch.dtype

    @classmethod
    def from_packed(cls, packed: Any, patch_size: int, action_dim: int | None) -> OutputLayout:
        if packed.vision is None or packed.action is None or getattr(packed, "sound", None) is not None:
            raise ValueError("WorldCache supports joint vision/action policy only, no sound")
        if len(packed.vision.tokens) != 1 or len(packed.action.tokens) != 1:
            raise ValueError("WorldCache supports batch=1 and one vision item only")
        vision, action = packed.vision.tokens[0], packed.action.tokens[0]
        if vision.ndim not in (4, 5) or (vision.ndim == 5 and vision.shape[0] != 1) or action.ndim != 2:
            raise ValueError("WorldCache expected vision [1,C,T,H,W] or [C,T,H,W], action [T,D]")
        c, t, h, w = vision.shape[-4:]
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        vm = packed.vision.condition_mask[0].reshape(-1)
        am = packed.action.condition_mask[0].reshape(-1)
        if vm.numel() != t or am.numel() != action.shape[0]:
            raise ValueError("WorldCache requires per-frame vision and per-row action condition masks")
        if not bool(((vm == 0) | (vm == 1)).all()) or not bool(((am == 0) | (am == 1)).all()):
            raise ValueError("WorldCache requires binary condition masks")
        future_ids = (vm == 0).nonzero().flatten()
        action_ids = (am == 0).nonzero().flatten()
        # The admitted experiment is L0 + L1..L8, q0 + 32 generated actions.
        if t != 9 or not torch.equal(future_ids, torch.arange(1, 9, device=vm.device)):
            raise ValueError("WorldCache D/D/D/C requires L0 conditioned and L1..L8 generated")
        if action.shape[0] != 33 or not torch.equal(action_ids, torch.arange(1, 33, device=am.device)):
            raise ValueError("WorldCache D/D/D/C requires q0 conditioned and q1..q32 generated")
        action_dim = action.shape[1] if action_dim is None else int(action_dim)
        if not 0 < action_dim <= action.shape[1]:
            raise ValueError("Invalid raw action dimension")
        return cls(
            tuple(vision.shape),
            tuple(action.shape),
            future_ids.clone(),
            action_ids.clone(),
            action_dim,
            patch_size,
            vision.dtype,
            action.dtype,
        )

    def encode(self, output: dict[str, Any], projected_vision: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if len(output["preds_vision"]) != 1 or len(output["preds_action"]) != 1:
            raise ValueError("WorldCache output batch changed")
        vision, action = output["preds_vision"][0], output["preds_action"][0]
        if tuple(vision.shape) != self.vision_shape or tuple(action.shape) != self.action_shape:
            raise ValueError("WorldCache output layout changed within request")
        self.vision_dtype, self.action_dtype = vision.dtype, action.dtype
        c, t, h, w = self.vision_shape[-4:]
        vision_unbatched = vision[0] if vision.ndim == 5 else vision
        p = self.patch_size
        spatial = ((h + p - 1) // p) * ((w + p - 1) // p)
        if projected_vision is not None:
            expected = (len(self.future_ids) * spatial, p * p * c)
            if tuple(projected_vision.shape) != expected:
                raise ValueError(f"Unexpected native llm2vae output {projected_vision.shape}; expected {expected}")
            vision_tokens = projected_vision
        else:
            if h % p or w % p:
                raise ValueError("Padded grids require native llm2vae outputs; never reconstruct padding with zeros")
            vision_tokens = patchify(vision_unbatched[:, self.future_ids], p)
        return {
            "vision": vision_tokens,
            "action": action[self.action_ids, : self.action_dim],
        }

    def decode(self, output: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        c, t, h, w = self.vision_shape[-4:]
        p = self.patch_size
        padded_h, padded_w = ((h + p - 1) // p) * p, ((w + p - 1) // p) * p
        vision = output["vision"].new_zeros(self.vision_shape, dtype=self.vision_dtype)
        vision_unbatched = vision[0] if vision.ndim == 5 else vision
        restored = unpatchify(output["vision"], (c, len(self.future_ids), padded_h, padded_w), p)
        vision_unbatched[:, self.future_ids] = restored[:, :, :h, :w].to(self.vision_dtype)
        action = output["action"].new_zeros(self.action_shape, dtype=self.action_dtype)
        action[self.action_ids, : self.action_dim] = output["action"].to(self.action_dtype)
        # Conditioned rows and padded action dimensions remain exactly zero;
        # _get_velocity still applies the original masks after this adapter.
        return {"preds_vision": [vision], "preds_action": [action]}


@dataclass
class BranchHistory:
    layout: OutputLayout
    outputs: list[dict[str, torch.Tensor]] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)


class WorldCacheRequest:
    """Exactly two branches x four steps. Construct afresh for EVERY request."""

    branches = ("conditional", "unconditional")

    def __init__(self, config: WorldCacheConfig):
        self.config = config
        self.step = -1
        self.seen: set[str] = set()
        self.histories: dict[str, BranchHistory] = {}
        self.events: list[dict[str, Any]] = []

    def begin_step(self) -> None:
        if self.step >= 0 and self.seen != set(self.branches):
            raise RuntimeError("WorldCache requires both CFG branches once per step")
        if self.step >= 3:
            raise RuntimeError("WorldCache D/D/D/C supports exactly four solver calls")
        self.step += 1
        self.seen.clear()

    def evaluate(
        self,
        branch: str,
        compute: Callable[[], dict[str, Any]],
        packed: Any,
        patch_size: int,
        action_dim: int | None,
        vision_projection: Any | None = None,
        projection_from_output: bool = False,
    ) -> dict[str, Any]:
        if self.step < 0 or branch not in self.branches or branch in self.seen:
            raise RuntimeError("Invalid WorldCache step/branch call order")
        mode = "FULL" if self.step < 3 else "CACHE"
        device = packed.vision.tokens[0].device
        scope = (
            torch.cuda.nvtx.range(f"worldcache/step{self.step}/{branch}/{mode}")
            if device.type == "cuda"
            else nullcontext()
        )
        with scope:
            if branch not in self.histories:
                if self.step != 0:
                    raise RuntimeError("Missing request-local WorldCache history")
                self.histories[branch] = BranchHistory(OutputLayout.from_packed(packed, patch_size, action_dim))
            state = self.histories[branch]
            if (
                tuple(packed.vision.tokens[0].shape) != state.layout.vision_shape
                or tuple(packed.action.tokens[0].shape) != state.layout.action_shape
            ):
                raise ValueError("WorldCache input layout changed within request")
            if mode == "FULL":
                projected = []
                hook = None
                if vision_projection is not None and not projection_from_output:
                    hook = vision_projection.register_forward_hook(
                        lambda module, inputs, output: projected.append(output.detach())
                    )
                try:
                    result = compute()  # The ONLY call site for the real Transformer.
                finally:
                    if hook is not None:
                        hook.remove()
                if projection_from_output:
                    # Returned by the compiled head; never recover padded patches
                    # from cropped pixels and never hold a replay-owned buffer.
                    projected.append(result.pop("_worldcache_projection").detach())
                if vision_projection is not None and len(projected) != 1:
                    raise RuntimeError(f"Expected one native vision projection, got {len(projected)}")
                tokens = state.layout.encode(result, projected[0] if projected else None)
                if self.config.check_finite:
                    for name, value in tokens.items():
                        _finite(value, f"step{self.step}/{branch}/FULL/{name}")
                state.outputs.append({name: value.detach().clone() for name, value in tokens.items()})
                state.steps.append(self.step)
                event = {"step": self.step, "branch": branch, "mode": mode}
            else:
                if state.steps != [0, 1, 2]:
                    raise RuntimeError("CACHE requires FULL steps 0, 1, 2 from this branch")
                curves, slopes = {}, {}
                for name in ("vision", "action"):
                    curves[name], curr, prev = curvature_and_slopes(
                        [out[name] for out in state.outputs], state.steps, self.config.eps
                    )
                    slopes[name] = (curr, prev)
                    if self.config.check_finite:
                        _finite(curves[name], f"{branch}/{name}/curvature")
                # Native vector dimensions may differ; scalar scores share global
                # quantiles over generated vision + action tokens, not conditions.
                joint = torch.cat(list(curves.values()))
                thresholds = torch.quantile(
                    joint, joint.new_tensor([self.config.percentile_stable, self.config.percentile_chaotic])
                )
                predicted, counts = {}, {}
                for name, curve in curves.items():
                    curr, prev = slopes[name]
                    predicted[name] = heterogeneous_predict(
                        state.outputs[-1][name], curr, prev, curve, thresholds, self.config
                    )
                    if self.config.check_finite:
                        _finite(predicted[name], f"{branch}/{name}/CACHE")
                    stable, chaotic = curve < thresholds[0], curve >= thresholds[1]
                    counts[name] = {
                        "total": curve.numel(),
                        "stable": int(stable.sum()),
                        "linear": int((~(stable | chaotic)).sum()),
                        "chaotic": int(chaotic.sum()),
                    }
                result = state.layout.decode(predicted)
                if self.config.check_finite:
                    for name, tensors in result.items():
                        _finite(tensors[0], f"{branch}/{name}/cast_to_model_dtype")
                event = {"step": self.step, "branch": branch, "mode": mode, "groups": counts}
                # Never insert an approximate output into FULL history.
            self.events.append(event)
            self.seen.add(branch)
            return result

    def finish(self) -> dict[str, Any]:
        if self.step != 3 or self.seen != set(self.branches) or len(self.events) != 8:
            raise RuntimeError("Incomplete WorldCache D/D/D/C request")
        return {
            "strategy": "worldcache_dddc_joint",
            "schedule": ["D", "D", "D", "C"],
            "config": asdict(self.config),
            "full_forwards": 6,
            "cache_forwards": 2,
            "events": list(self.events),
        }

    def clear(self) -> None:
        self.histories.clear()
