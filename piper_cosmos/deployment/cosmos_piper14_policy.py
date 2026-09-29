"""RTC-facing Piper14 wrapper for Cosmos3 action policy inference."""

from __future__ import annotations

import base64
import io
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
from PIL import Image

from piper_cosmos.cosmos3.domain import PIPER14_DOMAIN_NAME, PIPER14_RAW_ACTION_DIM


IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_PROMPT = "Assemble the mouse's battery."


class CosmosActionBackend(Protocol):
    accepts_concat_view: bool

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one Cosmos policy prediction."""


@dataclass(frozen=True)
class CosmosPiper14PolicyConfig:
    """Configuration for the Piper14 deployment policy wrapper."""

    checkpoint: str = ""
    config_file: str | None = None
    experiment_overrides: tuple[str, ...] = ()
    prompt: str = DEFAULT_PROMPT
    action_horizon: int = 32
    raw_action_dim: int = PIPER14_RAW_ACTION_DIM
    max_action_dim: int = 64
    camera_height: int = 480
    camera_width: int = 640
    resolution: str = "480"
    num_steps: int = 4
    guidance: float = 3.0
    shift: float = 5.0
    fps: int = 30
    seed: int = 0
    format_prompt_as_json: bool = True
    host: str = "127.0.0.1"
    port: int = 8766
    output_dir: str | None = None
    mock_backend: bool = False
    timing: bool = False
    use_torch_compile: bool = True
    use_cuda_graphs: bool = False
    compiled_region: str = "all"
    compile_dynamic: bool = True
    reuse_denoise_packing: bool = True
    allow_torch_sdpa_fallback: bool = False
    k112_enabled: bool = False
    k112_output_dir: str | None = None
    nano_version1_enabled: bool = False
    nano_version1_output_dir: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CosmosPiper14PolicyConfig":
        raw = payload.get("cosmos_piper14", payload)
        if not isinstance(raw, Mapping):
            raise TypeError("Expected a mapping or top-level `cosmos_piper14` mapping.")
        values = {
            field: raw[field] for field in cls.__dataclass_fields__ if field in raw
        }
        return cls(**values)


@dataclass
class Piper14Observation:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    prompt: str


class _SegmentedTimer:
    """Collect synchronized wall-clock timings for one policy request."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self._durations_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def measure(
        self,
        name: str,
        synchronize: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize is not None:
                synchronize()
            self.add(name, (time.perf_counter() - started) * 1000.0)

    def add(self, name: str, duration_ms: float) -> None:
        if not self.enabled:
            return
        self._durations_ms[name] = self._durations_ms.get(name, 0.0) + float(
            duration_ms
        )
        self._counts[name] = self._counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        if not self.enabled:
            return {}
        result = {
            name: {
                "ms": round(duration_ms, 3),
                "count": self._counts[name],
            }
            for name, duration_ms in self._durations_ms.items()
        }
        result["policy.total"] = {
            "ms": round((time.perf_counter() - self.started) * 1000.0, 3),
            "count": 1,
        }
        return result

    def log(self, request_id: int) -> dict[str, dict[str, float | int]]:
        result = self.snapshot()
        if not result:
            return result
        total_ms = float(result["policy.total"]["ms"])
        fields = []
        for name, value in result.items():
            count = int(value["count"])
            count_suffix = f"[{count}]" if count > 1 else ""
            duration_ms = float(value["ms"])
            percentage = 100.0 * duration_ms / total_ms if total_ms > 0 else 0.0
            fields.append(
                f"{name}{count_suffix}={duration_ms:.3f}ms({percentage:.1f}%)"
            )
        print(
            f"[cosmos-piper14-timing] request={request_id} " + " ".join(fields),
            flush=True,
        )
        return result


@contextmanager
def _profile_model_methods(
    model: Any,
    timer: _SegmentedTimer,
    synchronize_cuda: Callable[[], None] | None,
) -> Iterator[None]:
    """Temporarily time Cosmos model stages called by generate_samples_from_batch."""

    if not timer.enabled:
        yield
        return

    originals: list[tuple[str, Any, bool]] = []

    def install(method_name: str, timing_name: str) -> None:
        original = getattr(model, method_name, None)
        if original is None or not callable(original):
            return
        had_instance_attribute = method_name in vars(model)

        def timed_method(*args, **kwargs):
            with timer.measure(timing_name, synchronize_cuda):
                return original(*args, **kwargs)

        try:
            setattr(model, method_name, timed_method)
        except (AttributeError, TypeError):
            return
        originals.append((method_name, original, had_instance_attribute))

    install("get_data_and_condition", "model.prepare.vision_action")
    install("_get_inference_text_tokens", "model.prepare.text")
    install("_pack_input_sequence", "model.prepare.pack")
    install("_prepare_inference_data", "model.prepare.total")
    install("_get_velocity", "model.denoise.velocity")
    install("denoise", "model.denoise.network")
    try:
        yield
    finally:
        for method_name, original, had_instance_attribute in reversed(originals):
            if had_instance_attribute:
                setattr(model, method_name, original)
            else:
                delattr(model, method_name)


@contextmanager
def _reuse_denoise_packed_sequences(model: Any, enabled: bool) -> Iterator[None]:
    """Reuse single-sample sequence metadata across diffusion/CFG forwards.

    The policy server always infers one observation at a time. Within one
    request, text IDs, RoPE positions, attention splits, and condition masks
    remain fixed; only modality tokens and the scalar diffusion timestep
    change. Cosmos normally rebuilds and transfers that metadata for every
    conditional/unconditional forward.
    """

    original = getattr(model, "_pack_input_sequence", None)
    if not enabled or original is None or not callable(original):
        yield
        return

    had_instance_attribute = "_pack_input_sequence" in vars(model)
    cache: dict[tuple[bool, bool, tuple[tuple[int, ...], ...]], Any] = {}

    def cached_pack(*args, **kwargs):
        input_text_indexes = (
            kwargs["input_text_indexes"]
            if "input_text_indexes" in kwargs
            else args[1]
        )
        input_timesteps = (
            kwargs["input_timesteps"] if "input_timesteps" in kwargs else args[3]
        )
        # Batched/per-frame diffusion schedules can assign different timesteps
        # to different samples or frames. Keep the optimization fail-closed for
        # the scalar, batch-size-one policy path only.
        if int(input_timesteps.numel()) != 1:
            return original(*args, **kwargs)

        key = (
            bool(kwargs.get("skip_text_tokens", False)),
            bool(kwargs.get("include_end_of_generation_token", False)),
            tuple(tuple(int(token) for token in row) for row in input_text_indexes),
        )
        packed = cache.get(key)
        if packed is None:
            packed = original(*args, **kwargs)
            cache[key] = packed
            return packed

        timestep_value = float(input_timesteps.reshape(-1)[0].item())
        for modality in (packed.vision, packed.action, packed.sound):
            timesteps = None if modality is None else modality.timesteps
            if timesteps is not None:
                timesteps.fill_(timestep_value)
        return packed

    try:
        setattr(model, "_pack_input_sequence", cached_pack)
        yield
    finally:
        if had_instance_attribute:
            setattr(model, "_pack_input_sequence", original)
        else:
            delattr(model, "_pack_input_sequence")


@contextmanager
def _torch_sdpa_attention_fallback(enabled: bool) -> Iterator[None]:
    """Use PyTorch SDPA only when Cosmos has no compatible dense backend.

    This is an explicit compatibility path for inference hosts without the
    optional FlashAttention/NATTEN packages. It does not override a compatible
    Cosmos backend and intentionally rejects packed varlen attention.
    """

    if not enabled:
        yield
        return

    import torch
    import torch.nn.functional as functional
    from cosmos_framework.model.attention import frontend

    backend_name = "torch_sdpa_fallback"
    original_choose_backend = frontend.choose_backend
    had_backend = backend_name in frontend.BACKEND_MAP
    original_backend = frontend.BACKEND_MAP.get(backend_name)

    def torch_sdpa_attention(
        query,
        key,
        value,
        is_causal=False,
        causal_type=None,
        scale=None,
        cumulative_seqlen_Q=None,
        cumulative_seqlen_KV=None,
        max_seqlen_Q=None,
        max_seqlen_KV=None,
        return_lse=False,
        backend_kwargs=None,
        deterministic=False,
    ):
        del max_seqlen_Q, max_seqlen_KV, backend_kwargs, deterministic
        if cumulative_seqlen_Q is not None or cumulative_seqlen_KV is not None:
            raise RuntimeError("PyTorch SDPA fallback only supports dense attention.")
        if query.shape[0] != 1:
            raise RuntimeError("PyTorch SDPA fallback is limited to batch-one policy inference.")
        if is_causal and query.shape[1] != key.shape[1]:
            raise RuntimeError(
                "PyTorch SDPA fallback requires equal Q/KV lengths for causal attention; "
                f"got {query.shape[1]} and {key.shape[1]} ({causal_type=})."
            )
        if return_lse:
            raise RuntimeError("PyTorch SDPA fallback does not expose logsumexp.")
        query_bhsd = query.transpose(1, 2)
        key_bhsd = key.transpose(1, 2)
        value_bhsd = value.transpose(1, 2)
        result = functional.scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            dropout_p=0.0,
            is_causal=bool(is_causal),
            scale=scale,
            enable_gqa=query_bhsd.shape[1] != key_bhsd.shape[1],
        )
        return result.transpose(1, 2)

    def choose_backend_with_fallback(*args, **kwargs):
        compatible = original_choose_backend(*args, **kwargs)
        if compatible is not None:
            return compatible
        backend = kwargs.get("backend")
        is_varlen = bool(kwargs.get("is_varlen", False))
        requires_grad = bool(kwargs.get("requires_grad", False))
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        supported_dtype = dtype in (torch.float16, torch.bfloat16, torch.float32)
        if (
            backend is None
            and not is_varlen
            and not requires_grad
            and device is not None
            and device.type == "cuda"
            and supported_dtype
        ):
            return backend_name
        return None

    frontend.BACKEND_MAP[backend_name] = torch_sdpa_attention
    frontend.choose_backend = choose_backend_with_fallback
    try:
        yield
    finally:
        frontend.choose_backend = original_choose_backend
        if had_backend:
            frontend.BACKEND_MAP[backend_name] = original_backend
        else:
            del frontend.BACKEND_MAP[backend_name]


class MockCosmosActionBackend:
    """Small deterministic backend for no-robot/no-GPU tests."""

    accepts_concat_view = True

    def __init__(self, action_horizon: int, action_dim: int) -> None:
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        state = np.asarray(
            request.get("state", np.zeros(self.action_dim)), dtype=np.float32
        ).reshape(-1)
        if state.size != self.action_dim:
            state = np.zeros(self.action_dim, dtype=np.float32)
        increments = np.arange(self.action_horizon, dtype=np.float32)[:, None] * 0.001
        return {"action": (state[None, :] + increments).astype(np.float32).tolist()}


class LiberoActionServiceBackend:
    """Adapter around Cosmos Framework's official HTTP action service core.

    Piper14 training stores current qpos as the first action row, followed by
    future actions.  This backend mirrors the official LIBERO server batch
    construction but fills row 0 with the incoming Piper14 state and returns only
    the predicted future rows.
    """

    accepts_concat_view = True

    def __init__(self, config: CosmosPiper14PolicyConfig) -> None:
        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_libero import (
            ActionModelService,
            ActionServerArgs,
        )

        try:
            from cosmos_framework.data.generator.action.transforms import (
                ActionTransformPipeline,
            )
        except ModuleNotFoundError:
            from cosmos_framework.data.vfm.action.transforms import (
                ActionTransformPipeline,
            )

        from piper_cosmos.cosmos3.domain import register_piper14_domain

        self.config = config
        register_piper14_domain()
        checkpoint_kwargs = {"checkpoint_path": str(config.checkpoint)}
        if config.config_file:
            checkpoint_kwargs["config_file"] = str(config.config_file)
        if config.experiment_overrides:
            checkpoint_kwargs["experiment_overrides"] = list(
                config.experiment_overrides
            )
        checkpoint_overrides = CheckpointOverrides(**checkpoint_kwargs)
        if config.k112_enabled and config.nano_version1_enabled:
            raise ValueError("k112_enabled and nano_version1_enabled are mutually exclusive")
        sparse_controller_enabled = config.k112_enabled or config.nano_version1_enabled

        class Piper14ActionServerArgs(ActionServerArgs):
            def build_setup_overrides(self):
                setup_overrides = super().build_setup_overrides()
                setup_overrides.guardrails = False
                setup_overrides.use_torch_compile = bool(
                    config.use_torch_compile and not sparse_controller_enabled
                )
                setup_overrides.use_cuda_graphs = bool(
                    config.use_cuda_graphs and not sparse_controller_enabled
                )
                setup_overrides.compiled_region = config.compiled_region
                setup_overrides.compile_dynamic = bool(config.compile_dynamic)
                return setup_overrides

        server_kwargs: dict[str, Any] = {}
        if config.output_dir:
            server_kwargs["output_dir"] = Path(config.output_dir)
        args = Piper14ActionServerArgs(
            checkpoint=checkpoint_overrides,
            seed=int(config.seed),
            guidance=float(config.guidance),
            num_steps=int(config.num_steps),
            fps=int(config.fps),
            action_chunk_size=int(config.action_horizon),
            raw_action_dim=int(config.raw_action_dim),
            max_action_dim=int(config.max_action_dim),
            **server_kwargs,
        )
        self.service = ActionModelService(args)
        if config.k112_enabled:
            if int(config.num_steps) != 4:
                raise ValueError("K112 requires num_steps=4")
            if float(config.shift) != 5.0:
                raise ValueError("K112 requires shift=5")
            if float(config.guidance) == 1.0:
                raise ValueError("K112 requires conditional/unconditional CFG")
        if config.nano_version1_enabled:
            if int(config.num_steps) != 4:
                raise ValueError("Nano Version1 requires num_steps=4")
            if float(config.shift) != 5.0:
                raise ValueError("Nano Version1 requires shift=5")
            if float(config.guidance) == 1.0:
                raise ValueError("Nano Version1 requires conditional/unconditional CFG")
        self._k112_request_index = 0
        self.last_k112_summary: dict[str, Any] | None = None
        self._nano_version1_request_index = 0
        self.last_nano_version1_summary: dict[str, Any] | None = None
        self.transform = ActionTransformPipeline(
            tokenizer_config=None,
            cfg_dropout_rate=0.0,
            max_action_dim=int(config.max_action_dim),
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
            format_prompt_as_json=bool(config.format_prompt_as_json),
        )
        self.transform.video_resize = _TemporalExpandAfterResize(
            self.transform.video_resize
        )

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        try:
            from cosmos_framework.data.generator.action.domain_utils import (
                get_domain_id,
            )
        except ModuleNotFoundError:
            from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

        timer_value = request.get("_timing")
        timer = (
            timer_value
            if isinstance(timer_value, _SegmentedTimer)
            else _SegmentedTimer(self.config.timing)
        )
        owns_timer = timer is not timer_value
        cuda = getattr(torch, "cuda", None)
        synchronize_cuda = (
            cuda.synchronize if cuda is not None and cuda.is_available() else None
        )

        concat_view = request.get("concat_view")
        if concat_view is None:
            official_request = {
                key: value for key, value in request.items() if key != "_timing"
            }
            with timer.measure("backend.official_service", synchronize_cuda):
                response = self.service.predict_policy(official_request)
            if owns_timer:
                timer.log(request_id=0)
            return response

        with timer.measure("backend.validate"):
            image = ensure_rgb_uint8(concat_view, "concat_view")
            if not image.flags.writeable:
                image = image.copy()
            state = np.asarray(request.get("state"), dtype=np.float32).reshape(-1)
            if state.size != PIPER14_RAW_ACTION_DIM:
                raise ValueError(f"Expected Piper14 state dim 14, got {state.size}")

            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("'prompt' must be a string")
            domain_name = request.get("domain_name")
            if not isinstance(domain_name, str):
                raise ValueError("'domain_name' must be a string")

        future_horizon = int(self.service.cfg.action_chunk_size)
        action_rows = future_horizon + 1
        with timer.measure("backend.first_frame"):
            video = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(1)
        with timer.measure("backend.sample_build"):
            action = torch.zeros(
                (action_rows, PIPER14_RAW_ACTION_DIM), dtype=torch.float32
            )
            action[0, :PIPER14_RAW_ACTION_DIM] = torch.from_numpy(state)
            sample = {
                "ai_caption": prompt,
                "video": video,
                "action": action,
                "conditioning_fps": torch.tensor(
                    self.service.cfg.fps, dtype=torch.long
                ),
                "mode": "policy",
                "domain_id": torch.tensor(get_domain_id(domain_name), dtype=torch.long),
                "viewpoint": "concat_view",
                "additional_view_description": (
                    "The fixed overhead camera is on top; the left wrist camera is bottom-left; "
                    "the right wrist camera is bottom-right."
                ),
                "idle_frames": torch.tensor(0, dtype=torch.long),
                "inference_temporal_expand_after_resize": action_rows,
            }
        with timer.measure("backend.transform"):
            transformed = self.transform(sample, resolution=self.config.resolution)
        with timer.measure("backend.batch_build"):
            batch = build_data_batch_from_sample(transformed)

        lock_started = time.perf_counter()
        sparse_controller = None
        with self.service._lock:
            timer.add("model.lock_wait", (time.perf_counter() - lock_started) * 1000.0)
            with torch.inference_mode():
                with _profile_model_methods(
                    self.service.model, timer, synchronize_cuda
                ):
                    with _reuse_denoise_packed_sequences(
                        self.service.model,
                        self.config.reuse_denoise_packing,
                    ):
                        with _torch_sdpa_attention_fallback(
                            self.config.allow_torch_sdpa_fallback
                        ):
                            with timer.measure(
                                "model.generate.total", synchronize_cuda
                            ):
                                generate_kwargs = {
                                    "guidance": float(self.config.guidance),
                                    "seed": [int(self.config.seed)],
                                    "num_steps": int(self.config.num_steps),
                                    "shift": float(self.config.shift),
                                    "has_negative_prompt": False,
                                }
                                if not self.config.k112_enabled and not self.config.nano_version1_enabled:
                                    samples = self.service.model.generate_samples_from_batch(
                                        batch, **generate_kwargs
                                    )
                                elif self.config.k112_enabled:
                                    from piper_cosmos.cosmos3.k112_sparse import (
                                        K112Controller,
                                        K112VelocityCacheSampler,
                                    )

                                    request_index = self._k112_request_index
                                    self._k112_request_index += 1
                                    output_root = self.config.k112_output_dir
                                    request_dir = (
                                        Path(output_root) / f"request_{request_index:06d}"
                                        if output_root
                                        else None
                                    )
                                    sparse_controller = K112Controller(
                                        torch=torch,
                                        net=self.service.model.net,
                                        guidance=float(self.config.guidance),
                                        num_steps=int(self.config.num_steps),
                                        output_dir=request_dir,
                                    )
                                    generate_kwargs["sampler"] = K112VelocityCacheSampler(
                                        self.service.model.sampler, sparse_controller
                                    )
                                    with sparse_controller:
                                        samples = self.service.model.generate_samples_from_batch(
                                            batch, **generate_kwargs
                                        )
                                else:
                                    from piper_cosmos.cosmos3.nano_version1_graph import (
                                        NanoVersion1CompiledController,
                                    )
                                    from piper_cosmos.cosmos3.nano_version1_sparse import (
                                        NanoVersion1Controller,
                                    )

                                    request_index = self._nano_version1_request_index
                                    self._nano_version1_request_index += 1
                                    output_root = self.config.nano_version1_output_dir
                                    request_dir = (
                                        Path(output_root) / f"request_{request_index:06d}"
                                        if output_root
                                        else None
                                    )
                                    if self.config.use_torch_compile:
                                        sparse_controller = NanoVersion1CompiledController(
                                            torch=torch,
                                            net=self.service.model.net,
                                            guidance=float(self.config.guidance),
                                            num_steps=int(self.config.num_steps),
                                            output_dir=request_dir,
                                            use_cuda_graphs=bool(self.config.use_cuda_graphs),
                                        )
                                    else:
                                        sparse_controller = NanoVersion1Controller(
                                            torch=torch,
                                            net=self.service.model.net,
                                            guidance=float(self.config.guidance),
                                            num_steps=int(self.config.num_steps),
                                            output_dir=request_dir,
                                        )
                                    with sparse_controller:
                                        samples = self.service.model.generate_samples_from_batch(
                                            batch, **generate_kwargs
                                        )
            if sparse_controller is not None:
                summary = sparse_controller.finish()
                if self.config.k112_enabled:
                    self.last_k112_summary = summary
                else:
                    self.last_nano_version1_summary = summary
        with timer.measure("backend.action_output", synchronize_cuda):
            pred_action = samples["action"][0].float().squeeze(0)
            future = (
                pred_action[1 : future_horizon + 1, :PIPER14_RAW_ACTION_DIM]
                .detach()
                .cpu()
                .numpy()
            )
            response = {"action": future.astype(np.float32, copy=False).tolist()}
        if owns_timer:
            timer.log(request_id=0)
        return response


class CosmosPiper14PolicyClient:
    """Local policy object with an RTC-compatible interface."""

    def __init__(
        self,
        config: CosmosPiper14PolicyConfig | Mapping[str, Any],
        backend: CosmosActionBackend | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, CosmosPiper14PolicyConfig)
            else CosmosPiper14PolicyConfig.from_mapping(config)
        )
        if self.config.raw_action_dim != PIPER14_RAW_ACTION_DIM:
            raise ValueError(
                f"Piper14 raw_action_dim must be 14, got {self.config.raw_action_dim}"
            )
        if self.config.camera_height <= 0 or self.config.camera_width <= 0:
            raise ValueError("camera_height and camera_width must be positive")
        if self.config.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.config.compiled_region not in {"all", "language"}:
            raise ValueError(
                "compiled_region must be either 'all' or 'language', got "
                f"{self.config.compiled_region!r}"
            )
        if not self.config.mock_backend and not self.config.checkpoint:
            raise ValueError("A checkpoint path is required for Cosmos inference.")

        self.backend = backend or self._build_backend()
        self.observation: Piper14Observation | None = None
        self.last_timing: dict[str, dict[str, float | int]] = {}
        self._timing_request_id = 0

    def _build_backend(self) -> CosmosActionBackend:
        if self.config.mock_backend:
            return MockCosmosActionBackend(
                self.config.action_horizon, self.config.raw_action_dim
            )
        return LiberoActionServiceBackend(self.config)

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self.observation = self._coerce_observation(obs)

    def get_action(self) -> np.ndarray:
        if self.observation is None:
            raise RuntimeError(
                "Policy observation is empty. Call update_observation(obs) before get_action()."
            )
        timer = _SegmentedTimer(self.config.timing)
        return self._run_timed_inference(self.observation, timer)

    def infer(self, obs: Mapping[str, Any]) -> np.ndarray:
        timer = _SegmentedTimer(self.config.timing)
        with timer.measure("observation.coerce"):
            self.update_observation(obs)
        assert self.observation is not None
        return self._run_timed_inference(self.observation, timer)

    def reset(self) -> None:
        self.observation = None

    def timing(self) -> dict[str, dict[str, float | int]]:
        return {name: dict(values) for name, values in self.last_timing.items()}

    def metadata(self) -> dict[str, Any]:
        return {
            "domain_name": PIPER14_DOMAIN_NAME,
            "raw_action_dim": int(self.config.raw_action_dim),
            "action_horizon": int(self.config.action_horizon),
            "image_keys": list(IMAGE_KEYS),
            "action_type": "absolute_joint_position_command",
            "checkpoint": str(self.config.checkpoint),
            "prompt": str(self.config.prompt),
            "mock_backend": bool(self.config.mock_backend),
            "timing": bool(self.config.timing),
            "camera_height": int(self.config.camera_height),
            "camera_width": int(self.config.camera_width),
            "resolution": str(self.config.resolution),
            "num_steps": int(self.config.num_steps),
            "guidance": float(self.config.guidance),
            "shift": float(self.config.shift),
            "format_prompt_as_json": bool(self.config.format_prompt_as_json),
            "output_dir": self.config.output_dir,
            "use_torch_compile": bool(self.config.use_torch_compile),
            "use_cuda_graphs": bool(self.config.use_cuda_graphs),
            "compiled_region": str(self.config.compiled_region),
            "compile_dynamic": bool(self.config.compile_dynamic),
            "reuse_denoise_packing": bool(self.config.reuse_denoise_packing),
            "allow_torch_sdpa_fallback": bool(
                self.config.allow_torch_sdpa_fallback
            ),
            "k112_enabled": bool(self.config.k112_enabled),
            "k112_strategy": (
                "shared-motion-k112-v1" if self.config.k112_enabled else None
            ),
            "k112_output_dir": self.config.k112_output_dir,
            "nano_version1_enabled": bool(self.config.nano_version1_enabled),
            "nano_version1_strategy": (
                "nano-version1-core64-stable120-k184-live-l0-no-velocity-cache"
                if self.config.nano_version1_enabled
                else None
            ),
            "nano_version1_output_dir": self.config.nano_version1_output_dir,
            "effective_torch_compile": bool(
                self.config.use_torch_compile and not self.config.k112_enabled
            ),
            "effective_cuda_graphs": bool(
                self.config.use_cuda_graphs and not self.config.k112_enabled
            ),
        }

    def build_policy_request(
        self,
        obs: Mapping[str, Any],
        *,
        _timing: _SegmentedTimer | None = None,
    ) -> dict[str, Any]:
        timer = _timing or _SegmentedTimer(False)
        with timer.measure("request.coerce"):
            observation = self._coerce_observation(obs)
        with timer.measure("request.concat_view"):
            concat_view = compose_concat_view(
                observation.cam_high,
                observation.cam_left_wrist,
                observation.cam_right_wrist,
                camera_height=self.config.camera_height,
                camera_width=self.config.camera_width,
            )
        request: dict[str, Any] = {
            "concat_view": concat_view,
            "prompt": observation.prompt,
            "domain_name": PIPER14_DOMAIN_NAME,
            "state": observation.state.astype(np.float32, copy=True),
        }
        # In-process backends consume concat_view directly. Unknown or external
        # backends retain the legacy PNG/base64 request contract.
        if not bool(getattr(self.backend, "accepts_concat_view", False)):
            with timer.measure("request.png_base64"):
                request["image"] = encode_rgb_png_base64(concat_view)
            request["image_size"] = int(concat_view.shape[0])
        return request

    def _run_timed_inference(
        self,
        observation: Piper14Observation,
        timer: _SegmentedTimer,
    ) -> np.ndarray:
        self._timing_request_id += 1
        request_id = self._timing_request_id
        try:
            return self._infer_observation(observation, timer)
        finally:
            self.last_timing = timer.log(request_id)

    def _infer_observation(
        self,
        observation: Piper14Observation,
        timer: _SegmentedTimer,
    ) -> np.ndarray:
        request = self.build_policy_request(
            {
                "images": {
                    "cam_high": observation.cam_high,
                    "cam_left_wrist": observation.cam_left_wrist,
                    "cam_right_wrist": observation.cam_right_wrist,
                },
                "state": observation.state,
                "prompt": observation.prompt,
            },
            _timing=timer,
        )
        request["_timing"] = timer
        with timer.measure("backend.total"):
            response = self.backend.predict_policy(request)
        with timer.measure("policy.action_validate"):
            action = np.asarray(response.get("action", []), dtype=np.float32)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            if action.ndim != 2:
                raise ValueError(
                    f"Backend returned action with shape {action.shape}; expected [T,D]."
                )
            if action.shape[1] < self.config.raw_action_dim:
                raise ValueError(
                    f"Backend returned action dim {action.shape[1]}; expected at least {self.config.raw_action_dim}."
                )
            action = action[:, : self.config.raw_action_dim]
            if action.shape[0] == 0:
                raise ValueError("Backend returned an empty action chunk.")
            if action.shape[0] < self.config.action_horizon:
                tail = np.repeat(
                    action[-1:, :],
                    self.config.action_horizon - action.shape[0],
                    axis=0,
                )
                action = np.concatenate([action, tail], axis=0)
            elif action.shape[0] > self.config.action_horizon:
                action = action[: self.config.action_horizon]
            if not np.isfinite(action).all():
                raise ValueError("Backend returned non-finite action values.")
            return np.ascontiguousarray(action.astype(np.float32, copy=False))

    def _coerce_observation(self, obs: Mapping[str, Any]) -> Piper14Observation:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an `images` mapping.")
        missing = [
            key for key in IMAGE_KEYS if key not in images or images[key] is None
        ]
        if missing:
            raise ValueError(f"Observation is missing required image keys: {missing}")

        if "state" in obs:
            state_value = obs["state"]
        elif "qpos" in obs:
            state_value = obs["qpos"]
        else:
            raise ValueError("Observation must contain `state` or `qpos`.")
        state = np.asarray(state_value, dtype=np.float32).reshape(-1)
        if state.size != PIPER14_RAW_ACTION_DIM:
            raise ValueError(f"Expected state/qpos dim 14, got {state.size}")

        return Piper14Observation(
            cam_high=ensure_rgb_uint8(images["cam_high"], "images.cam_high"),
            cam_left_wrist=ensure_rgb_uint8(
                images["cam_left_wrist"], "images.cam_left_wrist"
            ),
            cam_right_wrist=ensure_rgb_uint8(
                images["cam_right_wrist"], "images.cam_right_wrist"
            ),
            state=np.ascontiguousarray(state),
            prompt=str(obs.get("prompt", self.config.prompt)),
        )


def ensure_rgb_uint8(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [H,W,3], got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image_f = image.astype(np.float32)
    if image_f.size and float(np.nanmax(image_f)) <= 1.0:
        image_f = image_f * 255.0
    return np.ascontiguousarray(np.clip(image_f, 0, 255).astype(np.uint8))


def resize_rgb_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    pil = Image.fromarray(image, mode="RGB")
    resized = pil.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).copy()


def compose_concat_view(
    cam_high: Any,
    cam_left_wrist: Any,
    cam_right_wrist: Any,
    *,
    camera_height: int = 480,
    camera_width: int = 640,
) -> np.ndarray:
    high = resize_rgb_uint8(
        ensure_rgb_uint8(cam_high, "cam_high"), camera_height, camera_width
    )
    left = resize_rgb_uint8(
        ensure_rgb_uint8(cam_left_wrist, "cam_left_wrist"), camera_height, camera_width
    )
    right = resize_rgb_uint8(
        ensure_rgb_uint8(cam_right_wrist, "cam_right_wrist"),
        camera_height,
        camera_width,
    )

    half_h = max(1, camera_height // 2)
    half_w = max(1, camera_width // 2)
    left = resize_rgb_uint8(left, half_h, half_w)
    right = resize_rgb_uint8(right, half_h, half_w)
    bottom = np.concatenate([left, right], axis=1)
    return np.ascontiguousarray(np.concatenate([high, bottom], axis=0))


def encode_rgb_png_base64(image: np.ndarray) -> str:
    image = ensure_rgb_uint8(image, "image")
    buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _TemporalExpandAfterResize:
    """Resize one condition frame, then restore the logical video length as a view."""

    def __init__(self, resize: Any) -> None:
        if not callable(resize):
            raise TypeError("ActionTransformPipeline.video_resize must be callable")
        self._resize = resize

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resize, name)

    def __call__(
        self, data_dict: dict[str, Any], resolution: str | int | None
    ) -> dict[str, Any]:
        import torch

        data_dict = self._resize(data_dict, resolution)
        temporal_expand = data_dict.pop("inference_temporal_expand_after_resize", None)
        if temporal_expand is None:
            return data_dict

        temporal_expand = int(temporal_expand)
        video = data_dict.get("video")
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 4
            or video.shape[1] != 1
        ):
            shape = None if not isinstance(video, torch.Tensor) else tuple(video.shape)
            raise ValueError(
                "inference_temporal_expand_after_resize requires video shape [C,1,H,W] "
                f"after resize, got {shape}"
            )
        if temporal_expand < 1:
            raise ValueError(
                "inference_temporal_expand_after_resize must be positive, "
                f"got {temporal_expand}"
            )
        data_dict["video"] = video.expand(-1, temporal_expand, -1, -1)
        return data_dict


def build_data_batch_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from cosmos_framework.data.generator.joint_dataloader import (
            IterativeJointDataLoader,
        )
    except ModuleNotFoundError:
        from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader

    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        else:
            data_batch[key] = [value]
    return data_batch


def resolve_checkpoint(path: str | Path) -> str:
    return str(Path(path).expanduser())
