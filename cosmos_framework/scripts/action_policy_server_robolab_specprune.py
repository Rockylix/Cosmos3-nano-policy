# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab server for Dense or observation-driven Nano SpecPrune-VLA.

SpecPrune remains disabled unless ``--mode specprune`` is selected.  The
Reasoner prefill/KV cache/offload path remains owned by Cosmos Framework.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from cosmos_framework.inference.action_specprune import (
    STRATEGY_VERSION,
    NanoSpecPruneConfig,
    ObservationPlan,
    condition_rgb,
)
from cosmos_framework.inference.action_specprune_offload import NanoSpecPruneController
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.utils import log


class NanoSpecPruneServerArgs(RobolabServerArgs):
    mode: Literal["dense", "specprune"] = "dense"
    specprune_output_dir: Path = Path("/tmp/cosmos3_action_server/nano_specprune")
    specprune_local_k: int = 58
    specprune_global_k: int = 72
    specprune_min_observation_tokens: int = 184
    specprune_keep_ratio: float = 0.9
    specprune_compile: bool = False
    specprune_cuda_graphs: bool = False


def _instruction_from_batch(model: Any, data_batch: dict[str, Any]) -> str:
    caption = data_batch[model.input_caption_key][0]
    if not isinstance(caption, str):
        raise ValueError("SpecPrune requires a string instruction/caption")
    try:
        payload = json.loads(caption)
        return str(payload["actions"][0]["description"])
    except (ValueError, KeyError, IndexError, TypeError):
        return caption


class NanoSpecPrunePolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {
            "use_torch_compile": False,
            "use_cuda_graphs": False,
            "dp_replicate_size": 1,
            "dp_shard_size": 1,
            "cp_size": 1,
            "cfgp_size": 1,
        }
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: NanoSpecPruneServerArgs) -> None:
        if str(args.mode) == "specprune":
            if (int(args.num_steps), float(args.shift), float(args.guidance)) != (4, 5.0, 3.0):
                raise ValueError("Frozen Nano SpecPrune requires steps=4, shift=5, guidance=3")
            if args.guidance_interval is not None:
                raise ValueError("Nano SpecPrune requires CFG on all four denoise steps")
            if args.specprune_cuda_graphs and not args.specprune_compile:
                raise ValueError("--specprune-cuda-graphs requires --specprune-compile")
        super().__init__(args)
        self._mode = str(args.mode)
        self._output_root = args.specprune_output_dir.expanduser().absolute()
        config = NanoSpecPruneConfig(
            local_k=int(args.specprune_local_k),
            global_k=int(args.specprune_global_k),
            min_observation_tokens=int(args.specprune_min_observation_tokens),
            keep_ratio=float(args.specprune_keep_ratio),
        )
        self._plan = ObservationPlan(config)
        self._prompt: str | None = None
        self._request_index = 0
        self.last_specprune_summary: dict[str, Any] | None = None
        self.last_generation_wall_s: float | None = None
        original_generate = self.model.generate_samples_from_batch

        def generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            kwargs = dict(generate_kwargs)
            data_batch = generate_args[0] if generate_args else kwargs["data_batch"]
            request_index = self._request_index
            self._request_index += 1
            request_dir = self._output_root / f"request_{request_index:06d}"
            controller = None
            if self._mode == "specprune":
                controller = NanoSpecPruneController(
                    net=self.model.net,
                    guidance=float(kwargs.get("guidance", self.cfg.guidance)),
                    num_steps=int(kwargs.get("num_steps", self.cfg.num_steps)),
                    observation_rgb=condition_rgb(data_batch, self.model.input_video_key),
                    instruction=_instruction_from_batch(self.model, data_batch),
                    tokenizer=self.model.vlm_tokenizer,
                    plan=self._plan,
                    output_dir=request_dir,
                    compiled=bool(args.specprune_compile),
                    cuda_graphs=bool(args.specprune_cuda_graphs),
                )

            torch.cuda.synchronize()
            start = time.perf_counter()
            if controller is None:
                samples = original_generate(*generate_args, **kwargs)
            else:
                with controller:
                    samples = original_generate(*generate_args, **kwargs)
            torch.cuda.synchronize()
            self.last_generation_wall_s = time.perf_counter() - start
            for key in ("action", "vision"):
                for value in samples.get(key, []):
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f"Nano {self._mode} produced NaN/Inf in {key}")
            self.last_specprune_summary = controller.finish() if controller is not None else None
            if controller is not None:
                summary = self.last_specprune_summary
                log.info(
                    f"[nano-specprune] request={request_index} "
                    f"future_retention={summary['mean_future_token_retention']:.4f} "
                    f"target={summary['asi_k184_average_future_retention_target']:.4f} "
                    f"generation_wall_s={self.last_generation_wall_s:.3f}"
                )
            return samples

        self.model.generate_samples_from_batch = generate

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        if prompt != self._prompt:
            self._plan.reset()
            self._prompt = prompt
        return super().infer(obs)


def serve(args: NanoSpecPruneServerArgs) -> None:
    log.info(f"[nano-specprune-server] host={socket.gethostname()} bind={args.host}:{int(args.port)} mode={args.mode}")
    service = NanoSpecPrunePolicyService(args)
    local_ip = get_local_ip()
    metadata = {
        "mode": str(args.mode),
        "strategy_version": STRATEGY_VERSION if str(args.mode) == "specprune" else "dense",
        "compile": bool(args.specprune_compile),
        "cuda_graphs": bool(args.specprune_cuda_graphs),
    }
    log.info(f"[nano-specprune-server] accessible at ws://{local_ip}:{int(args.port)}/")
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=int(args.port), metadata=metadata
    ).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        NanoSpecPruneServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
