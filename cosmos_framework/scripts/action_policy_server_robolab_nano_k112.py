# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab policy server for a dense or Shared-Motion K112 Cosmos3-Nano arm.

K112 is opt-in via ``--mode k112``.  Both modes in this entrypoint use eager,
single-rank execution so their later chunk timings are comparable.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from cosmos_framework.inference.action_k112 import (
    STRATEGY_VERSION,
    NanoK112Controller,
    NanoK112VelocityCacheSampler,
)
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.utils import log


class NanoK112ServerArgs(RobolabServerArgs):
    mode: Literal["dense", "k112"] = "dense"
    """Execution arm. K112 remains disabled unless explicitly selected."""

    k112_output_dir: Path = Path("/tmp/cosmos3_action_server/nano_k112")
    """Per-request K112 profile and token-saving summaries."""


class NanoK112PolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        # Shape changes between the dense profile stack and sparse stacks are
        # deliberate. Compile, graphs, CP, CFGP, and FSDP are not valid for the
        # current minimal controller.
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

    def __init__(self, args: NanoK112ServerArgs) -> None:
        if str(args.mode) == "k112":
            if int(args.num_steps) != 4:
                raise ValueError("Nano K112 requires --num-steps 4")
            if float(args.shift) != 5.0:
                raise ValueError("Nano K112 requires --shift 5")
            if float(args.guidance) == 1.0:
                raise ValueError("Nano K112 requires conditional/unconditional CFG")
            if args.guidance_interval is not None:
                raise ValueError("Nano K112 requires CFG on all four steps; omit --guidance-interval")
        super().__init__(args)
        self._mode = str(args.mode)
        self._k112_output_dir = args.k112_output_dir.expanduser().absolute()
        self._request_index = 0
        self.last_k112_summary: dict[str, Any] | None = None
        self.last_generation_wall_s: float | None = None
        original_generate = self.model.generate_samples_from_batch

        def generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            kwargs = dict(generate_kwargs)
            controller = None
            request_index = self._request_index
            self._request_index += 1
            request_dir = self._k112_output_dir / f"request_{request_index:06d}"
            if self._mode == "k112":
                controller = NanoK112Controller(
                    net=self.model.net,
                    guidance=float(kwargs.get("guidance", self.cfg.guidance)),
                    num_steps=int(kwargs.get("num_steps", self.cfg.num_steps)),
                    output_dir=request_dir,
                )
                kwargs["sampler"] = NanoK112VelocityCacheSampler(self.model.sampler, controller)

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
            self.last_k112_summary = controller.finish() if controller is not None else None
            request_dir.mkdir(parents=True, exist_ok=True)
            (request_dir / "generation.json").write_text(
                json.dumps(
                    {
                        "mode": self._mode,
                        "strategy_version": STRATEGY_VERSION if controller is not None else "dense",
                        "generation_wall_s": self.last_generation_wall_s,
                        "action_finite": True,
                        "vision_finite": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            log.info(
                f"[nano-k112] mode={self._mode} request={request_index} "
                f"generation_wall_s={self.last_generation_wall_s:.3f}"
            )
            return samples

        self.model.generate_samples_from_batch = generate


def serve(args: NanoK112ServerArgs) -> None:
    log.info(
        f"[nano-k112-server] host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"mode={args.mode} checkpoint={args.checkpoint_path!r}"
    )
    service = NanoK112PolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[nano-k112-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    metadata = {
        "mode": str(args.mode),
        "strategy_version": STRATEGY_VERSION if str(args.mode) == "k112" else "dense",
        "compile": False,
        "cuda_graphs": False,
    }
    server_cls(policy=service, host=args.host, port=int(args.port), metadata=metadata).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        NanoK112ServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
