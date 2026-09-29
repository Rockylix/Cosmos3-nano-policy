#!/usr/bin/env python3
"""Start the RTC-style Cosmos Piper14 policy server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.deployment.cosmos_piper14_policy import (  # noqa: E402
    CosmosPiper14PolicyConfig,
)
from piper_cosmos.deployment.cosmos_piper14_policy_server import (  # noqa: E402
    serve_cosmos_piper14_policy,
)
from piper_cosmos.cosmos3.local_edge_assets import (  # noqa: E402
    bootstrap_edge_hf_inference_assets,
    bootstrap_local_edge_assets,
    checkpoint_type,
    edge_hf_inference_overrides,
)


def _add_boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--edge-snapshot", type=Path, default=None)
    parser.add_argument("--wan-vae-path", type=Path, default=None)
    parser.add_argument("--reasoner-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--prompt", default="Assemble the mouse's battery.")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--mock-backend", action="store_true")
    parser.add_argument(
        "--k112",
        action="store_true",
        help=(
            "Enable Shared-Motion K112 sparse inference. Requires the bundled "
            "Cosmos Framework patch, num_steps=4, guidance CFG, and shift=5."
        ),
    )
    parser.add_argument(
        "--k112-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-request K112 mask/token summaries.",
    )
    parser.add_argument(
        "--nano-version1",
        action="store_true",
        help="Enable Nano Core64+Stable120 K184 sparse GEN inference with reasoner offload.",
    )
    parser.add_argument(
        "--nano-version1-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-request Nano Version1 mask/token summaries.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print synchronized per-stage CPU/GPU inference timings.",
    )
    _add_boolean_flag(
        parser,
        "torch-compile",
        default=True,
        help_text="Compile the configured model regions with torch.compile.",
    )
    _add_boolean_flag(
        parser,
        "cuda-graphs",
        default=False,
        help_text="Use torch.compile reduce-overhead CUDA Graph capture.",
    )
    _add_boolean_flag(
        parser,
        "compile-dynamic",
        default=True,
        help_text="Compile symbolic dynamic shapes; disable for fixed-shape policy serving.",
    )
    parser.add_argument(
        "--compiled-region",
        choices=("all", "language"),
        default="all",
        help="Model region covered by torch.compile.",
    )
    _add_boolean_flag(
        parser,
        "reuse-denoise-packing",
        default=True,
        help_text="Reuse static packed-sequence metadata across denoising forwards.",
    )
    _add_boolean_flag(
        parser,
        "allow-torch-sdpa-fallback",
        default=False,
        help_text="Use native PyTorch SDPA only if no Cosmos attention backend is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mock_backend and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --mock-backend is set")
    experiment_overrides: tuple[str, ...] = ()
    if not args.mock_backend:
        checkpoint = Path(args.checkpoint).expanduser().absolute()
        checkpoint_kind = checkpoint_type(checkpoint)
        if checkpoint_kind == "hf":
            inference_assets = bootstrap_edge_hf_inference_assets(
                checkpoint=checkpoint,
                wan_vae_path=args.wan_vae_path,
                reasoner_config=args.reasoner_config,
            )
            experiment_overrides = edge_hf_inference_overrides(inference_assets)
        else:
            bootstrap_local_edge_assets(
                edge_snapshot=args.edge_snapshot,
                wan_vae_path=args.wan_vae_path,
            )
        checkpoint_arg = str(checkpoint)
    else:
        checkpoint_arg = args.checkpoint
    output_dir = args.output_dir or (
        ROOT / "reports/cosmos3_edge_piper14/policy_server" / f"port_{args.port}"
    )
    config = CosmosPiper14PolicyConfig(
        checkpoint=checkpoint_arg,
        config_file=args.config_file,
        experiment_overrides=experiment_overrides,
        output_dir=str(output_dir),
        prompt=args.prompt,
        action_horizon=args.action_horizon,
        max_action_dim=args.max_action_dim,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        fps=args.fps,
        seed=args.seed,
        host=args.host,
        port=args.port,
        mock_backend=args.mock_backend,
        timing=args.timing,
        use_torch_compile=args.torch_compile,
        use_cuda_graphs=args.cuda_graphs,
        compiled_region=args.compiled_region,
        compile_dynamic=args.compile_dynamic,
        reuse_denoise_packing=args.reuse_denoise_packing,
        allow_torch_sdpa_fallback=args.allow_torch_sdpa_fallback,
        k112_enabled=args.k112,
        k112_output_dir=(
            str(args.k112_output_dir.expanduser().absolute())
            if args.k112_output_dir is not None
            else None
        ),
        nano_version1_enabled=args.nano_version1,
        nano_version1_output_dir=(
            str(args.nano_version1_output_dir.expanduser().absolute())
            if args.nano_version1_output_dir is not None
            else None
        ),
    )
    serve_cosmos_piper14_policy(
        config, host=args.host, port=args.port, authkey=args.authkey
    )


if __name__ == "__main__":
    main()
