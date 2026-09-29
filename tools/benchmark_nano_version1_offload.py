#!/usr/bin/env python3
"""Benchmark dense and Version1 Nano GEN inference with existing reasoner offload."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

MODES = ("dense", "version1", "version1_compile", "version1_graph")


def _clone(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(torch, item) for item in value)
    return copy.deepcopy(value)


def _flatten(torch: Any, samples: dict[str, Any], key: str) -> Any:
    values = samples.get(key)
    if not isinstance(values, list) or len(values) != 1 or not torch.is_tensor(values[0]):
        raise RuntimeError(f"Expected one tensor in samples[{key!r}]")
    value = values[0].detach().float().reshape(-1).cpu()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{key} contains NaN/Inf")
    return value


def _compare(torch: Any, reference: Any, candidate: Any) -> dict[str, float]:
    difference = candidate - reference
    eps = torch.finfo(reference.dtype).eps
    return {
        "mse": float(difference.square().mean()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(eps)),
        "cosine": float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0, eps=float(eps))),
        "max_absolute_error": float(difference.abs().max()),
    }


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p90_s": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min_s": ordered[0],
        "max_s": ordered[-1],
    }


def _compat_config(checkpoint: Path, vae: Path, tokenizer: Path) -> Path:
    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config = payload["model"]["config"]
    config["compile"]["enabled"] = False
    config["compile"]["use_cuda_graphs"] = False
    config["tokenizer"]["vae_path"] = str(vae.absolute())
    config["vlm_config"]["tokenizer"]["pretrained_model_name"] = str(tokenizer.resolve())
    output = Path("/tmp/cosmos3_nano_version1_offload/config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _build(args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import Piper14HDF5ActionDataset
    from piper_cosmos.deployment.cosmos_piper14_policy import build_data_batch_from_sample

    class BenchmarkArgs(ActionServerArgs):
        def build_setup_overrides(self):
            setup = super().build_setup_overrides()
            setup.dp_replicate_size = 1
            setup.dp_shard_size = 1
            setup.cp_size = 1
            setup.cfgp_size = 1
            setup.use_torch_compile = False
            setup.use_cuda_graphs = False
            setup.compile_dynamic = False
            setup.guardrails = False
            setup.offload_guardrail_models = False
            return setup

    checkpoint = args.checkpoint.resolve()
    config = _compat_config(checkpoint, args.vae_path, args.qwen_tokenizer_path)
    service = ActionModelService(
        BenchmarkArgs(
            checkpoint=CheckpointOverrides(checkpoint_path=str(checkpoint), config_file=str(config)),
            output_dir=Path("/tmp/cosmos3_nano_version1_offload"),
            action_chunk_size=32,
            raw_action_dim=14,
            max_action_dim=64,
            seed=args.seed,
            guidance=3.0,
            num_steps=4,
            fps=30,
            format_prompt_as_json=True,
        )
    )
    raw = Piper14HDF5ActionDataset(
        root=str(args.hdf_root.resolve()),
        config_path=str(args.dataset_config.resolve()),
        fps=30.0,
        chunk_length=32,
        mode="policy",
        use_state=True,
        viewpoint="concat_view",
        stride=1,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        max_action_dim=64,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=True,
    )
    sample = ActionSFTDataset(raw, transform, "480")[args.sample_index]
    batch = build_data_batch_from_sample(sample)
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "hdf_root": str(args.hdf_root.resolve()),
        "sample_index": args.sample_index,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "offload": {
            "COSMOS3_REASONER_OFFLOAD": os.environ["COSMOS3_REASONER_OFFLOAD"],
            "COSMOS3_REASONER_KVCACHE": os.environ["COSMOS3_REASONER_KVCACHE"],
            "COSMOS3_REASONER_EMPTY_CACHE": os.environ["COSMOS3_REASONER_EMPTY_CACHE"],
            "COSMOS3_POLICY_REASONER_CACHE": os.environ["COSMOS3_POLICY_REASONER_CACHE"],
        },
        "scope": "synchronized generate_samples_from_batch; excludes HDF transform, RPC, and VAE decode",
    }
    return service.model, batch, metadata


def _run(torch: Any, model: Any, batch: dict[str, Any], mode: str, seed: int) -> tuple[dict[str, Any], Any, float]:
    from piper_cosmos.cosmos3.nano_version1_graph import NanoVersion1CompiledController
    from piper_cosmos.cosmos3.nano_version1_sparse import NanoVersion1Controller

    controller = None
    if mode == "version1":
        controller = NanoVersion1Controller(torch=torch, net=model.net, guidance=3.0, num_steps=4)
    elif mode in ("version1_compile", "version1_graph"):
        controller = NanoVersion1CompiledController(
            torch=torch,
            net=model.net,
            guidance=3.0,
            num_steps=4,
            use_cuda_graphs=mode == "version1_graph",
        )
    elif mode != "dense":
        raise ValueError(mode)
    kwargs = dict(guidance=3.0, seed=[seed], num_steps=4, shift=5.0, has_negative_prompt=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        if controller is None:
            samples = model.generate_samples_from_batch(_clone(torch, batch), **kwargs)
        else:
            with controller:
                samples = model.generate_samples_from_batch(_clone(torch, batch), **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = controller.finish() if controller is not None else None
    _flatten(torch, samples, "action")
    _flatten(torch, samples, "vision")
    return samples, summary, elapsed


def _write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    dense = result["timing"]["dense"]["median_s"]
    labels = {
        "dense": "Dense eager",
        "version1": "Version1 eager",
        "version1_compile": "Version1 + compile-only",
        "version1_graph": "Version1 + compile + Global Graph",
    }
    rows = [
        f"| {labels[mode]} | {result['timing'][mode]['mean_s']:.6f} s | "
        f"{result['timing'][mode]['median_s']:.6f} s | {result['timing'][mode]['p90_s']:.6f} s | "
        f"{dense / result['timing'][mode]['median_s']:.3f}x |"
        for mode in MODES
    ]
    report = "\n".join(
        [
            "# Cosmos3 Nano Version1 offload 单 chunk 基准",
            "",
            "| 策略 | Mean | Median | P90 | 相对 Dense 加速 |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "- Reasoner/UND 使用 Framework 现有 offload 与逐层 KV cache；GEN 常驻 GPU。",
            "- step 0 conditional 为 dense profile；其余 7 个 CFG stack 使用固定 K184 future mask。",
            "- 同一模型、同一 HDF5 sample、相同 seed，四种模式交替测量并执行 CUDA synchronize。",
            "- 误差及 token accounting 见 `metrics.json`。",
            "",
        ]
    )
    (output_dir / "report_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/piper-cosmos3-quant/cosmos_battery/20k"))
    parser.add_argument("--hdf-root", type=Path, default=Path("/root/piper-cosmos3/hdf_data/perfect"))
    parser.add_argument("--dataset-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/dataset_configs/battery_assemble_hdf5.yaml")
    parser.add_argument("--vae-path", type=Path, default=Path("/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"))
    parser.add_argument("--qwen-tokenizer-path", type=Path, default=Path("/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measured-rounds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/piper-cosmos3/experiments/nano_version1_offload_hdf_v1"))
    args = parser.parse_args()

    os.environ.setdefault("COSMOS3_REASONER_OFFLOAD", "1")
    os.environ.setdefault("COSMOS3_REASONER_KVCACHE", "1")
    os.environ.setdefault("COSMOS3_REASONER_EMPTY_CACHE", "1")
    os.environ.setdefault("COSMOS3_REASONER_PIN_CPU_MASTER", "0")
    os.environ.setdefault("COSMOS3_POLICY_REASONER_CACHE", "1")

    import torch

    model, batch, metadata = _build(args)
    outputs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for round_index in range(args.warmup_rounds):
        sequence = MODES if round_index % 2 == 0 else tuple(reversed(MODES))
        for mode in sequence:
            samples, summary, elapsed = _run(torch, model, batch, mode, args.seed)
            print(f"warmup={round_index} mode={mode} generation_s={elapsed:.6f}", flush=True)
            outputs[mode] = {key: _flatten(torch, samples, key) for key in ("action", "vision")}
            if summary is not None:
                summaries[mode] = summary

    raw_timings = {mode: [] for mode in MODES}
    for repeat in range(args.measured_rounds):
        sequence = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in sequence:
            samples, summary, elapsed = _run(torch, model, batch, mode, args.seed)
            raw_timings[mode].append(elapsed)
            outputs[mode] = {key: _flatten(torch, samples, key) for key in ("action", "vision")}
            if summary is not None:
                summaries[mode] = summary
            print(f"repeat={repeat:02d} mode={mode} generation_s={elapsed:.6f}", flush=True)

    dense = outputs["dense"]
    result = {
        "metadata": metadata,
        "protocol": {"warmup_rounds": args.warmup_rounds, "measured_rounds": args.measured_rounds},
        "timing": {mode: _timing(values) for mode, values in raw_timings.items()},
        "accuracy_vs_dense": {
            mode: {key: _compare(torch, dense[key], outputs[mode][key]) for key in ("action", "vision")}
            for mode in MODES
        },
        "controller": summaries,
    }
    _write_report(result, args.output_dir)
    print(json.dumps(result["timing"], indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
