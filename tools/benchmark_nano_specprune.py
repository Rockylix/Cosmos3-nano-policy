#!/usr/bin/env python3
"""Matched Dense-eager versus Nano SpecPrune compile/Graph benchmark.

The benchmark uses three real HDF windows from one trajectory.  The first two
build chronological observation history; the third is the fixed timed target.
Reasoner prefill/KV cache/offload remain enabled and are verified not to repeat.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

for name in (
    "COSMOS3_REASONER_OFFLOAD",
    "COSMOS3_REASONER_KVCACHE",
    "COSMOS3_REASONER_EMPTY_CACHE",
    "COSMOS3_POLICY_REASONER_CACHE",
):
    os.environ[name] = "1"
os.environ["COSMOS3_REASONER_PIN_CPU_MASTER"] = "0"
os.environ["COSMOS3_LAYER_OFFLOAD"] = "0"

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_specprune() -> Any:
    _load("cosmos_framework.inference.action_k112", SPEC_ROOT / "cosmos_framework/inference/action_k112.py")
    module = _load(
        "cosmos_framework.inference.action_specprune",
        SPEC_ROOT / "cosmos_framework/inference/action_specprune.py",
    )

    class RuntimeController(module.NanoSpecPruneController):
        """Bridge the released module names to the offload runtime hook names."""

        def __enter__(self):
            result = super().__enter__()
            if hasattr(self.model, "_k112_sparse_controller"):
                raise RuntimeError("Offload runtime controller hook is occupied")
            self.model._k112_sparse_controller = self
            return result

        def __exit__(self, exc_type, exc, traceback):
            if getattr(self.model, "_k112_sparse_controller", None) is self:
                delattr(self.model, "_k112_sparse_controller")
            return super().__exit__(exc_type, exc, traceback)

    return module, RuntimeController


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


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p90_s": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min_s": ordered[0],
        "max_s": ordered[-1],
    }


def _instruction(model: Any, batch: dict[str, Any]) -> str:
    caption = batch[model.input_caption_key][0]
    if not isinstance(caption, str):
        raise ValueError("Expected transformed caption string")
    try:
        return str(json.loads(caption)["actions"][0]["description"])
    except (ValueError, KeyError, IndexError, TypeError):
        return caption


def _snapshot(plan: Any) -> dict[str, Any]:
    return {
        "previous_rgb": plan.previous_rgb.detach().clone(),
        "previous_global": {key: value.detach().clone() for key, value in plan.previous_global.items()},
        "confidence": {key: value.detach().clone() for key, value in plan.confidence.items()},
    }


def _restore(plan: Any, state: dict[str, Any]) -> None:
    plan.reset()
    plan.previous_rgb = state["previous_rgb"].detach().clone()
    plan.previous_global = {key: value.detach().clone() for key, value in state["previous_global"].items()}
    plan.confidence = {key: value.detach().clone() for key, value in state["confidence"].items()}


def _cost(summary: dict[str, Any]) -> dict[str, float]:
    # Same logical matrix boundary used by the Nano multi-strategy table.
    d, q, kv, intermediate, blocks, full = 4096, 4096, 1024, 12288, 36, 3273

    def layer(n: int, u: int) -> int:
        return n * (2 * d * (2 * q + 2 * kv) + 6 * d * intermediate) + 4 * n * (n + u) * q

    und = summary["und_tokens"]
    uc = int(und["conditional"])
    uu = int(und["unconditional"])
    dense = 4 * blocks * (layer(full, uc) + layer(full, uu))
    sparse = sum(layer(int(row["gen_tokens"]), uc if row["branch"] == "conditional" else uu) for row in summary["token_rows"])
    query_blocks = {0: 360, 1: 360, 17: 360, 18: 32, 25: 32, 31: 32, 35: 360}
    first = {(int(row["step"]), str(row["branch"]), int(row["block"])): row for row in summary["token_rows"]}
    scoring = sum(
        2 * nq * (int(first[(0, "conditional", block)]["gen_tokens"]) + uc) * q
        for block, nq in query_blocks.items()
    )
    total = sparse + scoring
    return {
        "dense_tflop": dense / 1e12,
        "sparse_transformer_tflop": sparse / 1e12,
        "scoring_tflop": scoring / 1e12,
        "specprune_tflop": total / 1e12,
        "percent": 100.0 * total / dense,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/piper-cosmos3-quant/cosmos_battery/20k"))
    parser.add_argument("--hdf-root", type=Path, default=ROOT / "hdf_data/perfect")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset_configs/battery_assemble_hdf5.yaml")
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer-path", type=Path, required=True)
    parser.add_argument("--history-indices", type=int, nargs=2, default=(0, 32))
    parser.add_argument("--target-index", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0, help="Compatibility seed sample used during model build.")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 5:
        parser.error("At least five warmups are required")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from benchmark_nano_version1_offload import _build
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import Piper14HDF5ActionDataset
    from piper_cosmos.deployment.cosmos_piper14_policy import build_data_batch_from_sample

    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 4096
    model, _, metadata = _build(args)
    spec_module, Controller = _load_specprune()
    attention_type = type(model.net.language_model.model.layers[0].self_attn)
    if not hasattr(attention_type, "_nano_k112_attention_capture_callback"):
        setattr(
            attention_type,
            "_nano_k112_attention_capture_callback",
            property(
                lambda self: self._k112_attention_capture_callback,
                lambda self, value: setattr(self, "_k112_attention_capture_callback", value),
            ),
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
    dataset = ActionSFTDataset(raw, transform, "480")
    indices = [*args.history_indices, args.target_index]
    batches = [build_data_batch_from_sample(dataset[index]) for index in indices]
    # The local Piper transform returns the JSON caption as a Python mapping.
    # Serialize it explicitly so the policy tokenizer sees the real task text
    # instead of the historical empty/dict fallback used by older timing runs.
    for batch in batches:
        captions = batch[model.input_caption_key]
        if len(captions) != 1:
            raise RuntimeError("Expected one caption per benchmark batch")
        if isinstance(captions[0], dict):
            batch[model.input_caption_key] = [json.dumps(captions[0], ensure_ascii=False)]
    instructions = [_instruction(model, batch) for batch in batches]
    if len(set(instructions)) != 1:
        raise RuntimeError(f"History and target instructions differ: {instructions}")

    kwargs = dict(guidance=3.0, seed=[args.seed], num_steps=4, shift=5.0, has_negative_prompt=False)
    prefill_calls: list[int] = []

    def prefill_hook(module, inputs, named):
        del module
        if named.get("und_only", inputs[2] if len(inputs) > 2 else False):
            prefill_calls.append(len(prefill_calls))

    audit = model.net.register_forward_pre_hook(prefill_hook, with_kwargs=True)
    graph_replays = [0]
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        graph_replays[0] += 1
        return original_replay(graph)

    torch.cuda.CUDAGraph.replay = counted_replay
    plan = spec_module.ObservationPlan()

    def dense(batch):
        with torch.inference_mode():
            return model.generate_samples_from_batch(_clone(torch, batch), **kwargs)

    def sparse(batch):
        controller = Controller(
            net=model.net,
            guidance=3.0,
            num_steps=4,
            observation_rgb=spec_module.condition_rgb(batch, model.input_video_key),
            instruction=instructions[-1],
            tokenizer=model.vlm_tokenizer,
            plan=plan,
            output_dir=None,
            compiled=True,
            cuda_graphs=True,
        )
        before = graph_replays[0]
        with torch.inference_mode(), controller:
            output = model.generate_samples_from_batch(_clone(torch, batch), **kwargs)
        summary = controller.finish()
        summary["observed_graph_replays"] = graph_replays[0] - before
        return output, summary

    try:
        dense(batches[0])
        if len(prefill_calls) != 2:
            raise RuntimeError(f"Expected two one-time CFG prefills, got {len(prefill_calls)}")
        for batch in batches[:2]:
            sparse(batch)
        frozen = _snapshot(plan)

        timings = {"dense": [], "specprune": []}
        last_summary = None
        for repeat in range(args.warmups + args.repeats):
            order = ("dense", "specprune") if repeat % 2 == 0 else ("specprune", "dense")
            for mode in order:
                if mode == "specprune":
                    _restore(plan, frozen)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if mode == "dense":
                    output = dense(batches[-1])
                else:
                    output, last_summary = sparse(batches[-1])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat >= args.warmups:
                    timings[mode].append(elapsed)
                for key in ("action", "vision"):
                    if not all(bool(torch.isfinite(value).all()) for value in output[key]):
                        raise RuntimeError(f"Nonfinite {key}")
                print(f"BENCH {repeat + 1}/{args.warmups + args.repeats} {mode} {elapsed:.6f}s", flush=True)
                if len(prefill_calls) != 2:
                    raise RuntimeError("Reasoner cache missed during timed benchmark")
        if last_summary is None:
            raise RuntimeError("Missing SpecPrune summary")
        stats = {name: _timing(values) for name, values in timings.items()}
        stats["dense"]["speedup"] = 1.0
        stats["specprune"]["speedup"] = stats["dense"]["median_s"] / stats["specprune"]["median_s"]
        report = {
            "metadata": metadata,
            "protocol": {
                "history_indices": list(args.history_indices),
                "target_index": args.target_index,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "scope": "synchronized generate_samples_from_batch; real chronological HDF history; one-time Reasoner prefill excluded",
            },
            "timing": stats,
            "flops": _cost(last_summary),
            "controller": last_summary,
            "prefill_calls": len(prefill_calls),
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"timing": stats, "flops": report["flops"]}, indent=2), flush=True)
    finally:
        audit.remove()
        torch.cuda.CUDAGraph.replay = original_replay
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
