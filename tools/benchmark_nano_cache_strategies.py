"""Paired HDF5 single-chunk benchmark, mandatory instruction cache and offload."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for name in ("COSMOS3_REASONER_OFFLOAD", "COSMOS3_REASONER_KVCACHE", "COSMOS3_REASONER_EMPTY_CACHE", "COSMOS3_POLICY_REASONER_CACHE"):
    os.environ[name] = "1"
os.environ["COSMOS3_REASONER_PIN_CPU_MASTER"] = "0"
os.environ["COSMOS3_LAYER_OFFLOAD"] = "0"

from benchmark_nano_version1_offload import _build, _clone, _compare, _flatten, _timing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/root/piper-cosmos3/cosmos_battery/20k"))
    p.add_argument("--hdf-root", type=Path, default=Path("/root/piper-cosmos3/hdf_data/perfect"))
    p.add_argument("--dataset-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/dataset_configs/battery_assemble_hdf5.yaml")
    p.add_argument("--vae-path", type=Path, default=Path("/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"))
    p.add_argument("--qwen-tokenizer-path", type=Path, default=Path("/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--measured-rounds", type=int, default=10)
    p.add_argument("--strategies", nargs="+", default=["dense_adapter", "asi", "toca", "worldcache", "c3ache"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--graphs", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import cosmos_framework
    from piper_cosmos.cosmos3.nano_cache_strategies import NanoCacheController

    expected_source = Path(__file__).resolve().parents[1] / "cosmos_framework"
    if Path(cosmos_framework.__file__).resolve().parent != expected_source:
        raise RuntimeError(f"Unexpected Framework source: {cosmos_framework.__file__}")
    if args.compile:
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.accumulated_recompile_limit = 4096
    model, batch, metadata = _build(args)
    # The Piper dataset transform keeps the JSON prompt as a Python mapping.
    # Serialize it explicitly so the tokenizer receives the real instruction;
    # older timing runs silently exercised the empty/dict fallback instead.
    captions = batch[model.input_caption_key]
    if len(captions) != 1:
        raise RuntimeError("Expected one caption per benchmark batch")
    if isinstance(captions[0], dict):
        batch[model.input_caption_key] = [json.dumps(captions[0], ensure_ascii=False)]
    if args.compile:
        # Framework initialization resets Dynamo defaults. ToCa's 36 fixed
        # fresh-token shapes must be admitted after that reset, before tracing.
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.accumulated_recompile_limit = 4096
    prefill_calls = []
    def prefill_hook(module, inputs, kwargs):
        if kwargs.get("und_only", inputs[2] if len(inputs) > 2 else False):
            prefill_calls.append(len(prefill_calls))
    audit = model.net.register_forward_pre_hook(prefill_hook, with_kwargs=True)
    graph_replays = [0]
    original_replay = torch.cuda.CUDAGraph.replay
    def counted_replay(graph):
        graph_replays[0] += 1
        return original_replay(graph)
    torch.cuda.CUDAGraph.replay = counted_replay
    kwargs = dict(guidance=3., seed=[args.seed], num_steps=4, shift=5., has_negative_prompt=False)

    def run(strategy, state=None, chunk=0):
        prepared = _clone(torch, batch)
        c = None if strategy == "dense" else NanoCacheController(model.net, strategy,
            compiled=args.compile, graphs=args.graphs, c3_state=state, chunk=chunk)
        torch.cuda.synchronize()
        replays_before = graph_replays[0]
        start = time.perf_counter()
        with torch.inference_mode():
            if c is None:
                output = model.generate_samples_from_batch(prepared, **kwargs)
            else:
                with c:
                    output = model.generate_samples_from_batch(prepared, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        summary = {} if c is None else c.finish()
        summary["observed_graph_replays"] = graph_replays[0] - replays_before
        tensors = {name: _flatten(torch, output, name) for name in ("action", "vision")}
        del output
        return tensors, summary, elapsed

    ref, _, cold_s = run("dense")
    assert len(prefill_calls) == 2, f"Expected 2 initial branch prefills, got {len(prefill_calls)}"
    cache = model._policy_reasoner_memory_cache
    assert cache[1].is_gen_only() and cache[2].is_gen_only()
    placement = {"reasoner_cuda_bytes": 0, "gen_cuda_bytes": 0}
    for layer in model.net.language_model.model.layers:
        for name, tensor in layer.named_parameters():
            if tensor.is_cuda:
                placement["gen_cuda_bytes" if "_moe_gen" in name else "reasoner_cuda_bytes"] += tensor.numel() * tensor.element_size()
    assert placement["reasoner_cuda_bytes"] == 0, placement
    assert placement["gen_cuda_bytes"] > 0, placement
    print(f"PREFILL VERIFIED calls=2 placement={placement} cold_s={cold_s:.3f}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    all_results = {}
    try:
        for strategy in args.strategies:
            durations, dense_durations, refresh_times, hit_times = [], [], [], []
            state = {}
            last, first = None, None
            for round_id in range(args.warmup_rounds + args.measured_rounds):
                sequence = ["dense", strategy] if round_id % 2 == 0 else [strategy, "dense"]
                measured = round_id >= args.warmup_rounds
                for mode in sequence:
                    if mode == "c3ache":
                        refresh, sr, tr = run(mode, state, 2 * round_id)
                        tensors, summary, th = run(mode, state, 2 * round_id + 1)
                        elapsed = (tr + th) / 2
                        assert sr["cache_hits"] == 0 and summary["cache_hits"] == 4
                        summary["refresh_summary"] = sr
                        if measured:
                            refresh_times.append(tr); hit_times.append(th)
                    else:
                        tensors, summary, elapsed = run(mode)
                    assert len(prefill_calls) == 2, "Instruction cache miss during warm benchmark"
                    assert model._policy_reasoner_memory_cache is cache, "Reasoner cache object replaced"
                    print(f"strategy={strategy} round={round_id} measured={measured} mode={mode} seconds={elapsed:.6f}", flush=True)
                    if mode == "dense":
                        assert all(torch.equal(ref[k], tensors[k]) for k in ref), "Dense repeat changed"
                        if measured:
                            dense_durations.append(elapsed)
                    else:
                        if measured:
                            durations.append(elapsed)
                            if first is None:
                                first = tensors
                        last = (tensors, summary)
                if measured:
                    result = dict(timing=_timing(durations), dense_timing=_timing(dense_durations),
                        speedup=_timing(dense_durations)["median_s"]/_timing(durations)["median_s"],
                        accuracy={k: _compare(torch, ref[k], last[0][k]) for k in ref},
                        repeat_error={k: _compare(torch, first[k], last[0][k]) for k in ref},
                        controller=last[1], raw_seconds=durations, dense_raw_seconds=dense_durations,
                        c3_refresh_seconds=refresh_times, c3_hit_seconds=hit_times)
                    all_results[strategy] = result
                    payload = dict(metadata=metadata, placement=placement, prefill_calls=len(prefill_calls),
                        compile=args.compile, graphs=args.graphs, cold_prefill_s=cold_s,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        results=all_results, c3_scope="same HDF input, refresh+hit mean, not cross-observation fidelity")
                    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2)+"\n")
            torch.save(last[0], args.output_dir / f"{strategy}_final.pt")
            print(f"COMPLETE {strategy} speedup={all_results[strategy]['speedup']:.3f}", flush=True)
    finally:
        audit.remove()
        torch.cuda.CUDAGraph.replay = original_replay
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
