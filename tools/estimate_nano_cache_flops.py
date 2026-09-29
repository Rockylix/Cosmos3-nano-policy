"""Offline logical-matmul accounting for the measured warm Nano chunks.

Reads weight headers and rebuilds HDF prompt preprocessing on CPU; no inference.
The current prompt-template mismatch is recorded, not silently repaired.
"""
import csv
import json
import math
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/nano_cache_adapters_comparison"
CHECKPOINT = ROOT / "cosmos_battery/20k"


def text_audit():
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens, compute_text_split_length
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
    from cosmos_framework.model.generator.tokenizers.tokenization_qwen2 import Qwen2Tokenizer
    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import Piper14HDF5ActionDataset
    from piper_cosmos.deployment.cosmos_piper14_policy import build_data_batch_from_sample

    raw = Piper14HDF5ActionDataset(root=str(ROOT / "hdf_data/perfect"),
        config_path=str(ROOT / "configs/dataset_configs/battery_assemble_hdf5.yaml"),
        fps=30., chunk_length=32, mode="policy", use_state=True, viewpoint="concat_view", stride=1)
    transform = ActionTransformPipeline(tokenizer_config=None, cfg_dropout_rate=0., max_action_dim=64,
        append_viewpoint_info=True, append_duration_fps_timestamps=True, append_resolution_info=True,
        append_idle_frames=False, format_prompt_as_json=True)
    batch = build_data_batch_from_sample(ActionSFTDataset(raw, transform, "480")[0])
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        local_files_only=True)
    tokenizer, special = add_special_tokens(tokenizer)
    special["eos_token_id"] = tokenizer.eos_token_id
    caption = batch["ai_caption"][0]
    cond, uncond = [tokenize_caption(c, tokenizer, use_system_prompt=False,
                                    system_prompt=batch.get("system_prompt")) for c in (caption, "")]
    return dict(caption=caption, caption_type=type(caption).__name__, conditional_ids=cond,
                unconditional_ids=uncond, conditional_decoded=tokenizer.decode(cond),
                conditional_equals_unconditional=cond == uncond,
                und_lengths=[compute_text_split_length(len(ids), special, True) for ids in (cond, uncond)])


def main():
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.net.language_model.model.layers."
    def shape(block, suffix):
        key = f"{prefix}{block}.{suffix}.weight"
        with safe_open(CHECKPOINT / index[key], framework="pt", device="cpu") as f:
            return tuple(f.get_slice(key).get_shape())
    blocks = len({int(k[len(prefix):].split('.')[0]) for k in index if k.startswith(prefix)})
    matrix_names = [f"self_attn.{p}_proj_moe_gen" for p in ("q", "k", "v", "o")]
    matrix_names += [f"mlp_moe_gen.{p}_proj" for p in ("up", "gate", "down")]
    shapes = {name: shape(0, name) for name in matrix_names}
    for b in range(blocks):
        assert all(shape(b, name) == dims for name, dims in shapes.items())
    q, d = shapes[matrix_names[0]]
    kv = shapes[matrix_names[1]][0]
    intermediate = shapes["mlp_moe_gen.up_proj"][0]
    mlp_row = 2 * sum(math.prod(shapes[n]) for n in matrix_names[4:])
    linear_row = 2 * sum(math.prod(shapes[n]) for n in matrix_names[:4])
    assert (blocks, d, q, kv, intermediate) == (36, 4096, 4096, 1024, 12288)
    assert mlp_row == 6 * d * intermediate
    benchmark = json.loads((OUT / "comparison.json").read_text())
    a = json.loads(Path(benchmark["audit"]["asi"]["source"]).read_text())["results"]["asi"]["controller"]
    n, ns = a["gen_tokens"], a["sparse_gen_tokens"]
    spatial = math.prod(a["token_grid"][1:])
    future, protected = 8 * spatial, n - 8 * spatial
    assert (n, ns, spatial, protected, a["sparse_stacks"]) == (3273, 1865, 360, 393, 7)
    prompt = text_audit()
    uc, uu = prompt["und_lengths"]
    def full(nq, u):
        return nq * (linear_row + mlp_row) + 4 * nq * (nq + u) * q
    def cached(u, fresh):
        return 4 * protected * d * q + 4 * n * d * kv + 4 * protected * (n + u) * q + (protected + fresh) * mlp_row
    pair = blocks * (full(n, uc) + full(n, uu))
    profile = blocks * (4 * 32 * (n + uc) * q + 2 * 32 * spatial * q)
    fresh = [int(.25 * (1.5 - b / (blocks - 1)) * future) for b in range(blocks)]
    totals = dict(dense=4 * pair,
        asi=blocks * (full(n, uc) + 3 * full(ns, uc) + 4 * full(ns, uu)) + profile,
        toca=2 * pair + 2 * sum(cached(uc, f) + cached(uu, f) for f in fresh),
        worldcache=3 * pair, c3ache=(4 * pair + 2 * pair) / 2)
    measured = {r["strategy"]: r for r in benchmark["rows"]}
    rows = [dict(strategy=s, tflop_per_chunk=f / 1e12, baseline_percent=100*f/totals['dense'],
                 reduction_percent=100*(1-f/totals['dense']), theoretical_speedup=totals['dense']/f,
                 measured_speedup=measured.get(s, {}).get('speedup', 1.)) for s, f in totals.items()]
    result = dict(scope="Warm instruction-cache-hit Transformer logical matmuls, FMA=2; excludes initial UND prefill",
        dimensions=dict(blocks=blocks, hidden=d, query_width=q, kv_width=kv, intermediate=intermediate,
                        gen_tokens=n, sparse_gen_tokens=ns, future_tokens=future, protected_tokens=protected),
        shapes=shapes, prompt_audit=prompt, asi_profile_flops=profile, toca_fresh_per_block=fresh, rows=rows)
    (OUT / "flops_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    with (OUT / "flops_comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
