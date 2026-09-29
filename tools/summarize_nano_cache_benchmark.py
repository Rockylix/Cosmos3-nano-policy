"""Combine complete paired Nano measurements; reject partial strategy runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "asi": "nano_cache_adapters_graph_v1",
    "toca": "nano_cache_adapters_graph_v2",
    "worldcache": "nano_cache_adapters_graph_v2",
    "c3ache": "nano_cache_adapters_graph_v2",
}


def main():
    rows, audit = [], {}
    identity = None
    for strategy, run in SOURCES.items():
        source = ROOT / "experiments" / run / "results.json"
        data = json.loads(source.read_text())
        result = data["results"][strategy]
        assert len(result["raw_seconds"]) == len(result["dense_raw_seconds"]) == 10
        assert data["prefill_calls"] == 2 and data["placement"]["reasoner_cuda_bytes"] == 0
        current = tuple(data["metadata"][k] for k in ("checkpoint", "hdf_root", "sample_index", "seed", "device", "torch"))
        identity = current if identity is None else identity
        assert identity == current
        controller = result["controller"]
        assert controller["observed_graph_replays"] > 0
        assert controller["layer_count"] == 36
        row = dict(strategy=strategy, mean_s=result["timing"]["mean_s"], median_s=result["timing"]["median_s"],
                   p90_s=result["timing"]["p90_s"], dense_median_s=result["dense_timing"]["median_s"],
                   speedup=result["speedup"], graph_replays=controller["observed_graph_replays"],
                   action_mse=result["accuracy"]["action"]["mse"], action_cos=result["accuracy"]["action"]["cosine"],
                   vision_cos=result["accuracy"]["vision"]["cosine"], vision_rel_l2=result["accuracy"]["vision"]["relative_l2"])
        rows.append(row)
        audit[strategy] = dict(source=str(source), peak_allocated_gib=data["peak_allocated_gib"],
                               repeat_error=result["repeat_error"])
    output = ROOT / "experiments/nano_cache_adapters_comparison"
    output.mkdir(exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(dict(rows=rows, audit=audit), indent=2)+"\n")
    with (output / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    report = ["# Nano 多策略单 chunk 测速结果", "", "每策略3轮预热、10轮采样，与同进程Dense eager交替；策略使用compile+CUDA Graph。", "",
              "| 策略 | Mean(s) | Median(s) | P90(s) | 配对 Dense Median(s) | 加速 | Graph replay/chunk |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['strategy']} | {r['mean_s']:.6f} | {r['median_s']:.6f} | {r['p90_s']:.6f} | {r['dense_median_s']:.6f} | {r['speedup']:.3f}× | {r['graph_replays']} |")
    report += ["", "C3ache时间按refresh+hit均摊，表中Graph次数为hit，refresh为288。相同HDF输入只用于测开销，不代表真实跨观测质量。",
               "Reasoner首次双分支prefill后持续命中缓存；每个进程全程prefill次数=2，UND decoder CUDA参数=0。",
               "计时为同步generate_samples_from_batch，不含初次prefill/编译/capture、HDF预处理和VAE decode。",
               "ASI首轮采样已完成后，ToCa首次预热触发Dynamo上限8；在框架初始化后设置容量256，重新加载模型补跑后三策略。各自使用配对Dense，未拼接未完成的ToCa样本。",
               "", "## 输出误差（完整返回tensor，非物理关节单位）", "",
               "| 策略 | action MSE | action cos | vision cos | vision rel-L2 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['strategy']} | {r['action_mse']:.8g} | {r['action_cos']:.6f} | {r['vision_cos']:.6f} | {r['vision_rel_l2']:.6f} |")
    report += ["", "没有运行RoboLab或真实机械臂，没有任务成功率结论。", "", "详见 [方法与命令](../../docs/nano_multistrategy_offload_cn.md)。", ""]
    (output / "report_cn.md").write_text("\n".join(report))
    (ROOT / "docs/nano_multistrategy_benchmark_results_cn.md").write_text("\n".join(report).replace("../../docs/nano_multistrategy_offload_cn.md", "nano_multistrategy_offload_cn.md"))
    print("\n".join(report))


if __name__ == "__main__":
    main()
