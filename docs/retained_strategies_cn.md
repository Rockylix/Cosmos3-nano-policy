# Nano 独立策略发布说明（2026-09-29）

本次按用户确认，以最近 HDF5 多策略测速实际使用的 offload Framework 为共同底座。
来源为 Framework `801bb20` 加当时运行源码修改及 `nano-cache-adapters` hooks。
原始文件 SHA256 保存在 `nano_runtime_source_manifest.json`。

`baseline` 使用原生 Dense 路径；四步、两 CFG branch。Reasoner/bridge 在指令首次
prefill 后 offload，后续复用指令 KV，GEN 常驻。策略控制器必须显式启用。

| 分支 | 内容 | 执行开关 |
|---|---|---|
| baseline | 原生 Dense + 共同 offload 底座 | 原生服务的 compile 配置；Dense 对照默认 eager |
| version/asi | 当前 HDF 实测 ASI，Core80 + Stable104，K184 | benchmark `--compile`、`--graphs` |
| experiment/toca-future | D/C/D/C、r=0.25、CFG independent | benchmark `--compile`、`--graphs` |
| experiment/worldcache | D/D/D/C | benchmark `--compile`、`--graphs` |
| experiment/c3ache | 两 chunk 周期，复用 chunk 前两步整栈 residual | benchmark `--compile`、`--graphs` |
| experiment/specprune-future | 观测选择、ASI 对齐预算、early-exit | 服务 `--specprune-compile`、`--specprune-cuda-graphs` |
| experiment/real-robot-inference | 原 Piper Nano 服务及 Version1 控制器 | 服务原有 compile、CUDA Graph 开关 |

各策略分支共享实现，`configs/release_strategy.json` 记录当前分支的发布策略与配置。
该文件是发布清单，不会自动启用 controller；启动时必须显式传入本页所列参数。
保留原来的 controller 和计算，不通过更改超参数“统一”历史实验。
特别是 Piper 原 Version1 默认预算与 HDF 多策略 ASI 不同，二者不冒充同一配置。

## HDF5 配对测速

所有路径显式传入。仓库源码、`piper_cosmos` 支持模块、dataset schema 已在同一仓库；
不用外部 Piper 源码或隐式 `.worktrees` 覆盖。环境和权重仍由使用者指定，不随 Git 提交。

```bash
cd /path/to/Cosmos3-nano-policy
export PYTHONPATH="$PWD:$PWD/tools"
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/existing/env/bin/python tools/benchmark_nano_cache_strategies.py \
  --checkpoint /path/to/checkpoint --hdf-root /path/to/hdf_data \
  --vae-path /path/to/Wan2.2_VAE.pth --qwen-tokenizer-path /path/to/tokenizer \
  --strategies asi --compile --graphs \
  --sample-index 64 --seed 0 --warmup-rounds 5 --measured-rounds 15 \
  --output-dir experiments/asi_graph_new
```

`--strategies` 可换成 `toca`、`worldcache`、`c3ache`。移除 `--graphs` 为 compile-only，
再移除 `--compile` 为 eager。Graph 要求 compile；同进程原生 Dense eager 配对对照自动执行。
C3ache 报告刷新与复用两个 chunk 的均摊时间。不能把复用 chunk 单独计成总体加速。

SpecPrune 使用 `tools/benchmark_nano_specprune.py`，数据路径同上，输出参数为 `--output`。
该工具保留已实测的 offload runtime hook 名称桥接，比较 eager/compile/Graph。
服务入口为 `cosmos_framework.scripts.action_policy_server_robolab_specprune`；
其 controller 使用同样的名称桥接，新的独立 checkout 尚需真实服务冒烟验证。

真机入口 `scripts/serve_cosmos_piper14_policy.py` 仅保留现有策略服务代码。
本次整理不连接、启动或移动机械臂。

请勿把 HDF 评测控制器的存在理解成所有策略已接到 RoboLab server。
当前通用 RoboLab server 是 Dense；SpecPrune 有独立 server；ASI/ToCa/WorldCache/C3ache
在本共同底座的已知入口是 HDF benchmark。需要在 Nano 上跑闭环时，必须先完成相应 server 接线和 GPU 冒烟。

## 验证边界

本次是源码归档和路径可移植化，运行 CPU 合同测试和语法检查。
历史 HDF GPU 测速记录保留在原 docs，但没有作为这个新 checkout 的重新测速结果。
全新 checkout 的 GPU 冒烟、RoboLab 闭环和真实硬件验证须随后按指定环境执行。
已有文档中的旧绝对路径是历史记录；当前启动以本页为准。
