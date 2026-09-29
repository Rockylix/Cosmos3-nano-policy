# Nano 多策略 FLOPs 对比

## 输入审计与适用范围

本表对应已有 `nano_cache_adapters_comparison` 单 chunk 测试，不重新推理。
权重为 `cosmos_battery/20k`，HDF 输入为 `hdf_data/perfect` 的 sample 0，4 步、guidance=3、shift=5。

**发现输入问题：** 沿相同数据预处理路径及模型实际使用的本地 Qwen2Tokenizer 离线重建时，`ai_caption` 为字典；chat template 未写入其中的任务文字。conditional/unconditional 均得到相同的 8 个文本 token，packing 加入 EOS/BOV 后均为 10 个 UND token。因此下表 FLOPs 和此前实测时间对应这一实际输入，不能视为正确注入任务文字后的完整验证。未擅自修复输入或重新测试。详见原始 JSON 的 `prompt_audit`。

## 口径

- Dense=100%；FMA（乘加）计 2 FLOPs。
- 统计预热后、instruction K/V cache 命中的 Transformer 逻辑矩阵乘法：Q/K/V/O、QK/AV、MLP，以及 ASI 额外打分矩阵乘法。
- Reasoner 每个 CFG 分支首次 prefill 一次后 offload。稳态 chunk 不重复计入 UND 参数投影/MLP，但 GEN query 对缓存 UND key/value 的 attention 仍计入。
- 不包含 VAE、输入/输出头、时间嵌入、UniPC、softmax/归一化/逐元素运算、排序/索引/拷贝，也不计硬件 tile padding。这不是完整端到端硬件指令 FLOPs。
- 实测为之前配对 Dense eager 与策略 compile+CUDA Graph 的同步 generation 时间；并非本次重测。

## 结果

| 策略 | TFLOP/chunk | 相对 Dense FLOPs | 减少量 | 等吞吐理论加速 | 之前配对实测加速 |
|---|---:|---:|---:|---:|---:|
| Dense eager | 414.438 | 100.00% | 0.00% | 1.000× | 1.000× |
| ASI | 247.662 | 59.76% | 40.24% | 1.673× | 1.811× |
| ToCa D/C/D/C，r=0.25 | 270.350 | 65.23% | 34.77% | 1.533× | 1.337× |
| WorldCache D/D/D/C | 310.829 | 75.00% | 25.00% | 1.333× | 1.488× |
| C3ache，refresh/hit 均摊 | 310.829 | 75.00% | 25.00% | 1.333× | 1.483× |

TFLOP/chunk 是计算量，不是 TFLOP/s。理论加速仅为 Dense FLOPs/策略 FLOPs，不是运行时间保证或严格速度上界。实测同时受编译、Graph、访存、缓存和内核效率影响。

## 参数与计算公式

权重 safetensors header 验证全部 36 层矩阵形状一致：

- hidden `D=4096`，Q 宽度 `Q=4096`，KV 宽度 `K=1024`，32 Q heads / 8 KV heads。
- SwiGLU intermediate `I=12288`，up/gate/down 三个矩阵。
- 每帧空间网格 `20×18=360`，L0+L1–L8 共 3240 个 vision token，另有 33 个 action token。
- Dense GEN `N=3273`；future `F=2880`；保留 L0/action 的基础部分 `P=393`。
- ASI 每个 future frame 保留 184，稀疏 GEN `Ns=393+8×184=1865`。
- 本次实际 `Uc=Uu=10`。

单层全量或稀疏 GEN 计算量，n 为当前实际 GEN 长度：

```text
LinearRow = 2 D (2Q + 2K)
MLPRow    = 6 D I
Full(n,u) = n (LinearRow + MLPRow) + 4 n (n+u) Q
Pair      = 36 [Full(N,Uc) + Full(N,Uu)]
Dense     = 4 Pair
```

ASI 首次 conditional 全量，剩余 7 次稀疏。Top-layer 选择前仍需对全部 36 层打分；额外 action-row attention 和 aligned future-key 点积也计入：

```text
Profile = 36 [4×32×(N+Uc)×Q + 2×32×360×Q]
ASI     = 36 [Full(N,Uc) + 3 Full(Ns,Uc) + 4 Full(Ns,Uu)] + Profile
```

Profile 为 0.065362 TFLOP/chunk。这里是已有 Nano adapter 的实测配置，不额外假定已加入 Edge velocity-cache 行为。

ToCa 在缓存步仍对全 GEN 计算 K/V；仅 P 个保护 token 计算 Q/O 和 attention，MLP 计算 P 加本层 fresh future token。评分来自同一次全量 attention，不重复计第二次全量 QK：

```text
fresh_l = floor[0.25 (1.5 - l/35) × 2880], l=0..35
Cached(u,l) = 4 P D Q + 4 N D K
              + 4 P (N+u) Q + (P+fresh_l) MLPRow
ToCa = 2 Pair + 2 sum_l [Cached(Uc,l) + Cached(Uu,l)]
```

`fresh_l` 从 B0 的 1080 降到 B35 的 360，使用逐层实际取整；CFG 独立，spatial bonus=0。

WorldCache 计算 6 个完整 forward，最后两次使用缓存，故 `3 Pair = 75% Dense`。
C3ache refresh 计算 8 次、hit 计算 4 次，周期 2：`(4 Pair + 2 Pair)/2 = 3 Pair`。单看 hit 是 50%，但不能把它作为均摊结果。此处复用相同 HDF 输入仅验证开销，不代表真实跨观测保真度。

## 复算与原始结果

```bash
cd /root/piper-cosmos3
PYTHONPATH=/root/piper-cosmos3/.worktrees/nano-cache-adapters:/root/piper-cosmos3 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python tools/estimate_nano_cache_flops.py
```

此命令只读权重 header、在 CPU 重建输入文本处理并计算公式，不加载模型推理、不下载资源。

- 脚本：`tools/estimate_nano_cache_flops.py`
- 数值及文本审计：`experiments/nano_cache_adapters_comparison/flops_comparison.json`
- 表格：`experiments/nano_cache_adapters_comparison/flops_comparison.csv`
- 原测速：[Nano 多策略单 chunk 测速结果](nano_multistrategy_benchmark_results_cn.md)

需用户确认后修复 caption 序列化、重新测量；任务文字改变可能影响 ASI token 选择、ToCa 排名、输出误差与延迟，不能仅修改表中的 UND 长度冒充新结果。
