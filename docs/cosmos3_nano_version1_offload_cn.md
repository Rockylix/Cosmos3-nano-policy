# Cosmos3 Nano Version1：Reasoner offload 与单 chunk 基准

## 实现边界

- 使用 Cosmos Framework 现有 `COSMOS3_REASONER_OFFLOAD`：UND/Reasoner 只在首次同指令请求中做 conditional/unconditional prefill，并保存逐层 K/V。
- `COSMOS3_POLICY_REASONER_CACHE=1` 按 conditional/unconditional token 序列缓存 K/V；指令 token 改变时自动 cache miss 并重新 prefill。
- prefill 后 UND 层及 bridge 位于 CPU，12.94 GiB GEN 参数常驻 GPU；实测生成阶段约占 14.5 GiB CUDA allocated。
- Version1 的 step 0 conditional stack 全量 profile B0-B35；随后 7 个 CFG stack 只计算 L0、全部 action token，以及每个 L1-L8 的 Core64 + Stable120，即 K184。
- 未选 future hidden 在完整 stack 输出处恢复为当前 stack 输入；不使用 velocity cache。
- compile/graph 只作用于已经常驻 GPU 的 GEN 稀疏路径。step 0 保持 eager；CUDA Graph 的边界是一整个 36-block sparse stack，不是逐 block graph。

## 文件

- `piper_cosmos/cosmos3/nano_version1_sparse.py`：profile、Core/Stable mask、稀疏 pack/restore。
- `piper_cosmos/cosmos3/nano_version1_graph.py`：compile-only 与 36-block Global Graph。
- `patches/cosmos-framework/0003-add-nano-version1-offload-hooks.patch`：Framework hook 与跨请求 K_AR cache。
- `tools/benchmark_nano_version1_offload.py`：HDF5 单 chunk 四模式交替基准。

## 应用 Framework patch

当前验证所用 Framework：`/root/cosmos3/cosmos/packages/cosmos3`，其 HEAD 含 Reasoner offload。

```bash
cd /root/piper-cosmos3
./scripts/manage_cosmos3_nano_version1_patch.sh check /root/cosmos3/cosmos/packages/cosmos3
./scripts/manage_cosmos3_nano_version1_patch.sh apply /root/cosmos3/cosmos/packages/cosmos3
```

正常路径默认关闭；只有创建 Nano Version1 controller 或传 `--nano-version1` 才会稀疏。

## 环境变量

```bash
export PYTHONPATH=/root/cosmos3/cosmos/packages/cosmos3:/root/piper-cosmos3
export COSMOS3_REASONER_OFFLOAD=1
export COSMOS3_REASONER_KVCACHE=1
export COSMOS3_REASONER_EMPTY_CACHE=1
export COSMOS3_REASONER_PIN_CPU_MASTER=0
export COSMOS3_POLICY_REASONER_CACHE=1
```

不要同时设置 `COSMOS3_LAYER_OFFLOAD=1`；Reasoner offload 已让 GEN 常驻 GPU，通用逐层 offload 会破坏 compile/graph 的固定参数地址。

## HDF5 基准命令

```bash
cd /root/piper-cosmos3
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/benchmark_nano_version1_offload.py \
  --checkpoint /root/piper-cosmos3/cosmos_battery/20k \
  --hdf-root /root/piper-cosmos3/hdf_data/perfect \
  --dataset-config /root/piper-cosmos3/configs/dataset_configs/battery_assemble_hdf5.yaml \
  --warmup-rounds 2 \
  --measured-rounds 10 \
  --output-dir /root/piper-cosmos3/experiments/nano_version1_offload_hdf_v1
```

计时范围为 CUDA synchronize 包围的 `generate_samples_from_batch`，不含 HDF5/图像预处理、RPC 和 VAE decode。首个 warmup 请求建立 cond/uncond K_AR，正式 10 轮均为同指令 cache-hit 的稳定 policy 路径。

## RTX 4090 实测

数据：`hdf_data/perfect` sample 0；checkpoint：`cosmos_battery/20k`；seed 0；4 steps；CFG 3.0；shift 5。

| 策略 | Mean | Median | P90 | 相对 Dense 加速（Median） |
|---|---:|---:|---:|---:|
| Dense eager | 3.336942 s | 3.343138 s | 3.354787 s | 1.000x |
| Version1 eager | 2.408374 s | 2.059603 s | 2.064652 s | 1.623x |
| Version1 + compile-only | 1.936190 s | 1.940740 s | 1.947395 s | 1.723x |
| Version1 + compile + Global Graph | 1.934862 s | 1.940823 s | 1.946280 s | 1.722x |

Version1 eager 的 raw mean 被一次 5.593 s 的孤立停顿拉高；其余 9 次集中在 2.034-2.065 s，因此稳定比较以 Median/P90 为主。Global Graph 在本轮没有超过 compile-only，说明固定 K184 下主体仍是 GPU kernel 计算，Python/kernel-launch 开销已经很小。

Token accounting：full GEN 3273，sparse GEN 1865，每个稀疏 block 少计算 1408 token，保留率 56.98%；1 个 dense stack + 7 个 sparse stacks。

单样本输出误差（Version1 compile/graph 相对 Dense）：action MSE `8.7125e-4`、cosine `0.999216`、relative-L2 `0.039621`；vision cosine `0.772309`、relative-L2 `0.635354`。compile-only 与 Global Graph 输出逐元素一致。

完整原始数据见：`experiments/nano_version1_offload_hdf_v1/metrics.json`。

## Policy server

完成 patch 并设置上述环境变量后，可在现有 server 命令末尾加入：

```bash
--nano-version1 --torch-compile --cuda-graphs \
--nano-version1-output-dir /tmp/cosmos3-nano-version1
```

如需 Dense，只移除 `--nano-version1`；如需 Version1 eager，使用 `--nano-version1 --no-torch-compile --no-cuda-graphs`。
