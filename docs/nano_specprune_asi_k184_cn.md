# Cosmos3 Nano SpecPrune-VLA（ASI K184 对齐版）

## 语义与预算

本分支把 Edge 已冻结的 observation-driven SpecPrune-VLA 迁移到 Nano。选择只依赖真实观测帧：

1. B0/B1 使用 observation query 对 instruction key 的 attention 做 Local 选择；
2. 上一 chunk 的 B17/B35 observation-to-instruction attention 形成 Global 历史；
3. 当前与上一 chunk 的同坐标观测 patch cosine 形成 Dynamic 保护集合；
4. B18/B25/B31 使用 action query 对 observation key 的 attention 更新重要性；
5. B13/B19/B26/B32 进行嵌套退出。

某层得到的一个 `[360]` 空间 mask 同时应用于 L1–L8。L0、32 个预测 action 和 condition action 始终保留。退出 token 保存该层输出 hidden，稀疏序列继续进入后续 block，最后在 RMSNorm、vision head 和 action head 前恢复完整 GEN 布局。没有 velocity cache，也不比较 future noise。

ASI K184 在一次 step0 conditional dense profile 加七次 K184 stack 下的平均 future-token 保留率为：

```text
(360 + 7 * 184) / (8 * 360) = 57.2222%
```

SpecPrune 保留其逐层退出语义，因此不强制每层恰好 K184；运行摘要中的 `mean_future_token_retention` 才是与 ASI 对齐的主指标。初始配置沿用 Edge 的绝对预算 `Local=58, Global=72, min=184`，只把 Dynamic 的稳定上限按 340→360 网格比例缩放到 331。这样不会因为 Nano 网格更大而错误放宽预算。配置位于 `configs/specprune_asi_k184_budget.json`，运行数据用于离线校准，不能用任务成功标签反向调参。

## 36 层映射

Edge 的 28 层位置按归一化深度映射到 Nano 36 层：

| 功能 | Edge | Nano |
|---|---|---|
| Local | B0/B1 | B0/B1 |
| Global | B13/B27 | B17/B35 |
| Action update | B14/B19/B24 | B18/B25/B31 |
| Prune | B10/B15/B20/B25 | B13/B19/B26/B32 |

## 启动

必须沿用 Nano 的 Reasoner prefill、KV cache 与 offload：

```bash
cd /path/to/Cosmos3-nano-policy
export COSMOS3_REASONER_OFFLOAD=1
export PYTHONPATH="$PWD"

uv run python -m cosmos_framework.scripts.action_policy_server_robolab_specprune \
  --checkpoint-path /path/to/Cosmos3-Nano-Policy-DROID \
  --mode specprune \
  --num-steps 4 \
  --shift 5 \
  --guidance 3 \
  --port 8000 \
  --specprune-output-dir /tmp/cosmos3_action_server/nano_specprune
```

Dense 对照只把 `--mode specprune` 改成 `--mode dense`。当前第一版固定为 eager；compile/CUDA Graph 必须在 eager 的 action/vision 数值和实际 token 计数通过后单独适配。

## 验证顺序

```bash
uv run pytest -q tests/inference/test_action_specprune.py
```

随后使用同一 HDF5、seed、四步去噪和 guidance=3 配对比较：

- action relative-L2/cosine/max error；
- vision latent relative-L2/cosine；
- 每个 block/step/branch 的 future token 数；
- `mean_future_token_retention` 与 57.2222% 的差距；
- warmed、CUDA-synchronized 单 chunk median/P90。

第一轮只做正确性与三任务 smoke。预算冻结后才运行更大规模闭环测试。
