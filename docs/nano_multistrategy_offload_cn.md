# Nano：ASI / ToCa / WorldCache / C3ache 单 chunk 对比

## 环境和显存约束

本轮用户明确允许使用现有宿主机环境：
`/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python`。
不使用 RoboLab，不连接机械臂，不下载环境或权重。

强制启用 `COSMOS3_REASONER_OFFLOAD=1`、`COSMOS3_REASONER_KVCACHE=1`、
`COSMOS3_POLICY_REASONER_CACHE=1`，关闭通用 `COSMOS3_LAYER_OFFLOAD`。
相同指令第一次 conditional/unconditional 各 prefill 一次，之后共享这两个分支的 UND K/V；
UND/bridge 卸载至 CPU，GEN 常驻 GPU。指令 token 改变会自动重新 prefill。
这是一次指令初始化中的两个 CFG 分支，不是每个 chunk 再运行两次 Reasoner。

测速脚本断言：首次恰好两次 UND-only forward，之后计数不变，两个 MemoryState 均为 gen-only，
且 `_policy_reasoner_memory_cache` 对象未被替换。还检查全部 decoder 的 UND 参数在 CPU。

## 迁移代码

- `piper_cosmos/cosmos3/nano_cache_strategies.py`：四策略的 GEN tensor kernels 和请求控制器。
- `piper_cosmos/cosmos3/nano_cache_vendor/worldcache.py`：Edge 的冻结预测公式与原始 patch-token 输出处理。
- `piper_cosmos/cosmos3/nano_cache_vendor/toca_joint_attention.py`：Edge 同一次 QK 产出 AV 和评分的 kernel。
- `tools/benchmark_nano_cache_strategies.py`：HDF5 配对测速及输出误差。
- `tests/test_nano_cache_strategies.py`：预算、布局、调度和 patchify 数学检查。

本轮 Framework 源码副本在 `.worktrees/nano-cache-adapters/cosmos_framework`。
它复制自已有 `/root/cosmos3/cosmos/packages/cosmos3/cosmos_framework`（含已有 offload 修改），
应用项目已有 `patches/cosmos-framework/0003-add-nano-version1-offload-hooks.patch`。
没有修改原 Framework、原 Version1 controller 或已有部署入口。
该副本不是真正 Git worktree，未新建分支、提交或推送；`.worktrees/` 由本地 Git exclude 忽略。
迁移服务器时不能只复制新 controller：还需保留含 offload 的 Framework 以及上述 hook patch。

## 冻结策略

| 策略 | 实际运算 |
|---|---|
| Dense | 原生 eager，4 step × 2 CFG；不运行 controller |
| ASI | step0 conditional 全量，全部实际层打分；Top6 block 构造逐帧 Core80，全部层构造共享 Stable104；其余7 stack使用K184 |
| ToCa | D/C/D/C，r=0.25，layer_slope=0.5，age_weight=0.25，period=2，spatial_bonus=0，CFG independent |
| WorldCache | D/D/D/C；最后一步对原始 vision patch / action prediction 做异质外推；分位点0.3/0.7，n_max=6 |
| C3ache | refresh_period=2；刷新chunk全量；复用chunk前2step整栈 residual复用，后2step全量；最后norm和输出头仍执行 |

ASI 未选 hidden 恢复为本次 stack 输入；不加入 velocity cache。
ASI 的 Core 池预留 Stable 预算，与本次 Edge Core80+Stable104 保持一致。
保留 L0、condition action 和全部预测 action，使用选中 token 的原始 RoPE，不重新编号。
ToCa 缓存步：保护 token 实算 Q/O，全部GEN实算K/V；future attention residual复用；
MLP实算保护token与逐层 fresh future token。Nano 36层下，fresh比例从37.5%降至12.5%。
ToCa不再次计算一套全量QK-softmax用于评分。
WorldCache使用native llm2vae输出，避免裁切后的latent丢失padding patch信息。
C3ache缓存位于最后RMSNorm之前，跨chunk使用当前输入加缓存整栈residual。

本次 HDF 输入布局已实测为 `[9,20,18]`：360 spatial token/frame，
L0=360，future=2880，action=33，GEN=3273；ASI 稀疏 GEN=1865。
层数、head、KV head、head dim 从实际模型取得，不使用 Edge 的28层常数。

## 运行

```bash
cd /root/piper-cosmos3
export PYTHONPATH=/root/piper-cosmos3/.worktrees/nano-cache-adapters:/root/piper-cosmos3
export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/benchmark_nano_cache_strategies.py \
  --compile --graphs \
  --strategies asi toca worldcache c3ache \
  --checkpoint /root/piper-cosmos3/cosmos_battery/20k \
  --hdf-root /root/piper-cosmos3/hdf_data/perfect \
  --sample-index 0 --seed 0 \
  --warmup-rounds 3 --measured-rounds 10 \
  --output-dir /root/piper-cosmos3/experiments/nano_cache_adapters_graph_v1
```

输出目录必须是新目录，已存在时会拒绝覆盖。
上述目录已用于本轮运行；重跑请换唯一后缀。
去掉 `--compile --graphs` 即 eager；增加 `dense_adapter` 用于与原生 Dense 的运算等价性检查。

时间范围为 CUDA synchronize 包围的 `generate_samples_from_batch`，不含输入clone、HDF读取/预处理、
RPC、VAE decode、首轮instruction prefill和编译/capture。
每策略都与同进程原生Dense交替运行；报告Mean/Median/P90与配对Median加速比。
compile/Graph作用于常驻GEN层的tensor kernel；调度、索引、native heads和solver保持在图外。
`graphs_requested` 只代表配置，实际生效应检查 `observed_graph_replays`。

C3ache每轮连续运行刷新和复用，再取两者平均作为该轮chunk时间。
本次复用相同HDF样本验证计时和接口，**不用于声称真实跨观测视频/动作保真度或任务成功率**。
action/vision误差针对完整返回tensor（含条件/填充维度），不是物理关节单位或单独future像素指标。

## 已完成的首轮检查

`experiments/nano_cache_adapters_eager_smoke_v1/results.json`：

- 原生Dense与适配Dense的action、vision逐元素一致（MSE=0）。
- 原生Dense重复请求逐元素一致；四策略均完成完整8次CFG调度。
- ASI为1全量+7稀疏，ToCa实际288层调用；WorldCache216层调用；C3ache复用chunk144层调用。
- 首次UND prefill=2，之后无重复prefill；UND decoder CUDA参数=0，GEN=12.94GiB。
- 首轮全模式峰值allocated约19.15GiB，无NaN/Inf输出。
- 3项CPU测试通过。

上述 eager 冒烟只做了1轮预热+1轮采样，**不是正式稳定加速比**。

## 正式采样已完成

见 [完整结果](nano_multistrategy_benchmark_results_cn.md) 和
`experiments/nano_cache_adapters_comparison/` 的 CSV/JSON/报告。

- ASI：`nano_cache_adapters_graph_v1`，10个正式样本。
- ToCa / WorldCache / C3ache：`nano_cache_adapters_graph_v2`，每策略10个正式样本。
- 各自3轮预热；所有加速分母均为同进程、同输入交替运行的Dense eager。
- ASI / ToCa 每chunk观察到288次Graph replay，WorldCache216次；C3ache刷新288、复用144。
- 四策略正式重复样本的action/vision最大绝对差均为0。
- ASI进程峰值allocated为14.97GiB，后三策略进程19.07GiB；这不是reserved或nvidia-smi进程占用。
- 全程保持每个进程仅2次UND prefill，之后同指令缓存命中。
- CPU测试3项通过；额外GPU测试1项通过：32 Query heads / 8 KV heads的ToCa真实AV与incoming score，与完整FP32 softmax参考比较。
- 原生Dense与适配Dense逐元素一致的检查为eager对eager；compile可能引入浮点舍入差异，不声称与eager逐元素相等。

ToCa首次编译预热因框架初始化将Dynamo容量恢复为8而停止；修正是在模型初始化后设置容量256，
允许36层不同的fresh-token形状。未更改算法、精度或预算；失败的预热没有纳入结果。

```bash
PYTHONPATH=/root/piper-cosmos3/.worktrees/nano-cache-adapters:/root/piper-cosmos3 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest \
  tests/test_nano_cache_strategies.py tests/test_nano_toca_joint_gpu.py -q
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python tools/summarize_nano_cache_benchmark.py
```

GPU测试必须在可见宿主机CUDA的授权环境运行；若CUDA不可见，它会skip，不能算GPU验证通过。
