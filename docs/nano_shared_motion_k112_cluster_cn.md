# Cosmos3-Nano Shared-Motion K112：集群下载、冒烟与 RoboLab 指南

## 1. 当前代码边界

本分支基于 NVIDIA `cosmos-framework`：

```text
base: 5eee9ed574255f017b192161bfbb5a10253d65cf
branch: experiment/nano-shared-motion-k112
```

没有下载或提交任何模型权重。K112 通过独立入口启用，官方 baseline 入口不变：

- 官方 baseline：`cosmos_framework.scripts.action_policy_server_robolab`
- 本分支 Dense/K112 对比入口：
  `cosmos_framework.scripts.action_policy_server_robolab_nano_k112`

`--mode dense` 是默认值。只有显式传入 `--mode k112` 才会启用稀疏。

## 2. Nano K112 实际执行过程

该实现只接受 Cosmos3-Nano 的实时结构：

```text
36 Transformer blocks
hidden = 4096
32 Query heads / 8 KV heads
condition latent L0 + future L1..L8
32 predicted action rows
4 denoise steps、shift=5、全程 CFG
```

每个 request：

1. `step 0 / conditional / B0-B35` 全量计算，每个 block 额外重算一次
   action-row QK softmax，用来保存 `[36,8,spatial]` 的 post-RoPE attention。
   Transformer block 本身不重复执行。
2. action horizon 与 latent 对齐时，四个 action 使用：

   \[
   \alpha=[1/6,\ 1/3,\ 1/3,\ 1/6].
   \]

3. 每个 block 计算：

   \[
   R_l=\frac1{8}\sum_f\sum_p V_{l,f,p},
   \]

   \[
   H_l=\frac1{8}\sum_f
   \frac{-\sum_p \bar V_{l,f,p}\log(\bar V_{l,f,p}+\epsilon)}{\log S},
   \]

   \[
   T_l=\operatorname{Mean}_{f<g}
   \cos(V_{l,f,:},V_{l,g,:}),
   \]

   \[
   Q_l=R_l(1-H_l)\max(T_l,0).
   \]

4. 选择 `Q_l` 最大的四个 Nano block，按 `Q_l` 加权合并。对每个空间位置取
   L1-L8 分数的第三小值，即要求它至少在 6/8 个 future latent 中保持强关注；
   最后一次性选择 Top112。
5. L1-L8 共用同一个 `[spatial]` mask；L0、全部 action、全部文本 token 始终保留。
6. 其余七个 CFG stack 在 B0 前 pack 一次，B0-B35 全部接收真实缩短的序列，
   B35 后 restore 一次。RoPE 从原始 position 对应的 cos/sin 裁剪，不重新编号。
7. step-0 unconditional 在 K112 外将 CFG delta 置零；step 1-3 的背景使用
   step-0 guided future velocity，K112 区域和 action 使用当前 step velocity。

当前实现不支持多 GPU FSDP、context parallel、CFGP、torch compile、CUDA graphs
或 NATTEN。请先在单张 80GB 级 GPU 上验证；这些限制会在不满足时直接报错，
不会静默退化。

## 3. 集群拉取代码

先把本地实验分支推到项目远程，然后在集群执行：

```bash
git clone git@github.com:Rockylix/Cosmos3-Task-relevant-Inference.git \
  cosmos-framework-nano-k112
cd cosmos-framework-nano-k112
git switch --track origin/experiment/nano-shared-motion-k112
git rev-parse HEAD
```

安装环境时使用仓库支持的 CUDA 组合。例如在 NVIDIA PyTorch/CUDA 13.0 容器内：

```bash
export LD_LIBRARY_PATH=''
uv sync --all-extras --group=cu130-train --group=policy-server
```

如果集群通过 Slurm 申请 GPU，先进入 GPU allocation 后再启动 server。例如：

```bash
srun --partition=normal --account=peilab --gres=gpu:1 \
  --cpus-per-task=16 --mem=128G --pty bash
```

## 4. 不使用 HF token，通过镜像下载权重

以下命令只写入集群 scratch/cache，不写 Git 工作树：

```bash
export ASSET_ROOT=/project/peilab/$USER/cosmos3_nano_assets
export HF_HOME="$ASSET_ROOT/hf_home"
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$HF_HOME"

uvx --from huggingface-hub hf download \
  nvidia/Cosmos3-Nano-Policy-DROID \
  --revision main

uvx --from huggingface-hub hf download \
  Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e

uvx --from huggingface-hub hf download \
  Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --include '*.json' '*.txt'
```

下载完成后可离线启动：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

若镜像明确返回 gated/approval 错误，镜像不能绕过上游授权；此时只能在已获授权的
缓存中下载。不要反复删除已完成的 snapshot。

## 5. 先跑 Dense server

在 Nano Framework 终端：

```bash
cd /path/to/cosmos-framework-nano-k112
export LD_LIBRARY_PATH=''
export HF_HOME=/project/peilab/$USER/cosmos3_nano_assets/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python -m \
  cosmos_framework.scripts.action_policy_server_robolab_nano_k112 \
  --mode dense \
  --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
  --host 0.0.0.0 \
  --port 8000 \
  --num-steps 4 \
  --guidance 3 \
  --shift 5 \
  --seed 0
```

这保留官方 policy seed 语义：`seed=0`、`deterministic_seed=False`。不同 request
从同一个 RNG 流依次取得扩散 seed；Dense/K112 只要使用相同 request 顺序即可配对。

## 6. 用 Piper14 HDF5 做结构冒烟

Piper14 HDF5 是 14D 双臂状态，官方 Nano DROID 是 7D arm + 1D gripper。
两者 action/state 语义不兼容。本工具只取 HDF5 的三路图像，并显式使用零 DROID
state，验证 HDF5 解码、WebSocket、VAE/Transformer/action head 和返回 shape；它不能
作为 Piper14 精度结论。

在另一终端执行：

```bash
cd /path/to/cosmos-framework-nano-k112

uv run python tools/smoke_nano_policy_from_piper_hdf5.py \
  --hdf5 /path/to/piper-cosmos3/hdf_data/perfect/episode_5.hdf5 \
  --frame-index 0 \
  --prompt 'Pick up the banana and place it in the bowl' \
  --host 127.0.0.1 \
  --port 8000 \
  --allow-droid-zero-state-smoke \
  --output /tmp/nano_dense_hdf5_smoke.json
```

验收：

- 返回 action 是二维张量；
- action 全部 finite；
- server 无 token-layout、VAE、checkpoint 或 CUDA 错误。

如果要做真实 Piper14 Nano 推理，必须换成训练后的 Piper14 Nano checkpoint，并使用
Piper14 14D domain/action pipeline；不能使用官方 DROID action head 替代。

## 7. 启动 K112 server

停止 Dense server 后执行：

```bash
cd /path/to/cosmos-framework-nano-k112
export LD_LIBRARY_PATH=''
export HF_HOME=/project/peilab/$USER/cosmos3_nano_assets/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python -m \
  cosmos_framework.scripts.action_policy_server_robolab_nano_k112 \
  --mode k112 \
  --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
  --host 0.0.0.0 \
  --port 8000 \
  --num-steps 4 \
  --guidance 3 \
  --shift 5 \
  --seed 0 \
  --decode-video \
  --k112-output-dir /project/peilab/$USER/nano_k112_runs/server
```

再次运行上一节 HDF5 命令，将输出改为：

```text
/tmp/nano_k112_hdf5_smoke.json
```

每个 K112 request 应生成：

```text
request_000000/
├── generation.json
├── shared_motion_k112_profile.pt
└── token_savings_summary.json
```

重点检查 `token_savings_summary.json`：

- `model_tier == Cosmos3-Nano`
- `transformer_blocks == 36`
- `shared_motion_tokens_per_future == 112`
- `dense_stack_count == 1`
- `sparse_stack_count == 7`
- `pack_count == restore_count == 7`
- `selected_profile_blocks` 均位于 `[0,35]`

## 8. RoboLab 单任务冒烟

Nano server 可以在大显存机器上运行，RoboLab/Isaac Sim 可以在另一台 4090 上通过
WebSocket 连接。RoboLab 终端执行：

```bash
cd /path/to/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y

uv run python policies/cosmos3/run.py \
  --remote-host <NANO_SERVER_IP> \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --headless \
  --video-mode all \
  --output-folder-name nano_k112_banana_smoke
```

先分别用 `--mode dense` 和 `--mode k112` 跑同一任务，保持 RoboLab 注册中的官方
simulation seed `1`，server 使用 policy seed `0`。

## 9. 官方 1200 episodes

单任务通过后，分别启动 Dense 和 K112 server，并各运行一次完整评测：

```bash
uv run python policies/cosmos3/run.py \
  --remote-host <NANO_SERVER_IP> \
  --remote-port 8000 \
  --num-envs 1 \
  --num-runs 10 \
  --headless \
  --video-mode none \
  --output-folder-name nano_dense_official_seed_120x10
```

K112 只替换输出目录：

```text
--output-folder-name nano_k112_official_seed_120x10
```

最终按 `episode_results.jsonl` 的 `success` 字段统计：

```text
120 tasks × 10 rollouts = 1200 episodes
success count / 1200
```

不要用 `score` 代替 `success`。K112 的成功率、动作一致性和加速都必须在 Nano 上
重新测量，不能引用 Edge 的结论。
