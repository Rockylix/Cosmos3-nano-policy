# Nano 多策略单 chunk 测速结果

> 后续 FLOPs 审计发现：该次 HDF caption 为字典，本地 chat template 未注入任务文字；conditional/unconditional 文本 token 相同。以下时间和误差仍是该实际输入的测量，但不是正确指令输入的验证。尚未修复或重跑，详见 [FLOPs 与输入审计](nano_multistrategy_flops_cn.md)。

每策略3轮预热、10轮采样，与同进程Dense eager交替；策略使用compile+CUDA Graph。

| 策略 | Mean(s) | Median(s) | P90(s) | 配对 Dense Median(s) | 加速 | Graph replay/chunk |
|---|---:|---:|---:|---:|---:|---:|
| asi | 1.834202 | 1.835955 | 1.840764 | 3.324753 | 1.811× | 288 |
| toca | 2.485525 | 2.487248 | 2.495542 | 3.325962 | 1.337× | 288 |
| worldcache | 2.250570 | 2.250256 | 2.253092 | 3.347462 | 1.488× | 216 |
| c3ache | 2.261112 | 2.261287 | 2.262896 | 3.353473 | 1.483× | 144 |

C3ache时间按refresh+hit均摊，表中Graph次数为hit，refresh为288。相同HDF输入只用于测开销，不代表真实跨观测质量。
Reasoner首次双分支prefill后持续命中缓存；每个进程全程prefill次数=2，UND decoder CUDA参数=0。
计时为同步generate_samples_from_batch，不含初次prefill/编译/capture、HDF预处理和VAE decode。
ASI首轮采样已完成后，ToCa首次预热触发Dynamo上限8；在框架初始化后设置容量256，重新加载模型补跑后三策略。各自使用配对Dense，未拼接未完成的ToCa样本。

## 输出误差（完整返回tensor，非物理关节单位）

| 策略 | action MSE | action cos | vision cos | vision rel-L2 |
|---|---:|---:|---:|---:|
| asi | 0.00058060774 | 0.999478 | 0.775826 | 0.630960 |
| toca | 0.00062278687 | 0.999440 | 0.989379 | 0.148763 |
| worldcache | 0.00037192131 | 0.999673 | 0.996095 | 0.088326 |
| c3ache | 0.00017173798 | 0.999867 | 0.999772 | 0.021534 |

没有运行RoboLab或真实机械臂，没有任务成功率结论。

详见 [方法与命令](nano_multistrategy_offload_cn.md)。
