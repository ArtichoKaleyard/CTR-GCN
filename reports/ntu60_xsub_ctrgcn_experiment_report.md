# CTR-GCN NTU60 Cross-Subject 复现实验报告

## 概述

本报告记录在当前 `modern` 分支上，基于 Foundry 现代化训练链路复现 CTR-GCN 在 NTU RGB+D 60 Cross-Subject 协议下的实验结果。实验覆盖两类目标：

- 三组 joint 单流拓扑消融：full CTR-GCN、关闭通道级拓扑、关闭可学习拓扑。
- full CTR-GCN 四流复现：joint、bone、joint motion、bone motion，并计算四流融合精度。

本次实验已经完成 6 个 65-epoch 训练任务，并额外导出四流验证集 per-sample score，用于融合复现。

## 实验环境

| 项目 | 值 |
|---|---|
| 节点 | `WSL2` |
| 分支 | `modern` |
| 当前提交 | `11422c17fcdf6712eae9ffc5f187178915edf001` |
| Python | 3.12 |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | `12.8` |
| Foundry | `0.3.9` |
| 依赖管理 | `uv` |

数据路径为 `./data/ntu`，当前是指向 `/home/atk/Datasets/ntu-rgbd` 的软链接。数据文件为 `data.npy` 和 `labels.pkl`。

## 数据与协议

本次实验使用 NTU RGB+D 60 skeleton 数据，Cross-Subject 协议。

数据检查结果：

| 项目 | 值 |
|---|---|
| 数据 shape | `(56578, 3, 300, 25, 2)` |
| 数据类型 | `float32` |
| 标签范围 | `0..59` |
| xsub train | `40091` |
| xsub val | `16487` |

Foundry 验证集顺序为 `shuffle=False`，因此四流 score 可以按样本名一致性进行融合。

## 配置与训练口径

主要训练配置：

| 项目 | 值 |
|---|---|
| epoch | `65` |
| optimizer | SGD |
| base learning rate | `0.1` |
| weight decay | `0.0004` |
| momentum | `0.9` |
| nesterov | `true` |
| scheduler | multistep |
| milestones | `[35, 55]` |
| gamma | `0.1` |
| batch size | `16` |
| gradient accumulation | `4` |
| window size | `64` |
| train `p_interval` | `[0.5, 1.0]` |
| train random rotation | `true` |
| DataLoader workers | `0` |

`num_workers` 保持为 0 是当前 WSL2 环境下的稳定选择。压测中 `num_workers=2` 在 4 个 CUDA batch 后 RSS 峰值约 28.7 GB，`num_workers=4` 被系统以 exit 137 终止；`num_workers=0` 同样 4 个 CUDA batch 约 1.5 GB RSS。

## 训练命令

四组 joint 单流消融：

```bash
bash run.sh ntu60_xsub_joint
bash run.sh ntu60_xsub_no_channel_topology
bash run.sh ntu60_xsub_no_dynamic
bash run.sh ntu60_xsub_q_only
```

full CTR-GCN 四流中的其他三流：

```bash
bash run.sh ntu60_xsub_bone
bash run.sh ntu60_xsub_joint_motion
bash run.sh ntu60_xsub_bone_motion
```

四流 score 导出与融合：

```bash
uv run python scripts/export_ctrgcn_scores.py \
  --batch-size 64 \
  --device cuda \
  --eval-p-interval 0.95 \
  --score-name epoch1_test_score_p095.pkl
```

其中 `--eval-p-interval 0.95` 用于在导出/评估阶段对齐原始仓库常见 test crop 口径；训练 checkpoint 未重训。

## 单流与消融结果

所有实验均完整跑完 65 epoch，并生成 `best.pt`、`last.pt`、`summary.json` 和 `metrics.json`。

| 实验 | artifact 目录 | best epoch | best val acc | final val acc |
|---|---|---:|---:|---:|
| full CTR-GCN joint | `artifacts/skeleton/ntu60_ctrgcn_xsub_joint` | 59 | 87.0625% | 86.6986% |
| no channel topology | `artifacts/skeleton/ntu60_ctrgcn_xsub_no_ctr_joint` | 63 | 86.7653% | 86.5106% |
| no learnable topology | `artifacts/skeleton/ntu60_ctrgcn_xsub_no_dynamic_joint` | 60 | 86.2134% | 85.8434% |
| pure Q topology | `artifacts/skeleton/ntu60_ctrgcn_xsub_q_only` | 65 | 85.5462% | 85.5462% |
| full CTR-GCN bone | `artifacts/skeleton/ntu60_ctrgcn_xsub_bone` | 62 | 88.3605% | 88.3120% |
| full CTR-GCN joint motion | `artifacts/skeleton/ntu60_ctrgcn_xsub_joint_motion` | 60 | 84.8123% | 84.8001% |
| full CTR-GCN bone motion | `artifacts/skeleton/ntu60_ctrgcn_xsub_bone_motion` | 61 | 85.4308% | 85.0306% |

消融对比以 joint 单流为基准：

| 对比 | best val acc | 相对 full joint 差值 |
|---|---:|---:|
| full CTR-GCN joint | 87.0625% | 0.0000 pp |
| no channel topology | 86.7653% | -0.2972 pp |
| no learnable topology | 86.2134% | -0.8491 pp |
| pure Q topology | 85.5462% | -1.5163 pp |

其中 pure Q topology 对齐论文中 `R = A + alpha * Q` 的拆项语义：移除共享拓扑先验 `A`，保留通道级相关性拓扑 `alpha * Q`。该实验的最终 `alpha` 没有停在 0；`best.pt` 中 10 个 `gcn1.alpha` 的范围为 `[-0.2252, 0.2069]`，绝对值均值为 `0.1472`。

这说明在当前 Foundry 复现口径下，完整的 `A + alpha * Q` 组合仍然最好；只保留共享拓扑 `A` 的 no channel topology 下降最小，固定共享拓扑的 no learnable topology 下降更明显，去掉共享拓扑先验后只保留 `alpha * Q` 的 pure Q topology 下降最大。

## 四流融合结果

四流融合使用官方仓库 `ensemble.py` 中常见权重：

| stream | weight |
|---|---:|
| joint | 0.6 |
| bone | 0.6 |
| joint motion | 0.4 |
| bone motion | 0.4 |

### 默认验证裁剪口径

使用训练配置默认验证裁剪，即验证阶段取 `p_interval` 上界 `1.0`，导出文件名为 `epoch1_test_score.pkl`。

| stream | best epoch | val acc |
|---|---:|---:|
| joint | 59 | 87.0625% |
| bone | 62 | 88.3605% |
| joint motion | 60 | 84.8123% |
| bone motion | 61 | 85.4309% |

融合结果：

| 指标 | 值 |
|---|---:|
| Top-1 | 90.0103% |
| Top-5 | 98.2896% |

### 官方式验证裁剪口径

在不重训的前提下，导出 score 时覆盖验证裁剪为 `p_interval=[0.95]`，导出文件名为 `epoch1_test_score_p095.pkl`。

| stream | best epoch | val acc |
|---|---:|---:|
| joint | 59 | 87.1899% |
| bone | 62 | 88.5122% |
| joint motion | 60 | 85.2975% |
| bone motion | 61 | 85.7706% |

融合结果：

| 指标 | 值 |
|---|---:|
| Top-1 | 90.2469% |
| Top-5 | 98.3805% |

相较 `p=1.0` 验证裁剪，`p=0.95` 带来：

| 指标 | 变化 |
|---|---:|
| Top-1 | +0.2366 pp |
| Top-5 | +0.0909 pp |

## 产物清单

训练产物目录：

```text
artifacts/skeleton/ntu60_ctrgcn_xsub_joint
artifacts/skeleton/ntu60_ctrgcn_xsub_bone
artifacts/skeleton/ntu60_ctrgcn_xsub_joint_motion
artifacts/skeleton/ntu60_ctrgcn_xsub_bone_motion
artifacts/skeleton/ntu60_ctrgcn_xsub_no_ctr_joint
artifacts/skeleton/ntu60_ctrgcn_xsub_no_dynamic_joint
artifacts/skeleton/ntu60_ctrgcn_xsub_q_only
```

四流 score 文件：

```text
artifacts/skeleton/ntu60_ctrgcn_xsub_joint/epoch1_test_score.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_bone/epoch1_test_score.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_joint_motion/epoch1_test_score.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_bone_motion/epoch1_test_score.pkl

artifacts/skeleton/ntu60_ctrgcn_xsub_joint/epoch1_test_score_p095.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_bone/epoch1_test_score_p095.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_joint_motion/epoch1_test_score_p095.pkl
artifacts/skeleton/ntu60_ctrgcn_xsub_bone_motion/epoch1_test_score_p095.pkl
```

实验配置与 manifest：

```text
conf/experiments/ntu60_xsub_joint.yaml
conf/experiments/ntu60_xsub_bone.yaml
conf/experiments/ntu60_xsub_joint_motion.yaml
conf/experiments/ntu60_xsub_bone_motion.yaml
conf/experiments/ntu60_xsub_no_channel_topology.yaml
conf/experiments/ntu60_xsub_no_dynamic.yaml
conf/experiments/ntu60_xsub_q_only.yaml

conf/manifests/ntu60_xsub_ctrgcn_joint.yaml
conf/manifests/ntu60_xsub_ctrgcn_bone.yaml
conf/manifests/ntu60_xsub_ctrgcn_joint_motion.yaml
conf/manifests/ntu60_xsub_ctrgcn_bone_motion.yaml
conf/manifests/ntu60_xsub_ctrgcn_no_ctr.yaml
conf/manifests/ntu60_xsub_ctrgcn_no_dynamic.yaml
conf/manifests/ntu60_xsub_ctrgcn_q_only.yaml
```

## 与论文和官方仓库的对齐情况

已经对齐的部分：

- 数据集：NTU RGB+D 60。
- 协议：Cross-Subject。
- 四流定义：joint、bone、joint motion、bone motion。
- 四流融合权重：`0.6 / 0.6 / 0.4 / 0.4`。
- 主要训练长度和学习率日程：65 epoch，初始学习率 0.1，milestones `[35, 55]`。
- skeleton 时间窗口：`window_size=64`。
- 训练裁剪比例：`p_interval=[0.5, 1.0]`。
- 训练随机旋转：启用。

仍未完全对齐的部分：

- 原始配置包含 `warm_up_epoch: 5`；当前 Foundry 训练未显式启用 warmup。
- 官方训练常用 `batch_size=64`；当前使用 `batch_size=16 + grad_accum_steps=4`。有效梯度累积接近 64，但 BatchNorm 统计仍按 micro-batch 16 计算。
- Foundry 预处理实现接近 CTR-GCN / 2s-AGCN feeder 口径，但不是直接复用官方 `feeders/tools.py`。
- 训练 runner、checkpoint、DataLoader 和随机性来自 Foundry 现代化链路，不是官方 `torchlight` 训练栈。

因此，本报告结果应表述为：

> 基于 Foundry 现代化链路的 CTR-GCN NTU60 Cross-Subject 四流复现。官方式验证裁剪 `p_interval=[0.95]` 下，四流融合 Top-1 为 90.2469%，Top-5 为 98.3805%。

不应表述为已经严格复现论文报告值。公开资料中常见的 CTR-GCN NTU60 论文结果为 X-Sub 92.4%、X-View 96.8%，当前结果与 X-Sub 论文值仍有约 2.15 pp 差距。

## 结论

本次实验已经足够支撑两个结论：

1. 当前仓库具备复现 CTR-GCN 多流训练、joint 单流拓扑消融和四流融合评估的完整工程链路。
2. 在 Foundry 现代化链路下，NTU60 xsub 四流融合达到 90.2469% Top-1；消融显示 `A + alpha * Q` 的完整拓扑最好，纯 Q 拓扑下降最大。

如果继续追求论文数值，下一步优先级为：

1. 在 Foundry 中补齐 `warm_up_epoch=5` 并重训。
2. 评估能否在不触发 WSL2 内存问题的前提下提高真实 batch 或修正 BatchNorm 统计口径。
3. 对照官方 feeder 做逐函数一致性检查，必要时直接复用官方 `valid_crop_resize` / `random_rot`。
4. 跑官方原始训练栈作为对照基线，区分模型/数据口径差异和训练框架差异。

## 参考

- CTR-GCN 官方仓库：<https://github.com/Uason-Chen/CTR-GCN>
- CTR-GCN arXiv 页面：<https://arxiv.org/abs/2107.12213>
- 公开结果表中常见的 CTR-GCN NTU60 X-Sub/X-View 结果：92.4% / 96.8%，例如 <https://pmc.ncbi.nlm.nih.gov/articles/PMC10303820/>。
