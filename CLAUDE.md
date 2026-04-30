# CTR-GCN

基于骨架的动作识别模型，ICCV 2021。使用 Channel-wise Topology Refinement Graph Convolution。

## 技术栈

- **Python**: 3.12（uv 管理）
- **框架**: PyTorch 2.11 + CUDA 12.8
- **训练管线**: [Foundry](https://github.com/ArtichoKaleyard/Foundry) v0.3.7
- **配置**: Foundry Python API（`get_ctrgcn_run_config()` / `get_baseline_run_config()`）

## 目录结构

```
CTR-GCN/
  model/ctrgcn.py           # CTR-GCN 核心模型（纯 nn.Module）
  model/baseline.py         # 现代化 ST-GCN 基线（消融用）
  foundry_entry.py          # Foundry 模型注册 + 预置实验配置
  pyproject.toml            # 依赖与项目元数据
  feeders/                  # 原始数据加载器（归档参考）
  graph/                    # 原始骨架图定义（归档参考）
  ensemble.py               # 多模态集成
```

## 约束

- 所有依赖通过 `uv add`/`uv remove` 管理，不用 pip
- Foundry 作为 git 依赖引入，不直接编辑 Foundry 源码
- 模型层保持纯 `nn.Module`，不引入框架耦合
- 注册在 `foundry_entry.py` 中完成，调用 `register_ctrgcn()` 即可接入 CTR-GCN 与 baseline

## 常用命令

```bash
uv sync                          # 同步依赖
uv run python -c "from foundry_entry import register_ctrgcn; register_ctrgcn()"
```

## 训练

通过 Foundry 的 `train_once()` 直接运行：

```python
from foundry_entry import register_ctrgcn, get_ctrgcn_run_config
from foundry.runtime.runner import train_once

register_ctrgcn()
run_config = get_ctrgcn_run_config("ntu60_xsub")  # 或 ntu60_xview / ntu120_xsub / ntu120_xset / nw_ucla
train_once(run_config)
```

baseline 消融使用同一套数据 preset：

```python
from foundry_entry import register_ctrgcn, get_baseline_run_config
from foundry.runtime.runner import train_once

register_ctrgcn()
run_config = get_baseline_run_config("ntu120_xsub")
train_once(run_config)
```

预置实验参数对齐原论文：SGD + nesterov momentum 0.9，初始 lr 0.1，step decay at epoch 35/55，weight decay 0.0004（UCLA 0.0001），65 epochs。
