"""CTR-GCN 的 Foundry 接入入口。

本模块把 Foundry 相关的注册、配置编译和模型构建逻辑集中在模型定义之外，
避免 `model/` 下的网络结构直接耦合训练框架。它会注册 CTR-GCN 主模型和
用于消融实验的 baseline 模型，并提供 NTU RGB+D 与 NW-UCLA 常见协议的
预置运行配置。

Examples:
    注册模型：

    ```python
    from foundry_entry import register_ctrgcn

    register_ctrgcn()
    ```

    编译预置实验配置：

    ```python
    from foundry_entry import (
        CTRGCN_EXPERIMENTS,
        get_ctrgcn_run_config,
        register_ctrgcn,
    )

    register_ctrgcn()
    run_config = get_ctrgcn_run_config("ntu60_xsub")
    ```
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from foundry import register_model
from foundry.core.config import RunConfig
from foundry.core.registry import DatasetSpec
from foundry.projects.skeleton import compile_skeleton_run_config
from foundry.projects.skeleton.compiler import load_skeleton_experiment_config
from foundry.projects.skeleton.graphs import build_graph
from foundry.projects.skeleton.types import SkeletonExperimentConfig

from model.baseline import Model as BaselineModel
from model.ctrgcn import Model as CTRGCNModel

MODEL_NAME = "ctrgcn"
BASELINE_MODEL_NAME = "baseline"

# 预置实验配置 —— 全部从 YAML 文件加载，确保每个字段显式锚定

_EXPERIMENTS_DIR = Path(__file__).resolve().parent / "conf" / "experiments"

_EXPERIMENT_FILES = {
    "ntu60_xsub": "ntu60_xsub_joint.yaml",
    "ntu60_xview": "ntu60_xview_joint.yaml",
    "ntu120_xsub": "ntu120_xsub_joint.yaml",
    "ntu120_xset": "ntu120_xset_joint.yaml",
    "nw_ucla": "nw_ucla_xview_joint.yaml",
    "ntu60_xsub_bone": "ntu60_xsub_bone.yaml",
    "ntu60_xsub_joint_motion": "ntu60_xsub_joint_motion.yaml",
    "ntu60_xsub_bone_motion": "ntu60_xsub_bone_motion.yaml",
    "ntu60_xview_bone": "ntu60_xview_bone.yaml",
    "ntu120_xsub_bone": "ntu120_xsub_bone.yaml",
    "ntu120_xset_bone": "ntu120_xset_bone.yaml",
    "nw_ucla_bone": "nw_ucla_xview_bone.yaml",
    "ntu60_xsub_no_ctr": "ntu60_xsub_no_channel_topology.yaml",
    "ntu60_xsub_no_dynamic": "ntu60_xsub_no_dynamic.yaml",
    "ntu60_xsub_q_only": "ntu60_xsub_q_only.yaml",
}


def _load_experiments() -> dict[str, SkeletonExperimentConfig]:
    """从 YAML 配置文件加载全部预置实验。

    Returns:
        以实验名为键的实验配置字典。每个配置由
        ``load_skeleton_experiment_config`` 解析得到。

    Raises:
        FileNotFoundError: 当任意配置文件缺失时抛出。
    """
    experiments: dict[str, SkeletonExperimentConfig] = {}
    for name, filename in _EXPERIMENT_FILES.items():
        path = _EXPERIMENTS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"实验配置文件不存在: {path}")
        experiments[name] = load_skeleton_experiment_config(str(path))
    return experiments


CTRGCN_EXPERIMENTS: dict[str, SkeletonExperimentConfig] = _load_experiments()


def get_ctrgcn_run_config(name: str) -> RunConfig:
    """把 CTR-GCN 预置实验编译成 Foundry 运行配置。

    Args:
        name: `CTRGCN_EXPERIMENTS` 中的预置实验名，例如
            ``"ntu60_xsub"``。

    Returns:
        已编译的 ``RunConfig``，可直接传给
        ``foundry.runtime.train_once()``。

    Raises:
        KeyError: 当 ``name`` 不是已知预置实验名时抛出。
    """
    experiment = CTRGCN_EXPERIMENTS[name]
    return compile_skeleton_run_config(experiment, resolved_model_name=MODEL_NAME)


def get_baseline_run_config(name: str) -> RunConfig:
    """把 baseline 消融预置实验编译成 Foundry 运行配置。

    Args:
        name: `CTRGCN_EXPERIMENTS` 中的预置实验名，例如
            ``"ntu120_xsub"``。

    Returns:
        使用 baseline 模型名编译后的 ``RunConfig``。数据集、加载器、
        优化器、调度器和运行时设置会沿用同名 CTR-GCN 预置实验。

    Raises:
        KeyError: 当 ``name`` 不是已知预置实验名时抛出。
    """
    experiment = CTRGCN_EXPERIMENTS[name]
    baseline_experiment = replace(
        experiment,
        model=BASELINE_MODEL_NAME,
        output_dir=experiment.output_dir.replace("ctrgcn", BASELINE_MODEL_NAME),
    )
    return compile_skeleton_run_config(baseline_experiment, resolved_model_name=BASELINE_MODEL_NAME)


# 模型构建与注册


def _resolve_common_model_params(model_params: dict[str, Any]) -> dict[str, Any]:
    """提取骨架分类模型共享的构造参数。

    Args:
        model_params: Foundry 传入的模型参数映射。运行时会在调用模型构建器前
            注入 ``_dataset_spec``。

    Returns:
        可同时传给 ``model.ctrgcn.Model`` 和 ``model.baseline.Model`` 的
        关键字参数。

    Raises:
        ValueError: 当 Foundry 未注入合法 ``DatasetSpec`` 时抛出。
    """

    dataset_spec = model_params.get("_dataset_spec")
    if not isinstance(dataset_spec, DatasetSpec):
        raise ValueError("骨架模型构建器需要 `_dataset_spec`。")

    num_class = int(model_params.get("num_classes", dataset_spec.num_classes or 0))
    num_point = int(model_params.get("num_joints", dataset_spec.num_joints or 0))
    num_person = int(model_params.get("num_persons", dataset_spec.num_persons or 2))
    in_channels = int(model_params.get("in_channels", dataset_spec.in_channels or 3))
    dropout = float(model_params.get("dropout", 0.0))
    adaptive = bool(model_params.get("adaptive", True))

    return dict(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        in_channels=in_channels,
        drop_out=dropout,
        adaptive=adaptive,
    )


def _resolve_ctrgcn_params(model_params: dict[str, Any]) -> dict[str, Any]:
    """从 Foundry 模型参数中提取 CTR-GCN 构造参数。

    Args:
        model_params: Foundry 传入的模型参数映射。运行时会在模型构建前注入
            数据集元信息和图结构参数。

    Returns:
        可传给 ``model.ctrgcn.Model`` 的关键字参数，其中包含根据 Foundry
        骨架图 layout 构建出的邻接矩阵。

    Raises:
        ValueError: 当 Foundry 未注入合法 ``DatasetSpec`` 时抛出。
    """

    dataset_spec = model_params.get("_dataset_spec")
    if not isinstance(dataset_spec, DatasetSpec):
        raise ValueError("CTR-GCN 构建器需要 `_dataset_spec`。")

    graph_params: dict[str, Any] = dict(model_params.get("graph", {}))
    layout = graph_params.get("layout", dataset_spec.layout_name or "ntu-rgb+d")
    strategy = graph_params.get("strategy", "spatial")
    max_hop = int(graph_params.get("max_hop", 1))
    dilation = int(graph_params.get("dilation", 1))

    graph = build_graph(
        layout=layout,
        strategy=strategy,
        max_hop=max_hop,
        dilation=dilation,
        normalization=graph_params.get("normalization", "partition-wise"),
    )
    adjacency = graph.adjacency  # (3, V, V) ndarray
    use_channel_topology = bool(model_params.get("use_channel_topology", True))
    use_shared_topology = bool(model_params.get("use_shared_topology", True))

    return {
        **_resolve_common_model_params(model_params),
        "adjacency": adjacency,
        "use_channel_topology": use_channel_topology,
        "use_shared_topology": use_shared_topology,
    }


def build_ctrgcn_model(
    model_params: dict[str, Any],
    device: str | torch.device | None,
) -> nn.Module:
    """为 Foundry 运行时构建 CTR-GCN 模型。

    Args:
        model_params: Foundry 模型参数，包含运行时注入的数据集元信息。
        device: 可选目标设备。传入后使用标准 ``nn.Module.to(...)`` 语义
            迁移模型。

    Returns:
        已配置完成的 CTR-GCN ``nn.Module``。
    """

    kwargs = _resolve_ctrgcn_params(model_params)
    model = CTRGCNModel(**kwargs)
    if device is not None:
        model = model.to(device)
    return model


def build_baseline_model(
    model_params: dict[str, Any],
    device: str | torch.device | None,
) -> nn.Module:
    """为 Foundry 运行时构建 baseline 消融模型。

    Args:
        model_params: Foundry 模型参数，包含运行时注入的数据集元信息。
        device: 可选目标设备。传入后使用标准 ``nn.Module.to(...)`` 语义
            迁移模型。

    Returns:
        已配置完成的 baseline ``nn.Module``。该模型保留原仓库用于消融的
        单位邻接矩阵语义。
    """

    kwargs = _resolve_common_model_params(model_params)
    model = BaselineModel(**kwargs)
    if device is not None:
        model = model.to(device)
    return model


def register_ctrgcn_models() -> None:
    """向 Foundry 注册 CTR-GCN 与 baseline 模型构建器。

    只要 Foundry 的模型注册表允许同名构建器重复注册或覆盖，本函数就可以
    安全地在脚本和 notebook 中多次调用。
    """
    register_model(MODEL_NAME, build_ctrgcn_model)
    register_model(BASELINE_MODEL_NAME, build_baseline_model)


def register_ctrgcn() -> None:
    """注册本项目的全部 CTR-GCN 相关模型。

    这个兼容入口让外部脚本只需调用一个短函数，就能同时注册主模型和消融
    baseline。模块导入时自动执行，无需手动调用。
    """
    register_ctrgcn_models()


# 模块导入即注册，确保 ``--register-module foundry_entry`` 能生效
register_ctrgcn_models()
