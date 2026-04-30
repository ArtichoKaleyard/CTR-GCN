"""CTR-GCN 消融实验使用的 ST-GCN 风格 baseline 模型。

原仓库把该模型作为简单强基线用于对比。本版本保留单位邻接矩阵消融语义，
同时移除旧式 autograd / CUDA 处理，使模型可以由 Foundry 构建，并通过
标准 ``nn.Module.to(...)`` 机制迁移设备。
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def import_class(name: str) -> Any:
    """从点分 Python 路径导入对象。

    Args:
        name: 点分导入路径，例如 ``"graph.ntu_rgb_d.Graph"``。

    Returns:
        导入得到的 Python 对象。

    Raises:
        AttributeError: 当模块后的任一属性组件不存在时抛出。
        ImportError: 当根模块无法导入时抛出。
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    """按分支数量初始化单个图卷积分支。

    Args:
        conv: 待初始化的卷积层。
        branches: 后续会求和的图子集或分支数量。
    """
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv: nn.Conv2d) -> None:
    """使用 Kaiming 正态分布初始化卷积层。

    Args:
        conv: 待初始化的卷积层。
    """
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm1d | nn.BatchNorm2d, scale: float) -> None:
    """初始化 BatchNorm 的仿射参数。

    Args:
        bn: 待初始化的 BatchNorm 层。
        scale: 写入 affine weight 的常数缩放值。
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    """baseline 模型使用的时间卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1) -> None:
        """创建时间卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            kernel_size: 时间维卷积核大小。
            stride: 时间维步幅。
        """
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行时间卷积和 BatchNorm。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T_out, V)`` 的输出张量。
        """
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    """单位邻接矩阵 baseline 使用的图卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, adaptive: bool = True) -> None:
        """创建 baseline 图卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            A: 初始邻接矩阵，形状为 ``(K, V, V)``。
            adaptive: 是否把邻接矩阵作为可学习参数。
        """
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.register_buffer("A", torch.from_numpy(A.astype(np.float32)))

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A: torch.Tensor) -> torch.Tensor:
        """沿源关节点维度对邻接矩阵子集做 L2 归一化。

        Args:
            A: 形状为 ``(K, V, V)`` 的邻接矩阵张量。

        Returns:
            与输入同形状的 L2 归一化邻接矩阵。
        """
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对所有邻接矩阵子集执行图卷积。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T, V)`` 的输出张量。
        """
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    """baseline 网络使用的残差图时序块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
    ) -> None:
        """创建图卷积加时间卷积的组合块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            A: 初始邻接矩阵，形状为 ``(K, V, V)``。
            stride: 时间维下采样步幅。
            residual: 是否启用残差分支。
            adaptive: 图卷积块是否学习邻接矩阵参数。
        """
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行残差图时序块。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T_out, V)`` 的输出张量。
        """
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    """为 CTR-GCN 消融实验保留的 ST-GCN 风格 baseline。"""

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph: str | None = None,
        graph_args: dict[str, Any] | None = None,
        in_channels: int = 3,
        drop_out: float = 0,
        adaptive: bool = True,
        num_set: int = 3,
        adjacency: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """创建 baseline 分类器。

        Args:
            num_class: 动作类别数。
            num_point: 骨架关节点数量。
            num_person: 每个样本最多包含的人数。
            graph: 可选的旧式图类导入路径。若未显式提供 ``adjacency``，该参数
                只用于兼容旧调用和推断关节点数量，不改变 baseline 的单位邻接
                矩阵语义。
            graph_args: 传给旧式图类的可选关键字参数。
            in_channels: 每个关节点的输入通道数。
            drop_out: 分类器前的 dropout 概率。``0`` 表示关闭 dropout。
            adaptive: 图卷积块是否学习邻接矩阵参数。
            num_set: baseline 中单位邻接矩阵子集的数量。
            adjacency: 高级调用方可显式传入的邻接矩阵。Foundry 构建 baseline
                时不会传入该参数，因此默认仍保持原始单位邻接消融语义。
        """
        super(Model, self).__init__()
        graph_args = {} if graph_args is None else graph_args

        if adjacency is not None:
            A = adjacency if isinstance(adjacency, np.ndarray) else adjacency.detach().cpu().numpy()
            num_set = A.shape[0]
            num_point = A.shape[-1]
        else:
            if graph is not None:
                # 保留旧式 graph 导入路径的兼容性，同时继续使用 baseline
                # 原始消融实验中的单位邻接矩阵初始化。
                Graph = import_class(graph)
                graph_instance = Graph(**graph_args)
                self.graph = graph_instance
                num_point = getattr(graph_instance, "num_node", num_point)
            A = np.stack([np.eye(num_point)] * num_set, axis=0)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 baseline 分类前向。

        Args:
            x: 形状为 ``(N, C, T, V, M)`` 的骨架张量。

        Returns:
            形状为 ``(N, num_class)`` 的分类 logits。
        """
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
