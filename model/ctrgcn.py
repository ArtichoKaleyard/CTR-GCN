"""用于骨架动作识别的 CTR-GCN 模型定义。

本实现保留 ICCV 2021 CTR-GCN 的原始网络结构，同时移除旧式 PyTorch 设备
处理逻辑。模型仍支持原仓库的 graph 导入路径构造方式，也支持 Foundry 通过
``adjacency`` 参数注入预构建邻接矩阵。
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


def weights_init(m: nn.Module) -> None:
    """递归初始化模块中的卷积和 BatchNorm 层。

    Conv 层使用 Kaiming 正态初始化，BatchNorm 层使用正态分布
    初始化 weight 并将 bias 置零。

    Args:
        m: 待初始化的模块（通常通过 ``nn.Module.apply`` 递归传入）。
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    """单分支时间卷积块（Conv2d + BatchNorm2d）。

    卷积只在时间维度上滑动，空间维固定为 1。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        """创建时间卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            kernel_size: 时间维卷积核大小。
            stride: 时间维步幅。
            dilation: 时间维膨胀系数。
        """
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行时间卷积。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T_out, V)`` 的输出张量。
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    """多尺度时间卷积块。

    包含多个不同膨胀率的时间卷积分支、一个 MaxPool 分支和
    一个 1x1 分支，所有分支输出沿通道维拼接后与残差相加。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] = 3,
        stride: int = 1,
        dilations: list[int] = [1, 2, 3, 4],
        residual: bool = True,
        residual_kernel_size: int = 1,
    ) -> None:
        """创建多尺度时间卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数，必须能被分支总数整除。
            kernel_size: 时间维卷积核大小，可为单个整数或与 dilations 等长
                的列表。
            stride: 时间维步幅。
            dilations: 各膨胀卷积分支的膨胀系数列表。
            residual: 是否启用残差连接。
            residual_kernel_size: 残差路径上的时间卷积核大小。
        """

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行多尺度时间卷积。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T_out, V)`` 的输出张量。
        """
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    """通道级拓扑优化图卷积。

    为不同输出通道建模不同的关节关系拓扑，对输入特征计算通道间相似度
    作为动态拓扑，与静态邻接矩阵融合后执行图卷积，是 CTR-GCN 的核心组件。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rel_reduction: int = 8,
        mid_reduction: int = 1,
        use_channel_topology: bool = True,
        use_shared_topology: bool = True,
    ) -> None:
        """创建通道级拓扑优化图卷积层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            rel_reduction: 关系通道数的除数因子。
            mid_reduction: 中间通道除数因子（保留参数，当前未使用）。
            use_channel_topology: 启用通道级动态拓扑。关闭后退化为标准 GCN。
            use_shared_topology: 启用共享拓扑 A。关闭后仅使用 α·Q（纯 Q 消融）。
        """
        super(CTRGC, self).__init__()
        self.use_channel_topology = use_channel_topology
        self.use_shared_topology = use_shared_topology
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        if self.use_channel_topology:
            if in_channels == 3 or in_channels == 9:
                self.rel_channels = 8
            else:
                self.rel_channels = in_channels // rel_reduction
            self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
            self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
            self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
            self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x: torch.Tensor, A: torch.Tensor | None = None, alpha: torch.Tensor | float = 1) -> torch.Tensor:
        x3 = self.conv3(x)
        if self.use_channel_topology:
            x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
            x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
            x1 = self.conv4(x1) * alpha
            if self.use_shared_topology:
                x1 = x1 + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        else:
            x1 = A.unsqueeze(0).unsqueeze(0) if A is not None else 0
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    """CTR-GCN 使用的时间卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1) -> None:
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
    """CTR-GCN 使用的图卷积块。

    内部由多个 CTRGC 子模块组成，每个子模块对应邻接矩阵的一个子集。
    支持自适应拓扑学习和残差连接。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        coff_embedding: int = 4,
        adaptive: bool = True,
        residual: bool = True,
        use_channel_topology: bool = True,
        use_shared_topology: bool = True,
    ) -> None:
        """创建 CTR-GCN 图卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            A: 初始邻接矩阵，形状为 ``(K, V, V)``。
            coff_embedding: 中间通道压缩系数。
            adaptive: 是否把邻接矩阵作为可学习参数。
            residual: 是否启用残差连接。
            use_channel_topology: 是否启用通道级动态拓扑。
            use_shared_topology: 是否启用共享拓扑 A。
        """
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels, use_channel_topology=use_channel_topology, use_shared_topology=use_shared_topology))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.register_buffer("A", torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对所有邻接矩阵子集执行 CTR-GC 图卷积。

        Args:
            x: 形状为 ``(N, C, T, V)`` 的输入张量。

        Returns:
            形状为 ``(N, out_channels, T, V)`` 的输出张量。
        """
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    """CTR-GCN 网络使用的残差图时序块。

    由图卷积（unit_gcn）和多尺度时间卷积（MultiScale_TemporalConv）
    组合而成的残差块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        kernel_size: int = 5,
        dilations: list[int] = [1, 2],
        use_channel_topology: bool = True,
        use_shared_topology: bool = True,
    ) -> None:
        """创建残差图时序块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            A: 初始邻接矩阵，形状为 ``(K, V, V)``。
            stride: 时间维下采样步幅。
            residual: 是否启用残差分支。
            adaptive: 图卷积块是否学习邻接矩阵参数。
            kernel_size: 多尺度时间卷积的卷积核大小。
            dilations: 多尺度时间卷积分支的膨胀系数列表。
            use_channel_topology: 是否启用通道级动态拓扑。
            use_shared_topology: 是否启用共享拓扑 A。
        """
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, use_channel_topology=use_channel_topology, use_shared_topology=use_shared_topology)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
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
    """Channel-wise Topology Refinement GCN 分类器。

    构造器同时支持原仓库 API（``graph="graph.ntu_rgb_d.Graph"``）和
    Foundry 接入路径（``adjacency=<ndarray>``）。调用时必须提供 ``graph``
    或 ``adjacency`` 之一。
    """

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
        adjacency: np.ndarray | torch.Tensor | None = None,
        use_channel_topology: bool = True,
        use_shared_topology: bool = True,
    ) -> None:
        """创建 CTR-GCN 分类器。

        Args:
            num_class: 动作类别数。
            num_point: 骨架关节点数量。
            num_person: 每个样本最多包含的人数。
            graph: 可选的旧式图类导入路径。
            graph_args: 传给旧式图类的可选关键字参数。
            in_channels: 每个关节点的输入通道数。
            drop_out: 分类器前的 dropout 概率。``0`` 表示关闭 dropout。
            adaptive: 图卷积块是否学习邻接矩阵参数。
            adjacency: 可选的显式邻接矩阵，形状为 ``(K, V, V)``。
            use_channel_topology: 是否启用通道级动态拓扑。关闭后 CTRGC
                退化为标准 GCN（保留骨架图先验和残差结构）。
            use_shared_topology: 是否启用共享拓扑 A。关闭后仅使用 α·Q
                （纯 Q 消融）。

        Raises:
            ValueError: 当 ``graph`` 和 ``adjacency`` 都未提供时抛出。
        """
        super(Model, self).__init__()
        graph_args = {} if graph_args is None else graph_args

        if adjacency is not None:
            A = adjacency if isinstance(adjacency, np.ndarray) else adjacency.detach().cpu().numpy()
        elif graph is not None:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            A = self.graph.A
        else:
            raise ValueError("Must provide either `graph` import string or `adjacency` matrix.")

        if A.ndim != 3:
            raise ValueError(f"adjacency must be 3-dimensional (K, V, V), got shape {A.shape}")
        if A.shape[-1] != num_point:
            raise ValueError(f"adjacency last dim {A.shape[-1]} != num_point {num_point}")

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        kws = dict(adaptive=adaptive, use_channel_topology=use_channel_topology, use_shared_topology=use_shared_topology)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, **kws)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, **kws)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, **kws)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, **kws)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, **kws)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, **kws)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, **kws)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, **kws)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, **kws)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, **kws)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 CTR-GCN 分类前向。

        Args:
            x: 形状为 ``(N, C, T, V, M)`` 的骨架张量。为了兼容旧预处理路径，
                也接受展平的 ``(N, T, V*C)`` 张量。

        Returns:
            形状为 ``(N, num_class)`` 的分类 logits。
        """
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
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
