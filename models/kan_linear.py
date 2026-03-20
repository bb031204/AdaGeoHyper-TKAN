"""
KANLinear: PyTorch 实现的 KAN 风格线性层
========================================
此处为 PyTorch 适配实现。

基于 Kolmogorov-Arnold Network (KAN) 思想，使用 B-spline 基函数实现可学习的激活函数，
替代传统的固定激活函数线性层。每条边上的激活函数由 B-spline 参数化，
可以学习到更复杂的非线性映射关系。

核心思路：
- 传统线性层: output = activation(W @ x + b)  （固定激活函数）
- KAN 线性层: output = Σ φ_ij(x_j)            （可学习激活函数）
  其中 φ_ij 是参数化的 B-spline 函数

输入: [*, in_features]
输出: [*, out_features]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN 风格线性层 (PyTorch 原生实现)

    Args:
        in_features:  输入特征维度
        out_features: 输出特征维度
        grid_size:    B-spline 网格区间数 (默认 5)
        spline_order: B-spline 阶数 (默认 3, 即三次样条)
        scale_noise:  初始化噪声缩放
        scale_base:   基础权重缩放
        scale_spline: 样条权重缩放
        grid_range:   B-spline 网格值域 (默认 [-1, 1])
        use_layernorm: 是否使用 LayerNorm
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_layernorm = use_layernorm

        # ---- 构建 B-spline 均匀网格 ----
        # 主区间有 grid_size 个子区间, 两侧各扩展 spline_order 个knot
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        # grid shape: [num_knots] = [grid_size + 2*spline_order + 1]
        # 扩展维度以便广播: [1, 1, num_knots]
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0))

        # 基函数数量 = grid_size + spline_order
        num_basis = grid_size + spline_order

        # ---- 基础线性权重 (类似标准线性层) ----
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (scale_base / math.sqrt(in_features))
        )
        self.base_bias = nn.Parameter(torch.zeros(out_features))

        # ---- B-spline 样条权重 ----
        # 每对 (in, out) 有 num_basis 个样条系数
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * (scale_spline / math.sqrt(in_features))
        )

        # ---- 可选 LayerNorm ----
        if use_layernorm:
            self.layernorm = nn.LayerNorm(out_features)

        # ---- 基础激活函数 (SiLU) ----
        self.base_activation = nn.SiLU()

        # 存储配置
        self.scale_noise = scale_noise

    def _compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 B-spline 基函数值。

        Args:
            x: [batch_size, in_features]

        Returns:
            bases: [batch_size, in_features, num_basis]
                   其中 num_basis = grid_size + spline_order
        """
        # x: [B, in] -> [B, in, 1] 用于与 grid 广播
        x = x.unsqueeze(-1)
        grid = self.grid  # [1, 1, num_knots]

        # ---- 0 阶 B-spline: 分段常数 ----
        # bases[b, i, j] = 1 if grid[j] <= x[b, i] < grid[j+1]
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
        # bases shape: [B, in, num_knots - 1]

        # ---- 递推计算高阶 B-spline ----
        for k in range(1, self.spline_order + 1):
            # 左项系数
            left_num = x - grid[:, :, : -(k + 1)]
            left_den = grid[:, :, k:-1] - grid[:, :, : -(k + 1)]

            # 右项系数
            right_num = grid[:, :, (k + 1) :] - x
            right_den = grid[:, :, (k + 1) :] - grid[:, :, 1:(-k) if (-k) != 0 else None]

            # 避免除零
            left = left_num / (left_den + 1e-8) * bases[:, :, :-1]
            right = right_num / (right_den + 1e-8) * bases[:, :, 1:]

            bases = left + right

        # 最终 bases shape: [B, in, grid_size + spline_order]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [*, in_features] 任意前导维度

        Returns:
            output: [*, out_features]
        """
        # 保存原始形状，展平为 2D
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # [B_flat, in_features]

        # ---- 基础线性部分 ----
        # output_base = base_activation(x) @ base_weight^T + base_bias
        base_output = F.linear(self.base_activation(x_flat), self.base_weight, self.base_bias)
        # base_output: [B_flat, out_features]

        # ---- B-spline 样条部分 ----
        spline_basis = self._compute_bspline_basis(x_flat)
        # spline_basis: [B_flat, in_features, num_basis]
        # spline_weight: [out_features, in_features, num_basis]
        # 展平为矩阵乘法 (比 einsum 更高效)
        spline_output = spline_basis.reshape(x_flat.shape[0], -1) @ self.spline_weight.reshape(self.out_features, -1).t()
        # spline_output: [B_flat, out_features]

        # ---- 合并输出 ----
        output = base_output + spline_output

        # ---- 可选 LayerNorm ----
        if self.use_layernorm:
            output = self.layernorm(output)

        # 恢复原始形状
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"use_layernorm={self.use_layernorm}"
        )
