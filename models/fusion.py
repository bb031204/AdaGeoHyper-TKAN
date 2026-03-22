"""
GatedFusion: 门控融合模块
=========================

用于融合空间特征 H_s 与时间特征 H_t。

融合机制:
    z = sigmoid(W1 * H_s + W2 * H_t + b)
    H_fused = z * H_s + (1 - z) * H_t

门控信号 z 自适应地决定每个位置上空间信息与时间信息的混合比例。

输入:
    H_s: [B, T, N, D]  空间特征
    H_t: [B, T, N, D]  时间特征

输出:
    H_fused: [B, T, N, D]  融合特征
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    门控融合层

    Args:
        feature_dim: 特征维度 D (H_s 和 H_t 的最后一个维度)
        dropout:     Dropout 比率
    """

    def __init__(self, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim

        # 门控权重: W1 用于空间特征, W2 用于时间特征
        self.W_s = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_t = nn.Linear(feature_dim, feature_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(feature_dim))

        # 初始化
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.W_t.weight)

        # 可选 LayerNorm
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_s: torch.Tensor, H_t: torch.Tensor) -> torch.Tensor:
        """
        门控融合前向传播。

        Args:
            H_s: [B, T, N, D] 或 [B, N, D]  空间特征
            H_t: [B, T, N, D] 或 [B, N, D]  时间特征

        Returns:
            H_fused: 与输入相同 shape 的融合特征
        """
        # 计算门控信号
        z = torch.sigmoid(self.W_s(H_s) + self.W_t(H_t) + self.bias)
        # z: 与输入相同 shape, 值域 [0, 1]

        # 门控融合
        H_fused = z * H_s + (1 - z) * H_t
        # H_fused: 与输入相同 shape

        # LayerNorm + Dropout
        H_fused = self.layer_norm(H_fused)
        H_fused = self.dropout(H_fused)

        return H_fused

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}"
