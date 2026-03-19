"""
PredictionHead: 预测头模块
===========================

将融合后的时空表示映射为未来 12 步预测结果。

设计:
- 输入: H_fused [B, T_in, N, D] (融合后的时空表示, T_in=12)
- 输出: Y_pred  [B, T_out, N, C] (未来预测, T_out=12, C=通道数)

实现方式:
- 将输入的 T_in 步编码重塑为 [B, N, T_in*D]
- 通过 MLP 映射到 [B, N, T_out*C]
- 重塑为 [B, T_out, N, C]

这是非自回归方式: 一次性生成完整未来序列, 避免逐步解码的误差累积。
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    多步预测头

    Args:
        input_dim:    输入隐藏维度 D
        output_dim:   输出通道数 C (温度=1, 风速=2, 等)
        input_len:    输入时间步数 (默认 12)
        pred_len:     预测时间步数 (默认 12)
        hidden_dim:   MLP 中间层维度 (可选)
        dropout:      Dropout 比率
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_len: int = 12,
        pred_len: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.pred_len = pred_len

        # 将 T_in * D 映射到 T_out * C
        total_input = input_len * input_dim
        total_output = pred_len * output_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_output),
        )

        # 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H_fused: torch.Tensor) -> torch.Tensor:
        """
        预测头前向传播。

        Args:
            H_fused: [B, T_in, N, D] 融合后的时空表示

        Returns:
            Y_pred: [B, T_out, N, C] 未来预测
        """
        B, T_in, N, D = H_fused.shape

        # 重塑: [B, T_in, N, D] -> [B, N, T_in * D]
        h = H_fused.permute(0, 2, 1, 3).reshape(B, N, T_in * D)
        # h: [B, N, T_in * D]

        # MLP 映射
        out = self.mlp(h)
        # out: [B, N, T_out * C]

        # 重塑: [B, N, T_out * C] -> [B, N, T_out, C] -> [B, T_out, N, C]
        out = out.reshape(B, N, self.pred_len, self.output_dim)
        Y_pred = out.permute(0, 2, 1, 3)
        # Y_pred: [B, T_out, N, C]

        return Y_pred

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"input_len={self.input_len}, pred_len={self.pred_len}"
        )
