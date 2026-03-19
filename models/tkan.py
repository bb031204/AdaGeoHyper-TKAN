"""
TKAN: Temporal Kolmogorov-Arnold Network (PyTorch 适配实现)
=========================================================
此处为 PyTorch 适配实现，基于原始 Keras TKAN 源码重写。

TKAN 的核心思想:
1. 基于 LSTM 的时间递推结构 (保留 input gate, forget gate, cell state)
2. 用 KAN 风格子层替代传统 output gate 的线性计算
3. 每个 KAN 子层维护自己的递归状态，增加表达能力
4. 最终通过聚合所有子层输出来计算 output gate

主要结构对比:
- 标准 LSTM: i, f, c_candidate, o 四个门均为线性变换 + sigmoid/tanh
- TKAN:     i, f, c_candidate 三个门为线性变换 + sigmoid
             o (output gate) 由 KAN 子层聚合生成

模块层次:
- KANLinear: B-spline 基函数非线性映射层
- TKANCell:  TKAN 单步递推单元 (类似 LSTMCell)
- TKANLayer: 封装 TKANCell 的序列处理层 (类似 LSTM)

输入输出:
- TKANLayer 输入:  [batch, seq_len, input_dim]
- TKANLayer 输出:  [batch, seq_len, hidden_dim] (return_sequences=True)
                   或 [batch, hidden_dim]        (return_sequences=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union

from models.kan_linear import KANLinear


class TKANCell(nn.Module):
    """
    TKAN 单步递推单元 (PyTorch 原生实现)

    在每个时间步处理:
    - 输入: x_t [batch, input_dim]
    - 状态: (h_{t-1}, c_{t-1}, sub_state_1, sub_state_2, ...)
    - 输出: h_t [batch, hidden_dim]

    核心机制:
    1. 计算 i(input), f(forget), c_candidate 三个门 (线性+sigmoid)
    2. 更新 cell state: c_t = f * c_{t-1} + i * tanh(c_candidate)
    3. 对每个 KAN 子层:
       - 将输入和子层状态混合后送入 KANLinear
       - 更新子层递归状态
    4. 聚合子层输出, 计算 output gate: o = sigmoid(aggregated)
    5. h_t = o * tanh(c_t)

    Args:
        input_dim:        输入特征维度
        hidden_dim:       隐藏状态维度
        sub_kan_configs:  KAN 子层配置列表
                          - None: 使用默认 KANLinear
                          - int:  指定 spline_order 的 KANLinear
                          - 'linear': 使用普通 Dense 层替代
        sub_kan_output_dim: KAN 子层输出维度 (默认=input_dim)
        sub_kan_input_dim:  KAN 子层输入维度 (默认=input_dim)
        dropout:          输入 dropout 比率
        recurrent_dropout: 递归 dropout 比率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sub_kan_configs: Optional[List] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sub_kan_configs = sub_kan_configs if sub_kan_configs is not None else [None]
        self.sub_kan_output_dim = sub_kan_output_dim if sub_kan_output_dim is not None else input_dim
        self.sub_kan_input_dim = sub_kan_input_dim if sub_kan_input_dim is not None else input_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.num_sub_layers = len(self.sub_kan_configs)

        # ---- LSTM 风格的门控权重 (input, forget, cell_candidate) ----
        # 三个门共享一组矩阵, 然后 split
        self.W_gates = nn.Linear(input_dim, hidden_dim * 3, bias=False)
        self.U_gates = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.bias_gates = nn.Parameter(torch.zeros(hidden_dim * 3))

        # forget gate 偏置初始化为 1 (unit_forget_bias)
        nn.init.ones_(self.bias_gates[hidden_dim: hidden_dim * 2])

        # 输入权重初始化 (glorot_uniform)
        nn.init.xavier_uniform_(self.W_gates.weight)
        # 递归权重初始化 (orthogonal)
        nn.init.orthogonal_(self.U_gates.weight)

        # ---- KAN 子层 ----
        self.tkan_sub_layers = nn.ModuleList()
        for config in self.sub_kan_configs:
            if config is None:
                # 默认 KANLinear
                layer = KANLinear(
                    self.sub_kan_input_dim, self.sub_kan_output_dim,
                    use_layernorm=True
                )
            elif isinstance(config, int):
                # 指定 spline_order
                layer = KANLinear(
                    self.sub_kan_input_dim, self.sub_kan_output_dim,
                    spline_order=config, use_layernorm=True
                )
            elif isinstance(config, str) and config == 'linear':
                # 普通线性层
                layer = nn.Sequential(
                    nn.Linear(self.sub_kan_input_dim, self.sub_kan_output_dim),
                    nn.SiLU()
                )
            else:
                # 默认 KANLinear
                layer = KANLinear(
                    self.sub_kan_input_dim, self.sub_kan_output_dim,
                    use_layernorm=True
                )
            self.tkan_sub_layers.append(layer)

        # ---- 子层输入变换 (将 input 和 sub_state 映射到 sub_kan_input_dim) ----
        # 输入变换: [input_dim] -> [sub_kan_input_dim] (每个子层独立)
        self.sub_input_proj = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, self.sub_kan_input_dim))
            for _ in range(self.num_sub_layers)
        ])
        # 状态变换: [sub_kan_output_dim] -> [sub_kan_input_dim]
        self.sub_state_proj = nn.ParameterList([
            nn.Parameter(torch.empty(self.sub_kan_output_dim, self.sub_kan_input_dim))
            for _ in range(self.num_sub_layers)
        ])
        # 子层递归混合权重: [sub_kan_output_dim * 2]
        self.sub_recurrent_kernel = nn.ParameterList([
            nn.Parameter(torch.empty(self.sub_kan_output_dim * 2))
            for _ in range(self.num_sub_layers)
        ])

        # 初始化子层参数
        for p in self.sub_input_proj:
            nn.init.orthogonal_(p)
        for p in self.sub_state_proj:
            nn.init.orthogonal_(p)
        for p in self.sub_recurrent_kernel:
            # 1D 参数不适合 orthogonal_ (需要 rows >= cols)
            # 使用 uniform 初始化, 与 Keras 中对 [num_layers, dim*2] 做 orthogonal 等效
            nn.init.uniform_(p, -1.0 / math.sqrt(self.sub_kan_output_dim),
                             1.0 / math.sqrt(self.sub_kan_output_dim))

        # ---- 聚合层: 将所有子层输出映射为 output gate ----
        agg_input_dim = self.num_sub_layers * self.sub_kan_output_dim
        self.aggregated_weight = nn.Linear(agg_input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.aggregated_weight.weight)

        # ---- Dropout ----
        self.input_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.recurrent_dropout_layer = nn.Dropout(recurrent_dropout) if recurrent_dropout > 0 else nn.Identity()

    def get_initial_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        获取初始状态。

        Returns:
            states: [h_0, c_0, sub_state_0, sub_state_1, ...]
                    h_0, c_0: [batch, hidden_dim]
                    sub_state_i: [batch, sub_kan_output_dim]
        """
        h_0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        sub_states = [
            torch.zeros(batch_size, self.sub_kan_output_dim, device=device)
            for _ in range(self.num_sub_layers)
        ]
        return [h_0, c_0] + sub_states

    def forward(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        单步前向传播。

        Args:
            x:        [batch, input_dim]   当前时间步输入
            states:   [h, c, sub_s_0, sub_s_1, ...] 上一步状态
            training: 是否训练模式 (影响 dropout)

        Returns:
            h_new:     [batch, hidden_dim]   当前时间步输出
            new_states: [h_new, c_new, new_sub_s_0, ...]
        """
        h_prev = states[0]   # [batch, hidden_dim]
        c_prev = states[1]   # [batch, hidden_dim]
        sub_states = states[2:]  # 每个: [batch, sub_kan_output_dim]

        # ---- Dropout ----
        if training:
            x = self.input_dropout(x)
            h_prev_dp = self.recurrent_dropout_layer(h_prev)
        else:
            h_prev_dp = h_prev

        # ---- 计算三个门 (i, f, c_candidate) ----
        # gates = sigmoid(W_x @ x + W_h @ h + bias)
        gates = self.W_gates(x) + self.U_gates(h_prev_dp) + self.bias_gates
        gates = torch.sigmoid(gates)

        # 分割为三个门
        i_gate, f_gate, c_candidate = gates.chunk(3, dim=-1)
        # i_gate: [batch, hidden_dim]  输入门
        # f_gate: [batch, hidden_dim]  遗忘门
        # c_candidate: [batch, hidden_dim]  候选cell (经过sigmoid)

        # ---- 更新 cell state ----
        # c_new = f * c_prev + i * tanh(c_candidate)
        # 注意: c_candidate 已经过 sigmoid, 再经 tanh 是 TKAN 的设计特点
        c_new = f_gate * c_prev + i_gate * torch.tanh(c_candidate)

        # ---- KAN 子层处理 (用于计算 output gate) ----
        sub_outputs = []
        new_sub_states = []

        for idx in range(self.num_sub_layers):
            sub_layer = self.tkan_sub_layers[idx]
            sub_state = sub_states[idx]

            # 将 input 和 sub_state 映射到 sub_kan_input_dim
            agg_input = x @ self.sub_input_proj[idx] + sub_state @ self.sub_state_proj[idx]
            # agg_input: [batch, sub_kan_input_dim]

            # 通过 KAN 子层
            sub_output = sub_layer(agg_input)
            # sub_output: [batch, sub_kan_output_dim]

            # 更新子层递归状态
            # sub_recurrent_kernel 前半部分用于 sub_output, 后半部分用于 sub_state
            kernel = self.sub_recurrent_kernel[idx]
            k_h, k_x = kernel[:self.sub_kan_output_dim], kernel[self.sub_kan_output_dim:]
            new_sub_state = k_h * sub_output + k_x * sub_state
            # new_sub_state: [batch, sub_kan_output_dim]

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        # ---- 聚合子层输出, 计算 output gate ----
        aggregated = torch.cat(sub_outputs, dim=-1)
        # aggregated: [batch, num_sub_layers * sub_kan_output_dim]

        o_gate = torch.sigmoid(self.aggregated_weight(aggregated))
        # o_gate: [batch, hidden_dim]

        # ---- 计算新的 hidden state ----
        h_new = o_gate * torch.tanh(c_new)
        # h_new: [batch, hidden_dim]

        new_states = [h_new, c_new] + new_sub_states
        return h_new, new_states

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_sub_layers={self.num_sub_layers}, "
            f"sub_kan_output_dim={self.sub_kan_output_dim}"
        )


class TKANLayer(nn.Module):
    """
    TKAN 序列处理层 (封装 TKANCell)

    对输入的整个时间序列进行逐步递推处理。

    Args:
        input_dim:         输入特征维度
        hidden_dim:        隐藏状态维度
        num_layers:        堆叠层数 (默认 1)
        sub_kan_configs:   KAN 子层配置
        sub_kan_output_dim: 子层输出维度
        sub_kan_input_dim:  子层输入维度
        dropout:           输入 dropout
        recurrent_dropout: 递归 dropout
        return_sequences:  是否返回完整序列 (True) 或仅最后一步 (False)

    输入:
        x: [batch, seq_len, input_dim]

    输出:
        return_sequences=True:  [batch, seq_len, hidden_dim]
        return_sequences=False: [batch, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        sub_kan_configs: Optional[List] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        return_sequences: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        # 构建多层 TKANCell
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            cell_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cell = TKANCell(
                input_dim=cell_input_dim,
                hidden_dim=hidden_dim,
                sub_kan_configs=sub_kan_configs,
                sub_kan_output_dim=sub_kan_output_dim if sub_kan_output_dim else cell_input_dim,
                sub_kan_input_dim=sub_kan_input_dim if sub_kan_input_dim else cell_input_dim,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
            self.cells.append(cell)

        # 层间 dropout
        self.layer_dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[List[torch.Tensor]]]]:
        """
        前向传播: 处理完整时间序列。

        Args:
            x:              [batch, seq_len, input_dim]
            initial_states: 可选, 每层的初始状态

        Returns:
            output:      [batch, seq_len, hidden_dim] 或 [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        training = self.training

        # 初始化状态
        if initial_states is None:
            initial_states = [
                cell.get_initial_state(batch_size, device) for cell in self.cells
            ]

        current_input = x  # [batch, seq_len, input_dim]

        all_layer_states = []

        for layer_idx, cell in enumerate(self.cells):
            states = initial_states[layer_idx]
            outputs = []

            # 逐时间步递推
            for t in range(seq_len):
                x_t = current_input[:, t, :]  # [batch, input_dim or hidden_dim]
                h_t, states = cell(x_t, states, training=training)
                outputs.append(h_t)  # h_t: [batch, hidden_dim]

            # 堆叠序列输出
            layer_output = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]
            all_layer_states.append(states)

            # 下一层输入
            if layer_idx < self.num_layers - 1:
                current_input = self.layer_dropout(layer_output)
            else:
                current_input = layer_output

        # 返回结果
        if self.return_sequences:
            return current_input  # [batch, seq_len, hidden_dim]
        else:
            return current_input[:, -1, :]  # [batch, hidden_dim]

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, return_sequences={self.return_sequences}"
        )
