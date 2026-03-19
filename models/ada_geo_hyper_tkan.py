"""
AdaGeoHyperTKAN: 完整主模型
============================

整合空间模块、时间模块、融合模块与预测头的完整模型。

主干流程 (严格方案 A):
    1. Spatial Module (超图)  →  H_s    [B, T, N, D]
    2. H_s → TKAN             →  H_t    [B, T, N, D]
    3. GatedFusion(H_s, H_t)  →  H_fused [B, T, N, D]
    4. Prediction Head         →  Y_pred  [B, T_out, N, C]

模块职责:
- AdaptiveGeoHypergraph: 空间关系建模 (不负责预测)
- TKANLayer:            时间演化建模 + 非自回归多步预测特征生成
- GatedFusion:          时空特征门控融合
- PredictionHead:       将融合表示映射为未来 12 步输出

输入: [B, T=12, N, F]  (B=batch, T=时间步, N=站点数, F=特征通道)
输出: [B, T=12, N, C]  (C=预测通道数)
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

from models.hypergraph import AdaptiveGeoHypergraph
from models.tkan import TKANLayer
from models.fusion import GatedFusion
from models.prediction_head import PredictionHead

logger = logging.getLogger(__name__)


class AdaGeoHyperTKAN(nn.Module):
    """
    AdaGeoHyper-TKAN 完整模型

    Args:
        input_dim:          输入特征维度 F (温度=1, 风速=2, 等)
        output_dim:         输出通道数 C
        hidden_dim:         空间模块隐藏维度
        tkan_hidden_dim:    TKAN 隐藏维度
        tkan_layers:        TKAN 层数
        tkan_sub_kan_configs: TKAN KAN子层配置
        input_len:          输入时间步 (默认 12)
        pred_len:           预测时间步 (默认 12)
        position_dim:       站点位置维度 (2=lon/lat, 3=lon/lat/alt)
        k_neighbors:        K近邻数
        lambda_geo:         平面距离权重
        lambda_alt:         海拔差权重
        summary_pool:       状态摘要方式
        scorer_hidden_dim:  打分函数隐藏维度
        hypergraph_layers:  超图卷积层数
        fusion_dim:         融合层维度 (需等于 hidden_dim)
        dropout:            Dropout 比率
        pred_head_hidden:   预测头 MLP 隐藏维度
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 64,
        tkan_hidden_dim: int = 64,
        tkan_layers: int = 2,
        tkan_sub_kan_configs: Optional[list] = None,
        input_len: int = 12,
        pred_len: int = 12,
        position_dim: int = 2,
        k_neighbors: int = 8,
        lambda_geo: float = 1.0,
        lambda_alt: float = 0.5,
        summary_pool: str = "mean",
        scorer_hidden_dim: int = 32,
        hypergraph_layers: int = 2,
        fusion_dim: int = 64,
        dropout: float = 0.1,
        pred_head_hidden: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.tkan_hidden_dim = tkan_hidden_dim
        self.input_len = input_len
        self.pred_len = pred_len

        # 确保 fusion_dim 一致
        assert hidden_dim == tkan_hidden_dim, (
            f"hidden_dim ({hidden_dim}) 必须等于 tkan_hidden_dim ({tkan_hidden_dim}) "
            f"以便 GatedFusion 可以对齐 H_s 和 H_t"
        )
        assert hidden_dim == fusion_dim, (
            f"hidden_dim ({hidden_dim}) 必须等于 fusion_dim ({fusion_dim})"
        )

        # ===============================
        # 1. 空间模块: 自适应地理邻域超图
        # ===============================
        self.spatial_module = AdaptiveGeoHypergraph(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            position_dim=position_dim,
            k_neighbors=k_neighbors,
            lambda_geo=lambda_geo,
            lambda_alt=lambda_alt,
            summary_pool=summary_pool,
            scorer_hidden_dim=scorer_hidden_dim,
            num_layers=hypergraph_layers,
            dropout=dropout,
        )

        # ===============================
        # 2. 时间模块: TKAN
        # ===============================
        if tkan_sub_kan_configs is None:
            tkan_sub_kan_configs = [None, 3]

        self.temporal_module = TKANLayer(
            input_dim=hidden_dim,           # TKAN 输入 = 空间模块输出 H_s 的维度
            hidden_dim=tkan_hidden_dim,
            num_layers=tkan_layers,
            sub_kan_configs=tkan_sub_kan_configs,
            sub_kan_output_dim=hidden_dim,
            sub_kan_input_dim=hidden_dim,
            dropout=dropout,
            recurrent_dropout=dropout * 0.5,
            return_sequences=True,          # 返回完整序列, 用于融合
        )

        # ===============================
        # 3. 门控融合模块
        # ===============================
        self.gated_fusion = GatedFusion(
            feature_dim=fusion_dim,
            dropout=dropout,
        )

        # ===============================
        # 4. 预测头
        # ===============================
        self.prediction_head = PredictionHead(
            input_dim=fusion_dim,
            output_dim=output_dim,
            input_len=input_len,
            pred_len=pred_len,
            hidden_dim=pred_head_hidden,
            dropout=dropout,
        )

        logger.info(f"[模型] AdaGeoHyperTKAN 初始化完成:")
        logger.info(f"  输入: [B, {input_len}, N, {input_dim}]")
        logger.info(f"  输出: [B, {pred_len}, N, {output_dim}]")
        logger.info(f"  hidden_dim={hidden_dim}, tkan_hidden={tkan_hidden_dim}")
        logger.info(f"  tkan_layers={tkan_layers}, hypergraph_layers={hypergraph_layers}")
        logger.info(f"  k_neighbors={k_neighbors}")

    def build_graph(self, **kwargs):
        """代理超图构建方法。"""
        self.spatial_module.build_graph(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整前向传播。

        数据流:
            X [B, T, N, F]
              ↓ Spatial Module (超图卷积)
            H_s [B, T, N, D]
              ↓ TKAN (时间建模, per-station)
            H_t [B, T, N, D]
              ↓ GatedFusion(H_s, H_t)
            H_fused [B, T, N, D]
              ↓ Prediction Head
            Y_pred [B, T_out, N, C]

        Args:
            x: [B, T=12, N, F] 输入时空序列

        Returns:
            Y_pred: [B, T_out=12, N, C] 预测结果
        """
        B, T, N, F_in = x.shape

        # ---- 1. 空间模块 ----
        # X → H_s: 通过超图卷积增强空间关系
        H_s = self.spatial_module(x)
        # H_s: [B, T, N, D]

        # ---- 2. TKAN 时间模块 ----
        # 将 H_s 按站点维度展开, 送入 TKAN
        # H_s: [B, T, N, D] -> [B*N, T, D]
        H_s_flat = H_s.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)

        # TKAN 处理时序
        H_t_flat = self.temporal_module(H_s_flat)
        # H_t_flat: [B*N, T, tkan_hidden_dim]

        # 恢复形状: [B*N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        H_t = H_t_flat.reshape(B, N, T, self.tkan_hidden_dim).permute(0, 2, 1, 3)
        # H_t: [B, T, N, D]

        # ---- 3. 门控融合 ----
        H_fused = self.gated_fusion(H_s, H_t)
        # H_fused: [B, T, N, D]

        # ---- 4. 预测头 ----
        Y_pred = self.prediction_head(H_fused)
        # Y_pred: [B, T_out, N, C]

        return Y_pred

    def get_model_info(self) -> dict:
        """获取模型信息摘要。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "AdaGeoHyperTKAN",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": f"[B, {self.input_len}, N, {self.input_dim}]",
            "output_shape": f"[B, {self.pred_len}, N, {self.output_dim}]",
            "hidden_dim": self.hidden_dim,
            "tkan_hidden_dim": self.tkan_hidden_dim,
        }
