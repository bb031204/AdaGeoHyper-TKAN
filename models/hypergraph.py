"""
AdaptiveGeoHypergraph: 自适应地理邻域超图空间模块
================================================

基于经度、纬度、海拔构建自适应邻域超图，用于刻画多站点之间
具有物理意义的高阶空间依赖关系。

设计思想:
1. 节点定义: 每个站点 v_i, 静态属性 p_i = [lon_i, lat_i, alt_i]
2. 距离定义: d_ij = sqrt(λ_g * d_geo(i,j)^2 + λ_h * (alt_i - alt_j)^2)
   - d_geo: 球面距离 (Haversine)
   - λ_g, λ_h: 距离权重
3. 超边构造: e_i = {v_i} ∪ N_i^geo (K近邻)
4. 自适应权重:
   - 状态摘要: s_i = Pool(X_i)
   - 打分函数: u_ij = φ([p_i, p_j, s_i, s_j])
   - 归一化:   α_ij = softmax(u_ij) 在邻域内
5. 空间传播: z_i = Σ_{j∈e_i} α_ij * x_j * W

关键特性:
- 地理邻域骨架固定、邻域贡献强度动态变化
- 支持构图缓存
- 与 TKAN 时间模块解耦

输入: [B, T, N, F]
输出: [B, T, N, D] (空间增强特征 H_s)
"""

import os
import math
import json
import hashlib
import logging
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def haversine_distance_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    计算所有站点之间的 Haversine 球面距离矩阵。

    Args:
        lon: [N] 经度 (度)
        lat: [N] 纬度 (度)

    Returns:
        dist: [N, N] 球面距离 (km)
    """
    R = 6371.0  # 地球半径 (km)
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # 广播计算
    dlon = lon_rad[:, None] - lon_rad[None, :]
    dlat = lat_rad[:, None] - lat_rad[None, :]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)  # 数值稳定
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_geographic_distance(
    positions: np.ndarray,
    lambda_geo: float = 1.0,
    lambda_alt: float = 0.5,
) -> np.ndarray:
    """
    计算站点之间的三维地理距离。

    d_ij = sqrt(λ_g * d_geo(i,j)^2 + λ_h * (alt_i - alt_j)^2)

    Args:
        positions: [N, 2] 或 [N, 3] (lon, lat) 或 (lon, lat, alt)
        lambda_geo: 平面距离权重
        lambda_alt: 海拔差权重

    Returns:
        distances: [N, N] 距离矩阵
    """
    lon = positions[:, 0]
    lat = positions[:, 1]

    # 球面距离 (km)
    geo_dist = haversine_distance_matrix(lon, lat)

    if positions.shape[1] >= 3:
        alt = positions[:, 2]
        alt_diff = alt[:, None] - alt[None, :]
        # 海拔差转为 km
        alt_diff_km = alt_diff / 1000.0
        combined = np.sqrt(lambda_geo * geo_dist ** 2 + lambda_alt * alt_diff_km ** 2)
    else:
        combined = np.sqrt(lambda_geo * geo_dist ** 2)

    return combined


def build_knn_hypergraph(
    distances: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于距离矩阵构建 K近邻超图结构。

    每个站点生成一条超边, 连接自身及其 K 个最近邻。

    Args:
        distances: [N, N] 距离矩阵
        k:         近邻数

    Returns:
        neighbor_indices: [N, K+1] 每个节点的超边成员 (含自身)
        neighbor_dists:   [N, K+1] 对应距离
    """
    N = distances.shape[0]
    k = min(k, N - 1)  # K 不超过 N-1

    # 对每个节点找 K+1 近邻 (包含自身, 距离为0)
    sorted_indices = np.argsort(distances, axis=-1)[:, : k + 1]
    # 对应距离
    sorted_dists = np.take_along_axis(distances, sorted_indices, axis=-1)

    return sorted_indices, sorted_dists


def get_position_dim_tag(position_dim: int) -> str:
    """根据位置维度返回可读标签。"""
    if position_dim >= 3:
        return "lonlat_alt"
    return "lonlat"


def get_cache_key(
    dataset_name: str,
    num_stations: int,
    k_neighbors: int,
    lambda_geo: float,
    lambda_alt: float,
    position_dim: int = 2,
    station_indices: Optional[np.ndarray] = None,
) -> Tuple[str, str]:
    """
    生成超图缓存键值。

    Returns:
        (cache_hash, dim_tag): 哈希值和维度标签
    """
    key_data = {
        "dataset": dataset_name,
        "num_stations": num_stations,
        "k_neighbors": k_neighbors,
        "lambda_geo": lambda_geo,
        "lambda_alt": lambda_alt,
        "position_dim": position_dim,
    }
    if station_indices is not None:
        key_data["station_hash"] = hashlib.md5(station_indices.tobytes()).hexdigest()
    key_str = json.dumps(key_data, sort_keys=True)
    cache_hash = hashlib.md5(key_str.encode()).hexdigest()
    dim_tag = get_position_dim_tag(position_dim)
    return cache_hash, dim_tag


class AdaptiveScorer(nn.Module):
    """
    自适应局部打分函数 φ

    对于中心站点 v_i 及其邻居 v_j:
    u_ij = φ([p_i, p_j, s_i, s_j])

    使用两层 MLP 实现。

    Args:
        position_dim: 位置特征维度 (2 或 3)
        summary_dim:  状态摘要维度
        hidden_dim:   MLP 隐藏维度
    """

    def __init__(self, position_dim: int, summary_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = position_dim * 2 + summary_dim * 2  # [p_i, p_j, s_i, s_j]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        p_center: torch.Tensor,
        p_neighbor: torch.Tensor,
        s_center: torch.Tensor,
        s_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p_center:   [B, N, K+1, P] 中心节点位置 (广播)
            p_neighbor: [B, N, K+1, P] 邻居节点位置
            s_center:   [B, N, K+1, D_s] 中心节点状态摘要 (广播)
            s_neighbor: [B, N, K+1, D_s] 邻居节点状态摘要

        Returns:
            scores: [B, N, K+1] 打分
        """
        combined = torch.cat([p_center, p_neighbor, s_center, s_neighbor], dim=-1)
        # combined: [B, N, K+1, input_dim]
        scores = self.mlp(combined).squeeze(-1)
        # scores: [B, N, K+1]
        return scores


class AdaptiveGeoHypergraph(nn.Module):
    """
    自适应地理邻域超图空间模块

    只负责空间关系建模，不负责最终预测。

    流程:
    1. 构建静态 K-NN 超图骨架 (基于地理距离, 可缓存)
    2. 计算站点状态摘要 s_i = Pool(X_i)
    3. 动态计算邻域权重 α_ij
    4. 加权超图卷积: z_i = Σ α_ij * x_j @ W

    Args:
        input_dim:    输入特征维度
        hidden_dim:   输出隐藏维度
        position_dim: 位置维度 (2=lon/lat, 3=lon/lat/alt)
        k_neighbors:  K近邻数
        lambda_geo:   平面距离权重
        lambda_alt:   海拔差权重
        summary_pool: 摘要方式 ('mean', 'last', 'linear')
        scorer_hidden_dim: 打分MLP隐藏维度
        num_layers:   超图卷积层数
        dropout:      Dropout比率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        position_dim: int = 2,
        k_neighbors: int = 8,
        lambda_geo: float = 1.0,
        lambda_alt: float = 0.5,
        summary_pool: str = "mean",
        scorer_hidden_dim: int = 32,
        degree_clamp_min: float = 1e-6,
        float32_norm: bool = False,
        num_layers: int = 2,
        dropout: float = 0.1,
        dynamic_pruning: bool = False,
        pruning_mode: str = "top_p",
        pruning_top_p: float = 0.8,
        pruning_threshold: float = 0.05,
        pruning_min_keep: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.position_dim = position_dim
        self.k_neighbors = k_neighbors
        self.lambda_geo = lambda_geo
        self.lambda_alt = lambda_alt
        self.degree_clamp_min = degree_clamp_min
        self.float32_norm = float32_norm
        self.summary_pool = summary_pool
        self.num_layers = num_layers

        # ---- 动态稀疏化配置 (方案3) ----
        self.dynamic_pruning = dynamic_pruning
        self.pruning_mode = pruning_mode
        self.pruning_top_p = float(pruning_top_p)
        self.pruning_threshold = float(pruning_threshold)
        self.pruning_min_keep = int(max(1, pruning_min_keep))
        self._last_pruning_stats = None

        # ---- 状态摘要 ----
        summary_dim = hidden_dim
        if summary_pool == "linear":
            self.summary_proj = nn.Sequential(
                nn.Linear(input_dim, summary_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.summary_proj = nn.Linear(input_dim, summary_dim)

        # ---- 自适应打分函数 ----
        self.scorer = AdaptiveScorer(
            position_dim=position_dim,
            summary_dim=summary_dim,
            hidden_dim=scorer_hidden_dim,
        )

        # ---- 超图卷积层 ----
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.conv_layers.append(nn.Linear(in_d, hidden_dim))

        # ---- 层归一化与激活 ----
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        # ---- 超图结构缓冲区 ----
        # 在 build_graph 时注册
        self.register_buffer("neighbor_indices", None)
        self.register_buffer("positions_tensor", None)

        self._graph_built = False

    def build_graph(
        self,
        positions: np.ndarray,
        cache_dir: Optional[str] = None,
        dataset_name: str = "",
        use_cache: bool = True,
        station_indices: Optional[np.ndarray] = None,
    ):
        """
        构建超图结构 (可缓存)。

        Args:
            positions:       [N, 2] 或 [N, 3] 站点位置 (lon, lat[, alt])
            cache_dir:       缓存目录
            dataset_name:    数据集名称
            use_cache:       是否使用缓存
            station_indices: 站点采样索引
        """
        N = positions.shape[0]
        actual_pos_dim = positions.shape[1]
        k = min(self.k_neighbors, N - 1)
        dim_tag = get_position_dim_tag(actual_pos_dim)

        # ---- 维度自检 ----
        has_alt = actual_pos_dim >= 3
        dim_desc_parts = ["经度(lon)", "纬度(lat)"]
        if has_alt:
            dim_desc_parts.append("海拔(alt)")
        dim_desc = " + ".join(dim_desc_parts)

        logger.info(f"[超图] ========== 超图构建信息 ==========")
        logger.info(f"[超图] 位置维度: {actual_pos_dim}D ({dim_tag}) -> {dim_desc}")
        logger.info(f"[超图] 站点数: {N}, K近邻: {k}")
        logger.info(f"[超图] λ_geo={self.lambda_geo}, λ_alt={self.lambda_alt}"
                    f"{'' if has_alt else ' (未生效: 无海拔数据)'}")

        if actual_pos_dim != self.position_dim:
            logger.warning(
                f"[超图] 维度不匹配! 模型 position_dim={self.position_dim}, "
                f"实际数据维度={actual_pos_dim}. 将以实际数据维度为准."
            )

        cache_hit = False
        cache_path = None

        # ---- 尝试加载缓存 ----
        if use_cache and cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_hash, _ = get_cache_key(
                dataset_name, N, k, self.lambda_geo, self.lambda_alt,
                actual_pos_dim, station_indices,
            )
            cache_filename = f"hypergraph_{dim_tag}_{cache_hash}.pkl"
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                neighbor_indices = cached["neighbor_indices"]
                cached_dim = cached.get("position_dim", "unknown")
                cached_dim_tag = cached.get("dim_tag", "unknown")
                cache_hit = True
                logger.info(f"[超图] 命中缓存:")
                logger.info(f"[超图]   路径: {cache_path}")
                logger.info(f"[超图]   文件: {cache_filename}")
                logger.info(f"[超图]   构建维度: {cached_dim}D ({cached_dim_tag})")
                logger.info(f"[超图]   站点数={N}, K={k}")

        # ---- 构建超图 ----
        if not cache_hit:
            logger.info(f"[超图] 未命中缓存, 开始构图 ({dim_tag})...")
            t_start = time.time()

            distances = compute_geographic_distance(positions, self.lambda_geo, self.lambda_alt)
            neighbor_indices, neighbor_dists = build_knn_hypergraph(distances, k)

            t_elapsed = time.time() - t_start
            logger.info(
                f"[超图] 构图完成: 站点数={N}, K={k}, "
                f"超边数={N}, 耗时={t_elapsed:.2f}s"
            )
            logger.info(
                f"[超图] 平均邻居距离: {neighbor_dists[:, 1:].mean():.2f}km, "
                f"最大距离: {neighbor_dists[:, -1].max():.2f}km"
            )

            # 保存缓存 (含维度元信息)
            if use_cache and cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "neighbor_indices": neighbor_indices,
                        "neighbor_dists": neighbor_dists,
                        "position_dim": actual_pos_dim,
                        "dim_tag": dim_tag,
                        "dim_desc": dim_desc,
                        "lambda_geo": self.lambda_geo,
                        "lambda_alt": self.lambda_alt,
                        "num_stations": N,
                        "k_neighbors": k,
                    }, f)
                logger.info(f"[超图] 缓存已保存:")
                logger.info(f"[超图]   路径: {cache_path}")
                logger.info(f"[超图]   文件: {os.path.basename(cache_path)}")
                logger.info(f"[超图]   构建维度: {actual_pos_dim}D ({dim_tag}) -> {dim_desc}")

        # ---- 注册为 buffer ----
        self.neighbor_indices = torch.from_numpy(neighbor_indices).long()

        self.positions_tensor = torch.from_numpy(positions).float()

        try:
            device = next(self.parameters()).device
            self.neighbor_indices = self.neighbor_indices.to(device)
            self.positions_tensor = self.positions_tensor.to(device)
        except StopIteration:
            pass

        self._graph_built = True
        logger.info(f"[超图] ========== 超图就绪 ==========")
        logger.info(f"[超图] N={N}, K={k}, edges={N}, dim={actual_pos_dim}D ({dim_desc})")

    def _compute_state_summary(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算站点状态摘要。

        Args:
            x: [B, T, N, F] 输入时间窗口

        Returns:
            summary: [B, N, D_summary] 状态摘要
        """
        if self.summary_pool == "mean":
            # 时间维度平均池化
            pooled = x.mean(dim=1)  # [B, N, F]
        elif self.summary_pool == "last":
            # 最后时刻
            pooled = x[:, -1, :, :]  # [B, N, F]
        else:
            # 平均池化 + 线性映射
            pooled = x.mean(dim=1)  # [B, N, F]

        summary = self.summary_proj(pooled)  # [B, N, D_summary]
        return summary

    def _compute_keep_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算动态稀疏保留掩码。

        Args:
            weights: [B, N, K+1] softmax 权重

        Returns:
            keep_mask: [B, N, K+1] bool
        """
        B, N, K_plus_1 = weights.shape
        min_keep = min(self.pruning_min_keep, K_plus_1)

        if self.pruning_mode == "threshold":
            keep_mask = weights >= self.pruning_threshold
            # 保底至少保留 min_keep 个邻居
            topk_idx = torch.topk(weights, k=min_keep, dim=-1).indices
            keep_mask.scatter_(-1, topk_idx, True)
        else:
            # 默认 top_p: 按权重从高到低累计到 p
            sorted_w, sorted_idx = torch.sort(weights, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_w, dim=-1)
            keep_sorted = cumsum <= self.pruning_top_p
            # 确保第一个被选中
            keep_sorted[..., 0] = True
            # 保底 min_keep
            keep_sorted[..., :min_keep] = True
            keep_mask = torch.zeros_like(weights, dtype=torch.bool)
            keep_mask.scatter_(-1, sorted_idx, keep_sorted)

        # 邻接表第0位是中心站点自身，始终保留以稳定传播
        keep_mask[..., 0] = True

        return keep_mask

    def _collect_pruning_stats(self, keep_mask: torch.Tensor, k_effective: torch.Tensor):
        """收集最近一次动态K统计信息。"""
        keep_ratio = float(keep_mask.float().mean().item())
        k_min = int(k_effective.min().item())
        k_max = int(k_effective.max().item())
        k_mean = float(k_effective.float().mean().item())
        k_median = float(k_effective.float().median().item())
        self._last_pruning_stats = {
            "enabled": True,
            "mode": self.pruning_mode,
            "k_min": k_min,
            "k_max": k_max,
            "k_mean": k_mean,
            "k_median": k_median,
            "keep_ratio": keep_ratio,
            "candidate_k": int(keep_mask.shape[-1]),
            "top_p": float(self.pruning_top_p),
            "threshold": float(self.pruning_threshold),
        }

    def get_last_pruning_stats(self):
        """获取最近一次前向传播的动态K统计。"""
        return self._last_pruning_stats

    def _compute_adaptive_weights(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自适应邻域权重。

        Args:
            x: [B, T, N, F] 输入

        Returns:
            weights: [B, N, K+1] 归一化的自适应权重
        """
        B, T, N, F_in = x.shape
        K_plus_1 = self.neighbor_indices.shape[1]  # K+1
        device = x.device

        # 状态摘要
        summary = self._compute_state_summary(x)  # [B, N, D_summary]

        # 位置信息
        positions = self.positions_tensor.to(device)  # [N, P]

        # ---- 收集邻居信息 ----
        nbr_idx = self.neighbor_indices.to(device)  # [N, K+1]

        # 中心节点位置 (广播)
        p_center = positions.unsqueeze(1).expand(-1, K_plus_1, -1)
        # p_center: [N, K+1, P]

        # 邻居节点位置
        p_neighbor = positions[nbr_idx]
        # p_neighbor: [N, K+1, P]

        # 中心节点摘要 (广播)
        s_center = summary.unsqueeze(2).expand(-1, -1, K_plus_1, -1)
        # s_center: [B, N, K+1, D_s]

        # 邻居节点摘要
        s_neighbor = summary[:, nbr_idx, :]
        # 索引: summary [B, N, D_s], nbr_idx [N, K+1]
        # 结果: [B, N, K+1, D_s]

        # 扩展位置维度以匹配 batch
        p_center = p_center.unsqueeze(0).expand(B, -1, -1, -1)
        # p_center: [B, N, K+1, P]
        p_neighbor = p_neighbor.unsqueeze(0).expand(B, -1, -1, -1)
        # p_neighbor: [B, N, K+1, P]

        # 计算打分
        if self.float32_norm:
            scores = self.scorer(
                p_center.float(),
                p_neighbor.float(),
                s_center.float(),
                s_neighbor.float(),
            )
            weights = F.softmax(scores.float(), dim=-1)
        else:
            scores = self.scorer(p_center, p_neighbor, s_center, s_neighbor)
            weights = F.softmax(scores, dim=-1)

        # 动态稀疏化: 在权重归一化后进行掩码裁剪，再二次归一化
        if self.dynamic_pruning:
            keep_mask = self._compute_keep_mask(weights)
            weights = weights * keep_mask.to(weights.dtype)
            weights = weights.clamp_min(0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.degree_clamp_min)
            k_effective = keep_mask.sum(dim=-1)
            self._collect_pruning_stats(keep_mask, k_effective)
        else:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.degree_clamp_min)
            # 关闭动态裁剪时也记录固定K统计
            self._last_pruning_stats = {
                "enabled": False,
                "mode": "fixed",
                "k_min": int(weights.shape[-1]),
                "k_max": int(weights.shape[-1]),
                "k_mean": float(weights.shape[-1]),
                "k_median": float(weights.shape[-1]),
                "keep_ratio": 1.0,
                "candidate_k": int(weights.shape[-1]),
                "top_p": float(self.pruning_top_p),
                "threshold": float(self.pruning_threshold),
            }

        weights = weights.clamp_min(self.degree_clamp_min)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.degree_clamp_min)
        # weights: [B, N, K+1]
        return weights.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        超图空间传播前向传播。

        Args:
            x: [B, T, N, F] 输入时空特征

        Returns:
            H_s: [B, T, N, D] 空间增强特征
        """
        assert self._graph_built, "请先调用 build_graph() 构建超图结构！"

        B, T, N, F_in = x.shape
        device = x.device
        K_plus_1 = self.neighbor_indices.shape[1]
        nbr_idx = self.neighbor_indices.to(device)  # [N, K+1]

        # ---- 计算自适应权重 ----
        weights = self._compute_adaptive_weights(x)
        # weights: [B, N, K+1]

        # ---- 多层超图卷积 ----
        # 将时间步展开: [B*T, N, F_in]
        h = x.reshape(B * T, N, F_in)

        # 预计算循环外常量 (避免重复 reshape/expand)
        nbr_idx_flat = nbr_idx.reshape(-1)
        w_bt = weights.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, K_plus_1, 1)

        for layer_idx in range(self.num_layers):
            nbr_features = h[:, nbr_idx_flat, :].reshape(B * T, N, K_plus_1, -1)
            aggregated = (nbr_features * w_bt).sum(dim=2)
            h = self.conv_layers[layer_idx](aggregated)
            h = self.layer_norms[layer_idx](h)
            h = self.activation(h)
            h = self.dropout_layer(h)

        # 恢复时间维度
        H_s = h.reshape(B, T, N, self.hidden_dim)
        # H_s: [B, T, N, D]

        return H_s

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"k_neighbors={self.k_neighbors}, num_layers={self.num_layers}, "
            f"graph_built={self._graph_built}"
        )
