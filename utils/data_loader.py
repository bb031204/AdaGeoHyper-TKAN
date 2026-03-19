"""
DataLoader: 数据加载与预处理模块
================================

负责读取 pkl 格式数据集, 构建 PyTorch DataLoader。

数据集格式 (基于现有数据):
- trn.pkl / val.pkl / test.pkl:
  dict with keys: 'x', 'y' (可能有 'context')
  x shape: (num_samples, 12, num_stations, num_channels)
  y shape: (num_samples, 12, num_stations, num_channels)

- position.pkl:
  dict with key: 'lonlat'
  lonlat shape: (num_stations, 2)  [经度, 纬度]
  可能有额外的 'alt' / 'altitude' 键

数据集说明:
- temperature:       单通道 (C=1)
- humidity:          单通道 (C=1)
- cloud_cover:       单通道 (C=1)
- component_of_wind: 双通道 (C=2, u风和v风)
"""

import os
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


class StandardScaler:
    """
    标准化工具 (零均值单位方差)

    与现有工程 lib.utils.StandardScaler 兼容。
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data) -> np.ndarray:
        """反标准化, 支持 numpy 和 torch。"""
        if isinstance(data, torch.Tensor):
            return data * self.std + self.mean
        return data * self.std + self.mean

    def __repr__(self):
        return f"StandardScaler(mean={self.mean:.4f}, std={self.std:.4f})"


class WeatherDataset(Dataset):
    """
    气象时空数据集

    基于现有 dataloader.py 的 Dataset 类改写,
    适配 AdaGeoHyper-TKAN 模型输入。

    数据格式:
        x: (num_samples, 12, num_stations, C)  输入
        y: (num_samples, 12, num_stations, C)  标签

    Args:
        data_dir:    数据集目录 (如 D:/bishe/WYB/temperature)
        mode:        'trn', 'val', 'test'
        scaler:      StandardScaler 列表 (每个通道一个)
        sample_ratio: 样本抽样比例
        num_stations: 站点抽样数 (None=全部)
        station_indices: 指定的站点索引 (优先于 num_stations)
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "trn",
        scaler: Optional[List[StandardScaler]] = None,
        sample_ratio: float = 1.0,
        num_stations: Optional[int] = None,
        station_indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode

        # ---- 读取数据 ----
        pkl_path = os.path.join(data_dir, f"{mode}.pkl")
        logger.info(f"[数据] 加载 {mode} 集: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.x = data["x"]  # (N_samples, 12, N_stations, C)
        self.y = data["y"]  # (N_samples, 12, N_stations, C)

        logger.info(f"[数据] {mode} 集原始: x={self.x.shape}, y={self.y.shape}, dtype={self.x.dtype}")

        # ---- 站点采样 ----
        if station_indices is not None:
            self.x = self.x[:, :, station_indices, :]
            self.y = self.y[:, :, station_indices, :]
            logger.info(f"[数据] 站点采样 (指定索引): {len(station_indices)} 个站点")
        elif num_stations is not None and num_stations < self.x.shape[2]:
            total_stations = self.x.shape[2]
            indices = np.random.choice(total_stations, num_stations, replace=False)
            indices.sort()
            self.x = self.x[:, :, indices, :]
            self.y = self.y[:, :, indices, :]
            logger.info(f"[数据] 站点采样: {total_stations} -> {num_stations}")
            self._station_indices = indices
        else:
            self._station_indices = None

        # ---- 样本抽样 ----
        if sample_ratio < 1.0:
            total_samples = self.x.shape[0]
            n_samples = max(1, int(total_samples * sample_ratio))
            indices = np.random.choice(total_samples, n_samples, replace=False)
            indices.sort()
            self.x = self.x[indices]
            self.y = self.y[indices]
            logger.info(f"[数据] 样本抽样: {total_samples} -> {n_samples} (ratio={sample_ratio})")

        # ---- 标准化 ----
        self.scaler = scaler
        if scaler is not None:
            feature_len = self.x.shape[-1]
            for i in range(feature_len):
                self.x[..., i] = scaler[i].transform(self.x[..., i])
                self.y[..., i] = scaler[i].transform(self.y[..., i])
            logger.info(f"[数据] 标准化完成, 通道数={feature_len}")

        # 转为 float32
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

        logger.info(f"[数据] {mode} 集最终: x={self.x.shape}, y={self.y.shape}")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [12, N_stations, C]
            y: [12, N_stations, C]
        """
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def load_positions(
    data_dir: str,
    num_stations: Optional[int] = None,
    station_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    加载站点位置信息。

    Args:
        data_dir:        数据集目录
        num_stations:    站点数 (用于采样)
        station_indices: 指定站点索引

    Returns:
        positions: [N, 2] 或 [N, 3] (lon, lat[, alt])
        position_dim: 位置维度 (2 或 3)
    """
    position_file = os.path.join(data_dir, "position.pkl")
    logger.info(f"[数据] 加载位置信息: {position_file}")

    with open(position_file, "rb") as f:
        pos_data = pickle.load(f)

    # 提取经纬度
    if "lonlat" in pos_data:
        lonlat = pos_data["lonlat"]  # [N, 2]
    elif "position" in pos_data:
        lonlat = pos_data["position"]
    else:
        raise KeyError(f"position.pkl 中未找到 'lonlat' 或 'position' 键, 可用键: {list(pos_data.keys())}")

    # 提取海拔 (可选)
    alt = None
    for alt_key in ["alt", "altitude", "elev", "elevation"]:
        if alt_key in pos_data:
            alt = pos_data[alt_key]
            break

    # 构建 positions 数组
    if alt is not None:
        if alt.ndim == 1:
            alt = alt[:, None]
        positions = np.concatenate([lonlat, alt], axis=-1).astype(np.float64)
        position_dim = 3
        logger.info(f"[数据] 位置信息: lon/lat + altitude, shape={positions.shape}")
    else:
        positions = lonlat.astype(np.float64)
        position_dim = 2
        logger.info(f"[数据] 位置信息: lon/lat (无海拔), shape={positions.shape}")

    # 站点采样
    if station_indices is not None:
        positions = positions[station_indices]
    elif num_stations is not None and num_stations < positions.shape[0]:
        # 此处假设采样索引在 dataset 中已确定, 这里需要传入
        logger.warning("[数据] 位置采样: 需要使用相同的 station_indices!")

    logger.info(f"[数据] 最终位置: {positions.shape}, "
                f"lon范围=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
                f"lat范围=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")

    return positions, position_dim


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_stations: Optional[int] = None,
    sample_ratio: float = 1.0,
    val_sample_ratio: float = 1.0,
    test_sample_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    """
    创建完整的数据加载管道。

    Args:
        data_dir:         数据集目录
        batch_size:       批大小
        num_stations:     站点采样数
        sample_ratio:     训练集样本抽样比例
        val_sample_ratio: 验证集样本抽样比例
        test_sample_ratio: 测试集样本抽样比例
        seed:             随机种子

    Returns:
        dict with keys:
            'train_loader', 'val_loader', 'test_loader',
            'scaler', 'positions', 'position_dim',
            'feature_dim', 'station_indices', 'num_stations'
    """
    np.random.seed(seed)

    # ---- 读取训练集确定标准化参数 ----
    trn_path = os.path.join(data_dir, "trn.pkl")
    with open(trn_path, "rb") as f:
        trn_raw = pickle.load(f)

    train_x = trn_raw["x"]  # (N, 12, N_stations, C)
    feature_dim = train_x.shape[-1]
    total_stations = train_x.shape[2]

    logger.info(f"[数据] 特征维度: {feature_dim}, 总站点数: {total_stations}")

    # ---- 站点采样 ----
    station_indices = None
    actual_stations = total_stations
    if num_stations is not None and num_stations < total_stations:
        station_indices = np.sort(
            np.random.choice(total_stations, num_stations, replace=False)
        )
        actual_stations = num_stations
        logger.info(f"[数据] 站点采样: {total_stations} -> {num_stations}")

    # ---- 计算标准化参数 (基于训练集) ----
    if station_indices is not None:
        train_data_for_scaler = train_x[:, :, station_indices, :]
    else:
        train_data_for_scaler = train_x

    scaler = []
    for i in range(feature_dim):
        ch_data = train_data_for_scaler[..., i]
        sc = StandardScaler(mean=float(ch_data.mean()), std=float(ch_data.std()))
        scaler.append(sc)
        logger.info(f"[数据] 通道{i} Scaler: {sc}")

    del trn_raw, train_x, train_data_for_scaler

    # ---- 创建 Dataset ----
    train_set = WeatherDataset(
        data_dir, mode="trn", scaler=scaler,
        sample_ratio=sample_ratio, station_indices=station_indices,
    )
    val_set = WeatherDataset(
        data_dir, mode="val", scaler=scaler,
        sample_ratio=val_sample_ratio, station_indices=station_indices,
    )
    test_set = WeatherDataset(
        data_dir, mode="test", scaler=scaler,
        sample_ratio=test_sample_ratio, station_indices=station_indices,
    )

    # ---- 创建 DataLoader ----
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)

    # ---- 加载位置信息 ----
    positions, position_dim = load_positions(
        data_dir, station_indices=station_indices
    )

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "positions": positions,
        "position_dim": position_dim,
        "feature_dim": feature_dim,
        "station_indices": station_indices,
        "num_stations": actual_stations,
    }

    logger.info(f"[数据] 数据加载完成:")
    logger.info(f"  训练集: {len(train_set)} 样本")
    logger.info(f"  验证集: {len(val_set)} 样本")
    logger.info(f"  测试集: {len(test_set)} 样本")
    logger.info(f"  站点数: {actual_stations}, 特征维度: {feature_dim}")
    logger.info(f"  位置维度: {position_dim}")

    return result
