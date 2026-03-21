import os
import pickle
import logging
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from elements_settings import get_element_settings, normalize_element_name

logger = logging.getLogger(__name__)


CONTEXT_FEATURE_ORDER = [
    "year",      # 0
    "month",     # 1
    "day",       # 2
    "time",      # 3 (time index in window)
    "region",    # 4 (land coverage / land-sea mask-like feature)
    "altitude",  # 5
    "latitude",  # 6
    "longitude", # 7
]


class StandardScaler:
    """Channel-wise standard scaler: mean=0, std=1."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, data_2d: np.ndarray):
        # data_2d: [N, C]
        self.mean = np.mean(data_2d, axis=0, dtype=np.float64)
        self.std = np.std(data_2d, axis=0, dtype=np.float64)
        self.std = np.maximum(self.std, 1e-8)
        return self

    def _reshape_stats_np(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = (1,) * (data.ndim - 1) + (-1,)
        return self.mean.reshape(shape), self.std.reshape(shape)

    def _reshape_stats_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (1,) * (data.ndim - 1) + (-1,)
        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device).reshape(shape)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device).reshape(shape)
        return mean, std

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            mean, std = self._reshape_stats_torch(data)
            return (data - mean) / (std + 1e-8)
        mean, std = self._reshape_stats_np(data)
        return (data - mean) / (std + 1e-8)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            mean, std = self._reshape_stats_torch(data)
            return data * std + mean
        mean, std = self._reshape_stats_np(data)
        return data * std + mean

    def __repr__(self):
        return f"StandardScaler(channels={0 if self.mean is None else self.mean.shape[0]})"


class MinMaxScaler:
    """Channel-wise min-max scaler to [0, 1]."""

    def __init__(self, data_min: Optional[np.ndarray] = None, data_max: Optional[np.ndarray] = None):
        self.data_min = None if data_min is None else np.asarray(data_min, dtype=np.float64)
        self.data_max = None if data_max is None else np.asarray(data_max, dtype=np.float64)

    def fit(self, data_2d: np.ndarray):
        self.data_min = np.min(data_2d, axis=0)
        self.data_max = np.max(data_2d, axis=0)
        return self

    def _reshape_stats_np(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = (1,) * (data.ndim - 1) + (-1,)
        return self.data_min.reshape(shape), self.data_max.reshape(shape)

    def _reshape_stats_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (1,) * (data.ndim - 1) + (-1,)
        data_min = torch.as_tensor(self.data_min, dtype=data.dtype, device=data.device).reshape(shape)
        data_max = torch.as_tensor(self.data_max, dtype=data.dtype, device=data.device).reshape(shape)
        return data_min, data_max

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            data_min, data_max = self._reshape_stats_torch(data)
            return (data - data_min) / (data_max - data_min + 1e-8)
        data_min, data_max = self._reshape_stats_np(data)
        return (data - data_min) / (data_max - data_min + 1e-8)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data_min, data_max = self._reshape_stats_torch(data)
            return data * (data_max - data_min + 1e-8) + data_min
        data_min, data_max = self._reshape_stats_np(data)
        return data * (data_max - data_min + 1e-8) + data_min

    def __repr__(self):
        return f"MinMaxScaler(channels={0 if self.data_min is None else self.data_min.shape[0]})"


def build_scaler(scaler_type: str, data_2d: np.ndarray):
    st = str(scaler_type).strip().lower()
    if st == "standard":
        return StandardScaler().fit(data_2d)
    if st == "minmax":
        return MinMaxScaler().fit(data_2d)
    raise ValueError(f"Unsupported scaler_type='{scaler_type}', supported: standard/minmax")


def resolve_context_indices(context_features: Optional[Dict[str, bool]]) -> Tuple[List[int], List[str]]:
    """
    Parse config flags to selected context channel indices.

    Supported keys:
      use_year/use_month/use_day/use_time/use_region/use_altitude/use_longitude/use_latitude
    """
    if context_features is None:
        return [], []

    key_map = {
        "use_year": 0,
        "use_month": 1,
        "use_day": 2,
        "use_time": 3,
        "use_region": 4,
        "use_altitude": 5,
        "use_latitude": 6,
        "use_longitude": 7,
    }

    selected = [idx for key, idx in key_map.items() if bool(context_features.get(key, False))]
    selected_names = [CONTEXT_FEATURE_ORDER[i] for i in selected]
    return selected, selected_names


def _preprocess_context_calendar(context_sel: np.ndarray, context_indices: List[int]) -> np.ndarray:
    """Apply simple modular mapping for periodic context features before scaling."""
    out = context_sel.copy()
    # month in [1,12] -> [-0.5, 0.5)
    if 1 in context_indices:
        local_idx = context_indices.index(1)
        out[..., local_idx] = ((out[..., local_idx] - 1.0) % 12.0) / 12.0 - 0.5
    # day in [1,31] -> [-0.5, 0.5)
    if 2 in context_indices:
        local_idx = context_indices.index(2)
        out[..., local_idx] = ((out[..., local_idx] - 1.0) % 31.0) / 31.0 - 0.5
    # time idx in [0, T) -> [-0.5, 0.5)
    if 3 in context_indices:
        local_idx = context_indices.index(3)
        max_t = np.maximum(np.max(out[..., local_idx]), 1.0)
        out[..., local_idx] = (out[..., local_idx] % max_t) / max_t - 0.5
    return out


class WeatherDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "trn",
        weather_scaler: Optional[object] = None,
        context_scaler: Optional[object] = None,
        sample_ratio: float = 1.0,
        num_stations: Optional[int] = None,
        station_indices: Optional[np.ndarray] = None,
        include_context: bool = False,
        context_indices: Optional[List[int]] = None,
        element_settings: Optional[Dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.include_context = include_context
        self.context_indices = context_indices or []
        self.element_settings = element_settings or {}
        self.weather_scaler = weather_scaler
        self.context_scaler = context_scaler

        pkl_path = os.path.join(data_dir, f"{mode}.pkl")
        logger.info(f"[Data] Loading {mode}: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        x = data["x"]
        y = data["y"]
        weather_dim = y.shape[-1]

        context = None
        if self.include_context and len(self.context_indices) > 0:
            if "context" not in data:
                raise KeyError(
                    f"{mode}.pkl has no 'context' but include_context=true and context_indices={self.context_indices}"
                )
            context = data["context"]

        # station sampling
        self._station_indices = None
        if station_indices is not None:
            x = x[:, :, station_indices, :]
            y = y[:, :, station_indices, :]
            if context is not None:
                context = context[:, :, station_indices, :]
            logger.info(f"[Data] Station sampling by fixed indices: {len(station_indices)}")
        elif num_stations is not None and num_stations < x.shape[2]:
            total_stations = x.shape[2]
            idx = np.random.choice(total_stations, num_stations, replace=False)
            idx.sort()
            x = x[:, :, idx, :]
            y = y[:, :, idx, :]
            if context is not None:
                context = context[:, :, idx, :]
            self._station_indices = idx
            logger.info(f"[Data] Station sampling: {total_stations} -> {num_stations}")

        # sample sampling
        if sample_ratio < 1.0:
            total_samples = x.shape[0]
            n_samples = max(1, int(total_samples * sample_ratio))
            idx = np.random.choice(total_samples, n_samples, replace=False)
            idx.sort()
            x = x[idx]
            y = y[idx]
            if context is not None:
                context = context[idx]
            logger.info(f"[Data] Sample sampling: {total_samples} -> {n_samples} (ratio={sample_ratio})")

        # temperature unit conversion before normalization
        if self.element_settings.get("kelvin_to_celsius", False):
            x[..., :weather_dim] = x[..., :weather_dim] - 273.15
            y[..., :weather_dim] = y[..., :weather_dim] - 273.15

        # weather normalization (global scaler, channel-wise stats)
        if self.weather_scaler is not None and self.element_settings.get("normalize", True):
            x[..., :weather_dim] = self.weather_scaler.transform(x[..., :weather_dim])
            y[..., :weather_dim] = self.weather_scaler.transform(y[..., :weather_dim])

        # context normalization + concat to x only (y keeps weather-only for loss)
        if context is not None and len(self.context_indices) > 0:
            context_dim = context.shape[-1]
            if max(self.context_indices) >= context_dim:
                raise IndexError(
                    f"context dimension={context_dim}, requested indices={self.context_indices}"
                )

            context_sel = context[..., self.context_indices]
            context_sel = _preprocess_context_calendar(context_sel, self.context_indices)
            if self.context_scaler is not None:
                context_sel = self.context_scaler.transform(context_sel)

            x = np.concatenate([x, context_sel], axis=-1)
            logger.info(
                f"[Data] Context concat to x only: x channels={x.shape[-1]}, "
                f"y channels={y.shape[-1]} (weather only), indices={self.context_indices}"
            )

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.weather_dim = weather_dim
        logger.info(f"[Data] Final {mode}: x={self.x.shape}, y={self.y.shape}")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def load_positions(
    data_dir: str,
    num_stations: Optional[int] = None,
    station_indices: Optional[np.ndarray] = None,
    context_altitude: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Altitude source priority:
      1) position.pkl keys: alt/altitude/elev/elevation
      2) context channel 5 passed by caller
    """
    position_file = os.path.join(data_dir, "position.pkl")
    logger.info(f"[Data] Loading positions: {position_file}")

    with open(position_file, "rb") as f:
        pos_data = pickle.load(f)

    if "lonlat" in pos_data:
        lonlat = pos_data["lonlat"]
    elif "position" in pos_data:
        lonlat = pos_data["position"]
    else:
        raise KeyError(f"position.pkl missing 'lonlat'/'position', keys={list(pos_data.keys())}")
    full_station_count = lonlat.shape[0]

    alt = None
    alt_source = None
    for alt_key in ["alt", "altitude", "elev", "elevation"]:
        if alt_key in pos_data:
            alt = pos_data[alt_key]
            alt_source = f"position.pkl[{alt_key}]"
            break

    if alt is None and context_altitude is not None:
        alt = context_altitude
        alt_source = "context channel 5"

    if station_indices is not None:
        lonlat = lonlat[station_indices]
        if alt is not None:
            if alt.ndim == 1:
                if alt.shape[0] == len(station_indices):
                    pass
                elif alt.shape[0] == full_station_count:
                    alt = alt[station_indices]
                else:
                    raise ValueError(
                        f"Altitude length {alt.shape[0]} does not match full stations or sampled stations "
                        f"(sampled={len(station_indices)})"
                    )
            else:
                if alt.shape[0] == len(station_indices):
                    pass
                elif alt.shape[0] == full_station_count:
                    alt = alt[station_indices]
                else:
                    raise ValueError(
                        f"Altitude first dimension {alt.shape[0]} does not match full stations or sampled stations "
                        f"(sampled={len(station_indices)})"
                    )

    if alt is not None:
        if alt.ndim == 1:
            alt = alt[:, None]
        positions = np.concatenate([lonlat, alt], axis=-1).astype(np.float64)
        position_dim = 3
        logger.info(f"[Data] Positions with altitude ({alt_source}), shape={positions.shape}")
    else:
        positions = lonlat.astype(np.float64)
        position_dim = 2
        logger.info(f"[Data] Positions lon/lat only, shape={positions.shape}")

    if station_indices is None and num_stations is not None and num_stations < positions.shape[0]:
        logger.warning("[Data] Position sampling needs same station_indices for strict alignment")

    return positions, position_dim


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_stations: Optional[int] = None,
    sample_ratio: float = 1.0,
    val_sample_ratio: float = 1.0,
    test_sample_ratio: float = 1.0,
    seed: int = 42,
    include_context: bool = False,
    context_features: Optional[Dict[str, bool]] = None,
    use_context_altitude: bool = True,
    element: Optional[str] = None,
) -> Dict:
    np.random.seed(seed)

    if element is None:
        element = "Temperature"
    element_name = normalize_element_name(element)
    element_settings = get_element_settings(element_name)

    trn_path = os.path.join(data_dir, "trn.pkl")
    with open(trn_path, "rb") as f:
        trn_raw = pickle.load(f)

    train_x = trn_raw["x"]  # (N, T, Stations, C_target)
    target_weather_dim = train_x.shape[-1]
    total_stations = train_x.shape[2]

    context_indices, context_feature_names = resolve_context_indices(context_features)

    station_indices = None
    actual_stations = total_stations
    if num_stations is not None and num_stations < total_stations:
        station_indices = np.sort(np.random.choice(total_stations, num_stations, replace=False))
        actual_stations = num_stations
        logger.info(f"[Data] Station sampling: {total_stations} -> {num_stations}")

    # fit weather scaler from training split only
    weather_for_fit = train_x[:, :, station_indices, :] if station_indices is not None else train_x
    if element_settings.get("kelvin_to_celsius", False):
        weather_for_fit = weather_for_fit - 273.15

    weather_scaler = None
    if element_settings.get("normalize", True):
        weather_fit_2d = weather_for_fit.reshape(-1, target_weather_dim)
        weather_scaler = build_scaler(element_settings.get("scaler_type", "standard"), weather_fit_2d)

    # context scaler fit from training split only
    context_scaler = None
    selected_context_dim = 0
    if include_context and len(context_indices) > 0:
        if "context" not in trn_raw:
            raise KeyError("include_context=true but trn.pkl has no 'context'")
        context_dim = trn_raw["context"].shape[-1]
        if max(context_indices) >= context_dim:
            raise IndexError(f"context dim={context_dim}, requested indices={context_indices}")

        context_fit = trn_raw["context"][:, :, station_indices, :] if station_indices is not None else trn_raw["context"]
        context_sel_fit = context_fit[..., context_indices]
        context_sel_fit = _preprocess_context_calendar(context_sel_fit, context_indices)
        selected_context_dim = context_sel_fit.shape[-1]
        context_fit_2d = context_sel_fit.reshape(-1, selected_context_dim)
        context_scaler = build_scaler(
            element_settings.get("context_scaler_type", "standard"),
            context_fit_2d,
        )
        logger.info(
            f"[Data] Context enabled: {context_feature_names} (indices={context_indices}), "
            f"input channels {target_weather_dim} -> {target_weather_dim + selected_context_dim}"
        )
    else:
        logger.info("[Data] Context disabled")

    # optional altitude from context channel 5 for hypergraph
    context_altitude = None
    if use_context_altitude:
        if "context" in trn_raw and trn_raw["context"].shape[-1] > 5:
            context_altitude = trn_raw["context"][0, 0, :, 5].astype(np.float64)
            if station_indices is not None:
                context_altitude = context_altitude[station_indices]
            logger.info(
                f"[Data] Using context altitude channel for graph, range=[{context_altitude.min():.2f}, {context_altitude.max():.2f}]"
            )
        else:
            logger.warning("[Data] use_context_altitude=true but context channel 5 not available")

    del trn_raw, train_x, weather_for_fit

    train_set = WeatherDataset(
        data_dir,
        mode="trn",
        weather_scaler=weather_scaler,
        context_scaler=context_scaler,
        sample_ratio=sample_ratio,
        station_indices=station_indices,
        include_context=include_context,
        context_indices=context_indices,
        element_settings=element_settings,
    )
    val_set = WeatherDataset(
        data_dir,
        mode="val",
        weather_scaler=weather_scaler,
        context_scaler=context_scaler,
        sample_ratio=val_sample_ratio,
        station_indices=station_indices,
        include_context=include_context,
        context_indices=context_indices,
        element_settings=element_settings,
    )
    test_set = WeatherDataset(
        data_dir,
        mode="test",
        weather_scaler=weather_scaler,
        context_scaler=context_scaler,
        sample_ratio=test_sample_ratio,
        station_indices=station_indices,
        include_context=include_context,
        context_indices=context_indices,
        element_settings=element_settings,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    positions, position_dim = load_positions(
        data_dir,
        station_indices=station_indices,
        context_altitude=context_altitude,
    )

    input_feature_dim = target_weather_dim + selected_context_dim
    total_target_dim = target_weather_dim

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": weather_scaler,
        "weather_scaler": weather_scaler,
        "context_scaler": context_scaler,
        "positions": positions,
        "position_dim": position_dim,
        "input_feature_dim": input_feature_dim,
        "target_dim": total_target_dim,
        "target_weather_dim": target_weather_dim,
        "feature_dim": total_target_dim,
        "station_indices": station_indices,
        "num_stations": actual_stations,
        "include_context": include_context,
        "context_indices": context_indices,
        "context_feature_names": context_feature_names,
        "context_dim_selected": selected_context_dim,
        "element": element_name,
        "element_settings": element_settings,
    }

    logger.info("[Data] Data loading done")
    logger.info(
        f"  Element: {element_name}, normalize={element_settings.get('normalize', True)}, "
        f"kelvin_to_celsius={element_settings.get('kelvin_to_celsius', False)}, "
        f"weather_scaler={element_settings.get('scaler_type', 'standard')}, "
        f"context_scaler={element_settings.get('context_scaler_type', 'standard')}"
    )
    logger.info(f"  Train samples: {len(train_set)}")
    logger.info(f"  Val samples: {len(val_set)}")
    logger.info(f"  Test samples: {len(test_set)}")
    logger.info(
        f"  Stations: {actual_stations}, input_dim={input_feature_dim}, "
        f"target_dim={total_target_dim}, weather_target_dim={target_weather_dim}"
    )
    logger.info(f"  Position dim: {position_dim}")

    return result
