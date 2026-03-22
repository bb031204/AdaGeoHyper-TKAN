import os
import pickle
import logging
from typing import Optional, Tuple, List, Dict, Any

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


_PERIODIC_INDICES = {1, 2, 3}  # month, day, time — use sin/cos encoding
_PERIODIC_PERIOD = {1: 12.0, 2: 31.0, 3: 12.0}  # month=12, day=31, time(in-window)=12


def _preprocess_context_calendar(context_sel: np.ndarray, context_indices: List[int]) -> np.ndarray:
    """
    Encode periodic calendar features (month/day/time) with sin/cos,
    pass through all other features unchanged.

    This correctly models cyclical proximity (e.g. December ↔ January)
    and doubles the column count for each periodic feature.
    """
    parts: List[np.ndarray] = []
    for local_idx, global_idx in enumerate(context_indices):
        col = context_sel[..., local_idx:local_idx + 1]
        if global_idx in _PERIODIC_INDICES:
            period = _PERIODIC_PERIOD.get(global_idx, None)
            if period is None:
                period = float(np.maximum(np.max(col), 1.0))
                angle = 2.0 * np.pi * col / period
            else:
                angle = 2.0 * np.pi * (col - 1.0) / period
            parts.append(np.sin(angle))
            parts.append(np.cos(angle))
        else:
            parts.append(col)
    return np.concatenate(parts, axis=-1)


def _serialize_scaler(scaler: Optional[object]) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        return {
            "type": "standard",
            "mean": scaler.mean.astype(np.float64),
            "std": scaler.std.astype(np.float64),
        }
    if isinstance(scaler, MinMaxScaler):
        return {
            "type": "minmax",
            "data_min": scaler.data_min.astype(np.float64),
            "data_max": scaler.data_max.astype(np.float64),
        }
    raise TypeError(f"Unsupported scaler type for serialization: {type(scaler)}")


def _deserialize_scaler(payload: Optional[Dict[str, Any]]) -> Optional[object]:
    if payload is None:
        return None
    st = str(payload.get("type", "")).strip().lower()
    if st == "standard":
        return StandardScaler(mean=np.asarray(payload["mean"]), std=np.asarray(payload["std"]))
    if st == "minmax":
        return MinMaxScaler(
            data_min=np.asarray(payload["data_min"]),
            data_max=np.asarray(payload["data_max"]),
        )
    raise ValueError(f"Unsupported serialized scaler type: {payload.get('type')}")


def save_preprocessing_artifact(
    artifact_path: str,
    *,
    station_indices: Optional[np.ndarray],
    weather_scaler: Optional[object],
    context_scaler: Optional[object],
    element_name: str,
    context_indices: List[int],
    context_feature_names: List[str],
    target_weather_dim: int,
):
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    payload = {
        "station_indices": None if station_indices is None else station_indices.astype(np.int64),
        "weather_scaler": _serialize_scaler(weather_scaler),
        "context_scaler": _serialize_scaler(context_scaler),
        "element_name": element_name,
        "context_indices": list(context_indices),
        "context_feature_names": list(context_feature_names),
        "target_weather_dim": int(target_weather_dim),
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"[Data] Preprocessing artifact saved: {artifact_path}")


def load_preprocessing_artifact(artifact_path: str) -> Dict[str, Any]:
    with open(artifact_path, "rb") as f:
        payload = pickle.load(f)
    payload["station_indices"] = (
        None
        if payload.get("station_indices") is None
        else np.asarray(payload["station_indices"], dtype=np.int64)
    )
    payload["weather_scaler"] = _deserialize_scaler(payload.get("weather_scaler"))
    payload["context_scaler"] = _deserialize_scaler(payload.get("context_scaler"))
    logger.info(f"[Data] Preprocessing artifact loaded: {artifact_path}")
    return payload


def _robust_clip_for_fit(data_2d: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    """Lightweight robust preprocessing for scaler fitting only."""
    lo = np.quantile(data_2d, lower_q, axis=0)
    hi = np.quantile(data_2d, upper_q, axis=0)
    return np.clip(data_2d, lo, hi)


def _extract_static_context_signature(
    raw_data: Dict,
    station_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Build per-station static signature from context channels (altitude/latitude/longitude)
    to verify station-order consistency across train/val/test.
    """
    context = raw_data.get("context", None)
    if context is None or context.ndim != 4 or context.shape[-1] <= 5:
        return None

    # Channels: 5=altitude, 6=latitude, 7=longitude (if available)
    chs = [c for c in [5, 6, 7] if c < context.shape[-1]]
    if not chs:
        return None

    # Use first sample/time because these channels are station-static by design.
    sig = context[0, 0, :, :][:, chs].astype(np.float64)
    if station_indices is not None:
        sig = sig[station_indices]
    return sig


def _validate_station_consistency(
    trn_raw: Dict,
    val_raw: Dict,
    test_raw: Dict,
    station_indices: Optional[np.ndarray],
):
    """Strict checks for station set/order alignment across splits."""
    trn_st = trn_raw["x"].shape[2]
    val_st = val_raw["x"].shape[2]
    tst_st = test_raw["x"].shape[2]
    if not (trn_st == val_st == tst_st):
        raise ValueError(
            f"Station count mismatch across splits: trn={trn_st}, val={val_st}, test={tst_st}"
        )

    if station_indices is not None:
        if station_indices.ndim != 1:
            raise ValueError(f"station_indices must be 1D, got shape={station_indices.shape}")
        if len(station_indices) == 0:
            raise ValueError("station_indices is empty")
        if station_indices.min() < 0 or station_indices.max() >= trn_st:
            raise IndexError(
                f"station_indices out of range: min={station_indices.min()}, "
                f"max={station_indices.max()}, total={trn_st}"
            )
        if not np.all(station_indices[:-1] <= station_indices[1:]):
            raise ValueError("station_indices must be sorted non-decreasing for deterministic ordering")

    trn_sig = _extract_static_context_signature(trn_raw, station_indices)
    val_sig = _extract_static_context_signature(val_raw, station_indices)
    tst_sig = _extract_static_context_signature(test_raw, station_indices)
    if trn_sig is not None and val_sig is not None and tst_sig is not None:
        if not np.allclose(trn_sig, val_sig, rtol=0.0, atol=1e-6):
            raise ValueError("Station order mismatch: train signature != val signature")
        if not np.allclose(trn_sig, tst_sig, rtol=0.0, atol=1e-6):
            raise ValueError("Station order mismatch: train signature != test signature")
        logger.info("[Data] Station consistency check passed (train/val/test static context signatures aligned)")
    else:
        logger.warning(
            "[Data] Station signature check skipped (context altitude/lat/lon not fully available in all splits)"
        )


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
    fixed_station_indices: Optional[np.ndarray] = None,
    weather_scaler_override: Optional[object] = None,
    context_scaler_override: Optional[object] = None,
    robust_preprocess: Optional[Dict[str, Any]] = None,
) -> Dict:
    np.random.seed(seed)

    if element is None:
        element = "Temperature"
    element_name = normalize_element_name(element)
    element_settings = get_element_settings(element_name)
    rp_cfg = dict(robust_preprocess or {})
    if "enabled" not in rp_cfg:
        rp_cfg["enabled"] = (element_name == "Temperature")
    rp_cfg.setdefault("lower_q", 0.001)
    rp_cfg.setdefault("upper_q", 0.999)

    trn_path = os.path.join(data_dir, "trn.pkl")
    with open(trn_path, "rb") as f:
        trn_raw = pickle.load(f)
    val_path = os.path.join(data_dir, "val.pkl")
    with open(val_path, "rb") as f:
        val_raw = pickle.load(f)
    test_path = os.path.join(data_dir, "test.pkl")
    with open(test_path, "rb") as f:
        test_raw = pickle.load(f)

    train_x = trn_raw["x"]  # (N, T, Stations, C_target)
    target_weather_dim = train_x.shape[-1]
    total_stations = train_x.shape[2]

    context_indices, context_feature_names = resolve_context_indices(context_features)

    station_indices = None
    actual_stations = total_stations
    if fixed_station_indices is not None:
        station_indices = np.asarray(fixed_station_indices, dtype=np.int64)
        actual_stations = len(station_indices)
        logger.info(f"[Data] Station sampling by provided fixed indices: {actual_stations}")
    elif num_stations is not None and num_stations < total_stations:
        station_indices = np.sort(np.random.choice(total_stations, num_stations, replace=False))
        actual_stations = num_stations
        logger.info(f"[Data] Station sampling: {total_stations} -> {num_stations}")

    _validate_station_consistency(trn_raw, val_raw, test_raw, station_indices)

    # fit weather scaler from training split only
    weather_for_fit = train_x[:, :, station_indices, :] if station_indices is not None else train_x
    if element_settings.get("kelvin_to_celsius", False):
        weather_for_fit = weather_for_fit - 273.15

    weather_scaler = weather_scaler_override
    if weather_scaler is None and element_settings.get("normalize", True):
        weather_fit_2d = weather_for_fit.reshape(-1, target_weather_dim)
        rp_enable = bool(rp_cfg.get("enabled", False))
        if rp_enable:
            low_q = float(rp_cfg.get("lower_q", 0.001))
            high_q = float(rp_cfg.get("upper_q", 0.999))
            if not (0.0 <= low_q < high_q <= 1.0):
                raise ValueError(f"Invalid robust quantile range: lower_q={low_q}, upper_q={high_q}")
            weather_fit_2d = _robust_clip_for_fit(weather_fit_2d, low_q, high_q)
            logger.info(
                f"[Data] Robust scaler-fit clipping enabled: lower_q={low_q:.4f}, upper_q={high_q:.4f}"
            )
        weather_scaler = build_scaler(element_settings.get("scaler_type", "standard"), weather_fit_2d)
    elif weather_scaler is not None:
        logger.info("[Data] Reusing provided weather scaler (cross-run consistency)")

    # context scaler fit from training split only
    context_scaler = context_scaler_override
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
        if context_scaler is None:
            context_fit_2d = context_sel_fit.reshape(-1, selected_context_dim)
            context_scaler = build_scaler(
                element_settings.get("context_scaler_type", "standard"),
                context_fit_2d,
            )
        else:
            logger.info("[Data] Reusing provided context scaler (cross-run consistency)")
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

    del trn_raw, val_raw, test_raw, train_x, weather_for_fit

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
    if positions.shape[0] != actual_stations:
        raise ValueError(
            f"Position/station mismatch after sampling: positions={positions.shape[0]}, expected={actual_stations}"
        )
    logger.info(
        f"[Data] Station index reuse check passed: sampled stations={actual_stations}, "
        "same fixed indices reused for train/val/test/positions"
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
    logger.info(
        f"  Robust preprocess (fit-only): enabled={bool(rp_cfg.get('enabled', False))}, "
        f"q=[{float(rp_cfg.get('lower_q', 0.001)):.4f}, {float(rp_cfg.get('upper_q', 0.999)):.4f}]"
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
