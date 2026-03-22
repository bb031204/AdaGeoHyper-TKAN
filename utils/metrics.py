"""
Metrics: 评估指标模块
======================

面向气象预测的通用+要素特定指标：
- 通用: MAE / RMSE / MAPE / sMAPE / WMAPE
- 风速: Vector MAE / Vector RMSE (针对 u,v 分量联合误差)
"""

import numpy as np
import torch
from typing import Dict, Union, Optional

Tensor = Union[np.ndarray, torch.Tensor]


ELEMENT_METRIC_CONFIG = {
    "Temperature": {
        "primary": ["MAE", "RMSE"],
        "secondary": ["sMAPE", "WMAPE"],
        "optional": ["MAPE"],
    },
    "Humidity": {
        "primary": ["MAE", "RMSE", "sMAPE"],
        "secondary": ["WMAPE"],
        "optional": ["MAPE"],
    },
    "Cloud": {
        "primary": ["MAE", "RMSE", "sMAPE"],
        "secondary": ["WMAPE"],
        "optional": ["MAPE"],
    },
    "Wind": {
        "primary": ["MAE", "RMSE", "VectorMAE", "VectorRMSE"],
        "secondary": ["sMAPE", "WMAPE"],
        "optional": ["MAPE"],
    },
}


def _to_numpy(x: Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def MAE(pred: Tensor, true: Tensor) -> float:
    pred, true = _to_numpy(pred), _to_numpy(true)
    return float(np.mean(np.abs(pred - true)))


def RMSE(pred: Tensor, true: Tensor) -> float:
    pred, true = _to_numpy(pred), _to_numpy(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def MAPE(pred: Tensor, true: Tensor, eps: float = 1e-8) -> float:
    pred, true = _to_numpy(pred), _to_numpy(true)
    denom = np.abs(true)
    mask = denom > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / (true[mask] + eps))) * 100)


def sMAPE(pred: Tensor, true: Tensor, eps: float = 1e-8) -> float:
    pred, true = _to_numpy(pred), _to_numpy(true)
    denom = np.abs(pred) + np.abs(true)
    mask = denom > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2.0 * np.abs(pred[mask] - true[mask]) / (denom[mask] + eps)) * 100)


def WMAPE(pred: Tensor, true: Tensor, eps: float = 1e-8) -> float:
    pred, true = _to_numpy(pred), _to_numpy(true)
    numerator = np.sum(np.abs(pred - true))
    denominator = np.sum(np.abs(true))
    if denominator <= eps:
        return 0.0
    return float((numerator / (denominator + eps)) * 100)


def vector_errors(pred: Tensor, true: Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    针对风速 (u,v) 计算向量误差。
    输入最后一维至少为2，前两维视为 u,v。
    """
    pred, true = _to_numpy(pred), _to_numpy(true)
    if pred.shape[-1] < 2 or true.shape[-1] < 2:
        return {"VectorMAE": 0.0, "VectorRMSE": 0.0}

    du = pred[..., 0] - true[..., 0]
    dv = pred[..., 1] - true[..., 1]
    mag_err = np.sqrt(np.maximum(du * du + dv * dv, eps))
    return {
        "VectorMAE": float(np.mean(mag_err)),
        "VectorRMSE": float(np.sqrt(np.mean(mag_err ** 2))),
    }


def compute_metrics(
    pred: Tensor,
    true: Tensor,
    element_name: Optional[str] = None,
) -> Dict[str, float]:
    metrics = {
        "MAE": MAE(pred, true),
        "RMSE": RMSE(pred, true),
        "sMAPE": sMAPE(pred, true),
        "WMAPE": WMAPE(pred, true),
        "MAPE": MAPE(pred, true),
    }

    if str(element_name or "").strip().lower() == "wind":
        metrics.update(vector_errors(pred, true))

    return metrics


def compute_per_step_metrics(
    pred: Tensor,
    true: Tensor,
    num_steps: int = 12,
    element_name: Optional[str] = None,
) -> Dict[str, list]:
    pred, true = _to_numpy(pred), _to_numpy(true)
    results = {"MAE": [], "RMSE": [], "sMAPE": [], "WMAPE": [], "MAPE": []}
    if str(element_name or "").strip().lower() == "wind":
        results["VectorMAE"] = []
        results["VectorRMSE"] = []

    for t in range(min(num_steps, pred.shape[1])):
        p_t = pred[:, t]
        t_t = true[:, t]
        step_metrics = compute_metrics(p_t, t_t, element_name=element_name)
        for k in results.keys():
            results[k].append(step_metrics.get(k, 0.0))

    return results
