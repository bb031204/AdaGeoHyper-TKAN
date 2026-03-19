"""
Metrics: 评估指标模块
======================

提供气象预测常用评估指标:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

支持 numpy 和 torch tensor 输入。
"""

import numpy as np
import torch
from typing import Dict, Union

Tensor = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Tensor) -> np.ndarray:
    """将 tensor 转为 numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def MAE(pred: Tensor, true: Tensor) -> float:
    """
    平均绝对误差。

    Args:
        pred: 预测值
        true: 真实值

    Returns:
        MAE 值
    """
    pred, true = _to_numpy(pred), _to_numpy(true)
    return float(np.mean(np.abs(pred - true)))


def RMSE(pred: Tensor, true: Tensor) -> float:
    """
    均方根误差。

    Args:
        pred: 预测值
        true: 真实值

    Returns:
        RMSE 值
    """
    pred, true = _to_numpy(pred), _to_numpy(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def MAPE(pred: Tensor, true: Tensor, eps: float = 1e-8) -> float:
    """
    平均绝对百分比误差。

    Args:
        pred: 预测值
        true: 真实值
        eps:  防止除零的小量

    Returns:
        MAPE 值 (百分比)
    """
    pred, true = _to_numpy(pred), _to_numpy(true)
    mask = np.abs(true) > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / (true[mask] + eps))) * 100)


def compute_metrics(pred: Tensor, true: Tensor) -> Dict[str, float]:
    """
    计算全部评估指标。

    Args:
        pred: 预测值
        true: 真实值

    Returns:
        dict: {'MAE': ..., 'RMSE': ..., 'MAPE': ...}
    """
    return {
        "MAE": MAE(pred, true),
        "RMSE": RMSE(pred, true),
        "MAPE": MAPE(pred, true),
    }


def compute_per_step_metrics(pred: Tensor, true: Tensor, num_steps: int = 12) -> Dict[str, list]:
    """
    计算每个预测步的指标。

    Args:
        pred: [N, T, ...]  预测值
        true: [N, T, ...]  真实值
        num_steps: 时间步数

    Returns:
        dict: {'MAE': [step1, step2, ...], 'RMSE': [...], 'MAPE': [...]}
    """
    pred, true = _to_numpy(pred), _to_numpy(true)
    results = {"MAE": [], "RMSE": [], "MAPE": []}

    for t in range(min(num_steps, pred.shape[1])):
        p_t = pred[:, t]
        t_t = true[:, t]
        results["MAE"].append(MAE(p_t, t_t))
        results["RMSE"].append(RMSE(p_t, t_t))
        results["MAPE"].append(MAPE(p_t, t_t))

    return results
