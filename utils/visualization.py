"""
Visualization: 可视化模块
==========================

提供训练过程和预测结果的可视化功能:
- 训练 loss 曲线
- 验证指标曲线
- 预测值 vs 真实值对比图
- 每个时间步预测对比图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 设置中文字体 (如果可用)
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training & Validation Loss",
):
    """
    绘制训练与验证 loss 曲线。

    Args:
        train_losses: 每个 epoch 的训练 loss
        val_losses:   每个 epoch 的验证 loss
        save_path:    图片保存路径
        title:        图标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 标注最佳验证 loss
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5, label=f"Best@epoch{best_epoch}")
    ax.annotate(
        f"Best: {best_val_loss:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + max(1, len(train_losses) // 10), best_val_loss),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] Loss 曲线已保存: {save_path}")


def plot_metrics_curve(
    metrics_history: Dict[str, List[float]],
    save_path: str,
    title: str = "Validation Metrics",
):
    """
    绘制验证指标随 epoch 变化曲线。

    Args:
        metrics_history: {'MAE': [...], 'RMSE': [...], 'MAPE': [...]}
        save_path:       保存路径
        title:           标题
    """
    fig, axes = plt.subplots(1, len(metrics_history), figsize=(5 * len(metrics_history), 5))
    if len(metrics_history) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]
        epochs = range(1, len(values) + 1)
        color = colors[idx % len(colors)]

        ax.plot(epochs, values, color=color, linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f"{metric_name}", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 标注最佳值
        best_idx = np.argmin(values)
        ax.scatter(best_idx + 1, values[best_idx], color="red", s=50, zorder=5)
        ax.annotate(f"{values[best_idx]:.4f}", xy=(best_idx + 1, values[best_idx]),
                    fontsize=9, color="red")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] 指标曲线已保存: {save_path}")


def plot_prediction_vs_truth(
    pred: np.ndarray,
    true: np.ndarray,
    save_dir: str,
    num_samples: int = 3,
    num_stations: int = 3,
    dataset_name: str = "",
):
    """
    绘制预测值与真实值对比图。

    对选定的样本和站点, 绘制 12 步预测 vs 真实值重叠图。

    Args:
        pred:         [N, T, N_s, C] 预测结果
        true:         [N, T, N_s, C] 真实值
        save_dir:     保存目录
        num_samples:  绘制的样本数
        num_stations: 每个样本绘制的站点数
        dataset_name: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)
    N, T, N_s, C = pred.shape

    # 选择样本和站点
    sample_indices = np.linspace(0, N - 1, min(num_samples, N), dtype=int)
    station_indices = np.linspace(0, N_s - 1, min(num_stations, N_s), dtype=int)

    for s_idx in sample_indices:
        fig, axes = plt.subplots(len(station_indices), C, figsize=(6 * C, 4 * len(station_indices)))
        if len(station_indices) == 1 and C == 1:
            axes = np.array([[axes]])
        elif len(station_indices) == 1:
            axes = axes[np.newaxis, :]
        elif C == 1:
            axes = axes[:, np.newaxis]

        for row, st_idx in enumerate(station_indices):
            for ch in range(C):
                ax = axes[row, ch]
                time_steps = range(1, T + 1)

                ax.plot(time_steps, true[s_idx, :, st_idx, ch], "b-o",
                        label="Truth", markersize=4, linewidth=1.5)
                ax.plot(time_steps, pred[s_idx, :, st_idx, ch], "r--s",
                        label="Pred", markersize=4, linewidth=1.5)

                ax.set_xlabel("Step", fontsize=10)
                ax.set_ylabel("Value", fontsize=10)
                ax.set_title(f"Sample {s_idx}, Station {st_idx}, Ch {ch}", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f"{dataset_name} - Prediction vs Truth (Sample {s_idx})", fontsize=13)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"pred_vs_truth_sample{s_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"[可视化] 预测对比图已保存: {save_dir}")


def plot_per_step_metrics(
    per_step: Dict[str, list],
    save_path: str,
    title: str = "Per-Step Metrics",
):
    """
    绘制每个预测步的指标柱状图。

    Args:
        per_step: {'MAE': [step1, ...], 'RMSE': [...], 'MAPE': [...]}
        save_path: 保存路径
    """
    num_metrics = len(per_step)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for idx, (name, values) in enumerate(per_step.items()):
        ax = axes[idx]
        steps = range(1, len(values) + 1)
        ax.bar(steps, values, color=colors[idx % len(colors)], alpha=0.7)
        ax.set_xlabel("Prediction Step", fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] 每步指标图已保存: {save_path}")
