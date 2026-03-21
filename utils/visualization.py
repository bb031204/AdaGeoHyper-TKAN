"""
Visualization utilities.
"""

import os
import logging
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training & Validation Loss",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

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
        ax.set_title(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)

        best_idx = np.argmin(values)
        ax.scatter(best_idx + 1, values[best_idx], color="red", s=50, zorder=5)
        ax.annotate(f"{values[best_idx]:.4f}", xy=(best_idx + 1, values[best_idx]), fontsize=9, color="red")

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
    Save one stitched figure (grid) for sampled samples/stations.
    """
    os.makedirs(save_dir, exist_ok=True)
    N, T, N_s, C = pred.shape

    # Prefer hard examples: choose high-error samples/stations for clearer comparison.
    ch = 0  # visualize primary target channel
    mae_sn = np.mean(np.abs(pred[..., ch] - true[..., ch]), axis=1)  # [N, N_s]
    sample_scores = np.max(mae_sn, axis=1)  # [N]
    sample_indices = np.argsort(sample_scores)[-min(num_samples, N):]
    sample_indices = np.sort(sample_indices)

    station_scores = np.mean(mae_sn[sample_indices, :], axis=0)  # [N_s]
    station_indices = np.argsort(station_scores)[-min(num_stations, N_s):]
    station_indices = np.sort(station_indices)

    n_rows = len(sample_indices)
    n_cols = len(station_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    time_steps = np.arange(1, T + 1)
    overall_mae = float(np.mean(np.abs(pred[..., ch] - true[..., ch])))

    for r, s_idx in enumerate(sample_indices):
        for c, st_idx in enumerate(station_indices):
            ax = axes[r, c]
            y_true = true[s_idx, :, st_idx, ch]
            y_pred = pred[s_idx, :, st_idx, ch]
            mae = float(np.mean(np.abs(y_pred - y_true)))

            # Error as area between prediction and ground truth curves.
            ax.fill_between(
                time_steps,
                y_true,
                y_pred,
                color="#9E9E9E",
                alpha=0.28,
                label="Error Area",
            )
            ax.plot(time_steps, y_true, color="#1565C0", marker="o", markersize=2.8, linewidth=1.6, label="Ground Truth")
            ax.plot(time_steps, y_pred, color="#D84315", linestyle="--", marker="s", markersize=2.4, linewidth=1.4, label="Prediction")

            ax.set_xlabel("Forecast Hour", fontsize=8)
            ax.set_ylabel("Temperature (°C)", fontsize=8)
            ax.set_title(f"Sample {s_idx}, Station {st_idx}  |  MAE={mae:.2f}°C", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6, loc="upper left", framealpha=0.8)

    fig.suptitle(
        f"{dataset_name} Temperature Prediction  (Overall MAE={overall_mae:.3f}°C)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(save_dir, "pred_vs_truth_grid.png")
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] 预测对比拼图已保存: {save_path}")


def plot_per_step_metrics(
    per_step: Dict[str, list],
    save_path: str,
    title: str = "Per-Step Metrics",
):
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
    logger.info(f"[可视化] 分步指标图已保存: {save_path}")
