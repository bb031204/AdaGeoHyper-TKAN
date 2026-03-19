"""
Predict: 预测与评估流程
========================

功能:
1. 加载最佳模型
2. 对测试集进行预测
3. 保存预测结果
4. 输出评估指标 (MAE, RMSE, MAPE)
5. 执行可视化
6. 生成 summary
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ada_geo_hyper_tkan import AdaGeoHyperTKAN
from utils.data_loader import create_data_loaders
from utils.metrics import compute_metrics, compute_per_step_metrics
from utils.logger import setup_logger
from utils.visualization import (
    plot_prediction_vs_truth,
    plot_per_step_metrics,
)

logger = logging.getLogger("AdaGeoHyperTKAN")


def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_best_model(
    config: dict,
    checkpoint_path: str,
    positions: np.ndarray,
    position_dim: int,
    feature_dim: int,
    device: torch.device,
    station_indices=None,
) -> AdaGeoHyperTKAN:
    """
    加载最佳模型。

    Args:
        config:          配置字典
        checkpoint_path: checkpoint 文件路径
        positions:       站点位置
        position_dim:    位置维度
        feature_dim:     特征维度
        device:          设备
        station_indices: 站点索引

    Returns:
        加载好的模型
    """
    model_cfg = config["model"]
    data_cfg = config["data"]
    hyper_cfg = config["hypergraph"]

    sub_kan_configs = model_cfg.get("tkan_sub_kan_configs", [None, 3])

    model = AdaGeoHyperTKAN(
        input_dim=feature_dim,
        output_dim=feature_dim,
        hidden_dim=model_cfg["hidden_dim"],
        tkan_hidden_dim=model_cfg["tkan_hidden_dim"],
        tkan_layers=model_cfg["tkan_layers"],
        tkan_sub_kan_configs=sub_kan_configs,
        input_len=data_cfg["input_len"],
        pred_len=data_cfg["pred_len"],
        position_dim=position_dim,
        k_neighbors=hyper_cfg["k_neighbors"],
        lambda_geo=hyper_cfg["lambda_geo"],
        lambda_alt=hyper_cfg["lambda_alt"],
        summary_pool=hyper_cfg["summary_pool"],
        scorer_hidden_dim=hyper_cfg["scorer_hidden_dim"],
        hypergraph_layers=model_cfg["hypergraph_layers"],
        fusion_dim=model_cfg["fusion_dim"],
        dropout=model_cfg["dropout"],
        pred_head_hidden=model_cfg["hidden_dim"] * 2,
    )

    model = model.to(device)

    # 构建超图
    cache_dir = os.path.join(project_root, config["hypergraph"]["cache_dir"])
    model.build_graph(
        positions=positions,
        cache_dir=cache_dir,
        dataset_name=config["data"]["dataset_name"],
        use_cache=config["hypergraph"]["use_hypergraph_cache"],
        station_indices=station_indices,
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"[预测] 模型权重加载完成: {checkpoint_path}")
    logger.info(f"[预测] Checkpoint epoch: {checkpoint['epoch']}, "
                f"val_loss: {checkpoint['best_val_loss']:.6f}")

    model.eval()
    return model


@torch.no_grad()
def predict_on_test(
    model: nn.Module,
    test_loader,
    device: torch.device,
    scaler_list,
):
    """
    在测试集上进行预测。

    Returns:
        predictions: [N_samples, T, N_stations, C] (反标准化后)
        ground_truth: [N_samples, T, N_stations, C] (反标准化后)
    """
    model.eval()
    all_preds = []
    all_trues = []

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)  # [B, T, N, C]

        pred_np = pred.cpu().numpy()
        true_np = y.cpu().numpy()

        # 反标准化
        for ch in range(pred_np.shape[-1]):
            pred_np[..., ch] = scaler_list[ch].inverse_transform(pred_np[..., ch])
            true_np[..., ch] = scaler_list[ch].inverse_transform(true_np[..., ch])

        all_preds.append(pred_np)
        all_trues.append(true_np)

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_trues, axis=0)

    return predictions, ground_truth


def generate_summary(
    metrics: dict,
    per_step: dict,
    config: dict,
    output_dir: str,
    model_info: dict,
):
    """生成实验 summary。"""
    summary = {
        "experiment": {
            "dataset": config["data"]["dataset_name"],
            "model": "AdaGeoHyperTKAN",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_len": config["data"]["input_len"],
            "pred_len": config["data"]["pred_len"],
        },
        "model_info": model_info,
        "test_metrics": metrics,
        "per_step_metrics": per_step,
        "config": {
            "hidden_dim": config["model"]["hidden_dim"],
            "tkan_layers": config["model"]["tkan_layers"],
            "k_neighbors": config["hypergraph"]["k_neighbors"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
        },
    }

    # 保存 JSON
    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存可读文本
    txt_path = os.path.join(output_dir, "test_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AdaGeoHyper-TKAN 测试结果 Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据集: {config['data']['dataset_name']}\n")
        f.write(f"时间: {summary['experiment']['timestamp']}\n")
        f.write(f"输入步数: {config['data']['input_len']}\n")
        f.write(f"预测步数: {config['data']['pred_len']}\n\n")

        f.write("--- 整体指标 ---\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write("\n--- 每步指标 ---\n")
        for metric_name, values in per_step.items():
            f.write(f"  {metric_name}: {[f'{v:.4f}' for v in values]}\n")

        f.write("\n--- 模型信息 ---\n")
        for k, v in model_info.items():
            f.write(f"  {k}: {v}\n")

    logger.info(f"[Summary] 已保存: {summary_path}")
    logger.info(f"[Summary] 文本版: {txt_path}")

    return summary


def predict(output_dir: str, config_path: str = None):
    """
    完整预测流程。

    Args:
        output_dir:  训练输出目录 (包含 checkpoints/, config_snapshot.yaml)
        config_path: 配置文件路径 (如果为 None, 使用 output_dir 中的快照)
    """
    # ---- 加载配置 ----
    if config_path is None:
        config_path = os.path.join(output_dir, "config_snapshot.yaml")
    config = load_config(config_path)
    dataset_name = config["data"]["dataset_name"]

    # ---- 设置日志 ----
    setup_logger("AdaGeoHyperTKAN", log_dir=output_dir)
    logger.info("=" * 60)
    logger.info("AdaGeoHyper-TKAN 测试预测")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"输出目录: {output_dir}")

    # ---- 设备 ----
    device_str = config["training"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"[设备] {device}")

    # ---- 数据 ----
    np.random.seed(config["training"]["seed"])
    data_dir = os.path.join(config["data"]["data_root"], dataset_name)
    data = create_data_loaders(
        data_dir=data_dir,
        batch_size=config["training"]["batch_size"],
        num_stations=config["data"]["num_stations"],
        sample_ratio=config["data"]["sample_ratio"],
        val_sample_ratio=config["data"].get("val_sample_ratio", 1.0),
        test_sample_ratio=config["data"].get("test_sample_ratio", 1.0),
        seed=config["training"]["seed"],
    )

    test_loader = data["test_loader"]
    scaler_list = data["scaler"]
    positions = data["positions"]
    position_dim = data["position_dim"]
    feature_dim = data["feature_dim"]

    # ---- 加载模型 ----
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"[错误] 最佳模型不存在: {checkpoint_path}")
        return

    model = load_best_model(
        config, checkpoint_path, positions, position_dim, feature_dim, device,
        station_indices=data["station_indices"],
    )

    model_info = model.get_model_info()

    # ---- 预测 ----
    logger.info("[预测] 开始测试集预测...")
    predictions, ground_truth = predict_on_test(model, test_loader, device, scaler_list)
    logger.info(f"[预测] 预测完成: pred={predictions.shape}, truth={ground_truth.shape}")

    # ---- 评估指标 ----
    metrics = compute_metrics(predictions, ground_truth)
    per_step = compute_per_step_metrics(predictions, ground_truth, num_steps=config["data"]["pred_len"])

    logger.info("\n" + "=" * 40)
    logger.info("测试集评估结果:")
    logger.info("=" * 40)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    logger.info("\n每步指标:")
    for metric_name, values in per_step.items():
        values_str = ", ".join([f"{v:.4f}" for v in values])
        logger.info(f"  {metric_name}: [{values_str}]")

    # ---- 保存预测结果 ----
    result_path = os.path.join(output_dir, "predictions.npz")
    np.savez(
        result_path,
        predictions=predictions,
        ground_truth=ground_truth,
    )
    logger.info(f"[保存] 预测结果已保存: {result_path}")

    # ---- 保存指标 ----
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"overall": metrics, "per_step": per_step}, f, indent=2)
    logger.info(f"[保存] 指标已保存: {metrics_path}")

    # ---- 可视化 ----
    fig_dir = os.path.join(output_dir, "figures")

    plot_prediction_vs_truth(
        predictions, ground_truth,
        save_dir=os.path.join(fig_dir, "pred_vs_truth"),
        num_samples=5,
        num_stations=4,
        dataset_name=dataset_name,
    )

    plot_per_step_metrics(
        per_step,
        save_path=os.path.join(fig_dir, "per_step_metrics.png"),
        title=f"Per-Step Metrics - {dataset_name}",
    )

    # ---- Summary ----
    generate_summary(metrics, per_step, config, output_dir, model_info)

    logger.info("\n" + "=" * 60)
    logger.info("预测与评估完成!")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AdaGeoHyper-TKAN 预测")
    parser.add_argument("--output_dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (可选)")
    args = parser.parse_args()

    predict(args.output_dir, args.config)
