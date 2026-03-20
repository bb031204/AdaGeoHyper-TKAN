"""
Predict: 预测与评估流程
========================

功能：
1. 加载最佳模型
2. 在测试集上推理
3. 计算评估指标
4. 保存结果文件与可视化图
"""

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ada_geo_hyper_tkan import AdaGeoHyperTKAN
from utils.data_loader import create_data_loaders
from utils.logger import setup_logger
from utils.metrics import compute_metrics, compute_per_step_metrics
from utils.visualization import plot_per_step_metrics, plot_prediction_vs_truth

logger = logging.getLogger("AdaGeoHyperTKAN")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_best_model(
    config: dict,
    checkpoint_path: str,
    positions: np.ndarray,
    position_dim: int,
    input_feature_dim: int,
    target_dim: int,
    device: torch.device,
    station_indices=None,
) -> AdaGeoHyperTKAN:
    """构建模型并加载最佳 checkpoint 权重。"""
    model_cfg = config["model"]
    data_cfg = config["data"]
    hyper_cfg = config["hypergraph"]

    sub_kan_configs = model_cfg.get("tkan_sub_kan_configs", [None, 3])

    model = AdaGeoHyperTKAN(
        input_dim=input_feature_dim,
        output_dim=target_dim,
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
        tkan_chunk_size=model_cfg.get("tkan_chunk_size", 0),
        use_gradient_checkpoint=False,
    ).to(device)

    cache_dir = os.path.join(project_root, config["hypergraph"]["cache_dir"])
    model.build_graph(
        positions=positions,
        cache_dir=cache_dir,
        dataset_name=config["data"]["dataset_name"],
        use_cache=config["hypergraph"]["use_hypergraph_cache"],
        station_indices=station_indices,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"[预测] 模型权重加载完成: {checkpoint_path}")
    logger.info(
        f"[预测] Checkpoint epoch: {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.6f}"
    )
    model.eval()
    return model


@torch.no_grad()
def predict_on_test(
    model: nn.Module,
    test_loader,
    device: torch.device,
    scaler_list,
    use_amp: bool = False,
):
    """在测试集上执行预测并反标准化。"""
    model.eval()
    all_preds = []
    all_trues = []

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)

        pred_np = pred.float().cpu().numpy()
        true_np = y.float().cpu().numpy()
        for ch in range(pred_np.shape[-1]):
            pred_np[..., ch] = scaler_list[ch].inverse_transform(pred_np[..., ch])
            true_np[..., ch] = scaler_list[ch].inverse_transform(true_np[..., ch])

        all_preds.append(pred_np)
        all_trues.append(true_np)

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_trues, axis=0)
    return predictions, ground_truth


def generate_summary(metrics: dict, per_step: dict, config: dict, output_dir: str, model_info: dict):
    """保存测试摘要（JSON + TXT）。"""
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
    }

    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

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
    """预测主流程入口。"""
    if config_path is None:
        config_path = os.path.join(output_dir, "config_snapshot.yaml")
    config = load_config(config_path)
    dataset_name = config["data"]["dataset_name"]

    setup_logger("AdaGeoHyperTKAN", log_dir=output_dir)
    logger.info("=" * 60)
    logger.info("AdaGeoHyper-TKAN 测试预测")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"输出目录: {output_dir}")

    device_str = config["training"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"[设备] {device}")

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
        include_context=config["data"].get("include_context", False),
        context_features=config["data"].get("context_features", {}),
        use_context_altitude=config["hypergraph"].get("use_context_altitude", True),
    )

    test_loader = data["test_loader"]
    scaler_list = data["scaler"]
    positions = data["positions"]
    position_dim = data["position_dim"]
    input_feature_dim = data.get("input_feature_dim", data.get("feature_dim"))
    target_dim = data.get("target_dim", data.get("feature_dim"))

    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"[错误] 最佳模型不存在: {checkpoint_path}")
        return

    model = load_best_model(
        config,
        checkpoint_path,
        positions,
        position_dim,
        input_feature_dim,
        target_dim,
        device,
        station_indices=data["station_indices"],
    )
    model_info = model.get_model_info()

    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    if use_amp:
        logger.info("[AMP] 混合精度推理已启用")
    logger.info("[预测] 开始测试集预测...")
    predictions, ground_truth = predict_on_test(model, test_loader, device, scaler_list, use_amp=use_amp)
    logger.info(f"[预测] 预测完成: pred={predictions.shape}, truth={ground_truth.shape}")

    metrics = compute_metrics(predictions, ground_truth)
    per_step = compute_per_step_metrics(predictions, ground_truth, num_steps=config["data"]["pred_len"])

    logger.info("\n" + "=" * 40)
    logger.info("测试集评估结果")
    logger.info("=" * 40)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("\n每步指标:")
    for metric_name, values in per_step.items():
        logger.info(f"  {metric_name}: [{', '.join([f'{v:.4f}' for v in values])}]")

    result_path = os.path.join(output_dir, "predictions.npz")
    np.savez(result_path, predictions=predictions, ground_truth=ground_truth)
    logger.info(f"[保存] 预测结果已保存: {result_path}")

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"overall": metrics, "per_step": per_step}, f, indent=2)
    logger.info(f"[保存] 指标已保存: {metrics_path}")

    fig_dir = os.path.join(output_dir, "figures")
    plot_prediction_vs_truth(
        predictions,
        ground_truth,
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

    generate_summary(metrics, per_step, config, output_dir, model_info)

    logger.info("\n" + "=" * 60)
    logger.info("预测与评估完成")
    logger.info("=" * 60)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AdaGeoHyper-TKAN 预测")
    parser.add_argument("--output_dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")
    args = parser.parse_args()

    predict(args.output_dir, args.config)

