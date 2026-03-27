"""
Train: 训练流程
================

完整训练流程:
1. 读取配置
2. 初始化数据集和 DataLoader
3. 检查或构建超图
4. 初始化模型
5. 训练 / 验证
6. 保存 checkpoint
7. 记录日志
8. 输出最佳模型

支持:
- 配置驱动训练
- checkpoint / resume
- 早停机制
- 学习率调度
- 完整日志
"""

import os
import sys
import time
import json
import shutil
import logging
import warnings
import threading
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*Not enough SMs.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# 将项目根目录加入路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Avoid UnicodeEncodeError on Windows GBK terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

from models.ada_geo_hyper_tkan import AdaGeoHyperTKAN
from utils.data_loader import (
    create_data_loaders,
    StandardScaler,
    save_preprocessing_artifact,
)
from utils.metrics import compute_metrics, compute_per_step_metrics, ELEMENT_METRIC_CONFIG
from utils.logger import setup_logger
from utils.visualization import plot_loss_curve, plot_metrics_curve
from elements_settings import resolve_element_from_config, get_element_settings

logger = logging.getLogger("AdaGeoHyperTKAN")


# ---- ANSI 终端颜色 ----
class C:
    """终端 ANSI 颜色码 (仅影响终端显示, 不影响日志文件)"""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列 (用于写入日志文件)"""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _log_file_only(msg: str):
    """仅写入日志文件, 不输出到控制台 (避免和 tqdm 冲突)"""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(logging.LogRecord(
                name=logger.name, level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=None, exc_info=None,
            ))


def tqdm_log(msg: str):
    """同时向 tqdm 终端(带颜色) 和 logger 文件(无颜色) 输出"""
    tqdm.write(msg)
    _log_file_only(_strip_ansi(msg))


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """获取计算设备。"""
    device_str = config["training"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"[设备] GPU: {gpu_name}, 显存: {gpu_mem:.1f}GB")
    else:
        logger.info(f"[设备] 使用 CPU")

    return device


def create_output_dir(config: dict) -> str:
    """创建输出目录 (以日期时间和数据集命名)。"""
    base_dir = config["output"]["output_dir"]
    dataset_name = config["data"]["dataset_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{timestamp}_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    return output_dir


def save_config_snapshot(config: dict, output_dir: str):
    """保存配置快照。"""
    import yaml
    config_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"[配置] 配置快照已保存: {config_path}")


def set_seed(seed: int):
    """设置随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    logger.info(f"[种子] 随机种子设置为 {seed} (cudnn.benchmark=True, TF32=high)")


def build_model(config: dict, input_feature_dim: int, target_dim: int, position_dim: int, device: torch.device) -> AdaGeoHyperTKAN:
    """根据配置构建模型。"""
    model_cfg = config["model"]
    data_cfg = config["data"]
    hyper_cfg = config["hypergraph"]

    # 解析 tkan_sub_kan_configs
    sub_kan_configs = model_cfg.get("tkan_sub_kan_configs", [None, 3])
    # YAML 中 null 会被解析为 None, int 保持原样

    scorer_mode = str(hyper_cfg.get("scorer_mode", "")).strip().lower()
    if scorer_mode in ("dynamic", "adaptive", "temporal"):
        use_state_summary_for_weights = True
    elif scorer_mode in ("static", "geo", "geography"):
        use_state_summary_for_weights = False
    else:
        use_state_summary_for_weights = bool(hyper_cfg.get("use_state_summary_for_weights", True))

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
        degree_clamp_min=hyper_cfg.get("degree_clamp_min", 1e-6),
        float32_norm=hyper_cfg.get("float32_norm", False),
        hypergraph_layers=model_cfg["hypergraph_layers"],
        dynamic_pruning=hyper_cfg.get("dynamic_pruning", {}).get("enabled", False),
        pruning_mode=hyper_cfg.get("dynamic_pruning", {}).get("mode", "top_p"),
        pruning_top_p=hyper_cfg.get("dynamic_pruning", {}).get("top_p", 0.8),
        pruning_threshold=hyper_cfg.get("dynamic_pruning", {}).get("threshold", 0.05),
        pruning_min_keep=hyper_cfg.get("dynamic_pruning", {}).get("min_keep", 2),
        use_state_summary_for_weights=use_state_summary_for_weights,
        fusion_dim=model_cfg["fusion_dim"],
        dropout=model_cfg["dropout"],
        pred_head_hidden=model_cfg["hidden_dim"] * 2,
        tkan_chunk_size=model_cfg.get("tkan_chunk_size", 0),
        use_gradient_checkpoint=model_cfg.get("use_gradient_checkpoint", False),
    )

    model = model.to(device)

    # 打印模型信息
    model_info = model.get_model_info()
    logger.info(f"[模型] 模型参数量: {model_info['total_params']:,} "
                f"(可训练: {model_info['trainable_params']:,})")

    return model


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    use_amp: bool = False,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    训练一个 epoch（只追踪 loss，不做逐 batch 指标收集以避免 GPU 同步开销）。

    Returns:
        dict: {'loss': ...}
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    grad_clip = config["training"].get("grad_clip", 0)
    accum_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    accum_steps = max(1, accum_steps)

    pbar = tqdm(
        train_loader,
        desc=f"    {C.CYAN}Train{C.RESET}",
        leave=False,
        ncols=120,
        bar_format="{l_bar}{bar:35}{r_bar}",
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            loss_for_backward = loss / accum_steps

            if use_amp and grad_scaler is not None:
                grad_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if grad_clip > 0:
                    if use_amp and grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if use_amp and grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"[OOM] CUDA 显存溢出! Batch {batch_idx}, "
                    f"x.shape={x.shape}, 请尝试:\n"
                    f"  1. 减小 batch_size (当前 {config['training']['batch_size']})\n"
                    f"  2. 减少 num_stations (当前 {config['data'].get('num_stations', 'all')})\n"
                    f"  3. 开启 use_amp: true (混合精度)\n"
                    f"  4. 设置 tkan_chunk_size (如 4096)"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            else:
                raise

        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    weather_scaler,
    target_weather_dim: int,
    element_name: str,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    验证/测试评估。

    Args:
        use_amp: 是否使用 AMP 混合精度

    Returns:
        dict: {'loss': ..., 'MAE': ..., 'RMSE': ..., 'MAPE': ...}
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_trues = []

    pbar = tqdm(
        data_loader,
        desc=f"    {C.MAGENTA}Val  {C.RESET}",
        leave=False,
        ncols=120,
        bar_format="{l_bar}{bar:35}{r_bar}",
    )

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        avg_loss = total_loss / num_batches
        pbar.set_postfix({"val_loss": f"{avg_loss:.4f}"})

        pred_np = pred.float().cpu().numpy()
        true_np = y.float().cpu().numpy()

        pred_weather = pred_np[..., :target_weather_dim]
        true_weather = true_np[..., :target_weather_dim]
        if weather_scaler is not None:
            pred_weather = weather_scaler.inverse_transform(pred_weather)
            true_weather = weather_scaler.inverse_transform(true_weather)

        all_preds.append(pred_weather)
        all_trues.append(true_weather)

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    metrics = compute_metrics(all_preds, all_trues, element_name=element_name)
    metrics["loss"] = avg_loss

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_mae: float,
    output_dir: str,
    is_best: bool = False,
    train_losses: list = None,
    val_losses: list = None,
    val_metrics_history: dict = None,
    no_improve_count: int = 0,
    config: dict = None,
    adaptive_events: list = None,
):
    """保存 checkpoint (含完整训练历史, 支持暂停恢复)。"""
    # torch.compile 包装后的模型, 取原始模块的 state_dict 以保持兼容性
    raw_model = getattr(model, "_orig_mod", model)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_mae": best_val_mae,
        # 训练历史 (用于暂停恢复)
        "train_losses": train_losses or [],
        "val_losses": val_losses or [],
        "val_metrics_history": val_metrics_history or {},
        "no_improve_count": no_improve_count,
        "adaptive_events": adaptive_events or [],
        "config": config,
    }

    # 保存最新 checkpoint
    ckpt_path = os.path.join(output_dir, "checkpoints", "latest.pth")
    torch.save(checkpoint, ckpt_path)

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
        torch.save(checkpoint, best_path)
        _log_file_only(f"[保存] 最佳模型已保存 (epoch {epoch}, MAE={best_val_mae:.6f})")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    加载 checkpoint, 返回恢复信息字典。

    Returns:
        dict: 包含 epoch, best_val_mae, train_losses, val_losses,
              val_metrics_history, no_improve_count
    """
    logger.info(f"[恢复] 从 checkpoint 恢复: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 与保存逻辑保持一致：若当前是 torch.compile 包装模型，权重加载到原始模块
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_val_mae = checkpoint.get("best_val_mae")
    if best_val_mae is None:
        # 向后兼容旧 checkpoint
        best_val_mae = checkpoint.get("best_val_loss", float("inf"))
    logger.info(f"[恢复] 恢复完成: epoch={epoch}, best_val_mae={best_val_mae:.6f}")

    return {
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "train_losses": checkpoint.get("train_losses", []),
        "val_losses": checkpoint.get("val_losses", []),
        "val_metrics_history": checkpoint.get(
            "val_metrics_history",
            {
                "MAE": [], "RMSE": [], "sMAPE": [], "WMAPE": [], "MAPE": [],
                "VectorMAE": [], "VectorRMSE": [],
            },
        ),
        "no_improve_count": checkpoint.get("no_improve_count", 0),
        "adaptive_events": checkpoint.get("adaptive_events", []),
    }


def check_pause_flag(output_dir: str) -> bool:
    """
    检查是否收到暂停信号。

    暂停信号通过 .pause 文件传递, 文件内容为目标暂停时间的时间戳。
    当前时间 >= 目标时间时返回 True。
    """
    pause_flag = os.path.join(output_dir, ".pause")
    if not os.path.exists(pause_flag):
        return False
    try:
        with open(pause_flag, "r") as f:
            target_time = float(f.read().strip())
        return time.time() >= target_time
    except (ValueError, OSError):
        return False


def clear_pause_flag(output_dir: str):
    """清除暂停标志文件。"""
    pause_flag = os.path.join(output_dir, ".pause")
    if os.path.exists(pause_flag):
        try:
            os.remove(pause_flag)
        except OSError:
            pass


def train(config_path: str = "config.yaml", resume_checkpoint: str = None,
          resume_output_dir: str = None):
    """
    完整训练流程入口。

    Args:
        config_path:       配置文件路径
        resume_checkpoint: 恢复训练的 checkpoint 路径 (由 pause_resume 系统传入)
        resume_output_dir: 恢复训练时复用的输出目录 (由 pause_resume 系统传入)
    """
    # ---- 1. 加载配置 ----
    config = load_config(config_path)
    element_name = resolve_element_from_config(config)
    element_settings = get_element_settings(element_name)
    config["data"]["element"] = element_name
    config_k_raw = config.get("hypergraph", {}).get("k_neighbors", None)
    element_k = int(element_settings["k"])
    dynamic_pruning_enabled = bool(config.get("hypergraph", {}).get("dynamic_pruning", {}).get("enabled", False))
    if dynamic_pruning_enabled and config_k_raw is not None:
        # 方案3启用时，k_neighbors 代表候选 K_max，优先尊重配置
        config["hypergraph"]["k_neighbors"] = int(config_k_raw)
    else:
        config["hypergraph"]["k_neighbors"] = element_k
    config["hypergraph"]["degree_clamp_min"] = float(element_settings.get("degree_clamp_min", 1e-6))
    config["hypergraph"]["float32_norm"] = bool(element_settings.get("float32_norm", False))
    dataset_name = config["data"]["dataset_name"]

    # ---- 2. 创建/复用输出目录 ----
    if resume_output_dir and os.path.isdir(resume_output_dir):
        output_dir = resume_output_dir
        # 确保子目录存在
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    else:
        output_dir = create_output_dir(config)

    # ---- 3. 设置日志 ----
    resume_log_path = None
    if resume_output_dir and os.path.isdir(resume_output_dir):
        try:
            log_candidates = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith("train_") and f.endswith(".log")
            ]
            if log_candidates:
                log_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                resume_log_path = log_candidates[0]
        except OSError:
            resume_log_path = None

    if resume_log_path is not None:
        setup_logger(
            "AdaGeoHyperTKAN",
            log_file_path=resume_log_path,
            file_mode="a",
        )
        logger.info(f"[恢复] 复用日志文件: {resume_log_path}")
    else:
        setup_logger("AdaGeoHyperTKAN", log_dir=output_dir)

    logger.info("=" * 60)
    logger.info("AdaGeoHyper-TKAN 训练启动")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    metric_plan = ELEMENT_METRIC_CONFIG.get(
        element_name,
        {"primary": ["MAE", "RMSE"], "secondary": ["sMAPE", "WMAPE"], "optional": ["MAPE"]},
    )
    if (not dynamic_pruning_enabled) and config_k_raw is not None and int(config_k_raw) != element_k:
        logger.warning(
            f"[K覆盖] config.hypergraph.k_neighbors={config_k_raw} 与 element={element_name} 预设不一致, "
            f"已覆盖为 {element_k}"
        )
    logger.info(
        f"要素: {element_name}, 生效k={config['hypergraph']['k_neighbors']}, "
        f"config_k={config_k_raw}, degree_clamp_min={config['hypergraph']['degree_clamp_min']}, "
        f"float32_norm={config['hypergraph']['float32_norm']}"
    )
    logger.info(f"输出目录: {output_dir}")

    # ---- 4. 设置种子 ----
    set_seed(config["training"]["seed"])

    # ---- 5. 设置设备 ----
    device = get_device(config)

    # ---- 6. 保存配置快照 ----
    save_config_snapshot(config, output_dir)

    # ---- 7. 加载数据 ----
    data_dir = os.path.join(config["data"]["data_root"], dataset_name)
    logger.info(f"[数据] 数据目录: {data_dir}")

    aggressive_cfg = config.get("aggressive_tuning", {})
    data_cfg = config.get("data", {})

    # 向后兼容：优先 aggressive_tuning；若缺失则回退旧字段，保证 resume 数据口径连续
    use_context_altitude = aggressive_cfg.get(
        "use_context_altitude",
        config.get("hypergraph", {}).get("use_context_altitude", True),
    )
    context_calendar_encoding = aggressive_cfg.get(
        "context_calendar_encoding",
        data_cfg.get("context_calendar_encoding", False),
    )
    robust_preprocess_cfg = aggressive_cfg.get(
        "robust_preprocess",
        data_cfg.get("robust_preprocess", {"enabled": False}),
    )

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
        context_calendar_encoding=context_calendar_encoding,
        use_context_altitude=use_context_altitude,
        element=element_name,
        robust_preprocess=robust_preprocess_cfg,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    weather_scaler = data["weather_scaler"]
    positions = data["positions"]
    position_dim = data["position_dim"]
    input_feature_dim = data.get("input_feature_dim", data.get("feature_dim"))
    target_dim = data.get("target_dim", data.get("feature_dim"))
    target_weather_dim = data["target_weather_dim"]
    preproc_artifact_path = os.path.join(output_dir, "preprocessing_artifact.pkl")
    save_preprocessing_artifact(
        preproc_artifact_path,
        station_indices=data["station_indices"],
        weather_scaler=data["weather_scaler"],
        context_scaler=data["context_scaler"],
        element_name=element_name,
        context_indices=data["context_indices"],
        context_feature_names=data["context_feature_names"],
        target_weather_dim=target_weather_dim,
    )

    # ---- 8. 构建模型 ----
    model = build_model(config, input_feature_dim, target_dim, position_dim, device)

    # ---- 9. 构建超图 ----
    cache_dir = os.path.join(project_root, config["hypergraph"]["cache_dir"])
    model.build_graph(
        positions=positions,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        use_cache=config["hypergraph"]["use_hypergraph_cache"],
        station_indices=data["station_indices"],
    )

    # ---- 10. 优化器与调度器 ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_epochs = config["training"].get("warmup_epochs", 3)
    sched_type = config["training"].get("scheduler", "none")
    scheduler = None

    if sched_type == "cosine":
        total_epochs = config["training"]["epochs"]
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
            logger.info(f"[Scheduler] Warmup({warmup_epochs}ep) + CosineAnnealing({total_epochs - warmup_epochs}ep)")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=1e-6
            )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("scheduler_step_size", 20),
            gamma=config["training"].get("scheduler_gamma", 0.5),
        )

    # ---- 10.5 torch.compile 编译加速 ----
    if config["training"].get("use_compile", True) and device.type == "cuda":
        _can_compile = True
        try:
            import triton  # noqa: F401
        except ImportError:
            _can_compile = False
            logger.warning("[Compile] Triton 未安装, 跳过 torch.compile "
                           "(pip install triton-windows 可启用)")
        if _can_compile:
            try:
                # 关键兜底: 编译后端失败时自动回退 eager，避免训练中断
                try:
                    import torch._dynamo as _dynamo  # type: ignore
                    _dynamo.config.suppress_errors = True
                except Exception:
                    pass

                model = torch.compile(model)
                logger.info("[Compile] torch.compile 已启用 (inductor backend, suppress_errors=True)")
            except Exception as e:
                logger.warning(f"[Compile] torch.compile 不可用, 跳过: {e}")

    # ---- 11. 损失函数（对齐 hyper_kan：mae/l1, mse/l2, huber） ----
    loss_type = str(config["training"].get("loss_type", "huber")).strip().lower()
    if loss_type in ("l1", "mae"):
        criterion = nn.L1Loss()
    elif loss_type in ("mse", "l2"):
        criterion = nn.MSELoss()
    elif loss_type == "huber":
        criterion = nn.HuberLoss(delta=config["training"].get("huber_delta", 1.0))
    else:
        raise ValueError(
            f"Unsupported training.loss_type='{loss_type}'. "
            "Supported: ['mae', 'l1', 'mse', 'l2', 'huber']."
        )
    logger.info(f"[Loss] {criterion.__class__.__name__}"
                + (f" (delta={criterion.delta})" if hasattr(criterion, "delta") else ""))

    # ---- 11.5 AMP 混合精度 ----
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("[AMP] 混合精度训练已启用 (float16)")
    else:
        logger.info("[AMP] 混合精度训练未启用 (float32)")

    # ---- 12. 恢复训练 ----
    start_epoch = 1
    best_val_mae = float("inf")
    no_improve_count = 0
    train_losses = []
    val_losses = []
    val_metrics_history = {
        "MAE": [], "RMSE": [], "sMAPE": [], "WMAPE": [], "MAPE": [],
        "VectorMAE": [], "VectorRMSE": [],
    }
    adaptive_events = []
    resumed_training = False

    # 优先使用 pause_resume 传入的 checkpoint, 其次使用 config 中的
    resume_path = resume_checkpoint or config["output"].get("resume_from_checkpoint")
    if resume_path and os.path.exists(resume_path):
        resumed_training = True
        restored = load_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )
        start_epoch = restored["epoch"] + 1  # 从下一个 epoch 开始
        best_val_mae = restored["best_val_mae"]
        train_losses = restored["train_losses"]
        val_losses = restored["val_losses"]
        val_metrics_history = restored["val_metrics_history"]
        no_improve_count = restored["no_improve_count"]
        adaptive_events = restored.get("adaptive_events", [])
        logger.info(f"[恢复] 已恢复训练历史: {len(train_losses)} epochs, "
                    f"best_val_mae={best_val_mae:.6f}, no_improve={no_improve_count}, "
                    f"adaptive_events={len(adaptive_events)}")

    # 清除可能残留的暂停标志
    preserve_pause_flag = os.environ.get("ADAGEO_PRESERVE_PAUSE_FLAG", "0") == "1"
    if preserve_pause_flag:
        logger.info("[暂停] preserve .pause flag on start (ADAGEO_PRESERVE_PAUSE_FLAG=1)")
    else:
        clear_pause_flag(output_dir)

    # ---- 13. 训练循环 ----
    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 15)
    use_early_stop = config["training"].get("use_early_stop", True)

    # 监控指标（New Best / Early-Stop 统一）
    monitor_metric = str(config["training"].get("monitor_metric", "MAE")).strip()
    monitor_mode = str(config["training"].get("monitor_mode", "min")).strip().lower()
    if monitor_mode not in ("min", "max"):
        raise ValueError(f"Unsupported training.monitor_mode='{monitor_mode}', expected 'min' or 'max'.")

    # C) 稳健早停：可选 EMA 平滑监控值
    early_stop_use_ema = bool(config["training"].get("early_stop_use_ema", True))
    early_stop_ema_alpha = float(config["training"].get("early_stop_ema_alpha", 0.3))
    early_stop_min_delta = float(config["training"].get("early_stop_min_delta", 0.0))
    early_stop_ema = None
    best_monitor_value = float("inf") if monitor_mode == "min" else float("-inf")

    # Resume 兼容：从历史重建监控器状态，避免恢复后首轮被误判为 New Best
    metric_hist = val_metrics_history.get(monitor_metric, []) if isinstance(val_metrics_history, dict) else []
    if metric_hist:
        if early_stop_use_ema:
            ema = None
            for v in metric_hist:
                fv = float(v)
                ema = fv if ema is None else (early_stop_ema_alpha * fv + (1.0 - early_stop_ema_alpha) * ema)
            early_stop_ema = ema
            best_monitor_value = float(ema)
        else:
            if monitor_mode == "min":
                best_monitor_value = float(min(metric_hist))
            else:
                best_monitor_value = float(max(metric_hist))

    # D) 动态K贴底反馈：仅在动态裁剪开启时才有意义（与 dynamic_pruning 联动）
    pruning_cfg = config.get("hypergraph", {}).get("dynamic_pruning", {})
    dynamic_pruning_enabled = bool(pruning_cfg.get("enabled", False))
    dynamic_k_feedback_requested = bool(config["training"].get("dynamic_k_feedback_enabled", True))
    dynamic_k_feedback_enabled = dynamic_pruning_enabled and dynamic_k_feedback_requested
    if dynamic_k_feedback_requested and not dynamic_pruning_enabled:
        logger.info("[DynamicK-Feedback] auto-disabled because hypergraph.dynamic_pruning.enabled=False")

    dynamic_k_floor_patience = int(config["training"].get("dynamic_k_floor_patience", 3))
    dynamic_k_floor_margin = float(config["training"].get("dynamic_k_floor_margin", 0.2))
    dynamic_k_top_p_step = float(config["training"].get("dynamic_k_top_p_step", 0.01))
    dynamic_k_top_p_max = float(config["training"].get("dynamic_k_top_p_max", 0.80))
    dynamic_k_floor_count = 0

    # 平台期自适应调参（不改模型结构）
    adaptive_enabled = bool(config["training"].get("adaptive_tuning_enabled", False))
    adaptive_patience = int(config["training"].get("adaptive_tuning_patience", 10))
    adaptive_loss_type = str(config["training"].get("adaptive_tuning_loss_type", "l1")).strip().lower()
    adaptive_lr_drop = float(config["training"].get("adaptive_tuning_lr_drop", 0.0005))
    adaptive_lr_floor = 0.002
    adaptive_applied = bool(adaptive_events)
    if adaptive_enabled:
        logger.info(
            "[Adaptive] enabled=True, "
            f"patience={adaptive_patience}, target_loss={adaptive_loss_type}, "
            f"lr_drop={adaptive_lr_drop:.6f}, lr_floor={adaptive_lr_floor:.4f}, "
            f"already_applied={adaptive_applied}"
        )
    resume_best_guard_value = best_monitor_value if resumed_training else None

    logger.info(
        "[EarlyStop] "
        f"monitor_metric={monitor_metric}, mode={monitor_mode}, "
        f"use_ema={early_stop_use_ema}, ema_alpha={early_stop_ema_alpha:.2f}, min_delta={early_stop_min_delta:.6f}, "
        f"best_monitor_init={best_monitor_value:.6f}"
    )
    logger.info(
        "[DynamicK-Feedback] "
        f"enabled={dynamic_k_feedback_enabled}, floor_patience={dynamic_k_floor_patience}, "
        f"margin={dynamic_k_floor_margin:.2f}, top_p_step={dynamic_k_top_p_step:.3f}, top_p_max={dynamic_k_top_p_max:.2f}"
    )

    # 显存预估
    if device.type == "cuda":
        num_st = data["num_stations"]
        bs = config["training"]["batch_size"]
        est_mem_gb = (bs * num_st * 64 * 12 * 4 * 6) / (1024**3)  # 粗估
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[显存] 预估峰值: ~{est_mem_gb:.1f}GB, GPU 可用: {gpu_mem_gb:.1f}GB")
        if est_mem_gb > gpu_mem_gb * 0.9 and not use_amp:
            logger.warning(
                f"[显存] ⚠️ 预估显存接近/超过 GPU 上限! 强烈建议:\n"
                f"  1. config.yaml 中设置 use_amp: true\n"
                f"  2. 减小 batch_size (当前 {bs})\n"
                f"  3. 减少 num_stations (当前 {num_st})"
            )

    W = 72
    tqdm_log(f"\n  {C.CYAN}{'━'*W}{C.RESET}")
    tqdm_log(f"  {C.CYAN}{C.BOLD}{'TRAINING':^{W}}{C.RESET}")
    tqdm_log(f"  {C.CYAN}{'━'*W}{C.RESET}")
    tqdm_log(
        f"    Element: {C.WHITE}{element_name}{C.RESET}  │  "
        f"Effective K: {C.WHITE}{config['hypergraph']['k_neighbors']}{C.RESET}  │  "
        f"Config K: {C.DIM}{config_k_raw}{C.RESET}"
    )
    tqdm_log(f"    Epochs: {C.WHITE}{epochs}{C.RESET}  │  Batch: {C.WHITE}{config['training']['batch_size']}{C.RESET}"
             f"  │  Early Stop: {C.WHITE}{'On' if use_early_stop else 'Off'}{C.RESET}"
             f" (patience={patience})"
             f"  │  AMP: {C.WHITE}{'On' if use_amp else 'Off'}{C.RESET}")
    tqdm_log(f"  {C.CYAN}{'─'*W}{C.RESET}")
    tqdm_log(
        f"    Metrics: primary={C.WHITE}{metric_plan['primary']}{C.RESET}"
        f"  secondary={C.DIM}{metric_plan['secondary']}{C.RESET}"
        f"  optional={C.DIM}{metric_plan['optional']}{C.RESET}"
    )

    # ---- 编译 warmup: 前向 + 反向全量编译, 避免训练循环内阻塞 ----
    _is_compiled = hasattr(model, "_orig_mod")
    _compile_active = _is_compiled
    # Windows 下 warmup 体感时间很长，默认关闭；可在 config.training.compile_warmup 显式开启
    _enable_compile_warmup = bool(config["training"].get("compile_warmup", os.name != "nt"))

    def _fallback_disable_compile(reason: str):
        nonlocal model, _compile_active
        if hasattr(model, "_orig_mod"):
            model = getattr(model, "_orig_mod")
            _compile_active = False
            logger.warning(f"[Compile] 自动回退到 eager 模式: {reason}")

    if _is_compiled and device.type == "cuda" and _enable_compile_warmup:
        tqdm_log(f"    {C.YELLOW}⏳ torch.compile 首次编译 (前向+反向, 仅一次, 约 5~10 分钟)...{C.RESET}")
        _warmup_start = time.time()

        # 保存原始状态 (warmup 会修改权重, 完成后恢复)
        _raw_model = getattr(model, "_orig_mod", model)
        _saved_state = {k: v.clone() for k, v in _raw_model.state_dict().items()}

        _sample_x, _sample_y = next(iter(train_loader))
        _sample_x = _sample_x.to(device, non_blocking=True)
        _sample_y = _sample_y.to(device, non_blocking=True)
        _bs = int(config["training"]["batch_size"])
        _train_tail_bs = len(train_loader.dataset) % _bs
        _val_tail_bs = len(val_loader.dataset) % _bs
        _warmup_steps = 3
        # tail-shape 预编译开关：默认 Linux/macOS 开启，Windows 默认关闭，可显式配置打开
        _enable_tail_warmup = bool(config["training"].get("compile_warmup_tail_shapes", os.name != "nt"))
        if _enable_tail_warmup and 0 < _train_tail_bs < _sample_x.shape[0]:
            _warmup_steps += 1
        if _enable_tail_warmup and 0 < _val_tail_bs < _sample_x.shape[0]:
            _warmup_steps += 1
        if _enable_tail_warmup and os.name == "nt":
            tqdm_log(f"    {C.YELLOW}↻ 已开启 tail-shape 预编译（Windows）: train_tail_bs={_train_tail_bs}, val_tail_bs={_val_tail_bs}{C.RESET}")
        # 让 warmup 进度条在“阶段内”也推进，避免长阶段一直显示 0/x
        _units_per_stage = 20
        _warmup_total_units = _warmup_steps * _units_per_stage
        _warmup_progress_units = 0
        _warmup_stage_idx = 0

        _warmup_bar = tqdm(
            total=_warmup_total_units,
            desc=f"    {C.YELLOW}Warmup{C.RESET}",
            leave=False,
            ncols=120,
            bar_format="{l_bar}{bar:35}| {n_fmt}/{total_fmt} [{elapsed}]",
        )

        def _run_warmup_stage(stage_name: str, fn):
            nonlocal _warmup_progress_units, _warmup_stage_idx
            stage_start = time.time()
            stop_event = threading.Event()
            stage_unit_start = _warmup_stage_idx * _units_per_stage
            stage_unit_end = stage_unit_start + _units_per_stage

            def _heartbeat():
                nonlocal _warmup_progress_units
                while not stop_event.wait(15):
                    stage_elapsed = time.time() - stage_start
                    total_elapsed = time.time() - _warmup_start
                    # 每 15s 推进 1 格，最多推进到本阶段倒数第1格
                    soft_units = min(_units_per_stage - 1, int(stage_elapsed // 15))
                    target_units = stage_unit_start + soft_units
                    if target_units > _warmup_progress_units:
                        _warmup_bar.update(target_units - _warmup_progress_units)
                        _warmup_progress_units = target_units
                    tqdm.write(
                        f"    [Warmup] {stage_name} 进行中... "
                        f"stage={stage_elapsed:.0f}s, total={total_elapsed:.0f}s"
                    )

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()
            try:
                fn()
            finally:
                stop_event.set()
                hb_thread.join(timeout=0.2)

            stage_elapsed = time.time() - stage_start
            total_elapsed = time.time() - _warmup_start
            _warmup_bar.set_postfix(
                {"stage": stage_name, "last": f"{stage_elapsed:.1f}s", "total": f"{total_elapsed:.1f}s"}
            )
            if _warmup_progress_units < stage_unit_end:
                _warmup_bar.update(stage_unit_end - _warmup_progress_units)
                _warmup_progress_units = stage_unit_end
            _warmup_stage_idx += 1
            tqdm_log(f"    [Warmup] {stage_name} 完成, 用时 {stage_elapsed:.1f}s")

        _warmup_failed = False
        _warmup_err = None
        try:
            # 1) 编译 train 前向 + 反向图
            def _stage_train_full():
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    warmup_pred = model(_sample_x)
                    warmup_loss = criterion(warmup_pred, _sample_y)
                if use_amp and grad_scaler is not None:
                    grad_scaler.scale(warmup_loss).backward()
                else:
                    warmup_loss.backward()
                del warmup_pred, warmup_loss

            _run_warmup_stage("train fwd+bwd", _stage_train_full)

            # 2) 编译 eval 前向图
            def _stage_val_full():
                model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    model(_sample_x)

            _run_warmup_stage("val fwd", _stage_val_full)

            # 2.5) 预编译尾 batch 形状，避免首个 epoch 在最后一个 batch 临时编译卡顿
            # train: 需要前向+反向图
            if _enable_tail_warmup and 0 < _train_tail_bs < _sample_x.shape[0]:
                def _stage_train_tail():
                    tail_x = _sample_x[:_train_tail_bs]
                    tail_y = _sample_y[:_train_tail_bs]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        tail_pred = model(tail_x)
                        tail_loss = criterion(tail_pred, tail_y)
                    if use_amp and grad_scaler is not None:
                        grad_scaler.scale(tail_loss).backward()
                    else:
                        tail_loss.backward()
                    del tail_pred, tail_loss, tail_x, tail_y

                _run_warmup_stage(f"train tail bs={_train_tail_bs}", _stage_train_tail)

            # val: 只需要前向图
            if _enable_tail_warmup and 0 < _val_tail_bs < _sample_x.shape[0]:
                def _stage_val_tail():
                    val_tail_x = _sample_x[:_val_tail_bs]
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                        model(val_tail_x)
                    del val_tail_x

                _run_warmup_stage(f"val tail bs={_val_tail_bs}", _stage_val_tail)

            # 3) 恢复原始模型权重和优化器 (warmup 不影响真正训练)
            def _stage_restore():
                nonlocal grad_scaler
                _raw_model.load_state_dict(_saved_state)
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
                if use_amp and grad_scaler is not None:
                    grad_scaler = torch.amp.GradScaler("cuda", enabled=True)

            _run_warmup_stage("restore state", _stage_restore)
        except Exception as e:
            _warmup_failed = True
            _warmup_err = e
        finally:
            _warmup_bar.close()

        if _warmup_failed:
            _fallback_disable_compile(f"warmup 失败: {_warmup_err}")
            # 回退后重置优化器状态，确保后续训练干净开始
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            if use_amp:
                grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
            tqdm_log(f"    {C.YELLOW}⚠ 编译warmup失败，已自动切换 eager 并继续训练{C.RESET}")
        else:
            _warmup_sec = time.time() - _warmup_start
            tqdm_log(f"    {C.GREEN}✓ 编译完成 ({_warmup_sec:.0f}s), 后续 epoch 无需重复编译{C.RESET}")

        del _sample_x, _sample_y, _saved_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif _is_compiled and device.type == "cuda" and not _enable_compile_warmup:
        tqdm_log(f"    {C.YELLOW}⏭ 已跳过 compile warmup（Windows 默认）; 首个 batch 可能出现一次性编译等待{C.RESET}")
    elif device.type == "cuda":
        tqdm_log(f"    {C.YELLOW}⏳ 首个 batch 需 CUDA warmup, 可能等待 1~3 分钟...{C.RESET}")
    tqdm_log("")

    training_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # 训练
        try:
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, config,
                use_amp=use_amp, grad_scaler=grad_scaler,
            )
        except RuntimeError as e:
            err_msg = str(e)
            compile_like_err = (
                ("compiler" in err_msg.lower())
                or ("torchinductor" in err_msg.lower())
                or ("torch._inductor" in err_msg.lower())
                or ("dynamo" in err_msg.lower())
                or ("cl is not found" in err_msg.lower())
            )
            if _compile_active and compile_like_err:
                _fallback_disable_compile(f"epoch {epoch} 训练阶段编译异常: {err_msg[:200]}")
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
                if use_amp:
                    grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
                tqdm_log(f"    {C.YELLOW}⚠ 已自动关闭 compile，本轮将以 eager 重试{C.RESET}")
                train_metrics = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    epoch, config,
                    use_amp=use_amp, grad_scaler=grad_scaler,
                )
            else:
                raise
        train_losses.append(train_metrics["loss"])

        # 验证
        try:
            val_metrics = evaluate(
                model, val_loader, criterion, device, weather_scaler, target_weather_dim,
                element_name=element_name,
                use_amp=use_amp,
            )
        except RuntimeError as e:
            err_msg = str(e)
            compile_like_err = (
                ("compiler" in err_msg.lower())
                or ("torchinductor" in err_msg.lower())
                or ("torch._inductor" in err_msg.lower())
                or ("dynamo" in err_msg.lower())
                or ("cl is not found" in err_msg.lower())
            )
            if _compile_active and compile_like_err:
                _fallback_disable_compile(f"epoch {epoch} 验证阶段编译异常: {err_msg[:200]}")
                if use_amp:
                    grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
                tqdm_log(f"    {C.YELLOW}⚠ 已自动关闭 compile，验证阶段改用 eager 重试{C.RESET}")
                val_metrics = evaluate(
                    model, val_loader, criterion, device, weather_scaler, target_weather_dim,
                    element_name=element_name,
                    use_amp=use_amp,
                )
            else:
                raise
        val_losses.append(val_metrics["loss"])

        tracked_metric_keys = ["MAE", "RMSE", "sMAPE", "WMAPE", "MAPE", "VectorMAE", "VectorRMSE"]
        for key in tracked_metric_keys:
            if key in val_metrics_history and key in val_metrics:
                val_metrics_history[key].append(val_metrics[key])

        # 学习率调度
        lr_before_step = float(optimizer.param_groups[0]["lr"])
        if scheduler:
            scheduler.step()
        lr_after_step = float(optimizer.param_groups[0]["lr"])

        epoch_time = time.time() - epoch_start

        # 监控值：按 monitor_metric 统一 New Best / Early-Stop
        val_loss = float(val_metrics["loss"])
        if monitor_metric not in val_metrics:
            raise ValueError(
                f"training.monitor_metric='{monitor_metric}' not found in val metrics. "
                f"Available keys: {list(val_metrics.keys())}"
            )
        metric_value = float(val_metrics[monitor_metric])

        # C) 稳健早停监控值：EMA 或原始监控值
        if early_stop_use_ema:
            if early_stop_ema is None:
                early_stop_ema = metric_value
            else:
                early_stop_ema = early_stop_ema_alpha * metric_value + (1.0 - early_stop_ema_alpha) * early_stop_ema
            monitor_value = float(early_stop_ema)
        else:
            monitor_value = metric_value

        if monitor_mode == "min":
            monitor_improved = monitor_value < (best_monitor_value - early_stop_min_delta)
        else:
            monitor_improved = monitor_value > (best_monitor_value + early_stop_min_delta)

        if monitor_improved:
            prev_best_monitor = best_monitor_value
            best_monitor_value = monitor_value
            no_improve_count = 0

            # 恢复训练保护：只有真正超过恢复前历史最佳才允许覆盖 best_model.pth
            if resume_best_guard_value is not None:
                if monitor_mode == "min":
                    is_best = monitor_value < (resume_best_guard_value - early_stop_min_delta)
                else:
                    is_best = monitor_value > (resume_best_guard_value + early_stop_min_delta)
            else:
                is_best = True

            if (monitor_mode == "min" and prev_best_monitor < float("inf")) or (monitor_mode == "max" and prev_best_monitor > float("-inf")):
                if monitor_mode == "min":
                    monitor_delta = prev_best_monitor - monitor_value
                else:
                    monitor_delta = monitor_value - prev_best_monitor
                best_tag = f"{C.GREEN}{C.BOLD}New Best [{monitor_metric} -{monitor_delta:.4f}]{C.RESET}"
                best_tag_plain = f"New Best [{monitor_metric} -{monitor_delta:.4f}]"
            else:
                best_tag = f"{C.GREEN}{C.BOLD}New Best{C.RESET}"
                best_tag_plain = "New Best"
        else:
            is_best = False
            no_improve_count += 1
            best_tag = ""
            best_tag_plain = ""

        # best_val_mae 作为唯一基准：严格用 MAE 记录最优
        if "MAE" in val_metrics and float(val_metrics["MAE"]) < best_val_mae:
            best_val_mae = float(val_metrics["MAE"])

        # 平台期触发一次自适应：降学习率 + 切换损失函数
        adaptive_triggered = False
        if adaptive_enabled and (not adaptive_applied) and no_improve_count >= adaptive_patience:
            old_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(adaptive_lr_floor, old_lr - adaptive_lr_drop)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            switched_loss_name = ""
            if adaptive_loss_type in ("l1", "mae"):
                criterion = nn.L1Loss()
                switched_loss_name = "L1Loss"
            elif adaptive_loss_type == "huber":
                criterion = nn.HuberLoss(delta=config["training"].get("huber_delta", 1.0))
                switched_loss_name = "HuberLoss"
            elif adaptive_loss_type in ("mse", "l2"):
                criterion = nn.MSELoss()
                switched_loss_name = "MSELoss"
            else:
                raise ValueError(
                    f"Unsupported training.adaptive_tuning_loss_type='{adaptive_loss_type}'. "
                    "Supported: ['mae', 'l1', 'mse', 'l2', 'huber']."
                )

            adaptive_applied = True
            adaptive_triggered = True
            no_improve_count = 0
            adaptive_event = {
                "epoch": int(epoch),
                "old_lr": float(old_lr),
                "new_lr": float(new_lr),
                "new_loss": str(switched_loss_name),
                "trigger_no_improve": int(adaptive_patience),
                "val_loss_at_trigger": float(val_metrics["loss"]),
            }
            adaptive_events.append(adaptive_event)
            adaptive_msg_plain = (
                "[Adaptive Applied] Plateau detected -> applied once: "
                f"epoch={epoch}, loss={switched_loss_name}, lr {old_lr:.6f} -> {new_lr:.6f}"
            )
            logger.info(adaptive_msg_plain)
            _log_file_only(adaptive_msg_plain)
            tqdm_log(
                f"    {C.YELLOW}{C.BOLD}⚠ Adaptive Applied{C.RESET} "
                f"{C.YELLOW}loss={switched_loss_name}, lr {old_lr:.6f} -> {new_lr:.6f}{C.RESET}"
            )

        # ---- 格式化终端输出 ----
        elapsed = time.time() - training_start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        avg_sec_per_epoch = elapsed / (epoch - start_epoch + 1)
        remaining_epochs = epochs - epoch
        eta_sec = remaining_epochs * avg_sec_per_epoch
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

        line1 = (
            f"  {C.CYAN}{C.BOLD}Epoch {epoch:3d}/{epochs}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  Train Loss: {C.WHITE}{train_metrics['loss']:.6f}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  Val Loss: {C.YELLOW}{val_metrics['loss']:.6f}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  {C.DIM}{epoch_time:.0f}s{C.RESET}"
            f"  {best_tag}"
        )
        current_loss_name = criterion.__class__.__name__.replace("Loss", "").lower()
        cosine_enabled = (sched_type == "cosine")
        metric_lines = [
            f"          {C.DIM}│{C.RESET}  MAE: {C.WHITE}{val_metrics['MAE']:.4f}{C.RESET}",
            f"          {C.DIM}│{C.RESET}  RMSE: {C.WHITE}{val_metrics['RMSE']:.4f}{C.RESET}",
            f"          {C.DIM}│{C.RESET}  sMAPE: {C.WHITE}{val_metrics.get('sMAPE', 0.0):.2f}%{C.RESET}",
            f"          {C.DIM}│{C.RESET}  WMAPE: {C.WHITE}{val_metrics.get('WMAPE', 0.0):.2f}%{C.RESET}",
            f"          {C.DIM}│{C.RESET}  MAPE(ref): {C.DIM}{val_metrics.get('MAPE', 0.0):.2f}%{C.RESET}",
            f"          {C.DIM}│{C.RESET}  LR(step): {C.DIM}{lr_before_step:.1e} -> {lr_after_step:.1e}{C.RESET}",
            f"          {C.DIM}│{C.RESET}  LossType: {C.WHITE}{current_loss_name}{C.RESET}",
            f"          {C.DIM}│{C.RESET}  CosineLR: {C.WHITE}{'On' if cosine_enabled else 'Off'}{C.RESET}",
            f"          {C.DIM}│{C.RESET}  Time: {C.DIM}[{elapsed_str}<{eta_str}]{C.RESET}",
        ]
        if "VectorMAE" in val_metrics:
            metric_lines.insert(4, f"          {C.DIM}│{C.RESET}  VecMAE: {C.WHITE}{val_metrics['VectorMAE']:.4f}{C.RESET}")
        if "VectorRMSE" in val_metrics:
            metric_lines.insert(5, f"          {C.DIM}│{C.RESET}  VecRMSE: {C.WHITE}{val_metrics['VectorRMSE']:.4f}{C.RESET}")

        raw_model = getattr(model, "_orig_mod", model)
        pruning_stats = raw_model.get_last_pruning_stats() if hasattr(raw_model, "get_last_pruning_stats") else None
        dynamic_k_feedback_msg = ""
        if pruning_stats is not None:
            # D) 动态K贴底反馈：连续贴近 min_keep 时，轻微提升 top_p
            if dynamic_k_feedback_enabled:
                k_mean = float(pruning_stats.get("k_mean", 0.0))
                k_floor = float(getattr(raw_model, "pruning_min_keep", 1)) + dynamic_k_floor_margin
                if k_mean <= k_floor:
                    dynamic_k_floor_count += 1
                else:
                    dynamic_k_floor_count = 0

                spatial_module = getattr(raw_model, "spatial_module", None)
                if dynamic_k_floor_count >= dynamic_k_floor_patience and spatial_module is not None and hasattr(spatial_module, "pruning_top_p"):
                    old_top_p = float(spatial_module.pruning_top_p)
                    new_top_p = min(dynamic_k_top_p_max, old_top_p + dynamic_k_top_p_step)
                    if new_top_p > old_top_p + 1e-12:
                        spatial_module.pruning_top_p = new_top_p
                        dynamic_k_feedback_msg = f"top_p {old_top_p:.2f}->{new_top_p:.2f}"
                        logger.info(
                            "[DynamicK-Feedback] floor detected -> "
                            f"k_mean={k_mean:.2f}, min_keep={getattr(spatial_module, 'pruning_min_keep', 1)}, "
                            f"raise top_p {old_top_p:.2f} -> {new_top_p:.2f}"
                        )
                        _log_file_only(
                            f"[DynamicK-Feedback] floor detected, raise top_p {old_top_p:.2f}->{new_top_p:.2f}"
                        )
                    dynamic_k_floor_count = 0

            mode_label = pruning_stats.get("mode", "fixed")
            p_cfg = pruning_stats.get("top_p", 0.0)
            t_cfg = pruning_stats.get("threshold", 0.0)
            line3 = (
                f"          "
                f"  {C.DIM}│{C.RESET}"
                f"  {C.BOLD}{C.CYAN}Dynamic-K[{mode_label}]{C.RESET} "
                f"range={C.WHITE}{pruning_stats['k_min']}-{pruning_stats['k_max']}{C.RESET} "
                f"mean={C.WHITE}{pruning_stats['k_mean']:.2f}{C.RESET} "
                f"median={C.WHITE}{pruning_stats['k_median']:.2f}{C.RESET} "
                f"keep={C.YELLOW}{pruning_stats['keep_ratio']*100:.1f}%{C.RESET} "
                f"cand={C.DIM}{pruning_stats.get('candidate_k', 0)}{C.RESET} "
                f"cfg(top_p={p_cfg:.2f}, tau={t_cfg:.3f})"
            )
        else:
            line3 = None

        tqdm.write(f"\n  {C.DIM}{'─' * W}{C.RESET}")
        tqdm.write(line1)
        for mline in metric_lines:
            tqdm.write(mline)
        if line3 is not None:
            tqdm.write(line3)
        if adaptive_triggered:
            tqdm.write(
                f"          {C.DIM}│{C.RESET}  {C.YELLOW}Adaptive Applied{C.RESET}: "
                f"loss->{criterion.__class__.__name__}, "
                f"lr->{optimizer.param_groups[0]['lr']:.6f}"
            )
        if dynamic_k_feedback_msg:
            tqdm.write(
                f"          {C.DIM}│{C.RESET}  {C.YELLOW}DynamicK Feedback{C.RESET}: "
                f"{dynamic_k_feedback_msg}"
            )

        # 文件日志 (无颜色, 紧凑一行, 不输出到终端)
        _log_file_only(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_metrics['loss']:.6f} | "
            f"Val: {val_metrics['loss']:.6f} | "
            f"MAE: {val_metrics['MAE']:.4f} | "
            f"RMSE: {val_metrics['RMSE']:.4f} | "
            f"sMAPE: {val_metrics.get('sMAPE', 0.0):.2f}% | "
            f"WMAPE: {val_metrics.get('WMAPE', 0.0):.2f}% | "
            f"MAPE(ref): {val_metrics.get('MAPE', 0.0):.2f}% | "
            f"VecMAE: {val_metrics.get('VectorMAE', 0.0):.4f} | "
            f"VecRMSE: {val_metrics.get('VectorRMSE', 0.0):.4f} | "
            f"LR(step): {lr_before_step:.2e}->{lr_after_step:.2e} | "
            f"LossType: {current_loss_name} | "
            f"CosineLR: {'On' if cosine_enabled else 'Off'} | "
            f"{epoch_time:.0f}s | "
            f"{best_tag_plain}"
        )

        # 保存 checkpoint (含完整训练历史)
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_mae, output_dir, is_best,
            train_losses=train_losses, val_losses=val_losses,
            val_metrics_history=val_metrics_history,
            no_improve_count=no_improve_count, config=config,
            adaptive_events=adaptive_events,
        )

        # 暂停检查 (来自 pause_resume/pause.py 的信号)
        if check_pause_flag(output_dir):
            clear_pause_flag(output_dir)
            tqdm.write(f"\n  {C.YELLOW}{'━' * W}{C.RESET}")
            tqdm.write(f"  {C.YELLOW}{C.BOLD}{'PAUSED':^{W}}{C.RESET}")
            tqdm.write(f"  {C.YELLOW}{'─' * W}{C.RESET}")
            tqdm.write(f"    收到暂停信号, 已在 epoch {epoch} 结束后安全暂停")
            tqdm.write(f"    {C.DIM}Checkpoint: {os.path.join(output_dir, 'checkpoints', 'latest.pth')}{C.RESET}")
            tqdm.write(f"    {C.CYAN}恢复训练: python pause_resume/resume.py{C.RESET}")
            tqdm.write(f"  {C.YELLOW}{'━' * W}{C.RESET}")
            _log_file_only(f"[暂停] 在 epoch {epoch} 安全暂停, checkpoint 已保存")
            return output_dir, best_val_mae

        # 早停检查
        if use_early_stop and no_improve_count >= patience:
            tqdm.write(f"\n  {C.DIM}{'─' * W}{C.RESET}")
            tqdm.write(f"  {C.RED}{C.BOLD}  Early Stop  {C.RESET}"
                       f"  连续 {patience} 个 epoch 无改善, 停止训练")
            _log_file_only(f"[早停] 连续 {patience} 个 epoch 无改善, 停止训练")
            break

    # ---- 14. 训练结束 ----
    total_time = time.time() - training_start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    tqdm.write(f"\n  {C.GREEN}{'━' * W}{C.RESET}")
    tqdm.write(f"  {C.GREEN}{C.BOLD}{'TRAINING COMPLETE':^{W}}{C.RESET}")
    tqdm.write(f"  {C.GREEN}{'─' * W}{C.RESET}")
    tqdm.write(f"    Best MAE: {C.GREEN}{C.BOLD}{best_val_mae:.6f}{C.RESET}"
               f"    Total Time: {C.CYAN}{total_time_str}{C.RESET}"
               f"    Epochs: {C.WHITE}{epoch}/{epochs}{C.RESET}")
    tqdm.write(f"  {C.GREEN}{'━' * W}{C.RESET}")
    _log_file_only(f"训练完成! 最佳 MAE: {best_val_mae:.6f}, 总耗时: {total_time_str}")

    # ---- 15. 可视化 ----
    fig_dir = os.path.join(output_dir, "figures")

    plot_loss_curve(
        train_losses, val_losses,
        save_path=os.path.join(fig_dir, "loss_curve.png"),
        title=f"Loss Curve - {dataset_name}",
    )

    plot_metrics_curve(
        val_metrics_history,
        save_path=os.path.join(fig_dir, "val_metrics.png"),
        title=f"Validation Metrics - {dataset_name}",
    )

    # ---- 16. 保存训练历史 ----
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics_history,
        "best_val_mae": best_val_mae,
        "total_epochs": epoch,
        "config": config,
    }
    history_path = os.path.join(output_dir, "training_history.json")
    # 清理不可序列化的项
    history_serializable = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics_history,
        "best_val_mae": best_val_mae,
        "total_epochs": epoch,
        "dataset_name": dataset_name,
    }
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    logger.info(f"[保存] 训练历史已保存: {history_path}")

    return output_dir, best_val_mae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AdaGeoHyper-TKAN 训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径 (由 pause_resume 系统使用)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="恢复训练时复用的输出目录 (由 pause_resume 系统使用)")
    args = parser.parse_args()

    train(args.config, resume_checkpoint=args.resume, resume_output_dir=args.output_dir)
