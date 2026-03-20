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
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

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
from utils.data_loader import create_data_loaders, StandardScaler
from utils.metrics import compute_metrics, compute_per_step_metrics
from utils.logger import setup_logger
from utils.visualization import plot_loss_curve, plot_metrics_curve

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
    logger.info(f"[种子] 随机种子设置为 {seed} (cudnn.benchmark=True)")


def build_model(config: dict, input_feature_dim: int, target_dim: int, position_dim: int, device: torch.device) -> AdaGeoHyperTKAN:
    """根据配置构建模型。"""
    model_cfg = config["model"]
    data_cfg = config["data"]
    hyper_cfg = config["hypergraph"]

    # 解析 tkan_sub_kan_configs
    sub_kan_configs = model_cfg.get("tkan_sub_kan_configs", [None, 3])
    # YAML 中 null 会被解析为 None, int 保持原样

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
    scaler_list,
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

    pbar = tqdm(
        train_loader,
        desc=f"    {C.CYAN}Train{C.RESET}",
        leave=False,
        ncols=120,
        bar_format="{l_bar}{bar:35}{r_bar}",
    )

    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            if use_amp and grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                if grad_clip > 0:
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

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
    scaler_list,
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

        for ch in range(pred_np.shape[-1]):
            pred_np[..., ch] = scaler_list[ch].inverse_transform(pred_np[..., ch])
            true_np[..., ch] = scaler_list[ch].inverse_transform(true_np[..., ch])

        all_preds.append(pred_np)
        all_trues.append(true_np)

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    metrics = compute_metrics(all_preds, all_trues)
    metrics["loss"] = avg_loss

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    is_best: bool = False,
    train_losses: list = None,
    val_losses: list = None,
    val_metrics_history: dict = None,
    no_improve_count: int = 0,
    config: dict = None,
):
    """保存 checkpoint (含完整训练历史, 支持暂停恢复)。"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        # 训练历史 (用于暂停恢复)
        "train_losses": train_losses or [],
        "val_losses": val_losses or [],
        "val_metrics_history": val_metrics_history or {},
        "no_improve_count": no_improve_count,
        "config": config,
    }

    # 保存最新 checkpoint
    ckpt_path = os.path.join(output_dir, "checkpoints", "latest.pth")
    torch.save(checkpoint, ckpt_path)

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
        torch.save(checkpoint, best_path)
        _log_file_only(f"[保存] 最佳模型已保存 (epoch {epoch}, val_loss={best_val_loss:.6f})")


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
        dict: 包含 epoch, best_val_loss, train_losses, val_losses,
              val_metrics_history, no_improve_count
    """
    logger.info(f"[恢复] 从 checkpoint 恢复: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    logger.info(f"[恢复] 恢复完成: epoch={epoch}, best_val_loss={best_val_loss:.6f}")

    return {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "train_losses": checkpoint.get("train_losses", []),
        "val_losses": checkpoint.get("val_losses", []),
        "val_metrics_history": checkpoint.get("val_metrics_history",
                                              {"MAE": [], "RMSE": [], "MAPE": []}),
        "no_improve_count": checkpoint.get("no_improve_count", 0),
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
    setup_logger("AdaGeoHyperTKAN", log_dir=output_dir)
    logger.info("=" * 60)
    logger.info("AdaGeoHyper-TKAN 训练启动")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
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

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    scaler_list = data["scaler"]
    positions = data["positions"]
    position_dim = data["position_dim"]
    input_feature_dim = data.get("input_feature_dim", data.get("feature_dim"))
    target_dim = data.get("target_dim", data.get("feature_dim"))

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

    scheduler = None
    sched_type = config["training"].get("scheduler", "none")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("scheduler_step_size", 20),
            gamma=config["training"].get("scheduler_gamma", 0.5),
        )

    # ---- 11. 损失函数 ----
    criterion = nn.MSELoss()

    # ---- 11.5 AMP 混合精度 ----
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("[AMP] 混合精度训练已启用 (float16)")
    else:
        logger.info("[AMP] 混合精度训练未启用 (float32)")

    # ---- 12. 恢复训练 ----
    start_epoch = 1
    best_val_loss = float("inf")
    no_improve_count = 0
    train_losses = []
    val_losses = []
    val_metrics_history = {"MAE": [], "RMSE": [], "MAPE": []}

    # 优先使用 pause_resume 传入的 checkpoint, 其次使用 config 中的
    resume_path = resume_checkpoint or config["output"].get("resume_from_checkpoint")
    if resume_path and os.path.exists(resume_path):
        restored = load_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )
        start_epoch = restored["epoch"] + 1  # 从下一个 epoch 开始
        best_val_loss = restored["best_val_loss"]
        train_losses = restored["train_losses"]
        val_losses = restored["val_losses"]
        val_metrics_history = restored["val_metrics_history"]
        no_improve_count = restored["no_improve_count"]
        logger.info(f"[恢复] 已恢复训练历史: {len(train_losses)} epochs, "
                    f"best_val_loss={best_val_loss:.6f}, no_improve={no_improve_count}")

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
    tqdm_log(f"    Epochs: {C.WHITE}{epochs}{C.RESET}  │  Batch: {C.WHITE}{config['training']['batch_size']}{C.RESET}"
             f"  │  Early Stop: {C.WHITE}{'On' if use_early_stop else 'Off'}{C.RESET}"
             f" (patience={patience})"
             f"  │  AMP: {C.WHITE}{'On' if use_amp else 'Off'}{C.RESET}")
    tqdm_log(f"  {C.CYAN}{'─'*W}{C.RESET}")
    if device.type == "cuda":
        tqdm_log(f"    {C.YELLOW}⏳ 首个 batch 需 CUDA warmup, 可能等待 1~3 分钟...{C.RESET}")
    tqdm_log("")

    training_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler_list, epoch, config,
            use_amp=use_amp, grad_scaler=grad_scaler,
        )
        train_losses.append(train_metrics["loss"])

        # 验证
        val_metrics = evaluate(
            model, val_loader, criterion, device, scaler_list,
            use_amp=use_amp,
        )
        val_losses.append(val_metrics["loss"])

        for key in ["MAE", "RMSE", "MAPE"]:
            val_metrics_history[key].append(val_metrics[key])

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # 判断是否为最佳
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            prev_best = best_val_loss
            best_val_loss = val_metrics["loss"]
            no_improve_count = 0
            if prev_best < float("inf"):
                loss_delta = prev_best - val_metrics["loss"]
                best_tag = f"{C.GREEN}{C.BOLD}New Best{C.RESET} ({C.GREEN}-{loss_delta:.6f}{C.RESET})"
                best_tag_plain = f"New Best (-{loss_delta:.6f})"
            else:
                best_tag = f"{C.GREEN}{C.BOLD}New Best{C.RESET}"
                best_tag_plain = "New Best"
        else:
            no_improve_count += 1
            best_tag = ""
            best_tag_plain = ""

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
            f"  Train {C.WHITE}{train_metrics['loss']:.6f}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  Val {C.YELLOW}{val_metrics['loss']:.6f}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  {C.DIM}{epoch_time:.0f}s{C.RESET}"
            f"  {best_tag}"
        )
        line2 = (
            f"          "
            f"  {C.DIM}│{C.RESET}"
            f"  MAE {C.WHITE}{val_metrics['MAE']:.4f}{C.RESET}"
            f"  RMSE {C.WHITE}{val_metrics['RMSE']:.4f}{C.RESET}"
            f"  MAPE {C.WHITE}{val_metrics['MAPE']:.2f}%{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  LR {C.DIM}{current_lr:.1e}{C.RESET}"
            f"  {C.DIM}│{C.RESET}"
            f"  {C.DIM}[{elapsed_str}<{eta_str}]{C.RESET}"
        )

        tqdm.write(f"\n  {C.DIM}{'─' * W}{C.RESET}")
        tqdm.write(line1)
        tqdm.write(line2)

        # 文件日志 (无颜色, 紧凑一行, 不输出到终端)
        _log_file_only(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_metrics['loss']:.6f} | "
            f"Val: {val_metrics['loss']:.6f} | "
            f"MAE: {val_metrics['MAE']:.4f} | "
            f"RMSE: {val_metrics['RMSE']:.4f} | "
            f"MAPE: {val_metrics['MAPE']:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.0f}s | "
            f"{best_tag_plain}"
        )

        # 保存 checkpoint (含完整训练历史)
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_loss, output_dir, is_best,
            train_losses=train_losses, val_losses=val_losses,
            val_metrics_history=val_metrics_history,
            no_improve_count=no_improve_count, config=config,
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
            return output_dir, best_val_loss

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
    tqdm.write(f"    Best Val Loss: {C.GREEN}{C.BOLD}{best_val_loss:.6f}{C.RESET}"
               f"    Total Time: {C.CYAN}{total_time_str}{C.RESET}"
               f"    Epochs: {C.WHITE}{epoch}/{epochs}{C.RESET}")
    tqdm.write(f"  {C.GREEN}{'━' * W}{C.RESET}")
    _log_file_only(f"训练完成! 最佳 Val Loss: {best_val_loss:.6f}, 总耗时: {total_time_str}")

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
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "config": config,
    }
    history_path = os.path.join(output_dir, "training_history.json")
    # 清理不可序列化的项
    history_serializable = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics_history,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "dataset_name": dataset_name,
    }
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    logger.info(f"[保存] 训练历史已保存: {history_path}")

    return output_dir, best_val_loss


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
