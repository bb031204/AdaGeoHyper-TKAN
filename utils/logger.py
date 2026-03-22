"""
Logger: 日志配置模块
=====================

统一配置项目日志, 支持控制台输出和文件输出。
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(
    name: str = "AdaGeoHyperTKAN",
    log_dir: str = None,
    level: int = logging.INFO,
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    """
    配置并返回 Logger。

    Args:
        name:    logger 名称
        log_dir: 日志文件保存目录
        level:   日志级别
        console: 是否输出到控制台
        log_file: 是否输出到文件

    Returns:
        配置好的 Logger
    """
    logger = logging.getLogger(name)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # 格式化器
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 Handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 Handler
    if log_file and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{timestamp}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_path}")

    # 同时设置 root logger 的子模块
    for module_name in ["models", "utils"]:
        sub_logger = logging.getLogger(module_name)
        sub_logger.setLevel(level)
        sub_logger.propagate = True
        # 使用 root logger 的 handlers
        if not sub_logger.handlers:
            sub_logger.parent = logger

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """获取已配置的 logger。"""
    if name is None:
        return logging.getLogger("AdaGeoHyperTKAN")
    return logging.getLogger(name)
