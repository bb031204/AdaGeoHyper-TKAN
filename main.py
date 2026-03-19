"""
Main: 统一启动入口
===================

串联训练与预测流程, 作为项目统一启动入口。

使用方式:
    # 完整流程 (训练 + 预测)
    python main.py --config config.yaml

    # 仅训练
    python main.py --config config.yaml --mode train

    # 仅预测 (需要指定训练输出目录)
    python main.py --mode predict --output_dir outputs/20240101_120000_temperature

    # 指定数据集快速运行
    python main.py --config config.yaml --dataset temperature

    # 指定设备
    python main.py --config config.yaml --device cuda
"""

import os
import sys
import argparse
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="AdaGeoHyper-TKAN: 多站点气象时空序列预测",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "train", "predict"],
        help="运行模式:\n"
             "  all     - 训练 + 预测 (默认)\n"
             "  train   - 仅训练\n"
             "  predict - 仅预测 (需要 --output_dir)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="预测模式下的训练输出目录"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="覆盖配置中的数据集名称 (temperature/cloud_cover/humidity/component_of_wind)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="覆盖配置中的设备 (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="覆盖配置中的批大小"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖配置中的训练轮数"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="覆盖配置中的学习率"
    )
    parser.add_argument(
        "--num_stations", type=int, default=None,
        help="覆盖配置中的站点数"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=None,
        help="覆盖配置中的样本抽样比例"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="覆盖配置中的随机种子"
    )

    return parser.parse_args()


def override_config(config: dict, args) -> dict:
    """根据命令行参数覆盖配置。"""
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset
    if args.device:
        config["training"]["device"] = args.device
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.num_stations is not None:
        config["data"]["num_stations"] = args.num_stations
    if args.sample_ratio is not None:
        config["data"]["sample_ratio"] = args.sample_ratio
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    return config


def main():
    """主入口函数。"""
    args = parse_args()

    # 导入训练和预测模块
    from train import train as run_train, load_config
    from predict import predict as run_predict

    print("=" * 60)
    print("  AdaGeoHyper-TKAN: 多站点气象时空序列预测")
    print("  基于自适应地理邻域超图 + TKAN 时间建模")
    print("=" * 60)

    if args.mode == "predict":
        # ---- 仅预测模式 ----
        if args.output_dir is None:
            print("[错误] 预测模式需要指定 --output_dir")
            sys.exit(1)
        run_predict(args.output_dir, args.config if args.config != "config.yaml" else None)

    elif args.mode == "train":
        # ---- 仅训练模式 ----
        config_path = os.path.join(project_root, args.config)
        if not os.path.exists(config_path):
            print(f"[错误] 配置文件不存在: {config_path}")
            sys.exit(1)

        # 加载并覆盖配置
        config = load_config(config_path)
        config = override_config(config, args)

        # 保存临时配置
        import yaml
        import tempfile
        tmp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=project_root
        )
        yaml.dump(config, tmp_config, default_flow_style=False, allow_unicode=True)
        tmp_config_path = tmp_config.name
        tmp_config.close()

        try:
            run_train(tmp_config_path)
        finally:
            os.unlink(tmp_config_path)

    else:
        # ---- 完整流程: 训练 + 预测 ----
        config_path = os.path.join(project_root, args.config)
        if not os.path.exists(config_path):
            print(f"[错误] 配置文件不存在: {config_path}")
            sys.exit(1)

        # 加载并覆盖配置
        config = load_config(config_path)
        config = override_config(config, args)

        # 保存临时配置
        import yaml
        import tempfile
        tmp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=project_root
        )
        yaml.dump(config, tmp_config, default_flow_style=False, allow_unicode=True)
        tmp_config_path = tmp_config.name
        tmp_config.close()

        try:
            # 训练
            print("\n[Phase 1] 训练...")
            output_dir, best_val_loss = run_train(tmp_config_path)

            # 预测
            print(f"\n[Phase 2] 预测... (output_dir: {output_dir})")
            run_predict(output_dir)

            print("\n" + "=" * 60)
            print(f"  全部完成!")
            print(f"  最佳验证损失: {best_val_loss:.6f}")
            print(f"  结果保存在: {output_dir}")
            print("=" * 60)
        finally:
            os.unlink(tmp_config_path)


if __name__ == "__main__":
    main()
