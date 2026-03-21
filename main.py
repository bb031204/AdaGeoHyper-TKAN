"""
Main entry for AdaGeoHyper-TKAN.
"""

import argparse
import os
import sys

from elements_settings import get_dataset_name_from_element, normalize_element_name

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaGeoHyper-TKAN runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "predict"],
        help="Run mode: all/train/predict",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for predict mode")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--element", type=str, default=None, help="Element override: Temperature/Cloud/Humidity/Wind")
    parser.add_argument("--dataset_selection", type=str, default=None, help="Alias of --element")

    parser.add_argument("--device", type=str, default=None, help="Device override: cuda/cpu/auto")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--num_stations", type=int, default=None, help="Num stations override")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Train sample ratio override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")

    return parser.parse_args()


def override_config(config: dict, args) -> dict:
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset

    selected_element = args.element or args.dataset_selection
    if selected_element:
        element_name = normalize_element_name(selected_element)
        config["data"]["element"] = element_name
        config["data"]["dataset_name"] = get_dataset_name_from_element(element_name)

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
    args = parse_args()

    from predict import predict as run_predict
    from train import load_config, train as run_train

    print("=" * 60)
    print("  AdaGeoHyper-TKAN: 多站点气象时空序列预测")
    print("  基于自适应地理邻域超图 + TKAN 时间建模")
    print("=" * 60)

    if args.mode == "predict":
        if args.output_dir is None:
            print("[错误] 预测模式需要 --output_dir")
            sys.exit(1)
        run_predict(args.output_dir, args.config if args.config != "config.yaml" else None)
        return

    config_path = os.path.join(project_root, args.config)
    if not os.path.exists(config_path):
        print(f"[错误] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config = override_config(config, args)

    import tempfile
    import yaml

    tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=project_root)
    yaml.dump(config, tmp_config, default_flow_style=False, allow_unicode=True)
    tmp_config_path = tmp_config.name
    tmp_config.close()

    try:
        if args.mode == "train":
            run_train(tmp_config_path)
        else:
            print("\n[Phase 1] 训练...")
            output_dir, best_val_loss = run_train(tmp_config_path)

            print(f"\n[Phase 2] 预测... (output_dir: {output_dir})")
            run_predict(output_dir)

            print("\n" + "=" * 60)
            print("  全流程完成!")
            print(f"  最优验证损失: {best_val_loss:.6f}")
            print(f"  输出目录: {output_dir}")
            print("=" * 60)
    finally:
        os.unlink(tmp_config_path)


if __name__ == "__main__":
    main()
