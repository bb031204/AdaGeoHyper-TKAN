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


REQUIRED_CONFIG_KEYS = {
    "data": ["dataset_name", "data_root", "input_len", "pred_len", "sample_ratio"],
    "training": ["device", "batch_size", "epochs", "learning_rate", "seed"],
    "hypergraph": ["cache_dir", "k_neighbors"],
    "output": ["output_dir"],
}


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


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _is_valid_device(device: str) -> bool:
    if device in {"auto", "cpu", "cuda"}:
        return True
    return device.startswith("cuda:")


def validate_cli_args(args):
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch_size 必须 > 0")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs 必须 > 0")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr 必须 > 0")
    if args.num_stations is not None and args.num_stations <= 0:
        raise ValueError("--num_stations 必须 > 0")
    if args.sample_ratio is not None and not (0 < args.sample_ratio <= 1):
        raise ValueError("--sample_ratio 必须在 (0, 1] 范围内")
    if args.device is not None and not _is_valid_device(args.device):
        raise ValueError("--device 仅支持 auto/cpu/cuda/cuda:N")


def validate_config(config: dict):
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            raise KeyError(f"配置缺失 section: {section}")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"配置缺失字段: {section}.{key}")

    training = config["training"]
    data = config["data"]

    if not _is_valid_device(str(training["device"])):
        raise ValueError("training.device 仅支持 auto/cpu/cuda/cuda:N")
    if int(training["batch_size"]) <= 0:
        raise ValueError("training.batch_size 必须 > 0")
    if int(training["epochs"]) <= 0:
        raise ValueError("training.epochs 必须 > 0")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate 必须 > 0")
    if int(data["input_len"]) <= 0 or int(data["pred_len"]) <= 0:
        raise ValueError("data.input_len 和 data.pred_len 必须 > 0")
    if float(data["sample_ratio"]) <= 0 or float(data["sample_ratio"]) > 1:
        raise ValueError("data.sample_ratio 必须在 (0, 1] 范围内")


def resolve_predict_config_path(output_dir: str, cli_config: str) -> str:
    if cli_config and cli_config != "config.yaml":
        config_path = _resolve_path(cli_config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"predict 配置文件不存在: {config_path}")
        return config_path

    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    if os.path.exists(snapshot_path):
        return snapshot_path

    default_config = _resolve_path("config.yaml")
    if os.path.exists(default_config):
        return default_config

    raise FileNotFoundError(
        f"未找到可用预测配置。已尝试: {snapshot_path} 与 {default_config}"
    )


def override_config(config: dict, args) -> dict:
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset

    selected_element = args.element or args.dataset_selection
    if selected_element:
        element_name = normalize_element_name(selected_element)
        config["data"]["element"] = element_name
        config["data"]["dataset_name"] = get_dataset_name_from_element(element_name)

    if args.device is not None:
        config["training"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.num_stations is not None:
        config["data"]["num_stations"] = args.num_stations
    if args.sample_ratio is not None:
        config["data"]["sample_ratio"] = args.sample_ratio
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    return config


def resolve_train_config_path(cli_config: str) -> str:
    """训练优先使用固定 outputs 快照配置，不再创建临时配置文件。"""
    preferred_snapshot = r"D:/bishe/AdaGeoHyper-TKAN/outputs/20260323_091932_temperature/config_snapshot.yaml"

    if cli_config and cli_config != "config.yaml":
        config_path = _resolve_path(cli_config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"训练配置文件不存在: {config_path}")
        return config_path

    if os.path.exists(preferred_snapshot):
        return preferred_snapshot

    default_config = _resolve_path("config.yaml")
    if os.path.exists(default_config):
        return default_config

    raise FileNotFoundError(
        f"未找到可用训练配置。已尝试: {preferred_snapshot} 与 {default_config}"
    )


def _has_cli_overrides(args) -> bool:
    return any([
        args.dataset is not None,
        args.element is not None,
        args.dataset_selection is not None,
        args.device is not None,
        args.batch_size is not None,
        args.epochs is not None,
        args.lr is not None,
        args.num_stations is not None,
        args.sample_ratio is not None,
        args.seed is not None,
    ])


def main():
    args = parse_args()

    from predict import predict as run_predict
    from train import load_config, train as run_train

    print("=" * 60)
    print("  AdaGeoHyper-TKAN: 多站点气象时空序列预测")
    print("  基于自适应地理邻域超图 + TKAN 时间建模")
    print("=" * 60)

    try:
        validate_cli_args(args)

        if args.mode == "predict":
            if args.output_dir is None:
                raise ValueError("预测模式需要 --output_dir")
            if not os.path.isdir(args.output_dir):
                raise FileNotFoundError(f"输出目录不存在: {args.output_dir}")

            predict_config_path = resolve_predict_config_path(args.output_dir, args.config)
            print(f"[配置] 预测使用配置: {predict_config_path}")
            run_predict(args.output_dir, predict_config_path)
            return

        config_path = resolve_train_config_path(args.config)
        print(f"[配置] 训练使用配置: {config_path}")

        config = load_config(config_path)
        validate_config(config)
        if _has_cli_overrides(args):
            raise ValueError(
                "当前模式已禁用临时配置文件。请直接修改配置文件后重试，"
                "或在 future 版本开启显式 override 模式。"
            )

        if args.mode == "train":
            run_train(config_path)
        else:
            print("\n[Phase 1] 训练...")
            output_dir, best_val_loss = run_train(config_path)

            predict_config_path = os.path.join(output_dir, "config_snapshot.yaml")
            print(f"\n[Phase 2] 预测... (output_dir: {output_dir})")
            print(f"[配置] 预测使用配置: {predict_config_path}")
            run_predict(output_dir, predict_config_path)

            print("\n" + "=" * 60)
            print("  全流程完成!")
            print(f"  最优验证损失: {best_val_loss:.6f}")
            print(f"  输出目录: {output_dir}")
            print("=" * 60)

    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"[配置错误] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[运行错误] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[未知错误] {e.__class__.__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
