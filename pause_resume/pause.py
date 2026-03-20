"""
pause.py - 暂停正在运行的训练

使用方法：
    # 立即暂停（默认）
    python pause_resume/pause.py

    # 100分钟后暂停
    python pause_resume/pause.py --pause-time 100

功能：
    1. 向训练器发送暂停信号（创建 .pause 标志文件）
    2. 训练器会在完成当前 epoch 后保存 checkpoint 并退出
    3. 训练状态完全保存，可通过 resume.py 恢复

参数：
    --pause-time: 距离现在多少分钟后暂停（默认0表示立即暂停）

注意：
    - 暂停会在当前 epoch 结束时生效，不会中断正在进行的 epoch
    - 训练器会自动保存 checkpoint（含完整训练历史）
    - 暂停后电脑可以进入休眠/关机以节省能耗
"""

import os
import sys
import time
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_latest_run(outputs_dir: str = "outputs"):
    """查找最新的训练目录"""
    full_outputs_dir = os.path.join(project_root, outputs_dir)

    if not os.path.exists(full_outputs_dir):
        return None

    # 获取所有符合条件的目录
    run_dirs = []
    for item in os.listdir(full_outputs_dir):
        item_path = os.path.join(full_outputs_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含 checkpoints 目录
            if os.path.exists(os.path.join(item_path, "checkpoints")):
                run_dirs.append((item_path, os.path.getmtime(item_path)))

    if not run_dirs:
        return None

    # 按修改时间排序，返回最新的
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    return run_dirs[0][0]


def create_pause_flag(out_dir: str, pause_minutes: float = 0) -> bool:
    """
    创建暂停标志文件

    Args:
        out_dir: 输出目录
        pause_minutes: 距离现在多少分钟后暂停（0表示立即）
    """
    pause_flag = os.path.join(out_dir, ".pause")

    # 检查是否已存在
    if os.path.exists(pause_flag):
        print(f"警告: 暂停标志已存在 ({pause_flag})")
        print("训练可能已经收到暂停信号")
        response = input("是否要覆盖？ (y/n): ").lower()
        if response != "y":
            print("已取消")
            return False

    try:
        if pause_minutes <= 0:
            # 立即暂停：写入当前时间戳
            content = str(time.time())
        else:
            # 定时暂停：写入目标时间戳
            target_time = time.time() + pause_minutes * 60
            content = str(target_time)

        with open(pause_flag, "w") as f:
            f.write(content)

        if pause_minutes > 0:
            hours = int(pause_minutes // 60)
            mins = int(pause_minutes % 60)
            if hours > 0:
                time_str = f"{hours}小时{mins}分钟" if mins > 0 else f"{hours}小时"
            else:
                time_str = f"{mins}分钟"
            print(f"  将在 {time_str} 后暂停训练")
            print(f"  预计暂停时间: {time.ctime(target_time)}")
        else:
            print(f"  路径: {pause_flag}")

        return True
    except Exception as e:
        print(f"✗ 创建暂停标志失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AdaGeoHyper-TKAN - 训练暂停工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pause_resume/pause.py                  # 立即暂停最新训练
  python pause_resume/pause.py --pause-time 100 # 100分钟后暂停
        """,
    )

    parser.add_argument(
        "--pause-time",
        type=float,
        default=0,
        help="距离现在多少分钟后暂停（默认0表示立即暂停）",
    )

    args = parser.parse_args()

    # 查找最新训练目录
    print("=" * 60)
    print("  AdaGeoHyper-TKAN - 训练暂停工具")
    print("=" * 60)
    print(f"\n正在查找最新训练目录...")

    target_run = find_latest_run()

    if target_run is None:
        print(f"✗ 未找到有效的训练目录")
        print(f"  请确保已经启动训练")
        return 1

    print(f"✓ 找到训练目录: {target_run}")

    # 检查是否有 latest.pth (说明训练正在/曾经运行)
    latest_ckpt = os.path.join(target_run, "checkpoints", "latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"✓ 找到 checkpoint: latest.pth")
    else:
        print(f"⚠️  未找到 checkpoint, 训练可能尚未开始")

    # 创建暂停标志
    print("\n" + "=" * 60)
    print("设置暂停信号")
    print("=" * 60)

    if create_pause_flag(target_run, args.pause_time):
        print("\n" + "=" * 60)
        if args.pause_time > 0:
            print(f"✓ 定时暂停信号已设置（{args.pause_time}分钟后）")
        else:
            print("✓ 暂停信号已发送")
        print("=" * 60)

        print(f"\n训练将在当前 epoch 结束后:")
        print("  1. 保存 checkpoint（含完整训练历史）")
        print("  2. 保存日志")
        print("  3. 清除暂停标志")
        print("  4. 优雅退出")

        print("\n恢复训练时，请运行:")
        print("  python pause_resume/resume.py")

        if args.pause_time > 0:
            print("\n您现在可以:")
            print("  • 继续使用电脑（训练将在后台自动暂停）")
            print("  • 关闭终端/IDE")
            print("  • 让电脑休眠以节省能耗")

        print("=" * 60)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
