# 训练暂停与恢复功能

## ⚡ 快速开始

**重要：请确保在项目根目录 `D:\bishe\AdaGeoHyper-TKAN` 下执行命令！**

```bash
# 进入项目目录
cd D:\bishe\AdaGeoHyper-TKAN

# 立即暂停
python pause_resume/pause.py

# 60分钟后暂停
python pause_resume/pause.py --pause-time 60

# 恢复训练
python pause_resume/resume.py

# 查看 checkpoint 信息（不启动训练）
python pause_resume/resume.py --info
```

---

## 📋 功能说明

### 1. 暂停训练 (`pause.py`)

在指定时间后完成当前 epoch 并自动暂停，保存完整训练状态。

**用法：**
```bash
# 立即暂停（完成当前 epoch 后）
python pause_resume/pause.py

# 5分钟后暂停
python pause_resume/pause.py --pause-time 5

# 60分钟后暂停
python pause_resume/pause.py --pause-time 60

# 2小时后暂停
python pause_resume/pause.py --pause-time 120
```

**工作原理：**
1. 创建 `.pause` 标志文件，包含目标时间戳
2. 训练器在每个 epoch 结束后检查该文件
3. 到达指定时间后，完成当前 epoch
4. 自动保存 checkpoint（含完整训练历史）并退出
5. 自动清除 `.pause` 标志文件

**暂停后保存的文件：**
```
outputs/20260319_211740_temperature/
├── checkpoints/
│   ├── latest.pth        # 暂停时的完整状态（用于恢复）
│   └── best_model.pth    # 最佳模型
├── figures/               # 训练曲线图
├── config_snapshot.yaml   # 训练配置快照
└── train_*.log            # 训练日志
```

---

### 2. 恢复训练 (`resume.py`)

自动查找最新的训练结果并恢复训练。

**用法：**
```bash
# 自动恢复最新训练
python pause_resume/resume.py

# 指定 checkpoint 恢复
python pause_resume/resume.py --checkpoint outputs/xxx/checkpoints/latest.pth

# 恢复后50分钟自动暂停
python pause_resume/resume.py --resume-time 50

# 仅查看 checkpoint 信息（不启动训练）
python pause_resume/resume.py --info

# 指定 config 恢复
python pause_resume/resume.py --config config.yaml
```

**恢复流程：**
1. 自动查找最新的训练目录
2. 查找最新的 checkpoint 文件（优先 `latest.pth`）
3. 自动加载训练时保存的 `config_snapshot.yaml`（确保配置一致）
4. 显示训练信息（epoch、best_val_loss、训练历史）
5. 在**同一个输出目录**中继续训练
6. 训练曲线、日志等完全连续

**恢复的状态包括：**
- ✅ 模型参数
- ✅ 优化器状态（包括 Adam 动量）
- ✅ 学习率调度器状态
- ✅ 训练/验证 Loss 历史（曲线连续）
- ✅ 验证指标历史（MAE/RMSE/MAPE）
- ✅ 最佳模型记录
- ✅ 早停无改善计数器

---

## 🔄 完整工作流程示例

### 场景1：长时间训练分批进行

```bash
# 第1天：正常启动训练
python main.py --config config.yaml

# 训练进行中... 需要关机休息
# 新开终端，发送暂停信号：
python pause_resume/pause.py

# 训练会在当前 epoch 结束后安全暂停
# 关机休息 💤

# 第2天：开机后恢复训练
python pause_resume/resume.py

# 第3天：再次恢复
python pause_resume/resume.py
```

### 场景2：定时训练

```bash
# 启动训练
python main.py --config config.yaml

# 在另一个终端设置2小时后自动暂停
python pause_resume/pause.py --pause-time 120

# 2小时后训练自动保存退出，电脑可以休眠

# 下次恢复并再训练1小时
python pause_resume/resume.py --resume-time 60
```

### 场景3：查看训练进度再决定

```bash
# 查看最新 checkpoint 信息
python pause_resume/resume.py --info

# 输出:
#   Epoch: 45
#   最佳验证 Loss: 0.123456
#   剩余轮数: 155
#   最近 Train Loss: [0.1234, 0.1198, 0.1167, 0.1145, 0.1132]

# 如果满意，继续训练
python pause_resume/resume.py
```

### 场景4：指定特定 checkpoint 恢复

```bash
python pause_resume/resume.py --checkpoint outputs/20260319_211740_temperature/checkpoints/latest.pth
```

---

## ⚠️ 注意事项

### Config 一致性

恢复训练时会**自动使用训练时保存的 `config_snapshot.yaml`**，确保：
- 模型结构完全相同
- 数据处理方式相同
- 超图参数相同

如果需要修改配置（如更换学习率），可以手动指定：
```bash
python pause_resume/resume.py --config config_modified.yaml
```
> ⚠️ **注意：** 修改模型结构相关的配置会导致 checkpoint 不兼容！

### Checkpoint 保存时机

- **每个 epoch 结束**：保存 `latest.pth`（含完整训练历史）
- **验证 Loss 改善时**：额外保存 `best_model.pth`
- **暂停时**：使用已保存的 `latest.pth`，无需额外操作

### 暂停的安全性

- 暂停**不会中断**正在进行的 epoch
- 会在当前 epoch **完整结束**后才暂停
- 所有状态已在 epoch 结束时自动保存
- 恢复训练的质量与不间断训练**完全一致**

---

## 🛠️ 故障排除

### 问题1：找不到 checkpoint

```
✗ 未找到可恢复的训练目录
```

**解决：**
- 确认已经完成过至少一次训练（至少完成 1 个 epoch）
- 检查 `outputs/` 目录是否存在
- 手动指定 checkpoint 路径

### 问题2：config 不匹配

```
⚠️ 警告: 找不到 config_snapshot.yaml
```

**解决：**
- 确认训练目录中存在 `config_snapshot.yaml`
- 或手动指定 config：`python pause_resume/resume.py --config config.yaml`

### 问题3：暂停未生效

如果发送暂停信号后训练仍在继续：
1. 确认 `.pause` 文件已在训练输出目录中创建
2. 训练会在**当前 epoch 完成后**才检查暂停信号
3. 如果一个 epoch 耗时很长，需要等待其完成

### 问题4：恢复后 Loss 曲线不连续

正常情况下 Loss 曲线应完全连续。如果出现断层：
- 确认使用的是 `latest.pth`（非 `best_model.pth`）
- 确认 config 没有被修改

---

## 📊 输出示例

### pause.py 输出

```
============================================================
  AdaGeoHyper-TKAN - 训练暂停工具
============================================================

正在查找最新训练目录...
✓ 找到训练目录: D:\bishe\AdaGeoHyper-TKAN\outputs\20260319_211740_temperature
✓ 找到 checkpoint: latest.pth

============================================================
设置暂停信号
============================================================
  路径: D:\bishe\AdaGeoHyper-TKAN\outputs\20260319_211740_temperature\.pause

============================================================
✓ 暂停信号已发送
============================================================

训练将在当前 epoch 结束后:
  1. 保存 checkpoint（含完整训练历史）
  2. 保存日志
  3. 清除暂停标志
  4. 优雅退出

恢复训练时，请运行:
  python pause_resume/resume.py
============================================================
```

### resume.py 输出

```
============================================================
  AdaGeoHyper-TKAN - 训练恢复工具
============================================================
🔍 自动查找最新 checkpoint...
✓ 最新训练目录: D:\bishe\AdaGeoHyper-TKAN\outputs\20260319_211740_temperature
✓ 找到 Checkpoint: ...\checkpoints\latest.pth

============================================================
  Checkpoint 信息
============================================================
  文件: ...\checkpoints\latest.pth
  Epoch: 15
  最佳验证 Loss: 0.234567
  训练历史长度: 15 epochs
  早停无改善计数: 3
  总计划轮数: 200
  剩余轮数: 185
  数据集: temperature
  最近 Train Loss: [0.2500, 0.2467, 0.2445, 0.2432, 0.2418]
  最近 Val Loss:   [0.2600, 0.2534, 0.2489, 0.2456, 0.2390]
============================================================
✓ 使用训练时保存的 Config: ...\config_snapshot.yaml

============================================================
  准备恢复训练...
============================================================

  将从 checkpoint 继续训练，恢复的状态包括:
    ✓ 模型参数
    ✓ 优化器状态（动量等）
    ✓ 学习率调度器状态
    ✓ 训练历史记录（loss 曲线连续）
    ✓ 最佳模型记录
    ✓ 早停计数器

  训练质量将与不间断训练完全一致

============================================================
  启动训练命令:
  python train.py --config ... --resume ... --output_dir ...
============================================================
```

---

## 🎯 工作原理

### 文件标志机制

暂停/恢复功能使用文件标志机制（`.pause` 文件）进行通信：

1. **暂停信号创建**：
   - `pause.py` 在训练目录创建 `.pause` 文件
   - 文件内容为目标暂停时间的时间戳
   - 支持立即暂停（当前时间）和定时暂停（未来时间）

2. **训练器检查**：
   - 训练器在每个 epoch 结束后检查 `.pause` 文件
   - 如果当前时间 >= 目标时间，触发暂停流程
   - 完成当前 epoch 后保存 checkpoint 并退出

3. **标志清除**：
   - 暂停完成后自动删除 `.pause` 文件
   - 恢复训练时也会清除残留标志，避免误触发

### Checkpoint 内容

`latest.pth` 中保存的完整状态：

| 字段 | 说明 |
|------|------|
| `epoch` | 当前 epoch 编号 |
| `model_state_dict` | 模型参数 |
| `optimizer_state_dict` | 优化器状态（含 Adam 动量） |
| `scheduler_state_dict` | 学习率调度器状态 |
| `best_val_loss` | 历史最佳验证 Loss |
| `train_losses` | 训练 Loss 历史列表 |
| `val_losses` | 验证 Loss 历史列表 |
| `val_metrics_history` | 验证指标历史 (MAE/RMSE/MAPE) |
| `no_improve_count` | 早停无改善计数器 |
| `config` | 训练配置字典 |
