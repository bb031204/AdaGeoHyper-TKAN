# AdaGeoHyper-TKAN 项目说明

## 项目简介

**AdaGeoHyper-TKAN** 是一个基于 PyTorch 的多站点气象时空序列预测模型。模型整合了：
- **自适应地理邻域超图 (Adaptive Geo Hypergraph)**：基于经度、纬度、海拔构建，用于空间关系建模
- **TKAN (Temporal Kolmogorov-Arnold Network)**：基于 KAN 思想的时间递推网络，负责时序建模
- **门控融合 (Gated Fusion)**：时空特征的自适应融合
- **非自回归多步预测头**：一次性输出未来 12 步预测

## 模型架构

```
输入 X [B, 12, N, F]
    ↓ 自适应地理邻域超图 (空间建模)
H_s [B, 12, N, D]
    ↓ TKAN (时间建模)
H_t [B, 12, N, D]
    ↓ GatedFusion(H_s, H_t)
H_fused [B, 12, N, D]
    ↓ Prediction Head
Y_pred [B, 12, N, C]  (未来12步预测)
```

## 项目结构

```
AdaGeoHyper-TKAN/
├── config.yaml              # 配置文件
├── main.py                  # 统一启动入口
├── train.py                 # 训练流程
├── predict.py               # 预测与评估流程
├── requirements.txt         # 依赖列表
├── PROJECT_README.md        # 本说明文件
│
├── models/                  # 模型模块
│   ├── __init__.py
│   ├── kan_linear.py        # KANLinear (B-spline KAN映射层)
│   ├── tkan.py              # TKANCell + TKANLayer (时间递推模块)
│   ├── hypergraph.py        # AdaptiveGeoHypergraph (自适应超图)
│   ├── fusion.py            # GatedFusion (门控融合)
│   ├── prediction_head.py   # PredictionHead (多步预测头)
│   └── ada_geo_hyper_tkan.py # AdaGeoHyperTKAN (完整主模型)
│
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载与预处理
│   ├── metrics.py           # 评估指标 (MAE/RMSE/MAPE)
│   ├── logger.py            # 日志配置
│   └── visualization.py     # 可视化工具
│
├── cache/                   # 超图缓存目录
├── outputs/                 # 训练输出目录
└── TKAN-main/              # 原始 TKAN 参考代码
```

## 环境配置

### 1. 安装 Python 环境
推荐 Python 3.9+ 。

### 2. 安装 PyTorch
根据你的 CUDA 版本选择安装命令：
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU
pip install torch
```

### 3. 安装其他依赖
```bash
pip install -r requirements.txt
```

## 使用方式

### 快速开始 (完整训练+预测)
```bash
python main.py --config config.yaml
```

### 指定数据集训练
```bash
# 温度预测
python main.py --config config.yaml --dataset temperature

# 湿度预测
python main.py --config config.yaml --dataset humidity

# 云量预测
python main.py --config config.yaml --dataset cloud_cover

# 风速预测
python main.py --config config.yaml --dataset component_of_wind
```

### 仅训练
```bash
python main.py --mode train --config config.yaml --dataset temperature
```

### 仅预测 (需要已有训练结果)
```bash
python main.py --mode predict --output_dir outputs/20240101_120000_temperature
```

### 常用命令行参数
```bash
python main.py \
    --config config.yaml \
    --dataset temperature \
    --device cuda \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num_stations 512 \
    --sample_ratio 0.5 \
    --seed 42
```

## 配置说明

编辑 `config.yaml` 来控制训练参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `data.dataset_name` | 数据集名称 | temperature |
| `data.num_stations` | 站点数 (null=全部) | null |
| `data.sample_ratio` | 样本抽样比例 | 1.0 |
| `hypergraph.k_neighbors` | K近邻数 | 8 |
| `model.hidden_dim` | 隐藏维度 | 64 |
| `model.tkan_layers` | TKAN层数 | 2 |
| `training.batch_size` | 批大小 | 32 |
| `training.epochs` | 训练轮数 | 100 |
| `training.learning_rate` | 学习率 | 0.001 |

## 输出说明

训练和预测完成后，结果保存在 `outputs/` 目录下，按 `YYYYMMDD_HHMMSS_数据集名称` 命名：

```
outputs/20240101_120000_temperature/
├── config_snapshot.yaml      # 配置快照
├── training_history.json     # 训练历史
├── test_summary.json         # 测试摘要 (JSON)
├── test_summary.txt          # 测试摘要 (文本)
├── test_metrics.json         # 测试指标
├── predictions.npz           # 预测结果
├── checkpoints/
│   ├── best_model.pth        # 最佳模型
│   └── latest.pth            # 最新checkpoint
├── figures/
│   ├── loss_curve.png        # Loss曲线
│   ├── val_metrics.png       # 验证指标曲线
│   ├── per_step_metrics.png  # 每步指标
│   └── pred_vs_truth/        # 预测对比图
└── train_*.log               # 训练日志
```

## 数据集格式

数据位于 `D:\bishe\WYB\` 下，每个数据集目录包含：
- `trn.pkl` / `val.pkl` / `test.pkl`：训练/验证/测试数据
  - 格式: dict with 'x', 'y' keys
  - Shape: (num_samples, 12, 2048, C)
- `position.pkl`：站点位置信息
  - 格式: dict with 'lonlat' key
  - Shape: (2048, 2) [经度, 纬度]

## 模型核心组件

### 1. KANLinear
B-spline 基函数参数化的非线性映射层，替代传统固定激活函数。

### 2. TKANCell
LSTM 风格的时间递推单元，用 KAN 子层替代 output gate 计算。

### 3. 自适应地理邻域超图
- 基于 Haversine 球面距离 + 海拔差构建三维地理距离
- K近邻搜索构建超边
- 动态自适应权重 (基于当前输入窗口的站点状态)

### 4. 门控融合
```
z = sigmoid(W1 · H_s + W2 · H_t)
H_fused = z · H_s + (1-z) · H_t
```

## 注意事项

1. **显存需求**：使用全部 2048 站点时，建议 16GB+ 显存。如显存不足，可通过 `num_stations` 减少站点数。
2. **首次运行**：首次运行时会构建超图并缓存，后续相同配置可直接使用缓存。
3. **每次只训练一个数据集**：不同数据集需要独立训练，不能混合。
4. **Checkpoint恢复**：在 `config.yaml` 中设置 `resume_from_checkpoint` 指向 `.pth` 文件即可恢复训练。
