# AdaGeoHyper-TKAN

> **基于自适应地理邻域超图与时序 KAN 的多站点气象时空序列预测模型**

---

## 目录

- [项目简介](#项目简介)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [命令行参数说明](#命令行参数说明)
- [配置文件详解](#配置文件详解)
- [训练流程说明](#训练流程说明)
- [预测与评估](#预测与评估)
- [输出目录结构](#输出目录结构)
- [核心模块详解](#核心模块详解)
- [GPU 训练说明](#gpu-训练说明)
- [常见问题 FAQ](#常见问题-faq)

---

## 项目简介

**AdaGeoHyper-TKAN** 是一种面向多站点气象时空序列预测的深度学习模型，专为 ERA5 再分析数据集上的温度、湿度、云量、风速等气象要素的 12 步超前预测任务设计。

模型创新性地融合了以下四大核心技术：

| 模块 | 技术 | 作用 |
|------|------|------|
| 空间建模 | **自适应地理邻域超图** (Adaptive Geo-Hypergraph) | 基于 Haversine 球面距离 + 海拔差构建 K 近邻超图，动态自适应加权捕获高阶空间依赖 |
| 时间建模 | **TKAN** (Temporal Kolmogorov-Arnold Network) | 基于 LSTM 结构 + KAN B-spline 可学习激活函数替代 output gate，增强时序非线性建模能力 |
| 特征融合 | **门控融合** (Gated Fusion) | 基于可学习门控机制自适应融合空间特征 H_s 与时间特征 H_t |
| 预测输出 | **非自回归多步预测头** (Prediction Head) | MLP 一次性映射融合特征到未来 12 步，避免误差累积 |

### 支持的气象变量

| 数据集 | 变量 | 通道数 |
|--------|------|--------|
| `temperature` | 2m 温度 | 1 |
| `humidity` | 2m 相对湿度 | 1 |
| `cloud_cover` | 总云量 | 1 |
| `component_of_wind` | 10m 风速 (u, v 分量) | 2 |

---

## 模型架构

```
输入 X [B, T=12, N, F]   ← B=批次, T=时间步, N=站点数, F=特征通道
        │
        ▼
┌─────────────────────────────────┐
│  自适应地理邻域超图 (空间模块)    │   ← Haversine + 海拔距离 → K-NN 超图
│  AdaptiveGeoHypergraph          │     自适应打分函数 → 动态权重
│  多层超图卷积                    │     加权聚合邻居特征
└─────────────────┬───────────────┘
                  │ H_s [B, T, N, D]
        ┌─────────┤
        │         ▼
        │  ┌──────────────────────┐
        │  │  TKAN (时间模块)      │   ← KAN-style LSTM
        │  │  多层 TKANCell        │     B-spline 可学习激活函数
        │  │  逐站点序列处理        │     多 KAN 子层聚合 output gate
        │  └──────────┬───────────┘
        │             │ H_t [B, T, N, D]
        ▼             ▼
┌─────────────────────────────────┐
│  门控融合 (Gated Fusion)         │
│  z = σ(W_s·H_s + W_t·H_t + b)  │
│  H_fused = z·H_s + (1-z)·H_t   │
└─────────────────┬───────────────┘
                  │ H_fused [B, T, N, D]
                  ▼
┌─────────────────────────────────┐
│  预测头 (Prediction Head)        │   ← 非自回归 MLP
│  Flatten(T×D) → MLP → Reshape   │     一次性输出 12 步
└─────────────────┬───────────────┘
                  │
                  ▼
输出 Y_pred [B, T_out=12, N, C]   ← 未来 12 步各站点预测值
```

---

## 项目结构

```
AdaGeoHyper-TKAN/
├── config.yaml                  # 🔧 全局配置文件 (数据/模型/训练/输出参数)
├── main.py                      # 🚀 统一启动入口 (训练+预测)
├── train.py                     # 📊 训练流程 (支持 checkpoint/早停/调度)
├── predict.py                   # 📈 预测与评估 (指标/可视化/summary)
│
├── models/                      # 🧠 模型模块
│   ├── __init__.py              #     包导出
│   ├── kan_linear.py            #     KANLinear — B-spline KAN 映射层
│   ├── tkan.py                  #     TKANCell + TKANLayer — 时间递推模块
│   ├── hypergraph.py            #     AdaptiveGeoHypergraph — 自适应超图空间模块
│   ├── fusion.py                #     GatedFusion — 时空门控融合
│   ├── prediction_head.py       #     PredictionHead — 多步非自回归预测头
│   └── ada_geo_hyper_tkan.py    #     AdaGeoHyperTKAN — 完整主模型
│
├── utils/                       # 🔧 工具模块
│   ├── __init__.py              #     包导出
│   ├── data_loader.py           #     数据加载/预处理/StandardScaler
│   ├── metrics.py               #     MAE / RMSE / MAPE 评估指标
│   ├── logger.py                #     统一日志管理
│   └── visualization.py         #     Loss曲线/指标曲线/预测对比可视化
│
├── cache/                       # 💾 超图缓存目录 (自动生成)
├── outputs/                     # 📁 训练输出目录 (自动生成)
├── files/                       # 📄 项目文档与参考资料
│   ├── requirements.txt         #     Python 依赖列表
│   └── ...
└── TKAN-main/                   # 📚 原始 TKAN 参考代码 (仅参考)
```

---

## 环境配置

### 1. 创建 Conda 环境 (推荐)

```bash
conda create -n hyper_tkan python=3.9 -y
conda activate hyper_tkan
```

### 2. 安装 PyTorch (GPU 版本优先)

> ⚡ **强烈建议使用 GPU 训练**，本项目默认配置 `device: "cuda"`。
> 2048 站点全量数据在 GPU 上约 20-40 分钟/epoch，CPU 上可能需要数小时。

根据你的 CUDA 版本选择安装命令：

```bash
# CUDA 12.8 (推荐, 与本项目测试环境一致)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

验证 GPU 是否可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

预期输出示例：

```
CUDA: True
GPU: NVIDIA GeForce RTX 4060
```

### 3. 安装其他依赖

```bash
pip install numpy scipy pyyaml matplotlib tqdm scikit-learn
```

或使用依赖文件：

```bash
pip install -r files/requirements.txt
```

### 4. 验证环境

```bash
python -c "import torch; import numpy; import scipy; import yaml; import matplotlib; print('All OK! torch:', torch.__version__)"
```

---

## 数据准备

### 数据目录结构

数据位于 `config.yaml` 中 `data.data_root` 指定的目录下（默认 `D:/bishe/WYB`），每个气象变量一个子目录：

```
D:/bishe/WYB/
├── temperature/           # 温度数据集
│   ├── trn.pkl            # 训练集
│   ├── val.pkl            # 验证集
│   ├── test.pkl           # 测试集
│   └── position.pkl       # 站点位置信息
├── humidity/              # 湿度数据集
│   ├── trn.pkl / val.pkl / test.pkl / position.pkl
├── cloud_cover/           # 云量数据集
│   ├── trn.pkl / val.pkl / test.pkl / position.pkl
└── component_of_wind/     # 风速数据集 (u, v 双通道)
    ├── trn.pkl / val.pkl / test.pkl / position.pkl
```

### 数据格式说明

**trn.pkl / val.pkl / test.pkl** (Python pickle dictionary):

```python
{
    'x': np.ndarray,        # shape: (num_samples, 12, 2048, C_target)
    'y': np.ndarray,        # shape: (num_samples, 12, 2048, C_target)
    'context': np.ndarray,  # shape: (num_samples, 12, 2048, C_context) (optional)
}
# C_target=1 (temperature/humidity/cloud_cover) or C_target=2 (wind u,v)
# C_context usually includes: year/month/day/time/region/altitude/latitude/longitude
# 12 = time steps, 2048 = total stations
```

> Notes:
> - `x/y` are target meteorological channels.
> - `context` supports two paths:
>   1) concatenate selected context channels to `x` via `include_context/context_features`;
>   2) provide altitude fallback for hypergraph when `position.pkl` has no altitude (`context[..., 5]`), controlled by `use_context_altitude`.

**position.pkl**（站点经纬度信息）:

```python
{
    'lonlat': np.ndarray,  # shape: (2048, 2) — [经度, 纬度]
    'alt': np.ndarray,     # shape: (2048,)   — 海拔 (可选)
}
```

---

## 快速开始

### 一键训练 + 预测 (推荐)

```bash
# 激活环境
conda activate hyper_tkan

# 进入项目目录
cd D:\bishe\AdaGeoHyper-TKAN

# 使用默认配置 (温度数据集, GPU 训练)
python main.py --config config.yaml
```

### 指定数据集训练

```bash
# 温度
python main.py --dataset temperature

# 湿度
python main.py --dataset humidity

# 云量
python main.py --dataset cloud_cover

# 风速 (u, v 双通道)
python main.py --dataset component_of_wind
```

### 仅训练

```bash
python main.py --mode train --config config.yaml
```

### 仅预测 (需要已有训练结果)

```bash
python main.py --mode predict --output_dir outputs/20260319_143000_temperature
```

### 直接调用 train.py / predict.py

```bash
# 训练
python train.py --config config.yaml

# 预测
python predict.py --output_dir outputs/20260319_143000_temperature
```

---

## 命令行参数说明

通过 `main.py` 启动时可使用以下命令行参数，**命令行参数会覆盖 `config.yaml` 中对应的设置**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `config.yaml` | 配置文件路径 |
| `--mode` | str | `all` | 运行模式：`all`(训练+预测) / `train`(仅训练) / `predict`(仅预测) |
| `--output_dir` | str | — | 预测模式下指定训练输出目录 |
| `--dataset` | str | — | 覆盖数据集名称 |
| `--device` | str | — | 覆盖设备：`cuda` / `cpu` / `auto` |
| `--batch_size` | int | — | 覆盖批大小 |
| `--epochs` | int | — | 覆盖训练轮数 |
| `--lr` | float | — | 覆盖学习率 |
| `--num_stations` | int | — | 覆盖站点数 (null=全部2048) |
| `--sample_ratio` | float | — | 覆盖训练集样本抽样比例 |
| `--seed` | int | — | 覆盖随机种子 |

### 使用示例

```bash
# 完整训练+预测, 使用 GPU, 温度数据集, 200 轮
python main.py --dataset temperature --device cuda --epochs 200

# 快速调试: 少量站点 + 少量样本 + 少量轮次
python main.py --dataset temperature --num_stations 256 --sample_ratio 0.1 --epochs 5 --batch_size 16

# 全量训练, 大批次
python main.py --dataset temperature --device cuda --batch_size 64 --epochs 200 --lr 0.001
```

---

## 配置文件详解

`config.yaml` 是项目的核心配置文件，分为 5 个部分：

### data — 数据配置

```yaml
data:
  dataset_name: "temperature"
  data_root: "D:/bishe/WYB"
  input_len: 12
  pred_len: 12
  num_stations: null
  sample_ratio: 1.0
  val_sample_ratio: 1.0
  test_sample_ratio: 1.0

  # 是否将 context 特征拼接到 x 输入通道
  include_context: false

  # 选择参与拼接的 context 通道（按需开启）
  context_features:
    use_year: false
    use_month: false
    use_day: false
    use_time: false
    use_region: false
    use_altitude: false
    use_latitude: false
    use_longitude: false
```

### hypergraph — 超图配置

```yaml
hypergraph:
  k_neighbors: 8
  lambda_geo: 1.0
  lambda_alt: 0.5

  # 若 position.pkl 无 alt/altitude/elev/elevation，
  # 可回退使用 context 第 5 通道（altitude）构图
  use_context_altitude: true

  use_hypergraph_cache: true
  cache_dir: "cache"
  summary_pool: "mean"
  scorer_hidden_dim: 32
```

### Context Features

The current version supports two context paths:

1. Input feature augmentation (for both training and prediction)
- Enable `data.include_context: true`.
- Channels selected by `data.context_features` are concatenated to the last dimension of `x`.
- Example: `x` changes from `[B, 12, N, 1]` to `[B, 12, N, 6]` when 5 context channels are enabled.
- `y` shape does not change; it still contains only target meteorological channels.

2. Altitude fallback for hypergraph construction
- Hypergraph first reads altitude from `position.pkl` (`alt/altitude/elev/elevation`).
- If missing and `hypergraph.use_context_altitude: true`, it falls back to `context[..., 5]`.

### model — 模型配置

```yaml
model:
  hidden_dim: 64                  # 空间模块隐藏维度 (也是 fusion_dim)
  tkan_hidden_dim: 64             # TKAN 隐藏维度 (须与 hidden_dim 一致)
  tkan_layers: 2                  # TKAN 堆叠层数
  tkan_sub_kan_configs: [null, 3] # TKAN KAN 子层配置 (null=默认线性, int=B-spline阶数)
  dropout: 0.1                    # Dropout 比率
  fusion_dim: 64                  # 融合层维度 (须与 hidden_dim 一致)
  kan_grid_size: 5                # KAN B-spline 网格区间数
  kan_spline_order: 3             # KAN B-spline 阶数 (3=三次样条)
  hypergraph_layers: 2            # 超图卷积层数
```

### training — 训练配置

```yaml
training:
  batch_size: 32                  # 批大小 (GPU 16GB 推荐 32, 8GB 推荐 16)
  epochs: 200                     # 最大训练轮数
  learning_rate: 0.001            # 初始学习率
  weight_decay: 0.0001            # AdamW 权重衰减
  device: "cuda"                  # 设备: cuda(GPU优先) / auto / cpu
  seed: 42                        # 随机种子 (确保可复现)
  patience: 15                    # 早停耐心值 (连续N轮无改善则停止)
  use_early_stop: true            # 是否启用早停
  scheduler: "cosine"             # 学习率调度: cosine / step / none
  scheduler_step_size: 20         # StepLR 步长 (仅 scheduler=step 生效)
  scheduler_gamma: 0.5            # StepLR 衰减因子
  grad_clip: 1.0                  # 梯度裁剪阈值 (防止梯度爆炸)
```

### output — 输出配置

```yaml
output:
  output_dir: "D:/bishe/AdaGeoHyper-TKAN/outputs"  # 输出根目录
  save_best_only: true            # 仅保存最佳模型 (始终保存 latest.pth)
  resume_from_checkpoint: null    # 恢复训练: 填写 .pth 路径, null=从头训练
  log_interval: 10                # 每 N 个 batch 打印一次训练日志
```

---

## 训练流程说明

执行 `python main.py` 后的完整训练流程：

```
1. 加载 config.yaml 配置
2. 创建带时间戳的输出目录 (outputs/YYYYMMDD_HHMMSS_数据集/)
3. 初始化日志系统 (控制台 + 文件双输出)
4. 设置随机种子 (确保可复现)
5. 检测并选择 GPU/CPU 设备
6. 保存配置快照 (config_snapshot.yaml)
7. 加载 pkl 数据 → 构建 DataLoader (训练/验证/测试)
   ├── StandardScaler 逐通道标准化
   ├── 可选: 站点采样 (num_stations)
   └── 可选: 样本抽样 (sample_ratio)
8. 构建 AdaGeoHyperTKAN 模型 → 移至 GPU
9. Build hypergraph (lon/lat + altitude; altitude from position.pkl or context channel-5), with cache acceleration
10. 初始化 Adam 优化器 + 学习率调度器
11. 训练循环 (每个 epoch):
    ├── 训练: MSE 损失 + 梯度裁剪 + 反标准化指标
    ├── 验证: 计算 Val Loss / MAE / RMSE / MAPE
    ├── 学习率调度 (Cosine Annealing)
    ├── 保存 checkpoint (latest.pth + best_model.pth)
    └── 早停判断 (patience=15)
12. 绘制 Loss 曲线 + 验证指标曲线
13. 保存训练历史 (training_history.json)
```

### 训练日志示例

```
[2026-03-19 14:30:00] [INFO] ============================================================
[2026-03-19 14:30:00] [INFO] AdaGeoHyper-TKAN 训练启动
[2026-03-19 14:30:00] [INFO] ============================================================
[2026-03-19 14:30:00] [INFO] 数据集: temperature
[2026-03-19 14:30:01] [INFO] [设备] GPU: NVIDIA GeForce RTX 4060, 显存: 8.0GB
[2026-03-19 14:30:02] [INFO] [数据] trn 集最终: x=(5000, 12, 2048, 1), y=(5000, 12, 2048, 1)
[2026-03-19 14:30:03] [INFO] [模型] 模型参数量: 381,389 (可训练: 381,389)
[2026-03-19 14:30:03] [INFO] [超图] 超图结构已就绪 (N=2048, K=8, edges=2048)
[2026-03-19 14:30:05] [INFO] Epoch   1/200 | Train Loss: 0.832154 | Val Loss: 0.654321 | Val MAE: 2.1234 | ★ Best
[2026-03-19 14:30:35] [INFO] Epoch   2/200 | Train Loss: 0.612543 | Val Loss: 0.543210 | Val MAE: 1.8765 | ★ Best
...
[2026-03-19 15:20:00] [INFO] [早停] 连续 15 个 epoch 无改善, 停止训练
[2026-03-19 15:20:00] [INFO] 训练完成! 最佳 Val Loss: 0.123456
```

---

## 预测与评估

训练完成后自动进入预测阶段（或手动执行预测）：

```
1. 加载 config_snapshot.yaml (确保与训练时配置一致)
2. 加载测试数据
3. 加载最佳模型权重 (best_model.pth)
4. 重建超图结构
5. 在测试集上前向推理 → 反标准化
6. 计算评估指标:
   ├── 整体指标: MAE / RMSE / MAPE
   └── 逐步指标: 12 个预测步各自的 MAE / RMSE / MAPE
7. 保存预测结果 (predictions.npz)
8. 生成可视化图表:
   ├── 预测 vs 真实对比图 (随机选取多个样本 + 多个站点)
   └── 逐步指标柱状图
9. 生成实验 Summary (JSON + TXT 双格式)
```

### 评估指标

| 指标 | 全称 | 公式 | 说明 |
|------|------|------|------|
| **MAE** | Mean Absolute Error | \|pred - true\| 的均值 | 预测的平均绝对误差 |
| **RMSE** | Root Mean Square Error | sqrt(MSE) | 对大误差更敏感 |
| **MAPE** | Mean Absolute Percentage Error | \|pred-true\|/\|true\| × 100% | 相对误差百分比 |

---

## 输出目录结构

每次训练+预测会在 `outputs/` 下生成独立的实验目录：

```
outputs/20260319_143000_temperature/
│
├── config_snapshot.yaml          # 📋 本次实验的完整配置快照
├── training_history.json         # 📊 训练历史 (所有 epoch 的 loss 和指标)
├── test_summary.json             # 📝 测试结果 JSON (整体+逐步指标)
├── test_summary.txt              # 📝 测试结果可读文本
├── test_metrics.json             # 📊 测试指标详情
├── predictions.npz               # 💾 预测结果 (pred + truth, numpy 格式)
├── train_20260319_143000.log     # 📄 完整训练日志
│
├── checkpoints/                  # 💾 模型权重
│   ├── best_model.pth            #     最佳模型 (按验证损失)
│   └── latest.pth                #     最新 checkpoint (可恢复训练)
│
└── figures/                      # 📊 可视化图表
    ├── loss_curve.png            #     训练/验证 Loss 曲线
    ├── val_metrics.png           #     验证集 MAE/RMSE/MAPE 曲线
    ├── per_step_metrics.png      #     每步预测指标柱状图
    └── pred_vs_truth/            #     预测 vs 真实对比图
        ├── sample_0_station_0.png
        ├── sample_0_station_1.png
        └── ...
```

---

## 核心模块详解

### 1. KANLinear (`models/kan_linear.py`)

**B-spline 基函数参数化的非线性映射层**，是 TKAN 的基础组件。

- 传统线性层：`y = σ(Wx + b)`，激活函数 σ 固定不可学习
- KAN 线性层：`y = Σ φ_ij(x_j)`，激活函数 φ 由 B-spline 参数化，可学习

关键特性：
- B-spline 网格大小可配置 (`grid_size=5`)
- 样条阶数可配置 (`spline_order=3`，三次样条)
- 兼有 base 权重（线性部分 + SiLU 激活）和 spline 权重（非线性部分）
- 支持 LayerNorm 正则化

### 2. TKANCell / TKANLayer (`models/tkan.py`)

**基于 LSTM 的时间递推单元，output gate 由 KAN 子层替代。**

与标准 LSTM 的核心区别：

| | 标准 LSTM | TKAN |
|---|---|---|
| input gate (i) | `σ(W_i · [h, x])` | 同 LSTM |
| forget gate (f) | `σ(W_f · [h, x])` | 同 LSTM |
| cell candidate (c̃) | `tanh(W_c · [h, x])` | 同 LSTM |
| **output gate (o)** | `σ(W_o · [h, x])` | **多个 KAN 子层聚合** |

每个 KAN 子层：
- 对输入进行线性投影
- 通过 KANLinear 非线性变换或简单线性变换（由 `sub_kan_configs` 控制）
- 维护独立的递归状态
- 最终聚合输出通过 sigmoid 生成 output gate

`TKANLayer` 封装多层 `TKANCell`，支持 `return_sequences=True/False`。

### 3. AdaptiveGeoHypergraph (`models/hypergraph.py`)

**基于地理信息的自适应加权超图空间模块。**

构建流程：
1. **距离计算**：Haversine 球面距离 + 标准化海拔差
2. **综合距离**：`d_ij = sqrt(λ_g · d_geo² + λ_h · Δalt²)`
3. **超边构造**：以每个站点为中心，K 近邻 + 自身构成超边
4. **自适应权重**：基于当前时间窗口数据，通过 MLP 打分函数动态计算邻域权重
5. **超图卷积**：加权聚合邻居特征 → 线性变换 → LayerNorm → ReLU

特色：
- 超图结构可缓存，避免重复计算
- 权重随输入动态变化（自适应）
- 支持 2D (经纬度) 和 3D (经纬度+海拔) 位置信息

### 4. GatedFusion (`models/fusion.py`)

**门控融合机制**，自适应平衡空间特征和时间特征：

```
z = sigmoid(W_s · H_s + W_t · H_t + bias)
H_fused = z ⊙ H_s + (1 - z) ⊙ H_t
```

- 当 z → 1 时，侧重空间特征
- 当 z → 0 时，侧重时间特征
- z 可学习，自动适应不同站点/不同时刻的空间-时间权重分配

### 5. PredictionHead (`models/prediction_head.py`)

**非自回归多步预测头**，将融合表示一次性映射为 12 步预测：

```
H_fused [B, T_in=12, N, D]
    → reshape 为 [B, N, T_in × D]
    → MLP (3层: 768→128→128→12)
    → reshape 为 [B, T_out=12, N, C]
```

非自回归的优势：避免自回归多步预测中的误差累积问题。

---

## GPU 训练说明

### 默认配置

本项目 `config.yaml` 默认设置 `device: "cuda"`，**优先使用 GPU 训练**。

### 显存需求估算

| 站点数 | batch_size | hidden_dim | 预估显存 |
|--------|-----------|-----------|---------|
| 256 | 32 | 64 | ~2 GB |
| 512 | 32 | 64 | ~4 GB |
| 1024 | 32 | 64 | ~6 GB |
| 2048 | 32 | 64 | ~10 GB |
| 2048 | 16 | 64 | ~6 GB |
| 2048 | 64 | 64 | ~16 GB |

### 显存不足时的解决方案

**方案 1**：减小 batch_size

```bash
python main.py --batch_size 16
```

**方案 2**：减少站点数

```bash
python main.py --num_stations 512
```

**方案 3**：组合使用

```bash
python main.py --num_stations 1024 --batch_size 16
```

**方案 4**：修改 `config.yaml` 降低 `hidden_dim`

```yaml
model:
  hidden_dim: 32
  tkan_hidden_dim: 32
  fusion_dim: 32
```

### 多 GPU

当前版本为单 GPU 训练。如需多 GPU，可使用 PyTorch 的 `DataParallel`，在 `train.py` 中 `model = model.to(device)` 后添加：

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### CPU 回退

如无可用 GPU，可设置：

```bash
python main.py --device cpu
```

或在 `config.yaml` 中修改 `device: "cpu"`。

---

## 常见问题 FAQ

### Q1: 首次运行特别慢？

**A**: 首次运行时需要构建超图结构（涉及距离矩阵计算），之后会缓存到 `cache/` 目录。后续运行相同站点配置时自动使用缓存，速度会快很多。

### Q2: 显存溢出 (CUDA out of memory)？

**A**: 参见 [GPU 训练说明](#gpu-训练说明) 中的显存不足解决方案。最快的方法是减小 `batch_size` 或 `num_stations`。

### Q3: 如何从断点恢复训练？

**A**: 在 `config.yaml` 中设置：

```yaml
output:
  resume_from_checkpoint: "outputs/20260319_143000_temperature/checkpoints/latest.pth"
```

然后重新运行 `python main.py`。

### Q4: 训练后如何只跑预测？

**A**:

```bash
python main.py --mode predict --output_dir outputs/20260319_143000_temperature
```

或：

```bash
python predict.py --output_dir outputs/20260319_143000_temperature
```

### Q5: 如何快速调试验证代码可用性？

**A**: 使用极小数据量快速跑通全流程：

```bash
python main.py --num_stations 64 --sample_ratio 0.05 --epochs 3 --batch_size 8
```

### Q6: 不同数据集需要分别训练吗？

**A**: 是的。每个数据集（temperature/humidity/cloud_cover/component_of_wind）需要独立训练，模型权重不共享。每次训练自动创建独立的输出目录。

### Q7: 如何切换学习率调度策略？

**A**: 在 `config.yaml` 中修改：

```yaml
training:
  scheduler: "cosine"   # 余弦退火 (推荐)
  # scheduler: "step"   # 阶梯衰减
  # scheduler: "none"   # 不使用调度
```

### Q8: 输出的预测结果如何加载使用？

**A**:

```python
import numpy as np

data = np.load("outputs/20260319_143000_temperature/predictions.npz")
predictions = data["predictions"]   # shape: (num_samples, 12, num_stations, C)
ground_truth = data["ground_truth"] # shape: (num_samples, 12, num_stations, C)
```

---

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```
AdaGeoHyper-TKAN: Adaptive Geo-Hypergraph Temporal Kolmogorov-Arnold Network
for Multi-site Meteorological Spatio-temporal Sequence Prediction
```

## 新增预处理逻辑（要素自适应）

- 要素配置统一在 `elements_settings.py`：
  - `kelvin_to_celsius`
  - `normalize`
  - `scaler_type`
  - `context_scaler_type`
  - `k`
  - `degree_clamp_min`
  - `float32_norm`

- 数据预处理（`utils/data_loader.py`）：
  - Temperature：先做 K->C（`-273.15`），再归一化。
  - 气象通道使用全局通道标准化器（训练集拟合，val/test复用）。
  - Context 使用独立 scaler（训练集拟合，val/test复用）。
  - 启用 context 时，标准化后的 context 会拼接到 `x` 和 `y`。

- K 值与稳定参数：
  - 训练/预测会按要素自动覆盖 `hypergraph.k_neighbors`、`degree_clamp_min`、`float32_norm`。
  - 训练开始会打印：Element、Effective K、Config K。

- 指标反标准化：
  - 仅对气象目标通道反标准化并计算 MAE/RMSE/MAPE。
  - 若 `y` 中拼接了 context 通道，不参与指标计算。
