# AdaGeoHyper-TKAN

基于**自适应地理超图（Adaptive Geo-Hypergraph）+ TKAN** 的多站点气象时空预测框架，支持温度、湿度、云量、风速四类任务，并已集成**动态K稀疏裁剪超图**。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [3. 项目结构](#3-项目结构)
- [4. 环境安装](#4-环境安装)
- [5. 数据格式与加载逻辑](#5-数据格式与加载逻辑)
- [6. 动态超图与动态K说明](#6-动态超图与动态k说明)
- [7. 配置文件说明（config.yaml）](#7-配置文件说明configyaml)
- [8. 不同数据集配置建议（新增）](#8-不同数据集配置建议新增)
- [9. 训练与预测](#9-训练与预测)
- [10. 评估指标说明](#10-评估指标说明)
- [11. 输出目录说明](#11-输出目录说明)
- [12. 常见问题 FAQ](#12-常见问题-faq)

---

## 1. 项目概览

AdaGeoHyper-TKAN 的总体流程：

1. 基于站点经纬度/海拔构建 KNN 超图骨架；
2. 基于输入窗口状态计算自适应邻域权重；
3. 可选动态裁剪（top-p / threshold）得到动态有效邻域；
4. 空间模块输出送入 TKAN 做时序建模；
5. Gated Fusion 融合时空表示；
6. PredictionHead 非自回归输出未来多步预测。

输入输出：
- 输入：`[B, T_in, N, F]`
- 输出：`[B, T_out, N, C]`

默认任务为 12→12 预测（`input_len=12, pred_len=12`）。

---

## 2. 核心特性

- 自适应超图：邻域骨架静态，邻域权重随样本动态变化
- 动态K裁剪（方案3）：
  - 候选 `K_max` 固定
  - 每次前向按权重做 `top_p` 或 `threshold` 裁剪
  - 保留 `min_keep`，并强制保留 self-loop
- 要素自适应预处理：
  - 温度支持开尔文转摄氏度
  - weather/context 分离缩放器
  - 周期特征（月/日/时）自动 sin/cos 编码
- 要素化评估：
  - 通用：MAE / RMSE / sMAPE / WMAPE / MAPE
  - 风速额外：VectorMAE / VectorRMSE
- 训练终端可视化：
  - 每个 epoch 输出动态K范围、均值、中位数、保留率

---

## 3. 项目结构

```text
AdaGeoHyper-TKAN/
├── config.yaml
├── main.py
├── train.py
├── predict.py
├── elements_settings.py
├── models/
│   ├── ada_geo_hyper_tkan.py
│   ├── hypergraph.py
│   ├── tkan.py
│   ├── kan_linear.py
│   ├── fusion.py
│   └── prediction_head.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
├── cache/
├── outputs/
└── pause_resume/
```

---

## 4. 环境安装

建议 Python 3.9+。

```bash
conda create -n hyper_tkan python=3.9 -y
conda activate hyper_tkan
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pyyaml matplotlib tqdm scikit-learn
```

如果你在 Windows + `torch.compile` 场景下遇到编译器问题，建议：
- 要么安装完整 C++ Build Tools；
- 要么在配置里 `use_compile: false`（更稳）。

---

## 5. 数据格式与加载逻辑

每个数据集目录应包含：

- `trn.pkl`
- `val.pkl`
- `test.pkl`
- `position.pkl`

`trn/val/test.pkl` 期望包含：
- `x`: `[S, Tin, N, C_weather]`
- `y`: `[S, Tout, N, C_weather]`
- `context`（可选）: `[S, Tin, N, C_ctx]`

`position.pkl` 至少包含站点经纬度：
- `lonlat` 或 `position`（`[N,2]`）
- 可选海拔字段：`alt / altitude / elev / elevation`

### 5.1 数据加载关键机制

在 `utils/data_loader.py` 中：

- 站点采样：train/val/test/position 共用同一组索引，严格对齐
- scaler 只在 train 拟合，val/test 复用
- 温度可先做 Kelvin→Celsius 再归一化
- context 先选通道，再对周期字段（month/day/time）做 sin/cos 编码
- context 仅拼接到 `x`，`y` 保持 weather-only（损失更干净）
- 若 position 无海拔，可回退到 context 通道 5 作为 altitude

---

## 6. 动态超图与动态K说明

动态K由以下参数共同决定：

- `k_neighbors`：候选 `K_max`
- `dynamic_pruning.enabled`：是否启用动态裁剪
- `dynamic_pruning.mode`：`top_p` 或 `threshold`
- `top_p`：累计权重阈值
- `threshold`：硬阈值
- `min_keep`：每个节点最少保留邻居数

当前实现细节：

1. 先在候选邻域上 softmax 得到权重；
2. 再按 `top_p/threshold` 生成掩码；
3. 强制保留 self-loop 与 `min_keep`；
4. 掩码后重新归一化，仅聚合保留邻域；
5. 训练日志输出动态K统计（range/mean/median/keep ratio）。

---

## 7. 配置文件说明（config.yaml）

核心配置块：

- `data`
  - `dataset_name`: `temperature / humidity / cloud_cover / component_of_wind`
  - `include_context` + `context_features`
  - `sample_ratio / val_sample_ratio / test_sample_ratio`
- `hypergraph`
  - `k_neighbors`, `lambda_geo`, `lambda_alt`
  - `dynamic_pruning` 子配置
- `model`
  - `hidden_dim`, `tkan_hidden_dim`, `tkan_layers`, `dropout`
- `training`
  - `batch_size`, `epochs`, `learning_rate`, `loss_type`
  - `scheduler`, `warmup_epochs`, `use_amp`, `use_compile`
- `output`
  - `output_dir`, `resume_from_checkpoint`

---

## 8. 不同数据集配置建议（新增）

> 目标：在不改模型结构前提下，用配置提高稳定性与指标表现。

### 8.1 温度（temperature）

推荐：局地相关更强，动态K不要太大。

- `k_neighbors (K_max)`: **8 ~ 12**（建议先 10）
- `dynamic_pruning.mode`: `top_p`
- `top_p`: **0.72 ~ 0.82**（建议 0.75）
- `min_keep`: 2
- `loss_type`: `l1` 或 `huber`
  - 若主目标是 MAE，优先 `l1`
- `robust_preprocess`: `q=[0.001, 0.999]`
- `context_features` 建议先精简：
  - 开：`month/day/time/altitude`
  - 可选再加：`latitude/longitude`
  - 先关：`region`

说明：温度任务引入过远邻居容易带来噪声，动态K均值建议落在约 4.5~6。

### 8.2 湿度（humidity）

推荐：局地差异更强，邻域应更小更稀疏。

- `k_neighbors (K_max)`: **6 ~ 10**（建议先 8）
- `top_p`: **0.65 ~ 0.75**（建议 0.70）
- `min_keep`: 2
- `loss_type`: `huber`（更抗异常）
- `robust_preprocess`: `q=[0.005, 0.995]`
- context 优先：`month/day/time/region`

说明：湿度对局地和边界层条件敏感，过大K常导致收敛慢、误差平台高。

### 8.3 云量（cloud_cover）

推荐：全局与局地都重要，K可以中等偏大。

- `k_neighbors (K_max)`: **10 ~ 16**（建议先 12）
- `top_p`: **0.72 ~ 0.82**（建议 0.78）
- `min_keep`: 2
- `loss_type`: `huber`
- `robust_preprocess`: `q=[0.005, 0.995]`
- context 优先：`month/day/time/region/lat/lon`

说明：云量受大尺度过程影响较明显，适度保留远邻可能有益。

### 8.4 风速（component_of_wind）

推荐：向量目标，保留一定全局关联。

- `k_neighbors (K_max)`: **12 ~ 20**（建议先 16）
- `top_p`: **0.78 ~ 0.88**（建议 0.82）
- `min_keep`: 3
- `loss_type`: `huber` 或 `l1`
- 指标重点看：`VectorMAE / VectorRMSE`
- context 优先：`time/altitude/lat/lon`

说明：风场空间连通性更强，适当较大K通常比温湿更稳定。

### 8.5 训练全局建议取值范围

- `learning_rate`: 1e-3 ~ 5e-3
- `batch_size`（8GB显存）：8 ~ 16
- `dropout`: 0.05 ~ 0.2
- `grad_clip`: 0.5 ~ 2.0
- `patience`: 15 ~ 35（湿度/风速可适当更大）
- `use_amp`: true（推荐）
- `use_compile`: 
  - Linux: 可开
  - Windows: 建议先关或使用自动回退机制

---

## 9. 训练与预测

### 9.1 训练 + 预测一体

```bash
python main.py --config config.yaml
```

### 9.2 仅训练

```bash
python main.py --mode train --config config.yaml
```

### 9.3 仅预测

```bash
python main.py --mode predict --output_dir "D:/bishe/AdaGeoHyper-TKAN/outputs/xxxxxx_temperature"
```

---

## 10. 评估指标说明

`utils/metrics.py` 当前支持：

- 通用：`MAE`, `RMSE`, `sMAPE`, `WMAPE`, `MAPE`
- 风速额外：`VectorMAE`, `VectorRMSE`

训练阶段：
- 验证集反标准化后计算指标
- 每 epoch 终端输出主指标 + 动态K统计

预测阶段：
- 保存 overall + per-step 指标
- 输出 `test_summary.json / test_summary.txt / test_metrics.json`

---

## 11. 输出目录说明

每次训练会在 `output.output_dir` 下创建时间戳目录，例如：

```text
outputs/20260322_172846_temperature/
├── config_snapshot.yaml
├── preprocessing_artifact.pkl
├── training_history.json
├── train_xxx.log
├── checkpoints/
│   ├── latest.pth
│   └── best_model.pth
├── figures/
│   ├── loss_curve.png
│   ├── val_metrics.png
│   └── per_step_metrics.png
├── predictions.npz
├── test_metrics.json
├── test_summary.json
└── test_summary.txt
```

---

## 12. 常见问题 FAQ

### Q1: 动态K怎么看是否合理？
看每个 epoch 的：
- `range`
- `mean`
- `median`
- `keep ratio`

温度任务通常不建议长期维持很高均值（如 >6.5）。

### Q2: Windows 下 compile 训练中断怎么办？
优先：`use_compile: false`。当前代码也已加入编译失败自动回退 eager 机制。

### Q3: 为什么 MAPE 很大？
当真实值接近0时，MAPE会放大，建议以 MAE/RMSE/sMAPE/WMAPE 为主，MAPE作参考。

### Q4: 如何保证 train/val/test 站点对齐？
当前加载器会复用同一 `station_indices` 并执行一致性检查。

### Q5: 训练慢怎么优化？
- 开启 AMP
- 适当减小 `num_stations`
- 调整 `k_neighbors/top_p` 降低有效邻域
- 检查 context 维度是否过多

---

如需更细的“温/湿/云/风四套即用配置文件”，可在当前 README 建议范围内直接扩展为 `config_temperature.yaml` 等专项配置。