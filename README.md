# AdaGeoHyper-TKAN

基于**自适应地理超图（Adaptive Geo-Hypergraph）+ TKAN** 的多站点气象时空预测框架，支持温度、湿度、云量、风速四类任务，并已集成**动态K稀疏裁剪超图**。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [2.1 数据流输出图](#21-数据流输出图)
- [2.2 模型结构说明模块](#22-模型结构说明模块)
- [2.3 模型结构图](#23-模型结构图)
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

### 2.1 数据流输出图

```text
[raw pkl 数据]
  trn/val/test: x, y, context
  position.pkl: lon/lat/(alt)
        │
        ▼
[数据加载与预处理 utils/data_loader.py]
  - 站点采样 / 样本采样
  - 温度 K→°C（按要素）
  - weather scaler(train fit, val/test transform)
  - context 周期特征 sin/cos + context scaler
  - x拼接context，y保持weather-only
        │
        ├──────────────► 产物A: DataLoader(train/val/test)
        ├──────────────► 产物B: preprocessing_artifact.pkl
        └──────────────► 产物C: positions(N,2/3)
                              │
                              ▼
[模型前向 AdaGeoHyperTKAN]
  输入: x [B, Tin, N, F]
  1) AdaptiveGeoHypergraph  -> H_s [B, Tin, N, D]
  2) TKAN                   -> H_t [B, Tin, N, D]
  3) GatedFusion            -> H_fused [B, Tin, N, D]
  4) PredictionHead         -> y_pred [B, Tout, N, C]
                              │
                              ▼
[训练 train.py]
  - loss(可配 l1/huber/mse)
  - 反向传播 + scheduler + early stop
  - 记录动态K统计(range/mean/median/keep)
  - 保存 checkpoint(best/latest)
                              │
                              ▼
[预测 predict.py]
  - 加载 best_model + preprocessing_artifact
  - 反标准化后评估 MAE/RMSE/sMAPE/WMAPE/MAPE
  - 风速额外 VectorMAE/VectorRMSE
  - 导出 summary / 图表 / npz
```

### 2.2 模型结构说明模块

#### 模块A：AdaptiveGeoHypergraph（空间模块）

- 输入：`x [B, Tin, N, F]`，站点位置 `positions [N, 2/3]`
- 核心：
  - KNN 构建候选邻域（`k_neighbors = K_max`）
  - 自适应打分 `φ([p_i,p_j,s_i,s_j])`
  - softmax 权重后可做动态裁剪（`top_p/threshold`）
- 输出：`H_s [B, Tin, N, D]`

#### 模块B：TKAN（时间模块）

- 输入：`x [B, Tin, N, F]`
- 核心：
  - TKANCell 堆叠建模时间依赖
  - KANLinear 提升非线性表达能力
- 输出：`H_t [B, Tin, N, D]`

#### 模块C：GatedFusion（时空融合）

- 输入：`H_s` 与 `H_t`
- 核心：学习门控权重 `z`，自适应融合空间/时间特征
- 输出：`H_fused [B, Tin, N, D]`

#### 模块D：PredictionHead（多步预测头）

- 输入：`H_fused [B, Tin, N, D]`
- 核心：按站点展平时间维后经 MLP 映射到 `Tout*C`
- 输出：`y_pred [B, Tout, N, C]`

#### 模块间接口一览

- `Hypergraph`: `[B,T,N,F] -> [B,T,N,D]`
- `TKAN`: `[B,T,N,F] -> [B,T,N,D]`
- `Fusion`: `([B,T,N,D],[B,T,N,D]) -> [B,T,N,D]`
- `Head`: `[B,T,N,D] -> [B,Tout,N,C]`

### 2.3 模型结构图

```text
                           AdaGeoHyper-TKAN (12 -> 12)

输入 X [B, Tin, N, F]  ────────────────────────────────────────────────────────────────┐
  (weather + optional context)                                                          │
                                                                                         │
                       ┌─────────────────────────────────────────────┐                    │
                       │ AdaptiveGeoHypergraph (空间分支)            │                    │
                       │ - KNN 超图骨架 (lon/lat/alt)               │                    │
                       │ - 自适应权重 scorer + dynamic K pruning     │                    │
                       │ - Hypergraph Conv x L                       │                    │
                       └──────────────────────┬──────────────────────┘                    │
                                              │ H_s [B,Tin,N,D]                           │
                                              │                                            │
                                              │                                            │
                       ┌──────────────────────▼──────────────────────┐                    │
                       │ TKAN (时间分支)                              │                    │
                       │ - TKANCell x Lt                              │                    │
                       │ - KANLinear 非线性时间映射                  │                    │
                       └──────────────────────┬──────────────────────┘                    │
                                              │ H_t [B,Tin,N,D]                           │
                                              │                                            │
                                ┌─────────────▼─────────────┐                              │
                                │ GatedFusion               │                              │
                                │ z = σ(Ws·Hs + Wt·Ht + b) │                              │
                                │ H = z·Hs + (1-z)·Ht      │                              │
                                └─────────────┬─────────────┘                              │
                                              │ H_fused [B,Tin,N,D]                        │
                                              ▼                                            │
                                ┌───────────────────────────┐                               │
                                │ PredictionHead (MLP)      │                               │
                                │ flatten(Tin*D per station)│                               │
                                │ -> Tout*C                 │                               │
                                └─────────────┬─────────────┘                               │
                                              ▼                                            │
                                      输出 Ŷ [B,Tout,N,C]                                   │
```

> 说明：
> - `Tin=12`, `Tout=12`（默认）
> - `D` 为隐藏维度（`model.hidden_dim`）
> - `C` 为目标通道数（温度/湿度/云量=1，风速=2）

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

参数范围的物理意义：
- `K_max`：决定每站点可感知的最大空间作用半径。温度主要受局地辐射、地形与下垫面影响，`K_max` 过大等于把远距离异质气团混入局地估计，容易拉低信噪比。
- `top_p`：控制保留的“有效相互作用质量”。`top_p` 越低，模型越强调少数主导邻居（更接近局地传输）；越高则纳入更多弱联系（更平滑但可能引噪）。
- `min_keep`：保证最小连通，避免极端天气或权重塌缩时图传播断裂。
- `robust_preprocess q`：仅用于 scaler 拟合的分位截断，抑制极端离群值对尺度估计的污染，不改变原始监督目标。
- context 取舍：`month/day/time` 对应季节-日变化与昼夜辐射周期；`altitude/lat/lon` 提供静态地形背景；`region` 为粗粒度分类，若边界效应强可能引入硬分区偏置。

说明：温度任务引入过远邻居容易带来噪声，动态K均值建议落在约 4.5~6。

### 8.2 湿度（humidity）

推荐：局地差异更强，邻域应更小更稀疏。

- `k_neighbors (K_max)`: **6 ~ 10**（建议先 8）
- `top_p`: **0.65 ~ 0.75**（建议 0.70）
- `min_keep`: 2
- `loss_type`: `huber`（更抗异常）
- `robust_preprocess`: `q=[0.005, 0.995]`
- context 优先：`month/day/time/region`

参数范围的物理意义：
- `K_max`：湿度受边界层混合、地形遮挡和局地水汽源影响显著，空间相关长度通常短于风场；较小 `K_max` 更符合局地水汽传输尺度。
- `top_p`：较低区间（0.65~0.75）让模型优先保留高置信邻居，减少“远邻同化”造成的虚假平滑。
- `loss_type=huber`：对偶发极端湿度误差更稳，避免少量异常样本主导梯度。
- `robust_preprocess q=[0.005,0.995]`：湿度分布常在高湿端聚集，适度收紧分位可稳定尺度估计。
- context 中 `region`：在海陆过渡、城市群与地形分区明显区域，可提供边界层类型先验。

说明：湿度对局地和边界层条件敏感，过大K常导致收敛慢、误差平台高。

### 8.3 云量（cloud_cover）

推荐：全局与局地都重要，K可以中等偏大。

- `k_neighbors (K_max)`: **10 ~ 16**（建议先 12）
- `top_p`: **0.72 ~ 0.82**（建议 0.78）
- `min_keep`: 2
- `loss_type`: `huber`
- `robust_preprocess`: `q=[0.005, 0.995]`
- context 优先：`month/day/time/region/lat/lon`

参数范围的物理意义：
- `K_max`：云系受中尺度到天气尺度系统共同驱动（锋面、槽脊、对流组织），中等偏大 `K_max` 有助于覆盖更长的相关尺度。
- `top_p`：取中高区间表示保留更多弱联系，模拟云场的连续扩展与团簇传播；过低会过度局地化，削弱云带连续性。
- `loss_type=huber`：对局地强对流导致的突变误差更鲁棒。
- `robust_preprocess`：云量接近0或1时分布偏态明显，分位稳健拟合可避免尺度失真。
- context 中 `lat/lon/region`：对应大尺度环流带与下垫面差异，可提升空间分型能力。

说明：云量受大尺度过程影响较明显，适度保留远邻可能有益。

### 8.4 风速（component_of_wind）

推荐：向量目标，保留一定全局关联。

- `k_neighbors (K_max)`: **12 ~ 20**（建议先 16）
- `top_p`: **0.78 ~ 0.88**（建议 0.82）
- `min_keep`: 3
- `loss_type`: `huber` 或 `l1`
- 指标重点看：`VectorMAE / VectorRMSE`
- context 优先：`time/altitude/lat/lon`

参数范围的物理意义：
- `K_max`：风是矢量场，受大尺度压强梯度与地形导流共同影响，相关长度通常更长；较大 `K_max` 能覆盖主导流线与下游传播关系。
- `top_p`：中高区间可保留多方向输送路径，避免仅保留单一局地主导边导致方向信息丢失。
- `min_keep=3`：保证至少多方向连接，减少矢量场在稀疏图下的方向退化。
- `VectorMAE/VectorRMSE`：直接度量(u,v)向量误差，比仅看标量分量更符合物理风场质量。
- context 中 `alt/lat/lon/time`：对应地形摩擦、纬向环流背景与日变化边界层风切变。

说明：风场空间连通性更强，适当较大K通常比温湿更稳定。

### 8.5 损失函数特点与推荐（按要素）

#### L1（MAE）
- 特点：直接优化绝对误差，对离群点比 MSE 更稳；与 MAE 指标一致。
- 适用：当主目标是把 MAE 压到更低（尤其温度任务）。
- 注意：曲线可能比 MSE 略抖，建议配合合适学习率（如 1e-3~3e-3）。

#### Huber
- 特点：小误差区近似 MSE、大误差区近似 L1，兼顾平滑优化与抗异常值能力。
- 适用：湿度/云量/风速这类突变或异常值更常见的任务。
- 参数：`huber_delta` 常用 1.0；若重尾噪声更明显，可尝试 0.5~0.8。

#### MSE
- 特点：对大误差惩罚更重，通常更利于优化 RMSE。
- 适用：噪声较小、以 RMSE 为主目标时。
- 注意：对离群点敏感，可能牺牲 MAE 稳定性。

#### 推荐起点（本项目）
- `temperature`：优先 `l1`，备选 `huber`
- `humidity`：优先 `huber`
- `cloud_cover`：优先 `huber`
- `component_of_wind`：优先 `huber`（并同时关注 `VectorMAE/VectorRMSE`）

### 8.6 训练全局建议取值范围

- `learning_rate`: 1e-3 ~ 5e-3
  - 物理含义：不是物理量本身，而是参数更新步长；过大易在多尺度过程下振荡，过小则难以追踪快速天气过程。
- `batch_size`（8GB显存）：8 ~ 16
  - 统计含义：控制梯度估计方差与时空样本混合程度；更大 batch 更平滑但显存压力更高。
- `dropout`: 0.05 ~ 0.2
  - 作用含义：抑制对个别站点/时段噪声的过拟合，过高会损失有效物理相关性。
- `grad_clip`: 0.5 ~ 2.0
  - 稳定含义：限制极端样本导致的梯度尖峰，防止训练不稳定。
- `patience`: 15 ~ 35（湿度/风速可适当更大）
  - 收敛含义：多变量/矢量任务通常收敛更慢，需要更长观测窗口判断是否真正停滞。
- `use_amp`: true（推荐）
  - 数值含义：提高吞吐并降低显存占用，通常不改变物理建模结论。
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