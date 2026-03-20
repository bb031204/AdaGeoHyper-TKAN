【本科毕设工程化代码生成 Prompt｜TKAN 可解释气象预测（最终版）】

一、项目背景
我正在完成本科毕业设计，题目为《基于 KAN 的可解释气象预测方法》。
目标是基于 Python，使用开源 TKAN（Temporal KAN）模型完成气象时间序列预测，并实现：
1）气象时序数据的预处理与特征构建；
2）TKAN 模型的构建、训练与调优；
3）以 LSTM 作为基线模型进行预测性能对比；
4）基于 TKAN 的可解释性分析与可视化，体现模型决策透明性。

二、技术环境与强约束

1. 必须使用 Keras 3（Multi-backend），后端已设置为 PyTorch：KERAS\_BACKEND=torch。
2. 严禁引入或依赖 TensorFlow（不得 import tensorflow / tf.keras）。
3. 当前环境tkan：

   * torch 2.9.1+cu128
   * CUDA 12.8
   * GPU：RTX 5070 Laptop（8GB）
   * keras 3.12.0，backend = torch

4. 使用的开源 TKAN 模型位于：D:\\bishe\\code\\TKAN-main

   * 不得修改该目录下任何文件
   * 仅通过 import / editable install 调用
   * 说明：实际训练时使用 pip 安装的 tkan==0.4.3（已验证可运行）。D:\\bishe\\code\\TKAN-main 仅作为源码参考，不要求从该路径直接 import。

三、路径说明

* 毕设任务说明文件：D:\\bishe\\毕设文件\\（仅用于理解，不修改）
* 数据集路径：D:\\bishe\\WYB\\（主要为 .pkl）
* 开源的TKAN，位于：D:\\bishe\\code\\TKAN-main（说明：说明：实际训练时使用 pip 安装的 tkan==0.4.3（已验证可运行）。D:\\bishe\\code\\TKAN-main 仅作为源码参考，不要求从该路径直接 import。）
* 新工程目录：D:\\bishe\\code\\TKAN\_wyb  （所有新代码必须放在该目录下，不得修改其他目录）

四、数据要求（PKL）

1. 数据以 .pkl 存储，结构可能为 dict 或 tuple，需自动探测。
2. 程序必须在终端打印：

   * pkl 数据类型
   * 若为 dict，打印 keys
   * 各数组 shape / dtype
   * NaN / Inf 统计

3. 不做图建模（不使用邻接矩阵进行 GNN）。
4. 保留数据集的时序特征
5. 允许两种模式：

   * 单站点时间序列
   * 多站点聚合（mean）

6. 最终输入 TKAN 的数据形状必须为：
   X: (N, L, F)
   y: (N, H\*F)

五、工程结构（必须严格一致）
在 D:\\bishe\\code\\TKAN\_wyb\\code\\ 下生成：

config.py                # 统一管理参数（或 yaml + 读取器）
dataloader.py            # 读取 pkl、预处理、滑窗、归一化
model\_tkan.py            # TKAN 模型封装
model\_lstm.py            # LSTM baseline
train.py                 # 训练与验证（含 tqdm）
predict.py               # 测试与指标计算
visualize.py             # 所有可视化
utils.py                 # 日志、时间、路径工具
main.py                  # 总入口
inspect\_pkl.py           # 可选：单独检查 pkl 结构

六、输出目录强约束
outputs 目录下已存在以下一级目录：

* cloud\_cover
* component\_of\_wind
* humidity
* temperature

程序每次只允许选择一个 target，所有输出只能写入对应子目录：

时间戳格式必须统一为：

YYYYMMDD\_HHMMSS

（例如：20260111\_172030）

完整输出路径必须为：

outputs/{target}/{model\_name}/{L}\_{H}/{YYYYMMDD\_HHMMSS}/
*├── log/
│   ├── train.log
│   ├── config.yaml
│   └── env.txt
├── model/
│   ├── best\_model.keras
│   └── scaler.pkl
├── figure/
│   ├── loss\_curve.png
│   ├── prediction\_vs\_gt.png
│   ├── model\_compare.png
│   └── interpret*\*.png
└── pred/

&nbsp;   └── test\_prediction.csv

target 取值必须严格限定为：
\["cloud\_cover", "component\_of\_wind", "humidity", "temperature"]

main.py 必须支持 --target 参数，并在终端打印 target 与 run\_dir。

七、终端日志要求（必须满足）

1. 使用 tqdm 显示：

   * 总 epoch 进度条
   * 每 epoch 内 batch 进度条
   * 总的训练时间和剩余时间

2. 每 epoch 打印：

   * Epoch x/y
   * Train Loss
   * Val Loss
   * Epoch Time
   * ETA

3. 训练日志同时输出到终端与 log/train.log，环境中有tabulate，可生成美观日志
4. 测试阶段打印 RMSE / MAE / MAPE，并对比 TKAN 与 LSTM。

八、模型与训练配置（按 TKAN 论文）

* 模型结构：
  TKAN(64, return\_sequences=True)
  TKAN(64, return\_sequences=False)
* optimizer：Adam
* loss：MSE（训练），报告 RMSE / MAE / MAPE
* epochs：200
* batch\_size：64（显存不足时自动降级）
* callbacks：
  EarlyStopping(patience=6, restore\_best\_weights=True)
  ReduceLROnPlateau(patience=3, factor=0.5)
* TKAN 参数必须使用真实 API（不使用 num\_splines，如需使用 sub\_kan\_configs）。

九、可解释性分析（必须实现）

1. 单变量扰动（Partial Dependence）：
   固定其他变量，扫描单一气象变量，绘制预测输出变化曲线。
2. 时间滞后敏感性分析：
   分别扰动 t-1 / t-3 / t-12，分析其对预测结果的影响。
   所有图保存到 figure 目录。

十、运行方式
在 PowerShell 中：

conda activate tkan
python D:\\bishe\\code\\TKAN\_wyb\\code\\main.py

程序需自动检查 GPU、路径、配置并创建输出目录。

目标是生成一套：
结构清晰、日志完整、可复现实验、完全绕开 TensorFlow、结果优异、基于tkan论文，符合本科毕设规范的 TKAN 气象预测工程代码。

严禁自行实现、复刻或改写 TKAN / KAN / EfficientKAN 等模型结构。模型部分必须直接 import 开源实现（from tkan import TKAN），只允许编写数据处理、训练流程、评估、可视化与工程化代码。

