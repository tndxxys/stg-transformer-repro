# 项目代码说明（更具体版）

本文档详细解释项目代码结构、核心函数、数据格式、张量形状与输出文件，帮助你精确定位每一步的实现。

---

## 0. 项目结构与用途

```
STG_Transformer/            # 本文模型实现（动态图 + FFT + 双通道 Transformer）
scripts/                    # 数据转换与图基线准备脚本
baselines/                  # 12 个对比模型仓库
UCI CBM Dataset/            # 原始数据集 (data.txt + Features.txt)
README.md                   # 快速使用说明
REPRODUCE.md                # 复现流程（逐步）
BASELINES.md                # 基线模型运行示例
example_usage.py            # 简单推理示例
```

---

## 1. STG_Transformer 代码详解

### 1.1 `STG_Transformer/model.py`
主类：`STGTransformer`

输入与输出：
- 输入 `x`: `(B, T, N, F)`，默认 `F=1`  
  - `B` 批大小  
  - `T` 序列长度 (`seq_len`)  
  - `N` 节点数 = 变量数（UCI CBM 为 18 列，不含 `date`）  
  - `F` 每节点特征维度（本项目中固定为 1）
- 输出 `out`: `(B, N, pred_len, 1)`  

内部流程（对应论文结构）：
1. `input_projection`: `F -> d_model`  
2. `compute_pearson_adjacency`: 动态邻接矩阵 `A`，形状 `(B, N, N)`  
3. `GraphConvStack`: 图卷积提取空间关系  
4. `frequency_decompose`: FFT 分解为低频/高频  
5. `TemporalEncoder`（低频/高频双通道）  
6. `fusion`: 拼接并融合  
7. `TemporalDecoder`: 解码预测序列  
8. `output_projection`: `d_model -> 1`  

关键参数：
- `top_k` / `corr_threshold`：用于动态邻接矩阵稀疏化  
- `freq_ratio` / `freq_cutoff`：控制 FFT 分解频段  

---

### 1.2 `STG_Transformer/layers.py`
重要函数与模块：

- `compute_pearson_adjacency(x, top_k, threshold)`  
  - 输入 `x: (B, T, N, F)`  
  - 先在 `(T*F)` 维度上计算相关系数  
  - 输出 `A: (B, N, N)`  

- `GraphConvStack(d_model, n_layers)`  
  - 逐层执行 `GraphConvLayer`  
  - 残差 + LayerNorm  

- `frequency_decompose(x, freq_ratio)`  
  - 输入 `x: (B, T, N, D)`  
  - 输出 `low/high`: 形状同输入  

- `TemporalEncoder`  
  - 将 `(B, T, N, D)` reshape 为 `(B*N, T, D)`  
  - 通过 `TransformerEncoderLayer` 堆叠  

- `TemporalDecoder`  
  - 使用 learnable query (`pred_len x D`)  
  - 输出 `(B, N, pred_len, D)`  

---

### 1.3 `STG_Transformer/data_provider.py`
负责数据读取、标准化和样本切分。

输入格式：
- CSV 或 `.txt`（空格分隔）  
- `Features.txt` 用于解析列名  

关键逻辑：
- `_load_dataframe`:  
  - `.txt` 走 `sep=r"\s+"`, 自动写列名  
  - `.csv` 直接读取  
- `get_dataloaders`:  
  - 训练集拟合 `StandardScaler`  
  - 验证集使用同一 scaler  
  - 默认目标列为 `kMc/kMt`（若存在）  

Dataset 输出：
- `x`: `(seq_len, N, 1)`  
- `y`: `(n_targets, pred_len, 1)`  

---

### 1.4 `STG_Transformer/train.py`
训练入口与循环：
- `train_epoch` / `validate`  
  - `model(x)` 输出 `(B, N, pred_len, 1)`  
  - 使用 `target_indices` 选择目标节点  
  - MSE 损失  
- 验证时逆标准化输出  

日志输出：
- TensorBoard: `./logs`  
- 最优权重: `./checkpoints/best_model.pth`  

---

### 1.5 `STG_Transformer/utils.py`
指标与逆变换：
- `calculate_metrics`: MSE/MAE/RMSE/R2  
- `inverse_transform_targets`: 按 target 列反归一化  

---

## 2. 数据转换脚本

### 2.1 `scripts/prepare_uci_cbm.py`
用途：把原始 `data.txt` 转为带 `date` 的 CSV。

输出：
- `uci_cbm.csv`  
- 列名包含 `kMc` / `kMt`  

主要逻辑：
- 读取 `Features.txt` 解析列名  
- 根据行号生成等间隔 `date`  
- 把 `kMc/kMt` 移到最后两列  

---

### 2.2 `scripts/prepare_graph_baselines.py`
用途：为图基线生成统一输入。

输出文件：
- DeSTGNN:  
  - `baselines/DeSTGNN_full/data/CBM/cbm.npz`  
  - `baselines/DeSTGNN_full/data/CBM/cbm_adj.npy`  
  - `baselines/DeSTGNN_full/configurations/CBM.conf`  
- matgcn:  
  - `baselines/matgcn/data/cbm/cbm.npz`  
  - `baselines/matgcn/data/cbm/cbm_distance.csv`  
  - `baselines/matgcn/config/cbm.json`  

邻接矩阵构建：
- `adj_mode=full`：全连接  
- `adj_mode=corr`：皮尔逊相关  
- `adj_mode=topk`：相关性 Top-K  

---

## 3. 对比基线模型说明

### 3.1 Time-Series-Library
路径：`baselines/Time-Series-Library/`  

包含模型：
- DLinear / FiLM / SCINet / TimesNet  
- iTransformer / Autoformer / Crossformer / PatchTST  

运行入口：`run.py`  
数据格式：`date + features` CSV  
关键参数：
- `--features MS` 表示多变量输入、单变量输出  
- `--enc_in/--dec_in` = 变量数（18）  

---

### 3.2 TimeXer
路径：`baselines/TimeXer/`  
运行入口：`run.py`  
数据格式与 Time-Series-Library 相同。  

---

### 3.3 WPMixer
路径：`baselines/WPMixer/`  
运行入口：`run_LTF.py`  
已支持 `custom`：  
必须提供 `--root_path/--data_path/--c_in/--c_out`。  

---

### 3.4 DeSTGNN（DEST-GNN）
路径：`baselines/DeSTGNN_full/`  

数据文件格式：
- `cbm.npz` 内部必须有 `data`  
  - 形状 `(T, N, 1)`  

预处理：
- `prepareData.py` 会生成 `cbm_rX_dY_wZ.npz`  
  - `train_x`: `(B, N, 1, T)`  
  - `train_target`: `(B, N, pred_len)`  

训练入口：
- `train_DeSTGNN.py`  

---

### 3.5 MA-T-GCN（matgcn）
路径：`baselines/matgcn/`  

数据文件格式：
- `cbm.npz` 内部有 `data`  
  - 形状 `(T, N, 1)`  
- `distance.csv`：边表 CSV  

我们已支持自定义 `hours`：
- 配置文件 `cbm.json` 中可写 `"hours": [1,2,3]`  
- 避免 7*24 长窗口导致样本不足  

训练入口：
- `main.py`  

---

## 4. 输出文件与日志位置

STG-Transformer：
- 日志：`./logs/`  
- 模型：`./checkpoints/best_model.pth`  

Time-Series-Library / TimeXer：
- 模型与日志：各自 `./checkpoints/`  

WPMixer：
- 输出目录：`baselines/WPMixer/saved/`  

DeSTGNN：
- 输出目录：`baselines/DeSTGNN_full/experiments/CBM/`  

matgcn：
- 输出目录：`baselines/matgcn/saved/cbm/`  

---

## 5. 常见问题定位

1) 维度错误  
   - 检查 `enc_in/dec_in/c_out` 是否为 18  
2) matgcn 样本为空  
   - 将 `hours` 改小，例如 `[1,2,3]`  
3) DeSTGNN 无法读取数据  
   - 确认 `cbm.npz` 中包含 `data` 键  

---

如需进一步补充“详细张量维度示意图”或“逐行代码注释版说明”，告诉我即可。  
