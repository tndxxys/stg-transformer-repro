# 论文模块与代码对应关系（STG-Transformer）

本文档将论文中的主要模块逐一对应到代码位置，便于你查找实现细节与复现实验。

---

## 1. 动态空间关系建模（动态图构建）

**论文描述**：基于传感器间皮尔逊相关系数动态构建邻接矩阵。  
**代码对应**：
- `STG_Transformer/layers.py`  
  - `compute_pearson_adjacency(...)`：计算相关矩阵，支持 `top_k` / `corr_threshold` 稀疏化  
  - `normalize_adjacency(...)`：对邻接矩阵做对称归一化  

**调用位置**：
- `STG_Transformer/model.py` → `STGTransformer.forward(...)`  
  - `adj = compute_pearson_adjacency(x, top_k=..., threshold=...)`

---

## 2. 空间特征提取（图卷积网络 GNN）

**论文描述**：图神经网络提取空间依赖关系。  
**代码对应**：
- `STG_Transformer/layers.py`  
  - `GraphConvLayer`：单层图卷积  
  - `GraphConvStack`：多层堆叠 + 残差 + LayerNorm  

**调用位置**：
- `STG_Transformer/model.py` → `self.spatial_gnn`  
- `STGTransformer.forward(...)` → `x = self.spatial_gnn(x, adj)`

---

## 3. 频域分解（低频/高频）

**论文描述**：傅里叶变换分解为低频趋势与高频波动。  
**代码对应**：
- `STG_Transformer/layers.py`  
  - `frequency_decompose(...)`：`rfft/irfft` 分解，`freq_ratio` / `freq_cutoff` 控制频段

**调用位置**：
- `STG_Transformer/model.py` → `low, high = frequency_decompose(...)`

---

## 4. 双通道时间建模（Transformer 编码器）

**论文描述**：低频/高频分别建模时间依赖。  
**代码对应**：
- `STG_Transformer/layers.py`  
  - `TemporalEncoder`：基于 `nn.TransformerEncoderLayer`  
  - `PositionalEncoding`：位置编码  

**调用位置**：
- `STG_Transformer/model.py`  
  - `self.low_encoder` / `self.high_encoder`  
  - `low = self.low_encoder(low)`  
  - `high = self.high_encoder(high)`

---

## 5. 低/高频融合（特征融合层）

**论文描述**：低频与高频通道融合。  
**代码对应**：
- `STG_Transformer/model.py`  
  - `self.fusion = nn.Linear(d_model * 2, d_model)`  
  - `fused = torch.cat([low, high], dim=-1)`

---

## 6. 解码器（预测序列重构）

**论文描述**：编码-解码结构预测未来序列。  
**代码对应**：
- `STG_Transformer/layers.py`  
  - `TemporalDecoder`：基于 `nn.TransformerDecoderLayer`  
  - 使用 `learnable query` 作为解码器输入  

**调用位置**：
- `STG_Transformer/model.py`  
  - `decoded = self.decoder(fused)`

---

## 7. 输入/输出投影

**论文描述**：将输入映射到模型维度、输出映射到预测值。  
**代码对应**：
- `STG_Transformer/model.py`  
  - `self.input_projection = nn.Linear(n_features, d_model)`  
  - `self.output_projection = nn.Linear(d_model, n_features)`  

---

## 8. 数据预处理与标准化

**论文描述**：标准化消除量纲差异。  
**代码对应**：
- `STG_Transformer/data_provider.py`  
  - `_load_dataframe(...)`：读取 CSV/TXT，转数值，去空列  
  - `StandardScaler()`：仅用训练集拟合标准化  
  - `GasTurbineDataset`：构造滑动窗口样本  

**逆标准化**：
- `STG_Transformer/utils.py`  
  - `inverse_transform_targets(...)`

---

## 9. 训练与评价

**论文描述**：MSE/MAE/RMSE 指标评估。  
**代码对应**：
- `STG_Transformer/train.py`  
  - `train_epoch(...)` / `validate(...)`  
  - `criterion = nn.MSELoss()`  
  - `calculate_metrics(...)` 输出 MSE/MAE/RMSE/R2  
  - 同时输出标准化尺度指标（`Std_*`）便于论文对齐  

**指标计算**：
- `STG_Transformer/utils.py`  
  - `calculate_metrics(...)`

---

## 10. 论文中的“时空特征增强”整体流程（对应 forward）

**论文流程**：
1) 动态构图  
2) 空间图卷积  
3) 频域分解  
4) 双通道时间建模  
5) 融合 + 解码预测

**代码入口**：
- `STG_Transformer/model.py` → `STGTransformer.forward(...)`

---

## 11. 目标变量选择（kMc/kMt）

**论文描述**：分别对 kMc / kMt 预测。  
**代码对应**：
- `STG_Transformer/data_provider.py`  
  - `target_cols` 参数或自动识别 `kMc/kMt`  
  - `target_indices` 传入训练/验证流程  

---

## 12. 关键张量形状（便于核对）

- 输入 `x`: `(B, T, N, F)`  
  - `T=seq_len`, `N=变量数`, `F=1`  
- 输出 `out`: `(B, N, pred_len, 1)`  

详见：
`STG_Transformer/model.py` 与 `STG_Transformer/data_provider.py`

---

如需我把论文公式与代码函数逐行对照（比如动态邻接矩阵计算公式、频域分解公式），告诉我具体想对照的章节或公式编号。  
