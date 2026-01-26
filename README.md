# STG-Transformer: 动态时空特征增强的舰船燃气轮机状态预测

基于论文《动态时空特征增强的舰船燃气轮机状态预测方法研究》的PyTorch实现。

## 模型架构

STG-Transformer结合动态图卷积与频域分解的双通道Transformer，用于捕获燃气轮机传感器数据中的时空依赖关系。

### 核心组件

1. **动态图构建 (Pearson Correlation)**: 基于滑动窗口动态计算传感器间邻接矩阵
2. **图卷积堆叠 (GraphConvStack)**: 捕获空间依赖关系
3. **频域分解 + 双通道编码器**: 低/高频分量分别建模时间依赖
4. **解码器重构 (TemporalDecoder)**: 输出预测序列

## 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib tensorboard tqdm
```

## 使用方法

### 训练模型

```bash
python -m STG_Transformer.train \
    --data_path <数据文件路径> \
    --batch_size 32 \
    --seq_len 96 \
    --pred_len 96 \
    --target_cols "kMc,kMt" \
    --d_model 128 \
    --n_heads 8 \
    --gnn_layers 2 \
    --enc_layers 2 \
    --dec_layers 1 \
    --d_ff 512 \
    --dropout 0.1 \
    --freq_ratio 0.1 \
    --top_k 8 \
    --lr 0.001 \
    --epochs 100
```

### UCI CBM Dataset 示例

```bash
python scripts/prepare_uci_cbm.py \
    --data_path "UCI CBM Dataset/data.txt" \
    --features_path "UCI CBM Dataset/Features.txt" \
    --output "UCI CBM Dataset/uci_cbm.csv"

python scripts/prepare_graph_baselines.py \
    --csv_path "UCI CBM Dataset/uci_cbm.csv" \
    --pred_len 96

python -m STG_Transformer.train \
    --data_path "UCI CBM Dataset/uci_cbm.csv" \
    --seq_len 96 \
    --pred_len 96 \
    --target_cols "kMc,kMt"
```

### 参数说明

- `data_path`: CSV格式的数据文件路径
- `batch_size`: 批次大小
- `seq_len`: 输入序列长度
- `pred_len`: 预测长度
- `target_cols`: 目标变量列名，逗号分隔，默认使用全部数值列
- `drop_cols`: 需要忽略的列名（如日期列）
- `features_path`: 特征名文件路径（用于无表头的data.txt）
- `d_model`: 模型维度
- `n_heads`: 注意力头数
- `gnn_layers`: 图卷积层数
- `enc_layers`: 编码器层数
- `dec_layers`: 解码器层数
- `d_ff`: 前馈网络维度
- `dropout`: Dropout比率
- `freq_ratio`: 低频比例（用于频域分解）
- `top_k`: 动态邻接矩阵每个节点保留的相关性Top-K
- `lr`: 学习率
- `epochs`: 训练轮数

## 项目结构

```
STG_Transformer/
├── __init__.py          # 包初始化
├── layers.py            # 核心层实现
├── model.py             # 主模型架构
├── data_provider.py     # 数据加载和预处理
├── train.py             # 训练脚本
└── utils.py             # 工具函数
```

## 数据格式

数据应为CSV格式，每行代表一个时间步，每列代表一个特征。若使用 `data.txt` 这类无表头文本，
可提供 `features_path` 自动补齐列名。模型会自动进行标准化处理，验证时使用同一标准化器进行逆变换以计算
MSE/MAE/RMSE。若列名包含 `kMc/kMt`，将默认使用它们作为目标变量，否则默认取最后两列。

## 引用

如果使用本代码，请引用原论文：
```
张祺. 动态时空特征增强的舰船燃气轮机状态预测方法研究.
```
