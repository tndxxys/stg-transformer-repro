"""
STG-Transformer使用示例
"""
import torch
from STG_Transformer import STGTransformer, count_parameters

# 创建模型实例
model = STGTransformer(
    n_features=1,      # 每个节点的特征数
    n_nodes=16,        # 传感器节点数
    d_model=128,       # 模型维度
    n_heads=8,         # 注意力头数
    gnn_layers=2,      # 图卷积层数
    enc_layers=2,      # 编码器层数
    dec_layers=1,      # 解码器层数
    d_ff=512,          # 前馈网络维度
    dropout=0.1,       # Dropout率
    pred_len=96        # 预测步长
)

# 准备输入数据 (batch_size, seq_len, n_nodes, n_features)
batch_size = 32
seq_len = 24
x = torch.randn(batch_size, seq_len, 16, 1)

# 前向传播
output = model(x)
print(f"输出形状: {output.shape}")  # (batch_size, n_nodes, pred_len, n_features)

# 计算模型参数量
print(f"模型参数量: {count_parameters(model):,}")
