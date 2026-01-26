import torch
import torch.nn as nn
from .layers import (
    GraphConvStack,
    TemporalEncoder,
    TemporalDecoder,
    compute_pearson_adjacency,
    frequency_decompose
)


class STGTransformer(nn.Module):
    """STG-Transformer主模型 - 用于燃气轮机状态预测"""
    def __init__(
        self,
        n_features,
        n_nodes,
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        pred_len=1,
        gnn_layers=2,
        enc_layers=2,
        dec_layers=1,
        freq_ratio=0.1,
        freq_cutoff=None,
        top_k=None,
        corr_threshold=None
    ):
        super().__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.pred_len = pred_len
        self.freq_ratio = freq_ratio
        self.freq_cutoff = freq_cutoff
        self.top_k = top_k
        self.corr_threshold = corr_threshold

        self.input_projection = nn.Linear(n_features, d_model)
        self.spatial_gnn = GraphConvStack(d_model, n_layers=gnn_layers, dropout=dropout)

        self.low_encoder = TemporalEncoder(d_model, n_heads, d_ff, n_layers=enc_layers, dropout=dropout)
        self.high_encoder = TemporalEncoder(d_model, n_heads, d_ff, n_layers=enc_layers, dropout=dropout)

        self.fusion = nn.Linear(d_model * 2, d_model)
        self.decoder = TemporalDecoder(d_model, n_heads, d_ff, dec_layers, pred_len, dropout)

        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: (B, T, N, F)
        # 1) 动态构图（基于窗口内皮尔逊相关）
        adj = compute_pearson_adjacency(
            x,
            top_k=self.top_k,
            threshold=self.corr_threshold
        )

        # 2) 输入投影到模型维度
        x = self.input_projection(x)
        # 3) 图卷积提取空间关系
        x = self.spatial_gnn(x, adj)

        # 4) 频域分解得到低频/高频序列
        low, high = frequency_decompose(x, freq_cutoff=self.freq_cutoff, freq_ratio=self.freq_ratio)
        # 5) 双通道时间编码
        low = self.low_encoder(low)
        high = self.high_encoder(high)

        # 6) 融合低频与高频表征
        fused = torch.cat([low, high], dim=-1)
        fused = self.fusion(fused)

        # 7) 解码得到预测序列
        decoded = self.decoder(fused)
        # 8) 投影到输出维度
        out = self.output_projection(decoded)

        return out
