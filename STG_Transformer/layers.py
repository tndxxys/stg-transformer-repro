import torch
import torch.nn as nn
import math


def compute_pearson_adjacency(x, top_k=None, threshold=None, eps=1e-6):
    """基于滑动窗口序列计算皮尔逊相关邻接矩阵"""
    # x: (B, T, N, F) -> 对每个样本的 N 个节点做相关性估计
    batch_size, seq_len, n_nodes, n_features = x.shape
    flat_len = max(seq_len * n_features, 1)

    # 将时间与特征维拼接为一个向量，便于计算节点间相关性
    x_flat = x.permute(0, 2, 1, 3).reshape(batch_size, n_nodes, flat_len)
    x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)

    # 协方差与方差，得到皮尔逊相关系数矩阵
    denom = max(flat_len - 1, 1)
    cov = torch.matmul(x_centered, x_centered.transpose(1, 2)) / denom
    var = torch.sum(x_centered ** 2, dim=-1) / denom
    std = torch.sqrt(var + eps)

    corr = cov / (std.unsqueeze(2) * std.unsqueeze(1) + eps)
    corr = torch.clamp(corr, -1.0, 1.0)
    # 图卷积要求非负权重，使用绝对相关作为邻接权重
    corr = corr.abs()

    if threshold is not None:
        # 基于相关性阈值进行稀疏化
        corr = torch.where(corr >= threshold, corr, torch.zeros_like(corr))

    if top_k is not None and top_k < n_nodes:
        # 基于 Top-K 进行稀疏化，保证对称性
        _, indices = torch.topk(corr, k=top_k + 1, dim=-1)
        mask = torch.zeros_like(corr)
        mask.scatter_(-1, indices, 1.0)
        mask = torch.maximum(mask, mask.transpose(1, 2))
        corr = corr * mask

    # 保留自环，保证数值稳定
    eye = torch.eye(n_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    corr = corr * (1.0 - eye) + eye
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return corr


def normalize_adjacency(adj, eps=1e-6):
    # 对称归一化：D^{-1/2} A D^{-1/2}
    batch_size, n_nodes, _ = adj.shape
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return deg_inv_sqrt @ adj @ deg_inv_sqrt


class GraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        # x: (B, T, N, D), adj: (B, N, N)
        adj_norm = normalize_adjacency(adj)
        # 基于邻接矩阵传播节点信息
        out = torch.einsum("bij,btjd->btid", adj_norm, x)
        out = self.linear(out)
        return self.dropout(self.act(out))


class GraphConvStack(nn.Module):
    """多层图卷积堆叠"""
    def __init__(self, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(d_model, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        # 多层残差堆叠，保持稳定训练
        for layer in self.layers:
            out = layer(x, adj)
            x = self.norm(x + out)
        return x


def frequency_decompose(x, freq_cutoff=None, freq_ratio=0.1):
    """频域分解为低频与高频分量"""
    # x: (B, T, N, D)
    seq_len = x.size(1)
    # MPS does not support rfft/irfft yet; run FFT on CPU then move back.
    use_cpu_fft = x.device.type == "mps"
    x_fft = torch.fft.rfft(x.to("cpu") if use_cpu_fft else x, dim=1)
    n_freq = x_fft.size(1)

    if freq_cutoff is None:
        freq_cutoff = max(1, int(n_freq * freq_ratio))
    freq_cutoff = min(freq_cutoff, n_freq)

    # 低频保留前半段，高频保留后半段
    low_fft = x_fft.clone()
    low_fft[:, freq_cutoff:] = 0
    high_fft = x_fft.clone()
    high_fft[:, :freq_cutoff] = 0

    low = torch.fft.irfft(low_fft, n=seq_len, dim=1)
    high = torch.fft.irfft(high_fft, n=seq_len, dim=1)
    if use_cpu_fft:
        low = low.to(x.device)
        high = high.to(x.device)
    return low, high


class TemporalEncoder(nn.Module):
    """时间编码器"""
    def __init__(self, d_model, n_heads, d_ff, n_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: (B, T, N, D) -> (B*N, T, D)
        batch_size, seq_len, n_nodes, d_model = x.shape
        x_reshaped = x.reshape(batch_size * n_nodes, seq_len, d_model)
        x_reshaped = self.pos_encoding(x_reshaped)
        for layer in self.layers:
            x_reshaped = layer(x_reshaped)
        # 恢复为 (B, T, N, D)
        return x_reshaped.reshape(batch_size, seq_len, n_nodes, d_model)


class TemporalDecoder(nn.Module):
    """时间解码器"""
    def __init__(self, d_model, n_heads, d_ff, n_layers, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.query_embed = nn.Parameter(torch.randn(pred_len, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, memory):
        # memory: (B, T, N, D)
        batch_size, seq_len, n_nodes, d_model = memory.shape
        memory = memory.reshape(batch_size * n_nodes, seq_len, d_model)
        # 使用 learnable query 作为解码器输入
        queries = self.query_embed.unsqueeze(0).expand(batch_size * n_nodes, -1, -1)
        queries = self.pos_encoding(queries)
        out = self.decoder(queries, memory)
        # 输出 (B, N, pred_len, D)
        return out.reshape(batch_size, n_nodes, self.pred_len, d_model)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
