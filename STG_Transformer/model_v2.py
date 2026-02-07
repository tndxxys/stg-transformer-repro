import torch
import torch.nn as nn

from .layers import (
    GraphConvStack,
    TemporalEncoder,
    TemporalDecoder,
    compute_pearson_adjacency,
    frequency_decompose,
)


class _GlobalTemporalBranch(nn.Module):
    """Temporal self-attention branch over time for each node."""

    def __init__(self, d_model, n_heads, d_ff, n_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        # x: (B, T, N, D) -> (B*N, T, D)
        b, t, n, d = x.shape
        out = x.reshape(b * n, t, d)
        for layer in self.layers:
            out = layer(out)
        return out.reshape(b, t, n, d)


class _ChannelAttentionBranch(nn.Module):
    """Channel attention branch over nodes for each time step."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, N, D) -> (B*T, N, D)
        b, t, n, d = x.shape
        out = x.reshape(b * t, n, d)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        out = self.ffn_norm(out + self.ffn(out))
        return out.reshape(b, t, n, d)


class STGTransformerV2(nn.Module):
    """
    STG-Transformer V2
    - Keeps original spatial + frequency decomposition pipeline.
    - Adds dual-branch attention fusion:
      1) temporal global branch
      2) channel attention branch
    - Fuses by learnable gate.
    """

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
        corr_threshold=None,
        fusion_layers=1,
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

        self.global_branch = _GlobalTemporalBranch(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=fusion_layers, dropout=dropout
        )
        self.channel_branch = _ChannelAttentionBranch(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.decoder = TemporalDecoder(d_model, n_heads, d_ff, dec_layers, pred_len, dropout)
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: (B, T, N, F)
        adj = compute_pearson_adjacency(
            x,
            top_k=self.top_k,
            threshold=self.corr_threshold,
        )
        x = self.input_projection(x)
        x = self.spatial_gnn(x, adj)

        low, high = frequency_decompose(
            x, freq_cutoff=self.freq_cutoff, freq_ratio=self.freq_ratio
        )
        low = self.low_encoder(low)
        high = self.high_encoder(high)

        base = self.fusion(torch.cat([low, high], dim=-1))
        global_feat = self.global_branch(base)
        channel_feat = self.channel_branch(base)

        mix_gate = self.gate(torch.cat([global_feat, channel_feat], dim=-1))
        fused = mix_gate * channel_feat + (1.0 - mix_gate) * global_feat

        decoded = self.decoder(fused)
        out = self.output_projection(decoded)
        return out
