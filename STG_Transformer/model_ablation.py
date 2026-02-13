import torch
import torch.nn as nn

from .layers import (
    GraphConvStack,
    TemporalEncoder,
    TemporalDecoder,
    compute_pearson_adjacency,
    frequency_decompose,
)


class STGAblationModel(nn.Module):
    """
    STG ablation model with explicit switches:
    - M1: temporal Transformer encoder-decoder
    - M2: dynamic Pearson adjacency
    - M3: frequency decomposition branch
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
        use_m1=True,
        use_m2=True,
        use_m3=True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.pred_len = pred_len
        self.freq_ratio = freq_ratio
        self.freq_cutoff = freq_cutoff
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.use_m1 = use_m1
        self.use_m2 = use_m2
        self.use_m3 = use_m3

        self.input_projection = nn.Linear(n_features, d_model)
        self.spatial_gnn = GraphConvStack(d_model, n_layers=gnn_layers, dropout=dropout)

        if self.use_m1:
            if self.use_m3:
                self.low_encoder = TemporalEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    n_layers=enc_layers,
                    dropout=dropout,
                )
                self.high_encoder = TemporalEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    n_layers=enc_layers,
                    dropout=dropout,
                )
                self.fusion = nn.Linear(d_model * 2, d_model)
            else:
                self.encoder = TemporalEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    n_layers=enc_layers,
                    dropout=dropout,
                )

            self.decoder = TemporalDecoder(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_layers=dec_layers,
                pred_len=pred_len,
                dropout=dropout,
            )
            self.output_projection = nn.Linear(d_model, n_features)
        else:
            # Lightweight temporal head for Backbone-only setting.
            self.no_m1_head = nn.Linear(d_model, pred_len * n_features)

    def _identity_adjacency(self, x):
        batch_size, _, n_nodes, _ = x.shape
        eye = torch.eye(n_nodes, device=x.device, dtype=x.dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x):
        # x: (B, T, N, F)
        if self.use_m2:
            adj = compute_pearson_adjacency(
                x,
                top_k=self.top_k,
                threshold=self.corr_threshold,
            )
        else:
            adj = self._identity_adjacency(x)

        x = self.input_projection(x)
        x = self.spatial_gnn(x, adj)

        if self.use_m1:
            if self.use_m3:
                low, high = frequency_decompose(
                    x, freq_cutoff=self.freq_cutoff, freq_ratio=self.freq_ratio
                )
                low = self.low_encoder(low)
                high = self.high_encoder(high)
                memory = self.fusion(torch.cat([low, high], dim=-1))
            else:
                memory = self.encoder(x)

            decoded = self.decoder(memory)
            out = self.output_projection(decoded)
            return out

        # Backbone only: direct one-step temporal projection from last hidden state.
        bsz, _, n_nodes, _ = x.shape
        last_hidden = x[:, -1, :, :]
        out = self.no_m1_head(last_hidden).reshape(
            bsz, n_nodes, self.pred_len, self.n_features
        )
        return out
