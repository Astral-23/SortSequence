import torch
import torch.nn as nn


# TransformerSortNet 定義
class TransformerSortNet(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, num_layers=2, dim_ff=128):
        super().__init__()
        # 各要素 scalar→高次元埋め込み
        self.embed = nn.Linear(1, d_model)
        # 位置エンコーディング（学習可能にしてもOK）
        self.pos_embed = nn.Parameter(torch.randn(seq_len, d_model))
        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 出力 projection
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, L]
        B, L = x.size()
        x = x.unsqueeze(-1)  # → [B, L, 1]
        h = self.embed(x)  # → [B, L, d_model]
        h = h + self.pos_embed  # 位置情報を加算: broadcast [L, d_model]
        h = self.encoder(h)  # → [B, L, d_model]
        out = self.out_proj(h)  # → [B, L, 1]
        return out.squeeze(-1)  # → [B, L]

