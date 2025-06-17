import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def suggest_embed_dim(cardinality: int) -> int:
    return max(4, int(round(1.6 * cardinality ** 0.56)))


class AttnPooling(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask):
        # h: [B, T, H], mask: [B, T]
        scores = self.attn(h).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        pooled = torch.sum(h * alpha, dim=1)  # [B, H]
        return pooled, alpha.squeeze(-1)


class GRUWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        embed_dim = suggest_embed_dim(vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.pool = AttnPooling(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        packed_out, _ = self.rnn(emb)
        pooled, alpha = self.pool(packed_out, mask)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        return logits, alpha
