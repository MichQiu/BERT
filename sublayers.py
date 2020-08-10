import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

class ScaleDotProductAttention(nn.Module):

    def __init__(self, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))

        if mask is not None:
            attn = attn.masked_fill_(mask, -1e9)

        attn = self.dropout(F.softmax(dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaleDotProductAttention()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        residual = q
        len_q = q.size(1)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = [l(x).view(batch_size, len_q, self.h, self.d_k).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]

        # Head broadcasting with extra dimension
        if mask is not None:
            mask = mask.unsqueeze(1)

        # (B, H, S, W) * (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S) * (B, H, S, W) -> (B, H, S, W)
        b, attn = self.attention(q, k, v, mask=mask)
        # (B, H, S, W) -trans-> (B, S, H, W) -view-> (B, S, W)
        b = b.transpose(1, 2).contiguous().view(batch_size, len_q, self.h * self.d_k)
        b = self.dropout(self.fc(q))
        b += residual

        b = self.layer_norm(b)

        return b, attn


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, model_d, ffn_d, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_d, ffn_d)
        self.w_2 = nn.Linear(ffn_d, model_d)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_d, eps=1e-6)

    def forward(self, sequence):
        residual = sequence

        # (B, S, D) -> (B, S, D_ffn) -> (B, S, D)
        sequence = self.dropout(self.w_2(F.gelu(self.w_1)))
        sequence += residual
        sequence = self.layer_norm(sequence)

        return sequence



