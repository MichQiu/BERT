import torch
import torch.nn as nn
import json
from typing import NamedTuple

from sublayers import MultiHeadAttention, PositionWiseFeedForward
from embedding import BertEmbedding

class Config(NamedTuple):
    """Configuration for BERT model"""
    n_vocab: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of hidden layer in Transformer encoder
    n_layers: int = 12 # Number of Hidden Layers
    n_heads: int = 12 # Number of heads in multi-head self-attention layers
    dim_ffn: int = 768*4 # Dimension of Intermediate Layers in Position-wise feed forward net
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of sentence segments
    n_sent: int = None # Length of sentences

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class TransformerBlock(nn.Module):
    """Transformer encoder blocks"""

    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(cfg.n_heads, cfg.dim)
        self.ffn = PositionWiseFeedForward(cfg.dim, cfg.dim_ffn)

    def forward(self, sequence, mask):
        sequence, _ = self.attention(sequence, sequence, sequence, mask=mask)
        sequence = self.ffn(sequence)
        return sequence


class BERT(nn.Module):
    """BERT model: Bidirectional Encoder Representations from Transformers"""

    def __init__(self, cfg):
        super(BERT, self).__init__()
        self.hidden = cfg.dim
        self.embedding = BertEmbedding(cfg.n_vocab, cfg.dim, cfg.n_segments, cfg.n_sent)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg.dim, cfg.dim_ffn, cfg.n_heads)
                                                 for _ in range(cfg.n_layers)])

    def forward(self, sequence, segment_label, mask):
        # embed input sequence
        sequence = self.embedding(sequence, segment_label)
        # Feed to transformer blocks
        for transformer in self.transformer_blocks:
            sequence = transformer(sequence, mask)

        return sequence


