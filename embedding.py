import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, model_d, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, model_d))

    def _get_sinusoid_encoding_table(self, n_position, model_d):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        # n_position: maximum sentence length

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (dim // 2) / model_d) for dim in range(model_d)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class SegmentEmbedding(nn.Embedding):
    """Segment embedding, split embeddings according to the number of segments provided"""
    def __init__(self, n_segment, embed_size=512):
        # 2 segments, 512 dimensions each
        super(SegmentEmbedding, self).__init__(n_segment, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    """Regular input word embeddings based on word tokens"""
    def __init__(self, n_vocab, embed_size=512):
        super(TokenEmbedding, self).__init__(n_vocab, embed_size, padding_idx=0)


class BertEmbedding(nn.Module):
    """
    BERT embedding consisting the sum of three type of embeddings
    TokenEmbedding: normal word embedding matrix
    SegmentEmbedding: embedding matrix containing sentence segment info (sent_A:1, sent_B:2)
    PositionalEncoding: embedding matrix containing position info
    """

    def __init__(self, n_vocab, model_d, n_segment, n_sent, dropout=0.1):
        """
        Args:
            n_vocab: number of embeddings
            model_d: dimension of embeddings
            n_segment: number of sentence segments
            n_sent: maximum sentence length
        """

        super(BertEmbedding, self).__init__()
        self.token = TokenEmbedding(n_vocab, embed_size=model_d)
        self.segment = SegmentEmbedding(n_segment, embed_size=model_d)
        self.position = PositionalEncoding(model_d, n_sent)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        # Sum the three embedding matrix together
        x = self.token(sequence) + self.segment(segment_label) + self.position(sequence)
        return self.dropout(x)

