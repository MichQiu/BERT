"""Pretrain transformer with Masked LM and Next Sentence Prediction"""

from random import randint, shuffle
from random import random as rand
import fire

import torch
import torch.nn as nn
import torch.optim as op
from tensorboardX import SummaryWriter # tensorboard for pytorch

import tokenizer
import model
import optim
import train

from model import BERT
from utils import set_seeds, get_device, get_random_word, truncate_tokens_pair

# Input file format:
# 1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text.
#    (Because we use the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task
#    doesn't span between documents

def seek_random_offset(f, back_margin=2000):
    """seek random offset of file pointer"""
    # sets file's current position at the offset (2 = seek relative to the file's end)
    f.seek(0, 2) # f.seek(0, 2) = 0th position relative to the file's end (set current position at the end)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin # current position (end of file) - 2000
    # set file reading range from the start of the file to a randint between (0, max_offset)
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence


class SentPairDataLoader():
    """Load sentence pair (sequential or random order) from corpus"""
    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super(SentPairDataLoader, self).__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample, ignore encoding error
        self.f_neg = open(file, "r", encoding='utf-8', errors='ignore') # for a negative (random) sample
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """Read tokens from file pointer with limited length"""
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob else int(self.max_len / 2)

                is_next = rand() < 0.5 # whether tokens_b is next to tokens_a or not (NSP)

                tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
                seek_random_offset(self.f_neg)
                f_next = self.f_pos if is_next else self.f_neg
                tokens_b = self.read_tokens(f_next, len_tokens, False)

                if tokens_a is None or tokens_b is None: # end of file
                    self.f_pos.seek(0, 0) # reset file pointer
                    return

                instance = (is_next, tokens_a, tokens_b)
                for proc in self.pipeline: # preprocess sentence pairs
                    instance = proc(instance)

                batch.append(instance) # append to batch

            # Convert To Tensors
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline():
    """Preprocess Pipeline class: callable"""
    def __init__(self):
        super(Pipeline, self).__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Preprocess4Pretrain(Pipeline):
    """Preprocessing steps for pretraining transformer"""
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super(Preprocess4Pretrain, self).__init__()
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (subwords)
        self.indexer = indexer # function of token to index
        self.max_len = max_len # max seq length

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3 in max_length for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add special tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        # segment id of 0 for [CLS] + tokens_a + [SEP} and 1 for tokens_b + [SEP]
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1) # [0, 0, ..., 0, 1, 1, ..., 1]
        # mask all tokens as inputs
        input_mask = [1]*len(tokens) # [1, 1, ..., 1]

        # For masked language models
        masked_tokens, masked_pos = [], []
        # the number of predictions is sometimes less than max_pred when sequence is short
        # min between max_pred and the (max between 1 and the number of tokens masked)
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens) if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = '[MASK]'
            else:
                if rand() < 0.5:
                    tokens[pos] = get_random_word(self.vocab_words)
                else:
                    continue
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens) # [1, 1, ..., 1]

        # Token Indexing
        input_ids = self.indexer(tokens) # index all tokens
        masked_ids = self.indexer(masked_tokens) # index only masked tokens

        # Zero Padding
        n_pad = self.max_len - len(input_ids) # number of zero paddings required
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)


class BertLM(nn.Module):
    """Bert model for pretrain: Masked LM and Next Sentence Prediction"""
    def __init__(self, cfg):
        super(BertLM, self).__init__()
        self.bert = BERT
        self.fc_nsp = nn.Linear(cfg.dim, cfg.dim) # Linear layers
        self.fc_mlm = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh() # Tanh activation for NSP
        self.activ2 = nn.GELU() # GELU activation for MLM
        self.layer_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.classifier_nsp = nn.Linear(cfg.dim, 2)
        # Masked LM prediction has the same output embedding weights as input but different output-only bias
        # for each token
        self.decoder_mlm = nn.Linear(cfg.dim, cfg.n_vocab, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        # Obtain sequence outputs through BERT layers
        h = self.bert(input_ids, segment_ids, input_mask) # [batch, seq_len, dim]
        pooled_h = self.activ1(self.fc_nsp(h[:, 0])) # forward [CLS] tokens through linear layer (NSP), [batch, dim]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1)) # [batch, max_pred, dim]
        # select masked word positions from final output, -1 means not changing the dim size
        h_masked = torch.gather(h, 1, masked_pos) # masking position, [batch, max_pred, dim]
        h_masked = self.layer_norm(self.activ2(self.fc_mlm(h_masked))) # Linear + gelu + layer normalization
        logits_mlm = self.decoder_mlm(h_masked) + self.decoder_bias # [batch, max_pred, n_vocab], projection to vocab
        logits_nsp = self.classifier_nsp(pooled_h) # [batch, 2]

        return logits_mlm, logits_nsp


def main(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/pretrain',
         log_dir='../exp/bert/pretrain/runs',
         max_len=512,
         max_pred=20,
         mask_prob=0.15):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = model.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    # Tokenize text
    tokenizer_ = tokenizer.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer_.tokenize(tokenizer_.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred, mask_prob, list(tokenizer_.vocab.keys()),
                                    tokenizer_.convert_tokens_to_ids, max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    BertModel = BertLM(model_cfg)
    # CrossEntropyloss includes both nn.LogSoftmax and nn.NLLLoss for training loss on n-class classification
    criterion1 = nn.CrossEntropyLoss(reduction='none') # no reduction will be applied
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.ScheduledOptim(op.Adam(BertModel.parameters(), betas=(0.9, 0.98), eps=1e-09), cfg.dim, 4000)
    trainer = train.Trainer(cfg, BertModel, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure the loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_mlm, logits_nsp = BertModel(input_ids, segment_ids, input_mask, masked_pos)
        # Masked LM, masked_ids is the target
        loss_mlm = criterion1(logits_mlm.view(-1, logits_mlm.size(-1)), masked_ids.view(-1)) # [batch * max_pred]
        # or can also be criterion1(logits_mlm.transpose(-2, -1), y), [batch, max_pred]
        loss_mlm = (loss_mlm*masked_weights.float()).mean() # obtain mean loss
        loss_nsp = criterion2(logits_nsp, is_next) # Next sentence prediction
        # Graph loss functions on tensorboardX
        writer.add_scalars('data/scalar_group',
                           {'loss_mlm': loss_mlm.item(),
                            'loss_nsp': loss_nsp.item(),
                            'loss_total': (loss_mlm + loss_nsp).item(),
                            'lr': optimizer._optimizer.get_lr()[0]},
                           global_step)

        return loss_mlm + loss_nsp

    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)





