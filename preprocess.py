import pickle
import tqdm
from collections import Counter
from torch.utils.data import Dataset
import torch
import random

class TorchVocab(object):
    """
    Defines a vocabulary object that will be used to numericalize a field
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens in the data used to build the Vocab
        token_to_idx = A collections.defaultdict instance mapping token strings to numerical identifiers
        idx_to_token = A list of token strings indexed by their numerical identifiers
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'], vectors=None, unk_init=None,
                 vectors_cache=None):
        """Create a Vocab object from a collections.Counter
        Args:
            counter: collections.Counter object
            max_size: The maximum size of the vocabulary or None is there is no maximum
            min_freq: The minimum frequency needed to include a token in the vocabulary.
            specials: The list of special tokens (e.g., padding or eos) that will be prepended to the vocabulary in
                      addition to an <unk> token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors or custom pretrained vectors
                     (see Vocab.load_vectors); or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors to zero vectors, can be any
                                 function that takes in a Tensor and returns a Tensor of the same size.
                                 Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1) # ensure that the minimum frequency is not less than 1

        self.idx_to_token = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for token in specials:
            del counter[token]

        # Add the total number of special tokens onto the maximum number of vocab in the data
        max_size = None if max_size is None else max_size + len(self.idx_to_token)

        # sort the token by frequency and alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda token: token[0]) # alphabetically
        words_and_frequencies.sort(key=lambda token: token[1], reverse=True) # frequency

        # adding the vocabs to the token list
        for word, freq in words_and_frequencies:
            # stop adding vocab to the token index list if list length is longer than maximum number of vocab
            if freq < min_freq or len(self.idx_to_token) == max_size:
                break
            self.idx_to_token.append(word)

        # token_to_idx is a reverse dict of idx_to_token
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        self.vectors = None
        if vectors is not None: # Load pretrained vectors
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    # equality condition
    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.token_to_idx != other.token_to_idx:
            return False
        if self.idx_to_token != other.idx_to_token:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    # check length of idx_to_token list
    def __len__(self):
        return len(self.idx_to_token)

    # Obtain token to index dict
    def vocab_rerank(self):
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    # Extend unseen tokens in self.idx_to_token and self.token_to_idx from imported Vocab object
    def extend(self, v, sort=False): # v: Vocab object
        words = sorted(v.idx_to_token) if sort else v.idx_to_token # return sorted version if sort=True
        for token in words:
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # new index = total length - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        """Set index for various special tokens"""
        self.pad_index = 0 # padding
        self.unk_index = 1 # unknown
        self.sos_index = 2 # start of sentence
        self.eos_index = 3 # end of sentence
        self.mask_index = 4 # masking
        super(Vocab, self).__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                                    max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    # load vocab object from file path
    # staticmethod is used to group functions which have some logical connection with a class to another class
    """
    Loading the vocab does not involve anything within the Vocab class but it is just convenient to store it in the
    class, hence staticmethod needs to be declared
    """
    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    # save Vocab object to file path
    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        # Preprocess text
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line # Obtain a list of words
            else:
                words = line.replace("\n", "").replace("\t", "").split() # remove any newlines and spaces

            # Add counts for each word
            for word in words:
                counter[word] += 1

        super(WordVocab, self).__init__(counter, max_size=max_size, min_freq=min_freq)

    # Convert raw sentence text into sequence
    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split() # split sentence into list of words if it is a string

        # get a list of token indices for every word in the sentence,
        seq = [self.token_to_idx.get(word, self.unk_index) for word in sentence]

        # If sos token is required
        if with_sos:
            seq = [self.sos_index] + seq # add sos index before seq

        # If eos token is required
        if with_sos:
            seq += [self.eos_index] # add eos index after seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        # if sentence sequence length is smaller than specified sequence length
        elif len(seq) < seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        # if sentence sequence length is larger, then only take the sentence sequence up to seq_length
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq # show original sequence length if with_len=True

    # Get word tokens from a sequence
    def from_seq(self, seq, join=False, with_pad=False):
        # If get words by index if index is smaller than idx_to_token length
        # Otherwise, add new token if index is not a padding index
        words = [self.idx_to_token[idx] if idx < len(self.idx_to_token)
                 else "<%d>" % idx for idx in seq if not with_pad or idx != self.pad_index]

        # join the tokens to form a sentence if join=True
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

# build Vocab and save
def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=str, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                # Get the total number of lines in the corpus
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            # Data stored on memory
            if on_memory:
                self.lines = [line[:-1].split("\t") for line in tqdm.tqdm(f, desc="Loading Dataset",
                                                                          total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    # Word masking
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = [] # returns an output label list with pre-masked token indices

        for i, token in enumerate(tokens):
            prob = random.random()
            # 15% likelihood to mask
            if prob < 0.15:
                prob /= 0.15 # new probability within the 15%

                # 80% randomly change token to MASK token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif 0.8 <= prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.token_to_idx.get(token, self.vocab.unk_index)

                # append the original token index to output label list (not masked, only include unk if not in vocab)
                output_label.append(self.vocab.token_to_idx.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.token_to_idx.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label


