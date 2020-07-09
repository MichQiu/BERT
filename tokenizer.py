from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata

def convert_to_unicode(text):
    """Converts 'text' to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")  # ignore errors
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def printable_text(text):
    """Returns text endocded in a way suitable for print or 'tf.logging'."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")  # ignore errors
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary"""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline()) # readline keeps a trailing newline character in the string
            if not token:
                break
            token = token.strip() # strip all default whitespace characters
            vocab[token] = index
            index += 1
    return vocab

def convert_token_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(char):
    """Checks whether char is a whitespace character"""
    # \t, \n, and \r are technically control characters but they are treated as whitespace in this case
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs": # if unicode category is separator/space
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether char is a punctuation character"""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation
    # Characters such as "^", "$", and "'" are not in the Unicode Punctuation class but are treated as punctuation
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class FullTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordPieceTokenizer(self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(text):
                split_tokens.append(sub_token)

        return split_tokens

    # Encapsulate these two functions as methods in the class, no need to import
    def convert_tokens_to_ids(self, tokens):
        return convert_token_to_ids(self.vocab, tokens)

    def convert_to_unicode(self, text):
        return convert_to_unicode(text)

class BasicTokenizer(object):
    """Run basic tokenization (punctuation splitting, lower casing, etc."""
    def __init__(self, do_lower_case=True):
        """
        Constructs a BasicTokenizer
        Args:
            do_lower_case: Whether to lower case the input
        """
        self.do_lower_case: do_lower_case

    def tokenize(self, text):
        """Tokenize a piece of text"""
        text = convert_to_unicode(text) # convert to unicode
        text = self._clean_text(text) # remove invalid characters and whitespace cleanup
        orig_tokens = whitespace_tokenize(text) # split text into a list of tokens
        split_tokens = [] # a list containing split tokens
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token) # strip accents
            split_tokens.extend(self._run_split_on_punc(token)) # split punctuations from characters and extend

        output_tokens = whitespace_tokenize(" ".join(split_tokens)) #
        return output_tokens

    def _run_strip_accents(self, text):
        """Strip accents from a piece of text"""
        text = unicodedata.normalize("NFD", text) # normalize unicode string by canonical decomposition
        output = []
        for char in text:
            cat = unicodedata.category(char) # return the general category assigned to the character chr as string
            if cat == "Mn": # 'Mn' = Mark, non-spacing character category, e.g. accents
                continue
            output.append(char)
        return "".join(output) # join stripped text

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text"""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char]) # append punctuation into individual lists (elements)
                start_new_word = True
            else:
                if start_new_word:
                    output.append([]) # append new list for new word
                start_new_word = False
                output[-1].append(char) # append characters to the last(new) list
            i += 1

        return ["".join(x) for x in output] # join split text

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text"""
        output = []
        for char in text:
            cp = ord(char) # ord() returns an integer representing the Unicode point of the character (len = 1)
            if cp == 0 or cp == 0xfffd or _is_control(char): # skip invalid characters
                continue
            if _is_whitespace(char):
                output.append(" ") # append whitespace character
            else:
                output.append(char)
        return "".join(output)

class WordPieceTokenizer(object):
    """Runs WordPiece tokenization"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization using the given vocabulary

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should have already been passed through
            'BasicTokenizer'.

        Returns:
            A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text): # split into a list of tokens
            chars = list(token) # split into a list of characters
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token) # add unknown token if total character length is larger than set
                continue

            is_bad = False
            start = 0 # starting character
            sub_tokens = []
            while start < len(chars):
                end = len(chars) # end position is the last character
                cur_substr = None # current substring
                while start < end: # loop from start to end
                    substr = "".join(chars[start:end]) # join characters from start to end
                    if start > 0:
                        substr = "##" + substr # insert ## for any subwords starting after the first character
                    if substr in self.vocab:
                        cur_substr = substr # use vocab for subwords that are included in the vocabulary
                        break
                    end -= 1 # decrement end position by one to check next vocab/subword
                if cur_substr is None: # if subword is unknown
                    is_bad = True
                    break
                sub_tokens.append(cur_substr) # append subwords to sub_tokens list
                start = end # equate start and end positions to finish loop

            if is_bad:
                output_tokens.append(self.unk_token) # append unknown token
            else:
                output_tokens.append(sub_tokens) # append subtokens list
        return output_tokens


