# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from collections import Counter, defaultdict
import json
from typing import Iterable, List, Callable, Optional

from .encoding import TextCodec

DEFAULT_UNK = '<unk>'
DEFAULT_PAD = '<pad>'
DEFAULT_EOS = '</s>'
DEFAULT_SPECIAL_SYMBOLS = [DEFAULT_PAD, DEFAULT_EOS, DEFAULT_UNK]
DEFAULT_PAD_ID = DEFAULT_SPECIAL_SYMBOLS.index(DEFAULT_PAD)
DEFAULT_UNK_ID = DEFAULT_SPECIAL_SYMBOLS.index(DEFAULT_UNK)
DEFAULT_EOS_ID = DEFAULT_SPECIAL_SYMBOLS.index(DEFAULT_EOS)


class Vocabulary(object):
    """
    Keeps direct and reverse token - token ID mappings, and provides functions
    to encode text and decode token IDs.
    """

    def __init__(self,
                 symbols: List[str],
                 pad_id: Optional[int]=None,
                 eos_id: Optional[int]=None,
                 unk_id: Optional[int]=None,
                 ):
        """
        Constructs a Vocabulary.
        :param symbols: List of symbols that are part of the vocabulary.
        :param pad_id: index of the PAD symbol in the symbols list.
        :param eos_id: index of the EOS symbol in the symbols list.
        :param unk_id: index of the UNK symbol in the symbols list.
        """
        # PAD, UNK and EOS must be at the beginning of the list
        pad_id = pad_id if pad_id is not None else symbols.index(DEFAULT_PAD)
        eos_id = eos_id if eos_id is not None else symbols.index(DEFAULT_EOS)
        unk_id = unk_id if unk_id is not None else symbols.index(DEFAULT_UNK)
        self.num_special = 3
        assert unk_id < self.num_special, "UNK must be at the beginning of the list"
        assert pad_id < self.num_special, "PAD must be at the beginning of the list"
        assert eos_id < self.num_special, "EOS must be at the beginning of the list"

        self.idx2symbol = list(symbols)
        self.symbol2idx = {}
        self.symbol2idx = {s: i for i, s in enumerate(self.idx2symbol)}
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.muted_ids = [self.eos_id, self.pad_id]

    def encode(self, tokenized_text: Iterable[str], use_unk=False, add_eos=True)->List[int]:
        """
        Encodes a piece of tokenized text into the associated token IDs.
        :param tokenized_text: Tokenized text to encode.
        :param use_unk: If True, the UNK symbol ID is used if a token is not
                        part of the vocabulary. Otherwise, the token is ignored.
        :param add_eos: Adds the EOS symbol ID at the end of the sequence.
        :return: List of token IDs.
        """
        unk_id = self.unk_id if use_unk else None
        eos_trail = [self.eos_id] if add_eos else []
        return [idx
                for idx in (self.symbol2idx.get(s, unk_id) for s in tokenized_text)
                if idx is not None] + eos_trail

    def decode(self, indexes: Iterable[int]) -> List[str]:
        """
        Decodes a list of token IDs into the associated tokens.
        :param indexes: token IDs to decode.
        :return: Tokenized text.
        """
        return [self.idx2symbol[idx] if idx not in self.muted_ids else ""
                for idx in indexes]

    def to_json(self) -> str:
        """
        Creates a json string with the internal state of the vocabulary.
        :return: Json string.
        """
        state = {'pad_id': self.pad_id,
                 'eos_id': self.eos_id,
                 'unk_id': self.unk_id,
                 'symbols': self.idx2symbol,
                 }
        return json.dumps(state)

    @staticmethod
    def from_json(json_string: str) -> 'Vocabulary':
        """
        Creates a Vocabulary instance from the Json serialized string.
        :param json_string: String with the internal state of the Vocabulary.
        :return: The reconstructed vocabulary object.
        """
        # To understand the return value, take into account that we
        # support Python 3.6 and read PEP 484:
        # https://www.python.org/dev/peps/pep-0484/#forward-references
        state = json.loads(json_string)
        return Vocabulary(symbols=state['symbols'],
                          pad_id=state['pad_id'],
                          eos_id=state['eos_id'],
                          unk_id=state['unk_id'],)


class VocabularyCollector(object):
    """
    Collects symbols and creates a Vocabulary with them, optionally taking only
    the most frequent symbols.
    """

    def __init__(self):
        """
        Constructs a VocabularyCollector.
        """
        self.symbol_count = defaultdict(int)

    def add_symbol(self, symbol: str):
        """
        Adds a symbol to the list of possible symbols in the Vocabulary.
        :param symbol:
        :return: None
        """
        self.symbol_count[symbol] += 1

    def consolidate(self,
                    max_num_symbols: int=None,
                    unk_symbol: str=DEFAULT_UNK,
                    pad_symbol: str=DEFAULT_PAD,
                    eos_symbol: str=DEFAULT_EOS,
                    sorting_key=None,
                    ) -> Vocabulary:
        """
        Creates a Vocabulary from the collected symbols.
        :param max_num_symbols: number of symbols in the resulting Vocabulary. If
                                None, it includes all symbols collected.
        :param unk_symbol: Symbol to use for UNK tokens.
        :param pad_symbol: PAD symbol.
        :param eos_symbol: End-of-sequence symbol.
        :param sorting_key: key to sort the symbols, or None if order
                            is not important.
        :return: The resulting Vocabulary.
        """
        counter = Counter(self.symbol_count)
        special_symbols = [pad_symbol, eos_symbol, unk_symbol]
        num_symbols = (None if max_num_symbols is None
                       else max_num_symbols - len(special_symbols))
        most_common_symbols = counter.most_common(num_symbols)
        symbols = [symbol for symbol, count in most_common_symbols]

        if sorting_key is not None:
            symbols = sorted(symbols, sorting_key)

        symbols = special_symbols + symbols
        return Vocabulary(symbols, pad_id=0, eos_id=1, unk_id=2)


def vocab_to_codec(vocab: Vocabulary,
                   tokenize: Callable[[str], List[str]],
                   detokenize: Callable[[List[str]], str],
                   use_unk=False,
                   add_eos=True,
                   ) -> TextCodec:
    """
    Utility function to create a TextCodec from a Vocabulary object.
    :param vocab: Vocabulary to use for encoding text.
    :param tokenize: tokenization function.
    :param detokenize: detokenization function.
    :param use_unk: Whether to use the UNK symbol for out-of-vocabulary
                    tokens.
    :param add_eos: Whether to include the EOS symbol at the
                    end of the sequences.
    :return: The resulting TextCodec.
    """
    import numpy as np

    class VocabCodec(TextCodec):
        def encode(self, tokens: List[str]) -> Optional[np.ndarray]:
            return vocab.encode(tokens, use_unk, add_eos)

        def decode(self, encoded: np.ndarray) -> List[str]:
            return vocab.decode(encoded)

        def tokenize(self, sentence: str) -> List[str]:
            return tokenize(sentence)

        def detokenize(self, tokens: List[str]) -> str:
            return detokenize(tokens)

    return VocabCodec()
