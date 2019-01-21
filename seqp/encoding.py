# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import numpy as np
from typing import Optional, List, Iterable, Callable


class TextCodec:
    """
    Abstract class with the contract of a text encoder. It can
    tokenize text and encode it as a numpy array, plus the
    opposite operations, that is, decode and detokenize.
    """

    def tokenize(self, sentence: str) -> List[str]:
        """
        Takes a piece of text and splits it into substrings.
        :param sentence: Text to split.
        :return: Split text.
        """
        raise NotImplementedError

    def encode(self, tokens: List[str]) -> Optional[np.ndarray]:
        """
        Encodes a tokenized text as a numpy array. The encoding
        can generate whatever data type, e.g. integer indices to
        a vocabulary table, embedded floating point vectors.
        :param tokens: Tokenized text.
        :return: Encoded numpy array. It can have any number of
                 dimensions as long as shape[0] == len(tokens)
        """
        raise NotImplementedError

    def decode(self, encoded: np.ndarray) -> List[str]:
        """
        Decodes a numpy array into
        :param encoded: Encoded numpy array.
        :return: Decoded text substrings (i.e. tokens), with
                 len(tokens) == encoded.shape[0]
        """
        raise NotImplementedError

    def detokenize(self, tokens: List[str]) -> str:
        """
        Assembles the given list of tokens into a single piece of text.
        :param tokens: Tokens to assemble.
        :return: Detokenized text string.
        """
        raise NotImplementedError


def record_generator(codec: TextCodec,
                     sentence_generator: Iterable[str],
                     progress: Callable[[int, str], None]=None):
    """
    Utility function to turn a text generator into a record generator
    by means of a TextCodec.
    :param codec: Codec to use to encode the text.
    :param sentence_generator: generator of the text to encode.
    :param progress: callable to report progress.
    :return: Record generator (useful for RecordWriter's).
    """
    if progress is None:
        def progress(_1, _2):
            pass

    for index, sentence in enumerate(sentence_generator):
        sentence = sentence.strip()
        tokens = codec.tokenize(sentence)
        encoded = codec.encode(tokens)
        progress(index, sentence)
        yield index, encoded
