# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

try:
    import torch
except ImportError:
    assert False, "Pytorch is needed for seqp integration with fairseq"

try:
    from fairseq.data import FairseqDataset, Dictionary
    from fairseq.data import data_utils as fairseq_data_utils
except ImportError:
    assert False, "Fairseq is needed for seqp integration with fairseq!"

import numpy as np

from ..record import RecordReader
from ..vocab import Vocabulary


class MonolingualDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self,
                 dictionary: Dictionary,
                 reader: RecordReader,
                 left_pad=True,
                 move_eos_to_beginning=False,
                 sequence_field=None):
        self.dictionary = dictionary
        self.reader = reader
        self.left_pad = left_pad
        self.move_eos_to_beginning = move_eos_to_beginning
        if reader.fields:
            self._get_seq = lambda record: record[reader.sequence_field]
        else:
            self._get_seq = lambda record: record

    def __getitem__(self, index):
        elem = self._get_seq(self.reader.retrieve(index))
        if isinstance(elem, np.ndarray):
            elem = torch.from_numpy(elem)
        return elem

    def __len__(self):
        return self.reader.num_records()

    def collater(self, samples):
        tokens = fairseq_data_utils.collate_tokens(
                        [s for s in samples],
                        self.dictionary.pad_index,
                        self.dictionary.eos_index,
                        self.left_pad,
                        move_eos_to_beginning=self.move_eos_to_beginning)
        lengths = torch.LongTensor([s.numel() for s in samples])
        lengths, sort_order = lengths.sort(descending=True)
        tokens = tokens.index_select(0, sort_order)
        return tokens

    def get_dummy_batch(self, num_tokens, max_positions):
        return self.dictionary.dummy_sentence(num_tokens)

    def num_tokens(self, index):
        return self.reader.length(index)

    def size(self, index):
        return self.reader.length(index)

    def ordered_indices(self):
        return [idx for idx, length in self.reader.indexes_and_lengths()]

    @property
    def sizes(self):
        return [length for idx, length in self.reader.indexes_and_lengths()]

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError


def vocab_to_dictionary(vocab: Vocabulary) -> Dictionary:
    """
    Creates a fairseq Dictionary from a seqp Vocabulary. It manipulates
    the Dictionary's internal state to avoid reserving token 0 for Lua
    compatibility in order to respect the token ID associations in the
    original Vocabulary.

    :param vocab: Vocabulary to convert to Dictionary.
    :return: Resulting Dictionary.
    """
    pad_symbol = vocab.idx2symbol[vocab.pad_id]
    eos_symbol = vocab.idx2symbol[vocab.eos_id]
    unk_symbol = vocab.idx2symbol[vocab.unk_id]

    dictionary = Dictionary(pad=pad_symbol, unk=unk_symbol, eos=eos_symbol)

    # We clear up the internal state to write it from scratch (and without
    # the Lua heritage token zero, to keep token IDs)
    dictionary.symbols = []
    dictionary.count = []
    dictionary.indices = {}
    dictionary.nspecial = 3

    for symbol in vocab.idx2symbol:
        unknown_frequency = 1   # frequency info is not available
        dictionary.add_symbol(symbol, unknown_frequency)

    dictionary.pad_index = vocab.pad_id
    dictionary.eos_index = vocab.eos_id
    dictionary.unk_index = vocab.unk_id

    return dictionary
