# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

try:
    from fairseq.data import Dictionary
except ImportError:
    assert False, "Fairseq is needed for seqp integration with fairseq!"

from seqp.vocab import Vocabulary


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
