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
    from fairseq.data import FairseqDataset, Dictionary, LanguagePairDataset
    from fairseq.data import data_utils as fairseq_data_utils
    from fairseq.tasks.translation import TranslationTask
    from fairseq.tasks import register_task
except ImportError:
    assert False, "Fairseq is needed for seqp integration with fairseq!"

import numpy as np
import os
from typing import Iterable, Optional
from tqdm import tqdm

from seqp.vocab import Vocabulary, VocabularyCollector
from seqp.record import RecordWriter, ShardedWriter
from seqp.hdf5 import Hdf5RecordWriter
from seqp.util import count_lines


def _write_vocabs(train_prefix, src, tgt, destdir, joined_vocabs):
    import itertools

    def create_vocab(sentences: Iterable[str]) -> Vocabulary:
        builder = VocabularyCollector()
        for sentence in sentences:
            for symbol in sentence.strip().split(" "):
                builder.add_symbol(symbol)
        vocab = builder.consolidate()
        return vocab

    src_file = f"{train_prefix}.{src}"
    tgt_file = f"{train_prefix}.{tgt}"

    if joined_vocabs:
        with open(src_file) as f1, open(tgt_file) as f2:
            sentences = itertools.chain(f1, f2)
            vocab = create_vocab(sentences)

        vocab_file = os.path.join(destdir, 'joint_vocab.json')
        with open(vocab_file, 'w') as f:
            f.write(vocab.to_json(indent=4))
    else:
        with open(src_file) as f:
            src_vocab = create_vocab(f)
        src_vocab_file = os.path.join(destdir, 'vocab.json')
        with open(src_vocab_file, 'w') as f:
            f.write(src_vocab.to_json(indent=4))
        with open(tgt_file) as f:
            tgt_vocab = create_vocab(f)
        tgt_vocab_file = os.path.join(destdir, f'vocab.{}')
        with open(tgt_vocab_file, 'w') as f:
            f.write(tgt_vocab.to_json(indent=4))

    return src_vocab, tgt_vocab


def _create_writer(file_prefix: str,
                   suffix: str,
                   num_sentences: int,
                   max_records_per_shard: int) -> RecordWriter:
    if num_sentences < max_records_per_shard:
        return Hdf5RecordWriter(f"{file_prefix}.{suffix}.hdf5")
    else:
        return ShardedWriter(Hdf5RecordWriter,
                             file_prefix + ".{}." + suffix + ".hdf5",
                             max_records_per_shard)


def main():
    import argparse

    parser = argparse.ArgumentParser("Data preparation for fairseq+seqp")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--joined-dictionary", action="store_true")
    parser.add_argument("--trainpref", required=True)
    parser.add_argument("--validpref", required=True)
    parser.add_argument("--testpref", required=True)
    parser.add_argument("--destdir", required=True)
    parser.add_argument("--max-shard", type=int, default=400000)
    args = parser.parse_args()

    prefixes = {'train': args.trainpref,
                'valid': args.validpref,
                'test': args.testpref}

    src_vocab, tgt_vocab = _write_vocabs(args.trainpref,
                                         args.source,
                                         args.target,
                                         args.destdir,
                                         args.joined_dictionary)

    vocabs = {args.source: src_vocab, args.target:tgt_vocab}

    for split, prefix in prefixes.items():
        for lang in [args.source, args.target]:
            vocab = vocabs[lang]
            input_filename = os.path.join(prefix + "." + lang)
            output_prefix = os.path.join(args.destdir, split)
            num = count_lines(input_filename)
            with open(input_filename) as sentences, \
                    _create_writer(output_prefix, lang, num, args.max_shard) as writer:
                for idx, sentence in tqdm(enumerate(sentences), total=num):
                    tokenized_text = sentence.strip().split("")
                    token_ids = vocab.encode(tokenized_text, use_unk=False, add_eos=True)
                    writer.write(idx, np.array(token_ids))


if __name__ == '__main__':
    main()
