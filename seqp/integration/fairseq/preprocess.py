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

import os
from typing import Iterable

from seqp.vocab import Vocabulary, VocabularyCollector


def _write_vocabs(args, src_file, tgt_file):
    import itertools

    def create_vocab(sentences: Iterable[str]) -> Vocabulary:
        builder = VocabularyCollector()
        for sentence in sentences:
            for symbol in sentence.strip().split(" "):
                builder.add_symbol(symbol)
        vocab = builder.consolidate()
        return vocab

    if args.joined_dictionary:
        with open(src_file) as f1, open(tgt_file) as f2:
            sentences = itertools.chain(f1, f2)
            vocab = create_vocab(sentences)

        vocab_file = os.path.join(args.destdir, 'joint_vocab.json')
        with open(vocab_file, 'w') as f:
            f.write(vocab.to_json(indent=4))
    else:
        with open(src_file) as f:
            src_vocab = create_vocab(f)
        src_vocab_file = os.path.join(args.destdir, 'src_vocab.json')
        with open(src_vocab_file, 'w') as f:
            f.write(src_vocab.to_json(indent=4))
        with open(tgt_file) as f:
            tgt_vocab = create_vocab(f)
        tgt_vocab_file = os.path.join(args.destdir, 'tgt_vocab.json')
        with open(tgt_vocab_file, 'w') as f:
            f.write(tgt_vocab.to_json(indent=4))

    return src_vocab, tgt_vocab


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
    args = parser.parse_args()

    prefixes = {'train': args.trainpref,
                'valid': args.validpref,
                'test': args.testpref}

    src_file, tgt_file = [os.path.join(args.trainpref, lang)
                          for lang in [args.source, args.target]]

    files = {args.source: src_file, args.target: tgt_file}

    src_vocab, tgt_vocab = _write_vocabs(args, src_file, tgt_file)

    for split, prefix in prefixes.items():
        for lang in [args.source, args.target]:
            with open(os.path.join(prefix + "." + lang)) as f:
                args.destdir


if __name__ == '__main__':
    main()
