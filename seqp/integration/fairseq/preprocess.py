# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.


import numpy as np
import os
from typing import Iterable
from tqdm import tqdm

from seqp.vocab import Vocabulary, VocabularyCollector
from seqp.record import RecordWriter, ShardedWriter
from seqp.hdf5 import Hdf5RecordWriter
from seqp.util import count_lines


def _create_vocab(sentences: Iterable[str],
                  num_tokens: int) -> Vocabulary:
    """
    Creates a Seqp vocabulary from the given sentences (which are assumed
    to be tokenized, i.e. blank-separated tokens) and maximum number
    of tokens.
    :param sentences: Sentences (already tokenized) to create the vocabulary.
    :param num_tokens: Maximum number of tokens in the vocabulary.
    :return: A newly created vocabulary.
    """
    builder = VocabularyCollector()
    for sentence in sentences:
        for symbol in sentence.strip().split(" "):
            builder.add_symbol(symbol)

    if num_tokens <= 0:
        num_tokens = None
    vocab = builder.consolidate(max_num_symbols=num_tokens)
    return vocab


def _write_vocabs(train_prefix: str,
                  src: str,
                  tgt: str,
                  destdir: str,
                  joined_vocabs: bool,
                  num_src_tokens: int,
                  num_tgt_tokens: int):
    """
    Writes source and target vocabularies (or a joint vocabulary)
    :param train_prefix: Prefix of the training files (except lang extension)
    :param src: Source language extension
    :param tgt: Target language extension
    :param destdir: Destination directory
    :param joined_vocabs: Whether to use a joint source-target vocabulary.
    :param num_src_tokens: Maximum number of tokens in the source vocabulary.
    :param num_tgt_tokens: Maximum number of tokens in the target vocabulary.
    :return: None
    """
    import itertools

    src_file = f"{train_prefix}.{src}"
    tgt_file = f"{train_prefix}.{tgt}"

    if joined_vocabs:
        num_tokens = num_src_tokens if num_src_tokens > 0 else num_tgt_tokens
        with open(src_file) as f1, open(tgt_file) as f2:
            sentences = itertools.chain(f1, f2)
            vocab = _create_vocab(sentences, num_tokens)

        vocab_file = os.path.join(destdir, f'vocab.{src}-{tgt}')
        with open(vocab_file, 'w') as f:
            f.write(vocab.to_json(indent=4))

        src_vocab = tgt_vocab = vocab
    else:
        with open(src_file) as f:
            src_vocab = _create_vocab(f, num_src_tokens)
        src_vocab_file = os.path.join(destdir, f'vocab.{tgt}')
        with open(src_vocab_file, 'w') as f:
            f.write(src_vocab.to_json(indent=4))
        with open(tgt_file) as f:
            tgt_vocab = _create_vocab(f, num_tgt_tokens)
        tgt_vocab_file = os.path.join(destdir, f'vocab.{src}')
        with open(tgt_vocab_file, 'w') as f:
            f.write(tgt_vocab.to_json(indent=4))

    return src_vocab, tgt_vocab


def _writer(file_prefix: str,
            lang: str,
            num_sentences: int,
            max_records_per_shard: int) -> RecordWriter:
    """
    Creates an appropriate RecordWriter object.
    :param file_prefix: File prefix where to store the data.
    :param lang: Language extension.
    :param num_sentences: Number of sentences to write to file (used
                          to compute shards).
    :param max_records_per_shard: Max amount of records per shard.
    :return: RecordWriter object.
    """
    if num_sentences < max_records_per_shard:
        return Hdf5RecordWriter(f"{file_prefix}.{lang}.hdf5")
    else:
        return ShardedWriter(Hdf5RecordWriter,
                             file_prefix + ".{}." + lang + ".hdf5",
                             max_records_per_shard)


def main():
    import argparse

    parser = argparse.ArgumentParser("Data preparation for fairseq+seqp")
    parser.add_argument("-s", "--source-lang",
                        required=True,
                        metavar="SRC",
                        help="source language")
    parser.add_argument("-t", "--target-lang",
                        required=True,
                        metavar="TARGET",
                        help="target language")
    parser.add_argument("--joined-dictionary", action="store_true")
    parser.add_argument("--trainpref", required=True)
    parser.add_argument("--validpref", required=True)
    parser.add_argument("--testpref", required=True)
    parser.add_argument("--nwordssrc",
                        metavar="N",
                        default=-1,
                        type=int,
                        help="number of source words to retain")
    parser.add_argument("--nwordstgt",
                        metavar="N",
                        default=-1,
                        type=int,
                        help="number of target words to retain")
    parser.add_argument("--destdir", required=True)
    parser.add_argument("--max-shard", type=int, default=400000)
    args = parser.parse_args()

    prefixes = {'train': args.trainpref,
                'valid': args.validpref,
                'test': args.testpref}

    os.makedirs(args.destdir, exist_ok=True)

    src_vocab, tgt_vocab = _write_vocabs(args.trainpref,
                                         args.source_lang,
                                         args.target_lang,
                                         args.destdir,
                                         args.joined_dictionary,
                                         args.nwordssrc,
                                         args.nwordstgt,
                                         )

    vocabs = {args.source_lang: src_vocab, args.target_lang: tgt_vocab}

    for split, prefix in prefixes.items():
        for lang in [args.source_lang, args.target_lang]:
            vocab = vocabs[lang]
            input_filename = f"{prefix}.{lang}"
            output_prefix = os.path.join(args.destdir, split)
            num_sents = count_lines(input_filename)
            with open(input_filename) as sentences, \
                    _writer(output_prefix, lang, num_sents, args.max_shard) as writer:
                for idx, sentence in tqdm(enumerate(sentences), total=num_sents):
                    tokenized_text = sentence.strip().split(" ")
                    token_ids = vocab.encode(tokenized_text, use_unk=False, add_eos=True)
                    writer.write(idx, np.array(token_ids))


if __name__ == '__main__':
    main()
