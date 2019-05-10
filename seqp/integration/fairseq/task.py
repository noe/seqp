# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

try:
    from fairseq.data import LanguagePairDataset
    from fairseq.tasks.translation import TranslationTask
    from fairseq.tasks import register_task
except ImportError:
    assert False, "Fairseq is needed for seqp integration with fairseq!"

from glob import glob
import os

from seqp.hdf5 import Hdf5RecordReader
from seqp.vocab import Vocabulary
from seqp.integration.fairseq.dictionary import vocab_to_dictionary
from seqp.integration.fairseq.dataset import MonolingualDataset


@register_task('translation-seqp')
class SeqpTranslationTask(TranslationTask):
    def __init__(self, args, src_vocab, tgt_vocab):
        src_dict = vocab_to_dictionary(src_vocab)
        tgt_dict = vocab_to_dictionary(tgt_vocab)
        super().__init__(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        if 'valid' in split and split[-1] == '1':
            raise FileNotFoundError

        prefix = os.path.join(self.args.data,
                              'train' if split == 'train' else
                              'valid' if 'valid' in split else
                              'test')

        src_files = glob(f'{prefix}*.{self.args.source_lang}.hdf5')
        tgt_files = glob(f'{prefix}*.{self.args.target_lang}.hdf5')
        src_reader = Hdf5RecordReader(src_files)
        tgt_reader = Hdf5RecordReader(tgt_files)

        if not hasattr(self.args, 'max_source_positions'):
            self.args.max_source_positions = 1024
        if not hasattr(self.args, 'max_target_positions'):
            self.args.max_target_positions = 1024

        max_source_positions = self.args.max_source_positions
        max_target_positions = self.args.max_target_positions

        src = MonolingualDataset(self.src_dict,
                                 src_reader,
                                 left_pad=False,
                                 move_eos_to_beginning=False)

        tgt = MonolingualDataset(self.tgt_dict,
                                 tgt_reader,
                                 left_pad=False,
                                 move_eos_to_beginning=False)

        src_sizes = [l for idx, l in src_reader.indexes_and_lengths()]
        tgt_sizes = [l for idx, l in tgt_reader.indexes_and_lengths()]

        dataset = LanguagePairDataset(src,
                                      src_sizes,
                                      self.src_dict,
                                      tgt=tgt,
                                      tgt_sizes=tgt_sizes,
                                      tgt_dict=self.tgt_dict,
                                      left_pad_source=False,
                                      left_pad_target=False,
                                      max_source_positions=max_source_positions,
                                      max_target_positions=max_target_positions)

        self.datasets[split] = dataset

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = False
        args.left_pad_target = False
        src = args.source_lang
        tgt = args.target_lang

        data_dir = args.data
        joint_vocab_file = os.path.join(data_dir, f'vocab.{src}-{tgt}')
        src_vocab_filename = os.path.join(data_dir, f'vocab.{src}')
        tgt_vocab_filename = os.path.join(data_dir, f'vocab.{tgt}')

        joint_vocab_exists = os.path.isfile(joint_vocab_file)
        individual_vocabs_exist = (os.path.isfile(src_vocab_filename)
                                   and os.path.isfile(tgt_vocab_filename))
        vocabs_exist = joint_vocab_exists or individual_vocabs_exist

        assert vocabs_exist, f"No vocabs found in {data_dir}"

        if joint_vocab_exists:
            with open(joint_vocab_file) as f:
                vocab = Vocabulary.from_json(f.read())
                src_vocab = tgt_vocab = vocab
        else:
            with open(src_vocab_filename) as src_vocab_file:
                src_vocab = Vocabulary.from_json(src_vocab_file.read())

            with open(tgt_vocab_filename) as tgt_vocab_file:
                tgt_vocab = Vocabulary.from_json(tgt_vocab_file.read())

        return SeqpTranslationTask(args, src_vocab, tgt_vocab)
