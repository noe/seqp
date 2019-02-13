#!/usr/bin/env python3
#
# Script to encode text as BERT embedded vectors in HDF5 format.
#
# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#


import argparse
import numpy as np
import sys
import torch
import tqdm
from typing import Optional, Union, List, Iterable, Callable

from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from seqp.record import RecordWriter, ShardedWriter
from seqp.hdf5 import Hdf5RecordWriter
from seqp.encoding import TextCodec
from seqp.util import count_lines


DEFAULT_BERT_WEIGHTS = 'bert-base-multilingual-cased'


class BertInterface(TextCodec):
    def __init__(self, use_gpu=False):
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_BERT_WEIGHTS)
        self.model = BertForMaskedLM.from_pretrained(DEFAULT_BERT_WEIGHTS)
        self.model.eval()
        use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model.to(self.device)

    def decode(self, embedded: Union[np.ndarray, torch.Tensor]) -> List[str]:
        if isinstance(embedded, np.ndarray):
            if len(embedded.shape) == 2:  # seq_length x emb_dim
                embedded = np.expand_dims(embedded, 0)  # add batch dimension
            assert len(embedded.shape) == 3
            embedded = torch.from_numpy(embedded).to(self.device)
        predictions = self.model.cls(embedded)
        predicted_indexes = torch.argmax(predictions, dim=2).cpu().numpy()
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexes[0].tolist())
        return predicted_tokens

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens).replace(" ##", "")

    def tokenize(self, sentence: str) -> List[str]:
        return self.tokenizer.tokenize(sentence)

    def encode(self, tokens: List[str]) -> Optional[np.ndarray]:
        tokenized_text = ['[CLS]'] + tokens
        if len(tokenized_text) > self.tokenizer.max_len:
            return None

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.Tensor([indexed_tokens]).to(self.device)
        sequence_output, _ = self.model.bert(tokens_tensor, output_all_encoded_layers=False)
        return sequence_output.detach()[0].cpu().numpy()


def show_sentence(key, sentence):
    print("{}: {}".format(key, sentence))


def write_records(sentences: Iterable[str],
                  codec: TextCodec,
                  writer: RecordWriter,
                  progress: Callable[[], None] = None):
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip("\r\n ")
        tokens = codec.tokenize(sentence)
        encoded = codec.encode(tokens)
        writer.write(idx, encoded)
        if progress is not None:
            progress()


def main():
    parser = argparse.ArgumentParser("BERT encoder into HDF5")
    parser.add_argument('--input', required=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_records', default=200000)
    args = parser.parse_args()
    extension = '.hdf5'
    if args.output[-len(extension):] == extension:
        args.output = args.output[:-len(extension)] # remove .hdf5 extension
    output_file_template = args.output + "_{:05d}.hdf5"
    embedder = BertInterface(use_gpu=True)

    writer = ShardedWriter(Hdf5RecordWriter, output_file_template, args.max_records)

    if args.input:
        total_sentences = count_lines(args.input)
        with open(args.input, 'r') as input_sentences:
            with tqdm.tqdm(total=total_sentences, ncols=100, leave=False, unit='segments') as pbar:
                def progress():
                    pbar.update(1)

                write_records(input_sentences, embedder, writer, progress)
    else:
        write_records(sys.stdin, embedder, writer)


if __name__ == '__main__':
    main()
