# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from typing import Iterable, Optional, Tuple, Dict
import numpy as np


class RecordReader:
    """
    Abstract class with the contract of a record reader, which offers the
    functionality of an indexed dataset of sequences.
    """

    def indexes(self) -> Iterable[int]:
        """
        Returns the indexes of the sequences in the dataset.
        :return: The indexes of the sequences in the dataset.
        """
        raise NotImplementedError

    def length(self, index) -> int:
        """
        Returns the length of the sequence at the given index.
        :param index: Index identifying the sequence to query.
        :return: Length of the sequence at the given index.
        """
        raise NotImplementedError

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        """
        Returns an Iterable of tuples of index and length of the
        associated sequence.
        :return: Indexes and lengths of the sequences in the dataset.
        """
        raise NotImplementedError

    def retrieve(self, index) -> np.ndarray:
        """
        Retrieves the sequence at the given index.
        :param index: Index of the sequence to retrieve.
        :return: Sequence at the given index.
        """
        raise NotImplementedError

    def num_records(self) -> int:
        """
        Gives the number of sequences in the dataset.
        :return: The number of sequences in the dataset.
        """
        raise NotImplementedError

    def metadata(self, key: str) -> Optional[str]:
        """
        Gets the piece of metadata associated to the given key.
        :param key: Key of the piece of metadata.
        :return: metadata associated to the key, or None if none.
        """
        return None


class RecordWriter(object):
    """
    Abstract class with the contract of a record writer, which
    basically writes a bunch of indexed sequences to a file,
    together with a metadata dictionary.
    """

    def write(self,
              encoded_records: Iterable[Tuple[int, np.ndarray]],
              output_file: str,
              max_records: Optional[int]=None,
              metadata: Optional[Dict[str, str]]=None) -> bool:
        """
        Writes the given records to output file.
        :param encoded_records: generator of tuples (index, numpy array) to
                                be written to file.
        :param output_file: file where to write the records.
        :param max_records: maximum number of records to write to output_file.
        :param metadata: metadata stored along with the sequences.
        :return: True if max_records was reached and therefore there can
                 be pending sequences in encoded_records; False otherwise.
        """
        raise NotImplementedError


def write_shards(writer: RecordWriter,
                 output_file_template: str,
                 encoded_records: Iterable[Tuple[int, np.ndarray]],
                 max_records_per_shard: int,
                 metadata: Optional[Dict[str, str]]=None):
    """
    Writes the given encoded record in several files.
    :param writer: RecordWriter to use.
    :param output_file_template: template for the output file names. It
                                 is used like `output_file_template.format(5)`
                                 to obtain each file name.
    :param encoded_records: generator of records.
    :param max_records_per_shard: maximum records per shard.
    :param metadata: dictionary with metadata that will be included in
                     each file.
    :return: None
    """
    assert output_file_template.format(1) != output_file_template
    remaining_records = True
    output_file_index = 1
    while remaining_records:
        output_file = output_file_template.format(output_file_index)
        remaining_records = writer.write(encoded_records,
                                         output_file,
                                         max_records_per_shard,
                                         metadata=metadata)
        output_file_index += 1
