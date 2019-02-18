# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from typing import Iterable, Optional, Tuple, Dict, Union
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

    def retrieve(self, index) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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

    def metadata(self, metadata_key: str) -> Optional[str]:
        """
        Gets the piece of metadata associated to the given key.
        :param metadata_key: Key of the piece of metadata.
        :return: metadata associated to the key, or None if none.
        """
        return None


class RecordWriter(object):
    """
    Abstract class with the contract of a record writer, which
    basically writes indexed sequences, probably to a file,
    together with a metadata dictionary. This class is a
    context manager so that it can properly close and release
    any resources at scope exit.

    Subclasses should allocate any needed resource in the
    constructor and release them in method `close`, and implement
    method `write`.
    """

    def __init__(self, fields: Optional[Iterable[str]]=None):
        """
        Constructor.
        :param fields: Optional fields for the records to write
        """
        self.metadata = dict()
        self.fields = fields

    def __enter__(self) -> "RecordWriter":
        """
        Invoked when entering the context scope.
        :return: returns itself so that close is invoked at the scope exit.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Invoked at with block scope exit. Calls self.close()
        in order to release any resource in use.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return: None
        """
        self.close()

    def close(self):
        """
        Closes any resource in use (e.g. open files).
        :return: None
        """
        pass

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        """
        Attaches a metadata dictionary to the records.
        :param metadata: Metadata to be attached to the records.
        :return: None
        """
        self.metadata.update(metadata)

    def write(self, idx: int,
              record: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
              ) -> None:
        """
        Writes a record to the underlying storage.
        :param idx: Index of the record. Must be unique.
        :param record: Record to be stored, or dictionary with
               records (dictionary keys must match fields provided in
               the constructor).
        :return: None
        """
        raise NotImplementedError


class ShardedWriter(RecordWriter):
    """
    Wrapper RecordWriter to have automatic sharding. Assumes the
    underlying writer is file-based and that it receives an
    argument specifying the file name in its constructor.
    """

    def __init__(self,
                 writer_class,
                 output_file_template: str,
                 max_records_per_shard: int,
                 output_file_param=0,
                 *args,
                 **kwargs):
        """
        Constructor.
        :param writer_class: Class of the actual writer to use.
        :param output_file_template: template of the file names. Internally
               it will be used with `format` to get the actual file name,
               passing a shard index as argument.
        :param max_records_per_shard: Maximum number of records in a
               single shard file.
        :param output_file_param: name or position of the parameter of writer_class
               constructor that specifies the file name.
        :param args: other positional arguments needed by writer_class constructor.
        :param kwargs: other keyword arguments needed by writer_class constructor.
        """
        super().__init__()
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.output_file_template = output_file_template
        self.max_records_per_shard = max_records_per_shard
        self.current_output_file_idx = 1
        self.writer_class = writer_class
        self.output_file_param = output_file_param
        if isinstance(self.output_file_param, int):
            self.args.insert(self.output_file_param,
                             output_file_template.format(1))
        else:
            self.kwargs[output_file_param] = output_file_template.format(1)
        self.current_writer = writer_class(*self.args, **self.kwargs)
        self.current_records = 0

    def close(self):
        self.current_writer.close()

    def _next_writer(self):
        self.current_writer.close()
        self.current_records = 0
        self.current_output_file_idx += 1
        shard_name = self.output_file_template.format(self.current_output_file_idx)
        if isinstance(self.output_file_param, int):
            self.args[self.output_file_param] = shard_name
        else:
            self.kwargs[self.output_file_param] = shard_name
        self.current_writer = self.writer_class(*self.args, **self.kwargs)

    def write(self, idx: int,
              record: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
              ) -> None:
        if self.current_records >= self.max_records_per_shard:
            self._next_writer()
        self.current_writer.write(idx, record)
