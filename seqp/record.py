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

    def __init__(self,
                 fields: Iterable[str] = None,
                 sequence_field: str = None):
        """
        Constructor.
        :param fields: fields in the records of this reader, None if none.
        :param sequence_field: field containing the main sequence.
        """
        self.fields = fields
        self.sequence_field = sequence_field

    def __enter__(self) -> "RecordReader":
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


class RecordWriter:
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

    def __init__(self,
                 fields: Optional[Iterable[str]]=None,
                 sequence_field: str = None,
                 append: bool = False,
                 initial_index: int = 0):
        """
        Constructor.
        :param fields: Optional fields for the records to write
        :param sequence_field: field containing the sequence itself (it
               will be used to compute the sequence length).
        :param append: whether to append or overwrite.
        :param initial_index: initial record index for the writer.
        """
        self.metadata = dict()
        self.fields = fields
        self.append = append
        self.next_index = initial_index
        self.sequence_field = sequence_field
        if fields or sequence_field:
            assert fields and sequence_field
            assert sequence_field in fields

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

    def write(self,
              record: Union[np.ndarray, Dict[str, np.ndarray]],
              ) -> int:
        """
        Writes a record to the underlying storage.
        :param idx: Index of the record. Must be unique.
        :param record: Record to be stored, or dictionary with
               records (dictionary keys must match fields provided in
               the constructor).
        :return: index of the just written record.
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
        :param args: other positional arguments needed by writer_class constructor.
        :param kwargs: other keyword arguments needed by writer_class
        constructor. Here you can specify key 'output_file_param', which can be
        either an integer (position in `args`) or a string (key in `kwargs`).
        """
        super().__init__()
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.output_file_template = output_file_template
        self.max_records_per_shard = max_records_per_shard
        initial_file_idx = 1
        self.current_output_file_idx = initial_file_idx
        self.writer_class = writer_class
        assert 'append' not in kwargs, "Flag 'append' not allowed in sharded writers"
        self.output_file_param = kwargs.pop('output_file_param', 0)
        self.current_records = 0
        self.former_files = []
        first_output_file = self._file_name(initial_file_idx)
        initial_index = 0
        self.current_writer = self._writer_for(first_output_file, initial_index)
        super().__init__(*args, **kwargs)

    def close(self):
        self.current_writer.close()

    def _file_name(self, file_idx) -> str:
        return self.output_file_template.format(file_idx)

    def _writer_for(self, shard_name, initial_index, **extra_kwags) -> RecordWriter:
        args = list(self.args)
        kwargs = dict(self.kwargs)
        kwargs.update(extra_kwags)
        kwargs['initial_index'] = initial_index

        if isinstance(self.output_file_param, int):
            args.insert(self.output_file_param, shard_name)
        else:
            kwargs[self.output_file_param] = shard_name

        return self.writer_class(*args, **kwargs)

    def _next_writer(self, initial_index) -> None:
        self.former_files.append(self._file_name(self.current_output_file_idx))
        self.current_writer.close()
        self.current_records = 0
        self.current_output_file_idx += 1
        shard_name = self._file_name(self.current_output_file_idx)
        self.current_writer = self._writer_for(shard_name, initial_index)
        self.current_writer.add_metadata(self.metadata)

    def write(self,
              record: Union[np.ndarray, Dict[str, np.ndarray]],
              ) -> int:
        if self.current_records >= self.max_records_per_shard:
            initial_index = 1 if self.current_writer is None else self.current_writer.next_index
            self._next_writer(initial_index)
        idx = self.current_writer.write(record)
        self.current_records += 1
        return idx

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        self.current_writer.add_metadata(metadata)
        self.metadata.update(metadata)
        self._update_metadata_in_previous_files()

    def _update_metadata_in_previous_files(self) -> None:
        for former_file in self.former_files:
            with self._writer_for(former_file, None, append=True) as writer:
                writer.add_metadata(self.metadata)
