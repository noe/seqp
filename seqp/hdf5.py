# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import json
import h5py
import numpy as np
import os
from typing import List, Iterable, Tuple, Optional, Dict, Union

from .record import RecordReader, RecordWriter

_LENGTHS_KEY = 'lengths'
_FIELDS_KEY = 'fields'
_SEQUENCE_FIELD_KEY = 'sequence_field'
_MAX_LENGTH = float('inf')


def _compose_key(idx: int, field: Optional[str]):
    suffix = ":{}".format(field) if field else ""
    return str(idx) + suffix


class Hdf5RecordWriter(RecordWriter):
    """
    Implementation of RecordWriter that persists the records in an HDF5 file.
    """

    def __init__(self,
                 output_file: str,
                 fields: Iterable[str]=None,
                 sequence_field: str=None):
        super().__init__(fields=fields, sequence_field=sequence_field)
        self.output_file = output_file
        self.hdf5_file = h5py.File(output_file, 'w')
        self.index_to_length = dict()

    def close(self):
        meta_dtype = h5py.special_dtype(vlen=str)

        def add_metadata(key, value):
            # add a dataset with the lengths of all sentences
            meta_dataset = self.hdf5_file.create_dataset(key, (1,), dtype=meta_dtype)
            meta_dataset[0] = value

        if self.metadata:
            for key, value in self.metadata.items():
                add_metadata(key, value)

        # add extra piece of metadata with sequence lengths
        add_metadata(_LENGTHS_KEY, json.dumps(self.index_to_length))

        # add extra piece of metadata with keys if they were provided
        if self.fields:
            add_metadata(_FIELDS_KEY, json.dumps(self.fields))
            add_metadata(_SEQUENCE_FIELD_KEY, self.sequence_field)

        self.hdf5_file.close()

        if len(self.index_to_length) == 0:
            os.remove(self.output_file)

    def write(self, idx: int,
              record: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
              ) -> None:
        if isinstance(record, dict):
            field_records = record
            assert all(field in self.fields for field in field_records.keys())
            lengths = {field: self._write(idx, record, field)
                       for field, record in field_records.items()}
            length = lengths[self.sequence_field]
        else:
            length = self._write(idx, record)

        self.index_to_length[str(idx)] = length

    def _write(self, idx: int,
               record: Optional[np.ndarray],
               field: str=None,
               ) -> int:
        internal_key = _compose_key(idx, field)
        assert internal_key not in self.index_to_length
        if record is None:
            # sentence did not match the needed criteria to be encoded (e.g. too long), so
            # we add an empty dataset
            # (see http://docs.h5py.org/en/stable/high/dataset.html#creating-and-reading-empty-or-null-datasets-and-attributes)
            self.hdf5_file.create_dataset(internal_key, data=h5py.Empty("f"))
            length = 0
        else:
            self.hdf5_file.create_dataset(internal_key, record.shape, dtype=record.dtype, data=record)
            length = record.shape[0]

        return length


def _to_numpy(hdf5_dataset):
    return hdf5_dataset[:]


class Hdf5RecordReader(RecordReader):
    """
    Implementation of RecordReader that reads records from HDF5 files.
    """

    def __init__(self,
                 file_names: List[str],
                 min_length: int = 0,
                 max_length: int = _MAX_LENGTH):
        """
        Constructs an Hdf5RecordReader.
        :param file_names: HDF5 files to read.
        :param min_length: Minimum sequence length threshold.
        :param max_length: Maximum sequence length threshold.
        """
        if isinstance(file_names, str):
            file_names = [file_names]
        self.file_names = file_names
        self.hdf5_stores = {file_name: h5py.File(file_name, 'r')
                            for file_name in file_names}
        self.index_to_filename = dict()
        self.total_count = 0
        self.index_to_length = {}

        fields = None
        sequence_field = None
        for file_name, store in self.hdf5_stores.items():
            file_index_to_length = json.loads(store[_LENGTHS_KEY][0])
            fields = json.load((store[_FIELDS_KEY][0])) if _FIELDS_KEY in store else None
            sequence_field = (json.load((store[_SEQUENCE_FIELD_KEY][0]))
                              if _SEQUENCE_FIELD_KEY in store else None)
            file_index_to_length = {int(index): length
                                    for index, length in file_index_to_length.items()
                                    if min_length <= length <= max_length}

            self.index_to_length.update(file_index_to_length)
            self.total_count += len(file_index_to_length)
            for index in file_index_to_length.keys():
                self.index_to_filename[index] = file_name
        super().__init__(fields, sequence_field)

    def close(self):
        """
        Closes all HDF5 files.
        :return: None
        """
        for hdf5_store in self.hdf5_stores.values():
            hdf5_store.close()
        self.hdf5_stores.clear()

    def indexes(self) -> Iterable[int]:
        """ See super class docstring. """
        yield from self.index_to_filename.keys()

    def length(self, index: int) -> int:
        """ See super class docstring. """
        return self.index_to_length.get(index, None)

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        """ See super class docstring. """
        yield from self.index_to_length.items()

    def retrieve(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ See super class docstring. """
        if self.fields:
            return {field: self._retrieve(idx, field) for field in self.fields}
        else:
            return self._retrieve(idx)

    def _retrieve(self, idx: int, field: str) -> Optional[np.ndarray]:
        file_name = self.index_to_filename.get(idx, None)
        internal_key = _compose_key(idx, field)
        return (_to_numpy(self.hdf5_stores[file_name][internal_key])
                if file_name else None)

    def num_records(self) -> int:
        """ See super class docstring. """
        return self.total_count

    def metadata(self, metadata_key) -> Optional[str]:
        """ See super class docstring. """
        random_hdf5_store: h5py.File = next(iter(self.hdf5_stores.values()))
        return str(random_hdf5_store[metadata_key][0])
