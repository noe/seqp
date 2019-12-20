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
_INITIAL_INDEX_KEY = 'initial_index'
_FINAL_INDEX_KEY = 'final_index'
_FIELDS_KEY = 'fields'
_SEQUENCE_FIELD_KEY = 'sequence_field'


def _compose_key(idx: int, field: Optional[str]):
    suffix = ":{}".format(field) if field else ""
    return str(idx) + suffix


class Hdf5RecordWriter(RecordWriter):
    """
    Implementation of RecordWriter that persists the records in an HDF5 file.
    """

    def __init__(self,
                 output_file: str,
                 fields: Iterable[str] = None,
                 sequence_field: str = None,
                 append: bool = False,
                 initial_index: int = 1):
        super().__init__(fields=fields, sequence_field=sequence_field, append=append)

        if fields is not None or sequence_field is not None:
            assert fields is not None and sequence_field is not None
            assert sequence_field in fields
        self.initial_index = initial_index
        self.next_index = initial_index
        self.output_file = output_file
        self.hdf5_file = h5py.File(output_file, 'a' if append else 'w')
        self.lengths = ([] if not append
                        else json.loads(self.hdf5_file[_LENGTHS_KEY][0]))
        if append and fields is not None:
            assert set(fields) == set(json.loads(self.hdf5_file[_FIELDS_KEY][0]))
            assert sequence_field == self.hdf5_file[_SEQUENCE_FIELD_KEY][0]

    def close(self):
        meta_dtype = h5py.special_dtype(vlen=str)

        def add_metadata(k, v):
            if self.append and k in self.hdf5_file:  # remove if already exists
                del self.hdf5_file[k]

            # add a dataset with the lengths of all sentences
            meta_dataset = self.hdf5_file.create_dataset(k,
                                                         (1,),
                                                         dtype=meta_dtype,
                                                         track_times=False)
            meta_dataset[0] = v

        if self.metadata:
            for key, value in self.metadata.items():
                add_metadata(key, value)

        # add extra piece of metadata with sequence lengths...
        add_metadata(_LENGTHS_KEY, json.dumps(self.lengths))

        # ...and initial/final indexes
        add_metadata(_INITIAL_INDEX_KEY, str(self.initial_index))
        add_metadata(_FINAL_INDEX_KEY, str(self.next_index - 1))

        # add extra piece of metadata with keys if they were provided
        if self.fields:
            add_metadata(_FIELDS_KEY, json.dumps(self.fields))
            add_metadata(_SEQUENCE_FIELD_KEY, self.sequence_field)

        self.hdf5_file.close()

        if len(self.lengths) == 0:
            os.remove(self.output_file)

    def write(self,
              record: Union[np.ndarray, Dict[str, np.ndarray]],
              ) -> int:
        idx = self.next_index
        self.next_index += 1
        if isinstance(record, dict):
            assert self.fields is not None, "Writers without fields need numpy arrays as record"
            field_records = record
            assert all(field in self.fields for field in field_records.keys())
            record_lengths = {field: self._write(idx, record, field)
                              for field, record in field_records.items()}
            length = record_lengths[self.sequence_field]
        else:
            assert self.fields is None, "Writers with fields need dictionaries as record"
            length = self._write(idx, record)

        self.lengths.append(length)
        return idx

    def _write(self,
               idx: int,
               record: np.ndarray,
               field: str=None,
               ) -> int:
        internal_key = _compose_key(idx, field)

        self.hdf5_file.create_dataset(internal_key,
                                      record.shape,
                                      dtype=record.dtype,
                                      data=record,
                                      track_times=False)
        length = record.shape[0]
        return length


def _to_numpy(hdf5_dataset):
    return hdf5_dataset[:]


class Hdf5RecordReader(RecordReader):
    """
    Implementation of RecordReader that reads records from HDF5 files.
    """

    def __init__(self,
                 file_names: Union[str, List[str]]):
        """
        Constructs an Hdf5RecordReader.
        :param file_names: HDF5 files to read.
        """
        if isinstance(file_names, str):
            file_names = [file_names]
        self.file_names = file_names
        self.hdf5_stores = {file_name: h5py.File(file_name, 'r')
                            for file_name in file_names}
        self.total_count = 0
        self.lengths = dict()
        self.initial_index = dict()
        self.final_index = dict()

        fields = None
        sequence_field = None

        for file_name, store in self.hdf5_stores.items():
            fields = json.loads(store[_FIELDS_KEY][0]) if _FIELDS_KEY in store else None
            sequence_field = (store[_SEQUENCE_FIELD_KEY][0]
                              if _SEQUENCE_FIELD_KEY in store else None)

            self.lengths[file_name] = json.loads(store[_LENGTHS_KEY][0])
            self.total_count += len(self.lengths[file_name])
            self.initial_index[file_name] = int(store[_INITIAL_INDEX_KEY][0])
            self.final_index[file_name] = int(store[_FINAL_INDEX_KEY][0])
        super().__init__(fields, sequence_field)

    def close(self):
        """
        Closes all HDF5 files.
        :return: None
        """
        for hdf5_store in self.hdf5_stores.values():
            hdf5_store.close()
        self.hdf5_stores.clear()

    def _find_file(self, index: int) -> Optional[str]:
        for file_name in self.file_names:
            if self.initial_index[file_name] <= index <= self.final_index[file_name]:
                return file_name
        return None

    def indexes(self) -> Iterable[int]:
        """ See super class docstring. """
        for file_name in self.file_names:
            initial = self.initial_index[file_name]
            final = self.final_index[file_name]
            yield from range(initial, final + 1)

    def length(self, index: int) -> int:
        """ See super class docstring. """
        file_name = self._find_file(index)
        if not file_name:
            return 0
        initial = self.initial_index[file_name]
        return self.lengths[file_name][index - initial]

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        """ See super class docstring. """
        for file_name in self.file_names:
            initial = self.initial_index[file_name]
            final = self.final_index[file_name]
            lengths = self.lengths[file_name]
            for index in range(initial, final + 1):
                length = lengths[index - initial]
                yield index, length

    def retrieve(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ See super class docstring. """
        if self.fields:
            return {field: self._retrieve(idx, field) for field in self.fields}
        else:
            return self._retrieve(idx, None)

    def _retrieve(self, idx: int, field: Optional[str]) -> Optional[np.ndarray]:
        file_name = self._find_file(idx)
        if not file_name:
            return None
        internal_key = _compose_key(idx, field)
        return (_to_numpy(self.hdf5_stores[file_name][internal_key])
                if file_name else None)

    def num_records(self) -> int:
        """ See super class docstring. """
        return self.total_count

    def metadata(self, metadata_key) -> Optional[str]:
        """ See super class docstring. """
        last_file = self.file_names[-1]
        last_hdf5_store: h5py.File = self.hdf5_stores[last_file]
        return str(last_hdf5_store[metadata_key][0])
