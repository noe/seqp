# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import json
import h5py
import numpy as np
import os
from typing import List, Iterable, Tuple, Optional, Dict

from .record import RecordReader, RecordWriter

_LENGTHS_KEY = 'lengths'
_MAX_LENGTH = float('inf')


class Hdf5RecordWriter(RecordWriter):
    """
    Implementation of RecordWriter that persists the records in an HDF5 file.
    """

    def __init__(self, output_file: str, **kwargs):
        super().__init__(output_file, **kwargs)
        self.hdf5_file = h5py.File(output_file, 'w')
        self.index_to_length = dict()

    def close(self):
        self.hdf5_file.close()

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

        if len(self.index_to_length) == 0:
            os.remove(self.output_file)

    def write(self, idx: int, record: Optional[np.ndarray]):
        key = str(idx)
        assert key not in self.index_to_length
        if record is None:
            # sentence did not match the needed criteria to be encoded (e.g. too long), so
            # we add an empty dataset
            # (see http://docs.h5py.org/en/stable/high/dataset.html#creating-and-reading-empty-or-null-datasets-and-attributes)
            self.hdf5_file.create_dataset(key, data=h5py.Empty("f"))
            length = 0
        else:
            self.hdf5_file.create_dataset(key, record.shape, dtype=record.dtype, data=record)
            length = record.shape[0]

        self.index_to_length[key] = length


def _to_numpy(hdf5_dataset):
    return hdf5_dataset[:]


class Hdf5RecordReader(RecordReader):
    """
    Implementation of RecordReader that reads records from HDF5 files.
    """

    def __init__(self,
                 file_names: List[str],
                 min_length: int=0,
                 max_length: int=_MAX_LENGTH):
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

        for file_name, store in self.hdf5_stores.items():
            file_index_to_length = json.loads(store[_LENGTHS_KEY][0])
            file_index_to_length = {int(index): length
                                    for index, length in file_index_to_length.items()
                                    if min_length <= length <= max_length}

            self.index_to_length.update(file_index_to_length)
            self.total_count += len(file_index_to_length)
            for index in file_index_to_length.keys():
                self.index_to_filename[index] = file_name

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

    def retrieve(self, index: int) -> np.ndarray:
        """ See super class docstring. """
        file_name = self.index_to_filename.get(index, None)
        return (_to_numpy(self.hdf5_stores[file_name][str(index)])
                if file_name else None)

    def num_records(self) -> int:
        """ See super class docstring. """
        return self.total_count

    def metadata(self, key) -> Optional[str]:
        """ See super class docstring. """
        random_hdf5_store: h5py.File = next(iter(self.hdf5_stores.values()))
        return str(random_hdf5_store[key][0])
