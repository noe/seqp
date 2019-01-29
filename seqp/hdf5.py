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
    def write(self,
              encoded_records: Iterable[Tuple[int, np.ndarray]],
              output_file: str,
              max_records: Optional[int]=None,
              metadata: Optional[Dict[str, str]] = None):
        """ See super class docstring. """
        remaining_records = True

        while remaining_records:
            num_records = 0
            index_to_length = {}
            remaining_records = False

            with h5py.File(output_file, 'w') as f:
                for idx, encoded in encoded_records:
                    key = str(idx)
                    if encoded is None:
                        # sentence did not match the needed criteria to be encoded (e.g. too long), so
                        # we add an empty dataset
                        # (see http://docs.h5py.org/en/stable/high/dataset.html#creating-and-reading-empty-or-null-datasets-and-attributes)
                        f.create_dataset(key, data=h5py.Empty("f"))
                        length = 0
                    else:
                        f.create_dataset(key, encoded.shape, dtype=encoded.dtype, data=encoded)
                        length = encoded.shape[0]

                    index_to_length[key] = length

                    num_records += 1

                    if max_records is not None and num_records >= max_records:
                        remaining_records = True
                        break

                def add_metadata(key, value):
                    # add a dataset with the lengths of all sentences
                    meta_dataset = f.create_dataset(key, (1,), dtype=h5py.special_dtype(vlen=str))
                    meta_dataset[0] = value

                if metadata:
                    for key, value in metadata.items():
                        add_metadata(key, value)

                # add extra piece of metadata with sequence lengths
                add_metadata(_LENGTHS_KEY, json.dumps(index_to_length))

            if num_records == 0:
                os.remove(output_file)

            return remaining_records


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

    def __del__(self):
        """
        Destructor of the Hdf5RecordReader. Closes all HDF5 files.
        :return: None
        """
        self.close()

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
