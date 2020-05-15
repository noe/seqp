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
                 initial_index: int = 0):
        super().__init__(fields=fields, sequence_field=sequence_field, append=append)

        if fields is not None or sequence_field is not None:
            assert fields is not None and sequence_field is not None
            assert sequence_field in fields

        self.output_file = output_file
        self.hdf5_file = h5py.File(output_file, 'a' if append else 'w')
        self.initial_index = (initial_index if not append
                              else int(self.hdf5_file[_INITIAL_INDEX_KEY][0]))
        self.next_index = (initial_index if not append
                           else 1 + int(self.hdf5_file[_FINAL_INDEX_KEY][0]))
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
               field: str = None,
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


class _ContiguousIndexing:
    """
    Indexing scheme that needs indexes to be contiguous. This
    assumption enables optimizations that make it faster.
    """

    def __init__(self,
                 initial_indexes: Dict[str, int],
                 final_indexes: Dict[str, int],
                 hdf5_stores: Dict[str, h5py.File],
                 lengths: Dict[str, List[int]]):

        # sort file_names by the initial index
        self.file_names = [f for f, i in sorted(initial_indexes.items(),
                                                key=lambda file_index: file_index[1])]

        # turn input params in lists sorted by initial index
        self.hdf5_stores = [hdf5_stores[f] for f in self.file_names]
        self.lengths = [lengths[f] for f in self.file_names]
        self.initial_indexes = [initial_indexes[f] for f in self.file_names]
        self.final_indexes = [final_indexes[f] for f in self.file_names]

        # ensure all buckets but the last one are the same size
        bucket_sizes = [f - i + 1 for i, f in zip(self.initial_indexes, self.final_indexes)]
        are_same_size = all(b == bucket_sizes[0] for b in bucket_sizes[:-1])
        if not are_same_size:
            raise ValueError("Buckets are of different sizes")

        self.initial_index = self.initial_indexes[0]
        self.final_index = self.final_indexes[-1]
        self.bucket_size = bucket_sizes[0]

        are_contiguous = all(f + 1 == i for f, i in zip(self.final_indexes[:-1],
                                                        self.initial_indexes[1:]))
        if not are_contiguous:
            raise ValueError("Buckets are not contiguous")

        store = self.hdf5_stores[0]
        self.fields = json.loads(store[_FIELDS_KEY][0]) if _FIELDS_KEY in store else None
        self.sequence_field = (store[_SEQUENCE_FIELD_KEY][0]
                               if _SEQUENCE_FIELD_KEY in store else None)

    def _bucket(self, index: int, return_remainder: bool = False) -> Union[int, Tuple[int, int]]:
        bucket, remainder = divmod(index - self.initial_index, self.bucket_size)
        return bucket if not return_remainder else (bucket, remainder)

    def indexes(self) -> Iterable[int]:
        yield from range(self.initial_index, self.final_index + 1)

    def length(self, index: int) -> int:
        bucket, remainder = self._bucket(index, return_remainder=True)
        return self.lengths[bucket][remainder]

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        for bucket in range(len(self.lengths)):
            indexes = range(self.initial_indexes[bucket], self.final_indexes[bucket] + 1)
            yield from zip(indexes, self.lengths[bucket])

    def retrieve(self, idx: int, field: Optional[str]) -> Optional[np.ndarray]:
        internal_key = _compose_key(idx, field)
        bucket = self._bucket(idx)
        return _to_numpy(self.hdf5_stores[bucket][internal_key])

    def num_records(self) -> int:
        return self.final_index - self.initial_index + 1

    def metadata(self, metadata_key) -> Optional[str]:
        last_hdf5_store: h5py.File = self.hdf5_stores[-1]
        return str(last_hdf5_store[metadata_key][0])

    def close(self):
        for hdf5_store in self.hdf5_stores:
            hdf5_store.close()
        self.hdf5_stores.clear()


class _NonContiguousIndexing:
    """
    Indexing scheme that does not need indexes to be contiguous.
    It is slower than _ContiguousIndexing.
    """

    def __init__(self,
                 initial_indexes: Dict[str, int],
                 final_indexes: Dict[str, int],
                 hdf5_stores: Dict[str, h5py.File],
                 lengths: Dict[str, List[int]]):
        self.file_names = list(initial_indexes.keys())
        self.initial_indexes = initial_indexes
        self.final_indexes = final_indexes
        self.hdf5_stores = hdf5_stores
        self.lengths = lengths
        self.total_count = sum(len(file_lengths) for file_lengths in lengths.values())
        store = next(iter(self.hdf5_stores.values()))
        self.fields = json.loads(store[_FIELDS_KEY][0]) if _FIELDS_KEY in store else None
        self.sequence_field = (store[_SEQUENCE_FIELD_KEY][0]
                               if _SEQUENCE_FIELD_KEY in store else None)

    def close(self):
        for hdf5_store in self.hdf5_stores.values():
            hdf5_store.close()
        self.hdf5_stores.clear()

    def _find_file(self, index: int) -> Optional[str]:
        for file_name in self.file_names:
            if self.initial_indexes[file_name] <= index <= self.final_indexes[file_name]:
                return file_name
        return None

    def indexes(self) -> Iterable[int]:
        for file_name in self.file_names:
            initial = self.initial_indexes[file_name]
            final = self.final_indexes[file_name]
            yield from range(initial, final + 1)

    def length(self, index: int) -> int:
        file_name = self._find_file(index)
        if not file_name:
            return 0
        initial = self.initial_indexes[file_name]
        return self.lengths[file_name][index - initial]

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        for file_name in self.file_names:
            initial = self.initial_indexes[file_name]
            final = self.final_indexes[file_name]
            lengths = self.lengths[file_name]
            for index in range(initial, final + 1):
                length = lengths[index - initial]
                yield index, length

    def retrieve(self, idx: int, field: Optional[str]) -> Optional[np.ndarray]:
        file_name = self._find_file(idx)
        if not file_name:
            return None
        internal_key = _compose_key(idx, field)
        return (_to_numpy(self.hdf5_stores[file_name][internal_key])
                if file_name else None)

    def num_records(self) -> int:
        return self.total_count

    def metadata(self, metadata_key) -> Optional[str]:
        last_file = self.file_names[-1]
        last_hdf5_store: h5py.File = self.hdf5_stores[last_file]
        return str(last_hdf5_store[metadata_key][0])


def _build_indexing(initial_indexes, final_indexes, hdf5_stores, lengths):
    try:
        return _ContiguousIndexing(initial_indexes, final_indexes, hdf5_stores, lengths)
    except ValueError:
        return _NonContiguousIndexing(initial_indexes, final_indexes, hdf5_stores, lengths)


class Hdf5RecordReader(RecordReader):
    """
    Implementation of RecordReader that reads records from HDF5 files.
    """

    def __init__(self, file_names: Union[str, List[str]]):
        """
        Constructs an Hdf5RecordReader.
        :param file_names: HDF5 files to read.
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        hdf5_stores = {file_name: h5py.File(file_name, 'r', libver='latest', swmr=True)
                       for file_name in file_names}

        lengths = dict()
        initial_index = dict()
        final_index = dict()

        fields = None
        sequence_field = None

        for file_name, store in hdf5_stores.items():
            fields = json.loads(store[_FIELDS_KEY][0]) if _FIELDS_KEY in store else None
            sequence_field = (store[_SEQUENCE_FIELD_KEY][0]
                              if _SEQUENCE_FIELD_KEY in store else None)

            lengths[file_name] = json.loads(store[_LENGTHS_KEY][0])
            initial_index[file_name] = int(store[_INITIAL_INDEX_KEY][0])
            final_index[file_name] = int(store[_FINAL_INDEX_KEY][0])

        self.indexing = _build_indexing(initial_index, final_index, hdf5_stores, lengths)
        super().__init__(fields, sequence_field)

    def retrieve(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ See super class docstring. """
        if self.fields:
            return {field: self.indexing.retrieve(idx, field) for field in self.fields}
        else:
            return self.indexing.retrieve(idx, None)

    def close(self):
        """Closes all HDF5 files."""
        self.indexing.close()

    def indexes(self) -> Iterable[int]:
        """ See super class docstring. """
        return self.indexing.indexes()

    def length(self, index: int) -> int:
        """ See super class docstring. """
        return self.indexing.length(index)

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        """ See super class docstring. """
        return self.indexing.indexes_and_lengths()

    def num_records(self) -> int:
        """ See super class docstring. """
        return self.indexing.num_records()

    def metadata(self, metadata_key) -> Optional[str]:
        """ See super class docstring. """
        return self.indexing.metadata(metadata_key)
