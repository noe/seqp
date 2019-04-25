# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import numpy as np
from seqp.record import RecordReader
from typing import Dict, Iterable, List, Optional, Tuple, Union


class InMemoryReader(RecordReader):
    """
    Implementation of RecordReader based on a List of records.
    """

    def __init__(self,
                 data: List[Union[np.ndarray, Dict[str, np.ndarray]]],
                 metadata: Dict[str, str] = None,
                 fields: Iterable[str] = None,
                 sequence_field: str = None):
        if fields:
            assert sequence_field
            if data:
                assert isinstance(data[0], dict)
        super().__init__(fields=fields, sequence_field=sequence_field)
        self.data = data
        self.metadata = metadata or dict()

    def indexes(self) -> Iterable[int]:
        return range(len(self.data))

    def length(self, index) -> int:
        item = self.retrieve(index)
        sequence = item[self.sequence_field] if self.sequence_field else item
        return sequence.shape[0]

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        return ((index, self.length(index)) for index in self.indexes())

    def retrieve(self, index) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        return self.data[index]

    def num_records(self) -> int:
        return len(self.data)

    def metadata(self, metadata_key: str) -> Optional[str]:
        return self.metadata.get(metadata_key, None)
