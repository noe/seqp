# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import numpy as np
from seqp.record import RecordReader
from typing import Iterable, Dict, Optional, Tuple, Union


def count_lines(file_name: str) -> int:
    """
    Counts the number of lines in a plain text file.
    :param file_name: File name.
    :return: The number of lines in the file.
    """
    with open(file_name) as f:
        i = 0
        for i, l in enumerate(f):
            pass
        return i + 1


class EosAppender(RecordReader):
    """
    Wrapper object to add EOS to the end of the sequences
    and adjusts lengths accordingly.
    """

    def __init__(self, wrapped, eos):
        super().__init__(wrapped.fields, wrapped.sequence_field)
        self.wrapped  = wrapped
        self.eos = eos

    def close(self):
        self.wrapped.close()

    def indexes(self) -> Iterable[int]:
        yield from self.wrapped.indexes()

    def length(self, index) -> int:
        return 1 + self.wrapped.length(index)

    def indexes_and_lengths(self) -> Iterable[Tuple[int, int]]:
        for i, l in self.wrapped.indexes_and_lengths():
            yield i, l + 1

    def retrieve(self, index) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        record = self.wrapped.retrieve(index)
        sequence = (record if not self.wrapped.fields
                    else record[self.wrapped.sequence_field])
        sequence = np.append(sequence, self.eos)
        if self.wrapped.fields:
            record[self.wrapped.sequence_field] = sequence
        else:
            record = sequence

        return record

    def num_records(self) -> int:
        return self.wrapped.num_records()

    def metadata(self, metadata_key: str) -> Optional[str]:
        return self.metadata(metadata_key)

