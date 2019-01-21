# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.


def count_lines(file_name: str) -> int:
    """
    Counts the number of lines in a plain text file.
    :param file_name: File name.
    :return: The number of lines in the file.
    """
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
