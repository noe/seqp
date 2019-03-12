# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.


from bisect import bisect_right
import math
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Union, Tuple, Callable

from .record import RecordReader


def _compute_bucket_limits(total_count: int,
                           num_buckets: int,
                           length_counter: Counter):
    """
    Computes bucket limits to have the same amount of sentences in all buckets
    :param total_count:
    :param num_buckets:
    :param min_length:
    :param length_counter:
    :return:
    """
    bucket_size = total_count // num_buckets
    bucket_limits = []
    bucket_min = 0
    bucket_max = 0

    length_counts = sorted(list(length_counter.items()), key=lambda x: x[0])
    length_idx = 0

    for bucket_idx in range(num_buckets):
        if length_idx >= len(length_counts):
            break

        elems_in_bucket = 0

        while elems_in_bucket < bucket_size and length_idx < len(length_counts):
            length, count = length_counts[length_idx]
            elems_in_bucket += count
            bucket_max = length
            length_idx += 1

        bucket_limits.append((bucket_min, bucket_max))
        bucket_min = bucket_max + 1

    is_last_bucket_too_small = elems_in_bucket < 0.5 * bucket_size

    if is_last_bucket_too_small:
        # fuse last bucket with the previous one
        new_last_bucket_limits = (bucket_limits[-2][0], bucket_limits[-1][1])
        bucket_limits = bucket_limits[:-2] + [new_last_bucket_limits]

    return bucket_limits


def _get_bucket_id(bucket_mins: List[int], max_possible_length, length: int):
    if length < bucket_mins[0]:
        # less than minimum length
        return -1
    if length > max_possible_length:
        # greater than maximum length
        return len(bucket_mins)

    bucket_id = bisect_right(bucket_mins, length) - 1
    return bucket_id


def _get_max_bucket_id(bucket_mins, bucket_limits, length):
    bucket_id = _get_bucket_id(bucket_mins, bucket_limits[-1][1], length)
    if bucket_id >= len(bucket_mins):
        return bucket_id - 1
    bucket_min, bucket_max = bucket_limits[bucket_id]
    return bucket_id if bucket_max == length else bucket_id - 1


def _get_min_bucket_id(bucket_mins, bucket_limits, length):
    bucket_id = _get_bucket_id(bucket_mins, bucket_limits[-1][1], length)
    if bucket_id < 0:
        return 0
    bucket_min, bucket_max = bucket_limits[bucket_id]
    return bucket_id if bucket_min == length else bucket_id + 1


def _get_clipped_bucket_id(bucket_mins, max_possible_length, length):
    bucket_id = _get_bucket_id(bucket_mins, max_possible_length, length)
    return max(min(bucket_id, len(bucket_mins) - 1), 0)


class DataLoader:
    def __init__(self, reader: RecordReader, pad_value=0, num_buckets=100):
        self.record_reader = reader
        self.pad_value = pad_value
        counter = Counter([l for i, l in reader.indexes_and_lengths()])
        self.bucket_limits = _compute_bucket_limits(self.record_reader.num_records(),
                                                    num_buckets,
                                                    counter)
        self.num_buckets = len(self.bucket_limits)  # number of buckets is determined in the limits

        self.buckets = defaultdict(list)
        self.bucket_mins = [self.bucket_limits[k][0] for k in range(self.num_buckets)]
        self.max_length = self.bucket_limits[-1][1]

        for index, length in self.record_reader.indexes_and_lengths():
            bucket_id = _get_bucket_id(self.bucket_mins, self.max_length, length)
            self.buckets[bucket_id].append(index)

    def iterator(self,
                 batch_size: Union[int, Callable[[int], int]],
                 is_size_in_tokens: bool=False,
                 dynamic_seq_length: Callable[[], Tuple[int, int]]=None,
                 dynamic_padded_length: Callable[[int], int]=None,
                 length_schedule: Callable[[int], float]=None):

        batch_size_func = batch_size if callable(batch_size) else lambda _: batch_size
        if is_size_in_tokens:
            assert not callable(batch_size), \
                "Dynamic batch size is not compatible with token-based size"

        assert not length_schedule or not dynamic_seq_length, \
            "Only one of length_schedule or dynamic_seq_length can be supplied"

        def pad(s, seq_length):
            s = s[:]
            rest_dim = len(s.shape) - 1
            pad_width = [(0, seq_length - s.shape[0])] + [(0, 0)] * rest_dim
            pad_values = [(0, self.pad_value)] + [(0, 0)] * rest_dim
            return np.pad(s, pad_width, 'constant', constant_values=pad_values)

        iteration = -1
        while True:
            iteration += 1

            # select random bucket
            lowest_bucket_id = 0
            highest_bucket_id = len(self.buckets) - 1

            if dynamic_seq_length:
                current_min_length, current_max_length = dynamic_seq_length()
                lowest_bucket_id = _get_min_bucket_id(self.bucket_limits, current_min_length)
                highest_bucket_id = _get_max_bucket_id(self.bucket_limits, current_max_length)
            elif length_schedule:
                ratio_min, ratio_max = length_schedule(iteration)
                assert ratio_max >= ratio_min
                lowest_bucket_id = int((len(self.bucket_limits) - 1) * ratio_min)
                highest_bucket_id = int((len(self.bucket_limits) - 1) * ratio_max)

            bucket_id = random.randint(lowest_bucket_id, highest_bucket_id)

            max_length = self.bucket_limits[bucket_id][1]
            padded_length = max_length
            if dynamic_padded_length:
                padded_length = dynamic_padded_length(max_length)

            assert padded_length >= max_length

            # sample from bucket (number of sentences)
            current_batch_size = batch_size_func(padded_length)

            num_sentences = (current_batch_size if not is_size_in_tokens
                             else int(math.floor(current_batch_size / float(padded_length))))

            bucket_len = len(self.buckets[bucket_id])
            assert num_sentences <= bucket_len, f"Bucket {bucket_id} only has {bucket_len} elems"

            sentence_keys = random.sample(self.buckets[bucket_id], num_sentences)

            sentences = np.array([pad(self.record_reader.retrieve(k), padded_length)
                                  for k in sentence_keys])

            yield sentences


class CoordinatedDataLoader:
    def __init__(self,
                 data_loaders: List[DataLoader],
                 length_ratios: List[float]=None):
        assert len(data_loaders) > 1
        if length_ratios is None:
            length_ratios = [0.] + [1.] * (len(data_loaders) - 1)
        assert len(length_ratios) == len(data_loaders)
        self.leader_dl = data_loaders[0]
        self.follower_dls = data_loaders[1:]
        self.follower_length_ratios = length_ratios[1:]
        self.largest_follower_ratio = max(self.follower_length_ratios)

    def iterator(self,
                 batch_size,
                 is_size_in_tokens=False,
                 leader_length_schedule=None):
        last_num_sentences = 0
        last_padded_length = 0

        def leader_batch_size(padded_length):
            nonlocal last_num_sentences
            num_sentences = (batch_size if not is_size_in_tokens
                             else int(math.floor(batch_size / float(padded_length))))
            last_num_sentences = num_sentences
            return num_sentences

        def leader_dynamic_padded_length(max_length):
            nonlocal last_padded_length
            length = max(f.bucket_limits[_get_clipped_bucket_id(f.bucket_limits, int(r * max_length))][1]
                         for f, r in zip(self.follower_dls, self.follower_length_ratios))
            length = max(length, max_length)
            last_padded_length = length
            return length

        def followers_dynamic_seq_length():
            return 0, last_padded_length

        def followers_dynamic_padded_length(max_length):
            return last_padded_length

        def followers_batch_size(padded_length):
            return last_num_sentences

        leader_it = self.leader_dl.iterator(leader_batch_size,
                                            is_size_in_tokens=False,
                                            dynamic_padded_length=leader_dynamic_padded_length,
                                            length_schedule=leader_length_schedule)
        follower_its = [dl.iterator(followers_batch_size,
                                    is_size_in_tokens=False,
                                    dynamic_seq_length=followers_dynamic_seq_length,
                                    dynamic_padded_length=followers_dynamic_padded_length)
                        for dl in self.follower_dls]

        yield from zip(leader_it, *follower_its)
