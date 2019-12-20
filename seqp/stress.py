import argparse
from glob import glob
import numpy as np
from seqp.hdf5 import Hdf5RecordWriter, Hdf5RecordReader
from seqp.record import ShardedWriter
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta
import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def get_size(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def load():
    #files = glob('/tmp/paco_seqp/test_*.hdf5')
    files = ['/tmp/paco_seqp/test_{}.hdf5'.format(k) for k in range(1, 27)]
    #files = ['/tmp/paco_seqp/test_{}.hdf5'.format(k) for k in [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

    start_time = timer()
    with Hdf5RecordReader(files) as reader:
        num_records = sum(1 for _ in reader.indexes())
        end_time = timer()
        size = get_size(reader)
        print("Num. records: {}".format(num_records))
        print("Load time: {}".format(timedelta(seconds=end_time - start_time)))
        print("Reader size: {} Mb".format(size // 1000000))

        idx = 2000000
        r = reader.retrieve(idx)
        print(f"Record {idx} is {r}")


def save():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('')
    seq_length = 5
    num_fields = 5
    num_segments = 8000000
    output_file_template = "/tmp/paco_seqp/test_{}.hdf5"
    fields = ['field_{}'.format(k) for k in range(num_fields)]
    with ShardedWriter(Hdf5RecordWriter,
                       output_file_template,
                       max_records_per_shard=200000,
                       fields=fields,
                       sequence_field=fields[0]) as writer:
        for idx in tqdm(range(num_segments)):
            k = (idx + 1) % 12347
            writer.write({f: k * np.ones(seq_length, dtype=np.int16) for f in fields})


if __name__ == '__main__':
    load()
