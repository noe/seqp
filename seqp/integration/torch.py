# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

try:
    import torch
    from torch.multiprocessing import Queue, Process
except ImportError:
    assert False, "Pytorch is needed for seqp integration with Pytorch!"

import numpy as np
from typing import Iterable, Union


def subprocess_prefetch(generator: Iterable[Union[np.array, Iterable[np.array]]],
                        prefetch_buffer_size: int,
                        )->Iterable[Union[np.array, Iterable[np.array]]]:
    """
    Wraps a generator to prefect batches in a separate subprocess. It can
    be used in a `with` block (which grants proper resource cleanup) or
    directly as a normal generator. It relies on the ability of
    torch.multiprocessing to load Tensors in shared memory; this way,
    the subprocess loads the numpy array from disk, creates a torch Tensor
    from it and then sends it through a Queue to the main process, which
    consumes it.

    :param generator: Generator to wrap.
    :param prefetch_buffer_size: Size of the prefetch buffer.
    :return: Wrapped generator.
    """
    batch_queue = Queue(prefetch_buffer_size)
    control_queue = Queue()
    Process(target=_enqueue_loader_output,
            args=(batch_queue, control_queue, generator)).start()
    control_queue.put(True)
    return _BatchIterator(batch_queue, control_queue)


class _BatchIterator(object):
    """
    Internal class to iterate batches gotten from a separate subprocess
    through a queue.
    """
    def __init__(self, batch_queue, control_queue):
        self.batch_queue = batch_queue
        self.control_queue = control_queue

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.batch_queue.get()
        if batch is None:
            raise StopIteration()
        return batch

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Request process ends
        self.control_queue.put(False)


def _enqueue_loader_output(batch_queue, control_queue, generator):
    while True:
        ctrl = control_queue.get()
        if ctrl is False:
            break

        for batches in generator:
            if isinstance(batches, tuple):
                batches = tuple([torch.from_numpy(b) for b in batches])
            batch_queue.put(batches)
        batch_queue.put(None)
