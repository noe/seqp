# seqp

`seqp` is a **sequence persistence** library for Python.

This library is for you if you want to encode text in binary form and
then load it back. Some use cases:

- In neural machine translation: to store token IDs and then load them
in batches to train a neural network.

- In text classification/NLI with contextual embeddings: you can encode your
training text with your favourite contextual embeddings (BERT, EMLo,...)
and save the sequences of vector representations in binary format once,
and in training you just load them back without having to encode them
every time.

`seqp` handles data as numpy arrays and it is not tied to any deep
learning framework. Nevertheless, it provides optional integration
components to some popular deep learning libraries such as fairseq.

# Functionality in seqp

- `seqp` saves to sequence data in binary format.
- `seqp` can load back your sequence data.
- `seqp` gives you padded + bucketed iterated data batching.
- `seqp` does not hold the full (potentially huge) dataset in memory.
- `seqp` integrates with other frameworks (if installed) to provide
  synergies.

# Examples

`seqp` offers several jupyter notebooks with usage examples:

- [Save text as token IDs and load them back](./examples/basic_read_write.ipynb)
- Use fields to add the dependency parse of each sequence **TODO**
- [Sharded storage of DNA data](./examples/sharded_storage.ipynb)
- Sharded storage of BERT representations **TODO**
- Reading records in batches, bucketing by sequence length **TODO**
- Integration with bare pytorch **TODO**
- Integration with fairseq **TODO**

Also, `seqp` offers some very simple command line tools that can
serve as usage examples:

- [`seqp-bert`](./tools/seqp-bert.py): saves the contextual BERT
representation of the given text into HDF5 files.
- [`seqp-elmo`](./tools/seqp-elmo.py): saves the contextual ELMo
representation of the given text into HDF5 files.

# Motivation

This library was created due to the fact that for each different deep
learning framework you have to use a different approach for sequence
persistence:
- TF records for tensorflow / tensor2tensor.
- custom binary format for for fairseq.
- ...

If your deep learning library is just another component in your pipeline
and you don't want to couple all the pipeline to it, `seqp` is for you.

# FAQ

#### Can I save extra pieces of information along with each sequence?

Yes.

`RecordWriter.write` is normally given a numpy array with
the sequence data. However, it can also be passed a dictionary
with "fields" associated to the sequence. `RecordReader` supports
reading fields analogously. Have a look at the [field usage
example](./examples/fields.ipynb) to learn more.


#### Should `RecordReader` and `RecordWriter` be used as context managers?

Record readers and record writers can be used as context
managers in a `with` block for safe resource deallocation.
This is the appropriate way of using them if you are going
to read or write everything in a single loop, after which
their resources can be deallocated.

You can also instantiate them as normal variables and release
them by invoking method `close` on them. This is the appropriate
way of using them if you want to keep them in object member
variables.

#### Why `RecordReader` returns an iterator to indexes and lengths instead of the sequences themselves?

`seqp` is designed to eagerly read sequence indexes and their lengths,
and lazily read sequences themselves. This design allows to
efficiently implement sequence loading strategies like
length-driven sequence bucketing.
That is the reason why sequences have to be retrieved by index
separately.

#### Why there is `ShardedWriter` but not `ShardedReader`?

While sharded writing seemed more natural as two split
functionalities, reading from multiple files seemed too
coupled to _factor out_ the sharding functionality. Merely
seeking symmetry seemed just not good enough reason to
move the sharded reading out of `Hdf5Reader`.

