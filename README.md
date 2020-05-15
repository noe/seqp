# seqp

`seqp` is a **sequence persistence** library for Python.

This library is for you if you want to encode text (or any
 other discrete sequential data) in binary form, persist
 it to file and then load it back (e.g. to
 train a machine learning model).
 
You can install it with:
 
```
pip install https://github.com/noe/seqp/archive/0.1.tar.gz
```


# Features

- `seqp` persists sequence data to file in indexed binary format.
- `seqp` can load back your sequence data, retrieving by index.
- `seqp` allows storing metadata along with your sequences
  (e.g. info on how to decode it, like a dictionary in NLP).
- `seqp` gives you padded + bucketed data loading.
- When loading data, `seqp` does not hold the full
  (potentially huge) dataset in memory, but only loads
  data as it is needed.
- `seqp` integrates with other frameworks (if installed)
  to provide synergies.
- `seqp` handles data as numpy arrays and it is not tied to
  any deep learning framework. Nevertheless, it provides
  optional integration components to some popular deep
  learning libraries such as [fairseq](https://github.com/pytorch/fairseq/).

# Use Cases

These are some use cases `seqp` can be used for:

- **Neural machine translation**: to store token IDs and then load them
in batches to train a neural network. See the
[basic text encoding/decoding example](./examples/basic_read_write.ipynb)

- **Text classification/NLI with contextual embeddings**: you can encode your
training text with your favourite contextual embeddings (BERT, EMLo,...)
and save the sequences of vector representations in binary format once,
and in training you just load them back without having to encode them
every time.

- **DNA sequence classification**: store DNA sequences in binary
format and load them back in length-bucketed batches.
See the [DNA example](./examples/sharded_storage.ipynb).

# Quick example

This snippet shows how to serialize sequences in shards of 100000:

```Python
output_file_template = "data_{:02d}.hdf5"

with ShardedWriter(Hdf5RecordWriter,
                   output_file_template,
                   max_records_per_shard=100000) as writer:
    for seq in sequences:
        binarized_seq = binarize_sequence(seq)
        writer.write(np.array(binarized_seq, dtype=np.uint32))
```

And this one shows how to read them back:

```Python
with Hdf5RecordReader(glob('data_*.hdf5')) as reader:
    for seq_idx in reader.indexes():
        binarized_seq = reader.retrieve(seq_idx)
```

# Complete examples

`seqp` offers several jupyter notebooks with usage examples:

- [Save text as token IDs and load them back](./examples/basic_read_write.ipynb)
- [Use fields to add the dependency parse of each sequence](./examples/fields.ipynb)
- [Sharded storage of DNA data](./examples/sharded_storage.ipynb)
- [Sharded storage of BERT representations](./examples/bert.ipynb)
- [Reading records in batches, bucketing by sequence length](./examples/data_load.ipynb)


Also, `seqp` offers some very simple command line tools that can
serve as usage examples:

- [`seqp-bert`](./tools/seqp-bert.py): saves the contextual
[BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
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

