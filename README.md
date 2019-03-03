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

- Save text as token IDs and load them back **TODO**
- Sharded storage **TODO**
- Use fields to add the dependency parse of each sequence **TODO**
- Reading records in batches, bucketing by sequence length **TODO**

Also, `seqp` offers some command line tools
- `seqp-bert` **TODO**
- `seqp-elmo` **TODO**

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

### Should `RecordReader` and `RecordWriter` be used as context managers?

All record readers and