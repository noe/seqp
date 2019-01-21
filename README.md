
`seqp` is a sequence persistence library for Python.

This library is for you if you want to encode text in binary form and
then load it back. Some use cases:

- In neural machine translation: to store token IDs and then load them
in batches to train a neural network.

- In text classification/NLI with contextual embeddings: you can encode your
training text with your favourite contextual embeddings (BERT, EMLo,...)
and save the sequences of vector representations in binary format once,
and in training you just load them back without having to encode them
every time.

This library was created due to the fact that for each different deep
learning framework you have to use a different approach for that: TF records
for tensorflow (+ tensor2tensor), custom binary format for for fairseq, etc.

If your deep learning library is just another component in your pipeline
and you don't want to couple all the pipeline to it, `seqp` is for you.

Under the hoods it uses HDF5 to store the data in binary format.
