from seqp.vocab import (Vocabulary,
                        VocabularyCollector,
                        DEFAULT_SPECIAL_SYMBOLS,
                        vocab_to_codec,
                        )

import pytest

text = [
        "My friend's name is David, he is in Miss Krabappel's class.",
        "Don't underestimate the strength of the dark side."
       ]


def test_collector():
    collector = VocabularyCollector()
    for t in text:
        for s in t.split(' '):
            collector.add_symbol(s)
    vocab1 = collector.consolidate()
    assert "name" in vocab1.idx2symbol

    vocab2 = collector.consolidate(max_num_symbols=5)
    assert "the" in vocab2.idx2symbol


@pytest.fixture
def vocab():
    """
    Returns a dummy Vocabulary with three symbols 'mouse', 'dog' and 'tree'
    (apart from the default special symbols)
    """
    symbols = DEFAULT_SPECIAL_SYMBOLS + ["mouse", "dog", "tree"]
    return Vocabulary(symbols)


def test_vocab(vocab):
    assert [vocab.num_special] == vocab.encode(["mouse"], add_eos=False)
    assert [vocab.num_special, vocab.eos_id] == vocab.encode(["mouse"], add_eos=True)


def test_vocab_persistence(vocab):
    json_string = vocab.to_json()
    assert isinstance(json_string, str)
    restored_vocab = Vocabulary.from_json(json_string)
    assert vars(vocab) == vars(restored_vocab)


def test_vocab2codec():
    # TODO
    pass
