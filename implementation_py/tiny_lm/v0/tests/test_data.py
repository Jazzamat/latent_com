import pytest
import torch

from data import CharTokenizer, TextDataset


def test_tokenizer_builds_sorted_vocabulary() -> None:
    tokenizer = CharTokenizer.from_text("cbabac")
    assert list(tokenizer.stoi.keys()) == ["a", "b", "c"]
    assert tokenizer.vocab_size == 3


def test_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = CharTokenizer.from_text("hello world")
    text = "world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_raises_on_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tokenizer.encode("d")


def test_dataset_split_uses_train_split_ratio() -> None:
    encoded = list(range(20))
    dataset = TextDataset(encoded_text=encoded, block_size=4, train_split=0.75)
    assert len(dataset.train_data) == 15
    assert len(dataset.val_data) == 5


def test_get_batch_has_expected_shape_and_next_token_shift() -> None:
    encoded = list(range(200))
    dataset = TextDataset(encoded_text=encoded, block_size=8, train_split=0.8)

    x, y = dataset.get_batch(split="train", batch_size=4, device="cpu")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)


def test_get_batch_supports_validation_split() -> None:
    encoded = list(range(200))
    dataset = TextDataset(encoded_text=encoded, block_size=8, train_split=0.8)

    x, y = dataset.get_batch(split="val", batch_size=3, device="cpu")
    assert x.shape == (3, 8)
    assert y.shape == (3, 8)
