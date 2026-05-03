import random
from types import SimpleNamespace

import pytest
import torch

from config import ModelConfig
from data import CharTokenizer, TextDataset
from model import TinyLM
from utils import estimate_loss, load_checkpoint, save_checkpoint, set_seed


def _build_tiny_stack():
    text = ("hello tiny lm\n" * 40).strip()
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layers=2,
        n_heads=2,
        d_model=16,
        mlp_hidden=32,
        dropout=0.0,
    )
    dataset = TextDataset(encoded, block_size=model_config.block_size, train_split=0.8)
    model = TinyLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, dataset, optimizer, model_config, tokenizer


def test_set_seed_makes_random_and_torch_deterministic() -> None:
    set_seed(1234)
    python_random_1 = random.random()
    torch_random_1 = torch.rand(5)

    set_seed(1234)
    python_random_2 = random.random()
    torch_random_2 = torch.rand(5)

    assert python_random_1 == python_random_2
    assert torch.equal(torch_random_1, torch_random_2)


@pytest.mark.parametrize("start_in_train_mode", [True, False])
def test_estimate_loss_returns_train_and_val_and_restores_mode(start_in_train_mode: bool) -> None:
    model, dataset, _optimizer, _model_config, _tokenizer = _build_tiny_stack()
    train_cfg = SimpleNamespace(eval_iters=3, batch_size=4)

    if start_in_train_mode:
        model.train()
    else:
        model.eval()

    losses = estimate_loss(model, dataset, train_cfg, device="cpu")

    assert set(losses.keys()) == {"train", "val"}
    assert isinstance(losses["train"], float)
    assert isinstance(losses["val"], float)
    assert model.training is start_in_train_mode


def test_checkpoint_roundtrip_preserves_parameters_and_tokenizer(tmp_path) -> None:
    model, _dataset, optimizer, model_config, tokenizer = _build_tiny_stack()
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(str(checkpoint_path), model, optimizer, model_config, tokenizer)
    loaded_model, loaded_optimizer, loaded_model_config, loaded_tokenizer = load_checkpoint(
        str(checkpoint_path),
        device="cpu",
    )

    assert loaded_model_config == model_config
    assert loaded_tokenizer.stoi == tokenizer.stoi
    assert loaded_tokenizer.itos == tokenizer.itos
    for p_expected, p_actual in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p_expected, p_actual)
    assert loaded_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
