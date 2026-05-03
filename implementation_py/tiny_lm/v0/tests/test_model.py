import pytest
import torch

from config import ModelConfig
from model import TinyLM


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        block_size=8,
        n_layers=2,
        n_heads=2,
        d_model=16,
        mlp_hidden=32,
        dropout=0.0,
    )


def test_forward_returns_logits_loss_and_none_hidden_by_default() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))

    logits, loss, hidden = model(idx, targets)

    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss is not None
    assert loss.shape == torch.Size([])
    assert hidden is None


def test_forward_without_targets_returns_none_loss() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    logits, loss, hidden = model(idx)

    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is None
    assert hidden is None


def test_forward_returns_minimal_hidden_state_contract() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    logits, loss, hidden = model(idx, return_hidden_states=True)

    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is None
    assert isinstance(hidden, dict)
    assert "layers" in hidden
    layers = hidden["layers"]
    assert isinstance(layers, list)
    assert len(layers) == cfg.n_layers
    for layer_state in layers:
        assert layer_state.shape == (2, cfg.block_size, cfg.d_model)


def test_forward_raises_for_sequence_longer_than_block_size() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))

    with pytest.raises(ValueError):
        model(idx)


def test_logits_for_earlier_positions_do_not_change_when_future_token_changes() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    model.eval()

    seq_a = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    seq_b = seq_a.clone()
    seq_b[0, -1] = 7

    with torch.no_grad():
        logits_a, _loss_a, _hidden_a = model(seq_a)
        logits_b, _loss_b, _hidden_b = model(seq_b)

    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6)


def test_generate_appends_requested_number_of_tokens() -> None:
    cfg = _small_model_config()
    model = TinyLM(cfg)
    model.eval()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    out = model.generate(prompt, max_new_tokens=5)

    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], prompt)
    assert out.min().item() >= 0
    assert out.max().item() < cfg.vocab_size
