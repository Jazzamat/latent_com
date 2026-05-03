import pytest
import torch

from comm_model import CommAgents, CommModelConfig


def _build_model() -> CommAgents:
    config = CommModelConfig(
        sender_vocab_size=12,
        receiver_vocab_size=4,
        n_locations=4,
        d_model=16,
        hidden_size=16,
        message_dim=8,
    )
    return CommAgents(config)


def test_forward_shapes_with_message() -> None:
    model = _build_model()
    sender_tokens = torch.randint(0, 12, (5, 4), dtype=torch.long)
    receiver_tokens = torch.randint(0, 4, (5, 4), dtype=torch.long)

    logits, message = model(sender_tokens, receiver_tokens, message_mode="with_message")

    assert logits.shape == (5, 4)
    assert message.shape == (5, 8)


def test_zero_message_mode_returns_zero_message_tensor() -> None:
    model = _build_model()
    sender_tokens = torch.randint(0, 12, (3, 4), dtype=torch.long)
    receiver_tokens = torch.randint(0, 4, (3, 4), dtype=torch.long)

    _logits, message = model(sender_tokens, receiver_tokens, message_mode="zero_message")

    assert torch.equal(message, torch.zeros_like(message))


def test_invalid_message_mode_raises() -> None:
    model = _build_model()
    sender_tokens = torch.randint(0, 12, (2, 4), dtype=torch.long)
    receiver_tokens = torch.randint(0, 4, (2, 4), dtype=torch.long)

    with pytest.raises(ValueError):
        model(sender_tokens, receiver_tokens, message_mode="unknown")


def test_gradients_flow_through_model() -> None:
    model = _build_model()
    sender_tokens = torch.randint(0, 12, (4, 4), dtype=torch.long)
    receiver_tokens = torch.randint(0, 4, (4, 4), dtype=torch.long)
    targets = torch.randint(0, 4, (4,), dtype=torch.long)

    logits, _message = model(sender_tokens, receiver_tokens, message_mode="with_message")
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    sender_grads = [p.grad for p in model.sender.parameters() if p.requires_grad]
    receiver_grads = [p.grad for p in model.receiver.parameters() if p.requires_grad]
    assert any(grad is not None for grad in sender_grads)
    assert any(grad is not None for grad in receiver_grads)
