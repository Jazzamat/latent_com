from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class CommModelConfig:
    sender_vocab_size: int
    receiver_vocab_size: int
    n_locations: int
    d_model: int = 64
    hidden_size: int = 64
    message_dim: int = 32


class SenderEncoder(nn.Module):
    def __init__(self, config: CommModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.sender_vocab_size, config.d_model)
        self.rnn = nn.GRU(config.d_model, config.hidden_size, batch_first=True)
        self.proj = nn.Linear(config.hidden_size, config.message_dim)

    def forward(self, sender_tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sender_tokens)
        _outputs, h_n = self.rnn(x)
        final_hidden = h_n[-1]
        return self.proj(final_hidden)


class ReceiverPolicy(nn.Module):
    def __init__(self, config: CommModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.receiver_vocab_size, config.d_model)
        self.rnn = nn.GRU(config.d_model, config.hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size + config.message_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.n_locations),
        )

    def forward(self, receiver_tokens: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        x = self.embedding(receiver_tokens)
        _outputs, h_n = self.rnn(x)
        final_hidden = h_n[-1]
        fused = torch.cat([final_hidden, message], dim=-1)
        return self.head(fused)


class CommAgents(nn.Module):
    def __init__(self, config: CommModelConfig):
        super().__init__()
        self.config = config
        self.sender = SenderEncoder(config)
        self.receiver = ReceiverPolicy(config)

    @staticmethod
    def _apply_message_mode(message: torch.Tensor, message_mode: str) -> torch.Tensor:
        if message_mode == "with_message":
            return message
        if message_mode == "zero_message":
            return torch.zeros_like(message)
        if message_mode == "shuffled_message":
            if message.size(0) <= 1:
                return message
            perm = torch.randperm(message.size(0), device=message.device)
            return message[perm]
        raise ValueError(f"Unsupported message_mode: {message_mode}")

    def forward(
        self,
        sender_tokens: torch.Tensor,
        receiver_tokens: torch.Tensor,
        message_mode: str = "with_message",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_message = self.sender(sender_tokens)
        message = self._apply_message_mode(raw_message, message_mode)
        logits = self.receiver(receiver_tokens, message)
        return logits, message
