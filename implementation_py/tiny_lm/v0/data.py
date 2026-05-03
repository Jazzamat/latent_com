from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class TextDataset:
    def __init__(self, encoded_text: List[int], block_size: int, train_split: float = 0.9):
        data = torch.tensor(encoded_text, dtype=torch.long)
        split_idx = int(len(data) * train_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        self.block_size = block_size

    def get_batch(self, split: str, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(0, len(source) - self.block_size - 1, (batch_size,))

        x = torch.stack([source[i:i + self.block_size] for i in ix])
        y = torch.stack([source[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
