from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    mlp_hidden: int = 1024
    dropout: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    text_path: str = "data/input.txt"
    train_split: float = 0.9

    batch_size: int = 32
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 50

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    checkpoint_path: str = "checkpoints/tiny_lm.pt"
