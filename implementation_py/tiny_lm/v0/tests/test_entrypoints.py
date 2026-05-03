import importlib
from types import SimpleNamespace

import torch


def test_train_main_runs_training_and_saves_checkpoint(monkeypatch, tmp_path) -> None:
    train = importlib.import_module("train")

    train_cfg = SimpleNamespace(
        seed=7,
        device="cpu",
        text_path="ignored.txt",
        train_split=0.8,
        batch_size=2,
        max_iters=3,
        eval_interval=2,
        eval_iters=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,
        checkpoint_path=str(tmp_path / "checkpoints" / "tiny.pt"),
    )

    class FakeTokenizer:
        def __init__(self) -> None:
            self.vocab_size = 8

        @classmethod
        def from_text(cls, _text: str):
            return cls()

        def encode(self, _text: str):
            return [0, 1, 2, 3, 4, 5, 6, 7] * 4

    class FakeDataset:
        def __init__(self, encoded_text, block_size, train_split):
            self.encoded_text = encoded_text
            self.block_size = block_size
            self.train_split = train_split

        def get_batch(self, split: str, batch_size: int, device: str):
            assert split in {"train", "val"}
            x = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
            y = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
            return x, y

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, xb, yb=None):
            logits = torch.zeros((xb.shape[0], xb.shape[1], 8), device=xb.device)
            loss = self.param * 0 + torch.tensor(1.0, device=xb.device)
            return logits, loss, None

    calls = {"estimate_loss": 0, "save_checkpoint": 0}

    monkeypatch.setattr(train, "TrainConfig", lambda: train_cfg)
    monkeypatch.setattr(train, "set_seed", lambda _seed: None)
    monkeypatch.setattr(train, "load_text", lambda _path: "tiny corpus")
    monkeypatch.setattr(train, "CharTokenizer", FakeTokenizer)
    monkeypatch.setattr(train, "TextDataset", FakeDataset)
    monkeypatch.setattr(train, "TinyLM", lambda _cfg: FakeModel())
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    def fake_estimate_loss(*_args, **_kwargs):
        calls["estimate_loss"] += 1
        return {"train": 1.0, "val": 1.2}

    def fake_save_checkpoint(*_args, **_kwargs):
        calls["save_checkpoint"] += 1

    monkeypatch.setattr(train, "estimate_loss", fake_estimate_loss)
    monkeypatch.setattr(train, "save_checkpoint", fake_save_checkpoint)

    train.main()

    assert calls["estimate_loss"] == 2
    assert calls["save_checkpoint"] == 1


def test_generate_main_loads_checkpoint_and_prints_decoded_text(monkeypatch, capsys) -> None:
    generate = importlib.import_module("generate")

    train_cfg = SimpleNamespace(device="cpu", checkpoint_path="ignored.pt")

    class FakeTokenizer:
        def encode(self, text: str):
            assert text == "Hello"
            return [1, 2, 3]

        def decode(self, ids):
            assert ids == [1, 2, 3, 4]
            return "decoded text"

    class FakeModel:
        def eval(self):
            return self

        def generate(self, idx, max_new_tokens: int):
            assert max_new_tokens == 200
            assert idx.shape == (1, 3)
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    monkeypatch.setattr(generate, "TrainConfig", lambda: train_cfg)
    monkeypatch.setattr(generate.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        generate,
        "load_checkpoint",
        lambda _path, _device: (FakeModel(), None, None, FakeTokenizer()),
    )

    generate.main()
    printed = capsys.readouterr().out.strip()
    assert printed == "decoded text"
