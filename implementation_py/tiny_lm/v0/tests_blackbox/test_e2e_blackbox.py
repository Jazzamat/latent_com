import math

import torch

from config import ModelConfig
from data import CharTokenizer, TextDataset
from model import TinyLM
from utils import load_checkpoint, save_checkpoint, set_seed


def test_end_to_end_train_save_load_generate(tmp_path, blackbox_device: str) -> None:
    set_seed(123)
    device = blackbox_device
    text = ("tiny lm learns short patterns.\n" * 200).strip()
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        mlp_hidden=64,
        dropout=0.0,
    )
    dataset = TextDataset(encoded, block_size=model_config.block_size, train_split=0.9)
    model = TinyLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    last_loss = None
    model.train()
    for _step in range(40):
        xb, yb = dataset.get_batch("train", batch_size=16, device=device)
        _logits, loss, _hidden = model(xb, yb)
        assert loss is not None
        assert torch.isfinite(loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()

    assert last_loss is not None
    assert math.isfinite(last_loss)

    checkpoint_path = tmp_path / "tiny_blackbox.pt"
    save_checkpoint(str(checkpoint_path), model, optimizer, model_config, tokenizer)
    assert checkpoint_path.exists()

    loaded_model, _loaded_optimizer, loaded_config, loaded_tokenizer = load_checkpoint(
        str(checkpoint_path),
        device=device,
    )
    loaded_model.eval()

    assert loaded_config == model_config
    assert loaded_tokenizer.stoi == tokenizer.stoi
    assert loaded_tokenizer.itos == tokenizer.itos

    prompt = "tiny "
    idx = torch.tensor([loaded_tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = loaded_model.generate(idx, max_new_tokens=24, temperature=0.8, top_k=8)

    assert out.shape == (1, len(prompt) + 24)
    assert out.min().item() >= 0
    assert out.max().item() < loaded_config.vocab_size

    generated_text = loaded_tokenizer.decode(out[0].detach().cpu().tolist())
    assert len(generated_text) == len(prompt) + 24
