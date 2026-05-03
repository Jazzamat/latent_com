import random
from typing import Dict

import torch

from config import ModelConfig
from model import TinyLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(model, dataset, train_config, device: str) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            xb, yb = dataset.get_batch(split, train_config.batch_size, device)
            _, loss, _ = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    if was_training:
        model.train()
    else:
        model.eval()
    return out


def save_checkpoint(path, model, optimizer, model_config, tokenizer) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        path,
    )


def load_checkpoint(path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config: ModelConfig = checkpoint["model_config"]
    model = TinyLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    from data import CharTokenizer
    tokenizer = CharTokenizer(stoi=checkpoint["stoi"], itos=checkpoint["itos"])
    return model, optimizer, model_config, tokenizer
