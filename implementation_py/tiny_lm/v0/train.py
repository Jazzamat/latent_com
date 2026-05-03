import os

import torch

from config import ModelConfig, TrainConfig
from data import CharTokenizer, TextDataset, load_text
from model import TinyLM
from utils import estimate_loss, save_checkpoint, set_seed


def main() -> None:
    train_config = TrainConfig()
    set_seed(train_config.seed)

    device = train_config.device if torch.cuda.is_available() else "cpu"

    text = load_text(train_config.text_path)
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    dataset = TextDataset(encoded, block_size=model_config.block_size, train_split=train_config.train_split)
    model = TinyLM(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    for step in range(train_config.max_iters):
        if step % train_config.eval_interval == 0:
            losses = estimate_loss(model, dataset, train_config, device)
            print(f"step={step} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")

        xb, yb = dataset.get_batch("train", train_config.batch_size, device)
        _, loss, _ = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

    checkpoint_dir = os.path.dirname(train_config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(train_config.checkpoint_path, model, optimizer, model_config, tokenizer)


if __name__ == "__main__":
    main()
