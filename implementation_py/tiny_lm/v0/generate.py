import torch

from config import TrainConfig
from utils import load_checkpoint


def main() -> None:
    train_config = TrainConfig()
    device = train_config.device if torch.cuda.is_available() else "cpu"

    model, _optimizer, _model_config, tokenizer = load_checkpoint(train_config.checkpoint_path, device)
    model.eval()

    prompt = "Hello"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=200)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
