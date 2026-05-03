from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from comm_data import CommDataConfig, build_datasets_with_signature_ood
from comm_model import CommAgents, CommModelConfig
from eval_comm import evaluate_all_modes


@dataclass
class CommTrainConfig:
    seed: int = 42
    device: str = "cuda"

    n_locations: int = 4
    n_moves: int = 3
    protagonist_sees_prob: float = 0.5
    train_size: int = 4000
    val_size: int = 800
    ood_val_size: int = 800
    ood_holdout_ratio: float = 0.25
    ood_signature_seed: int | None = None

    d_model: int = 64
    hidden_size: int = 64
    message_dim: int = 32

    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    print_every: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _with_prefix(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_experiment(config: CommTrainConfig) -> dict:
    set_seed(config.seed)
    device = config.device if torch.cuda.is_available() else "cpu"

    data_config = CommDataConfig(
        n_locations=config.n_locations,
        n_moves=config.n_moves,
        protagonist_sees_prob=config.protagonist_sees_prob,
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    train_dataset, iid_val_dataset, ood_val_dataset, heldout_signatures = build_datasets_with_signature_ood(
        config=data_config,
        holdout_ratio=config.ood_holdout_ratio,
        ood_val_size=config.ood_val_size,
        signature_seed=config.ood_signature_seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    iid_val_loader = DataLoader(iid_val_dataset, batch_size=config.batch_size, shuffle=False)
    ood_val_loader = DataLoader(ood_val_dataset, batch_size=config.batch_size, shuffle=False)

    model_config = CommModelConfig(
        sender_vocab_size=train_dataset.sender_vocab_size,
        receiver_vocab_size=train_dataset.receiver_vocab_size,
        n_locations=config.n_locations,
        d_model=config.d_model,
        hidden_size=config.hidden_size,
        message_dim=config.message_dim,
    )
    model = CommAgents(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for sender_tokens, receiver_tokens, targets in train_loader:
            sender_tokens = sender_tokens.to(device)
            receiver_tokens = receiver_tokens.to(device)
            targets = targets.to(device)

            logits, _message = model(
                sender_tokens=sender_tokens,
                receiver_tokens=receiver_tokens,
                message_mode="with_message",
            )
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % config.print_every == 0 or epoch == config.epochs - 1:
            iid_val_metrics = evaluate_all_modes(model, iid_val_loader, device)
            ood_val_metrics = evaluate_all_modes(model, ood_val_loader, device)
            avg_loss = epoch_loss / max(1, n_batches)
            print(
                f"epoch={epoch + 1} loss={avg_loss:.4f} "
                f"iid_with={iid_val_metrics['with_message_acc']:.3f} "
                f"iid_zero={iid_val_metrics['zero_message_acc']:.3f} "
                f"ood_with={ood_val_metrics['with_message_acc']:.3f} "
                f"ood_zero={ood_val_metrics['zero_message_acc']:.3f}"
            )

    iid_final_metrics = evaluate_all_modes(model, iid_val_loader, device)
    ood_final_metrics = evaluate_all_modes(model, ood_val_loader, device)

    return {
        "device": device,
        "train_size": config.train_size,
        "val_size": config.val_size,
        "ood_val_size": config.ood_val_size,
        "ood_holdout_ratio": config.ood_holdout_ratio,
        "ood_signature_seed": config.ood_signature_seed,
        "heldout_visibility_signatures": heldout_signatures,
        **_with_prefix(iid_final_metrics, "iid"),
        **_with_prefix(ood_final_metrics, "ood"),
    }


def main() -> None:
    config = CommTrainConfig()
    results = run_experiment(config)
    print("Final metrics:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
