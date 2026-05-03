from typing import Dict

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(model, dataloader: DataLoader, device: str, message_mode: str) -> float:
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    for sender_tokens, receiver_tokens, targets in dataloader:
        sender_tokens = sender_tokens.to(device)
        receiver_tokens = receiver_tokens.to(device)
        targets = targets.to(device)

        logits, _message = model(
            sender_tokens=sender_tokens,
            receiver_tokens=receiver_tokens,
            message_mode=message_mode,
        )
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()

    if was_training:
        model.train()
    else:
        model.eval()

    return correct / max(1, total)


@torch.no_grad()
def evaluate_all_modes(model, dataloader: DataLoader, device: str) -> Dict[str, float]:
    with_acc = evaluate_accuracy(model, dataloader, device, message_mode="with_message")
    zero_acc = evaluate_accuracy(model, dataloader, device, message_mode="zero_message")
    shuffled_acc = evaluate_accuracy(model, dataloader, device, message_mode="shuffled_message")
    return {
        "with_message_acc": with_acc,
        "zero_message_acc": zero_acc,
        "shuffled_message_acc": shuffled_acc,
        "comm_gain": with_acc - zero_acc,
    }
