from dataclasses import dataclass
from random import Random
from typing import List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class Episode:
    initial_location: int
    true_location: int
    belief_location: int
    sender_tokens: List[int]
    receiver_tokens: List[int]
    saw_flags: List[bool]


@dataclass
class CommDataConfig:
    n_locations: int = 4
    n_moves: int = 3
    protagonist_sees_prob: float = 0.5
    train_size: int = 4000
    val_size: int = 800
    seed: int = 42


class BeliefCommunicationDataset(Dataset):
    def __init__(
        self,
        sender_tokens: torch.Tensor,
        receiver_tokens: torch.Tensor,
        targets: torch.Tensor,
        n_locations: int,
    ):
        if sender_tokens.ndim != 2:
            raise ValueError("sender_tokens must be rank-2")
        if receiver_tokens.ndim != 2:
            raise ValueError("receiver_tokens must be rank-2")
        if targets.ndim != 1:
            raise ValueError("targets must be rank-1")
        if len(sender_tokens) != len(receiver_tokens) or len(sender_tokens) != len(targets):
            raise ValueError("sender, receiver, and targets must have identical batch size")

        self.sender_tokens = sender_tokens.long()
        self.receiver_tokens = receiver_tokens.long()
        self.targets = targets.long()

        self.n_locations = n_locations
        self.sender_vocab_size = 3 * n_locations
        self.receiver_vocab_size = n_locations

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sender_tokens[idx], self.receiver_tokens[idx], self.targets[idx]


def visibility_signature(saw_flags: List[bool]) -> int:
    signature = 0
    for saw_move in saw_flags:
        signature = (signature << 1) | int(saw_move)
    return signature


def generate_episode(
    n_locations: int,
    n_moves: int,
    protagonist_sees_prob: float,
    rng: Random,
) -> Episode:
    if n_locations < 2:
        raise ValueError("n_locations must be >= 2")
    if n_moves < 1:
        raise ValueError("n_moves must be >= 1")
    if not 0.0 <= protagonist_sees_prob <= 1.0:
        raise ValueError("protagonist_sees_prob must be in [0, 1]")

    initial_location = rng.randrange(n_locations)
    true_location = initial_location
    belief_location = initial_location

    sender_tokens = [initial_location + 2 * n_locations]
    receiver_tokens = [initial_location]
    saw_flags: List[bool] = []

    for _ in range(n_moves):
        next_location = rng.randrange(n_locations - 1)
        if next_location >= true_location:
            next_location += 1
        true_location = next_location

        saw_move = rng.random() < protagonist_sees_prob
        saw_flags.append(saw_move)

        if saw_move:
            belief_location = true_location

        receiver_tokens.append(true_location)
        sender_tokens.append(true_location + (n_locations if saw_move else 0))

    return Episode(
        initial_location=initial_location,
        true_location=true_location,
        belief_location=belief_location,
        sender_tokens=sender_tokens,
        receiver_tokens=receiver_tokens,
        saw_flags=saw_flags,
    )


def _feasible_visibility_signatures(n_moves: int, protagonist_sees_prob: float) -> List[int]:
    n_signatures = 1 << n_moves
    if protagonist_sees_prob <= 0.0:
        return [0]
    if protagonist_sees_prob >= 1.0:
        return [n_signatures - 1]
    return list(range(n_signatures))


def select_heldout_visibility_signatures(
    config: CommDataConfig,
    holdout_ratio: float,
    signature_seed: Optional[int] = None,
) -> Tuple[Set[int], Set[int]]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")

    feasible_signatures = _feasible_visibility_signatures(
        n_moves=config.n_moves,
        protagonist_sees_prob=config.protagonist_sees_prob,
    )
    if len(feasible_signatures) < 2:
        raise ValueError(
            "Need at least two feasible visibility signatures for OOD holdout; "
            "use protagonist_sees_prob strictly between 0 and 1"
        )

    holdout_count = int(round(len(feasible_signatures) * holdout_ratio))
    holdout_count = max(1, min(len(feasible_signatures) - 1, holdout_count))

    if signature_seed is None:
        signature_seed = config.seed

    rng = Random(signature_seed + 17_531)
    shuffled_signatures = feasible_signatures[:]
    rng.shuffle(shuffled_signatures)

    heldout_signatures = set(shuffled_signatures[:holdout_count])
    in_distribution_signatures = set(shuffled_signatures[holdout_count:])
    return in_distribution_signatures, heldout_signatures


def _sample_episode_with_allowed_signatures(
    config: CommDataConfig,
    rng: Random,
    allowed_signatures: Set[int],
    max_attempts: int = 10_000,
) -> Episode:
    for _ in range(max_attempts):
        episode = generate_episode(
            n_locations=config.n_locations,
            n_moves=config.n_moves,
            protagonist_sees_prob=config.protagonist_sees_prob,
            rng=rng,
        )
        if visibility_signature(episode.saw_flags) in allowed_signatures:
            return episode

    raise ValueError(
        "Unable to sample episode matching requested visibility signatures; "
        "adjust holdout_ratio, n_moves, or protagonist_sees_prob"
    )


def _build_split(
    size: int,
    config: CommDataConfig,
    rng: Random,
    allowed_signatures: Optional[Set[int]] = None,
) -> BeliefCommunicationDataset:
    sender_tokens = torch.empty((size, config.n_moves + 1), dtype=torch.long)
    receiver_tokens = torch.empty((size, config.n_moves + 1), dtype=torch.long)
    targets = torch.empty(size, dtype=torch.long)

    for i in range(size):
        if allowed_signatures is None:
            episode = generate_episode(
                n_locations=config.n_locations,
                n_moves=config.n_moves,
                protagonist_sees_prob=config.protagonist_sees_prob,
                rng=rng,
            )
        else:
            episode = _sample_episode_with_allowed_signatures(
                config=config,
                rng=rng,
                allowed_signatures=allowed_signatures,
            )
        sender_tokens[i] = torch.tensor(episode.sender_tokens, dtype=torch.long)
        receiver_tokens[i] = torch.tensor(episode.receiver_tokens, dtype=torch.long)
        targets[i] = episode.belief_location

    return BeliefCommunicationDataset(
        sender_tokens=sender_tokens,
        receiver_tokens=receiver_tokens,
        targets=targets,
        n_locations=config.n_locations,
    )


def build_datasets(config: CommDataConfig) -> Tuple[BeliefCommunicationDataset, BeliefCommunicationDataset]:
    train_rng = Random(config.seed)
    val_rng = Random(config.seed + 1)

    train_dataset = _build_split(config.train_size, config, train_rng)
    val_dataset = _build_split(config.val_size, config, val_rng)
    return train_dataset, val_dataset


def build_datasets_with_signature_ood(
    config: CommDataConfig,
    holdout_ratio: float = 0.25,
    ood_val_size: Optional[int] = None,
    signature_seed: Optional[int] = None,
) -> Tuple[
    BeliefCommunicationDataset,
    BeliefCommunicationDataset,
    BeliefCommunicationDataset,
    List[int],
]:
    if ood_val_size is None:
        ood_val_size = config.val_size
    if ood_val_size <= 0:
        raise ValueError("ood_val_size must be > 0")

    in_dist_signatures, heldout_signatures = select_heldout_visibility_signatures(
        config=config,
        holdout_ratio=holdout_ratio,
        signature_seed=signature_seed,
    )

    train_rng = Random(config.seed)
    iid_val_rng = Random(config.seed + 1)
    ood_val_rng = Random(config.seed + 2)

    train_dataset = _build_split(
        size=config.train_size,
        config=config,
        rng=train_rng,
        allowed_signatures=in_dist_signatures,
    )
    iid_val_dataset = _build_split(
        size=config.val_size,
        config=config,
        rng=iid_val_rng,
        allowed_signatures=in_dist_signatures,
    )
    ood_val_dataset = _build_split(
        size=ood_val_size,
        config=config,
        rng=ood_val_rng,
        allowed_signatures=heldout_signatures,
    )

    return train_dataset, iid_val_dataset, ood_val_dataset, sorted(heldout_signatures)
