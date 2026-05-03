from random import Random

import pytest

from comm_data import (
    CommDataConfig,
    build_datasets,
    build_datasets_with_signature_ood,
    generate_episode,
)


def _visibility_signatures_from_dataset(dataset, n_locations: int) -> set[int]:
    signatures: set[int] = set()
    for sender_row in dataset.sender_tokens:
        signature = 0
        for token in sender_row[1:]:
            signature = (signature << 1) | int(token.item() >= n_locations)
        signatures.add(signature)
    return signatures


def test_episode_all_seen_tracks_reality() -> None:
    episode = generate_episode(
        n_locations=5,
        n_moves=4,
        protagonist_sees_prob=1.0,
        rng=Random(1),
    )
    assert episode.belief_location == episode.true_location


def test_episode_never_seen_keeps_initial_belief() -> None:
    episode = generate_episode(
        n_locations=5,
        n_moves=4,
        protagonist_sees_prob=0.0,
        rng=Random(2),
    )
    assert episode.belief_location == episode.initial_location


def test_build_datasets_shapes_and_vocab_sizes() -> None:
    config = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=32,
        val_size=16,
        seed=42,
    )
    train_dataset, val_dataset = build_datasets(config)

    assert len(train_dataset) == 32
    assert len(val_dataset) == 16
    assert train_dataset.sender_tokens.shape == (32, 4)
    assert train_dataset.receiver_tokens.shape == (32, 4)
    assert train_dataset.targets.shape == (32,)
    assert train_dataset.sender_vocab_size == 12
    assert train_dataset.receiver_vocab_size == 4


def test_signature_ood_split_is_deterministic_for_same_config() -> None:
    config = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=64,
        val_size=16,
        seed=123,
    )

    _train_a, _iid_a, _ood_a, heldout_a = build_datasets_with_signature_ood(
        config=config,
        holdout_ratio=0.25,
        ood_val_size=16,
    )
    _train_b, _iid_b, _ood_b, heldout_b = build_datasets_with_signature_ood(
        config=config,
        holdout_ratio=0.25,
        ood_val_size=16,
    )

    assert heldout_a == heldout_b


def test_signature_seed_can_freeze_heldout_patterns_across_data_seeds() -> None:
    config_seed_1 = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=32,
        val_size=16,
        seed=100,
    )
    config_seed_2 = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=32,
        val_size=16,
        seed=200,
    )

    _train_a, _iid_a, _ood_a, heldout_a = build_datasets_with_signature_ood(
        config=config_seed_1,
        holdout_ratio=0.25,
        ood_val_size=16,
        signature_seed=999,
    )
    _train_b, _iid_b, _ood_b, heldout_b = build_datasets_with_signature_ood(
        config=config_seed_2,
        holdout_ratio=0.25,
        ood_val_size=16,
        signature_seed=999,
    )

    assert heldout_a == heldout_b


def test_signature_ood_split_has_no_train_or_iid_leakage() -> None:
    config = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=128,
        val_size=32,
        seed=42,
    )
    train_dataset, iid_val_dataset, ood_val_dataset, heldout = build_datasets_with_signature_ood(
        config=config,
        holdout_ratio=0.25,
        ood_val_size=32,
    )

    heldout_set = set(heldout)
    train_signatures = _visibility_signatures_from_dataset(train_dataset, config.n_locations)
    iid_signatures = _visibility_signatures_from_dataset(iid_val_dataset, config.n_locations)
    ood_signatures = _visibility_signatures_from_dataset(ood_val_dataset, config.n_locations)

    assert train_signatures.isdisjoint(heldout_set)
    assert iid_signatures.isdisjoint(heldout_set)
    assert ood_signatures.issubset(heldout_set)
    assert len(ood_signatures) > 0


def test_signature_ood_requires_multiple_feasible_signatures() -> None:
    config = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=1.0,
        train_size=16,
        val_size=8,
        seed=42,
    )

    with pytest.raises(ValueError):
        build_datasets_with_signature_ood(config=config, holdout_ratio=0.25, ood_val_size=8)


def test_signature_ood_validates_holdout_ratio() -> None:
    config = CommDataConfig(
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=16,
        val_size=8,
        seed=42,
    )

    with pytest.raises(ValueError):
        build_datasets_with_signature_ood(config=config, holdout_ratio=0.0, ood_val_size=8)
