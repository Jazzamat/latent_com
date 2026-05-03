import argparse
from dataclasses import replace
from statistics import mean, pstdev
from typing import List, Sequence

from train_comm import CommTrainConfig, run_experiment


SWEEP_METRIC_KEYS = (
    "iid_with_message_acc",
    "iid_zero_message_acc",
    "iid_shuffled_message_acc",
    "iid_comm_gain",
    "ood_with_message_acc",
    "ood_zero_message_acc",
    "ood_shuffled_message_acc",
    "ood_comm_gain",
)


def aggregate_float_metrics(run_results: List[dict], metric_keys: Sequence[str]) -> dict:
    if not run_results:
        raise ValueError("run_results must not be empty")

    aggregate = {"n_runs": len(run_results)}
    for key in metric_keys:
        values = [float(result[key]) for result in run_results]
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
    return aggregate


def run_seed_sweep(base_config: CommTrainConfig, seeds: Sequence[int]) -> dict:
    if not seeds:
        raise ValueError("seeds must not be empty")

    holdout_seed = base_config.ood_signature_seed
    if holdout_seed is None:
        holdout_seed = base_config.seed

    per_seed_results = []
    for seed in seeds:
        config = replace(
            base_config,
            seed=int(seed),
            ood_signature_seed=holdout_seed,
        )
        results = run_experiment(config)
        per_seed_results.append(results)
        print(
            f"seed={seed} "
            f"iid_gain={results['iid_comm_gain']:.4f} "
            f"ood_gain={results['ood_comm_gain']:.4f} "
            f"heldout={results['heldout_visibility_signatures']} "
            f"holdout_seed={holdout_seed}"
        )

    aggregate = aggregate_float_metrics(per_seed_results, SWEEP_METRIC_KEYS)
    return {
        "seeds": [int(seed) for seed in seeds],
        "ood_signature_seed": holdout_seed,
        "runs": per_seed_results,
        "aggregate": aggregate,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed communication robustness sweep")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
        help="Seed list for training runs (default: 42..51)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs",
    )
    parser.add_argument(
        "--ood-signature-seed",
        type=int,
        default=None,
        help="Seed used for selecting held-out visibility signatures; defaults to base seed",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    base_config = CommTrainConfig()
    if args.epochs is not None:
        base_config = replace(base_config, epochs=int(args.epochs))
    if args.ood_signature_seed is not None:
        base_config = replace(base_config, ood_signature_seed=int(args.ood_signature_seed))

    seeds = [int(seed) for seed in args.seeds]
    sweep_results = run_seed_sweep(base_config=base_config, seeds=seeds)

    print("Aggregate metrics:")
    print(f"- seeds: {sweep_results['seeds']}")
    print(f"- ood_signature_seed: {sweep_results['ood_signature_seed']}")
    for key in sorted(sweep_results["aggregate"].keys()):
        value = sweep_results["aggregate"][key]
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
