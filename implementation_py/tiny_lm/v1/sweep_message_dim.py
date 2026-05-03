import argparse
from dataclasses import replace
from typing import Sequence

from sweep_comm import run_seed_sweep
from train_comm import CommTrainConfig


def run_message_dim_sweep(
    base_config: CommTrainConfig,
    message_dims: Sequence[int],
    seeds: Sequence[int],
) -> list[dict]:
    if not message_dims:
        raise ValueError("message_dims must not be empty")

    summaries = []
    for message_dim in message_dims:
        config = replace(base_config, message_dim=int(message_dim))
        sweep = run_seed_sweep(base_config=config, seeds=seeds)
        agg = sweep["aggregate"]

        summary = {
            "message_dim": int(message_dim),
            "n_seeds": agg["n_runs"],
            "ood_signature_seed": sweep["ood_signature_seed"],
            "iid_comm_gain_mean": agg["iid_comm_gain_mean"],
            "iid_comm_gain_std": agg["iid_comm_gain_std"],
            "ood_comm_gain_mean": agg["ood_comm_gain_mean"],
            "ood_comm_gain_std": agg["ood_comm_gain_std"],
        }
        summaries.append(summary)

        print(
            f"message_dim={summary['message_dim']} "
            f"iid_gain={summary['iid_comm_gain_mean']:.4f}+/-{summary['iid_comm_gain_std']:.4f} "
            f"ood_gain={summary['ood_comm_gain_mean']:.4f}+/-{summary['ood_comm_gain_std']:.4f}"
        )

    return summaries


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run message-dimension robustness sweep")
    parser.add_argument(
        "--message-dims",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Message dimensions to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
        help="Seed list for each message-dimension run",
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
        help="Seed used for selecting held-out visibility signatures",
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

    summaries = run_message_dim_sweep(
        base_config=base_config,
        message_dims=[int(dim) for dim in args.message_dims],
        seeds=[int(seed) for seed in args.seeds],
    )

    ranked = sorted(summaries, key=lambda row: row["ood_comm_gain_mean"], reverse=True)
    print("Ranked by ood_comm_gain_mean:")
    for row in ranked:
        print(
            f"- message_dim={row['message_dim']} "
            f"ood_comm_gain_mean={row['ood_comm_gain_mean']:.4f} "
            f"ood_comm_gain_std={row['ood_comm_gain_std']:.4f}"
        )


if __name__ == "__main__":
    main()
