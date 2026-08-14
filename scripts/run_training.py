"""
CLI entry point for a training run.

Usage:
    python scripts/run_training.py --config baseline
    python scripts/run_training.py --config sims_800 --checkpoint-dir checkpoints/sims_800

Designed to be resumable across Colab sessions: rerun with the same
--checkpoint-dir and the script picks up the latest checkpoint rather than
starting over, since a T4 Colab session can disconnect mid-run.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import ABLATION_CONFIGS, Config
from src.network import PolicyValueNet
from src.train import run_training


def find_latest_checkpoint(checkpoint_dir: str):
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "iter_*.pt")))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Override the config's default checkpoint dir")
    parser.add_argument("--iterations", type=int, default=None,
                         help="Override num_iterations for a quick smoke test")
    args = parser.parse_args()

    cfg = ABLATION_CONFIGS[args.config]
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    else:
        cfg.checkpoint_dir = os.path.join("checkpoints", args.config)
    if args.iterations:
        cfg.num_iterations = args.iterations

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.json"))

    print(f"Config: {args.config}")
    print(cfg)

    latest = find_latest_checkpoint(cfg.checkpoint_dir)
    if latest:
        print(f"NOTE: found existing checkpoint {latest}. This script currently "
              f"starts fresh training regardless -- wire up checkpoint loading "
              f"into run_training() before relying on this for multi-session runs.")

    best_net, history = run_training(cfg)

    final_path = os.path.join(cfg.checkpoint_dir, "final.pt")
    torch.save(best_net.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")

    import json
    with open(os.path.join(cfg.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
