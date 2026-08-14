"""
CLI entry point for a training run.

Usage:
    python scripts/run_training.py --config baseline
    python scripts/run_training.py --config sims_800 --checkpoint-dir checkpoints/sims_800

Resumable across Colab sessions: rerun with the same --checkpoint-dir and
the script picks up exactly where it left off (network weights, optimizer
state, replay buffer, RNG state, iteration count) via
`<checkpoint-dir>/resume_state.pt`. A dropped Colab session costs at most
one in-progress iteration, not the whole run. Use --no-resume to force a
fresh start in the same directory (e.g. after deliberately changing the
config for that dir, since resume state and a changed config can conflict).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import ABLATION_CONFIGS, Config
from src.train import run_training, _resume_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Override the config's default checkpoint dir")
    parser.add_argument("--iterations", type=int, default=None,
                         help="Override num_iterations for a quick smoke test")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing resume_state.pt and start fresh")
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

    if os.path.exists(_resume_path(cfg)) and not args.no_resume:
        print("Existing resume state found -- continuing that run. "
              "Pass --no-resume to discard it and start fresh instead.")

    best_net, history = run_training(cfg, resume=not args.no_resume)

    final_path = os.path.join(cfg.checkpoint_dir, "final.pt")
    torch.save(best_net.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")

    import json
    with open(os.path.join(cfg.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
