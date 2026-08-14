"""
Central hyperparameter config. Keeping everything here (rather than
scattered across scripts) makes ablations a one-line diff and makes it
easy to log the exact config alongside each training run's results.

Baseline MCTS/self-play/eval sizes were tuned down from the original
draft after measuring real Colab T4 timing: 1 iteration at
(n_simulations=200, games_per_iteration=100, eval_games=40) took ~28 min
on a T4, which would put a 40-iteration run at ~19 hours -- not viable on
free-tier Colab. Current defaults target roughly a 4x reduction in total
MCTS work per iteration (~7 min/iteration, ~4.5 hrs for 40 iterations),
based on cost scaling roughly as (games_per_iteration + eval_games) *
n_simulations. Re-measure after any config change before committing to a
full run -- see scripts/run_training.py --iterations 1 for a quick check.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # --- Network ---
    channels: int = 64
    num_res_blocks: int = 6

    # --- MCTS ---
    n_simulations: int = 100          # sims/move during self-play (tuned down from 200 -- see below)
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    games_per_iteration: int = 50      # tuned down from 100 -- see below
    temperature_moves: int = 15        # sample by visit-count temp for first N plies, then play greedy
    temperature: float = 1.0

    # --- Training ---
    num_iterations: int = 40
    epochs_per_iteration: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 200_000  # in (state, pi, z) samples, not games

    # --- Eval / checkpointing ---
    eval_games: int = 20               # tuned down from 40 -- see below
    eval_win_rate_threshold: float = 0.55  # new net must beat old net by this to become "best"
    checkpoint_dir: str = "checkpoints"

    # --- Misc ---
    seed: int = 0
    device: str = "cuda"  # falls back to cpu automatically in code that uses this

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Named configs for the ablation runs described in the README
ABLATION_CONFIGS = {
    "baseline": Config(),
    "sims_50": Config(n_simulations=50),
    "sims_800": Config(n_simulations=800),
    "no_dirichlet_noise": Config(dirichlet_epsilon=0.0),
    "shallow_net": Config(num_res_blocks=3),
    "wide_net": Config(channels=128),
}
