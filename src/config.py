"""
Central hyperparameter config. Keeping everything here (rather than
scattered across scripts) makes ablations a one-line diff and makes it
easy to log the exact config alongside each training run's results.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # --- Network ---
    channels: int = 64
    num_res_blocks: int = 6

    # --- MCTS ---
    n_simulations: int = 200          # sims/move during self-play
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    games_per_iteration: int = 100
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
    eval_games: int = 40
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
