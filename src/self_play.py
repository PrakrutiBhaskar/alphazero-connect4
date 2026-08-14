"""
Self-play game generation.

Each self-play game produces training samples of the form
(canonical_board, pi, z) where:
  - canonical_board: board from the perspective of the player to move
  - pi: the MCTS visit-count distribution over columns (policy target)
  - z: the eventual game outcome from that player's perspective (+1 win,
       -1 loss, 0 draw) -- filled in only once the game finishes, which is
       why we buffer (board, pi, player) tuples for the whole game first.
"""

from __future__ import annotations
import numpy as np

from .env import ConnectFour, COLS
from .mcts import MCTS
from .config import Config


def _sample_move(pi: dict[int, float], temperature: float) -> int:
    cols = list(pi.keys())
    visits = np.array([pi[c] for c in cols], dtype=np.float64)
    if temperature == 0:
        return cols[int(np.argmax(visits))]
    logits = np.log(visits + 1e-10) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(np.random.choice(cols, p=probs))


def play_one_game(network, cfg: Config, device: str):
    """Play a single self-play game to completion. Returns a list of
    (canonical_board, pi_vector, z) training samples."""
    game = ConnectFour()
    mcts = MCTS(network, device=device,
                c_puct=cfg.c_puct,
                n_simulations=cfg.n_simulations,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon)

    history = []  # (canonical_board, pi_vector, player_to_move)
    done, winner = False, 0
    ply = 0

    while not done:
        pi = mcts.run(game, add_root_noise=True)
        pi_vector = np.zeros(COLS, dtype=np.float32)
        for col, frac in pi.items():
            pi_vector[col] = frac

        history.append((game.canonical_board(), pi_vector, game.to_play))

        temperature = cfg.temperature if ply < cfg.temperature_moves else 0.0
        move = _sample_move(pi, temperature)
        _, done, winner = game.step(move)
        ply += 1

    samples = []
    for board, pi_vector, player in history:
        z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        samples.append((board, pi_vector, z))
    return samples


def generate_self_play_data(network, cfg: Config, device: str, n_games: int):
    """Run n_games sequential self-play games and pool their samples.

    NOTE: this is intentionally simple/sequential for clarity and to keep
    the reproduction debuggable. The straightforward speedup for a T4 is
    batching MCTS leaf evaluations across multiple simultaneous games
    (see README "Performance notes") -- left as a documented extension
    rather than built in, since correctness-first matters more early on.
    """
    all_samples = []
    for i in range(n_games):
        samples = play_one_game(network, cfg, device)
        all_samples.extend(samples)
    return all_samples
