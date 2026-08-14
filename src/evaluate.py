"""
Two kinds of evaluation, matching what's proposed in the README:

1. play_match(): net-vs-net arena, used inside the training loop to decide
   whether a newly trained network should be promoted to "best" (mirrors
   the paper's self-play evaluation gate).

2. solver_agreement(): grades an agent's MCTS move choice against a
   perfect Connect Four solver's move oracle on sampled positions. This is
   the "ground truth" benchmark that makes the reproduction's evaluation
   story stronger than self-play Elo alone -- see README for how to wire
   up a solver (e.g. an existing open-source Connect Four solver used only
   as a move oracle, not as code you copy).
"""

from __future__ import annotations
import numpy as np

from .env import ConnectFour
from .mcts import MCTS
from .config import Config


def _play_single_game(net_a, net_b, cfg: Config, device: str, a_plays_first: bool):
    """net_a vs net_b, greedy (temperature=0) move selection, no root
    noise -- this is an evaluation match, not self-play data generation."""
    game = ConnectFour()
    mcts_a = MCTS(net_a, device=device, c_puct=cfg.c_puct, n_simulations=cfg.n_simulations)
    mcts_b = MCTS(net_b, device=device, c_puct=cfg.c_puct, n_simulations=cfg.n_simulations)

    current_is_a = a_plays_first
    done, winner = False, 0
    while not done:
        mcts = mcts_a if current_is_a else mcts_b
        pi = mcts.run(game, add_root_noise=False)
        move = max(pi, key=pi.get)  # greedy: most-visited move
        _, done, winner = game.step(move)
        current_is_a = not current_is_a

    if winner == 0:
        return "draw"
    a_won = (winner == 1) == a_plays_first
    return "a" if a_won else "b"


def play_match(net_a, net_b, cfg: Config, device: str, n_games: int) -> float:
    """Play n_games between net_a and net_b, alternating who goes first
    (removes first-move-advantage bias from the win-rate estimate).
    Returns net_a's win rate, counting draws as half a win."""
    a_wins, b_wins, draws = 0, 0, 0
    for i in range(n_games):
        result = _play_single_game(net_a, net_b, cfg, device, a_plays_first=(i % 2 == 0))
        if result == "a":
            a_wins += 1
        elif result == "b":
            b_wins += 1
        else:
            draws += 1
    return (a_wins + 0.5 * draws) / n_games


def solver_agreement(net, cfg: Config, device: str, solver_move_fn, positions):
    """
    Grade move quality against a perfect-play oracle.

    solver_move_fn: callable(board, to_play) -> best_column (or a set of
        equally-good best columns), from an external Connect Four solver.
        Not implemented here -- plug in a solver's move oracle. This
        function only defines the evaluation protocol.
    positions: list of ConnectFour game states to grade (e.g. sampled
        opening/midgame positions, or self-play game states at various plies).

    Returns the fraction of positions where the agent's top MCTS move
    matches the solver's optimal move (or set of optimal moves).
    """
    mcts = MCTS(net, device=device, c_puct=cfg.c_puct, n_simulations=cfg.n_simulations)
    agree = 0
    for game in positions:
        pi = mcts.run(game, add_root_noise=False)
        agent_move = max(pi, key=pi.get)
        best_moves = solver_move_fn(game.board, game.to_play)
        if isinstance(best_moves, int):
            best_moves = {best_moves}
        if agent_move in best_moves:
            agree += 1
    return agree / len(positions) if positions else float("nan")
