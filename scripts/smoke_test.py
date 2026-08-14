"""
Fast sanity check (~seconds on CPU) that the whole pipeline runs
end-to-end before burning Colab GPU time: env rules, network forward
pass, MCTS search, and one self-play game.

Run this FIRST after cloning, before touching Colab.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env import ConnectFour
from src.network import PolicyValueNet
from src.mcts import MCTS
from src.self_play import play_one_game
from src.config import Config


def test_env_basic():
    game = ConnectFour()
    assert len(game.legal_moves()) == 7
    game.step(3)
    game.step(3)
    print("env: basic drop/legal_moves OK")


def test_env_horizontal_win():
    game = ConnectFour()
    # Player 1 drops in cols 0,1,2,3 with player -1 dropping elsewhere between
    moves = [0, 0, 1, 1, 2, 2, 3]
    winner = 0
    done = False
    for m in moves:
        _, done, winner = game.step(m)
    assert done and winner == 1, f"expected player 1 to win, got done={done} winner={winner}"
    print("env: horizontal win detection OK")


def test_network_forward():
    net = PolicyValueNet(channels=16, num_res_blocks=2)  # tiny, just for shape check
    game = ConnectFour()
    planes = net.board_to_planes(game.canonical_board()).unsqueeze(0)
    logits, value = net(planes)
    assert logits.shape == (1, 7)
    assert value.shape == (1, 1)
    print("network: forward pass shapes OK")


def test_mcts_runs():
    net = PolicyValueNet(channels=16, num_res_blocks=2)
    game = ConnectFour()
    mcts = MCTS(net, n_simulations=20)
    pi = mcts.run(game)
    assert abs(sum(pi.values()) - 1.0) < 1e-4
    assert set(pi.keys()) == set(game.legal_moves())
    print("mcts: search runs and returns a valid distribution OK")


def test_one_self_play_game():
    net = PolicyValueNet(channels=16, num_res_blocks=2)
    cfg = Config(n_simulations=15, games_per_iteration=1)
    samples = play_one_game(net, cfg, device="cpu")
    assert len(samples) > 0
    for board, pi, z in samples:
        assert board.shape == (6, 7)
        assert pi.shape == (7,)
        assert z in (-1.0, 0.0, 1.0)
    print(f"self_play: completed one game, {len(samples)} training samples OK")


if __name__ == "__main__":
    test_env_basic()
    test_env_horizontal_win()
    test_network_forward()
    test_mcts_runs()
    test_one_self_play_game()
    print("\nAll smoke tests passed. Safe to move to Colab for real training.")
