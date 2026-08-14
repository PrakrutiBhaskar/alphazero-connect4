"""
PUCT Monte Carlo Tree Search, as used in AlphaZero / AlphaGo Zero.

This is the part of the reproduction worth spending real attention on --
search quality (not just the network) is what makes AlphaZero strong.
Details that are easy to get subtly wrong and are handled carefully here:
  - PUCT selection formula (balances prior + visit-count exploration)
  - Value is always from the perspective of the player TO MOVE at a node,
    so it must be negated on backup as we walk up the tree (perspective
    flips every ply)
  - Dirichlet noise mixed into the ROOT prior only, not every node, so
    self-play gets exploration without corrupting the network's learned
    priors deeper in the tree
"""

from __future__ import annotations
import math
import numpy as np
import torch

from .env import ConnectFour


class Node:
    __slots__ = ("parent", "prior", "children", "visit_count",
                 "value_sum", "to_play")

    def __init__(self, parent: "Node | None", prior: float, to_play: int):
        self.parent = parent
        self.prior = prior
        self.children: dict[int, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.to_play = to_play

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, network, device="cpu",
                 c_puct: float = 1.5,
                 n_simulations: int = 200,
                 dirichlet_alpha: float = 1.0,
                 dirichlet_epsilon: float = 0.25):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def _evaluate(self, game: ConnectFour):
        """Run the network on the canonical board; return (priors, value)
        both from the perspective of `game`'s current player to move."""
        planes = self.network.board_to_planes(game.canonical_board())
        planes = planes.unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(planes)
        priors = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        mask = game.legal_moves_mask()
        priors = priors * mask
        total = priors.sum()
        priors = priors / total if total > 0 else mask / mask.sum()
        return priors, float(value.item())

    def run(self, game: ConnectFour, add_root_noise: bool = True) -> dict[int, float]:
        """Run n_simulations of MCTS from `game`'s current state.
        Returns {column: visit_fraction}, usable directly as the policy
        training target and for move selection."""
        root = Node(parent=None, prior=1.0, to_play=game.to_play)
        priors, _ = self._evaluate(game)

        if add_root_noise:
            legal = game.legal_moves()
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            for i, col in enumerate(legal):
                priors[col] = (1 - self.dirichlet_epsilon) * priors[col] + \
                              self.dirichlet_epsilon * noise[i]

        self._expand(root, priors, game)

        for _ in range(self.n_simulations):
            node = root
            sim_game = game.clone()
            path = [node]
            done, winner = False, 0

            # SELECT: descend via PUCT until we reach an unexpanded node
            while node.expanded:
                col, node = self._select_child(node)
                _, done, winner = sim_game.step(col)
                path.append(node)
                if done:
                    break

            if done:
                # `winner` is absolute (1/-1/0). Convert to the value from
                # the perspective of the player now to move (sim_game.to_play),
                # who is the LOSER whenever winner != 0 (since the mover who
                # just won flips to_play right after the winning move).
                value = 0.0 if winner == 0 else -1.0
            else:
                priors, value = self._evaluate(sim_game)
                self._expand(node, priors, sim_game)

            self._backup(path, value)

        return {col: child.visit_count / root.visit_count
                for col, child in root.children.items()}

    def _expand(self, node: Node, priors: np.ndarray, game: ConnectFour):
        for col in game.legal_moves():
            node.children[col] = Node(parent=node, prior=float(priors[col]),
                                       to_play=-game.to_play)

    def _select_child(self, node: Node) -> tuple[int, Node]:
        best_score, best_col, best_child = -float("inf"), None, None
        sqrt_total = math.sqrt(max(1, node.visit_count))
        for col, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            # child.value is from the CHILD's (opponent's) POV, so negate
            # it to score from the parent's POV before adding the bonus
            score = -child.value + u
            if score > best_score:
                best_score, best_col, best_child = score, col, child
        return best_col, best_child

    def _backup(self, path: list[Node], value: float):
        """`value` is from the perspective of the player to move at the
        leaf. Walk back up, flipping sign each ply since perspective
        alternates between parent and child."""
        v = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += v
            v = -v
