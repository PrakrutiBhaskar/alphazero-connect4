"""
Main training loop: self-play -> replay buffer -> optimize -> evaluate
-> (maybe) promote new checkpoint as "best" -> repeat.

This mirrors the loop structure in the AlphaZero/AlphaGo Zero papers:
the network that generates self-play data is only replaced by a newly
trained one once it wins a majority of an evaluation match against the
current best, so the self-play data source never regresses.
"""

from __future__ import annotations
import os
import random
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .network import PolicyValueNet
from .self_play import generate_self_play_data
from .evaluate import play_match


class ReplayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, pi, z = self.samples[idx]
        planes = PolicyValueNet.board_to_planes(board)
        return planes, torch.from_numpy(pi), torch.tensor(z, dtype=torch.float32)


def train_one_epoch(net, loader, optimizer, device):
    net.train()
    total_loss, total_policy_loss, total_value_loss, n_batches = 0.0, 0.0, 0.0, 0
    for planes, pi, z in loader:
        planes, pi, z = planes.to(device), pi.to(device), z.to(device)

        logits, value = net(planes)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(-1), z)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "policy_loss": total_policy_loss / n_batches,
        "value_loss": total_value_loss / n_batches,
    }


def run_training(cfg: Config, log_fn=print):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        log_fn(f"CUDA not available, falling back to CPU (cfg requested {cfg.device})")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    best_net = PolicyValueNet(cfg.channels, cfg.num_res_blocks).to(device)
    train_net = PolicyValueNet(cfg.channels, cfg.num_res_blocks).to(device)
    train_net.load_state_dict(best_net.state_dict())

    optimizer = torch.optim.Adam(train_net.parameters(),
                                  lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)

    replay_buffer = collections.deque(maxlen=cfg.replay_buffer_size)
    history = []  # per-iteration metrics, for plotting training curves later

    for iteration in range(1, cfg.num_iterations + 1):
        log_fn(f"\n=== Iteration {iteration}/{cfg.num_iterations} ===")

        # 1. Self-play with the current BEST network
        log_fn(f"Generating {cfg.games_per_iteration} self-play games...")
        new_samples = generate_self_play_data(best_net, cfg, device, cfg.games_per_iteration)
        replay_buffer.extend(new_samples)
        log_fn(f"Replay buffer size: {len(replay_buffer)}")

        # 2. Train on the replay buffer
        dataset = ReplayDataset(list(replay_buffer))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        epoch_metrics = []
        for epoch in range(cfg.epochs_per_iteration):
            metrics = train_one_epoch(train_net, loader, optimizer, device)
            epoch_metrics.append(metrics)
            log_fn(f"  epoch {epoch+1}: loss={metrics['loss']:.4f} "
                   f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})")

        # 3. Evaluate: does train_net beat best_net?
        log_fn(f"Evaluating candidate vs best ({cfg.eval_games} games)...")
        win_rate = play_match(train_net, best_net, cfg, device, n_games=cfg.eval_games)
        log_fn(f"Candidate win rate vs best: {win_rate:.3f}")

        promoted = win_rate >= cfg.eval_win_rate_threshold
        if promoted:
            best_net.load_state_dict(train_net.state_dict())
            log_fn("-> Candidate PROMOTED to best.")
        else:
            train_net.load_state_dict(best_net.state_dict())
            log_fn("-> Candidate rejected; reverted to best.")

        ckpt_path = os.path.join(cfg.checkpoint_dir, f"iter_{iteration:03d}.pt")
        torch.save(best_net.state_dict(), ckpt_path)

        history.append({
            "iteration": iteration,
            "win_rate_vs_prev_best": win_rate,
            "promoted": promoted,
            "replay_buffer_size": len(replay_buffer),
            **epoch_metrics[-1],
        })

    return best_net, history
