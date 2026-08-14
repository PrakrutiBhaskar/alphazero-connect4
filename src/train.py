"""
Main training loop: self-play -> replay buffer -> optimize -> evaluate
-> (maybe) promote new checkpoint as "best" -> repeat.

This mirrors the loop structure in the AlphaZero/AlphaGo Zero papers:
the network that generates self-play data is only replaced by a newly
trained one once it wins a majority of an evaluation match against the
current best, so the self-play data source never regresses.

RESUME SUPPORT: a full "resume state" (both networks, optimizer, replay
buffer, RNG states, iteration count, history) is written to
`<checkpoint_dir>/resume_state.pt` after every iteration, separate from
the per-iteration `iter_NNN.pt` files (which hold only the best network's
weights, for lightweight inference/eval use). A Colab disconnect mid-run
costs at most one iteration of self-play/training, not the whole run.
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

RESUME_FILENAME = "resume_state.pt"


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


def _resume_path(cfg: Config) -> str:
    return os.path.join(cfg.checkpoint_dir, RESUME_FILENAME)


def save_resume_state(cfg, best_net, train_net, optimizer, replay_buffer,
                       history, completed_iteration, log_fn=print):
    """Atomic-ish save: write to a temp file then rename, so a crash mid-write
    (e.g. Colab killing the process) can't leave a corrupt resume file that
    silently fails to load next session."""
    state = {
        "completed_iteration": completed_iteration,
        "best_net_state": best_net.state_dict(),
        "train_net_state": train_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "replay_buffer": list(replay_buffer),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cfg": cfg,
    }
    path = _resume_path(cfg)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on same filesystem
    log_fn(f"Saved resume state ({len(replay_buffer)} buffer samples) -> {path}")


def load_resume_state(cfg: Config, device: str, log_fn=print):
    """Returns None if no resume file exists, else a dict with everything
    needed to pick training back up exactly where it left off."""
    path = _resume_path(cfg)
    if not os.path.exists(path):
        return None
    log_fn(f"Found resume state at {path}, loading...")
    state = torch.load(path, map_location=device, weights_only=False)
    return state


def run_training(cfg: Config, log_fn=print, resume: bool = True):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        log_fn(f"CUDA not available, falling back to CPU (cfg requested {cfg.device})")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    best_net = PolicyValueNet(cfg.channels, cfg.num_res_blocks).to(device)
    train_net = PolicyValueNet(cfg.channels, cfg.num_res_blocks).to(device)
    optimizer = torch.optim.Adam(train_net.parameters(),
                                  lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)

    replay_buffer = collections.deque(maxlen=cfg.replay_buffer_size)
    history = []
    start_iteration = 1

    resume_state = load_resume_state(cfg, device, log_fn) if resume else None
    if resume_state is not None:
        best_net.load_state_dict(resume_state["best_net_state"])
        train_net.load_state_dict(resume_state["train_net_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        replay_buffer = collections.deque(resume_state["replay_buffer"],
                                           maxlen=cfg.replay_buffer_size)
        history = resume_state["history"]
        start_iteration = resume_state["completed_iteration"] + 1
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        np.random.set_state(resume_state["numpy_rng_state"])
        random.setstate(resume_state["python_rng_state"])
        log_fn(f"Resuming from iteration {start_iteration}/{cfg.num_iterations} "
               f"(replay buffer: {len(replay_buffer)} samples)")
    else:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        train_net.load_state_dict(best_net.state_dict())
        log_fn("No resume state found, starting fresh.")

    if start_iteration > cfg.num_iterations:
        log_fn(f"Already completed {start_iteration - 1}/{cfg.num_iterations} "
               f"iterations. Nothing to do (bump cfg.num_iterations to continue).")
        return best_net, history

    for iteration in range(start_iteration, cfg.num_iterations + 1):
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

        # Lightweight checkpoint: best network's weights only, for eval/inference
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"iter_{iteration:03d}.pt")
        torch.save(best_net.state_dict(), ckpt_path)

        history.append({
            "iteration": iteration,
            "win_rate_vs_prev_best": win_rate,
            "promoted": promoted,
            "replay_buffer_size": len(replay_buffer),
            **epoch_metrics[-1],
        })

        # Full resume state: everything needed to continue exactly from here
        # if this process gets killed before the next iteration starts.
        save_resume_state(cfg, best_net, train_net, optimizer, replay_buffer,
                           history, completed_iteration=iteration, log_fn=log_fn)

    return best_net, history
