# AlphaZero on Connect Four — Reproduction

A from-scratch reproduction of the core AlphaZero algorithm (Silver et al.,
2017/2018, *"Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm"* / *"A general reinforcement learning
algorithm that masters chess, shogi, and Go through self-play"*), scaled
down from Chess/Shogi/Go to Connect Four so it's trainable end-to-end on a
single free-tier Colab T4 GPU.

Connect Four was chosen deliberately over other toy benchmarks: it's a
**solved game** (first player wins with optimal play), which means agent
quality can be graded against a perfect-play oracle rather than relying
on self-play Elo alone — a meaningfully stronger evaluation story than
most from-scratch RL reproductions.

## What this reproduces

- **PUCT-guided MCTS** using a neural network's policy/value outputs to
  guide search (`src/mcts.py`)
- **Self-play data generation** with temperature-based move sampling and
  Dirichlet noise at the root for exploration (`src/self_play.py`)
- **Policy-value ResNet** trained on self-play data via a combined
  cross-entropy (policy) + MSE (value) loss (`src/network.py`, `src/train.py`)
- **Iterative self-play → train → evaluate → promote** loop, where a
  newly trained network only replaces the self-play data generator once
  it beats it in a head-to-head arena (`src/train.py`, `src/evaluate.py`)

Scaled down from the paper: 4-6 residual blocks at 64-128 channels
(vs ~20-40 blocks in the original), and 50-800 MCTS simulations/move
(vs ~800 in the paper for Chess, though Connect Four's much smaller
branching factor makes far fewer sims go a lot further).

## Repo structure

```
src/
  env.py         Connect Four rules/environment
  network.py     Policy-value ResNet
  mcts.py        PUCT Monte Carlo Tree Search
  self_play.py   Self-play game generation
  train.py       Main training loop (self-play/train/eval/promote)
  evaluate.py    Net-vs-net arena + solver-agreement grading
  config.py      All hyperparameters, incl. named ablation configs
scripts/
  smoke_test.py     Fast CPU correctness check (run this first, always)
  run_training.py   CLI entry point for real training runs (Colab)
notebooks/          Colab notebooks for GPU training runs (add here)
tests/               Unit tests
```

## Status

- [x] Environment + rules (tested: horizontal, vertical, diagonal wins)
- [x] Policy-value network
- [x] PUCT MCTS
- [x] Self-play generation
- [x] Training loop with promotion gate + full cross-session resume (tested)
- [x] Net-vs-net evaluation arena
- [x] Config tuned for T4 feasibility (measured: ~28min/iter -> ~7min/iter target)
- [ ] Solver integration for ground-truth move-quality grading
- [ ] Full baseline training run on Colab
- [ ] Batched MCTS across self-play games (perf, optional -- see Performance notes)
- [ ] Ablation runs
- [ ] Training curve / ablation plots + writeup

## Quickstart

```bash
pip install -r requirements.txt

# Always run this first -- fast CPU check that env/network/MCTS/self-play
# are wired correctly before spending Colab GPU time
python scripts/smoke_test.py

# Real training run (intended for Colab with a T4)
python scripts/run_training.py --config baseline
```

## Evaluation plan

1. **Training curves**: policy loss, value loss, and win-rate-vs-previous-best
   per iteration (logged automatically to `checkpoints/<config>/history.json`)
2. **Solver agreement**: sample positions from self-play games at various
   plies, compare the trained agent's top MCTS move against a perfect
   Connect Four solver's optimal move set. Report % agreement, and how it
   improves across training iterations. *(Not yet wired up — plug a
   solver's move oracle into `evaluate.solver_agreement()`.)*
3. **Ablations** (see `config.ABLATION_CONFIGS`):
   - MCTS simulation count: 50 vs 200 (baseline) vs 800 sims/move
   - Root Dirichlet noise on vs off
   - Network depth/width: 3 vs 6 (baseline) vs deeper, 64 vs 128 channels

## Performance notes (for Colab/T4 budget)

**Measured on Colab T4:** the original draft config (200 sims, 100
self-play games, 40 eval games) took **~28 min/iteration**, which would
put a 40-iteration run at ~19 hours -- not viable on free-tier Colab.
Current baseline defaults (100 sims, 50 self-play games, 20 eval games)
target ~7 min/iteration (~4.5 hrs total), based on cost scaling roughly
as `(games_per_iteration + eval_games) * n_simulations`. **Re-measure
with `--iterations 1` after any config change** before committing to a
full run -- don't trust the math blindly.

Self-play and eval currently run MCTS strictly sequentially: each
simulation is a separate GPU forward pass on a single board (batch size
1), which badly underutilizes a T4. This is the actual root cause of the
slowdown, and tuning down game/sim counts (above) is a workaround, not a
fix. The real fix -- **batching MCTS leaf evaluations across several
simultaneous self-play games** so the GPU processes e.g. 32 boards per
forward pass instead of 1 -- is a documented, not-yet-built extension.
Expected to be a 5-10x speedup on GPU. Worth doing once the sequential
version is confirmed to train correctly end-to-end (it has been, see
Status above), since it's a meaningful systems optimization with low
algorithmic risk, and a stronger story than just cutting hyperparameters:
"identified sequential single-sample GPU calls as the self-play
bottleneck and rewrote it to batch across parallel games" vs. "reduced
simulation count." Priority: implement if time remains after the
baseline run + ablations are done and reproducibly checkpointed --
correctness and a complete evaluation story matter more than speed for
the writeup.

## What's deliberately *not* reproduced

- **Board size / game**: Chess/Shogi/Go → Connect Four (compute budget)
- **Network scale**: ~20-40 residual blocks → 4-6 (compute budget)
- **Distributed self-play**: the paper used thousands of TPUs generating
  self-play games in parallel; this reproduction runs self-play
  sequentially on a single GPU

These are called out explicitly (rather than glossed over) because being
precise about what was and wasn't reproduced, and why, is what makes a
scaled-down reproduction read as a considered engineering decision in an
interview rather than an unacknowledged shortcut.
