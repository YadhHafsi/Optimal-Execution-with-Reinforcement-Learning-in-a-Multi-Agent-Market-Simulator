"""Train the DQN execution agent of Hafsi & Vittori (2026) with RLlib 2.2.

Reproduces the training setup of Section V-B of the paper:

* Double DQN with dueling head (RLlib defaults ``double_q=True``,
  ``dueling=True``), Q-head hidden layers ``[50, 20]``;
* learning rate decreasing linearly from ``--lr`` to 0 over 90 000 environment
  steps; epsilon-greedy exploration decreasing linearly from 1.0 to 0.02 over
  10 000 steps; discount factor 0.9999;
* running mean/std observation normalisation (``MeanStdFilter``);
* replay buffer, batch size and target-network update frequency left at RLlib's
  defaults (50 000 transitions, 32, 500 steps), see ``DESIGN.md``.

Usage (one process per learning rate, see ``Makefile`` / ``run_paper.sh``)::

    python -m experiments.train_dqn --config experiments/configs/paper.yaml \
        --lr 1e-3 --seed 10 --total-timesteps 100000 --out results/training/lr_1e-3

The script writes ``progress.csv`` (one row per RLlib iteration, including
``episode_reward_mean`` used for Figure 4), ``episodes.csv`` (one row per
training episode) and a final checkpoint in ``<out>/checkpoint``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import env_kwargs_from_config, load_config  # noqa: E402


def build_dqn_config(env_config: Dict[str, Any], lr: float, seed: int, lr_decay_steps: int = 90_000,
                     epsilon_timesteps: int = 10_000, final_epsilon: float = 0.02, gamma: float = 0.9999,
                     hiddens: Optional[List[int]] = None, num_rollout_workers: int = 0):
    """RLlib ``DQNConfig`` implementing the paper's hyper-parameters."""
    from ray.rllib.algorithms.dqn import DQNConfig

    import abides_gym  # noqa: F401  registers "markets-execution-v0" with RLlib

    hiddens = [50, 20] if hiddens is None else list(hiddens)
    config = (
        DQNConfig()
        .environment("markets-execution-v0", env_config=copy.deepcopy(env_config), disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=num_rollout_workers, observation_filter="MeanStdFilter")
        .resources(num_gpus=0)
        .debugging(seed=seed, log_level="ERROR")
        .training(
            lr=lr,
            lr_schedule=[[0, lr], [lr_decay_steps, 0.0]],
            gamma=gamma,
            hiddens=hiddens,
            double_q=True,
            dueling=True,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            }
        )
    )
    return config


def apply_dqn_overrides(config, overrides: Dict[str, Any]):
    """Apply ``{key: value}`` overrides to an RLlib ``AlgorithmConfig``.

    Keys are RLlib configuration keys (``n_step``, ``train_batch_size``,
    ``target_network_update_freq``, ``num_steps_sampled_before_learning_starts``,
    ``grad_clip``, ...). Dotted keys address entries of dictionary-valued
    settings and are merged into them, e.g. ``replay_buffer_config.capacity``
    or ``model.fcnet_hiddens``.
    """
    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key:
            top, _, sub = key.partition(".")
            current = getattr(config, top)
            if not isinstance(current, dict):
                raise ValueError(f"{top} is not a dictionary setting; cannot set {key}")
            merged = dict(flat.get(top, current))
            merged[sub] = value
            flat[top] = merged
        else:
            flat[key] = value
    config.update_from_dict(flat)
    return config


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    # defaults of the training settings come from the config's ``training`` section
    parser.add_argument("--lr", type=float, default=None, help="initial learning rate (decays linearly to 0)")
    parser.add_argument("--seed", type=int, default=None, help="RLlib seed (environment seed of the paper)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--epsilon-timesteps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--hiddens", type=int, nargs="+", default=None)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="save a checkpoint every N iterations")
    parser.add_argument("--env-override", action="append", default=[], metavar="KEY=VALUE",
                        help="override an environment parameter of the config (e.g. beta_not_enough=100); repeatable")
    parser.add_argument("--dqn-override", action="append", default=[], metavar="KEY=VALUE",
                        help="override an RLlib DQN setting (e.g. n_step=20, replay_buffer_config.capacity=500000, "
                             "model.fcnet_hiddens=[128,128]); repeatable")
    args = parser.parse_args(argv)

    import ray

    cfg = load_config(args.config)
    tr = cfg.get("training", {})
    if args.lr is None:
        args.lr = float(tr.get("selected_learning_rate", 1e-3))
    if args.seed is None:
        args.seed = int(tr.get("seed", 10))
    if args.total_timesteps is None:
        args.total_timesteps = int(tr.get("total_timesteps", 100_000))
    if args.lr_decay_steps is None:
        args.lr_decay_steps = int(tr.get("lr_decay_steps", 90_000))
    if args.epsilon_timesteps is None:
        args.epsilon_timesteps = int(tr.get("epsilon_timesteps", 10_000))
    if args.gamma is None:
        args.gamma = float(tr.get("gamma", 0.9999))
    if args.hiddens is None:
        args.hiddens = [int(h) for h in tr.get("hiddens", [50, 20])]
    env_config = env_kwargs_from_config(cfg)
    for item in args.env_override:
        key, _, raw = item.partition("=")
        env_config[key.strip()] = yaml.safe_load(raw)
    dqn_overrides: Dict[str, Any] = {}
    for item in args.dqn_override:
        key, _, raw = item.partition("=")
        dqn_overrides[key.strip()] = yaml.safe_load(raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_config.json", "w") as fh:
        json.dump(
            {"env_config": env_config, "lr": args.lr, "seed": args.seed, "total_timesteps": args.total_timesteps,
             "lr_decay_steps": args.lr_decay_steps, "epsilon_timesteps": args.epsilon_timesteps,
             "gamma": args.gamma, "hiddens": args.hiddens, "env_override": args.env_override,
             "dqn_override": dqn_overrides},
            fh, indent=2, default=str,
        )

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1, log_to_driver=False,
             object_store_memory=200_000_000)
    config = build_dqn_config(env_config, lr=args.lr, seed=args.seed, lr_decay_steps=args.lr_decay_steps,
                              epsilon_timesteps=args.epsilon_timesteps, gamma=args.gamma, hiddens=args.hiddens)
    if dqn_overrides:
        apply_dqn_overrides(config, dqn_overrides)
    algo = config.build()

    progress_path = out / "progress.csv"
    episodes_path = out / "episodes.csv"
    prog_fields = ["iteration", "timesteps_total", "episodes_total", "episode_reward_mean", "episode_reward_min",
                   "episode_reward_max", "episode_len_mean", "cur_lr", "cur_epsilon", "time_total_s"]
    with open(progress_path, "w", newline="") as fh:
        csv.writer(fh).writerow(prog_fields)
    with open(episodes_path, "w", newline="") as fh:
        csv.writer(fh).writerow(["episode_index", "timesteps_total_at_end", "episode_reward", "episode_len"])

    n_episodes_logged = 0
    t0 = time.time()
    iteration = 0
    last_ckpt = None
    while True:
        result = algo.train()
        iteration += 1
        ts = result["timesteps_total"]
        info = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
        cur_lr = info.get("cur_lr", np.nan)
        try:
            cur_eps = algo.get_policy().exploration.get_state()["cur_epsilon"]
        except Exception:  # pragma: no cover - defensive
            cur_eps = np.nan
        row = [iteration, ts, result.get("episodes_total"), result.get("episode_reward_mean"),
               result.get("episode_reward_min"), result.get("episode_reward_max"), result.get("episode_len_mean"),
               cur_lr, cur_eps, round(time.time() - t0, 1)]
        with open(progress_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        # per-episode rewards of the episodes completed during this iteration
        hist = result.get("sampler_results", result).get("hist_stats", {})
        rewards = hist.get("episode_reward", [])
        lengths = hist.get("episode_lengths", [])
        new = result.get("episodes_this_iter", 0)
        if new and rewards:
            with open(episodes_path, "a", newline="") as fh:
                w = csv.writer(fh)
                for r, l in zip(rewards[-new:], lengths[-new:]):
                    n_episodes_logged += 1
                    w.writerow([n_episodes_logged, ts, r, l])
        print(f"[lr={args.lr:g} seed={args.seed}] iter {iteration:4d} | steps {ts:7d} | episodes {result.get('episodes_total')} "
              f"| reward mean {result.get('episode_reward_mean')!s:>10} | lr {cur_lr:.2e} | eps {cur_eps:.3f} | {time.time()-t0:7.0f}s",
              flush=True)
        if iteration % args.checkpoint_every == 0:
            last_ckpt = algo.save(str(out / "checkpoints"))
        if ts >= args.total_timesteps:
            break

    final = algo.save(str(out / "checkpoint"))
    with open(out / "final_checkpoint.txt", "w") as fh:
        fh.write(str(final) + "\n")
    print(f"done: {ts} steps in {time.time()-t0:.0f}s, checkpoint {final}", flush=True)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
