"""Run a grid of DQN trainings in parallel, evaluate every policy, rank them.

Stage 1 launches one ``experiments.train_dqn`` process per (terminal penalty,
learning rate, seed) combination, ``--parallel`` at a time.  Stage 2 evaluates
every trained policy out of sample on the standard scenario (``--eval-seeds``
seeds) in the environment it was trained in (same terminal penalty), all
policies in parallel.  Stage 3 ranks the policies and evaluates the ``--top``
best on every scenario of the configuration.

Ranking rule (see ``rank_policies``): a policy must complete the parent order
in at least ``--min-completion`` of the evaluation episodes; among those, the
highest mean episode implementation shortfall wins, ties broken by the lower
standard deviation.  Policies that do not meet the completion requirement are
ranked after the others by completion rate.

Outputs under ``<out>/``::

    training/<run>/...                 train_dqn.py outputs (progress, episodes, checkpoint)
    eval/<run>/standard/episodes.csv   stage-2 evaluation of every policy
    eval/<run>/<scenario>/...          stage-3 evaluation of the top policies
    ranking.csv, ranking.md            all policies with their statistics and rank
    campaign.json                      the grid and the arguments used

Example (the campaign used for the released policy)::

    python -m experiments.train_campaign --config experiments/configs/paper.yaml \
        --betas 5 20 50 100 --lrs 1e-3 1e-4 --seeds 10 20 30 \
        --total-timesteps 1000000 --lr-decay-steps 900000 --epsilon-timesteps 100000 \
        --parallel 24 --out results/campaign_1M
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.make_figures import df_to_markdown  # noqa: E402


def run_name(beta: float, lr: float, seed: int) -> str:
    return f"beta_{beta:g}_lr_{lr:g}_seed_{seed}"


def _run(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))


def rank_policies(stats: pd.DataFrame, min_completion: float) -> pd.DataFrame:
    """Rank trained policies: completion first, then mean IS, then lower IS dispersion."""
    stats = stats.copy()
    stats["meets_completion"] = stats["P(complete)"] >= min_completion
    stats = stats.sort_values(
        ["meets_completion", "E(IS total)", "SD(IS total)", "P(complete)"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    stats.insert(0, "rank", range(1, len(stats) + 1))
    return stats


def episode_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n": int(len(df)),
        "E(IS total)": float(df["implementation_shortfall"].mean()),
        "SD(IS total)": float(df["implementation_shortfall"].std(ddof=1)),
        "E(IS step mean)": float(df["step_is_mean"].mean()),
        "E(Pen)": float(df["terminal_penalty"].mean()),
        "E(depth)": float(df["depth_penalty"].mean()),
        "E(reward)": float(df["episode_reward"].mean()),
        "E(T)": float(df["time_pct"].mean()),
        "E(executed)": float(df["executed_quantity"].mean()),
        "P(complete)": float((df["remaining_quantity"] <= 0).mean()),
        "P(idle)": float((df["executed_quantity"] == 0).mean()),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    parser.add_argument("--betas", type=float, nargs="+", default=[5, 20, 50, 100], help="terminal penalties (both sides)")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr-decay-steps", type=int, default=900_000)
    parser.add_argument("--epsilon-timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--eval-seeds", type=int, default=50)
    parser.add_argument("--min-completion", type=float, default=0.95)
    parser.add_argument("--top", type=int, default=3, help="number of policies evaluated on every scenario")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-training", action="store_true", help="only run the evaluation stages on existing runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [(b, lr, s) for b, lr, s in itertools.product(args.betas, args.lrs, args.seeds)]
    with open(out / "campaign.json", "w") as fh:
        json.dump({"args": vars(args), "grid": [run_name(*g) for g in grid]}, fh, indent=2)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    py = sys.executable

    # ------------------------------------------------------------ stage 1
    def train(g):
        beta, lr, seed = g
        name = run_name(beta, lr, seed)
        run_dir = out / "training" / name
        if (run_dir / "final_checkpoint.txt").exists():
            return name, 0
        cmd = [py, "-m", "experiments.train_dqn", "--config", args.config, "--lr", f"{lr:g}", "--seed", str(seed),
               "--total-timesteps", str(args.total_timesteps), "--lr-decay-steps", str(args.lr_decay_steps),
               "--epsilon-timesteps", str(args.epsilon_timesteps), "--checkpoint-every", str(args.checkpoint_every),
               "--env-override", f"beta_not_enough={beta:g}", "--env-override", f"beta_too_much={beta:g}",
               "--out", str(run_dir)]
        if args.dry_run:
            print(" ".join(cmd))
            return name, 0
        return name, _run(cmd, out / "logs" / f"train_{name}.log", env)

    if not args.skip_training:
        t0 = time.time()
        print(f"[stage 1] {len(grid)} training runs, {args.parallel} in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for name, rc in pool.map(train, grid):
                print(f"[stage 1] {name}: exit {rc} | {time.time() - t0:7.0f}s", flush=True)
    if args.dry_run:
        return

    # ------------------------------------------------------------ stage 2
    def evaluate(g, scenarios: Optional[List[str]]):
        beta, lr, seed = g
        name = run_name(beta, lr, seed)
        run_dir = out / "training" / name
        if not (run_dir / "final_checkpoint.txt").exists():
            return name, -1
        cmd = [py, "-m", "experiments.evaluate", "--config", args.config, "--policies", "RL", "--checkpoint", str(run_dir),
               "--num-seeds", str(args.eval_seeds), "--env-override", f"beta_not_enough={beta:g}",
               "--env-override", f"beta_too_much={beta:g}", "--out", str(out / "eval" / name)]
        if scenarios:
            cmd += ["--scenarios", *scenarios]
        return name, _run(cmd, out / "logs" / f"eval_{name}.log", env)

    t0 = time.time()
    print(f"[stage 2] evaluating {len(grid)} policies on the standard scenario ({args.eval_seeds} seeds)", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, rc in pool.map(lambda g: evaluate(g, ["standard"]), grid):
            print(f"[stage 2] {name}: exit {rc} | {time.time() - t0:6.0f}s", flush=True)

    rows = []
    for beta, lr, seed in grid:
        name = run_name(beta, lr, seed)
        p = out / "eval" / name / "standard" / "episodes.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["policy"] == "RL"]
        rows.append({"run": name, "beta": beta, "lr": lr, "seed": seed, **episode_stats(df)})
    if not rows:
        print("no evaluation results")
        return
    ranking = rank_policies(pd.DataFrame(rows), args.min_completion)
    ranking.to_csv(out / "ranking.csv", index=False)
    (out / "ranking.md").write_text(df_to_markdown(ranking))
    print(ranking.to_string(index=False), flush=True)

    # ------------------------------------------------------------ stage 3
    top = ranking.head(args.top)
    t0 = time.time()
    print(f"[stage 3] evaluating the top {len(top)} policies on every scenario", flush=True)
    top_grid = [(r["beta"], r["lr"], int(r["seed"])) for _, r in top.iterrows()]
    with ThreadPoolExecutor(max_workers=min(args.parallel, len(top_grid))) as pool:
        for name, rc in pool.map(lambda g: evaluate(g, None), top_grid):
            print(f"[stage 3] {name}: exit {rc} | {time.time() - t0:6.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
