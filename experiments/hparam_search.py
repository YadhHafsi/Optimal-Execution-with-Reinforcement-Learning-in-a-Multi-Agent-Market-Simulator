"""Hyper-parameter search: train a list of DQN variants, evaluate each against
TWAP at the same execution granularity, rank them.

The search is described by a YAML file (see ``experiments/configs/search/``)::

    name: stage1
    config: experiments/configs/paper.yaml      # base configuration
    eval_seeds: 50                              # out-of-sample seeds (eval_seed_offset + i)
    eval_seed_offset: 1000                      # use a fresh offset to confirm finalists
    reference_episodes: results/paper/standard/episodes.csv   # TWAP of the paper's setting, same seeds
    scenarios: [standard]
    parallel: 8
    runs:
      - name: b100_nstep20
        total_timesteps: 200000
        lr: 1.0e-3
        seed: 10
        gamma: 0.9999
        lr_decay_steps: 180000
        epsilon_timesteps: 20000
        hiddens: [50, 20]
        env_override: {beta_not_enough: 100, beta_too_much: 100}
        dqn_override: {n_step: 20}
      - name: c10_q40_s10_at_150k          # evaluation-only entry: an existing (intermediate) checkpoint
        checkpoint: results/search_stage2/training/c10_q40_s10/checkpoints/checkpoint_000150
        env_override: {timestep_duration: "10s", q_min: 40, num_action_levels: 4, beta_not_enough: 100, beta_too_much: 100}

Stages:

1. every run is trained with ``experiments.train_dqn`` (``--parallel`` at a
   time), under ``<out>/training/<run>/``;
2. every trained policy is evaluated out of sample with ``experiments.evaluate``
   in its own environment (same overrides), under ``<out>/eval/<run>/``;
3. TWAP, Passive and Random are evaluated once per distinct execution setting
   (time step, lot size, action levels, window, terminal penalty), under
   ``<out>/baselines/<setting>/``;
4. ``ranking.csv|md``: per run, completion, executed quantity, mean and sd of
   the implementation shortfall (IS, higher is better), reward, fraction of the
   window used, discarded shares, TWAP's IS in the same setting, the paired
   difference RL - TWAP with its two-sided p-value, and, when
   ``reference_episodes`` (default ``results/paper/standard/episodes.csv``)
   exists, the difference to the reference TWAP of the paper configuration
   (1 s steps, 20-share lots) on the common seeds.
   ``beats_twap`` is true when the policy completes >= 95% of the episodes and
   its paired difference against the same-setting TWAP is positive with
   p < 0.05; ``beats_ref_twap`` likewise against the reference TWAP.

Usage::

    python -m experiments.hparam_search --spec experiments/configs/search/stage1.yaml --out results/search_stage1
    python -m experiments.hparam_search --spec ... --out ... --skip-training   # re-evaluate / re-rank only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.make_figures import df_to_markdown  # noqa: E402

SETTING_KEYS = ["timestep_duration", "action_repeat", "q_min", "num_action_levels", "execution_window", "first_interval",
                "parent_order_size", "beta_not_enough", "beta_too_much", "depth_penalty_alpha", "is_weight", "schedule_penalty",
                "action_mode", "schedule_band", "schedule_corridor", "reward_mode", "state_features", "direction",
                "order_style", "passive_horizon", "passive_offset"]


def _fmt(v: Any) -> str:
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _run(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))


def setting_of(run: Dict[str, Any]) -> Dict[str, Any]:
    ov = run.get("env_override", {}) or {}
    return {k: ov[k] for k in SETTING_KEYS if k in ov}


def setting_name(setting: Dict[str, Any]) -> str:
    if not setting:
        return "default"
    key = "_".join(f"{k}={_fmt(v)}" for k, v in sorted(setting.items()))
    return key if len(key) <= 60 else "setting_" + hashlib.md5(key.encode()).hexdigest()[:10]


def episode_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n": int(len(df)),
        "E(IS)": float(df["implementation_shortfall"].mean()),
        "SD(IS)": float(df["implementation_shortfall"].std(ddof=1)),
        "E(Pen)": float(df["terminal_penalty"].mean()),
        "E(depth)": float(df["depth_penalty"].mean()),
        "E(reward)": float(df["episode_reward"].mean()),
        "E(T)": float(df["time_pct"].mean()),
        "E(executed)": float(df["executed_quantity"].mean()),
        "P(complete)": float((df["remaining_quantity"] <= 0).mean()),
        "P(exact)": float((df["remaining_quantity"] == 0).mean()),
        "P(idle)": float((df["executed_quantity"] == 0).mean()),
        "unfilled": int(df["unfilled_quantity"].sum()) if "unfilled_quantity" in df else -1,
    }


def paired(rl: pd.DataFrame, bench: pd.DataFrame) -> Dict[str, float]:
    from scipy import stats

    a = rl.set_index("seed")["implementation_shortfall"]
    b = bench.set_index("seed")["implementation_shortfall"]
    idx = a.index.intersection(b.index)
    d = a.loc[idx] - b.loc[idx]
    if len(idx) < 3:
        return {"diff": float("nan"), "p": float("nan"), "P(RL better)": float("nan")}
    t = stats.ttest_rel(a.loc[idx], b.loc[idx])
    return {"diff": float(d.mean()), "p": float(t.pvalue), "P(RL better)": float((d > 0).mean())}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parallel", type=int, default=None, help="override the spec's parallelism")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="only rank existing evaluations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    spec = yaml.safe_load(Path(args.spec).read_text())
    base_config = spec.get("config", "experiments/configs/paper.yaml")
    runs: List[Dict[str, Any]] = spec["runs"]
    scenarios: List[str] = spec.get("scenarios", ["standard"])
    eval_seeds = int(spec.get("eval_seeds", 50))
    eval_seed_offset = int(spec.get("eval_seed_offset", 1000))
    reference_episodes = Path(spec.get("reference_episodes", "results/paper/standard/episodes.csv"))
    if not reference_episodes.is_absolute():
        reference_episodes = REPO_ROOT / reference_episodes
    parallel = args.parallel or int(spec.get("parallel", max(1, (os.cpu_count() or 2) - 2)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.yaml").write_text(Path(args.spec).read_text())
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    py = sys.executable

    # ------------------------------------------------------------ stage 1: training
    def train(run: Dict[str, Any]):
        name = run["name"]
        run_dir = out / "training" / name
        if run.get("checkpoint"):
            return name, 0  # evaluation-only entry (an existing checkpoint, e.g. an intermediate one)
        if (run_dir / "final_checkpoint.txt").exists():
            return name, 0
        cmd = [py, "-m", "experiments.train_dqn", "--config", base_config,
               "--lr", f"{float(run.get('lr', 1e-3)):g}", "--seed", str(run.get("seed", 10)),
               "--total-timesteps", str(run.get("total_timesteps", 100_000)),
               "--lr-decay-steps", str(run.get("lr_decay_steps", int(0.9 * run.get("total_timesteps", 100_000)))),
               "--epsilon-timesteps", str(run.get("epsilon_timesteps", int(0.1 * run.get("total_timesteps", 100_000)))),
               "--gamma", f"{float(run.get('gamma', 0.9999)):g}",
               "--checkpoint-every", str(run.get("checkpoint_every", 50)), "--out", str(run_dir)]
        if run.get("hiddens"):
            cmd += ["--hiddens", *[str(h) for h in run["hiddens"]]]
        for k, v in (run.get("env_override") or {}).items():
            cmd += ["--env-override", f"{k}={_fmt(v)}"]
        for k, v in (run.get("dqn_override") or {}).items():
            cmd += ["--dqn-override", f"{k}={_fmt(v)}"]
        if args.dry_run:
            print(" ".join(cmd))
            return name, 0
        return name, _run(cmd, out / "logs" / f"train_{name}.log", env)

    if not args.skip_training:
        t0 = time.time()
        print(f"[stage 1] {len(runs)} training runs, {parallel} in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for name, rc in pool.map(train, runs):
                print(f"[stage 1] {name}: exit {rc} | {time.time() - t0:7.0f}s", flush=True)
    if args.dry_run:
        return

    # ------------------------------------------------------------ stage 2: RL evaluation
    def evaluate_rl(run: Dict[str, Any]):
        name = run["name"]
        run_dir = out / "training" / name
        checkpoint = run.get("checkpoint") or str(run_dir)
        if not run.get("checkpoint") and not (run_dir / "final_checkpoint.txt").exists():
            return name, -1
        if all((out / "eval" / name / sc / "episodes.csv").exists() for sc in scenarios):
            return name, 0
        cmd = [py, "-m", "experiments.evaluate", "--config", base_config, "--policies", "RL", "--checkpoint", checkpoint,
               "--num-seeds", str(eval_seeds), "--seed-offset", str(eval_seed_offset), "--scenarios", *scenarios,
               "--out", str(out / "eval" / name)]
        for k, v in (run.get("env_override") or {}).items():
            cmd += ["--env-override", f"{k}={_fmt(v)}"]
        return name, _run(cmd, out / "logs" / f"eval_{name}.log", env)

    # ------------------------------------------------------------ stage 3: baselines per setting
    settings = {}
    for run in runs:
        s = setting_of(run)
        settings[setting_name(s)] = s

    def evaluate_baselines(item):
        sname, s = item
        bdir = out / "baselines" / sname
        if all((bdir / sc / "episodes.csv").exists() for sc in scenarios):
            return sname, 0
        cmd = [py, "-m", "experiments.evaluate", "--config", base_config, "--policies", "TWAP", "Passive", "Random",
               "--num-seeds", str(eval_seeds), "--seed-offset", str(eval_seed_offset), "--scenarios", *scenarios,
               "--processes", "3", "--out", str(bdir)]
        for k, v in s.items():
            cmd += ["--env-override", f"{k}={_fmt(v)}"]
        return sname, _run(cmd, out / "logs" / f"baselines_{sname}.log", env)

    if not args.skip_eval:
        t0 = time.time()
        print(f"[stage 2] evaluating {len(runs)} policies on {scenarios} ({eval_seeds} seeds)", flush=True)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for name, rc in pool.map(evaluate_rl, runs):
                print(f"[stage 2] {name}: exit {rc} | {time.time() - t0:6.0f}s", flush=True)
        print(f"[stage 3] baselines for {len(settings)} execution setting(s)", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, parallel // 3)) as pool:
            for sname, rc in pool.map(evaluate_baselines, list(settings.items())):
                print(f"[stage 3] {sname}: exit {rc} | {time.time() - t0:6.0f}s", flush=True)

    # ------------------------------------------------------------ stage 4: ranking
    ref_twap = None
    if reference_episodes.exists():
        ref = pd.read_csv(reference_episodes)
        ref_twap = ref[ref["policy"] == "TWAP"]
    rows = []
    for run in runs:
        name = run["name"]
        sname = setting_name(setting_of(run))
        for sc in scenarios:
            p = out / "eval" / name / sc / "episodes.csv"
            if not p.exists():
                continue
            rl = pd.read_csv(p)
            rl = rl[rl["policy"] == "RL"]
            row: Dict[str, Any] = {"run": name, "scenario": sc, "setting": sname, "steps": run.get("total_timesteps", 100_000),
                                   "seed": run.get("seed", 10), "checkpoint": run.get("checkpoint", ""), **episode_stats(rl)}
            bp = out / "baselines" / sname / sc / "episodes.csv"
            if bp.exists():
                b = pd.read_csv(bp)
                for pol in ["TWAP", "Passive", "Random"]:
                    bb = b[b["policy"] == pol]
                    if len(bb):
                        row[f"{pol} E(IS)"] = float(bb["implementation_shortfall"].mean())
                        pr = paired(rl, bb)
                        row[f"diff vs {pol}"] = pr["diff"]
                        row[f"p vs {pol}"] = pr["p"]
                if "TWAP E(IS)" in row:
                    row["P(RL better than TWAP)"] = paired(rl, b[b["policy"] == "TWAP"])["P(RL better)"]
            if ref_twap is not None and sc == "standard":
                common = ref_twap[ref_twap["seed"].isin(rl["seed"])]
                pr = paired(rl, common)
                row["ref TWAP 1s E(IS)"] = float(common["implementation_shortfall"].mean()) if len(common) else float("nan")
                row["ref seeds"] = int(len(common))
                row["diff vs ref TWAP 1s"] = pr["diff"]
                row["p vs ref TWAP 1s"] = pr["p"]
                row["P(RL better than ref TWAP)"] = pr["P(RL better)"]
            complete = row["P(complete)"] >= 0.95
            row["beats_twap"] = bool(complete and row.get("diff vs TWAP", -1) > 0 and row.get("p vs TWAP", 1) < 0.05)
            row["beats_ref_twap"] = bool(complete and row.get("diff vs ref TWAP 1s", -1) > 0 and row.get("p vs ref TWAP 1s", 1) < 0.05)
            rows.append(row)
    if not rows:
        print("no evaluation results")
        return
    ranking = pd.DataFrame(rows)
    ranking["meets_completion"] = ranking["P(complete)"] >= 0.95
    ranking = ranking.sort_values(["scenario", "meets_completion", "diff vs TWAP", "E(IS)"], ascending=[True, False, False, False]).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.groupby("scenario").cumcount() + 1)
    ranking.to_csv(out / "ranking.csv", index=False)
    (out / "ranking.md").write_text(df_to_markdown(ranking.round(4)))
    pd.set_option("display.width", 250)
    cols = ["rank", "run", "scenario", "steps", "P(complete)", "E(executed)", "E(IS)", "SD(IS)", "E(reward)", "E(T)", "unfilled",
            "TWAP E(IS)", "diff vs TWAP", "p vs TWAP", "ref TWAP 1s E(IS)", "diff vs ref TWAP 1s", "p vs ref TWAP 1s", "beats_twap", "beats_ref_twap"]
    print(ranking[[c for c in cols if c in ranking.columns]].round(3).to_string(index=False), flush=True)
    json.dump({"spec": spec, "settings": settings}, open(out / "search.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
