"""Out-of-sample evaluation of execution policies (Figures 5-8, Tables I-III).

For every scenario (background-agent configuration) and every policy, the
script runs one episode per evaluation seed and stores

* ``results/<config>/<scenario>/episodes.csv``: one row per episode with the
  normalised implementation shortfall, depth penalty, terminal penalty, episode
  reward, executed quantity, completion time, number of steps, ...;
* ``results/<config>/<scenario>/steps.parquet``: one row per environment step
  (best bid/ask, spread, imbalance, holdings, action, fills, reward and, for the
  RL policy, the Q-values of every action).

Hand-crafted baselines run in a process pool; the RL policy (an RLlib
checkpoint) runs in the main process.  Seeds are ``seed_offset + i`` so that
evaluation windows are disjoint from the training seed.

Examples::

    # baselines only, all scenarios of the paper configuration
    python -m experiments.evaluate --config experiments/configs/paper.yaml --policies TWAP Passive Random

    # RL policy from a checkpoint, standard scenario only
    python -m experiments.evaluate --config experiments/configs/paper.yaml --policies RL \
        --checkpoint results/paper/training/lr_1e-3_seed_10 --scenarios standard

``--checkpoint`` accepts the training output directory, its ``checkpoint``
sub-directory or the RLlib ``checkpoint_NNNNNN`` directory itself.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import env_kwargs_from_config, load_config, policy_env_params  # noqa: E402

STEP_COLUMNS = [
    "scenario", "policy", "seed", "step", "time_pct", "current_time", "best_bid", "best_ask", "mid_price",
    "spread", "imbalance_5", "holdings_pct", "action", "step_executed_quantity", "step_avg_fill_price",
    "step_implementation_shortfall", "step_depth_penalty", "reward",
]


def _make_env(env_kwargs: Dict[str, Any]):
    warnings.filterwarnings("ignore")
    import gym
    import abides_gym  # noqa: F401

    return gym.make("markets-execution-v0", **env_kwargs)


def run_episode(env, policy, seed: int, scenario: str, record_q: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one full episode and return (episode summary, per-step records)."""
    env.seed(seed)
    policy.reset(seed)
    state = env.reset()
    done = False
    steps: List[Dict[str, Any]] = []
    t0 = time.time()
    entry_price = None
    names = list(getattr(env.unwrapped, "state_feature_names", []))
    idx = {n: i for i, n in enumerate(names)}
    prev_info: Optional[Dict[str, Any]] = None

    def obs(name: str, fallback: Optional[str] = None) -> float:
        # pre-action value: from the state when the feature is in it, else from the previous step's info
        if name in idx:
            return float(state[idx[name]])
        key = fallback or name
        return float(prev_info[key]) if prev_info is not None and key in prev_info else float("nan")

    while not done:
        action = policy.get_action(state)
        q = policy.q_values(state) if record_q else None
        next_state, reward, done, info = env.step(action)
        rec = {
            "scenario": scenario, "policy": policy.name, "seed": seed, "step": info["num_steps"],
            "time_pct": info["time_pct"], "current_time": info["current_time"], "best_bid": info["best_bid"],
            "best_ask": info["best_ask"], "mid_price": info["mid_price"], "spread": info["spread"],
            "imbalance_5": info["imbalance_5"], "holdings_pct": info["holdings_pct"], "action": action,
            "step_executed_quantity": info["step_executed_quantity"], "step_avg_fill_price": info["step_avg_fill_price"],
            "step_implementation_shortfall": info["step_implementation_shortfall"],
            "step_depth_penalty": info["step_depth_penalty"], "reward": reward,
            "step_order_quantity": info["step_order_quantity"],
            "step_unfilled_quantity": info["step_unfilled_quantity"],
        }
        # observation at decision time (state before the step)
        rec["obs_best_bid"] = obs("best_bid")
        rec["obs_best_ask"] = obs("best_ask")
        rec["obs_holdings_pct"] = obs("holdings_pct")
        rec["obs_time_pct"] = obs("time_pct")
        rec["obs_imbalance_5"] = obs("imbalance_5")
        for n in names:
            if n not in ("best_bid", "best_ask", "holdings_pct", "time_pct", "imbalance_5"):
                rec[f"obs_{n}"] = float(state[idx[n]])
        rec["step_execution_cost"] = info.get("step_execution_cost", float("nan"))
        rec["step_inventory_pnl"] = info.get("step_inventory_pnl", float("nan"))
        rec["step_limit_quantity"] = info.get("step_limit_quantity", 0)
        rec["step_passive_fill_quantity"] = info.get("step_passive_fill_quantity", 0)
        if q is not None:
            for k, qv in enumerate(q):
                rec[f"q_{k}"] = float(qv)
        steps.append(rec)
        entry_price = info["entry_price"]
        prev_info = info
        state = next_state
    summary = {
        "scenario": scenario, "policy": policy.name, "seed": seed,
        "implementation_shortfall": info["implementation_shortfall"],
        "depth_penalty": info["depth_penalty"],
        "terminal_penalty": info["terminal_penalty"],
        "episode_reward": info["episode_reward"],
        "executed_quantity": info["executed_quantity"],
        "remaining_quantity": info["remaining_quantity"],
        "time_pct": info["time_pct"],
        "num_steps": len(steps),
        "num_trading_steps": info["num_trading_steps"],
        "num_orders": info["num_orders"],
        "num_unfilled_orders": info["num_unfilled_orders"],
        "unfilled_quantity": info["unfilled_quantity"],
        "schedule_penalty": info.get("schedule_penalty", 0.0),
        "execution_cost": info.get("execution_cost", float("nan")),
        "inventory_pnl": info.get("inventory_pnl", float("nan")),
        "passive_fill_quantity": info.get("passive_fill_quantity", 0),
        "num_limit_orders": info.get("num_limit_orders", 0),
        "entry_price": entry_price,
        "final_mid_price": info["mid_price"],
        "step_is_final": info["step_implementation_shortfall"],
        "step_is_mean": float(np.mean([s["step_implementation_shortfall"] for s in steps])),
        "step_is_mean_trading": float(np.mean([s["step_implementation_shortfall"] for s in steps if s["step_executed_quantity"] > 0])) if info["num_trading_steps"] > 0 else 0.0,
        "avg_fill_price": float(sum(s["step_executed_quantity"] * s["step_avg_fill_price"] for s in steps) / max(1, sum(s["step_executed_quantity"] for s in steps))),
        "mean_spread_at_trade": float(np.mean([s["spread"] for s in steps if s["step_executed_quantity"] > 0])) if info["num_trading_steps"] > 0 else np.nan,
        "mean_imbalance_at_trade": float(np.mean([s["imbalance_5"] for s in steps if s["step_executed_quantity"] > 0])) if info["num_trading_steps"] > 0 else np.nan,
        "wall_time_s": round(time.time() - t0, 2),
    }
    for k, v in info["action_counter"].items():
        summary[k] = v
    return summary, steps


def _worker(task: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process-pool worker: one (scenario, policy, seed) episode of a hand-crafted policy."""
    warnings.filterwarnings("ignore")
    from experiments.policies import make_policy

    env = _make_env(task["env_kwargs"])
    policy = make_policy(task["policy"], policy_env_params(task["env_kwargs"]), seed=task["seed"])
    try:
        return run_episode(env, policy, task["seed"], task["scenario"])
    finally:
        env.close()


def evaluate_baselines(cfg: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]], policies: List[str], seeds: List[int],
                       out_dir: Path, processes: int) -> None:
    tasks = []
    for scenario, overrides in scenarios.items():
        env_kwargs = env_kwargs_from_config(cfg, overrides)
        for pol in policies:
            for seed in seeds:
                tasks.append({"scenario": scenario, "policy": pol, "seed": seed, "env_kwargs": env_kwargs})
    print(f"[baselines] {len(tasks)} episodes on {processes} processes", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    results: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {s: ([], []) for s in scenarios}
    done = 0
    with ctx.Pool(processes=processes, maxtasksperchild=20) as pool:
        for summary, steps in pool.imap_unordered(_worker, tasks, chunksize=1):
            results[summary["scenario"]][0].append(summary)
            results[summary["scenario"]][1].extend(steps)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[baselines] {done}/{len(tasks)} episodes | {el:6.0f}s elapsed | ETA {el/done*(len(tasks)-done):6.0f}s", flush=True)
    for scenario, (summaries, steps) in results.items():
        _append_results(out_dir / scenario, summaries, steps)


def evaluate_rl(cfg: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]], seeds: List[int], out_dir: Path,
                checkpoint: str) -> None:
    from experiments.policies import RLlibDQNPolicy

    std_env_kwargs = env_kwargs_from_config(cfg)  # the policy was trained on the standard scenario
    policy = RLlibDQNPolicy(checkpoint, env_config=std_env_kwargs)
    t0 = time.time()
    n_total = len(scenarios) * len(seeds)
    done = 0
    for scenario, overrides in scenarios.items():
        env = _make_env(env_kwargs_from_config(cfg, overrides))
        summaries, steps = [], []
        for seed in seeds:
            s, st = run_episode(env, policy, seed, scenario, record_q=True)
            summaries.append(s)
            steps.extend(st)
            done += 1
            el = time.time() - t0
            print(f"[RL] {scenario} seed {seed}: IS={s['implementation_shortfall']:+.4f} exec={s['executed_quantity']} "
                  f"time_pct={s['time_pct']:.3f} | {done}/{n_total} | {el:6.0f}s | ETA {el/done*(n_total-done):6.0f}s", flush=True)
        env.close()
        _append_results(out_dir / scenario, summaries, steps)


def _append_results(scenario_dir: Path, summaries: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    ep_path = scenario_dir / "episodes.csv"
    st_path = scenario_dir / "steps.parquet"
    new_ep = pd.DataFrame(summaries)
    new_st = pd.DataFrame(steps)
    if ep_path.exists():
        old = pd.read_csv(ep_path)
        # replace previous results of the same (policy, seed)
        key = ["policy", "seed"]
        old = old.merge(new_ep[key].drop_duplicates(), on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new_ep = pd.concat([old, new_ep], ignore_index=True)
    new_ep.sort_values(["policy", "seed"]).to_csv(ep_path, index=False)
    if st_path.exists():
        old = pd.read_parquet(st_path)
        key = ["policy", "seed"]
        drop = old.merge(new_st[key].drop_duplicates(), on=key, how="left", indicator=True)["_merge"] == "both"
        old = old[~drop.values]
        new_st = pd.concat([old, new_st], ignore_index=True)
    new_st.to_parquet(st_path, index=False)
    print(f"saved {ep_path} ({len(new_ep)} episodes) and {st_path} ({len(new_st)} steps)", flush=True)


def report_unfilled(out_dir: Path, scenarios: List[str]) -> int:
    """Print, per scenario and policy, the market orders that were not filled in full.

    The exchange discards the part of a market order that meets no resting
    liquidity (``abides_markets/order_book.py``) without any message to the
    agent.  A valid run has zero unfilled shares for every policy; the total is
    returned so that callers can fail on it.
    """
    rows = []
    for sc in scenarios:
        p = out_dir / sc / "episodes.csv"
        if not p.exists():
            continue
        ep = pd.read_csv(p)
        if "unfilled_quantity" not in ep.columns:
            continue
        for pol, d in ep.groupby("policy"):
            rows.append({
                "scenario": sc, "policy": pol, "episodes": len(d), "orders": int(d["num_orders"].sum()),
                "unfilled_orders": int(d["num_unfilled_orders"].sum()), "unfilled_shares": int(d["unfilled_quantity"].sum()),
                "P(complete)": float((d["remaining_quantity"] <= 0).mean()),
                "E(executed)": float(d["executed_quantity"].mean()),
            })
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    print("\nliquidity check (market orders not filled in full):")
    print(df.to_string(index=False))
    total = int(df["unfilled_shares"].sum())
    if total > 0:
        print(f"WARNING: {total} shares of market orders were discarded by the exchange for lack of resting "
              "liquidity; check the market configuration with experiments/check_market.py", flush=True)
    else:
        print("all market orders were filled in full", flush=True)
    return total


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    parser.add_argument("--policies", nargs="+", default=None, help="subset of the policies listed in the config")
    parser.add_argument("--scenarios", nargs="+", default=None, help="subset of the scenarios listed in the config")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="training output directory (or its checkpoint / checkpoint_NNNNNN sub-directory) of the RL policy")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default=None, help="output directory (default results/<config name>)")
    parser.add_argument("--env-override", action="append", default=[], metavar="KEY=VALUE",
                        help="override an environment parameter of the config for every scenario; repeatable")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    import yaml as _yaml

    for item in args.env_override:
        key, _, raw = item.partition("=")
        cfg.setdefault("environment", {})[key.strip()] = _yaml.safe_load(raw)
    ev = cfg["evaluation"]
    policies = args.policies or ev.get("policies", ["TWAP", "Passive", "Random"])
    scenarios_all = ev.get("scenarios", {"standard": {}})
    scenarios = {k: scenarios_all[k] for k in (args.scenarios or scenarios_all)}
    n = args.num_seeds or int(ev.get("num_seeds", 20))
    off = args.seed_offset if args.seed_offset is not None else int(ev.get("seed_offset", 1000))
    seeds = [off + i for i in range(n)]
    out_dir = Path(args.out) if args.out else REPO_ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = [p for p in policies if p.lower() not in ("rl", "dqn")]
    rl = [p for p in policies if p.lower() in ("rl", "dqn")]
    print(f"config={cfg['name']} scenarios={list(scenarios)} policies={policies} seeds={seeds[0]}..{seeds[-1]} -> {out_dir}", flush=True)
    if baselines:
        evaluate_baselines(cfg, scenarios, baselines, seeds, out_dir, args.processes)
    if rl:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required to evaluate the RL policy")
        evaluate_rl(cfg, scenarios, seeds, out_dir, args.checkpoint)
    report_unfilled(out_dir, list(scenarios))


if __name__ == "__main__":
    main()
