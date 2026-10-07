"""Check that the simulated market can absorb the execution agent's orders.

For every evaluation seed of a configuration (and every scenario) the script
runs a do-nothing episode and a TWAP episode and reports, per seed:

* whether each adaptive market maker is quoting at the end of the episode
  (it has observed a mid price and has orders resting on the book);
* the fraction of wake-ups at which the published book had no ask (or no bid);
* the executed quantity of TWAP and the shares of its market orders that the
  exchange discarded for lack of resting liquidity (``unfilled_quantity``).

A sound market has every market maker quoting, an empty side of the book at
(close to) no wake-up and zero discarded shares.  The script exits with status
1 if any share was discarded or any market maker is silent, so it can guard a
pipeline.

Usage::

    python -m experiments.check_market --config experiments/configs/paper.yaml
    python -m experiments.check_market --config experiments/configs/paper.yaml \\
        --background-override mm_wake_up_freq=1S --num-seeds 50 --out results/check_market_1S.csv
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import env_kwargs_from_config, load_config, policy_env_params  # noqa: E402
from experiments.policies import DoNothingPolicy, TWAPPolicy  # noqa: E402


def _run(task: Dict[str, Any]) -> Dict[str, Any]:
    warnings.filterwarnings("ignore")
    from experiments.evaluate import _make_env
    from abides_markets.agents.market_makers.adaptive_market_maker_agent import AdaptiveMarketMakerAgent

    env_kwargs = dict(task["env_kwargs"])
    if task["policy"] != "TWAP":
        env_kwargs["action_mode"] = "lots"  # a do-nothing agent must really send nothing (no corridor)
    env = _make_env(env_kwargs)
    params = policy_env_params(task["env_kwargs"])
    if task["policy"] == "TWAP":
        policy = TWAPPolicy(params["parent_order_size"], params["num_steps"], params["q_min"], params["num_action_levels"])
    else:
        policy = DoNothingPolicy()
    env.seed(task["seed"])
    policy.reset(task["seed"])
    state = env.reset()
    done, steps, ask_empty, bid_empty = False, 0, 0, 0
    while not done:
        state, _, done, info = env.step(policy.get_action(state))
        steps += 1
        ask_empty += int(info["imbalance_5"] >= 0.999)  # bid share of the 5-level volume == 1 -> no ask
        bid_empty += int(info["imbalance_5"] <= 0.001)
    mms = [a for a in env.unwrapped.kernel.agents if isinstance(a, AdaptiveMarketMakerAgent)]
    return {
        "scenario": task["scenario"], "policy": task["policy"], "seed": task["seed"],
        "market_makers": len(mms),
        "mm_with_mid": sum(m.last_mid is not None for m in mms),
        "mm_quoting": sum(len(m.orders) > 0 for m in mms),
        "steps": steps,
        "frac_ask_empty": ask_empty / max(1, steps),
        "frac_bid_empty": bid_empty / max(1, steps),
        "executed": int(info["executed_quantity"]),
        "remaining": int(info["remaining_quantity"]),
        "orders": int(info["num_orders"]),
        "unfilled_orders": int(info["num_unfilled_orders"]),
        "unfilled_shares": int(info["unfilled_quantity"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--policies", nargs="+", default=["DoNothing", "TWAP"])
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--env-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--background-override", action="append", default=[], metavar="KEY=VALUE",
                        help="override a parameter of the background (RMSC-4) section, e.g. mm_wake_up_freq=1S")
    parser.add_argument("--out", default=None, help="CSV with one row per (scenario, policy, seed)")
    args = parser.parse_args(argv)

    import yaml

    cfg = load_config(args.config)
    for item in args.env_override:
        k, _, v = item.partition("=")
        cfg.setdefault("environment", {})[k.strip()] = yaml.safe_load(v)
    for item in args.background_override:
        k, _, v = item.partition("=")
        cfg.setdefault("background", {})[k.strip()] = yaml.safe_load(v)
    ev = cfg["evaluation"]
    scenarios_all = ev.get("scenarios", {"standard": {}})
    scenarios = {k: scenarios_all[k] for k in (args.scenarios or scenarios_all)}
    n = args.num_seeds or int(ev.get("num_seeds", 20))
    off = args.seed_offset if args.seed_offset is not None else int(ev.get("seed_offset", 1000))
    seeds = [off + i for i in range(n)]

    tasks = []
    for sc, bg in scenarios.items():
        env_kwargs = env_kwargs_from_config(copy.deepcopy(cfg), bg)
        for pol in args.policies:
            for seed in seeds:
                tasks.append({"scenario": sc, "policy": pol, "seed": seed, "env_kwargs": env_kwargs})
    print(f"config={cfg['name']} scenarios={list(scenarios)} policies={args.policies} seeds={seeds[0]}..{seeds[-1]} "
          f"background={cfg.get('background', {})}", flush=True)
    with get_context("spawn").Pool(args.processes) as pool:
        rows = pool.map(_run, tasks)
    df = pd.DataFrame(rows).sort_values(["scenario", "policy", "seed"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    summary = df.groupby(["scenario", "policy"]).agg(
        seeds=("seed", "size"),
        mm_all_quoting=("mm_quoting", lambda s: float((s == df["market_makers"].iloc[0]).mean())),
        mm_silent=("mm_with_mid", lambda s: int((df["market_makers"].iloc[0] - s).sum())),
        frac_ask_empty=("frac_ask_empty", "mean"),
        frac_bid_empty=("frac_bid_empty", "mean"),
        E_executed=("executed", "mean"),
        P_exact=("remaining", lambda s: float((s == 0).mean())),
        P_complete=("remaining", lambda s: float((s <= 0).mean())),
        unfilled_orders=("unfilled_orders", "sum"),
        unfilled_shares=("unfilled_shares", "sum"),
    ).reset_index()
    print(summary.to_string(index=False), flush=True)
    bad = int(summary["unfilled_shares"].sum()) > 0 or int(summary["mm_silent"].sum()) > 0
    print("MARKET CHECK " + ("FAILED: silent market makers or discarded market orders" if bad else "PASSED"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
