"""Sample undisturbed price paths from the simulator (Figure 3 of the paper).

Runs the environment with the do-nothing policy for several seeds and stores
the best bid / best ask / mid price observed at every 1-second wake-up::

    python -m experiments.sample_price_paths --config experiments/configs/paper.yaml \
        --seeds 5 --window 04:00:00 --out results/paper/price_paths.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import env_kwargs_from_config, load_config  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--window", default="04:00:00", help="length of the sampled path (paper: about 14 000 steps)")
    parser.add_argument("--out", default=str(REPO_ROOT / "results/paper/price_paths.parquet"))
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    import gym
    import abides_gym  # noqa: F401

    cfg = load_config(args.config)
    kw = env_kwargs_from_config(cfg)
    kw["execution_window"] = args.window
    kw["first_interval"] = kw.get("first_interval", "00:05:00")
    rows = []
    for i in range(args.seeds):
        seed = args.seed_offset + i
        env = gym.make("markets-execution-v0", **kw)
        env.seed(seed)
        state = env.reset()
        done, step = False, 0
        while not done:
            state, r, done, info = env.step(0)
            rows.append({"seed": seed, "step": step, "current_time": info["current_time"], "best_bid": info["best_bid"],
                         "best_ask": info["best_ask"], "mid_price": info["mid_price"], "spread": info["spread"],
                         "imbalance_5": info["imbalance_5"]})
            step += 1
        env.close()
        print(f"seed {seed}: {step} steps, ask range {min(r_['best_ask'] for r_ in rows if r_['seed']==seed):.0f}-{max(r_['best_ask'] for r_ in rows if r_['seed']==seed):.0f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
