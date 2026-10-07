"""Copy a trained policy out of a campaign into ``models/`` with its provenance.

Reads ``<campaign>/ranking.csv`` (written by ``experiments.train_campaign``),
takes the run at ``--rank`` (default 1) and copies into ``--dest``:

* ``checkpoint/checkpoint_NNNNNN/``: the final RLlib checkpoint;
* ``run_config.json``, ``progress.csv``, ``episodes.csv``: training settings and logs;
* ``evaluation/<scenario>/episodes.csv``: the out-of-sample evaluation of the policy;
* ``README.md``: settings, evaluation statistics and the command to use the policy.

Usage::

    python -m experiments.release_policy --campaign results/campaign_1M --dest models/dqn_execution_policy
    python -m experiments.release_policy --run-dir results/learn/training/lr_5e-4_seed_10 \
        --eval-dir results/learn --dest models/dqn_schedule_policy --config experiments/configs/learn.yaml

The second form releases a run trained with ``train_dqn.py`` directly (no
campaign ranking): ``--eval-dir`` is a ``results/<config>`` directory produced
by ``evaluate.py`` whose ``RL`` rows are that run's evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import load_config  # noqa: E402
from experiments.make_figures import df_to_markdown  # noqa: E402
from experiments.policies import resolve_checkpoint  # noqa: E402
from experiments.train_campaign import episode_stats  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=None, help="campaign output directory (contains ranking.csv)")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--run-dir", default=None, help="a training run directory (alternative to --campaign)")
    parser.add_argument("--eval-dir", default=None, help="results/<config> directory with the run's RL evaluation (with --run-dir)")
    parser.add_argument("--dest", default=str(REPO_ROOT / "models" / "dqn_execution_policy"))
    parser.add_argument("--config", default="experiments/configs/paper.yaml", help="configuration the campaign was run with")
    args = parser.parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run = run_dir.name
        eval_root = Path(args.eval_dir) if args.eval_dir else None
        n_ranked = 1
    else:
        campaign = Path(args.campaign)
        ranking = pd.read_csv(campaign / "ranking.csv")
        row = ranking[ranking["rank"] == args.rank].iloc[0]
        run = row["run"]
        run_dir = campaign / "training" / run
        eval_root = campaign / "eval" / run
        n_ranked = len(ranking)
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    ckpt = Path(resolve_checkpoint(str(run_dir)))
    shutil.copytree(ckpt, dest / "checkpoint" / ckpt.name)
    for f in ["run_config.json", "progress.csv", "episodes.csv"]:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, dest / f)
    (dest / "final_checkpoint.txt").write_text(str(Path("checkpoint") / ckpt.name) + "\n")

    # evaluation results of this run (every scenario that was evaluated)
    eval_rows = []
    for sc_dir in sorted(eval_root.glob("*")) if eval_root is not None else []:
        p = sc_dir / "episodes.csv"
        if p.exists():
            (dest / "evaluation" / sc_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / "evaluation" / sc_dir.name / "episodes.csv")
            df = pd.read_csv(p)
            df = df[df["policy"] == "RL"]
            eval_rows.append({"scenario": sc_dir.name, **episode_stats(df)})
    ev = pd.DataFrame(eval_rows)
    if len(ev):
        ev["E(executed)"] = ev["E(executed)"].round(0).astype(int)
        ev["n"] = ev["n"].astype(int)

    rc = json.loads((dest / "run_config.json").read_text())
    env = rc["env_config"]
    beta_ne = env.get("beta_not_enough", 5.0)
    beta_tm = env.get("beta_too_much", 5.0)
    cfg_env = load_config(args.config).get("environment", {})
    overrides = " ".join(f"--env-override {k}={env[k]:g}" for k in ["beta_not_enough", "beta_too_much"]
                         if k in env and float(env[k]) != float(cfg_env.get(k, 5.0)))
    provenance = (f"Trained with `experiments/train_campaign.py` (run `{run}`, rank {args.rank} of {n_ranked} in the campaign "
                  "ranking: completion rate first, then mean implementation shortfall)." if not args.run_dir
                  else f"Trained with `experiments/train_dqn.py` (run `{run}` of `{args.config}`, see docs/CHANGES.md C11).")
    readme = f"""# Released DQN execution policy

{provenance}

## Training settings

| Setting | Value |
|---|---|
| Configuration | `{args.config}` |
| Terminal penalty (per share not executed / in excess) | {beta_ne:g} / {beta_tm:g} |
| Initial learning rate (linear decay to 0) | {rc['lr']:g} over {rc['lr_decay_steps']} steps |
| Exploration | epsilon 1.0 to 0.02 over {rc['epsilon_timesteps']} steps |
| Discount | {rc['gamma']} |
| Q-head hidden layers | {rc['hiddens']} |
| Environment steps | {rc['total_timesteps']} |
| Seed | {rc['seed']} |

Files: `checkpoint/` (RLlib 2.2.0 checkpoint), `run_config.json`,
`progress.csv` and `episodes.csv` (training logs), `evaluation/` (out-of-sample
episodes per scenario).

## Out-of-sample evaluation

Evaluation seeds 1000 + i, one execution window each; statistics of the
normalised implementation shortfall (episode total, cents per share of the
parent order), terminal penalty, fraction of the window used and completion.

{df_to_markdown(ev[["scenario", "n", "E(IS total)", "SD(IS total)", "E(IS step mean)", "E(Pen)", "E(T)", "E(executed)", "P(complete)"]]) if len(ev) else "(no evaluation found)"}

## Using the policy

```bash
python -m experiments.evaluate --config {args.config} --policies RL \\
    --checkpoint {dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest} {overrides}
```

In Python:

```python
from experiments.policies import RLlibDQNPolicy
from experiments.common import env_kwargs_from_config, load_config

cfg = load_config("{args.config}")
env_kwargs = env_kwargs_from_config(cfg)
env_kwargs.update({{k: v for k, v in dict(beta_not_enough={beta_ne:g}, beta_too_much={beta_tm:g}).items() if v != env_kwargs.get(k)}})
policy = RLlibDQNPolicy("{dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest}", env_config=env_kwargs)
action = policy.get_action(state)      # state: the 8-feature observation of markets-execution-v0
q_values = policy.q_values(state)
```
"""
    (dest / "README.md").write_text(readme)
    print(f"released {run} -> {dest}")
    print(ev.to_string(index=False) if len(ev) else "no evaluation")


if __name__ == "__main__":
    main()
