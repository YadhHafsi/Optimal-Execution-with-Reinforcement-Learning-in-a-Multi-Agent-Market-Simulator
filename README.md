# Optimal Execution with Reinforcement Learning in a Multi-Agent Market Simulator

Code accompanying

> Y. Hafsi and E. Vittori, "Optimal Execution with Reinforcement Learning in a
> Multi-Agent Market Simulator", *2026 International Conference on Artificial
> Intelligence, Computer, Data Sciences and Applications (ACDSA)*, 2026.
> DOI: [10.1109/ACDSA67686.2026.11467851](https://ieeexplore.ieee.org/document/11467851),
> preprint [arXiv:2411.06389](https://arxiv.org/abs/2411.06389).

A Deep Q-Network executes a 20 000-share buy order over 30 minutes in ABIDES, a
multi-agent limit-order-book simulator (RMSC-4: market makers, value, momentum
and noise agents). Policies are evaluated out of sample against TWAP, a passive
and a random policy, in five market scenarios, by implementation shortfall (IS,
cents per share, higher is better). The repository provides the Gym environment,
training and evaluation scripts, figures, and trained policies.

The environment is a corrected version of the one used for the paper (fixes to
the execution environment and to the ABIDES market-maker and order-book code,
see [docs/CHANGES.md](docs/CHANGES.md)). Reported numbers are therefore not
expected to match the paper's tables exactly.

## Installation

Python 3.10 (Ray RLlib 2.2.0, gym 0.23.1, torch 1.13.1).

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -e ./abides-core -e ./abides-markets -e ./abides-gym
python -m pytest tests -q
```

## Usage

```bash
make check-market      # market validity on every evaluation seed and scenario
make train             # DQN runs of the selected configuration (learning rates, seeds)
make evaluate-baselines  # TWAP, Passive, Random on the five scenarios
make evaluate-rl       # selected run on the five scenarios
make price-paths figures # figures, tables, t-tests and report in results/ and figures/
```

`CONFIG` selects the configuration (default `learn.yaml`), e.g.
`make all CONFIG=experiments/configs/paper.yaml`.

```python
import gym, abides_gym
from experiments.common import load_config, env_kwargs_from_config

env = gym.make("markets-execution-v0", **env_kwargs_from_config(load_config("experiments/configs/learn.yaml")))
env.seed(1000); state = env.reset()
state, reward, done, info = env.step(2)   # participation 1.0 x TWAP rate for 10 s
```

## Configurations

| File | Decision process |
|---|---|
| `experiments/configs/paper.yaml` | The paper's MDP: a market order of 0 to 80 shares every second, terminal penalty on unexecuted shares. |
| `experiments/configs/learn.yaml` | Participation multipliers (0 to 2) of the TWAP rate every 10 s inside a completion corridor around the TWAP schedule; reward relative to TWAP. |
| `experiments/configs/learn_trend.yaml` | As `learn.yaml`, in a market whose fundamental drifts upward or downward (`fund_drift`). |

## Trained policies

| Directory | Configuration |
|---|---|
| `models/dqn_schedule_policy/` | `learn.yaml`; mean shortfall equal to TWAP's (paired difference +0.07 cents per share, 200 seeds), all orders completed. |
| `models/dqn_trend_policy/` | `learn_trend.yaml`; see its README. |
| `models/dqn_execution_policy/` | `paper.yaml` with terminal penalty 100, 1M training steps; all orders completed. |

Each directory has a README with training settings and out-of-sample evaluation.
Evaluate a policy with
`python -m experiments.evaluate --config <config> --policies RL --checkpoint <dir>`.

## Documentation

* [docs/DESIGN.md](docs/DESIGN.md): market, decision processes, algorithm, baselines, evaluation protocol, results.
* [docs/CHANGES.md](docs/CHANGES.md): changes with respect to the original code.

## Citation

```bibtex
@inproceedings{hafsi2026optimal,
  author    = {Hafsi, Yadh and Vittori, Edoardo},
  title     = {Optimal Execution with Reinforcement Learning in a Multi-Agent Market Simulator},
  booktitle = {2026 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA)},
  year      = {2026},
  doi       = {10.1109/ACDSA67686.2026.11467851}
}
```

Please also cite ABIDES (Byrd, Hybinette and Balch, 2020) and ABIDES-Gym
(Amrouni et al., 2022); entries are in [docs/ABIDES_README.md](docs/ABIDES_README.md#citing-abides).

## License

BSD-3-Clause, see [LICENSE](LICENSE). ABIDES is copyright J.P. Morgan Chase
(2021) and Georgia Tech Research Corporation (2019); the additions of this
repository are copyright Yadh Hafsi (2026).
