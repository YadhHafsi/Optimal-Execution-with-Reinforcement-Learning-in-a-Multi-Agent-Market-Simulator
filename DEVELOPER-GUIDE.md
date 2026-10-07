# Developer Guide

This guide describes the layout of the repository, the conventions the code
follows and how to work on each part. It is adapted from the ABIDES developer
guide of the upstream project (jpmorganchase/abides-jpmc-public); the parts of
that guide that described JPMorgan-internal processes have been removed. For
the contribution workflow (environment setup, tests, pull requests) see
`CONTRIBUTING.md`; for the design and the changes made to the original code see
`docs/DESIGN.md` and `docs/CHANGES.md`.

## Repository layout

```
Optimal-Execution-with-Reinforcement-Learning-in-a-Multi-Agent-Market-Simulator/
    abides-core/        package abides_core:    kernel, agents base class, messages, latency, generators
    abides-markets/     package abides_markets: exchange, order book, trading agents, oracles, RMSC configs
    abides-gym/         package abides_gym:     Gym wrapper + the execution environment of the paper
    experiments/        training, evaluation, price paths, figures (run with python -m experiments.<name>)
        configs/        YAML experiment configurations (paper.yaml, smoke.yaml)
    tests/              tests of the execution environment and of the experiments package
    docs/               paper text/PDF, DESIGN.md, CHANGES.md
    results/            outputs of the scripts (git-ignored)
    logs/               stdout of background runs (git-ignored)
    legacy/             upstream utility scripts of the original fork, kept for reference, not maintained
    Makefile            install / test / train / evaluate / figures targets
    .github/workflows/  CI: tests + smoke training on Python 3.10
```

### The three ABIDES packages

The simulator is split into three independently installable packages. The
dependency direction is strictly `abides_core <- abides_markets <- abides_gym`.

**abides-core.** A general-purpose discrete-event simulator. The `Kernel`
owns the event queue and the simulated clock (nanosecond integers since the
epoch, see `abides_core.utils.str_to_ns`), delivers `Message` objects between
`Agent` instances through a `LatencyModel`, and drives the agents' `wakeup`
calls. Nothing in this package knows about markets. Tests are in
`abides-core/tests`.

**abides-markets.** Financial-market extension of the core: the
`ExchangeAgent` with its `OrderBook`, the order and market-data message types,
the stylised background agents (noise, value, momentum, adaptive POV market
maker), the oracles that generate the fundamental price
(`SparseMeanRevertingOracle` is the one used here), and the background
configurations in `abides_markets/configs` (`rmsc03`, `rmsc04`, `rmsc05`).
`rmsc04.build_config(seed=..., **kwargs)` returns the dictionary the kernel is
started from; the execution environment forwards the `background` section of
the YAML configuration to it. Tests are in `abides-markets/tests`.

Two deliberate deviations from upstream ABIDES live in this package and are
documented in `docs/CHANGES.md` (decisions D3 and D6): the fundamental process
parameters and innovation scale of the oracle, which are the ones the paper's
results were produced with, and a numpy re-implementation of the order-size
mixture that used to depend on `pomegranate`.

**abides-gym.** Wraps a kernel as a single-agent Gym environment.
`abides_gym/envs/core_environment.py` and `markets_environment.py` contain the
plumbing (a `GymAgent` is inserted into the background configuration, the
kernel is run until that agent's next wake-up, the raw state is buffered).
`markets_execution_environment_v0.py` is the environment of the paper,
registered as `markets-execution-v0`; its module docstring is the reference
description of the MDP (state, actions, reward, termination) and every
constant of the paper is a constructor argument. `markets_daily_investor_environment_v0.py`
is the upstream example environment, kept unchanged.

### The experiments package

`experiments/` is a plain Python package imported from the repository root
(it is not installed). Every script adds the repository root to `sys.path` so
that `python -m experiments.<script>` works from the root.

| module | role |
|---|---|
| `common.py` | `load_config`, `env_kwargs_from_config`, `num_steps_from_env_kwargs`, `policy_env_params`; the list `ENV_KEYS` of environment arguments read from YAML |
| `policies.py` | `BasePolicy`, `TWAPPolicy`, `PassivePolicy`, `RandomPolicy`, `AggressivePolicy`, `DoNothingPolicy`, `RLlibDQNPolicy`, and the factory `make_policy` |
| `train_dqn.py` | RLlib DQN training (`build_dqn_config` + CLI); writes `progress.csv`, `episodes.csv`, checkpoints |
| `evaluate.py` | out-of-sample evaluation of the policies on every scenario; writes `episodes.csv` and `steps.parquet` per scenario |
| `sample_price_paths.py` | undisturbed best-ask paths for Figure 3 |
| `make_figures.py` | figures and tables of the paper from `results/` |

Run `python -m experiments.<script> --help` for the exact flags; the module
docstrings are the help text.

Conventions specific to this package:

* The hand-crafted baselines must not import RLlib or torch; they are
  executed in a `multiprocessing` pool. `RLlibDQNPolicy` imports them lazily.
* Ray is initialised with `num_cpus=1` and a small object store; the market
  simulation is single-threaded and several runs are meant to share a machine
  (`OMP_NUM_THREADS=1`).
* Output directories are created by the scripts; results files are appended
  to, not overwritten, by `evaluate.py`.
* Long runs print one progress line per iteration with `flush=True` so that
  `logs/*.log` can be tailed.

## Code style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* Format with [black](https://github.com/psf/black) (default settings) and
  sort imports with [isort](https://pypi.org/project/isort/) (black profile).
  Neither tool is enforced by a hook in this repository; run them before
  committing. Both are available through `requirements-dev.txt` once the
  maintainer regenerates it, or install them ad hoc with `uv pip install black isort`.
* Provide type annotations for function signatures and class attribute
  declarations. `mypy.ini` in the root ignores missing stubs for numpy, pandas
  and scipy; run `mypy experiments abides-gym/abides_gym/envs` occasionally.
* Prefer explicit constructor arguments with defaults equal to the paper's
  values over module-level constants, so that configuration files can override
  them.
* Pre-commit hooks: the upstream project used
  [pre-commit](https://pre-commit.com/) with a `pytest-check` hook. The
  configuration file is not part of this repository; if you want the same
  behaviour locally, install `pre-commit` from `requirements-dev.txt` and add a
  `.pre-commit-config.yaml` to your clone (do not commit it unless the
  maintainers agree).

## Testing

The project uses pytest. `pytest.ini` in the root sets `testpaths` to
`tests`, `abides-core/tests` and `abides-markets/tests`, filters the
deprecation warnings of the pinned dependencies and defines one marker:

* `slow`: tests that run a full (short) ABIDES simulation.

Commands:

```bash
python -m pytest tests -q               # execution environment + experiments (about 5 s)
python -m pytest tests -q -m 'not slow' # unit tests only
python -m pytest -q                     # also the upstream ABIDES tests
```

Guidelines:

* A unit test should construct the smallest object that exhibits the
  behaviour. For the reward and state functions this means calling
  `raw_state_to_state`, `raw_state_to_reward` and
  `raw_state_to_update_reward` on a hand-built raw state, not stepping a
  kernel. `tests/test_execution_env.py` shows how.
* Tests that need a running market use the `smoke.yaml` configuration
  (2-minute window, 2000 shares) and are marked `slow`.
* Determinism is a feature under test: `test_seeding_is_reproducible` checks
  that two environments seeded identically produce identical trajectories.
  Any change that legitimately alters trajectories must update the affected
  tests and be recorded in `docs/CHANGES.md`.
* Regression against the previous behaviour of the simulator (the upstream
  "macro-testing" of order books between commits) is done here by comparing
  `results/<config>/<scenario>/episodes.csv` before and after a change on the
  same seeds; decision D9 in `docs/CHANGES.md` is an example (bit-identical
  trajectories on three seeds after removing the deep copies).

## Documentation

* Code is documented when written. Classes and functions have
  [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
  docstrings.
* Non-code documentation lives in `docs/` in Markdown: `DESIGN.md` (experiment design, notation, formulas,
  baselines, evaluation protocol), `CHANGES.md` (every change to the code and
  the MDP with its rationale and timeline).
* Use metric units and avoid em-dashes.

### Writing documentation strings

```python
class ExampleClass:
    """A short sentence describing the class should be on the first row.

    Each class should have a description of the class and its functionality.

    Class attributes that are relevant for users of the class are listed as
    follows; types are taken from the annotations and are not repeated here.

    Attributes:
        price: The order limit price in cents.
        quantity: The number of shares.

    Single backticks highlight code strings, e.g. `ExampleClass`.
    """

    price: int
    quantity: int

    def do_something(self, x: int, y: int) -> bool:
        """Does something with x and y.

        The same style is used for functions and methods. The `self` parameter
        is not documented.

        Arguments:
            x: The first number.
            y: The second number.

        Returns:
            True if x > y else False.
        """
        return x > y
```

## Working on the environment

The execution environment is the scientific core of the repository, so a few
extra rules apply to `abides-gym/abides_gym/envs/markets_execution_environment_v0.py`:

1. Read the module docstring and the corresponding paper sections (IV-A, V-A,
   V-B) first; the docstring is kept in sync with the code and is the
   reference description of the MDP.
2. Every quantity of the paper (`X0`, `Q_min`, `alpha`, `beta`, window, step,
   start time, history length) is a constructor argument with the paper's
   value as default. Add new knobs the same way and add the key to
   `experiments.common.ENV_KEYS`.
3. Metrics returned in `info` are per-step quantities (best bid/ask, spread,
   imbalance, step implementation shortfall, step depth penalty, fills). Episode
   statistics are computed by `experiments/evaluate.py`, not by the
   environment. The original code mixed the two (bug B7 in `docs/CHANGES.md`);
   do not reintroduce that.
4. Before and after any change, run the slow tests and compare
   `episodes.csv` of the smoke configuration on a few seeds; record the
   outcome in `docs/CHANGES.md`.

## Useful links

* PEP 8: https://www.python.org/dev/peps/pep-0008/
* Type annotations cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
* Google-style docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
* black: https://black.readthedocs.io/en/stable/
* ABIDES (upstream): https://github.com/jpmorganchase/abides-jpmc-public
* ABIDES-Gym paper: https://arxiv.org/abs/2110.14771
* Paper reproduced here: https://ieeexplore.ieee.org/document/11467851 (arXiv:2411.06389)
