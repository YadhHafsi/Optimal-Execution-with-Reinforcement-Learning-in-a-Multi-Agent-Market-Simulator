# Code changes

Every change made to the original research code
(github.com/YadhHafsi/execution, branch `yadh-branch`, commit `722f23c`, a fork
of jpmorganchase/abides-jpmc-public) is listed here with the reason for the
change and its effect. Identifiers B1-B12 refer to the fixes and D1-D16 to
the design decisions made along the way (the identifiers are used in the
tables below and in [DESIGN.md](DESIGN.md)).

> **Replication of the paper.** The paper's results are not replicated exactly.
> The environment and the simulator were changed (execution environment, C1;
> ABIDES market makers, order book and data feed, C9; market-maker setting,
> D4), so a run of this repository is a different experiment from the one
> reported in the paper, and its numbers are not expected to coincide with the
> paper's tables and figures. The original runs' settings are also only partly
> recorded (execution start time, market-maker configuration, definition of the
> reported IS statistic), which prevents a bit-for-bit reconstruction.

## C1. Execution environment (`markets_execution_environment_v0.py`)

The 859-line environment of the fork carried three action-space variants
(`discrete`, `multi_discrete`, `continuous`), commented-out state features and
several reward formulas. It was rewritten from scratch (about 500 lines, fully
documented) to implement exactly the MDP of the paper (DESIGN.md Section 3).

| # | Change | Why | Effect |
|---|---|---|---|
| C1.1 | Action space is `Discrete(1 + num_action_levels)`; action 0 returns an empty action list; action $k$ sends `MKT` of `k * q_min` shares. The old `multi_discrete` branch computed `size = max(parent_order_size * action / 1000, 1)`, so action 0 sent a **1-share market order** (B1), and a `CCL_ALL` was sent before every order (B6). | Paper: "do nothing, or consume $Q_{\min} k$". | Removes a spurious 1-share-per-second flow (about 1800 shares per 30-minute episode). |
| C1.2 | `q_min` and `num_action_levels` are explicit arguments (were derived from `parent_order_size/1000`). | Clarity; the paper states $Q_{\min} = 20$. | None with the paper's values (20000/1000 = 20). |
| C1.3 | `reset()` is overridden and resets `step_index`, `entry_price`, the metrics tracker and the action counters. In the fork `step_index` was set only in `__init__`, so the arrival price $P_0$ was frozen at the value of the **first** episode for the lifetime of the environment object (B2). | RLlib reuses one environment object across all training episodes, so training rewards were computed against a stale $P_0$. | Correct implementation shortfall in every episode; changes training rewards materially whenever the fundamental drifts between episodes. |
| C1.4 | Reward $= (w_{IS}\,\mathrm{IS}_k - \alpha d_k)/X_0$ with defaults $w_{IS} = 1$, $\alpha = 2$. The fork computed `(ob_reward + 0.5*pnl)/parent_order_size` $= (0.5\,\mathrm{IS}_k - 1\cdot d_k)/X_0$ (B3), exactly half of the paper's step reward. The original is recovered with `is_weight=0.5` **and** `depth_penalty_alpha=1.0`. | Paper equation (4) has unit weight on the shortfall and $\alpha = 2$. | Step rewards are twice the original ones; the terminal penalty is not scaled, so its weight relative to the shortfall halves. |
| C1.5 | Depth penalty $d_k = \sum_{\text{fills}} \max(0, p - P^{ask}_{k})$ (best ask at the previous wake-up), unchanged in substance from the fork's `ob_reward`, but the previous-wake-up snapshot is now guarded (`len(far_side) >= 2 and len(far_side[-2]) > 0`) instead of indexing `asks[-2]` unconditionally. | Robustness at the first step and with an empty book. | None on normal steps. |
| C1.6 | Terminal penalties are `beta_not_enough` and `beta_too_much` in cents per share, defaults 5 / 5 (paper); the fork's defaults were -1000 / -10 (B4). The sign is applied by the environment (arguments are positive). | Paper Section V-A. | Sets the paper's trade-off between finishing and paying impact. |
| C1.7 | State vector: `[holdings_pct, time_pct, imbalance_5, best_bid, best_ask] + 3 mid-price changes`, shape `(8,)`, finite bounds $\pm 10^8$ cents (B8). The fork produced the same features with shape `(8, 1)` and `float32` max/min bounds. | RLlib handles 1-D boxes without reshaping; finite bounds keep `Box.contains` meaningful. | None on values. |
| C1.8 | `holdings_pct` and executed quantity are signed by direction, so `SELL` parent orders work symmetrically (the fork mixed `holdings` and `-holdings`). | Correctness for sells. | None for buys. |
| C1.9 | `info` reports **episode accumulators** (`implementation_shortfall`, `depth_penalty`, `terminal_penalty`, `episode_reward`, `num_trading_steps`) and **last-step** quantities (`step_*`) separately (B7). The fork's `implementation_shortfall` was the last step's value. | Both statistics are useful. | New columns; the last-step statistic is still available as `step_implementation_shortfall`. |
| C1.10 | Removed the unused `square_root_impact_reward` (a $\sigma\sqrt{q/V}$ heuristic computed from a hard-coded "daily volatility" of 0.01), `diff_pct`, `price_impact`, `direction_feature`, `imbalance_all`, `just_quantity_reward_update`, `starting_cash` risk logic, and the `smc_01` config name (B5); `rmsc05` is accepted. | Dead code. | None. |
| C1.11 | Argument validation rewritten (`first_interval + execution_window` must fit in the day, `state_history_length >= 2`, mutable default `background_config_extra_kvargs={}` replaced by `None`). | Safety. | None with valid arguments. |

## C2. Performance of the Gym plumbing (D9)

Profiling one episode showed 98% of the step time in `copy.deepcopy`:
`core_environment.step` deep-copied the raw state four times, `act_on_wakeup`
once, `update_raw_state` deep-copied the 50-snapshot market-data buffer, and
`ignore_mkt_data_buffer_decorator` deep-copied its input and then mutated the
caller's object. Changes:

| File | Change |
|---|---|
| `abides-markets/abides_markets/agents/utils.py` | `ignore_mkt_data_buffer_decorator` and `ignore_buffers_decorator` build reduced dictionaries that share the immutable L2 snapshots; no copy, no mutation of the input. The dead commented block in `get_volume` was removed. |
| `abides-markets/.../background_v2/core_background_agent.py` | `update_raw_state` snapshots the buffers with `list(...)` and copies the containers of `internal_data` (lists shallowly, dictionaries one level deeper because `order_executed` updated the per-order entries of `order_status` in place; since C9.7 those entries are rebound and every copy is shallow) instead of `deepcopy`; per-step lists are rebound at every step, so the snapshot is frozen. Debug comments removed from `apply_actions`. |
| `abides-gym/.../financial_gym_agent.py` | `act_on_wakeup` returns `list(self.raw_state)` instead of `deepcopy(...)`. |
| `abides-gym/.../core_environment.py` | `reset`/`step` hand the same read-only raw state to every consumer. |

Verification: three full episodes (seeds 0, 1, 7, random actions) were
recorded before and after the change; states, rewards, `done` flags and every
`info` field are bit-identical (`tests` reproduce a shorter version). Wall
clock per 30-minute episode went from about 60 s to 0.5-0.8 s (the ABIDES
kernel itself costs about 0.4 ms per step). This also applies to the daily
investor environment, which uses the same decorators.

## C3. Order-size model without `pomegranate` (D6)

`abides_markets/models/order_size_model.py` now implements the eleven-component
mixture in numpy (`OrderSizeModel.sample(random_state)` draws the component
then the value, as `GeneralMixtureModel.sample` does; `OrderSizeModel.mean()`
added). `pomegranate` 0.14 needs Cython 0.29 and numpy < 1.24 and does not build
on current toolchains. Verified against `pomegranate` 0.14.8 (built in a
throwaway environment) on 300 000 samples: mean 105.9 vs 105.9 (analytic
105.94), P(size = 100) 0.699 vs 0.698, P(size < 100) 0.185 vs 0.185,
P(size >= 300) 0.042 vs 0.042, two-sample KS test on the log-normal component
p = 0.74. The random stream differs, so simulations are not bit-identical to
the original code (they never were across platforms).

## C4. Core utilities

* `abides_core/utils.py::str_to_ns` returns `int(pd.to_timedelta(s).to_timedelta64().astype(np.int64))`.
  The fork returned a float (`/ np.timedelta64(1, 'ns')`) to dodge a Windows
  `int32` overflow of upstream's `.astype(int)`; integer nanosecond times are
  restored on every platform (D7).
* `parse_logs_df`: the fork's debug counters and prints removed; the
  `EventTime` guard accepts any `np.integer`.
* `dtype="uint64"` in the seed draws of `rmsc03/04/05.py` and
  `abides_markets/utils/__init__.py` (from the fork) is kept: harmless and
  needed on Windows.

## C5. Gym / Ray compatibility (D8)

* `core_environment.reset` uses `np_random.integers` when available (gym >= 0.22
  returns a `Generator`) and falls back to `randint`; the seed range is
  `[0, 2**32 - 1)`.
* Dependencies pinned in `requirements.txt`: `ray[rllib]==2.2.0`, `gym==0.23.1`
  (the last gym release supported by RLlib 2.2 without the broken `gym==0.21`
  metadata), `torch==1.13.1`, `numpy==1.23.5`, `pandas==1.5.3`, `scipy==1.10.1`,
  `setuptools<70` (RLlib 2.2 imports `pkg_resources`). The original
  `environment.yml` (Windows conda export) and the two `requirements*.txt`
  variants were removed.

## C6. Experiment package (`experiments/`)

New code.

| File | Purpose |
|---|---|
| `common.py` | YAML configuration loading; environment keyword construction; scenario overrides |
| `policies.py` | TWAP, Passive, Random, Aggressive, DoNothing baselines and `RLlibDQNPolicy` (greedy action and Q-values from a checkpoint, with the training-time `MeanStdFilter`) |
| `train_dqn.py` | RLlib 2.2 DQN with the paper's hyper-parameters (`build_dqn_config`), CSV logging of every iteration and every episode, checkpoints; `--env-override KEY=VALUE` changes an environment parameter of the config |
| `evaluate.py` | Parallel out-of-sample rollouts over scenarios x policies x seeds; per-episode and per-step outputs |
| `sample_price_paths.py` | Undisturbed price paths (Fig. 3) |
| `make_figures.py` | All figures, Tables I-III, t-tests (pooled, Welch, paired), HTML report |
| `train_campaign.py` | Grid of trainings run in parallel (terminal penalties x learning rates x seeds), out-of-sample evaluation of every policy and ranking (completion rate first, then mean IS) |
| `release_policy.py` | Copies a ranked campaign policy with its provenance and evaluation into `models/` and writes its README |
| `configs/paper.yaml` | Standard configuration; `smoke.yaml` (tests/CI) |


## C7. Tests (`tests/`, `pytest.ini`)

* `test_execution_env.py`: action mapping, observation space, reward equation
  on synthetic raw states (fills at and beyond the touch), zero reward without
  fills, symmetric terminal penalty, termination conditions, state vector
  values, a full short episode with reset consistency, and seed reproducibility.
* `test_policies_and_models.py`: TWAP schedule sums to $X_0$ and catches up
  after missed fills, action frequencies of Passive and Random, policy
  re-seeding, factory, and the order-size model against the `pomegranate`
  reference statistics.
* Upstream ABIDES tests (`abides-core/tests`, `abides-markets/tests`) are kept.

## C8. Repository hygiene

* Removed from version control: `build/`, `dist/`, `*.egg-info`, `__pycache__`
  and `.pyc` files (including 52 committed byte-code files), the `.conda`
  directory (Windows DLL leftovers), `notebooks/` and `notebooks/log/*`,
  `.ipynb_checkpoints`, `.vscode`, `spinningup/` (empty), `3.19`,
  `3.19.0`, `3.20.0` (pip output accidentally committed), `results_files.xlsx`
  (a two-line note), `environment.yml`.
* The original notebooks are not part of this repository (their code targets
  the old environment API); `version_testing/`, `profiling/` and the original
  install scripts moved to `legacy/`.
* `.gitignore` extended (`.venv/`, `logs/`, `results/`, `figures/`).
* Licence: BSD-3-Clause with the JPMorgan Chase (2021), Georgia Tech Research
  Corporation (2019) and Yadh Hafsi (2026) notices; `CITATION.cff` added.

## C9. Market liquidity (B10-B12, D4, D13)

Investigating why TWAP did not complete the parent order in 30% of the
evaluation episodes showed that 8.1% of its market orders were discarded by
the exchange for lack of resting liquidity, and that the market had no market
maker in 43 of the 50 evaluation seeds. Changes (2026-09-12):

| # | File | Change | Why | Effect |
|---|---|---|---|---|
| C9.1 | `abides-markets/.../market_makers/adaptive_market_maker_agent.py` | When both query responses are in and no mid price is known yet, the agent resets its state and schedules the next wake-up instead of returning; the book-imbalance re-quote is skipped when no mid price is known (it raised inside a bare `except`). | B10: with `mm_wake_up_freq="1S"` the first spread query (open + Exp(1 s)) found an empty or one-sided book and the market maker never woke again; both silent in 43/50 seeds, one in 5/50. Identical code upstream. | Both market makers quote in 50/50 seeds at 1 s and at 60 s. |
| C9.2 | `abides-markets/abides_markets/order_book.py` | `handle_limit_order` stamps `last_update_ts` after a resting order or a match; `handle_market_order` after a fill. | B11: the timestamp was written only by cancel/modify, and the exchange gates L2 publication on it, so the execution agent's snapshot was refreshed at cancellations only (median age 5-7 s, max 69 s). | Snapshot at decision at most 0.1 s old; 2326 snapshots per episode instead of about 250. State features, $P_0$ and the depth-penalty reference are current. |
| C9.3 | `abides-gym/.../markets_execution_environment_v0.py` | `step_order_quantity`, `step_unfilled_quantity`, `num_orders`, `num_unfilled_orders`, `unfilled_quantity` in the metrics tracker and `info`. | B12: the order book discards the unfilled remainder of a market order silently. | Every discarded share is visible per step and per episode. |
| C9.4 | `experiments/evaluate.py`, `experiments/make_figures.py` | New columns in `episodes.csv` / `steps.parquet`; a liquidity check printed after every evaluation (orders, unfilled orders, unfilled shares, completion per policy and scenario, with a warning if any share was discarded); `Unfilled orders` / `Unfilled shares` columns in the summary tables. | D13. | The pipeline reports "all market orders were filled in full" or warns. |
| C9.5 | `experiments/check_market.py`, `Makefile` (`check-market`) | Do-nothing and TWAP episodes on every evaluation seed and scenario; reports market makers quoting, empty-side fraction, executed quantity, discarded shares; exit status 1 on any silent market maker or discarded share. | D13. | `paper.yaml` passes on 250/250 (scenario, seed) pairs: TWAP executes exactly 20 000 shares everywhere, 0 discarded shares. |
| C9.6 | `experiments/configs/paper.yaml`, `smoke.yaml`, tests | `mm_wake_up_freq: "60S"` (RMSC-4 default). | D4: at 1 s the market maker sizes its ladder from one second of volume, 1 share per level; even with C9.1 TWAP still loses 660 shares over 50 episodes (complete in 74%). | Market makers hold about 60% of the 10-level ask depth (about 30 shares per level). Baseline results change: the previous results were produced in a market without market makers. |
| C9.7 | `abides-markets/.../core_background_agent.py`, environment `raw_state_to_info` | `order_executed` rebinds the per-order `order_status` entries (no in-place mutation), so `update_raw_state` copies shallowly; `info` is a shallow copy of the tracker instead of `dataclasses.asdict`. | Copying the growing `order_status` table entry by entry cost 0.6 ms per step, `asdict` 0.3 ms. | TWAP episode 3.2 s (1.8 ms per step) instead of 4.6 s; trajectories identical (seed 1000: IS 6.0954 before and after). |
| C9.8 | `tests/test_market_liquidity.py` | Order-book timestamps, discarded remainder, environment accounting, market makers quoting after an empty book at their first wake-up (1 s, seeds 1001 and 1008), TWAP executing exactly $X_0$ with no discarded order at 60 s and 1 s. | Regression protection. | 80 tests pass. |

Verification of the diagnosis before the change (results of 2026-09-08,
paper configuration with 1 s market makers): 8.1% of TWAP's 52 303 market
orders filled zero shares; every zero fill arrived on an empty ask side
(order-book instrumentation, 301/301 in three seeds); the four seeds with a
material shortfall (981, 271, 240, 83 shares) ended inside an empty-ask spell
and eleven more were short by 1-9 shares (rounding of `round(deficit/20)`);
with 60 s market makers and the same seeds, the only seeds with any discarded
order were exactly the four seeds whose market makers had died at the open.
After the change, on the same 50 seeds and five scenarios: 0 discarded
shares, TWAP executes exactly 20 000 shares in every episode, both market
makers quote in every seed, no empty side of the book at any wake-up.

Every result of the repository was re-run after C9 (baselines, the three
Fig. 4 trainings, the RL evaluations, price paths, figures and tables). The
released policy of `models/dqn_execution_policy/` was retrained in the
corrected market with `train_campaign.py` (9 runs of 1 000 000 steps, terminal
penalties 5 / 50 / 100, seeds 10 / 20 / 30, learning rate 1e-3, one AWS
c6a.4xlarge, 6.8 h) and the rank-1 policy released with `release_policy.py`;
its README lists the settings and the out-of-sample statistics.

## C10. Hyper-parameter search tooling (2026-09-14)

Added to look for DQN settings that beat TWAP in the corrected market. None of
these changes affects the paper configuration (`paper.yaml`) or the results of
C9 unless the new options are used.

| # | File | Change | Why | Effect |
|---|---|---|---|---|
| C10.1 | `experiments/train_dqn.py` | `--dqn-override KEY=VALUE` (repeatable): any RLlib DQN setting, dotted keys for dictionary settings (`replay_buffer_config.capacity`, `model.fcnet_hiddens`); recorded in `run_config.json` (`apply_dqn_overrides`). | Tune n-step returns, replay size, target update, batch size, trunk width without editing code. | None by default. |
| C10.2 | `experiments/policies.py` | `RLlibDQNPolicy` rebuilds the network from the run's `run_config.json` (Q-head sizes, discount, DQN overrides; `load_run_config`) before restoring the checkpoint. | A checkpoint trained with a different Q-head could not be restored into the default network. | Any trained variant can be evaluated; unchanged for the paper's settings. |
| C10.3 | `experiments/hparam_search.py`, `experiments/configs/search/*.yaml` | Spec-driven search: trains a list of variants in parallel, evaluates each out of sample in its own environment, evaluates TWAP / Passive / Random once per distinct execution setting, and ranks the variants by completion then paired shortfall difference against the same-setting TWAP (and against the paper's 1 s TWAP when `results/paper` exists). | One command per search stage; comparable baselines for coarser decision settings. | `ranking.csv|md`, `search.json`. |
| C10.4 | `abides-gym/.../markets_execution_environment_v0.py` | `action_repeat` (default 1): the chosen action is repeated at that many consecutive wake-ups; the decision reward is the sum of the step rewards and the `step_*` diagnostics are accumulated. `experiments.common.policy_env_params` gives the hand-crafted policies `Q_min * action_repeat` lots and `N / action_repeat` decision points. | Coarser decisions with 1 s child orders (a decision every 30 s spans 60 decisions instead of 1800 steps), an MDP variant not in the paper (D14). | None with the default. Verified: a decision with `action_repeat=3` equals three 1 s steps (test). |
| C10.5 | `experiments/policies.py` (TWAP) | At the last decision point a positive deficit is rounded up instead of to the nearest lot. | With coarse lots (`Q_min * action_repeat`) $X_0$ is not a multiple of the lot and TWAP ended 20 shares short. | Never triggers with the paper's values in the corrected market (the last-step deficit is 0 or 20); TWAP completes with any lot size. |
| C10.6 | `abides-gym/.../markets_execution_environment_v0.py` | `schedule_penalty` (default 0): optional shaping term $-\lambda\,|x_t - X_0 t/T| / X_0$ subtracted from the step reward, accumulated in `info` (`schedule_penalty`, `step_schedule_penalty`). | Search stage 3: start the agent from the TWAP schedule and let it gain only by timing. Not part of the paper's MDP; the evaluation metrics (shortfall, terminal penalty) are unchanged, `episode_reward` includes the term when it is on. | None with the default. |
| C10.7 | `tests/` | Override helper, run-config lookup, action-repeat equivalence, TWAP completion with `action_repeat`, schedule-penalty term. | Regression protection. | 84 tests pass. |

## C11. Learning configuration: environment extensions (2026-09-25)

Added so that the repository ships a configuration in which the DQN agent
learns a state-dependent execution policy (see `experiments/configs/` and
the learning configuration `learn.yaml`). All three options are off by default; `paper.yaml` and
every result of C9 are unchanged (the test-suite checks the paper-mode state
vector and reward bit for bit).

| # | File | Change | Why | Effect |
|---|---|---|---|---|
| C11.1 | `abides-gym/.../markets_execution_environment_v0.py` | `action_mode="schedule"`: action $j$ sets the participation multiplier $m_j = j/(K/2)$ of the TWAP rate $X_0/N$ for the coming decision ($K$ = `num_action_levels`; $j = K/2$ is TWAP), executed as one child market order per wake-up (`action_repeat` wake-ups per decision); the executed quantity is kept inside a corridor $[S(t) - wX_0,\ S(t) + wX_0]$ around the TWAP schedule $S(t) = X_0 t/T$ (`schedule_band` $= w$), whose lower edge closes on $X_0$ at twice the TWAP rate, so the order is always complete at $T$ (`schedule_bounds`, `_schedule_child_order`). | Completion by construction instead of by a terminal penalty: the reward becomes pure execution quality, the policy class is "TWAP plus timing", and no penalty tuning is needed. | Every policy in this mode executes exactly $X_0$ (0 discarded shares in the corrected market); TWAP is the constant middle action. |
| C11.2 | same | `reward_mode="mark_to_market"`: step reward $[\sum_{\text{fills}} q\,(m_{t-1} - p) - \alpha d_t + R_t\,(m_{t-1} - m_t)]/X_0$ with $m$ the mid at the wake-ups and $R_t$ the quantity remaining after the fills. Summing over a complete episode gives exactly $\sum q (P_0 - p)/X_0 - \sum \alpha d_t / X_0$ (Abel summation; the boundary term vanishes when $R_N = 0$). | The paper's reward depends on the arrival price $P_0$, which is not in the state, so the return of a state is not a function of the state; the decomposition removes $P_0$ and separates the spread/impact cost (about 1 cent per share) from the inventory drift (about 12 cents per share), which is the part timing can act on. | Same episode total (test), lower-variance, Markov reward. `info` reports `execution_cost` and `inventory_pnl` (episode and step). |
| C11.3 | same | `state_features`: explicit feature list (paper names plus `schedule_dev`, `spread`, `imbalance_1`, `mid_minus_p0`, `ret_k`, `imb_mean_k`, `dev_k` from a per-episode history). | Features chosen from the predictability study of the predictability study (the 5-level imbalance predicts the next-minute mid move; past returns carry weak momentum; the spread nothing). | `None` reproduces the paper's 8-feature vector exactly. |
| C11.4 | `experiments/common.py`, `experiments/policies.py`, `experiments/evaluate.py`, `experiments/hparam_search.py` | The new keys are forwarded from the YAML `environment` section; `policy_env_params` exposes `action_mode`, `twap_action` and the position of `holdings_pct`; TWAP is the constant middle action in schedule mode; the evaluation logger records every state feature as `obs_<name>` and the reward decomposition; the search driver treats the new options as execution settings (baselines are re-evaluated per setting). | Plumbing. | None for the paper configuration. |
| C11.5 | `tests/` | Schedule mode completes exactly for TWAP, back- and front-loaded policies and the reward identity holds; window features and bounds; unknown feature names rejected; passive and mixed styles; config-driven training defaults. | Regression protection. | 92 tests pass. |
| C11.6 | same environment | `order_style` `"passive"` / `"mixed"` (resting limit orders at the best bid, corridor-forced market catch-up; `passive_horizon`, `passive_offset`; `resting_pct` feature; passive-fill accounting), `reward_mode="vs_twap"`, `schedule_corridor="linear"`; `ImbalanceRulePolicy` (`Rule-lo0.4-hi0.6-s0-f4`) and `PassiveTWAP` baselines. | Instruments of the study of the learning-configuration study (all negative results; kept for reproducibility). | Off by default. |
| C11.7 | `experiments/configs/learn.yaml`, `Makefile`, `train_dqn.py`, `release_policy.py`, `check_market.py` | The learning configuration; the Makefile reads name, learning rates, seeds and budget from `CONFIG` (default `learn.yaml`); `train_dqn.py` takes its defaults from the config's `training` section; `release_policy.py --run-dir/--eval-dir` releases a plain run; the do-nothing census of `check_market.py` runs in lot mode. | One command path for both configurations. | `make all` trains and evaluates the learning configuration. |

## C12. Trending regime (2026-10-07)

| # | File | Change | Why | Effect |
|---|---|---|---|---|
| C12.1 | `abides-markets/.../oracles/sparse_mean_reverting_oracle.py`, `configs/rmsc04.py` | `fund_drift` (cents per second, default 0): the fundamental drifts at that rate, upward or downward, the sign drawn once per simulation (only when the drift is non-zero, so the random stream is unchanged otherwise). | A market in which TWAP is suboptimal, for the study of the trending-regime study (C12.3). | None by default (zero-drift paths are bit-identical, tested). |
| C12.2 | `experiments/policies.py` | `TrendRulePolicy` (`Trend-ret_300-thr10`, `Trend-mid_minus_p0-thr10`): fastest action above +threshold, pause below -threshold, TWAP between, on a state feature. | Hand-written reference for the trending regime. | Variants carry distinct policy names. |
| C12.3 | `experiments/configs/learn_trend.yaml` | Trending-regime configuration (drift 0.04 cents per second, flat corridor, trend features, 150 000 decisions, `no_drift` control scenario). | As C12.1. | `make train CONFIG=experiments/configs/learn_trend.yaml`. |
| C12.4 | `models/dqn_trend_policy` | Median run of five (seed 20, learning rate 5e-4) with its evaluation on 200 fresh seeds in both markets. | Released with the control result and the failed runs stated in its README. | |

## What was deliberately not changed

* The fork's fundamental-price oracle (`fund_vol = 5e-10` and the linear-in-time
  innovation scale, D3), the RMSC-4 agent counts and parameters, the
  momentum agent logic and the kernel; the market maker only in the wake-up
  scheduling of C9.1 and the order book only in the timestamping of C9.2.
* RLlib DQN defaults that the paper does not mention (replay buffer, batch
  size, target update, trunk size), listed in DESIGN.md Section 4.
