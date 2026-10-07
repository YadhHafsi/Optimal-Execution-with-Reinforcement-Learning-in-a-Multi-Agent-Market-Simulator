# Experimental design

This document specifies, in a self-contained way, the experiment that this
repository reproduces: the optimal-execution study of

> Y. Hafsi and E. Vittori, "Optimal Execution with Reinforcement Learning in a
> Multi-Agent Market Simulator", *2026 International Conference on Artificial
> Intelligence, Computer, Data Sciences and Applications (ACDSA)*, pp. 1-9,
> doi:10.1109/ACDSA67686.2026.11467851 (preprint arXiv:2411.06389v2).

It gives the notation, every formula implemented in the code, the baseline
policies and the training and evaluation protocols. The code changes that
were necessary are documented in [CHANGES.md](CHANGES.md).

The environment specified here is a corrected version of the one used for the
paper: the fixes to the execution environment and to the ABIDES market-maker
and order-book code (CHANGES.md, B1-B12) mean that the results of this
repository are not expected to reproduce the paper's reported numbers exactly.

## 1. Problem

A trader must buy a parent order of $X_0$ shares of a single asset within a
time window $[0, T]$, choosing at each of $N + 1$ decision points
$t_k = kT/N$ how many shares $\Delta x_k \ge 0$ to buy with a market order.
Let $x_k = \sum_{j \le k} \Delta x_j$ be the cumulative executed quantity,
$P_0$ the arrival price (mid price at $t_0$) and $P_k$ the average fill price
of the order sent at $t_k$. The execution quality is measured by the
implementation shortfall (IS) of Perold (1988), written here for a buy order so
that positive values are good:

$$
\mathrm{IS} = \sum_{k=0}^{N} \Delta x_k \,(P_0 - P_k), \qquad
\overline{\mathrm{IS}} = \frac{\mathrm{IS}}{X_0}\ \text{(cents per share)}.
$$

The paper studies whether a reinforcement-learning (RL) agent trained in the
multi-agent market simulator ABIDES can obtain a better and less variable
$\overline{\mathrm{IS}}$ than standard execution schedules while keeping market
impact small.

## 2. Market simulator

### 2.1 ABIDES and the RMSC-4 configuration

The market is simulated with ABIDES (Byrd, Hybinette and Balch, 2020) through
its Gym interface ABIDES-Gym (Amrouni et al., 2022). The background population
is the reference configuration RMSC-4 (`abides_markets/configs/rmsc04.py`):

| Agent type | Count | Behaviour (parameters as in the code) |
|---|---|---|
| Exchange | 1 | Price-time priority limit order book, 10 levels published, 500-message stream history |
| Noise agents | 1000 (10 / 2000 in the robustness study) | One randomly timed market or limit order per day, U-quadratic arrival between 09:00 and 16:00, size from the order-size mixture of Section 2.3 |
| Value agents | 102 | Trade on a noisy observation of the fundamental, mean-reversion estimate $\kappa_{va} = 1.67\times10^{-15}$, arrival rate $\lambda_{va} = 5.7\times10^{-12}$ ns$^{-1}$ |
| Adaptive market makers | 2 | Ladder of 10 levels on each side spaced 5 ticks apart, size 2.5% of recent transacted volume (minimum 1 share), quotes refreshed every **60 s** (Poisson, the RMSC-4 default), see D4 below |
| Momentum agents | 12 (6 / 24 in the robustness study) | Compare the 20- and 50-observation mid-price averages, wake up every 37 s (Poisson), order size from the mixture |

Prices are integers in cents; the tick is 1 cent and the fundamental is
centred on $\mu = 100\,000$ cents.

**Market makers (D4).** Each adaptive market maker wakes up at Poisson times
with mean `mm_wake_up_freq`, cancels its orders, queries the spread and the
volume transacted since its previous wake-up, and posts a ladder of 10 levels
per side with $\max(1, \mathrm{round}(0.025\,V))$ shares per level, $V$ being
that volume. The paper's text gives a wake-up of 1 s; the repository uses the
RMSC-4 default of 60 s for two reasons documented in CHANGES.md (B10, D4): in
the ABIDES code as found, a market maker whose first query preceded the first
two-sided book never woke again (fixed here, C9.1), which at 1 s silenced both
market makers in 43 of the 50 evaluation seeds, and at 1 s the ladder is sized
from one second of volume, hence 1 share per level, so that a 20-share market
order walks ten levels. At 60 s the two market makers hold about 60% of the
ten-level depth (about 30 shares per level). The paper's reported TWAP
statistics (zero terminal penalty, complete execution) are those of a market
with quoting market makers. A market order that finds no resting liquidity is
discarded by the exchange without any message (B12); the environment reports
the discarded quantity and the pipeline checks that it is zero (Section 6).

### 2.2 Fundamental value

The oracle (`SparseMeanRevertingOracle`) advances the fundamental $X_t$
between two requests $d$ nanoseconds apart with a mean-reverting update plus
Poisson jumps ("megashocks", rate $2.78\times10^{-18}$ ns$^{-1}$, mean 1000
cents, variance $5\times10^{4}$):

$$
X_{t+d} = \mu + (X_t - \mu)\,e^{-\theta d} + \sigma_d\,\varepsilon, \qquad
\varepsilon \sim \mathcal N(0, 1),\quad \theta = 1.67\times10^{-16}\ \text{ns}^{-1}.
$$

The research code that produced the paper changed the innovation scale of
upstream ABIDES. Upstream uses the exact Ornstein-Uhlenbeck (OU) variance
$\sigma_d = \sqrt{\tfrac{\sigma^2}{2\theta}(1 - e^{-2\theta d})}$ with
$\sigma = 5\times10^{-5}$; the fork uses

$$
\sigma_d = \frac{\sigma}{2\theta}\,\bigl(1 - e^{-2\theta d}\bigr), \qquad \sigma = 5\times10^{-10},
$$

which the paper describes as "a volatility of $5\times10^{-10}$". For $d$ = 1 s
this gives $\sigma_d \approx 0.5$ cents (upstream: 1.6 cents) and the scale
grows linearly rather than as $\sqrt d$ for short horizons. **The fork's
process is kept** (decision D3), because reproducing the paper means using the
simulator the paper used; it is documented as a deviation from upstream ABIDES.

### 2.3 Order-size model

Background agents draw order sizes from an eleven-component mixture: a
log-normal (log-mean 2.9, log-sd 1.2, weight 0.20) and ten narrow normals
(sd 0.15) centred on the round lots 100, 200, ..., 1000 shares with weights
0.70, 0.06, 0.004, 0.0329, 0.001, 0.0006, 0.0004, 0.0005, 0.0003, 0.0003; the
sample is rounded to an integer. Upstream implements this with `pomegranate`
0.14; this repository re-implements the identical mixture in numpy (D6) and
verifies it against `pomegranate` 0.14.8 in the test-suite.

## 3. Markov decision process

Implemented in `abides-gym/abides_gym/envs/markets_execution_environment_v0.py`
(registered as `markets-execution-v0`). Default values are those of the paper's
Section V-A/V-B; every constant is a constructor argument and a key of
`experiments/configs/paper.yaml`.

### 3.1 Timing

| Symbol | Meaning | Value |
|---|---|---|
| $\Delta t$ | control interval (`timestep_duration`) | 1 s |
| $T$ | execution window (`execution_window`) | 30 min ($N = 1800$ decision points) |
| $t_0$ | arrival, `first_interval` after the 09:30 open | 09:35:00 (D5) |
| $X_0$ | parent order (`parent_order_size`) | 20 000 shares, buy |

The agent wakes up at $t_0 + k\Delta t$; the order sent at decision point $k$
is matched immediately and its fills are observed at decision point $k + 1$.

### 3.2 State

$s_k \in \mathbb R^{8}$ (with `state_history_length` $= 4$):

| Index | Feature | Definition |
|---|---|---|
| 0 | `holdings_pct` | $x_k / X_0$ |
| 1 | `time_pct` | $k / N$ |
| 2 | `imbalance_5` | $\dfrac{\sum_{j\le5} Q^{bid}_j}{\sum_{j\le5} Q^{bid}_j + \sum_{j\le5} Q^{ask}_j}$ over the best 5 levels (0.5 if the book is empty) |
| 3 | `best_bid` | best bid price in cents (mid or last trade if the side is empty) |
| 4 | `best_ask` | best ask price in cents |
| 5-7 | `mid_return_lag{3,2,1}` | the last three one-step changes of the mid price, zero-padded |

The paper lists features 0-4 and adds "a state history length of 4 and a
market data buffer of length 50": the history length produces features 5-7;
the market-data buffer (50 L2 snapshots at 0.1 s) is the agent's internal
buffer from which the latest snapshot before each wake-up is taken. The
exchange sends the agent a new 10-level snapshot whenever the book has
changed, at most every 0.1 s (B11), so the features are at most 0.1 s old at
the decision.
RLlib's `MeanStdFilter` normalises every feature online during training and
the frozen statistics are applied at evaluation.

### 3.3 Actions

$a_k \in \{0, 1, 2, 3, 4\}$: $a_k = 0$ sends no order; $a_k = j \ge 1$ sends a
market buy order of $Q^{(j)} = j\,Q_{\min}$ shares with $Q_{\min} = 20$
(`q_min`, `num_action_levels`). Orders are not capped by the remaining
quantity; over-execution is penalised (Section 3.4). The environment also
offers `action_repeat` (default 1, not in the paper, D14): the chosen action is
repeated at that many consecutive wake-ups, so that a decision every
`action_repeat` seconds sends $j\,Q_{\min}$ shares per second over the
interval; the decision reward is the sum of the step rewards.

### 3.4 Reward

With $F_k$ the set of fills $(q, p)$ obtained from the order sent at $k$ and
$P^{ask}_{k}$ the best ask observed at decision point $k$ (before the order):

$$
r_{k+1} = \frac{1}{X_0}\Bigl[\; w_{IS}\sum_{(q,p)\in F_k} q\,(P_0 - p)
\;-\;\alpha\, d_k \;-\; \beta\,|x_N - X_0|\,\mathbb 1\{k+1 = \text{last}\}\Bigr],
\qquad
d_k = \sum_{(q,p)\in F_k} \max(0,\, p - P^{ask}_{k}).
$$

| Symbol | Argument | Paper | Notes |
|---|---|---|---|
| $w_{IS}$ | `is_weight` | 1 | the original code used $w_{IS} = 0.5$ with $\alpha = 1$, i.e. half of the paper's step reward (B3) |
| $\alpha$ | `depth_penalty_alpha` | 2 | depth consumed, in cents beyond the pre-trade touch, summed over fills (not weighted by quantity), as in the original code (D2) |
| $\beta$ | `beta_not_enough` / `beta_too_much` | 5 / 5 | cents per share missing / in excess at the end (paper Sec. V-A); the original notebooks used 100-200 / 5-20 (B4) |

For a sell parent order the shortfall term becomes $q\,(p - P_0)$ and the depth
uses the best bid. The normalisation by $X_0$ makes every reward component a
quantity in cents per share of the parent order; `episode_reward` is the sum of
all step rewards including the terminal penalty.

### 3.5 Termination

The episode ends when $x_k \ge X_0$ or when $t_k \ge t_0 + T$. The paper
states that "once the execution is completed, the reward remains zero for the
rest of the period"; ending the episode at completion is equivalent for the
return since no further reward can be earned.

### 3.6 Info / logged quantities

At every step the environment reports (in `info`): cumulative
`implementation_shortfall` ($\sum \mathrm{IS}_k / X_0$), `depth_penalty`
($\sum \alpha d_k / X_0$), `terminal_penalty`, `episode_reward`,
`executed_quantity`, `remaining_quantity`, `time_pct`, `num_steps`,
`num_trading_steps`, the last-step quantities (`step_*`), the observables
`best_bid`, `best_ask`, `mid_price`, `spread`, `imbalance_5`, `entry_price`
($P_0$) and the action counters, and the liquidity accounting of the market
orders sent: `step_order_quantity`, `step_unfilled_quantity`, `num_orders`,
`num_unfilled_orders`, `unfilled_quantity` (shares that met no resting
liquidity and were discarded by the exchange, B12).

## 4. Learning algorithm

Double DQN with a dueling head (RLlib 2.2.0, torch), configuration
`experiments/train_dqn.py::build_dqn_config`:

| Hyper-parameter | Value | Source |
|---|---|---|
| Q-head hidden layers (`hiddens`) | [50, 20] | paper ("2 layers of 50 and 20 neurons"), original notebooks |
| Model trunk (`fcnet_hiddens`) | [256, 256] (RLlib default) | not stated in the paper (D11) |
| Learning rate | linear from $\eta_0$ to 0 over 90 000 steps; $\eta_0 \in \{10^{-2}, 10^{-3}, 10^{-4}\}$, selected $10^{-3}$ | paper |
| Exploration | $\epsilon$-greedy, 1.0 to 0.02 over 10 000 steps | paper (= RLlib default) |
| Discount $\gamma$ | 0.9999 | paper |
| Training budget | 100 000 environment steps | paper Fig. 4 |
| Replay buffer | prioritized, 50 000 transitions, $\alpha = 0.6$, $\beta = 0.4$ | RLlib default |
| Batch / target update / learning start | 32 / 500 steps / 1000 steps | RLlib default |
| n-step, Adam $\epsilon$, gradient clip | 1 / $10^{-8}$ / 40 | RLlib default |
| Trunk / heads | 8 -> 256 -> 256 (tanh) trunk; dueling advantage and value heads 256 -> 50 -> 20 -> (5, 1) with relu | RLlib default / notebooks |
| Loss / update rhythm | Huber TD loss; one gradient step on a batch of 32 per environment step after the 1000 warm-up steps; hard target update every 500 steps; reporting every 1000 steps | RLlib default |
| Observation filter | MeanStdFilter | original notebooks |
| Seed | 10 (environment seed of Fig. 4); 20 and 30 added for a seed-robustness check | paper |

One training run of 100 000 steps corresponds to roughly 60-100 episodes.

## 5. Baseline policies (`experiments/policies.py`)

| Policy | Definition (paper Sec. V-C) | Implementation |
|---|---|---|
| TWAP | buy $X_0/N$ shares at each decision point (eq. 3) | tracks the cumulative target $X_0 (k+1)/N$ and sends $\mathrm{round}((\text{target} - x_k)/Q_{\min})$ lots, clipped to $\{0,\dots,4\}$, rounding a positive deficit up at the last decision; one 20-share order every 1.8 s when fills succeed |
| Passive (PP) | 60% do nothing; otherwise one of the four sizes with equal probability | $P(a{=}0) = 0.6$, $P(a{=}j) = 0.1$ for $j = 1..4$ |
| Random (RP) | 50% do nothing; otherwise one of {do nothing, $Q_{\min}k$, $k = 1,2,3$} with equal probability | $P(a{=}0) = 0.625$, $P(a{=}j) = 0.125$ for $j = 1..3$ |
| RL | greedy policy of the trained DQN | `RLlibDQNPolicy`, `explore=False` |

Nominal execution rates: TWAP 11.1 shares/s (finishes at $T$), Passive 20
shares/s, Random 15 shares/s.

## 6. Evaluation protocol

* **Seeds.** Evaluation seeds are $1000 + i$, $i = 0..49$, disjoint from the
  training seed; a seed determines the whole ABIDES simulation (all background
  agents) and the policy's own randomness. Every policy is evaluated on the
  same seeds (paired design).
* **Windows.** 50 episodes per (scenario, policy); the first 20 seeds are used
  for the distribution plots (paper: "20 30-minute windows"), all 50 for the
  tables and the t-tests (paper: $n = 50$).
* **Scenarios** (background overrides): standard (1000 noise, 12 momentum),
  10 noise, 2000 noise, 6 momentum, 24 momentum agents. The RL policy is the
  one trained on the standard scenario (robustness of a fixed policy).
* **Market validity (D13).** Before results are used, `experiments/check_market.py`
  runs a do-nothing and a TWAP episode on every evaluation seed of every
  scenario and requires: both market makers quoting, no empty side of the
  published book at any wake-up, no discarded share of any market order, and
  TWAP executing exactly $X_0$. `evaluate.py` prints the discarded-share
  count of every policy after each run. With `paper.yaml` all 250 (scenario,
  seed) pairs pass; TWAP executes 20 000 shares in 100% of the episodes.
* **Per-episode metrics** (`results/<config>/<scenario>/episodes.csv`):
  $\overline{\mathrm{IS}}$ (episode total), the mean per-step
  $\overline{\mathrm{IS}}_k$ over the $N_{ep}$ steps of the episode
  (`step_is_mean`), the last-step value (`step_is_final`), the terminal penalty
  E(Pen), the fraction of the window used E(T) (`time_pct` at the end), the
  cumulative depth penalty, executed quantity, number of steps and of trading
  steps, spread and imbalance at the trading steps.
* **Per-step logs** (`steps.parquet`): time, bid, ask, mid, spread, imbalance,
  holdings, action, fills, reward and, for RL, the Q-value of every action.
* **Tables I-III.** For each scenario and policy: E(IS), E(Pen), E(T), Var(IS)
  over the 50 episodes, for the three IS definitions above.
* **Statistical test.** One-sided two-sample Student t-test with pooled
  variance, $H_0: E(\overline{\mathrm{IS}})_{RL} \le E(\overline{\mathrm{IS}})_S$
  against $H_1: >$, $n = 50$ per strategy, $\mathrm{df} = 98$, critical value
  $t_{0.95}(98) = 1.660$, as in the paper; Welch's version and, because the
  same seeds are used for all policies, a paired t-test are reported alongside.
* **Figures.** Fig. 3: best-ask paths of 6 seeds over 4 hours with no trading
  (and a 30-minute variant). Fig. 4: per-episode training rewards and their rolling
  mean over 10 episodes against environment steps for the three learning
  rates (RLlib's `episode_reward_mean` is a trailing 100-episode mean, which
  for runs of 60-100 episodes is a cumulative mean; it is shown as Fig. S4).
  Fig. 5 / Fig. 8: kernel density and strip plots of $\overline{\mathrm{IS}}$
  per policy (20 seeds), standard scenario / all scenarios. Fig. 6a: bid, ask
  and actions of the first evaluation episode of the RL policy. Fig. 6b: mean
  executed fraction against time fraction with a 95% t-interval over the 20
  episodes. Fig. 6c: mean Q-value of each action in 2.5% time bins. Fig. 7:
  distribution of the spread (bars, cents) and of the 5-level imbalance (kernel
  density) at the decision steps where the policy traded, per policy.

## References

* Amrouni, S., Moulin, A., Vann, J., Vyetrenko, S., Balch, T., Veloso, M. (2022). ABIDES-Gym: Gym environments for multi-agent discrete event simulation and application to financial markets. ICAIF '21.
* Byrd, D., Hybinette, M., Balch, T. H. (2020). ABIDES: Towards high-fidelity multi-agent market simulation. SIGSIM-PADS '20.
* Perold, A. F. (1988). The implementation shortfall: paper versus reality. Journal of Portfolio Management 14(3).
* van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI.
