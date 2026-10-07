"""ABIDES-Gym environment for the optimal-execution problem studied in

    Y. Hafsi and E. Vittori, "Optimal Execution with Reinforcement Learning in a
    Multi-Agent Market Simulator", ACDSA 2026 (arXiv:2411.06389).

The environment wraps an ABIDES market simulation (background configuration
``rmsc04`` by default) and exposes the execution problem as a single-agent Gym
environment.  Everything that defines the Markov decision process of the paper
(Section IV-A and V-A) lives in this module; the base classes only handle the
plumbing between Gym and the ABIDES kernel.

Markov decision process
-----------------------
Notation: ``X0`` is the parent order size, ``P0`` the arrival price (mid price
at the first wake-up of the agent), ``Q_min`` the minimum child order size.

* **Time.** The agent wakes up every ``timestep_duration`` (paper: 1 second),
  starting ``first_interval`` after the market open (default 30 seconds), and
  has ``execution_window`` (paper: 30 minutes) to complete the parent order.
* **State** (``raw_state_to_state``), a vector of
  ``5 + state_history_length - 1`` features (8 with the paper's history of 4):

  1. ``holdings_pct``  executed fraction of the parent order, ``x_t / X0``;
  2. ``time_pct``      elapsed fraction of the execution window;
  3. ``imbalance_5``   bid volume share over the best 5 levels,
     ``TD^5_bid / (TD^5_bid + TD^5_ask)``;
  4. ``best_bid``      best bid price (cents);
  5. ``best_ask``      best ask price (cents);
  6-8. the last ``state_history_length - 1`` one-step mid-price changes
     (zero-padded at the start of the episode).

  The paper lists items 1-5 explicitly and describes items 6-8 as the
  "state history length of 4".
* **Actions** (``_map_action_space_to_ABIDES_SIMULATOR_SPACE``):
  ``Discrete(1 + num_action_levels)``; action ``0`` does nothing and action
  ``k >= 1`` sends a market order of ``k * Q_min`` shares in the direction of
  the parent order (paper: ``Q_min = 20``, ``k = 1..4``). With
  ``action_repeat > 1`` (not in the paper) the chosen action is repeated at
  that many consecutive wake-ups before the next decision.
* **Reward** (``raw_state_to_reward`` and ``raw_state_to_update_reward``),
  equation (4) of the paper, normalised by ``X0``::

      r_t = [ sum_fills q (P0 - p)          # implementation shortfall (buy)
              - alpha * d_t                  # depth penalty
              - beta  * |x_T - X0| 1{t = T}  # terminal penalty
            ] / X0

  (an optional shaping term ``- lambda * |x_t - X0 t/T| / X0``, off by default,
  can be added with ``schedule_penalty``; see CHANGES.md C10)
  where the sum runs over the fills obtained between two wake-ups, ``d_t`` is
  the depth consumed by those fills, measured as the sum over fills of the
  distance (in cents) between the fill price and the best quote observed at the
  *previous* wake-up (zero when the fill happened at or inside that quote), and
  the terminal penalty applies ``beta_not_enough`` per share left unexecuted or
  ``beta_too_much`` per share executed beyond ``X0``.  For a sell parent order
  the sign of the shortfall term is flipped.
* **Termination** (``raw_state_to_done``): the parent order is complete
  (``x_t >= X0``) or the execution window is over.

Extensions (off by default, not part of the paper's MDP; CHANGES.md C11)
------------------------------------------------------------------------
* ``action_mode="schedule"``: action ``j`` sets the participation rate for the
  coming decision interval to ``m_j = j / (num_action_levels / 2)`` times the
  TWAP rate ``X0 / N`` (``j = num_action_levels / 2`` is TWAP), executed as
  one child market order per wake-up; the cumulative executed quantity is
  kept inside a corridor ``[S(t) - w X0, S(t) + w X0]`` around the TWAP
  schedule ``S(t) = X0 t / T`` (``schedule_band = w``) whose lower edge closes
  on ``X0`` at rate ``2 X0 / N`` so that the order is always complete at ``T``.
* ``order_style`` (schedule mode): ``"market"`` (child market orders, as
  above), ``"passive"`` (the quantity still missing to reach the decision's
  intended quantity, ``m_j`` times the TWAP rate over the decision plus
  ``passive_horizon`` extra wake-ups, rests as a limit buy at the best bid
  plus ``passive_offset`` cents, kept while its price and size stay right;
  unfilled parts carry over as a deviation from the schedule and a market
  order is sent only when the executed quantity falls behind the lower edge
  of the corridor) or ``"mixed"`` (five actions: 0 pause and
  cancel, 1 passive at the TWAP rate, 2 passive at twice the TWAP rate,
  3 market at the TWAP rate, 4 market at twice the TWAP rate).
* ``reward_mode="mark_to_market"``: the step reward is
  ``[sum_fills q (m_prev - p) - alpha d_t + R_t (m_prev - m_t)] / X0``
  (execution cost against the pre-trade mid, plus the inventory ``R_t``
  remaining after the fills marked against the mid move).  It telescopes to
  the same episode total as the arrival-price reward when the order is
  complete, but does not depend on ``P0`` and is therefore Markov in the
  state.  The implementation shortfall reported in ``info`` is the
  arrival-price definition in both modes.
* ``state_features``: an explicit list of feature names replacing the paper's
  vector: the paper's names above plus ``schedule_dev`` (``x_t/X0 - t/T``),
  ``spread``, ``imbalance_1``, ``mid_minus_p0``, ``ret_{k}`` (mid change over
  the last ``k`` wake-ups), ``imb_mean_{k}`` (mean 5-level imbalance over the
  last ``k`` wake-ups), ``dev_{k}`` (mid minus its mean over the last ``k``
  wake-ups).
* **Liquidity accounting** (``info``): ``step_order_quantity`` and
  ``step_unfilled_quantity`` are the shares sent at the step and the part of
  them that met no resting liquidity (the exchange discards the unfilled
  remainder of a market order; it never rests and no message is sent);
  ``num_orders``, ``num_unfilled_orders`` and ``unfilled_quantity`` accumulate
  them over the episode.  A sound simulation has ``unfilled_quantity == 0``
  for every policy (``experiments/check_market.py`` verifies this).

Differences with the original research code are listed in ``CHANGES.md`` at
the root of the repository.
"""
import importlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_markets.orders import LimitOrder
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymMarketsExecutionEnv_v0(AbidesGymMarketsEnv):
    """Optimal-execution environment (see module docstring for the MDP).

    Arguments:
        background_config: name of the ABIDES background configuration module in
            ``abides_markets.configs`` (``"rmsc03"``, ``"rmsc04"`` or ``"rmsc05"``).
        mkt_close: end of the simulated trading day (kernel stop time).
        timestep_duration: time between two wake-ups of the agent (control
            frequency).  Paper: ``"1s"``.
        starting_cash: cash of the agent at the beginning of the episode (cents).
        state_history_length: number of past wake-ups kept in the raw-state
            buffer; the state contains ``state_history_length - 1`` mid-price
            changes.  Paper: 4.
        market_data_buffer_length: number of L2 snapshots buffered by the agent
            between two wake-ups.  Paper: 50.
        first_interval: delay between the market open and the first wake-up of
            the agent (start of the execution window).
        parent_order_size: total number of shares to execute (``X0``).  Paper: 20000.
        execution_window: length of the execution window.  Paper: ``"00:30:00"``.
        direction: ``"BUY"`` or ``"SELL"``.
        q_min: minimum child order size (``Q_min``).  Paper: 20.
        num_action_levels: number of non-zero order sizes ``k * Q_min``,
            ``k = 1..num_action_levels``.  Paper: 4.
        depth_penalty_alpha: weight ``alpha`` of the depth penalty.  Paper: 2.
        beta_not_enough: terminal penalty per share left unexecuted.  Paper: 5.
        beta_too_much: terminal penalty per share executed beyond ``X0``.  Paper: 5.
        is_weight: weight of the implementation-shortfall term (1 in the paper).
            The original research code used ``(0.5 * IS - 1 * d_t) / X0``, i.e.
            exactly half of the paper's step reward; it is recovered with
            ``is_weight=0.5`` together with ``depth_penalty_alpha=1.0``.
        action_mode: ``"lots"`` (paper: action ``k`` sends ``k * q_min`` shares)
            or ``"schedule"`` (action ``j`` is a participation multiplier of
            the TWAP rate, executed as child orders inside a corridor around
            the TWAP schedule; see the module docstring).
        schedule_band: half-width ``w`` of the corridor in ``action_mode
            ="schedule"``, as a fraction of ``X0``.
        reward_mode: ``"arrival"`` (paper, equation (4)), ``"mark_to_market"``
            (same episode total, Markov in the state) or ``"vs_twap"`` (the
            mark-to-market reward minus the TWAP schedule's own inventory term:
            ``[sum_fills q (m_prev - p) - alpha d_t + D_t (m_t - m_prev)] / X0``
            with ``D_t = x_t - S(t)`` the deviation from the TWAP schedule after
            the fills; it sums to the shortfall difference against a TWAP
            executed at the mid and removes the drift noise that no action
            controls).
        schedule_corridor: ``"flat"`` (corridor of full half-width ``w X0``
            whose lower edge is closed on ``X0`` by a ramp at twice the TWAP
            rate) or ``"linear"`` (both edges converge linearly to ``X0`` at
            ``T``: half-width ``w X0 (1 - t/T)``).
        state_features: optional list of feature names (see the module
            docstring); ``None`` gives the paper's 8-feature vector.
        order_style: ``"market"``, ``"passive"`` or ``"mixed"`` (schedule mode
            only; see the module docstring).
        passive_horizon: extra wake-ups of intended flow (beyond the current
            decision) kept resting as a limit order in the passive styles.
        passive_offset: cents above the best bid at which the limit order is
            placed (never at or above the best ask).
        schedule_penalty: weight ``lambda`` of an optional per-step penalty
            ``lambda * |x_t - X0 * t/T| / X0`` on the deviation of the executed
            quantity from the linear (TWAP) schedule (0 in the paper: no such
            term). Reward shaping for the hyper-parameter search of CHANGES.md
            C10; the evaluation statistics (implementation shortfall, terminal
            penalty) are unaffected, ``episode_reward`` includes the term and
            ``schedule_penalty`` in ``info`` accumulates it.
        action_repeat: number of consecutive wake-ups over which a chosen action
            is repeated before the agent decides again (1 in the paper). With
            ``action_repeat=30`` and 1-second wake-ups the agent decides every
            30 s and action ``k`` sends ``k * Q_min`` shares at each of the 30
            seconds; the reward of a decision is the sum of the 30 step rewards
            and the ``step_*`` entries of ``info`` are accumulated over them.
            Hand-crafted policies must then use ``Q_min * action_repeat`` as
            their lot size and ``N / action_repeat`` decision points
            (``experiments.common.policy_env_params`` does this).
        debug_mode: if True ``info`` also contains raw simulator quantities.
        background_config_extra_kvargs: extra keyword arguments forwarded to the
            background configuration builder, e.g.
            ``{"num_noise_agents": 2000, "num_momentum_agents": 12,
            "mm_wake_up_freq": "1S"}`` (used for the robustness experiments).
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    @dataclass
    class CustomMetricsTracker:
        """Metrics returned in ``info`` at every step (all numeric, RLlib friendly).

        Quantities suffixed ``_normalized`` are divided by the parent order size
        and therefore expressed in cents per share of the parent order.
        """

        # episode accumulators
        episode_reward: float = 0.0
        implementation_shortfall: float = 0.0  # sum_t IS_t / X0
        depth_penalty: float = 0.0  # sum_t alpha * d_t / X0 (positive number)
        schedule_penalty: float = 0.0  # sum_t lambda * |x_t - X0 t/T| / X0 (positive number, 0 unless shaping is on)
        execution_cost: float = 0.0  # sum_t sum_fills q (m_prev - p) / X0 (mark-to-market decomposition, <= 0)
        inventory_pnl: float = 0.0  # sum_t R_t (m_prev - m_t) / X0 (mark-to-market decomposition)
        terminal_penalty: float = 0.0  # -beta * |x_T - X0| / X0 (<= 0)
        executed_quantity: int = 0
        remaining_quantity: int = 0
        num_steps: int = 0
        num_trading_steps: int = 0  # steps with at least one fill
        num_orders: int = 0  # market orders sent (steps with action > 0)
        passive_fill_quantity: int = 0  # shares filled by resting limit orders (passive styles)
        num_limit_orders: int = 0  # limit orders placed
        resting_quantity: int = 0  # shares resting as limit orders at the current wake-up
        resting_price: float = 0.0
        num_unfilled_orders: int = 0  # market orders that were not filled in full
        unfilled_quantity: int = 0  # shares sent that met no resting liquidity (discarded by the exchange)
        # last-step quantities (the original notebooks reported these)
        step_implementation_shortfall: float = 0.0
        step_depth_penalty: float = 0.0
        step_schedule_penalty: float = 0.0
        step_execution_cost: float = 0.0
        step_inventory_pnl: float = 0.0
        step_reward: float = 0.0
        step_executed_quantity: int = 0
        step_avg_fill_price: float = 0.0
        step_order_quantity: int = 0  # shares sent as market orders at this step
        step_limit_quantity: int = 0  # shares placed as a limit order at this step
        step_passive_fill_quantity: int = 0  # limit-order fills received at this step
        step_unfilled_quantity: int = 0  # shares sent at this step that were not filled
        action: int = 0
        # market / agent observables at the current wake-up
        holdings_pct: float = 0.0
        time_pct: float = 0.0
        imbalance_5: float = 0.5
        spread: float = 0.0
        best_bid: float = 0.0
        best_ask: float = 0.0
        mid_price: float = 0.0
        entry_price: float = 0.0
        current_time: int = 0
        num_max_steps_per_episode: float = 0.0
        action_counter: Dict[str, int] = field(default_factory=dict)

    def __init__(
        self,
        background_config: str = "rmsc04",
        mkt_close: str = "16:00:00",
        timestep_duration: str = "1s",
        starting_cash: int = 1_000_000,
        state_history_length: int = 4,
        market_data_buffer_length: int = 50,
        first_interval: str = "00:00:30",
        parent_order_size: int = 20_000,
        execution_window: str = "00:30:00",
        direction: str = "BUY",
        q_min: int = 20,
        num_action_levels: int = 4,
        depth_penalty_alpha: float = 2.0,
        beta_not_enough: float = 5.0,
        beta_too_much: float = 5.0,
        is_weight: float = 1.0,
        schedule_penalty: float = 0.0,
        action_repeat: int = 1,
        action_mode: str = "lots",
        schedule_band: float = 0.2,
        schedule_corridor: str = "flat",
        reward_mode: str = "arrival",
        state_features: Optional[List[str]] = None,
        order_style: str = "market",
        passive_horizon: int = 0,
        passive_offset: int = 0,
        debug_mode: bool = False,
        background_config_extra_kvargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.background_config_name: str = background_config
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.parent_order_size: int = parent_order_size
        self.execution_window: NanosecondTime = str_to_ns(execution_window)
        self.direction: str = direction
        self.q_min: int = q_min
        self.num_action_levels: int = num_action_levels
        self.depth_penalty_alpha: float = float(depth_penalty_alpha)
        self.beta_not_enough: float = float(beta_not_enough)
        self.beta_too_much: float = float(beta_too_much)
        self.is_weight: float = float(is_weight)
        self.schedule_penalty: float = float(schedule_penalty)
        assert self.schedule_penalty >= 0, "schedule_penalty must be non-negative"
        self.action_repeat: int = int(action_repeat)
        assert self.action_repeat >= 1, "action_repeat must be a positive integer"
        self.action_mode: str = action_mode
        assert self.action_mode in ("lots", "schedule"), "action_mode must be 'lots' or 'schedule'"
        self.schedule_band: float = float(schedule_band)
        assert 0.0 < self.schedule_band <= 1.0, "schedule_band must be in (0, 1]"
        self.schedule_corridor: str = schedule_corridor
        assert self.schedule_corridor in ("flat", "linear"), "schedule_corridor must be 'flat' or 'linear'"
        self.reward_mode: str = reward_mode
        assert self.reward_mode in ("arrival", "mark_to_market", "vs_twap"), "reward_mode must be 'arrival', 'mark_to_market' or 'vs_twap'"
        self.state_features: Optional[List[str]] = list(state_features) if state_features else None
        self.order_style: str = order_style
        assert self.order_style in ("market", "passive", "mixed"), "order_style must be 'market', 'passive' or 'mixed'"
        assert self.order_style == "market" or self.action_mode == "schedule", "passive/mixed order styles need action_mode='schedule'"
        assert self.order_style != "mixed" or num_action_levels == 4, "order_style='mixed' needs num_action_levels=4 (five actions)"
        self.passive_horizon: int = int(passive_horizon)
        self.passive_offset: int = int(passive_offset)
        assert self.passive_horizon >= 0 and self.passive_offset >= 0
        self.debug_mode: bool = debug_mode
        self.background_config_extra_kvargs: Dict[str, Any] = dict(
            background_config_extra_kvargs or {}
        )

        # ------------------------------------------------------------------
        # parameter validation
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "rmsc05",
        ], "Select rmsc03, rmsc04 or rmsc05 as background_config"
        assert (
            str_to_ns("00:00:00") <= self.first_interval <= str_to_ns("06:30:00")
        ), "Select authorized first_interval delay"
        assert (
            str_to_ns("09:30:00") <= self.mkt_close <= str_to_ns("16:00:00")
        ), "Select authorized market hours"
        assert (
            str_to_ns("00:00:00") < self.timestep_duration <= str_to_ns("06:30:00")
        ), "Select authorized timestep_duration"
        assert (
            str_to_ns("00:00:00") < self.execution_window <= str_to_ns("06:30:00")
        ), "Select authorized execution_window"
        assert (
            self.first_interval + self.execution_window <= str_to_ns("06:30:00")
        ), "first_interval + execution_window must fit in the trading day"
        assert isinstance(self.starting_cash, int) and self.starting_cash >= 0
        assert isinstance(self.state_history_length, int) and self.state_history_length >= 2
        assert isinstance(self.market_data_buffer_length, int) and self.market_data_buffer_length >= 2
        assert isinstance(self.parent_order_size, int) and self.parent_order_size > 0
        assert isinstance(self.q_min, int) and self.q_min > 0
        assert isinstance(self.num_action_levels, int) and self.num_action_levels >= 1
        assert self.depth_penalty_alpha >= 0 and self.beta_not_enough >= 0 and self.beta_too_much >= 0
        assert self.direction in ["BUY", "SELL"], "direction needs to be BUY or SELL"
        assert isinstance(self.debug_mode, bool)

        background_config_args = {"end_time": mkt_close}
        background_config_args.update(self.background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        # ------------------------------------------------------------------
        # action space: 0 = do nothing, k = market order of k * q_min shares
        self.num_actions: int = 1 + self.num_action_levels
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # ------------------------------------------------------------------
        # observation space
        paper_features = ["holdings_pct", "time_pct", "imbalance_5", "best_bid", "best_ask"] + [
            f"mid_return_lag{i}" for i in range(self.state_history_length - 1, 0, -1)
        ]
        self.state_feature_names: List[str] = list(self.state_features) if self.state_features else paper_features
        for name in self.state_feature_names:
            self._feature_bounds(name)  # validates the name
        self.num_state_features: int = len(self.state_feature_names)
        bounds = [self._feature_bounds(n) for n in self.state_feature_names]
        self.state_lows: np.ndarray = np.array([b[0] for b in bounds], dtype=np.float32)
        self.state_highs: np.ndarray = np.array([b[1] for b in bounds], dtype=np.float32)
        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features,),
            dtype=np.float32,
        )
        self._history_length: int = max(
            [self.state_history_length] + [self._feature_window(n) + 1 for n in self.state_feature_names]
        )

        self.num_max_steps_per_episode: float = (
            self.execution_window / self.timestep_duration
        )
        # schedule mode: TWAP rate per wake-up and participation multipliers
        self.twap_rate: float = self.parent_order_size / self.num_max_steps_per_episode
        self.action_multipliers: List[float] = [
            j / (self.num_action_levels / 2.0) for j in range(self.num_actions)
        ]
        self.max_child_order: int = int(math.ceil(2.0 * self.action_multipliers[-1] * self.twap_rate))
        # mixed style: (multiplier, style) per action
        self.mixed_actions: List[tuple] = [(0.0, "passive"), (1.0, "passive"), (2.0, "passive"), (1.0, "market"), (2.0, "market")]
        self._reset_episode_variables()

    # ----------------------------------------------------------------------
    # state features
    _PRICE_BOUND = 1e8  # cents; prices in rmsc0x are of the order of 1e5

    @classmethod
    def _feature_window(cls, name: str) -> int:
        """Number of past wake-ups a feature needs (0 for instantaneous ones)."""
        for prefix in ("ret_", "imb_mean_", "dev_"):
            if name.startswith(prefix):
                return int(name[len(prefix):])
        if name.startswith("mid_return_lag"):
            return int(name[len("mid_return_lag"):])
        return 0

    @classmethod
    def _feature_bounds(cls, name: str):
        pb = cls._PRICE_BOUND
        if name in ("holdings_pct", "time_pct", "schedule_dev", "resting_pct"):
            return (-2.0, 2.0)
        if name in ("imbalance_5", "imbalance_1") or name.startswith("imb_mean_"):
            return (0.0, 1.0)
        if name in ("best_bid", "best_ask", "spread", "mid_minus_p0") or name.startswith(("mid_return_lag", "ret_", "dev_")):
            return (-pb, pb)
        raise ValueError(f"unknown state feature {name!r}")

    # ----------------------------------------------------------------------
    # episode bookkeeping
    def _reset_episode_variables(self) -> None:
        """Reset everything that must not leak from one episode to the next."""
        self.step_index: int = 0
        self.entry_price: Optional[float] = None  # arrival price P0
        self.last_action: int = 0
        self.pnl: float = 0.0
        self.reward: float = 0.0
        self._hist_mid: List[float] = []  # mid price at every wake-up of the episode
        self._hist_imb: List[float] = []  # 5-level imbalance at every wake-up
        self._decision_executed: int = 0  # executed quantity at the start of the current decision
        self._decision_step: int = 0  # wake-up index at the start of the current decision
        self._child_index: int = 0  # child step within the current decision
        self.custom_metrics_tracker = self.CustomMetricsTracker(
            action_counter={f"action_{i}": 0 for i in range(self.num_actions)},
            num_max_steps_per_episode=self.num_max_steps_per_episode,
        )

    def reset(self):
        """Start a new episode (new ABIDES simulation) and return the first state."""
        self._reset_episode_variables()
        return super().reset()

    def step(self, action: int):
        """One decision: ``action_repeat`` consecutive wake-ups with the same action.

        With the default ``action_repeat=1`` this is the plain Gym step. Otherwise
        the rewards of the repeated steps are summed and the ``step_*``
        diagnostics of ``info`` accumulated, so that a decision looks like one
        (longer) step to the learner and to the evaluation logs.
        """
        self._begin_decision()
        if self.action_repeat == 1:
            return super().step(action)
        total_reward = 0.0
        qty, notional, sent, unfilled, is_sum, depth_sum = 0, 0.0, 0, 0, 0.0, 0.0
        exec_sum, inv_sum, limit_sum, passive_sum = 0.0, 0.0, 0, 0
        for child in range(self.action_repeat):
            self._child_index = child
            state, reward, done, info = super().step(action)
            total_reward += reward
            qty += info["step_executed_quantity"]
            notional += info["step_executed_quantity"] * info["step_avg_fill_price"]
            sent += info["step_order_quantity"]
            unfilled += info["step_unfilled_quantity"]
            is_sum += info["step_implementation_shortfall"]
            depth_sum += info["step_depth_penalty"]
            exec_sum += info["step_execution_cost"]
            inv_sum += info["step_inventory_pnl"]
            limit_sum += info["step_limit_quantity"]
            passive_sum += info["step_passive_fill_quantity"]
            if done:
                break
        info = dict(info)
        info.update(
            step_executed_quantity=qty,
            step_avg_fill_price=notional / qty if qty > 0 else 0.0,
            step_order_quantity=sent,
            step_unfilled_quantity=unfilled,
            step_implementation_shortfall=is_sum,
            step_depth_penalty=depth_sum,
            step_execution_cost=exec_sum,
            step_inventory_pnl=inv_sum,
            step_limit_quantity=limit_sum,
            step_passive_fill_quantity=passive_sum,
            step_reward=total_reward,
        )
        return state, total_reward, done, info

    def _begin_decision(self) -> None:
        """Record the executed quantity and wake-up index at the start of a decision."""
        tracker = self.custom_metrics_tracker
        self._decision_executed = int(tracker.executed_quantity)
        self._decision_step = int(round(tracker.time_pct * self.num_max_steps_per_episode))
        self._child_index = 0

    # ----------------------------------------------------------------------
    # schedule mode: corridor around the TWAP schedule
    def schedule_bounds(self, k: float):
        """Lower and upper bound of the executed quantity ``k`` wake-ups into the window.

        ``S(k) = X0 k / N``; lower ``max(0, S - w X0, X0 - 2 r (N - k))`` (the
        closing ramp guarantees completion at ``N`` at twice the TWAP rate);
        upper ``min(X0, S + w X0)``.
        """
        x0, n, r, w = self.parent_order_size, self.num_max_steps_per_episode, self.twap_rate, self.schedule_band
        s = x0 * k / n
        if self.schedule_corridor == "linear":
            half = w * x0 * (1.0 - k / n)
            lower, upper = max(0.0, s - half), min(float(x0), s + half)
        else:
            m_max = self.action_multipliers[-1]
            lower = max(0.0, s - w * x0, x0 - m_max * r * (n - k))
            upper = min(float(x0), s + w * x0)
        return lower, max(lower, upper)

    def _schedule_child_order(self, action: int) -> int:
        """Shares to send at this wake-up so that the executed quantity follows
        ``x_dec + m_j r (child + 1)`` clipped to the corridor at the next wake-up."""
        m = self.action_multipliers[action]
        return self._child_order_for_multiplier(m)

    def _child_order_for_multiplier(self, m: float) -> int:
        k_next = self._decision_step + self._child_index + 1
        intended = self._decision_executed + m * self.twap_rate * (self._child_index + 1)
        lower, upper = self.schedule_bounds(min(k_next, self.num_max_steps_per_episode))
        target = min(max(intended, lower), upper)
        executed = int(self.custom_metrics_tracker.executed_quantity)
        order = int(round(target - executed))
        return int(min(max(order, 0), self.max_child_order))

    def _schedule_actions(self, action: int) -> List[Dict[str, Any]]:
        """ABIDES actions of one wake-up in schedule mode for the market, passive and mixed styles."""
        tracker = self.custom_metrics_tracker
        tracker.step_limit_quantity = 0
        if self.order_style == "market":
            size = self._schedule_child_order(action)
            tracker.step_order_quantity = size
            if size > 0:
                tracker.num_orders += 1
                return [{"type": "MKT", "direction": self.direction, "size": size}]
            return []
        m, style = self.mixed_actions[action] if self.order_style == "mixed" else (self.action_multipliers[action], "passive")
        resting = int(tracker.resting_quantity)
        actions: List[Dict[str, Any]] = []
        if style == "market":
            size = self._child_order_for_multiplier(m)
            tracker.step_order_quantity = size
            if resting > 0:
                actions.append({"type": "CCL_ALL"})
            if size > 0:
                tracker.num_orders += 1
                actions.append({"type": "MKT", "direction": self.direction, "size": size})
            return actions
        # passive: market order only for the part behind the corridor's lower edge
        executed = int(tracker.executed_quantity)
        n = self.num_max_steps_per_episode
        k_next = self._decision_step + self._child_index + 1
        lower, _ = self.schedule_bounds(min(k_next, n))
        forced = int(min(max(int(math.ceil(lower - executed)), 0), self.max_child_order))
        tracker.step_order_quantity = forced
        if forced > 0:
            tracker.num_orders += 1
            actions.append({"type": "MKT", "direction": self.direction, "size": forced})
        # resting size: what is still missing to reach the decision's intended quantity
        # (m x TWAP rate over the decision plus ``passive_horizon`` extra wake-ups), inside
        # the corridor's upper edge; unfilled parts carry over as a deviation from the schedule
        h_end = self.action_repeat + self.passive_horizon
        k_h = min(n, self._decision_step + h_end)
        _, upper_h = self.schedule_bounds(k_h)
        intended_h = self._decision_executed + m * self.twap_rate * h_end
        size = int(round(min(intended_h, upper_h) - executed - forced))
        size = int(min(max(size, 0), math.ceil(self.action_multipliers[-1] * self.twap_rate * h_end)))
        if size <= 0:
            if resting > 0:
                actions.append({"type": "CCL_ALL"})
            return actions
        best_bid, best_ask = tracker.best_bid, tracker.best_ask
        if self.direction == "BUY":
            price = int(min(best_bid + self.passive_offset, max(best_bid, best_ask - 1)))
        else:
            price = int(max(best_ask - self.passive_offset, min(best_ask, best_bid + 1)))
        if resting > 0 and int(tracker.resting_price) == price and 0.7 * size <= resting <= 1.3 * size:
            return actions  # keep the resting order and its queue priority
        if resting > 0:
            actions.append({"type": "CCL_ALL"})
        tracker.step_limit_quantity = size
        tracker.num_limit_orders += 1
        actions.append({"type": "LMT", "direction": self.direction, "size": size, "limit_price": price})
        return actions

    # ----------------------------------------------------------------------
    # helpers
    def _signed(self, holdings: int) -> int:
        """Executed quantity, positive for both buy and sell parent orders."""
        return holdings if self.direction == "BUY" else -holdings

    def _time_pct(self, raw_internal: Dict[str, Any], last: bool) -> float:
        mkt_open = raw_internal["mkt_open"][-1] if last else raw_internal["mkt_open"]
        current_time = (
            raw_internal["current_time"][-1] if last else raw_internal["current_time"]
        )
        assert (
            current_time >= mkt_open + self.first_interval
        ), "Agent has woken up earlier than its first interval"
        return (current_time - mkt_open - self.first_interval) / self.execution_window

    # ----------------------------------------------------------------------
    # gym <-> abides action mapping
    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        """Map a Gym action to the ABIDES agent API.

        - ``0``: do nothing;
        - ``k`` in ``1..num_action_levels``: market order of ``k * q_min`` shares.
        """
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(
                f"Action {action} is not part of the actions supported by the function."
            )
        self.last_action = action
        self.custom_metrics_tracker.action_counter[f"action_{action}"] += 1
        self.custom_metrics_tracker.action = action
        if self.action_mode == "schedule":
            return self._schedule_actions(action)
        size = action * self.q_min
        self.custom_metrics_tracker.step_order_quantity = size
        if size > 0:
            self.custom_metrics_tracker.num_orders += 1
        if size == 0:
            return []
        return [
            {
                "type": "MKT",
                "direction": self.direction,
                "size": size,
            }
        ]

    # ----------------------------------------------------------------------
    # state
    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Build the state vector from the buffered raw simulator state.

        After the pre-processing decorator every entry of ``raw_state`` is a
        list indexed by wake-up (oldest first, at most ``state_history_length``
        entries) holding the *latest* L2 snapshot received before that wake-up.
        """
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) holdings
        holdings = raw_state["internal_data"]["holdings"][-1]
        holdings_pct = self._signed(holdings) / self.parent_order_size

        # 2) time
        time_pct = self._time_pct(raw_state["internal_data"], last=True)

        # 3) imbalance over the best 5 levels (bid share of the volume)
        imbalance_5 = markets_agent_utils.get_imbalance(bids[-1], asks[-1], depth=5)

        # 4) prices
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]
        if self.step_index == 0 or self.entry_price is None:
            self.entry_price = mid_price  # arrival price P0
        best_bid = bids[-1][0][0] if len(bids[-1]) > 0 else mid_price
        best_ask = asks[-1][0][0] if len(asks[-1]) > 0 else mid_price

        # 5) lagged mid-price changes, zero padded at the start of the episode
        returns = np.diff(mid_prices)
        padded_returns = np.zeros(self.state_history_length - 1)
        if len(returns) > 0:
            padded_returns[-len(returns) :] = returns[-(self.state_history_length - 1) :]

        # per-episode history for the window features (ret_k, imb_mean_k, dev_k)
        self._hist_mid.append(float(mid_price))
        self._hist_imb.append(float(imbalance_5))
        if len(self._hist_mid) > self._history_length:
            del self._hist_mid[: len(self._hist_mid) - self._history_length]
            del self._hist_imb[: len(self._hist_imb) - self._history_length]

        # resting limit orders (passive styles): active entries of the agent's order table
        resting_qty, resting_price = 0, 0.0
        status_hist = raw_state["internal_data"].get("order_status")
        status = status_hist[-1] if isinstance(status_hist, list) and status_hist else (status_hist or {})
        for entry in status.values():
            if isinstance(entry, dict) and entry.get("status") == "active" and entry.get("active_qty", 0) > 0:
                order = entry.get("order")
                if isinstance(order, LimitOrder):
                    resting_qty += int(entry["active_qty"])
                    resting_price = float(order.limit_price)

        # log observables
        tracker = self.custom_metrics_tracker
        tracker.resting_quantity = resting_qty
        tracker.resting_price = resting_price
        tracker.holdings_pct = holdings_pct
        tracker.time_pct = time_pct
        tracker.imbalance_5 = imbalance_5
        tracker.spread = best_ask - best_bid
        tracker.best_bid = best_bid
        tracker.best_ask = best_ask
        tracker.mid_price = mid_price
        tracker.entry_price = self.entry_price
        tracker.current_time = int(raw_state["internal_data"]["current_time"][-1])
        tracker.num_steps = self.step_index

        if self.state_features is None:
            computed_state = np.array(
                [holdings_pct, time_pct, imbalance_5, best_bid, best_ask]
                + padded_returns.tolist(),
                dtype=np.float32,
            )
        else:
            values = {
                "holdings_pct": holdings_pct,
                "time_pct": time_pct,
                "imbalance_5": imbalance_5,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": best_ask - best_bid,
                "schedule_dev": holdings_pct - min(1.0, time_pct),
                "imbalance_1": markets_agent_utils.get_imbalance(bids[-1], asks[-1], depth=1),
                "mid_minus_p0": mid_price - self.entry_price,
                "resting_pct": resting_qty / self.parent_order_size,
            }
            for i in range(1, self.state_history_length):
                values[f"mid_return_lag{i}"] = float(padded_returns[-i])
            computed_state = np.array(
                [values[n] if n in values else self._window_feature(n) for n in self.state_feature_names],
                dtype=np.float32,
            )
        self.step_index += 1
        return computed_state

    def _window_feature(self, name: str) -> float:
        """``ret_k``, ``imb_mean_k`` or ``dev_k`` from the per-episode history
        (the current wake-up is the last entry; fewer than ``k`` past wake-ups
        use what is available, zero at the first wake-up)."""
        k = self._feature_window(name)
        mids, imbs = self._hist_mid, self._hist_imb
        if name.startswith("ret_"):
            return mids[-1] - mids[max(0, len(mids) - 1 - k)]
        if name.startswith("imb_mean_"):
            window = imbs[-k:] if k > 0 else imbs[-1:]
            return float(np.mean(window))
        if name.startswith("dev_"):
            window = mids[-k:] if k > 0 else mids[-1:]
            return mids[-1] - float(np.mean(window))
        raise ValueError(name)

    # ----------------------------------------------------------------------
    # reward
    @raw_state_to_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """Step reward: implementation shortfall minus depth penalty, over X0."""
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        fills = raw_state["internal_data"]["inter_wakeup_executed_orders"][-1]
        entry_price = self.entry_price

        # best quote on the far side at the previous wake-up (pre-trade touch)
        far_side = asks if self.direction == "BUY" else bids
        pre_trade_touch = None
        if len(far_side) >= 2 and len(far_side[-2]) > 0:
            pre_trade_touch = far_side[-2][0][0]

        step_is = 0.0
        step_depth = 0.0
        step_qty = 0
        step_notional = 0.0
        step_passive = 0
        for order in fills:
            q = order.quantity
            p = order.fill_price
            step_qty += q
            step_notional += q * p
            if isinstance(order, LimitOrder):
                step_passive += q
            if self.direction == "BUY":
                step_is += q * (entry_price - p)
                if pre_trade_touch is not None:
                    step_depth += max(0.0, p - pre_trade_touch)
            else:
                step_is += q * (p - entry_price)
                if pre_trade_touch is not None:
                    step_depth += max(0.0, pre_trade_touch - p)

        # mark-to-market decomposition of the same objective (module docstring)
        mids = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, raw_state["parsed_mkt_data"]["last_transaction"])
        ]
        mid_now = mids[-1]
        mid_prev = mids[-2] if len(mids) >= 2 else mid_now
        sign = 1.0 if self.direction == "BUY" else -1.0
        step_exec_cost = sum(
            o.quantity * sign * (mid_prev - o.fill_price) for o in fills
        )  # <= 0 when paying the spread
        executed_after = self._signed(raw_state["internal_data"]["holdings"][-1])
        remaining_after = max(0, self.parent_order_size - executed_after)
        step_inventory = remaining_after * sign * (mid_prev - mid_now)

        if self.reward_mode == "mark_to_market":
            reward = (
                self.is_weight * (step_exec_cost + step_inventory) - self.depth_penalty_alpha * step_depth
            ) / self.parent_order_size
        elif self.reward_mode == "vs_twap":
            # inventory term relative to the TWAP schedule: D_t (m_t - m_prev), D_t = x_t - X0 t/T
            t_pct = min(1.0, self._time_pct(raw_state["internal_data"], last=True))
            deviation = executed_after - self.parent_order_size * t_pct
            step_vs_twap = deviation * sign * (mid_now - mid_prev)
            reward = (
                self.is_weight * (step_exec_cost + step_vs_twap) - self.depth_penalty_alpha * step_depth
            ) / self.parent_order_size
        else:
            reward = (
                self.is_weight * step_is - self.depth_penalty_alpha * step_depth
            ) / self.parent_order_size

        # optional shaping: distance to the linear schedule after this step's fills
        step_schedule = 0.0
        if self.schedule_penalty > 0:
            executed_now = self._signed(raw_state["internal_data"]["holdings"][-1])
            t_pct = min(1.0, self._time_pct(raw_state["internal_data"], last=True))
            deviation = abs(executed_now - self.parent_order_size * t_pct)
            step_schedule = self.schedule_penalty * deviation / self.parent_order_size
            reward -= step_schedule

        self.pnl = step_is
        self.reward = reward
        tracker = self.custom_metrics_tracker
        tracker.step_schedule_penalty = step_schedule
        tracker.schedule_penalty += step_schedule
        tracker.step_execution_cost = step_exec_cost / self.parent_order_size
        tracker.step_inventory_pnl = step_inventory / self.parent_order_size
        tracker.execution_cost += tracker.step_execution_cost
        tracker.inventory_pnl += tracker.step_inventory_pnl
        tracker.step_implementation_shortfall = step_is / self.parent_order_size
        tracker.step_depth_penalty = (
            self.depth_penalty_alpha * step_depth / self.parent_order_size
        )
        tracker.step_reward = reward
        tracker.step_executed_quantity = step_qty
        tracker.step_avg_fill_price = step_notional / step_qty if step_qty > 0 else 0.0
        # A market order never rests: the part that met no liquidity was discarded
        # by the exchange without any message.  Account for it here.
        tracker.step_passive_fill_quantity = step_passive
        tracker.passive_fill_quantity += step_passive
        tracker.step_unfilled_quantity = max(0, tracker.step_order_quantity - (step_qty - step_passive))
        if tracker.step_unfilled_quantity > 0:
            tracker.unfilled_quantity += tracker.step_unfilled_quantity
            tracker.num_unfilled_orders += 1
        tracker.implementation_shortfall += tracker.step_implementation_shortfall
        tracker.depth_penalty += tracker.step_depth_penalty
        tracker.episode_reward += reward
        if step_qty > 0:
            tracker.num_trading_steps += 1
        return reward

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """Terminal penalty added to the last reward of the episode."""
        executed = self._signed(raw_state["internal_data"]["holdings"])
        diff = executed - self.parent_order_size
        if diff < 0:
            update_reward = -self.beta_not_enough * abs(diff)
        elif diff > 0:
            update_reward = -self.beta_too_much * diff
        else:
            update_reward = 0.0
        update_reward = update_reward / self.parent_order_size
        tracker = self.custom_metrics_tracker
        tracker.terminal_penalty = update_reward
        tracker.episode_reward += update_reward
        tracker.step_reward += update_reward
        return update_reward

    # ----------------------------------------------------------------------
    # termination
    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """Episode ends when the parent order is complete or the window is over."""
        executed = self._signed(raw_state["internal_data"]["holdings"])
        current_time = raw_state["internal_data"]["current_time"]
        mkt_open = raw_state["internal_data"]["mkt_open"]
        time_limit = mkt_open + self.first_interval + self.execution_window

        tracker = self.custom_metrics_tracker
        tracker.executed_quantity = int(executed)
        tracker.remaining_quantity = int(self.parent_order_size - executed)

        if executed >= self.parent_order_size:
            return True  # parent order executed
        if current_time >= time_limit:
            return True  # execution window over
        return False

    # ----------------------------------------------------------------------
    # info
    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnostics (never used by the agent)."""
        # shallow copy (dataclasses.asdict recursed into every field and was a
        # measurable share of the step time); action_counter is the only container
        info = dict(vars(self.custom_metrics_tracker))
        info["action_counter"] = dict(info["action_counter"])
        if self.debug_mode:
            bids = raw_state["parsed_mkt_data"]["bids"]
            asks = raw_state["parsed_mkt_data"]["asks"]
            info.update(
                {
                    "last_transaction": raw_state["parsed_mkt_data"]["last_transaction"],
                    "raw_best_bid": bids[0][0] if len(bids) > 0 else None,
                    "raw_best_ask": asks[0][0] if len(asks) > 0 else None,
                    "holdings": raw_state["internal_data"]["holdings"],
                    "cash": raw_state["internal_data"]["cash"],
                    "parent_size": self.parent_order_size,
                    "pnl": self.pnl,
                    "reward": self.reward,
                }
            )
        return info
