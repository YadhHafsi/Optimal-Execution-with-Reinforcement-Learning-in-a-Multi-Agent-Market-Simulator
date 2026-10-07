"""Tests of the market-liquidity fixes (CHANGES.md, C9).

* the adaptive market makers keep waking up when their first spread query finds
  an empty or one-sided book (upstream ABIDES silenced them for the day);
* the order book stamps new resting orders and trades as updates, so the L2
  feed of the execution agent refreshes on every change of the book;
* the execution environment reports market orders that were not filled in
  full (the exchange discards them silently);
* TWAP executes the parent order exactly, with no discarded order.
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from abides_markets.agents.market_makers.adaptive_market_maker_agent import AdaptiveMarketMakerAgent  # noqa: E402
from abides_markets.order_book import OrderBook  # noqa: E402
from abides_markets.orders import LimitOrder, MarketOrder, Side  # noqa: E402

from test_execution_env import _Fill, _raw_state, make_env  # noqa: E402


class _Owner:
    """Minimal stand-in for the exchange agent that owns an order book."""

    def __init__(self):
        self.current_time = 1_000
        self.mkt_open = 1_000
        self.book_logging = None
        self.book_log_depth = 10
        self.stream_history = 10
        self.oracle = None
        self.random_state = np.random.RandomState(0)

    def send_message(self, *args, **kwargs):
        pass

    def logEvent(self, *args, **kwargs):
        pass


def test_book_timestamp_advances_on_resting_orders_and_trades():
    owner = _Owner()
    book = OrderBook(owner, "ABM")
    assert book.last_update_ts == 1_000

    owner.current_time = 2_000
    book.handle_limit_order(LimitOrder(1, 2_000, "ABM", 100, Side.ASK, 100_010))
    assert book.last_update_ts == 2_000, "a new resting order is a book update"

    owner.current_time = 3_000
    book.handle_market_order(MarketOrder(2, 3_000, "ABM", 40, Side.BID))
    assert book.last_update_ts == 3_000, "a trade is a book update"
    assert book.asks[0].total_quantity == 60

    owner.current_time = 4_000
    book.handle_market_order(MarketOrder(2, 4_000, "ABM", 40, Side.ASK))  # bid side empty
    assert book.last_update_ts == 3_000, "a discarded market order changes nothing"
    assert not book.bids


def test_market_order_remainder_is_discarded_not_rested():
    owner = _Owner()
    book = OrderBook(owner, "ABM")
    book.handle_limit_order(LimitOrder(1, 1_000, "ABM", 30, Side.ASK, 100_010))
    book.handle_market_order(MarketOrder(2, 1_000, "ABM", 80, Side.BID))
    assert not book.asks and not book.bids, "50 unfilled shares must not rest on the book"


def test_env_reports_unfilled_market_orders():
    env = make_env(parent_order_size=2000)
    env.entry_price = 100_000.0
    env.step_index = 1
    assert env._map_action_space_to_ABIDES_SIMULATOR_SPACE(2) == [{"type": "MKT", "direction": "BUY", "size": 40}]
    t = env.custom_metrics_tracker
    assert t.step_order_quantity == 40 and t.num_orders == 1
    raw = _raw_state(
        bids_prev=[(99_998, 50)], asks_prev=[(100_002, 20)],
        bids_now=[(99_998, 50)], asks_now=[],
        fills=[_Fill(20, 100_002)], holdings=20,
    )
    env.raw_state_to_reward(raw)
    assert t.step_executed_quantity == 20
    assert t.step_unfilled_quantity == 20
    assert t.unfilled_quantity == 20 and t.num_unfilled_orders == 1
    # a fully filled order adds nothing
    env._map_action_space_to_ABIDES_SIMULATOR_SPACE(1)
    env.raw_state_to_reward(_raw_state([(99_998, 50)], [(100_002, 20)], [(99_998, 50)], [(100_002, 0)],
                                       fills=[_Fill(20, 100_002)], holdings=40))
    assert t.step_unfilled_quantity == 0 and t.unfilled_quantity == 20 and t.num_orders == 2
    # doing nothing sends nothing
    env._map_action_space_to_ABIDES_SIMULATOR_SPACE(0)
    assert t.step_order_quantity == 0 and t.num_orders == 2


def _market_makers(env):
    return [a for a in env.kernel.agents if isinstance(a, AdaptiveMarketMakerAgent)]


@pytest.mark.parametrize("seed", [1001, 1008])
def test_market_makers_quote_after_an_empty_book_at_their_first_wakeup(seed):
    """With a 1 s wake-up the first spread query lands on an empty book; the
    market makers must nevertheless be quoting one minute after the open."""
    env = make_env(first_interval="00:01:00", execution_window="00:00:30", parent_order_size=200,
                   background_config_extra_kvargs={"mm_wake_up_freq": "1S"})
    env.seed(seed)
    env.reset()
    mms = _market_makers(env)
    assert len(mms) == 2
    for mm in mms:
        assert mm.last_mid is not None, "market maker never observed a mid price"
        assert len(mm.orders) > 0, "market maker has no order on the book"


@pytest.mark.parametrize("wake_up_freq", ["60S", "1S"])
def test_twap_executes_the_parent_order_exactly_with_no_discarded_order(wake_up_freq):
    from experiments.policies import TWAPPolicy

    env = make_env(parent_order_size=2000, execution_window="00:02:00",
                   background_config_extra_kvargs={"mm_wake_up_freq": wake_up_freq})
    policy = TWAPPolicy(parent_order_size=2000, num_steps=120, q_min=20, num_action_levels=4)
    env.seed(1001)
    policy.reset(1001)
    state = env.reset()
    done = False
    while not done:
        state, _, done, info = env.step(policy.get_action(state))
    assert info["unfilled_quantity"] == 0 and info["num_unfilled_orders"] == 0
    assert info["executed_quantity"] == 2000 and info["remaining_quantity"] == 0
    assert info["terminal_penalty"] == 0.0
    mms = _market_makers(env)
    assert all(mm.last_mid is not None for mm in mms)


def test_action_repeat_sums_rewards_and_accumulates_fills():
    """A decision with action_repeat=3 equals three 1 s steps with the same action."""
    from experiments.policies import TWAPPolicy
    from experiments.common import policy_env_params

    kw = dict(parent_order_size=2000, execution_window="00:02:00", first_interval="00:05:00",
              background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    env1 = make_env(**kw)
    env3 = make_env(action_repeat=3, **kw)
    seq = [1, 0, 2, 0, 1, 1]  # decisions of the repeated environment
    env1.seed(1002); s1 = env1.reset(); env3.seed(1002); s3 = env3.reset()
    np.testing.assert_array_equal(s1, s3)
    for a in seq:
        r1 = 0.0; q1 = 0; sent1 = 0
        for _ in range(3):
            s1, r, d1, i1 = env1.step(a)
            r1 += r; q1 += i1["step_executed_quantity"]; sent1 += i1["step_order_quantity"]
        s3, r3, d3, i3 = env3.step(a)
        np.testing.assert_array_equal(s1, s3)
        assert r3 == pytest.approx(r1)
        assert i3["step_executed_quantity"] == q1 and i3["step_order_quantity"] == sent1 == a * 20 * 3
        assert i3["executed_quantity"] == i1["executed_quantity"] and d1 == d3
    # hand-crafted policies see 3 x Q_min per decision and N / 3 decisions
    env_kwargs = dict(kw, action_repeat=3, q_min=20, num_action_levels=4, timestep_duration="1s")
    p = policy_env_params(env_kwargs)
    assert p["q_min"] == 60 and p["num_steps"] == 40
    twap = TWAPPolicy(p["parent_order_size"], p["num_steps"], p["q_min"], p["num_action_levels"])
    twap.reset(1002); s = env3.reset(); done = False
    while not done:
        s, _, done, info = env3.step(twap.get_action(s))
    assert info["executed_quantity"] == 2000 and info["unfilled_quantity"] == 0


def test_schedule_mode_completes_exactly_and_reward_telescopes():
    """Schedule mode (CHANGES.md C11): every policy completes exactly; the
    mark-to-market reward sums to the arrival-price shortfall minus the depth penalty."""
    from experiments.common import policy_env_params
    from experiments.policies import make_policy

    kw = dict(parent_order_size=2000, execution_window="00:02:00", first_interval="00:05:00",
              action_mode="schedule", action_repeat=10, schedule_band=0.2, reward_mode="mark_to_market",
              state_features=["schedule_dev", "time_pct", "imbalance_5", "imb_mean_10", "spread", "ret_10", "dev_30"],
              background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    env = make_env(**kw)
    assert env.observation_space.shape == (7,)
    assert env.action_multipliers == [0.0, 0.5, 1.0, 1.5, 2.0]
    lo, hi = env.schedule_bounds(120); assert lo == hi == 2000  # corridor closes on X0 at T
    lo, hi = env.schedule_bounds(60); assert lo == pytest.approx(600) and hi == pytest.approx(1400)
    params = policy_env_params(dict(kw, timestep_duration="1s", q_min=20, num_action_levels=4))
    assert params["twap_action"] == 2 and params["num_steps"] == 12
    for policy_name, constant in [("TWAP", None), ("back", 0), ("front", 4)]:
        pol = make_policy("TWAP", params, seed=1003) if constant is None else None
        env.seed(1003); s = env.reset(); done = False; total = 0.0; n = 0
        while not done:
            a = pol.get_action(s) if pol else constant
            s, r, done, info = env.step(a); total += r; n += 1
        assert info["executed_quantity"] == 2000 and info["remaining_quantity"] == 0, policy_name
        assert info["unfilled_quantity"] == 0 and info["terminal_penalty"] == 0.0
        assert total == pytest.approx(info["episode_reward"])
        assert info["execution_cost"] + info["inventory_pnl"] == pytest.approx(info["implementation_shortfall"], abs=1e-6)
        assert info["episode_reward"] == pytest.approx(info["implementation_shortfall"] - info["depth_penalty"], abs=1e-6)
        if constant is None:
            assert n == 12 and info["time_pct"] == pytest.approx(1.0)
        if constant == 4:
            assert info["time_pct"] < 1.0  # front-loaded: done before the end


def test_window_features_and_bounds():
    env = make_env(parent_order_size=2000, execution_window="00:02:00",
                   state_features=["holdings_pct", "ret_3", "imb_mean_2", "dev_3", "spread", "mid_minus_p0"])
    env._reset_episode_variables()
    env.step_index = 0
    # feed three synthetic wake-ups through the state function
    states = []
    for mid_shift, imb in [(0, (50, 50)), (2, (80, 20)), (-1, (50, 150))]:
        bid = 99_999 + mid_shift; ask = 100_001 + mid_shift
        raw = _raw_state([(bid, imb[0])], [(ask, imb[1])], [(bid, imb[0])], [(ask, imb[1])], fills=[], holdings=0)
        states.append(env.raw_state_to_state(raw))
    s = states[-1]
    # mids: 100000, 100002, 99999 -> ret_3 = 99999 - 100000 = -1; dev_3 = 99999 - mean = -1.333
    assert s[1] == pytest.approx(-1.0) and s[3] == pytest.approx(99_999 - (100_000 + 100_002 + 99_999) / 3)
    assert s[2] == pytest.approx((0.8 + 0.25) / 2)  # mean of the last two imbalances
    assert s[4] == 2.0 and s[5] == pytest.approx(-1.0)  # spread; mid minus P0 (100000)
    assert env.state_lows[1] < 0 < env.state_highs[1]
    with pytest.raises(ValueError):
        make_env(state_features=["not_a_feature"])


def test_vs_twap_reward_and_linear_corridor():
    """vs_twap reward: sums to IS minus the mid-executed TWAP shortfall; linear corridor closes on X0."""
    from experiments.common import policy_env_params
    from experiments.policies import make_policy

    kw = dict(parent_order_size=2000, execution_window="00:02:00", first_interval="00:05:00",
              action_mode="schedule", action_repeat=10, schedule_band=0.2, schedule_corridor="linear",
              reward_mode="vs_twap", state_features=["schedule_dev", "time_pct", "imbalance_5", "dev_60"],
              background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    env = make_env(**kw)
    assert env.schedule_bounds(120) == (2000, 2000)
    lo, hi = env.schedule_bounds(60); assert lo == pytest.approx(1000 - 200) and hi == pytest.approx(1000 + 200)
    lo, hi = env.schedule_bounds(90); assert lo == pytest.approx(1500 - 100) and hi == pytest.approx(1500 + 100)
    params = policy_env_params(dict(kw, timestep_duration="1s", q_min=20, num_action_levels=4))
    twap = make_policy("TWAP", params, seed=1004)
    env.seed(1004); s = env.reset(); done = False; total = 0.0; mids = []
    while not done:
        s, r, done, info = env.step(twap.get_action(s)); total += r
    assert info["executed_quantity"] == 2000 and info["unfilled_quantity"] == 0
    assert total == pytest.approx(info["episode_reward"])
    # TWAP itself follows the schedule, so its vs_twap inventory term is (nearly) zero: the
    # episode reward is essentially its execution cost against the mid minus the depth penalty
    assert abs(info["episode_reward"] - (info["execution_cost"] - info["depth_penalty"])) < 0.3
    assert info["execution_cost"] < 0  # pays the half spread


def test_passive_and_mixed_styles_complete_and_account_fills():
    """Passive style: limit orders at the bid fill the order, market orders only when behind the
    corridor; mixed style: TWAP (market x1) and PassiveTWAP (passive x1) are distinct actions."""
    from experiments.common import policy_env_params
    from experiments.policies import make_policy

    base = dict(parent_order_size=2000, execution_window="00:02:00", first_interval="00:05:00",
                action_mode="schedule", action_repeat=10, schedule_band=0.2, reward_mode="vs_twap",
                state_features=["schedule_dev", "time_pct", "imbalance_5", "resting_pct"],
                background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    with pytest.raises(AssertionError):
        make_env(**dict(base, action_mode="lots", order_style="passive"))
    env = make_env(order_style="passive", **base)
    params = policy_env_params(dict(base, timestep_duration="1s", q_min=20, num_action_levels=4, order_style="passive"))
    assert params["passive_twap_action"] == 2 and params["twap_action"] == 2
    pol = make_policy("PassiveTWAP", params, seed=1005)
    env.seed(1005); s = env.reset(); done = False; total = 0.0
    while not done:
        s, r, done, info = env.step(pol.get_action(s)); total += r
    assert info["executed_quantity"] == 2000 and info["remaining_quantity"] == 0 and info["unfilled_quantity"] == 0
    assert info["passive_fill_quantity"] > 0 and info["num_limit_orders"] > 0
    assert info["passive_fill_quantity"] + info["num_orders"] * 0 <= 2000
    assert info["execution_cost"] > -0.5  # passive fills earn (part of) the spread
    assert total == pytest.approx(info["episode_reward"])
    # mixed style
    envm = make_env(order_style="mixed", **base)
    assert envm.mixed_actions[3] == (1.0, "market") and envm.mixed_actions[1] == (1.0, "passive")
    pm = policy_env_params(dict(base, timestep_duration="1s", q_min=20, num_action_levels=4, order_style="mixed"))
    assert pm["twap_action"] == 3 and pm["passive_twap_action"] == 1
    twap = make_policy("TWAP", pm, seed=1005)
    envm.seed(1005); s = envm.reset(); done = False
    while not done:
        s, r, done, info = envm.step(twap.get_action(s))
    assert info["executed_quantity"] == 2000 and info["passive_fill_quantity"] == 0 and info["num_limit_orders"] == 0
    with pytest.raises(AssertionError):
        make_env(order_style="mixed", **dict(base, num_action_levels=2))
