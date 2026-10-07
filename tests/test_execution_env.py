"""Unit and integration tests of the execution environment (paper MDP)."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

import gym  # noqa: E402
import abides_gym  # noqa: E402,F401
from abides_gym.envs.markets_execution_environment_v0 import SubGymMarketsExecutionEnv_v0  # noqa: E402


class _Fill:
    def __init__(self, quantity, fill_price):
        self.quantity = quantity
        self.fill_price = fill_price


def make_env(**kw):
    params = dict(first_interval="00:05:00", execution_window="00:02:00", parent_order_size=2000,
                  background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    params.update(kw)
    return SubGymMarketsExecutionEnv_v0(**params)


def test_action_space_and_mapping():
    env = make_env()
    assert env.action_space.n == 5
    assert env._map_action_space_to_ABIDES_SIMULATOR_SPACE(0) == []
    for k in range(1, 5):
        acts = env._map_action_space_to_ABIDES_SIMULATOR_SPACE(k)
        assert acts == [{"type": "MKT", "direction": "BUY", "size": 20 * k}]
    with pytest.raises(ValueError):
        env._map_action_space_to_ABIDES_SIMULATOR_SPACE(5)
    assert env.custom_metrics_tracker.action_counter == {"action_0": 1, "action_1": 1, "action_2": 1, "action_3": 1, "action_4": 1}


def test_observation_space_shape_and_names():
    env = make_env(state_history_length=4)
    assert env.observation_space.shape == (8,)
    assert env.state_feature_names[:5] == ["holdings_pct", "time_pct", "imbalance_5", "best_bid", "best_ask"]
    assert len(env.state_feature_names) == 8


def _raw_state(bids_prev, asks_prev, bids_now, asks_now, fills, holdings=0, t_offset_s=1):
    """Two-wake-up raw state in the (unflipped) format produced by the gym agent."""
    mkt_open = 0
    first_interval = 5 * 60 * 10**9
    t_prev = mkt_open + first_interval
    t_now = t_prev + t_offset_s * 10**9

    def rs(bids, asks, t, fills_, holdings_):
        return {
            "parsed_mkt_data": [{"bids": bids, "asks": asks, "last_transaction": 100_000, "exchange_ts": t}],
            "parsed_volume_data": [{"bid_volume": 0, "ask_volume": 0, "total_volume": 0, "last_transaction": 100_000, "exchange_ts": t}],
            "internal_data": {"holdings": holdings_, "cash": 0, "inter_wakeup_executed_orders": fills_, "current_time": t,
                              "mkt_open": mkt_open, "mkt_close": 10**12, "episode_executed_orders": [], "order_status": {}},
        }

    return [rs(bids_prev, asks_prev, t_prev, [], 0), rs(bids_now, asks_now, t_now, fills, holdings)]


def test_reward_formula_matches_paper_equation_4():
    env = make_env(depth_penalty_alpha=2.0, parent_order_size=2000)
    env.entry_price = 100_000.0  # arrival price P0
    env.step_index = 1
    # previous best ask 100_002; fills: 20 @ 100_002 (at touch, no depth) and 20 @ 100_005 (3 cents deep)
    raw = _raw_state(
        bids_prev=[(99_998, 50)], asks_prev=[(100_002, 20), (100_005, 100)],
        bids_now=[(99_998, 50)], asks_now=[(100_005, 80)],
        fills=[_Fill(20, 100_002), _Fill(20, 100_005)], holdings=40,
    )
    r = env.raw_state_to_reward(raw)
    is_term = 20 * (100_000 - 100_002) + 20 * (100_000 - 100_005)  # -140
    depth = max(0, 100_002 - 100_002) + max(0, 100_005 - 100_002)  # 3
    expected = (is_term - 2.0 * depth) / 2000
    assert r == pytest.approx(expected)
    t = env.custom_metrics_tracker
    assert t.step_implementation_shortfall == pytest.approx(is_term / 2000)
    assert t.step_depth_penalty == pytest.approx(2.0 * depth / 2000)
    assert t.step_executed_quantity == 40
    assert t.step_avg_fill_price == pytest.approx(100_003.5)
    assert t.num_trading_steps == 1


def test_reward_zero_without_fills():
    env = make_env()
    env.entry_price = 100_000.0
    raw = _raw_state([(99_998, 50)], [(100_002, 20)], [(99_998, 50)], [(100_002, 20)], fills=[])
    assert env.raw_state_to_reward(raw) == 0.0


def test_terminal_penalty_symmetric():
    env = make_env(beta_not_enough=5.0, beta_too_much=5.0, parent_order_size=2000)
    raw_under = _raw_state([], [], [], [], fills=[], holdings=1900)
    raw_over = _raw_state([], [], [], [], fills=[], holdings=2040)
    raw_exact = _raw_state([], [], [], [], fills=[], holdings=2000)
    assert env.raw_state_to_update_reward(raw_under) == pytest.approx(-5.0 * 100 / 2000)
    assert env.raw_state_to_update_reward(raw_over) == pytest.approx(-5.0 * 40 / 2000)
    assert env.raw_state_to_update_reward(raw_exact) == 0.0


def test_done_conditions():
    env = make_env(parent_order_size=2000)
    assert env.raw_state_to_done(_raw_state([], [], [], [], [], holdings=2000)) is True
    assert env.raw_state_to_done(_raw_state([], [], [], [], [], holdings=1999)) is False
    # window over: 2 minutes after the first wake-up
    assert env.raw_state_to_done(_raw_state([], [], [], [], [], holdings=0, t_offset_s=120)) is True


def test_state_vector_from_raw_state():
    env = make_env(parent_order_size=2000, state_history_length=4)
    env._reset_episode_variables()
    raw = _raw_state(
        bids_prev=[(99_998, 50), (99_997, 50)], asks_prev=[(100_002, 20), (100_003, 30)],
        bids_now=[(99_999, 10), (99_998, 90)], asks_now=[(100_001, 40), (100_002, 60)],
        fills=[], holdings=200,
    )
    s = env.raw_state_to_state(raw)
    assert s.shape == (8,)
    assert s[0] == pytest.approx(0.1)  # holdings_pct
    assert s[1] == pytest.approx(1 / 120)  # 1 second into a 2-minute window
    assert s[2] == pytest.approx(100 / (100 + 100))  # imbalance over 5 levels
    assert s[3] == 99_999 and s[4] == 100_001
    # mid moved from 100_000 to 100_000 -> return 0; padded zeros before
    assert np.allclose(s[5:], [0, 0, 0])
    assert env.entry_price == pytest.approx(100_000.0)  # set at step 0


@pytest.mark.slow
def test_full_episode_and_reset_are_consistent():
    env = gym.make("markets-execution-v0", first_interval="00:05:00", execution_window="00:01:00",
                   parent_order_size=1000, background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
    env.seed(0)
    s = env.reset()
    assert env.observation_space.contains(s)
    total, done, n = 0.0, False, 0
    while not done:
        s, r, done, info = env.step(4)
        total += r
        n += 1
    assert n <= 60
    assert info["executed_quantity"] >= 1000 or info["time_pct"] >= 1.0
    assert total == pytest.approx(info["episode_reward"])
    assert info["implementation_shortfall"] - info["depth_penalty"] + info["terminal_penalty"] == pytest.approx(total)
    # second episode starts from scratch
    s2 = env.reset()
    assert env.unwrapped.step_index == 1
    assert env.unwrapped.custom_metrics_tracker.executed_quantity == 0
    assert sum(env.unwrapped.custom_metrics_tracker.action_counter.values()) == 0


@pytest.mark.slow
def test_seeding_is_reproducible():
    def rollout(seed):
        env = gym.make("markets-execution-v0", first_interval="00:05:00", execution_window="00:00:30",
                       parent_order_size=500, background_config_extra_kvargs={"mm_wake_up_freq": "60S"})
        env.seed(seed)
        s = env.reset()
        out = [s.tolist()]
        for a in [1, 2, 0, 4, 3]:
            s, r, d, i = env.step(a)
            out.append((s.tolist(), r))
            if d:
                break
        return out

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_schedule_penalty_shaping_term():
    """Optional shaping: lambda * |x_t - X0 t/T| / X0 is subtracted from the step reward."""
    env0 = make_env(parent_order_size=2000, execution_window="00:02:00")
    env1 = make_env(parent_order_size=2000, execution_window="00:02:00", schedule_penalty=1.5)
    for env in (env0, env1):
        env.entry_price = 100_000.0
        env.step_index = 1
    # the helper puts the first wake-up at open + 5 min; 60 s later is half of the 2-minute window
    raw = _raw_state(
        bids_prev=[(99_998, 50)], asks_prev=[(100_002, 20)],
        bids_now=[(99_998, 50)], asks_now=[(100_002, 20)],
        fills=[_Fill(20, 100_002)], holdings=500, t_offset_s=60,
    )
    r0 = env0.raw_state_to_reward(raw)
    r1 = env1.raw_state_to_reward(raw)
    # schedule at t/T = 0.5 is 1000 shares; executed 500 -> deviation 500
    expected = 1.5 * 500 / 2000
    assert r0 - r1 == pytest.approx(expected)
    assert env1.custom_metrics_tracker.step_schedule_penalty == pytest.approx(expected)
    assert env1.custom_metrics_tracker.schedule_penalty == pytest.approx(expected)
    assert env0.custom_metrics_tracker.schedule_penalty == 0.0
