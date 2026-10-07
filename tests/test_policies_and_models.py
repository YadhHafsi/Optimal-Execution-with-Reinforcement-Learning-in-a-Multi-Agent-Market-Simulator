"""Tests of the hand-crafted policies and of the numpy order-size model."""
import numpy as np
import pytest

from abides_markets.models import OrderSizeModel
from experiments.policies import PassivePolicy, RandomPolicy, TWAPPolicy, make_policy


def test_twap_schedule_tracks_x0_over_n():
    pol = TWAPPolicy(parent_order_size=20_000, num_steps=1800, q_min=20, num_action_levels=4, seed=0)
    executed = 0
    actions = []
    for n in range(1800):
        state = np.array([executed / 20_000, n / 1800, 0.5, 100_000, 100_001, 0, 0, 0], dtype=np.float32)
        a = pol.get_action(state)
        actions.append(a)
        executed += 20 * a  # assume every order is filled
    assert executed == 20_000
    assert set(actions) <= {0, 1}  # 20 shares every 1.8 s -> never more than one lot per step
    assert abs(np.mean(actions) - 20_000 / 20 / 1800) < 1e-9


def test_twap_catches_up_after_missed_fills():
    pol = TWAPPolicy(parent_order_size=20_000, num_steps=1800, q_min=20, num_action_levels=4, seed=0)
    for n in range(100):  # nothing gets filled for 100 steps
        pol.get_action(np.array([0.0, n / 1800, 0.5, 0, 0, 0, 0, 0]))
    a = pol.get_action(np.array([0.0, 100 / 1800, 0.5, 0, 0, 0, 0, 0]))
    assert a == 4  # deficit of ~1100 shares -> largest admissible order


def test_passive_and_random_action_frequencies():
    n = 200_000
    passive = PassivePolicy(num_action_levels=4, seed=1)
    acts = np.array([passive.get_action(None) for _ in range(n)])
    freq = np.bincount(acts, minlength=5) / n
    assert freq[0] == pytest.approx(0.6, abs=0.01)
    assert np.allclose(freq[1:], 0.1, atol=0.01)

    rnd = RandomPolicy(max_level=3, seed=1)
    acts = np.array([rnd.get_action(None) for _ in range(n)])
    freq = np.bincount(acts, minlength=5) / n
    assert freq[0] == pytest.approx(0.5 + 0.5 / 4, abs=0.01)
    assert np.allclose(freq[1:4], 0.125, atol=0.01)
    assert freq[4] == 0


def test_policy_reset_reproducible():
    p = PassivePolicy(seed=5)
    a = [p.get_action(None) for _ in range(50)]
    p.reset(5)
    b = [p.get_action(None) for _ in range(50)]
    assert a == b


def test_make_policy_factory():
    params = {"parent_order_size": 20_000, "q_min": 20, "num_action_levels": 4, "num_steps": 1800, "env_config": {}}
    assert make_policy("twap", params).name == "TWAP"
    assert make_policy("Passive", params).name == "Passive"
    assert make_policy("random", params).name == "Random"
    with pytest.raises(ValueError):
        make_policy("nope", params)


# Reference statistics of the original pomegranate-based model, measured with
# pomegranate 0.14.8 on 300 000 samples (see CHANGES.md, C3).
POMEGRANATE_REFERENCE = {"mean": 105.9, "p_lt_100": 0.185, "p_100": 0.699, "p_200": 0.060, "p_ge_300": 0.042, "median": 100}


def test_order_size_model_matches_pomegranate_reference():
    model = OrderSizeModel()
    rs = np.random.RandomState(0)
    x = np.array([model.sample(rs) for _ in range(200_000)])
    assert x.mean() == pytest.approx(POMEGRANATE_REFERENCE["mean"], abs=1.0)
    assert (x < 100).mean() == pytest.approx(POMEGRANATE_REFERENCE["p_lt_100"], abs=0.005)
    assert (x == 100).mean() == pytest.approx(POMEGRANATE_REFERENCE["p_100"], abs=0.005)
    assert (x == 200).mean() == pytest.approx(POMEGRANATE_REFERENCE["p_200"], abs=0.005)
    assert (x >= 300).mean() == pytest.approx(POMEGRANATE_REFERENCE["p_ge_300"], abs=0.005)
    assert np.median(x) == POMEGRANATE_REFERENCE["median"]
    assert model.mean() == pytest.approx(105.94, abs=0.05)
    assert x.min() >= 0 and np.issubdtype(x.dtype, np.integer)


def test_resolve_checkpoint_accepts_run_checkpoint_and_leaf_dirs(tmp_path):
    from experiments.policies import resolve_checkpoint

    run = tmp_path / "lr_1e-3_seed_10"
    leaf = run / "checkpoint" / "checkpoint_000100"
    leaf.mkdir(parents=True)
    (run / "final_checkpoint.txt").write_text(str(leaf) + "\n")
    assert resolve_checkpoint(str(run)) == str(leaf)
    assert resolve_checkpoint(str(run / "checkpoint")) == str(leaf)
    assert resolve_checkpoint(str(leaf)) == str(leaf)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "nowhere"))


def test_apply_dqn_overrides_merges_nested_settings():
    pytest.importorskip("ray")
    from experiments.common import env_kwargs_from_config, load_config
    from experiments.train_dqn import apply_dqn_overrides, build_dqn_config

    cfg = load_config("experiments/configs/smoke.yaml")
    config = build_dqn_config(env_kwargs_from_config(cfg), lr=1e-3, seed=0)
    apply_dqn_overrides(config, {"n_step": 20, "replay_buffer_config.capacity": 123456,
                                 "model.fcnet_hiddens": [64, 32], "train_batch_size": 64})
    d = config.to_dict()
    assert d["n_step"] == 20 and d["train_batch_size"] == 64
    assert d["replay_buffer_config"]["capacity"] == 123456
    assert d["replay_buffer_config"]["type"] == "MultiAgentPrioritizedReplayBuffer"  # other entries kept
    assert d["model"]["fcnet_hiddens"] == [64, 32] and d["model"]["fcnet_activation"] == "tanh"
    with pytest.raises(ValueError):
        apply_dqn_overrides(config, {"n_step.foo": 1})


def test_load_run_config_walks_up_from_checkpoint(tmp_path):
    import json

    from experiments.policies import load_run_config

    run = tmp_path / "run"
    ckpt = run / "checkpoint" / "checkpoint_000010"
    ckpt.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"hiddens": [256, 128], "gamma": 0.99, "dqn_override": {"n_step": 5}}))
    rc = load_run_config(str(ckpt))
    assert rc["hiddens"] == [256, 128] and rc["dqn_override"] == {"n_step": 5}
    assert load_run_config(str(tmp_path / "elsewhere")) == {}


def test_config_forwards_extension_keys(tmp_path):
    import yaml

    from experiments.common import env_kwargs_from_config, policy_env_params

    cfg = {"name": "x", "environment": {"parent_order_size": 2000, "action_mode": "schedule", "action_repeat": 10,
                                        "schedule_band": 0.25, "reward_mode": "mark_to_market",
                                        "state_features": ["schedule_dev", "imbalance_5"], "not_a_key": 1},
           "background": {"mm_wake_up_freq": "60S"}}
    kw = env_kwargs_from_config(cfg, {"num_noise_agents": 10})
    assert kw["action_mode"] == "schedule" and kw["schedule_band"] == 0.25 and kw["state_features"] == ["schedule_dev", "imbalance_5"]
    assert "not_a_key" not in kw and kw["background_config_extra_kvargs"] == {"mm_wake_up_freq": "60S", "num_noise_agents": 10}
    p = policy_env_params(kw)
    assert p["twap_action"] == 2 and p["num_steps"] == 180 and p["holdings_index"] == 0


def test_imbalance_rule_policy_thresholds():
    import numpy as np

    from experiments.policies import ImbalanceRulePolicy, make_policy

    pol = ImbalanceRulePolicy(num_action_levels=4, imbalance_index=2, twap_action=2)
    assert pol.get_action(np.array([0.0, 0.1, 0.75])) == 4  # bid-heavy: fastest
    assert pol.get_action(np.array([0.0, 0.1, 0.25])) == 0  # ask-heavy: pause
    assert pol.get_action(np.array([0.0, 0.1, 0.50])) == 2  # balanced: TWAP
    params = {"parent_order_size": 2000, "q_min": 200, "num_action_levels": 4, "num_steps": 12,
              "action_mode": "schedule", "twap_action": 2, "holdings_index": 0, "imbalance_index": 1, "env_config": {}}
    p2 = make_policy("Rule", params, seed=0)
    assert p2.name == "Rule" and p2.get_action(np.array([0.0, 0.9])) == 4


def test_rule_policy_from_name():
    import numpy as np

    from experiments.policies import make_policy

    params = {"parent_order_size": 20000, "q_min": 200, "num_action_levels": 4, "num_steps": 180,
              "action_mode": "schedule", "twap_action": 2, "holdings_index": 0, "imbalance_index": 2, "env_config": {}}
    p = make_policy("Rule-lo0.35-hi0.65-s1-f3", params, seed=0)
    assert [p.get_action(np.array([0, 0, x])) for x in (0.3, 0.5, 0.7)] == [1, 2, 3]
    assert p.name == "Rule-lo0.35-hi0.65-s1-f3"
    with pytest.raises(ValueError):
        make_policy("Rule-xyz", params, seed=0)


def test_train_dqn_defaults_come_from_the_config(tmp_path, monkeypatch):
    """The training settings of the config's ``training`` section are the CLI defaults."""
    import json

    import experiments.train_dqn as td

    captured = {}

    class FakeAlgo:
        def train(self):
            return {"timesteps_total": 10**9, "episodes_total": 1, "episode_reward_mean": 0.0, "episode_reward_min": 0.0,
                    "episode_reward_max": 0.0, "episode_len_mean": 1.0, "info": {}, "sampler_results": {"hist_stats": {}}}

        def get_policy(self):
            raise RuntimeError

        def save(self, path):
            return path

        def stop(self):
            pass

    def fake_build(env_config, lr, seed, lr_decay_steps, epsilon_timesteps, gamma, hiddens, **kw):
        captured.update(lr=lr, seed=seed, lr_decay_steps=lr_decay_steps, epsilon_timesteps=epsilon_timesteps, gamma=gamma, hiddens=hiddens)

        class C:
            def build(self):
                return FakeAlgo()

        return C()

    class FakeRay:
        @staticmethod
        def init(*a, **k):
            pass

        @staticmethod
        def shutdown():
            pass

    monkeypatch.setattr(td, "build_dqn_config", fake_build)
    monkeypatch.setitem(__import__("sys").modules, "ray", FakeRay)
    td.main(["--config", "experiments/configs/learn.yaml", "--out", str(tmp_path / "run")])
    assert captured == {"lr": 5e-4, "seed": 10, "lr_decay_steps": 90000, "epsilon_timesteps": 20000, "gamma": 0.99, "hiddens": [50, 20]}
    rc = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert rc["total_timesteps"] == 100000 and rc["env_config"]["action_mode"] == "schedule"
