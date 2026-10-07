"""Execution policies evaluated in the paper.

Every policy exposes ``name`` and ``get_action(state) -> int`` on the action
grid of ``SubGymMarketsExecutionEnv_v0`` (``0`` = do nothing, ``k`` = market
order of ``k * q_min`` shares).  The three hand-crafted baselines follow
Section V-C of Hafsi & Vittori (2026); the RL policy wraps an RLlib DQN
checkpoint.

The baselines only use information available in the state vector
(``holdings_pct`` and ``time_pct``) and their own random number generator, so
they can be run in worker processes without any RLlib dependency.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# index of the state features (see SubGymMarketsExecutionEnv_v0.state_feature_names)
HOLDINGS_PCT = 0
TIME_PCT = 1


class BasePolicy:
    name: str = "base"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the policy at the start of an episode (keeps rollouts reproducible)."""
        self.rng = np.random.RandomState(seed)

    def get_action(self, state: np.ndarray) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    # Optional hook used by the evaluation runner for Figure 6c.
    def q_values(self, state: np.ndarray) -> Optional[np.ndarray]:
        return None


class TWAPPolicy(BasePolicy):
    """Time-Weighted Average Price schedule, equation (3) of the paper.

    The parent order ``X0`` is split evenly over the ``N`` decision points of the
    window (``X0 / N`` shares per step).  Because the environment only accepts
    multiples of ``q_min``, the policy tracks the cumulative TWAP target
    ``X0 * (n + 1) / N`` and, at every step, sends the smallest admissible market
    order that keeps it on schedule: ``k = round((target - executed) / q_min)``
    clipped to the action grid.  With ``X0 = 20000``, ``N = 1800`` and
    ``q_min = 20`` this is one 20-share market order every 1.8 seconds.  At the
    last decision point any remaining deficit is rounded *up*, so that the
    order is finished whenever the market fills it (this only matters when
    ``X0`` is not a multiple of ``q_min`` or after partial fills; with the
    paper's values the rule never triggers).
    """

    name = "TWAP"

    def __init__(
        self,
        parent_order_size: int,
        num_steps: int,
        q_min: int,
        num_action_levels: int,
        seed: Optional[int] = None,
        constant_action: Optional[int] = None,
        holdings_index: int = HOLDINGS_PCT,
    ) -> None:
        super().__init__(seed)
        self.parent_order_size = parent_order_size
        self.num_steps = num_steps
        self.q_min = q_min
        self.num_action_levels = num_action_levels
        # schedule mode of the environment: TWAP is the constant "participation 1" action
        self.constant_action = constant_action
        self.holdings_index = holdings_index
        self.step = 0

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.step = 0

    def get_action(self, state: np.ndarray) -> int:
        if self.constant_action is not None:
            self.step += 1
            return int(self.constant_action)
        executed = float(state[self.holdings_index]) * self.parent_order_size
        # the order sent at decision point n is filled before the state at n + 1
        target = self.parent_order_size * min(1.0, (self.step + 1) / self.num_steps)
        deficit = target - executed
        k = int(round(deficit / self.q_min))
        if self.step >= self.num_steps - 1 and deficit > 0:
            k = max(k, int(math.ceil(deficit / self.q_min)))  # finish the order at the last decision
        self.step += 1
        return int(np.clip(k, 0, self.num_action_levels))


class PassivePolicy(BasePolicy):
    """Passive Policy (PP): 60% do nothing, otherwise a uniformly random size.

    "This policy mostly keeps the agent inactive, with a 60% chance of doing
    nothing. With a 40% chance, it randomly chooses and executes a quantity
    among four available options, each with equal likelihood."
    """

    name = "Passive"

    def __init__(self, num_action_levels: int = 4, p_nothing: float = 0.6, seed=None):
        super().__init__(seed)
        self.num_action_levels = num_action_levels
        self.p_nothing = p_nothing

    def get_action(self, state: np.ndarray) -> int:
        if self.rng.rand() < self.p_nothing:
            return 0
        return int(self.rng.randint(1, self.num_action_levels + 1))


class RandomPolicy(BasePolicy):
    """Random Policy (RP), paper Section V-C: 50% do nothing, otherwise uniform over {0, 1, 2, 3}.

    P(0) = 0.5 + 0.5 / 4 = 0.625 and P(k) = 0.125 for k = 1..3.
    """

    name = "Random"

    def __init__(self, max_level: int = 3, p_nothing: float = 0.5, seed=None):
        super().__init__(seed)
        self.max_level = max_level
        self.p_nothing = p_nothing

    def get_action(self, state: np.ndarray) -> int:
        if self.rng.rand() < self.p_nothing:
            return 0
        return int(self.rng.randint(0, self.max_level + 1))


class ImbalanceRulePolicy(BasePolicy):
    """Hand-crafted timing rule for the schedule mode of the environment.

    Participation multiplier 2 (fastest action) when the 5-level imbalance is
    above ``hi`` (bid-heavy book, mid expected to rise), 0 (pause) when it is
    below ``lo``, TWAP otherwise.  This is the rule the predictability study of
    the predictability study evaluates on replayed paths; here it runs in the live
    simulator, with its own impact.  In lot mode it sends ``k = 4 / 0 / 2``
    lots with the same thresholds (a crude schedule that does not complete).
    """

    name = "Rule"

    def __init__(self, num_action_levels: int, imbalance_index: int = 2, twap_action: Optional[int] = None,
                 hi: float = 0.6, lo: float = 0.4, fast: Optional[int] = None, slow: int = 0,
                 seed: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(seed)
        self.num_action_levels = num_action_levels
        self.imbalance_index = imbalance_index
        self.twap_action = twap_action if twap_action is not None else num_action_levels // 2
        self.hi, self.lo = hi, lo
        self.fast = int(fast) if fast is not None else int(num_action_levels)
        self.slow = int(slow)
        if name:
            self.name = name

    @classmethod
    def from_name(cls, name: str, env_params: Dict[str, Any], seed: Optional[int] = None) -> "ImbalanceRulePolicy":
        """``Rule`` or ``Rule-lo0.4-hi0.6-s0-f4`` (thresholds, slow and fast action indices)."""
        kw: Dict[str, Any] = {}
        for part in name.split("-")[1:]:
            for key, attr, typ in (("lo", "lo", float), ("hi", "hi", float), ("s", "slow", int), ("f", "fast", int)):
                if part.startswith(key) and part[len(key):].replace(".", "", 1).isdigit():
                    kw[attr] = typ(part[len(key):])
                    break
            else:
                raise ValueError(f"cannot parse rule policy name {name!r}")
        return cls(num_action_levels=env_params["num_action_levels"], imbalance_index=int(env_params.get("imbalance_index", 2)),
                   twap_action=env_params.get("twap_action"), seed=seed, name=name, **kw)

    def get_action(self, state: np.ndarray) -> int:
        imb = float(state[self.imbalance_index])
        if imb > self.hi:
            return self.fast
        if imb < self.lo:
            return self.slow
        return int(self.twap_action)


class TrendRulePolicy(BasePolicy):
    """Hand-crafted trend follower for the schedule mode: buy at the fastest rate when the
    chosen return feature (``ret_300`` by default) exceeds ``+threshold`` cents, pause below
    ``-threshold``, TWAP in between.  Reference for the trending regime (``fund_drift``)."""

    name = "Trend"

    def __init__(self, feature_index: int, num_action_levels: int, twap_action: int, threshold: float = 5.0,
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.feature_index, self.fast, self.slow = feature_index, int(num_action_levels), 0
        self.twap_action, self.threshold = int(twap_action), float(threshold)

    def get_action(self, state: np.ndarray) -> int:
        x = float(state[self.feature_index])
        return self.fast if x > self.threshold else (self.slow if x < -self.threshold else self.twap_action)


class AggressivePolicy(BasePolicy):
    """Always send the largest market order (used in a few sanity checks)."""

    name = "Aggressive"

    def __init__(self, num_action_levels: int = 4, seed=None):
        super().__init__(seed)
        self.num_action_levels = num_action_levels

    def get_action(self, state: np.ndarray) -> int:
        return self.num_action_levels


class DoNothingPolicy(BasePolicy):
    """Never trades (used to sample undisturbed price paths, Figure 3)."""

    name = "DoNothing"

    def get_action(self, state: np.ndarray) -> int:
        return 0


def resolve_checkpoint(path: str) -> str:
    """Accept a run directory, its ``checkpoint`` directory or an RLlib ``checkpoint_NNNNNN`` directory.

    ``train_dqn.py`` writes ``<out>/checkpoint/checkpoint_<iteration>``; passing
    ``<out>`` or ``<out>/checkpoint`` resolves to the latest ``checkpoint_*``.
    """
    import os

    p = os.path.abspath(os.path.expanduser(path))
    final_txt = os.path.join(p, "final_checkpoint.txt")
    if os.path.isfile(final_txt):
        cand = open(final_txt).read().strip()
        if os.path.isdir(cand):
            return cand
        p = os.path.join(p, "checkpoint")
    if os.path.isdir(p) and os.path.basename(p).startswith("checkpoint_"):
        return p
    subs = sorted(d for d in os.listdir(p) if d.startswith("checkpoint_") and os.path.isdir(os.path.join(p, d))) if os.path.isdir(p) else []
    if subs:
        return os.path.join(p, subs[-1])
    raise FileNotFoundError(f"No RLlib checkpoint found at {path}")


class RLlibDQNPolicy(BasePolicy):
    """Greedy policy of a trained RLlib DQN checkpoint.

    The observation filter learned during training (``MeanStdFilter``) is applied
    exactly as RLlib does at inference time; ``q_values`` returns the (dueling)
    Q-values of every action for Figure 6c.
    """

    name = "RL"

    def __init__(self, checkpoint_path: str, env_config: Dict[str, Any], seed=None):
        super().__init__(seed)
        # Imported lazily so that the baselines do not need RLlib.
        import ray
        import torch
        from ray.rllib.algorithms.dqn import DQNConfig
        from ray.rllib.algorithms.dqn.dqn_torch_policy import compute_q_values
        from ray.rllib.policy.sample_batch import SampleBatch

        import abides_gym  # noqa: F401  (registers the environment with RLlib)
        from experiments.train_dqn import apply_dqn_overrides, build_dqn_config

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1, log_to_driver=False,
             object_store_memory=200_000_000)
        self._torch = torch
        self._compute_q_values = compute_q_values
        self._SampleBatch = SampleBatch
        self.checkpoint_path = resolve_checkpoint(checkpoint_path)
        # rebuild the network exactly as trained: the run's run_config.json (written by
        # train_dqn.py next to the checkpoint) carries the Q-head sizes and the RLlib overrides
        run_cfg = load_run_config(self.checkpoint_path)
        config: DQNConfig = build_dqn_config(
            env_config=env_config, lr=1e-3, seed=0,
            gamma=float(run_cfg.get("gamma", 0.9999)), hiddens=run_cfg.get("hiddens"),
        )
        if run_cfg.get("dqn_override"):
            apply_dqn_overrides(config, run_cfg["dqn_override"])
        self.algo = config.build()
        self.algo.restore(self.checkpoint_path)
        self.policy = self.algo.get_policy()
        self.filter = self.algo.workers.local_worker().filters["default_policy"]

    def get_action(self, state: np.ndarray) -> int:
        return int(self.algo.compute_single_action(state, explore=False))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        obs = self.filter(np.asarray(state, dtype=np.float32), update=False)
        obs_t = self._torch.as_tensor(obs[None, :], dtype=self._torch.float32)
        input_dict = self._SampleBatch({self._SampleBatch.OBS: obs_t, "is_training": False})
        q, _, _, _ = self._compute_q_values(
            self.policy, self.policy.model, input_dict, explore=False, is_training=False
        )
        return q.detach().cpu().numpy()[0]


def load_run_config(checkpoint_path: str) -> Dict[str, Any]:
    """Return the ``run_config.json`` written by ``train_dqn.py`` for a checkpoint, or ``{}``.

    The file is searched in the checkpoint directory and its three parents
    (``<run>/checkpoint/checkpoint_NNNNNN`` and ``models/<name>/checkpoint/checkpoint_NNNNNN``).
    """
    import json

    p = Path(checkpoint_path)
    for d in [p, *list(p.parents)[:3]]:
        f = d / "run_config.json"
        if f.exists():
            with open(f) as fh:
                return json.load(fh)
    return {}


def make_policy(name: str, env_params: Dict[str, Any], seed: Optional[int] = None, checkpoint: Optional[str] = None) -> BasePolicy:
    """Factory used by the evaluation runner.

    ``env_params`` must contain ``parent_order_size``, ``q_min``,
    ``num_action_levels`` and ``num_steps`` (execution window / timestep).
    """
    name_l = name.lower()
    if name_l == "twap":
        return TWAPPolicy(
            parent_order_size=env_params["parent_order_size"],
            num_steps=env_params["num_steps"],
            q_min=env_params["q_min"],
            num_action_levels=env_params["num_action_levels"],
            constant_action=env_params.get("twap_action"),
            holdings_index=int(env_params.get("holdings_index", HOLDINGS_PCT)),
            seed=seed,
        )
    if name_l == "passive":
        return PassivePolicy(num_action_levels=env_params["num_action_levels"], seed=seed)
    if name_l in ("rule", "imbalancerule") or name_l.startswith("rule-"):
        return ImbalanceRulePolicy.from_name(name, env_params, seed=seed)
    if name_l.startswith("trend"):
        feats = list(env_params["env_config"].get("state_features") or [])
        feature = "ret_300"
        thr = 5.0
        for part in name.split("-")[1:]:
            if part.startswith("thr"):
                thr = float(part[3:])
            else:
                feature = part
        if feature not in feats:
            raise ValueError(f"Trend policy needs {feature!r} in state_features")
        pol = TrendRulePolicy(feats.index(feature), env_params["num_action_levels"], env_params["twap_action"], thr, seed=seed)
        pol.name = name  # distinct names so that variants do not overwrite each other in the results
        return pol
    if name_l == "passivetwap":
        if env_params.get("passive_twap_action") is None:
            raise ValueError("PassiveTWAP needs action_mode='schedule' with order_style 'passive' or 'mixed'")
        pol = TWAPPolicy(parent_order_size=env_params["parent_order_size"], num_steps=env_params["num_steps"], q_min=env_params["q_min"],
                         num_action_levels=env_params["num_action_levels"], constant_action=int(env_params["passive_twap_action"]))
        pol.name = "PassiveTWAP"
        return pol
    if name_l == "random":
        return RandomPolicy(max_level=min(3, env_params["num_action_levels"]), seed=seed)
    if name_l == "aggressive":
        return AggressivePolicy(num_action_levels=env_params["num_action_levels"], seed=seed)
    if name_l in ("donothing", "do_nothing", "nothing"):
        return DoNothingPolicy(seed=seed)
    if name_l in ("rl", "dqn"):
        if checkpoint is None:
            raise ValueError("The RL policy needs a checkpoint path")
        return RLlibDQNPolicy(checkpoint, env_config=env_params["env_config"], seed=seed)
    raise ValueError(f"Unknown policy {name}")
