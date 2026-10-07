"""Shared helpers: experiment configuration files and environment construction."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from abides_core.utils import str_to_ns

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys of the YAML "environment" section that are forwarded verbatim to
# SubGymMarketsExecutionEnv_v0.
ENV_KEYS = [
    "background_config",
    "mkt_close",
    "timestep_duration",
    "starting_cash",
    "state_history_length",
    "market_data_buffer_length",
    "first_interval",
    "parent_order_size",
    "execution_window",
    "direction",
    "q_min",
    "num_action_levels",
    "depth_penalty_alpha",
    "beta_not_enough",
    "beta_too_much",
    "is_weight",
    # extensions (CHANGES.md C10, C11); absent from paper.yaml
    "schedule_penalty",
    "action_repeat",
    "action_mode",
    "schedule_band",
    "schedule_corridor",
    "reward_mode",
    "state_features",
    "order_style",
    "passive_horizon",
    "passive_offset",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("environment", {})
    cfg.setdefault("background", {})
    cfg.setdefault("evaluation", {})
    return cfg


def env_kwargs_from_config(cfg: Dict[str, Any], background_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Keyword arguments for ``gym.make("markets-execution-v0", **kwargs)``.

    ``background_overrides`` (e.g. ``{"num_noise_agents": 2000}``) are merged
    into ``background_config_extra_kvargs`` on top of the YAML ``background``
    section; this is how the robustness experiments of Figure 8 are generated.
    """
    env = {k: v for k, v in cfg.get("environment", {}).items() if k in ENV_KEYS}
    background = copy.deepcopy(cfg.get("background", {}))
    if background_overrides:
        background.update(background_overrides)
    if background:
        env["background_config_extra_kvargs"] = background
    return env


def num_steps_from_env_kwargs(env_kwargs: Dict[str, Any]) -> int:
    """Number of decision points in the execution window (N in the paper).

    With ``action_repeat > 1`` a decision spans that many wake-ups.
    """
    window = str_to_ns(env_kwargs.get("execution_window", "00:30:00"))
    step = str_to_ns(env_kwargs.get("timestep_duration", "1s"))
    repeat = int(env_kwargs.get("action_repeat", 1))
    return int(round(window / step / repeat))


def policy_env_params(env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Parameters the hand-crafted policies need (see ``experiments.policies``)."""
    levels = int(env_kwargs.get("num_action_levels", 4))
    action_mode = str(env_kwargs.get("action_mode", "lots"))
    style = str(env_kwargs.get("order_style", "market"))
    if action_mode != "schedule":
        twap_action, passive_twap_action = None, None
    elif style == "mixed":
        twap_action, passive_twap_action = 3, 1  # market x1, passive x1
    else:
        twap_action = levels // 2
        passive_twap_action = levels // 2 if style == "passive" else None
    return {
        "parent_order_size": int(env_kwargs.get("parent_order_size", 20_000)),
        # per decision: action k sends k * q_min shares at each of the action_repeat wake-ups
        "q_min": int(env_kwargs.get("q_min", 20)) * int(env_kwargs.get("action_repeat", 1)),
        "num_action_levels": levels,
        "num_steps": num_steps_from_env_kwargs(env_kwargs),
        # schedule mode: actions are participation multipliers of the TWAP rate; the middle action is TWAP
        "action_mode": action_mode,
        "twap_action": twap_action,
        "passive_twap_action": passive_twap_action,
        # position of the executed fraction in the state vector (custom feature lists may reorder it)
        "holdings_index": (list(env_kwargs["state_features"]).index("holdings_pct")
                           if env_kwargs.get("state_features") and "holdings_pct" in env_kwargs["state_features"] else 0),
        "imbalance_index": (list(env_kwargs["state_features"]).index("imbalance_5")
                            if env_kwargs.get("state_features") and "imbalance_5" in env_kwargs["state_features"] else 2),
        "env_config": copy.deepcopy(env_kwargs),
    }
