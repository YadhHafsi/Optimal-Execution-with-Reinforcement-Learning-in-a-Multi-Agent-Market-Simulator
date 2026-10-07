"""Produce every figure and table of the paper from the stored results.

Inputs (all produced by the other scripts of this package)::

    results/<config>/training/lr_<lr>_seed_<seed>/{progress.csv,episodes.csv}   (train_dqn.py)
    results/<config>/<scenario>/{episodes.csv,steps.parquet}                    (evaluate.py)
    results/<config>/price_paths_4h.parquet, price_paths_30min.parquet          (sample_price_paths.py)

Outputs::

    figures/<config>/fig3_price_paths.png Fig. 3   sample ask-price paths
    figures/<config>/fig4_learning_curves.png  Fig. 4   DQN training reward (3 learning rates; per-episode rewards, rolling mean)
    figures/fig4b_learning_curves_seeds.png        seed robustness of the selected learning rate
    figures/fig5_is_distribution.png      Fig. 5   normalised implementation shortfall, standard scenario
    figures/fig6a_trajectory.png          Fig. 6a  bid/ask and actions of one RL episode
    figures/fig6b_execution_profile.png   Fig. 6b  executed fraction vs time, mean and 95% CI
    figures/fig6c_q_values.png            Fig. 6c  Q-values along the episode
    figures/fig7a_spreads.png             Fig. 7a  spread distribution at trading steps
    figures/fig7b_imbalance.png           Fig. 7b  imbalance distribution at trading steps
    figures/fig8_robustness.png           Fig. 8   IS distributions across agent configurations
    figures/figS*_...png                  supplementary variants (per-step IS, all-step spreads, ...)
    results/<config>/tables/table{1,2,3}.csv|md, ttests_*.csv|md, summary_*.csv
    results/<config>/report.html          self-contained HTML report with the tables and figures

Usage::

    python -m experiments.make_figures --config experiments/configs/paper.yaml
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import load_config  # noqa: E402

# ----------------------------------------------------------------------------
# style: thin marks, hairline solid grid, text in ink tokens, fixed colour order
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POLICY_ORDER = ["RL", "TWAP", "Passive", "Random"]
POLICY_COLOR = dict(zip(POLICY_ORDER, CATEGORICAL[:4]))
SEQ_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal ramp for action sizes 1..4

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

SCENARIO_TITLES = {"standard": "1000 noise / 12 momentum agents", "noise_10": "10 noise agents",
                   "noise_2000": "2000 noise agents", "momentum_6": "6 momentum agents", "momentum_24": "24 momentum agents"}

# Statistics of the normalised implementation shortfall that can be plotted and
# tabulated: the episode total, the mean per-step value and the last-step value.
IS_METRICS = {
    "implementation_shortfall": "episode total IS / X0 (cents per share)",
    "step_is_mean": "mean per-step IS / X0",
    "step_is_final": "IS / X0 of the last step",
}


# ----------------------------------------------------------------------------
# helpers
def _tidy(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="semibold")
    ax.tick_params(length=3)


def _save(fig, path: Path, figures: Dict[str, Path], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    figures[key] = path
    print(f"wrote {path}")


def _kde(ax, values: np.ndarray, color: str, label: str, x_grid: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.std(values) == 0:
        ax.axvline(values.mean() if len(values) else 0, color=color, label=label)
        return
    kde = stats.gaussian_kde(values)
    y = kde(x_grid)
    ax.plot(x_grid, y, color=color, label=label)
    ax.fill_between(x_grid, 0, y, color=color, alpha=0.10, linewidth=0)


def _strip(ax, values: np.ndarray, color: str, y: float, rng: np.random.RandomState) -> None:
    values = np.asarray(values, dtype=float)
    ax.scatter(values, np.full(len(values), y) + rng.uniform(-0.15, 0.15, len(values)), s=22, color=color,
               edgecolors=SURFACE, linewidths=1.0, zorder=3)


def _run_lr(run_dir: Path) -> Optional[float]:
    """Learning rate of a training run, read from its run_config.json."""
    p = run_dir / "run_config.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["lr"])
    except Exception:
        return None


def find_training_run(results_dir: Path, lr: float, seed: int) -> Optional[Path]:
    """Locate ``results/<config>/training/lr_*_seed_<seed>`` for a learning rate.

    Directory names are free-form (``lr_1e-3_seed_10`` or ``lr_0.001_seed_10``);
    the learning rate is matched on the value stored in ``run_config.json``.
    """
    for d in sorted((results_dir / "training").glob(f"lr_*_seed_{seed}")):
        v = _run_lr(d)
        if v is not None and abs(v - lr) <= 1e-12 * max(1.0, abs(lr)) and (d / "progress.csv").exists():
            return d
    return None


def load_episodes(results_dir: Path, scenarios: List[str]) -> pd.DataFrame:
    frames = []
    for sc in scenarios:
        p = results_dir / sc / "episodes.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["scenario"] = sc
            frames.append(df)
        else:
            print(f"missing {p}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["policy"] = pd.Categorical(df["policy"], [p for p in POLICY_ORDER if p in set(df["policy"])] + sorted(set(df["policy"]) - set(POLICY_ORDER)))
    return df


def load_steps(results_dir: Path, scenario: str) -> Optional[pd.DataFrame]:
    p = results_dir / scenario / "steps.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ----------------------------------------------------------------------------
# tables and tests
def summary_table(ep: pd.DataFrame, is_metric: str) -> pd.DataFrame:
    g = ep.groupby(["scenario", "policy"], observed=True)
    out = {
        "n": g.size(),
        "E(IS)": g[is_metric].mean(),
        "E(Pen)": g["terminal_penalty"].mean(),
        "E(T)": g["time_pct"].mean(),
        "Var(IS)": g[is_metric].var(ddof=1),
        "Std(IS)": g[is_metric].std(ddof=1),
        "E(depth penalty)": g["depth_penalty"].mean(),
        "E(reward)": g["episode_reward"].mean(),
        "E(executed)": g["executed_quantity"].mean(),
        "P(complete)": g["remaining_quantity"].apply(lambda s: float((s <= 0).mean())),
        "E(steps)": g["num_steps"].mean(),
        "E(trading steps)": g["num_trading_steps"].mean(),
    }
    if "num_unfilled_orders" in ep.columns:
        # market orders that met no resting liquidity (must be 0 in a valid run)
        out["Unfilled orders"] = g["num_unfilled_orders"].sum()
        out["Unfilled shares"] = g["unfilled_quantity"].sum()
    out = pd.DataFrame(out).reset_index()
    return out


def pooled_t_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """One-sided two-sample Student t-test with pooled variance, H1: mean(x) > mean(y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    t = (x.mean() - y.mean()) / np.sqrt(sp2 * (1 / nx + 1 / ny))
    df = nx + ny - 2
    p = 1 - stats.t.cdf(t, df)
    tw, pw = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    return {"t": t, "df": df, "p_one_sided": p, "t_crit_0.95": stats.t.ppf(0.95, df), "welch_t": tw, "welch_p": pw}


def t_tests(ep: pd.DataFrame, is_metric: str) -> pd.DataFrame:
    rows = []
    for sc, d in ep.groupby("scenario"):
        if "RL" not in set(d["policy"]):
            continue
        rl = d.loc[d["policy"] == "RL", is_metric].values
        for pol in ["TWAP", "Random", "Passive"]:
            if pol not in set(d["policy"]):
                continue
            other = d.loc[d["policy"] == pol, is_metric].values
            r = pooled_t_test(rl, other)
            # paired version: the same evaluation seeds are used for every policy
            rl_s = d[d["policy"] == "RL"].set_index("seed")[is_metric]
            ot_s = d[d["policy"] == pol].set_index("seed")[is_metric]
            common = rl_s.index.intersection(ot_s.index)
            if len(common) > 2:
                tp, pp = stats.ttest_rel(rl_s.loc[common].values, ot_s.loc[common].values, alternative="greater")
            else:
                tp, pp = np.nan, np.nan
            rows.append({"scenario": sc, "benchmark": pol, "n_RL": len(rl), "n_bench": len(other),
                         "mean_RL": rl.mean(), "mean_bench": other.mean(), **r,
                         "significant_5pct": bool(r["t"] > r["t_crit_0.95"]),
                         "paired_t": tp, "paired_p_one_sided": pp})
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# figures
def fig_price_paths(results_dir: Path, figdir: Path, figures: Dict[str, Path]) -> None:
    for name, key, title in [("price_paths_4h.parquet", "fig3", "Fig. 3: sample paths of the best ask (no trading), 4 hours at 1 s"),
                             ("price_paths_30min.parquet", "figS3_30min", "Fig. S3: sample paths of the best ask (no trading), 30 minutes at 1 s")]:
        p = results_dir / name
        if not p.exists():
            print(f"missing {p}")
            continue
        df = pd.read_parquet(p)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (seed, d) in enumerate(df.groupby("seed")):
            ax.plot(d["step"], d["best_ask"], color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=1.2, label=f"seed {seed}")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        _tidy(ax, "Time-steps (1 s)", "Ask price (cents)", title)
        ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18))
        _save(fig, figdir / f"{key}_price_paths.png", figures, key)


def _robust_ylim(ax, lo_q: float = 0.01, hi_q: float = 0.99) -> None:
    """Limit the y-axis to the [lo_q, hi_q] quantile range of the plotted y-data.

    A single diverged training episode (reward of order -10^4) would otherwise
    flatten every curve; the clipped range is stated in the y-label.
    """
    ys = []
    for line in ax.get_lines():
        ys.append(np.asarray(line.get_ydata(), dtype=float))
    for coll in ax.collections:
        offs = coll.get_offsets()
        if len(offs):
            ys.append(np.asarray(offs)[:, 1].astype(float))
    if not ys:
        return
    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return
    lo, hi = np.quantile(y, lo_q), np.quantile(y, hi_q)
    if y.min() < lo - 0.5 * (hi - lo) or y.max() > hi + 0.5 * (hi - lo):
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ax.get_ylabel() + f"\n(axis clipped to the {int(lo_q*100)}-{int(hi_q*100)}% range of the data)")


def _episode_curve(run_dir: Path, window: int = 10):
    """Per-episode rewards of a training run and their rolling mean (x = environment steps)."""
    p = run_dir / "episodes.csv"
    if not p.exists():
        return None
    epi = pd.read_csv(p)
    if epi.empty:
        return None
    epi = epi.sort_values("timesteps_total_at_end")
    epi["rolling"] = epi["episode_reward"].rolling(window, min_periods=1).mean()
    return epi


def fig_learning_curves(results_dir: Path, figdir: Path, figures: Dict[str, Path], cfg: Dict, window: int = 10) -> None:
    """Fig. 4: training curves.

    The main panel plots the per-episode rewards (``episodes.csv``) smoothed
    with a rolling mean over ``window`` episodes; RLlib's ``episode_reward_mean``
    (a trailing mean over 100 episodes) is kept as a supplementary figure.
    """
    tr = cfg.get("training", {})
    seed = int(tr.get("seed", 10))
    lrs = tr.get("learning_rates", [1e-2, 1e-3, 1e-4])
    sel = float(tr.get("selected_learning_rate", 1e-3))

    # Fig. 4: rolling mean of the episode rewards, one curve per learning rate
    fig, ax = plt.subplots(figsize=(8, 4.2))
    any_curve = False
    for i, lr in enumerate(lrs):
        d = find_training_run(results_dir, float(lr), seed)
        epi = _episode_curve(d, window) if d else None
        if epi is None:
            print(f"missing training run lr={lr:g} seed={seed} under {results_dir / 'training'}")
            continue
        ax.scatter(epi["timesteps_total_at_end"], epi["episode_reward"], s=10, color=CATEGORICAL[i], alpha=0.25, edgecolors="none")
        ax.plot(epi["timesteps_total_at_end"], epi["rolling"], color=CATEGORICAL[i], label=f"lr = {lr:g} ({len(epi)} episodes)")
        any_curve = True
    if any_curve:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        _tidy(ax, "Environment steps", f"Episode reward (dots) and rolling mean over {window} episodes",
              f"Fig. 4: DQN training reward, environment seed {seed}")
        _robust_ylim(ax)
        ax.legend(loc="lower right")
        _save(fig, figdir / "fig4_learning_curves.png", figures, "fig4")
    else:
        plt.close(fig)

    # Fig. 4b: seed robustness of the selected learning rate
    runs = sorted(d for d in results_dir.glob("training/lr_*_seed_*") if _run_lr(d) is not None and abs(_run_lr(d) - sel) <= 1e-12)
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, d in enumerate(runs):
            epi = _episode_curve(d, window)
            if epi is None:
                continue
            ax.plot(epi["timesteps_total_at_end"], epi["rolling"], color=CATEGORICAL[i % 8], label=d.name.replace("_", " "))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        _tidy(ax, "Environment steps", f"Rolling mean episode reward ({window} episodes)",
              f"Fig. 4b: seed robustness of DQN training at lr = {sel:g}")
        _robust_ylim(ax)
        ax.legend(loc="lower right")
        _save(fig, figdir / "fig4b_learning_curves_seeds.png", figures, "fig4b")

    # Fig. S4: RLlib's episode_reward_mean (trailing 100 episodes = cumulative mean here)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    any_curve = False
    for i, lr in enumerate(lrs):
        d = find_training_run(results_dir, float(lr), seed)
        if d is None:
            continue
        prog = pd.read_csv(d / "progress.csv")
        ax.plot(prog["timesteps_total"], prog["episode_reward_mean"], color=CATEGORICAL[i], label=f"lr = {lr:g}")
        any_curve = True
    if any_curve:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        _tidy(ax, "Environment steps", "RLlib episode_reward_mean (trailing 100 episodes)",
              f"Fig. S4: RLlib's smoothed training reward (trailing 100 episodes), environment seed {seed}")
        _robust_ylim(ax)
        ax.legend(loc="lower right")
        _save(fig, figdir / "figS4_rllib_reward_mean.png", figures, "figS4")
    else:
        plt.close(fig)


def fig_is_distribution(ep: pd.DataFrame, scenario: str, is_metric: str, figdir: Path, figures: Dict[str, Path],
                        key: str, title: str, n_plot: int) -> None:
    d = ep[(ep["scenario"] == scenario)]
    if d.empty:
        return
    d = d.sort_values("seed").groupby("policy", observed=True).head(n_plot)
    pols = [p for p in POLICY_ORDER if p in set(d["policy"])]
    vals = d[is_metric].values
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = 0.15 * (hi - lo if hi > lo else 1)
    grid = np.linspace(lo - pad, hi + pad, 400)
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    rng = np.random.RandomState(0)
    for i, pol in enumerate(pols):
        v = d.loc[d["policy"] == pol, is_metric].values
        _kde(ax, v, POLICY_COLOR[pol], f"{pol} (mean {v.mean():+.3g}, n = {len(v)})", grid)
        _strip(axs, v, POLICY_COLOR[pol], len(pols) - 1 - i, rng)
        axs.plot([v.mean(), v.mean()], [len(pols) - 1 - i - 0.3, len(pols) - 1 - i + 0.3], color=POLICY_COLOR[pol], linewidth=2)
    axs.set_yticks(range(len(pols)))
    axs.set_yticklabels(list(reversed(pols)))
    axs.grid(False)
    axs.set_xlabel(f"{IS_METRICS.get(is_metric, is_metric)}")
    axs.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
    _tidy(ax, None, "Density", title)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
    ax.legend(loc="upper left")
    _save(fig, figdir / f"{key}.png", figures, key)


def fig_robustness(ep: pd.DataFrame, is_metric: str, figdir: Path, figures: Dict[str, Path], key: str, title: str, n_plot: int) -> None:
    panels = [("noise_10", "(a) 10 noise agents"), ("standard", "(b) 1000 noise agents"), ("noise_2000", "(c) 2000 noise agents"),
              ("momentum_6", "(d) 6 momentum agents"), ("standard", "(e) 12 momentum agents"), ("momentum_24", "(f) 24 momentum agents")]
    have = [sc for sc, _ in panels if sc in set(ep["scenario"])]
    if not have:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=False)
    for ax, (sc, sub) in zip(axes.ravel(), panels):
        d = ep[ep["scenario"] == sc].sort_values("seed").groupby("policy", observed=True).head(n_plot)
        if d.empty:
            ax.set_visible(False)
            continue
        vals = d[is_metric].values
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = 0.15 * (hi - lo if hi > lo else 1)
        grid = np.linspace(lo - pad, hi + pad, 300)
        for pol in [p for p in POLICY_ORDER if p in set(d["policy"])]:
            _kde(ax, d.loc[d["policy"] == pol, is_metric].values, POLICY_COLOR[pol], pol, grid)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
        _tidy(ax, IS_METRICS.get(is_metric, is_metric) if sc != "standard" or sub.startswith("(e)") else None, "Density", sub)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, figdir / f"{key}.png", figures, key)


def fig_trajectory(steps: pd.DataFrame, figdir: Path, figures: Dict[str, Path], seed: Optional[int] = None) -> None:
    rl = steps[steps["policy"] == "RL"]
    if rl.empty:
        return
    if seed is None:
        # the most illustrative episode: the one with the largest number of trading steps
        trades = rl[rl["step_executed_quantity"] > 0].groupby("seed").size()
        seed = int(trades.idxmax()) if len(trades) else int(rl["seed"].min())
    d = rl[rl["seed"] == seed].sort_values("step")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(d["step"], d["obs_best_bid"], color=INK2, linewidth=1.4, label="best bid")
    ax.plot(d["step"], d["obs_best_ask"], color=MUTED, linewidth=1.4, label="best ask")
    for k in range(1, 5):
        dk = d[d["action"] == k]
        if dk.empty:
            continue
        ax.scatter(dk["step"], dk["step_avg_fill_price"].where(dk["step_executed_quantity"] > 0, dk["obs_best_ask"]),
                   s=28 + 10 * k, color=SEQ_BLUE[k - 1], edgecolors=SURFACE, linewidths=1.0, zorder=3, label=f"action {k} ({20*k} shares)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _tidy(ax, "Time-step (1 s) since arrival", "Price (cents)", f"Fig. 6a: bid, ask and RL actions along one episode (seed {seed}, {int((d['step_executed_quantity'] > 0).sum())} trading steps)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18))
    _save(fig, figdir / "fig6a_trajectory.png", figures, "fig6a")


def fig_execution_profile(steps: pd.DataFrame, figdir: Path, figures: Dict[str, Path], n_plot: int) -> None:
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pol in [p for p in POLICY_ORDER if p in set(steps["policy"])]:
        d = steps[steps["policy"] == pol]
        seeds = sorted(d["seed"].unique())[:n_plot]
        curves = []
        for s in seeds:
            e = d[d["seed"] == s].sort_values("step")
            t = np.concatenate([[0], e["time_pct"].values, [1.0]])
            h = np.concatenate([[0], np.minimum(e["holdings_pct"].values, 1.0), [min(e["holdings_pct"].iloc[-1], 1.0)]])
            curves.append(np.interp(grid, t, h))
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        if pol == "RL":
            se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
            ci = stats.t.ppf(0.975, max(1, len(curves) - 1)) * se
            ax.fill_between(grid, mean - ci, mean + ci, color=POLICY_COLOR[pol], alpha=0.15, linewidth=0, label="RL 95% CI")
            ax.plot(grid, mean, color=POLICY_COLOR[pol], label=f"RL average ({len(curves)} episodes)")
        else:
            ax.plot(grid, mean, color=POLICY_COLOR[pol], linewidth=1.4, label=f"{pol} average")
    _tidy(ax, "Time (fraction of the 30-minute window)", "Executed fraction of the parent order", "Fig. 6b: execution trajectory")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _save(fig, figdir / "fig6b_execution_profile.png", figures, "fig6b")


def fig_q_values(steps: pd.DataFrame, figdir: Path, figures: Dict[str, Path], n_plot: int) -> None:
    rl = steps[steps["policy"] == "RL"]
    qcols = [c for c in rl.columns if c.startswith("q_")]
    if rl.empty or not qcols:
        return
    seeds = sorted(rl["seed"].unique())[:n_plot]
    rl = rl[rl["seed"].isin(seeds)].copy()
    rl["tbin"] = np.clip((rl["obs_time_pct"] * 40).astype(int), 0, 39)
    agg = rl.groupby("tbin")[qcols].mean()
    x = (agg.index.values + 0.5) / 40
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, c in enumerate(sorted(qcols, key=lambda s: int(s.split("_")[1]))):
        ax.plot(x, agg[c], color=CATEGORICAL[k], label=f"action {k}" + (" (nothing)" if k == 0 else f" ({20*k} shares)"))
    _tidy(ax, "Time (fraction of the window)", "Q-value (mean over episodes and 2.5% time bins)", "Fig. 6c: Q-values along the episode")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18))
    _save(fig, figdir / "fig6c_q_values.png", figures, "fig6c")


def fig_spreads(steps: pd.DataFrame, figdir: Path, figures: Dict[str, Path], key: str, title: str, trading_only: bool) -> None:
    d = steps[steps["step_executed_quantity"] > 0] if trading_only else steps
    pols = [p for p in POLICY_ORDER if p in set(d["policy"])]
    if not pols:
        return
    d = d.copy()
    d["spread_c"] = d["obs_best_ask"] - d["obs_best_bid"]
    max_s = int(min(15, np.nanpercentile(d["spread_c"], 99))) if len(d) else 10
    bins = np.arange(0, max_s + 2)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(pols)
    for i, pol in enumerate(pols):
        v = d.loc[d["policy"] == pol, "spread_c"].clip(upper=max_s + 1).values
        hist, _ = np.histogram(v, bins=bins)
        freq = hist / max(1, hist.sum())
        ax.bar(bins[:-1] + (i - (len(pols) - 1) / 2) * width, freq, width=width * 0.9, color=POLICY_COLOR[pol], label=pol,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(b) for b in bins[:-2]] + [f">{max_s}"])
    _tidy(ax, "Bid-ask spread observed at the decision time (cents)", "Frequency", title)
    ax.legend(ncol=4, loc="upper right")
    _save(fig, figdir / f"{key}.png", figures, key)


def fig_imbalance(steps: pd.DataFrame, figdir: Path, figures: Dict[str, Path], key: str, title: str, trading_only: bool) -> None:
    d = steps[steps["step_executed_quantity"] > 0] if trading_only else steps
    pols = [p for p in POLICY_ORDER if p in set(d["policy"])]
    if not pols:
        return
    grid = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for pol in pols:
        _kde(ax, d.loc[d["policy"] == pol, "obs_imbalance_5"].values, POLICY_COLOR[pol], pol, grid)
    ax.axvline(0.5, color=AXIS, linewidth=0.8, zorder=0)
    _tidy(ax, "Volume imbalance over the best 5 levels (bid share) at the decision time", "Density", title)
    ax.legend(loc="upper left")
    _save(fig, figdir / f"{key}.png", figures, key)


# ----------------------------------------------------------------------------
# report
def _img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}" style="max-width:100%;height:auto;">'


def _md_to_html(md_text: str) -> str:
    try:
        import markdown  # type: ignore

        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        import html

        return f"<pre>{html.escape(md_text)}</pre>"


def write_report(out: Path, cfg: Dict, figures: Dict[str, Path], tables: Dict[str, pd.DataFrame], analysis_md: Optional[Path],
                 meta: Dict) -> None:
    css = f"""
    body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; color: {INK}; background: #f9f9f7; margin: 0; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
    h1, h2, h3 {{ color: {INK}; }} h2 {{ margin-top: 40px; border-bottom: 1px solid {GRID}; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; }} th, td {{ padding: 5px 10px; border-bottom: 1px solid {GRID}; text-align: right; font-variant-numeric: tabular-nums; }}
    th {{ color: {INK2}; font-weight: 600; }} td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
    figure {{ margin: 18px 0; background: {SURFACE}; padding: 12px; border: 1px solid {GRID}; border-radius: 6px; }}
    figcaption {{ color: {INK2}; font-size: 13px; margin-top: 6px; }}
    .meta {{ color: {INK2}; font-size: 14px; }} code {{ background: #efeeea; padding: 1px 4px; border-radius: 3px; }}
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Reproduction report: {cfg['name']}</title><style>{css}</style></head><body><main>"]
    parts.append("<h1>Optimal Execution with Reinforcement Learning in a Multi-Agent Market Simulator: results report</h1>")
    parts.append(f"<p class='meta'>Configuration <code>{cfg['name']}</code>. Generated by <code>experiments/make_figures.py</code>. "
                 f"Episodes per (scenario, policy): {meta.get('n_episodes', {})}. Distribution plots use the first {meta.get('n_plot')} seeds; tables and tests use all seeds.</p>")
    parts.append("<h2>Tables</h2>")
    for name, df in tables.items():
        parts.append(f"<h3>{name}</h3>")
        parts.append(df.to_html(index=False, float_format=lambda v: f"{v:.4g}", na_rep="", border=0))
    parts.append("<h2>Figures</h2>")
    for key in ["fig3", "fig4", "fig4b", "figS4", "fig5", "figS5_step_mean", "figS5_last_step", "fig6a", "fig6b", "fig6c", "fig7a", "fig7b",
                "figS7a_all_steps", "figS7b_all_steps", "fig8", "figS8_step_mean", "figS3_30min"]:
        if key in figures:
            parts.append(f"<figure>{_img_tag(figures[key])}<figcaption>{figures[key].name}</figcaption></figure>")
    parts.append("</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/paper.yaml"))
    parser.add_argument("--results", default=None, help="results directory (default results/<config name>)")
    parser.add_argument("--figures", default=None, help="figure directory (default figures/<config name>)")
    parser.add_argument("--report", default=None, help="HTML report path (default results/<config>/report.html)")
    parser.add_argument("--is-metric", default="implementation_shortfall", choices=list(IS_METRICS),
                        help="IS statistic used for the main figures and tables")
    parser.add_argument("--trajectory-seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results_dir = Path(args.results) if args.results else REPO_ROOT / "results" / cfg["name"]
    figdir = Path(args.figures) if args.figures else REPO_ROOT / "figures" / cfg["name"]
    report_path = Path(args.report) if args.report else results_dir / "report.html"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir = results_dir / "tables"
    tabdir.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("evaluation", {})
    n_plot = int(ev.get("plot_seeds", 20))
    scenarios = list(ev.get("scenarios", {"standard": {}}).keys())
    figures: Dict[str, Path] = {}
    tables: Dict[str, pd.DataFrame] = {}

    # Fig 3 and Fig 4 do not need evaluation results
    fig_price_paths(results_dir, figdir, figures)
    fig_learning_curves(results_dir, figdir, figures, cfg)

    ep = load_episodes(results_dir, scenarios)
    meta = {"n_plot": n_plot, "n_episodes": {}}
    if not ep.empty:
        meta["n_episodes"] = ep.groupby(["scenario", "policy"], observed=True).size().to_dict()
        meta["n_episodes"] = {f"{k[0]}/{k[1]}": int(v) for k, v in meta["n_episodes"].items()}
        # tables (all IS definitions) -----------------------------------
        for metric, label in IS_METRICS.items():
            summ = summary_table(ep, metric)
            summ.to_csv(tabdir / f"summary_{metric}.csv", index=False)
            if metric == args.is_metric:
                tables[f"Summary by scenario and policy, IS = {label}"] = summ
                for tname, scs in [("table1", ["standard"]), ("table2", ["noise_10", "noise_2000"]), ("table3", ["momentum_6", "momentum_24"])]:
                    t = summ[summ["scenario"].isin(scs)][["scenario", "policy", "n", "E(IS)", "E(Pen)", "E(T)", "Var(IS)"]]
                    t.to_csv(tabdir / f"{tname}.csv", index=False)
                    (tabdir / f"{tname}.md").write_text(df_to_markdown(t))
            tt = t_tests(ep, metric)
            if len(tt):
                tt.to_csv(tabdir / f"ttests_{metric}.csv", index=False)
                (tabdir / f"ttests_{metric}.md").write_text(df_to_markdown(tt))
                if metric == args.is_metric:
                    tables[f"One-sided pooled t-tests, H1: E(IS)_RL > E(IS)_benchmark, IS = {label}"] = tt
        ep.to_csv(tabdir / "all_episodes.csv", index=False)
        # figures ---------------------------------------------------------
        fig_is_distribution(ep, "standard", args.is_metric, figdir, figures, "fig5_is_distribution",
                            f"Fig. 5: implementation shortfall normalised by X0, {SCENARIO_TITLES['standard']}, {n_plot} windows", n_plot)
        fig_is_distribution(ep, "standard", "step_is_mean", figdir, figures, "figS5_step_mean",
                            f"Fig. S5a: mean per-step normalised IS, {n_plot} windows", n_plot)
        fig_is_distribution(ep, "standard", "step_is_final", figdir, figures, "figS5_last_step",
                            f"Fig. S5b: last-step normalised IS, {n_plot} windows", n_plot)
        fig_robustness(ep, args.is_metric, figdir, figures, "fig8_robustness",
                       f"Fig. 8: normalised implementation shortfall across background-agent configurations, {n_plot} windows each", n_plot)
        fig_robustness(ep, "step_is_mean", figdir, figures, "figS8_step_mean",
                       f"Fig. S8: mean per-step normalised IS across background-agent configurations, {n_plot} windows each", n_plot)
        steps = load_steps(results_dir, "standard")
        if steps is None:
            print(f"no per-step log at {results_dir / 'standard' / 'steps.parquet'} (not committed to git); "
                  "Fig. 6a-c, 7a-b and S7 need `python -m experiments.evaluate ...` to be run first")
        if steps is not None and len(steps):
            fig_trajectory(steps, figdir, figures, args.trajectory_seed)
            fig_execution_profile(steps, figdir, figures, n_plot)
            fig_q_values(steps, figdir, figures, n_plot)
            fig_spreads(steps, figdir, figures, "fig7a_spreads", "Fig. 7a: spread at the steps where the policy trades", True)
            fig_imbalance(steps, figdir, figures, "fig7b_imbalance", "Fig. 7b: volume imbalance at the steps where the policy trades", True)
            fig_spreads(steps, figdir, figures, "figS7a_all_steps", "Fig. S7a: spread at every decision step", False)
            fig_imbalance(steps, figdir, figures, "figS7b_all_steps", "Fig. S7b: volume imbalance at every decision step", False)
    else:
        print("no evaluation results found; only Fig. 3 / Fig. 4 produced")

    write_report(report_path, cfg, figures, tables, None, meta)
    def _rel(v: Path) -> str:
        try:
            return str(v.relative_to(REPO_ROOT))
        except ValueError:
            return str(v)

    (tabdir / "figures_index.json").write_text(json.dumps({k: _rel(v) for k, v in figures.items()}, indent=2))


if __name__ == "__main__":
    main()
