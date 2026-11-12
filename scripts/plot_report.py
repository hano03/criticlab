"""The standard criticlab figure for one or more runs.

Usage (from the repo root; wildcards are expanded here, so they also work in PowerShell):
    python scripts/plot_report.py "runs/Hopper-v5__*" --out figures/hopper.png
    python scripts/plot_report.py runs --out figures/all.png     # every run under runs/

Runs are grouped by algorithm (TD3 without clipped double-Q counts as its own group) and
drawn as mean +/- sample std across seeds. Six panels:
    1 return                 2 normalised Q bias       3 calibration at the final probe
    4 rank calibration       5 twin disagreement       6 TD error |delta|
It also writes <out>_hero.png: the README figure, built to be robust twice over: the median
over states within each probe (--hero-metric), then the median and interquartile range across
seeds (--hero-agg). Panels use mean ± sample std across seeds.
If the runs span several environments, one figure per environment is written (<out>__<env>.png).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour follows the algorithm, never its rank (validated categorical slots 1-4, light mode).
COLORS = {"td3": "#2a78d6", "ddpg": "#eb6834", "sac": "#1baf7a", "td3_no_cdq": "#eda100"}
LABELS = {"td3": "TD3", "ddpg": "DDPG", "sac": "SAC", "td3_no_cdq": "TD3 w/o clipped double-Q"}
STYLES = {"td3_no_cdq": "--"}  # secondary encoding for the ablation
ORDER = ["ddpg", "td3_no_cdq", "td3", "sac"]
FALLBACK = "#6250d6"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def find_reports(inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        for match in (glob.glob(item) or [item]):
            p = Path(match)
            if p.is_file() and p.suffix == ".json":
                paths.add(p)
            elif (p / "critic_report.json").is_file():
                paths.add(p / "critic_report.json")
            elif p.is_dir():
                paths.update(p.rglob("critic_report.json"))
    # runs set aside by run_week2.py as unfinished never enter a figure
    return sorted(p for p in paths if "_incomplete" not in p.parts)


def group_key(meta: dict) -> str:
    algo = meta.get("algo", "unknown")
    if algo == "td3" and meta.get("clipped_double_q") is False:
        return "td3_no_cdq"
    return algo


def stack(reports: list[dict], key: str):
    """steps, mean, std of one metric across seeds (NaN where a seed lacks it)."""
    steps = sorted({p["step"] for r in reports for p in r["probes"]})
    grid = np.full((len(reports), len(steps)), np.nan)
    col = {s: j for j, s in enumerate(steps)}
    for i, r in enumerate(reports):
        for p in r["probes"]:
            v = p["metrics"].get(key)
            if v is not None:
                grid[i, col[p["step"]]] = v
    with np.errstate(all="ignore"), _quiet():
        mean = np.nanmean(grid, axis=0)
        count = np.sum(np.isfinite(grid), axis=0)
        std = np.where(count > 1, np.nanstd(grid, axis=0, ddof=1), 0.0)  # sample std across seeds
    return np.array(steps), mean, std


def stack_quantiles(reports: list[dict], key: str):
    """steps, median, 25th and 75th percentile of one metric across seeds (robust to one bad seed)."""
    steps, _, _ = stack(reports, key)
    grid = np.full((len(reports), len(steps)), np.nan)
    col = {s: j for j, s in enumerate(steps)}
    for i, r in enumerate(reports):
        for p in r["probes"]:
            v = p["metrics"].get(key)
            if v is not None:
                grid[i, col[p["step"]]] = v
    with np.errstate(all="ignore"), _quiet():
        q25, med, q75 = np.nanpercentile(grid, [25, 50, 75], axis=0)
    return steps, med, q25, q75


class _quiet:
    """Silence 'Mean of empty slice' warnings for steps no seed has."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def return_key(reports: list[dict]) -> str:
    # SAC's probe runs the stochastic policy; its deterministic evaluation is the comparable return
    has_eval = any("eval/deterministic_return_mean" in p["metrics"] for r in reports for p in r["probes"])
    return "eval/deterministic_return_mean" if has_eval else "probe/return_mean"


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line_with_band(ax, steps, mean, std, key: str, lo=None, hi=None):
    ok = np.isfinite(mean)
    if not ok.any():
        return None
    color = COLORS.get(key, FALLBACK)
    if lo is None:
        lo, hi = mean - std, mean + std
    ax.fill_between(steps[ok], lo[ok], hi[ok], color=color, alpha=0.15, linewidth=0)
    (line,) = ax.plot(steps[ok], mean[ok], color=color, linewidth=1.8, linestyle=STYLES.get(key, "-"))
    return line


def direct_labels(ax, ends: list[tuple[float, float, str]]) -> None:
    """Label each line at its right end in text ink, nudged apart so labels never overlap."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    placed: list[float] = []
    for _, y, _ in ends:  # push apart upwards...
        placed.append(max(y, placed[-1] + gap) if placed else y)
    # ...then shift each crowded cluster back down so it is centred on its lines, not above them
    i = 0
    while i < len(placed):
        j = i
        while j + 1 < len(placed) and placed[j + 1] - placed[j] <= gap + 1e-12:
            j += 1
        shift = np.mean([placed[k] - ends[k][1] for k in range(i, j + 1)])
        for k in range(i, j + 1):
            placed[k] -= shift
        i = j + 1
    for (x, _, text), y_text in zip(ends, placed):
        ax.text(x, y_text, "  " + text, va="center", ha="left", fontsize=8, color=INK_2, clip_on=False)


def calibration(ax, reports: list[dict], key: str, n_bins: int = 10):
    """Mean Monte Carlo return per bin of predicted Q, pooled over seeds at each seed's final probe."""
    q = np.concatenate([np.asarray(r["probes"][-1]["raw"]["q"], float) for r in reports])
    g = np.concatenate([np.asarray(r["probes"][-1]["raw"]["g"], float) for r in reports])
    if q.size < n_bins:
        return None, None
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, n_bins - 1)
    qm = np.array([q[idx == b].mean() for b in range(n_bins) if (idx == b).any()])
    gm = np.array([g[idx == b].mean() for b in range(n_bins) if (idx == b).any()])
    color = COLORS.get(key, FALLBACK)
    (line,) = ax.plot(qm, gm, color=color, linewidth=1.8, linestyle=STYLES.get(key, "-"),
                      marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    return line, (min(qm.min(), gm.min()), max(qm.max(), gm.max()))


def figure_for_env(env_id: str, groups: dict[str, list[dict]], out: Path, hero: bool,
                   hero_metric: str = "median", hero_agg: str = "median") -> list[Path]:
    keys = [k for k in ORDER if k in groups] + sorted(k for k in groups if k not in ORDER)
    seeds = {k: len(groups[k]) for k in keys}
    name = {k: f"{LABELS.get(k, k)} ({seeds[k]} seed{'s' if seeds[k] != 1 else ''})" for k in keys}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), dpi=150, facecolor=SURFACE)
    (a_ret, a_bias, a_cal), (a_rank, a_twin, a_td) = axes
    panels = [
        (a_ret, "1  Return", "episode return", None),
        (a_bias, "2  Normalised Q bias  (> 0 = overestimation)", "(Q − G) / max(|G|, 10)", "bias/norm_bias_mean"),
        (a_rank, "4  Rank calibration", "Spearman(Q, G)", "bias/spearman"),
        (a_twin, "5  Twin-critic disagreement", "mean |Q1 − Q2|", "twin/abs_diff_mean"),
        (a_td, "6  TD error on replay batches", "mean |δ|", "td/abs_mean"),
    ]
    handles = {}
    bias_ends = []
    for ax, title, ylabel, metric in panels:
        style_axis(ax, title, ylabel)
        drew = False
        for k in keys:
            m = metric or return_key(groups[k])
            steps, mean, std = stack(groups[k], m)
            line = line_with_band(ax, steps, mean, std, k)
            if line is not None:
                drew = True
                handles.setdefault(k, line)
                if metric == "bias/norm_bias_mean":
                    last = np.where(np.isfinite(mean))[0][-1]
                    bias_ends.append((steps[last], mean[last], LABELS.get(k, k)))
        if metric == "bias/norm_bias_mean":
            ax.axhline(0, color=INK_2, linewidth=1, zorder=1)
        if not drew:
            ax.text(0.5, 0.5, "not recorded for these runs", transform=ax.transAxes,
                    ha="center", va="center", color=INK_2, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("environment steps", color=INK_2, fontsize=9)
    direct_labels(a_bias, bias_ends)

    style_axis(a_cal, "3  Calibration at the final probe", "mean Monte Carlo return G")
    span = []
    for k in keys:
        line, rng = calibration(a_cal, groups[k], k)
        if line is not None:
            handles.setdefault(k, line)
            span.append(rng)
    if span:
        lo, hi = min(s[0] for s in span), max(s[1] for s in span)
        a_cal.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle=":", zorder=1)
        # key in the bottom-right corner, inside the axes: never collides with the axis title
        a_cal.text(0.98, 0.03, "dotted line: perfect calibration (G = Q)", transform=a_cal.transAxes,
                   ha="right", va="bottom", fontsize=8, color=INK_2)
    a_cal.set_xlabel("predicted Q (10 equal-count bins)", color=INK_2, fontsize=9)

    fig.legend([handles[k] for k in keys if k in handles], [name[k] for k in keys if k in handles],
               loc="upper center", ncol=len(handles), frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"{env_id}: critic report", x=0.01, y=0.995, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.985, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    written = [out]

    if hero:
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
        # median per probe by default: robust to the few near-terminal states that dominate the
        # mean when a policy briefly collapses (see docs/diagnostics.md, section 3)
        metric = f"bias/norm_bias_{hero_metric}"
        if not any(metric in p["metrics"] for g in groups.values() for r in g for p in r["probes"]):
            metric, hero_metric = "bias/norm_bias_mean", "mean"  # older runs have no median
        style_axis(ax, f"{env_id}: normalised Q bias ({hero_metric} per probe)",
                   "(Q − G) / max(|G|, 10)")
        ends, hh = [], []
        for k in keys:
            if hero_agg == "median":
                # median across seeds, band = interquartile range: one collapsed seed cannot drive it
                steps, centre, lo, hi = stack_quantiles(groups[k], metric)
                line = line_with_band(ax, steps, centre, None, k, lo=lo, hi=hi)
            else:
                steps, centre, std = stack(groups[k], metric)
                line = line_with_band(ax, steps, centre, std, k)
            if line is not None:
                hh.append((line, name[k]))
                last = np.where(np.isfinite(centre))[0][-1]
                ends.append((steps[last], centre[last], LABELS.get(k, k)))
        ax.axhline(0, color=INK_2, linewidth=1, zorder=1)
        ax.text(0.005, 0.0, " unbiased", transform=ax.get_yaxis_transform(), va="bottom",
                fontsize=8, color=INK_2)
        ax.set_xlabel("environment steps", color=INK_2, fontsize=9)
        if hh:
            leg = ax.legend([h for h, _ in hh], [t for _, t in hh], frameon=False, fontsize=9,
                            labelcolor=INK, loc="upper right",
                            title=("line: median across seeds · band: interquartile range"
                                   if hero_agg == "median" else "line: mean across seeds · band: ± std"),
                            title_fontsize=8,
                            alignment="left")
            leg.get_title().set_color(INK_2)
        direct_labels(ax, ends)
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        hero_path = out.with_name(out.stem + "_hero" + out.suffix)
        fig.savefig(hero_path, facecolor=SURFACE)
        plt.close(fig)
        written.append(hero_path)
    return written


def plot(inputs: list[str], out: str, hero: bool = True, hero_metric: str = "median",
         hero_agg: str = "median") -> list[Path]:
    paths = find_reports(inputs)
    if not paths:
        raise SystemExit(f"no critic_report.json found under: {' '.join(inputs)}")
    by_env: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        report = json.loads(p.read_text(encoding="utf-8"))
        if report.get("probes"):
            by_env[report["meta"]["env_id"]][group_key(report["meta"])].append(report)
    out_path = Path(out)
    written = []
    for env_id, groups in sorted(by_env.items()):
        target = out_path if len(by_env) == 1 else out_path.with_name(
            f"{out_path.stem}__{env_id}{out_path.suffix}")
        written += figure_for_env(env_id, groups, target, hero, hero_metric, hero_agg)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", help="run folders, critic_report.json files, or wildcards")
    parser.add_argument("--out", default=os.path.join("figures", "critic_report.png"))
    parser.add_argument("--no-hero", action="store_true", help="skip the single-panel README figure")
    parser.add_argument("--hero-metric", choices=["median", "mean"], default="median",
                        help="per-probe statistic in the hero figure (panel 2 always shows the mean)")
    parser.add_argument("--hero-agg", choices=["median", "mean"], default="median",
                        help="across seeds: median + interquartile band (default), or mean ± std")
    args = parser.parse_args()
    for path in plot(args.inputs, args.out, hero=not args.no_hero, hero_metric=args.hero_metric,
                     hero_agg=args.hero_agg):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
