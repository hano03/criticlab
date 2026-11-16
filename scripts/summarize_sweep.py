"""Per-seed results table for a sweep, with Welch t-tests, ready to paste into a findings note.

    python scripts/summarize_sweep.py runs/week3                       # every group, final probe
    python scripts/summarize_sweep.py runs/week3 --ref ppo             # + Welch tests against PPO, lambda 0.95
    python scripts/summarize_sweep.py runs/week2 runs/week3 --at-step 300000 --only ddpg td3 td3_no_cdq sac ppo
    python scripts/summarize_sweep.py runs/week3 --ref ppo --out figures/week3_summary.md
    python scripts/summarize_sweep.py "runs/week3/Hopper-v5__*" --window 20000 250000 --ref ppo

With --window START END, each run's value is instead the mean over its probes in [START, END]
(e.g. early training), so a transient effect is tested on one number per seed, not per probe.

For each run, the value of each metric is taken at its final probe at or before --at-step (the
week-2 definition). Across seeds: mean ± sample std (ddof = 1), median, and n. Welch's t-test
compares each group with --ref in the same environment, on the per-seed values.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_report  # noqa: E402

METRICS = [  # (key, column name, format)
    ("return", "Return", ".0f"),                         # deterministic eval where recorded
    ("bias/norm_bias_median", "Norm. bias (median)", "+.3f"),
    ("bias/norm_bias_mean", "Norm. bias (mean)", "+.3f"),
    ("bias/spearman", "Spearman(Q, G)", ".2f"),
    ("ppo/explained_variance", "Explained var.", "+.2f"),
    ("probe/length_mean", "Episode length", ".0f"),
]


def _value(metrics: dict, metric: str):
    if metric == "return":
        return metrics.get("eval/deterministic_return_mean", metrics.get("probe/return_mean"))
    return metrics.get(metric)


def final_values(report: dict, metric: str, window=None):
    """(step label, value) of the metric at the run's final probe, or, with window=(a, b), its mean
    over the probes with a <= step <= b. None if the run has no such value."""
    if not report["probes"]:
        return None
    if window is None:
        last = report["probes"][-1]
        v = _value(last["metrics"], metric)
        return None if v is None else (last["step"], float(v))
    vals = [_value(p["metrics"], metric) for p in report["probes"] if window[0] <= p["step"] <= window[1]]
    vals = [float(v) for v in vals if v is not None]
    return (len(vals), float(np.mean(vals))) if vals else None


def describe(values: np.ndarray, fmt: str) -> str:
    if len(values) == 0:
        return "–"
    if len(values) == 1:
        return f"{values[0]:{fmt}}"
    sd = np.std(values, ddof=1)
    return f"{values.mean():{fmt}} ± {sd:{fmt.replace('+', '')}} (med {np.median(values):{fmt}})"


def summarize(inputs, at_step=None, only=None, ref=None, window=None) -> str:
    if window is not None and at_step is not None:
        raise ValueError("use either at_step or window, not both")
    by_env = plot_report.load_groups(inputs, window[1] if window else at_step, only)
    lines = []
    for env_id, groups in sorted(by_env.items()):
        keys = [k for k in plot_report.ORDER if k in groups] + sorted(k for k in groups if k not in plot_report.ORDER)
        cols = [m for m in METRICS if any(final_values(r, m[0], window) for k in keys for r in groups[k])]
        if window:
            title = f" (per-seed mean over probes from {window[0]:,} to {window[1]:,} steps)"
        else:
            title = f" (final probe at or before {at_step:,} steps)" if at_step else ""
        lines.append(f"\n### {env_id}" + title)
        lines.append("")
        lines.append(f"| Group | n | {'Probes per seed' if window else 'Probe step(s)'} | "
                     + " | ".join(c[1] for c in cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(cols))
        per_seed = {}
        for k in keys:
            vals, steps = {}, set()
            for key, _, _ in cols:
                pairs = [final_values(r, key, window) for r in groups[k]]
                pairs = [p for p in pairs if p is not None]
                steps.update(s for s, _ in pairs)
                vals[key] = np.array([v for _, v in pairs])
            per_seed[k] = vals
            step_txt = ", ".join(f"{s:,}" for s in sorted(steps))
            lines.append(f"| {plot_report.label(k)} | {len(groups[k])} | {step_txt} | "
                         + " | ".join(describe(vals[c[0]], c[2]) for c in cols) + " |")
        if ref and ref in per_seed:
            lines.append("")
            lines.append(f"Welch's t-test against {plot_report.label(ref)} (per-seed values; p, two-sided):")
            lines.append("")
            lines.append("| Group | " + " | ".join(c[1] for c in cols) + " |")
            lines.append("|---|" + "---|" * len(cols))
            for k in keys:
                if k == ref:
                    continue
                cells = []
                for key, _, _ in cols:
                    a, b = per_seed[k][key], per_seed[ref][key]
                    if len(a) > 1 and len(b) > 1:
                        if np.ptp(a) == 0 and np.ptp(b) == 0:  # both groups constant: Welch is undefined
                            cells.append("– (no spread)")
                            continue
                        with warnings.catch_warnings():  # one constant group: scipy warns, result stands
                            warnings.simplefilter("ignore", RuntimeWarning)
                            res = stats.ttest_ind(a, b, equal_var=False)
                        if not np.isfinite(res.pvalue):
                            cells.append("– (no spread)")
                            continue
                        p = "p < 0.001" if res.pvalue < 0.001 else f"p = {res.pvalue:.3f}"
                        cells.append(f"t = {res.statistic:+.2f}, {p}")
                    else:
                        cells.append("–")
                lines.append(f"| {plot_report.label(k)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="sweep folders, run folders or wildcards")
    ap.add_argument("--at-step", type=int, default=None, help="use the last probe at or before this step")
    ap.add_argument("--window", nargs=2, type=int, default=None, metavar=("START", "END"),
                    help="per-seed mean over probes in [START, END] instead of the final probe")
    ap.add_argument("--only", nargs="+", default=None, help="only these groups")
    ap.add_argument("--ref", default=None, help="group to test the others against, e.g. ppo")
    ap.add_argument("--out", default=None, help="also write the tables to this Markdown file (UTF-8)")
    args = ap.parse_args()
    if not plot_report.find_reports(args.inputs):
        raise SystemExit(f"no critic_report.json found under: {' '.join(args.inputs)}")
    if args.window and args.at_step:
        raise SystemExit("use either --at-step or --window, not both")
    text = summarize(args.inputs, args.at_step, args.only, args.ref, args.window)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # λ and ± survive redirection on Windows too
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
