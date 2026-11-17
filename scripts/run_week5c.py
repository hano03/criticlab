"""Week 5c: does the replay buffer's size change DQN's overshoot on CartPole, and in which direction?

    python scripts/run_week5c.py --dry-run            # the 20 new jobs and the time estimate
    python scripts/run_week5c.py --max-parallel 10    # run them, then draw figures and the summary
    python scripts/run_week5c.py --summary-only       # summary of existing runs only

Three buffer sizes, everything else at CleanRL's defaults (CartPole-v1, plain DQN, target copied
every 500 env steps, 0.1 updates per env step, 500k steps, 10 probe episodes of up to 1,000 steps):
  buffer  2,500   dqn_buf2500    runs/week5c, seeds 0-9 (new)     5 copies per buffer turnover
  buffer 10,000   dqn            runs/week5,  seeds 0-9 (seen)   20 copies per buffer turnover
  buffer 40,000   dqn_buf40000   runs/week5c, seeds 0-9 (new)    80 copies per buffer turnover
Each cell is read only from these folders and seeds (sources()); runs/week5 also holds dqn_g3 and
dqn_g10, which are other configurations and never read here.

Why. Week 5 (docs/findings_week5.md) found that the copy interval in env steps, not updates per
copy, controls the overshoot. Two explanations fit and predict opposite effects of buffer size:
  TURNOVER     targets are bootstrapped again and again on the same transitions; copies per buffer
               turnover (buffer / copy interval) drive the overshoot -> larger buffer, MORE overshoot.
  OVERFITTING  the network fits a small, narrow dataset; a larger, more varied buffer generalises
               better -> larger buffer, LESS overshoot.
Buffer size leaves the replays per transition unchanged (updates per env step x batch = 12.8), so
it moves copies per turnover and the breadth of the data, not how often each transition is reused.

Pre-registration (Day 33, before the new runs; also docs/preregistration_week5c.md and spec v3.7).
Primary measure, per seed: peak excess of max-Q over the copy envelope E(n) = 100 (1 - 0.99^(n+1)),
all probes up to 500k (test 1's measure; the copy schedule is identical in every run).
Outcome, in this order:
  NO VERDICT   the 10,000 cell's mean is below +50 (the overshoot was not there to change)
  TURNOVER     40,000 - 2,500: 95% Welch interval above 0; means rise 2,500 < 10,000 < 40,000;
               Spearman rho(log buffer, excess) > 0 with p < 0.05 over the 30 runs
  OVERFITTING  the mirror image: interval below 0; means fall; rho < 0 with p < 0.05
  NO EFFECT    the interval for 40,000 - 2,500 lies within +-25% of the 10,000 cell's mean
  INCONCLUSIVE anything else (a mixed or non-monotone pattern, or an interval too wide to say)
Reported, not predicted: excess within the first 290 copies, peak max-Q and its step, share of
probes above 110, greedy return and median normalised bias over 400k-500k, and the pairwise
contrasts.

Resumable like run_week2.py. Figures: figures/week5c.png (+ _hero), figures/week5c_buffer.png;
tables and the outcome: figures/week5c_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_week2 as sweep  # noqa: E402
import run_week4 as week4  # noqa: E402
from launch import run_all  # noqa: E402

ENV = "CartPole-v1"
CEILING, R_MAX = 100.0, 1.0
PERIOD = 500
TOTAL = 500_000
CELLS = {2_500: "dqn_buf2500", 10_000: "dqn", 40_000: "dqn_buf40000"}   # buffer size -> configuration
NEW = [2_500, 40_000]
SEEDS = frozenset(range(10))
LATE = week4.LATE
MATCHED_COPIES = week4.MATCHED_COPIES
MARGIN = week4.MARGIN
MIN_DEFAULT = 50.0
EQUIVALENCE = 0.25
OUTCOME_TEXT = {
    "TURNOVER": "A larger buffer made the overshoot larger, in order across all three sizes. This supports the "
                "turnover explanation: the more target copies are fitted on the same transitions, the further "
                "the values run away.",
    "OVERFITTING": "A larger buffer made the overshoot smaller, in order across all three sizes. This supports the "
                   "overfitting explanation and counts against turnover: broader data, not fewer copies per "
                   "turnover, is what keeps the values down.",
    "NO EFFECT": "Buffer size between 2,500 and 40,000 does not change the overshoot by more than a quarter of its "
                 "size. Neither explanation, as stated, accounts for the copy interval's effect.",
    "INCONCLUSIVE": "The pattern is mixed, non-monotone or too uncertain to support either explanation. Report the "
                    "cell means and intervals; draw no mechanism from them.",
    "NO VERDICT": "The default buffer did not reproduce the overshoot, so the test cannot answer the question. Check "
                  "the runs before reading anything else.",
}


def sources(week5_dir, runs_dir) -> dict:
    """Pre-registered source of every cell: buffer size -> (folder, seeds read from it)."""
    return {2_500: (Path(runs_dir), SEEDS), 10_000: (Path(week5_dir), SEEDS), 40_000: (Path(runs_dir), SEEDS)}


def week5c_jobs(runs_dir) -> list[dict]:
    jobs = sweep.build_jobs([CELLS[b] for b in NEW], [ENV], sorted(SEEDS), TOTAL, 10, runs_dir)
    for job in jobs:
        job["argv"] = list(job["argv"]) + week4.CARTPOLE_FLAGS
    return jobs


def finished(report: dict) -> bool:
    probes = report.get("probes") or []
    return bool(probes) and probes[-1]["step"] == report["meta"].get("total_timesteps")


def load(src: dict) -> dict:
    """buffer size -> per-seed measures, read only from that cell's designated folder and seeds.
    Unfinished runs, runs of another length and other configurations are ignored; the same
    (cell, seed) twice within the designated sources is an error."""
    out: dict[int, list[dict]] = {b: [] for b in CELLS}
    seen: dict[tuple, Path] = {}
    cache: dict[Path, list] = {}
    for buffer, (folder, seeds) in src.items():
        if folder not in cache:
            cache[folder] = [(p, json.loads(p.read_text(encoding="utf-8")))
                             for p in sorted(folder.glob("*/critic_report.json"))]
        for path, report in cache[folder]:
            meta = report["meta"]
            if (sweep.algo_key(meta) != CELLS[buffer] or meta.get("env_id") != ENV or meta.get("seed") not in seeds
                    or int(meta.get("total_timesteps", -1)) != TOTAL or not finished(report)):
                continue
            ident = (buffer, meta["seed"])
            if ident in seen:
                raise SystemExit(f"seed counted twice: buffer {buffer}, seed {ident[1]}: {seen[ident]} and {path}")
            seen[ident] = path
            row = week4.per_seed(report, CEILING, R_MAX, late=LATE, matched_steps=TOTAL)
            row["primary"] = row["excess_steps"]
            row["seed"] = meta["seed"]
            ls = int(meta.get("learning_starts", 0))
            row["curve"] = [(q["step"], q["metrics"]["bias/q_max"]) for q in report["probes"]
                            if q["step"] > ls and q["metrics"].get("bias/q_max") is not None]
            out[buffer].append(row)
    return out


def verdict(per_cell: dict[int, list[float]]) -> dict:
    """The pre-registered outcome from per-seed primary values of the three buffer sizes."""
    small, default, large = (np.asarray(per_cell[b], float) for b in (2_500, 10_000, 40_000))
    diff, lo, hi, p = week4.welch_ci(large, small)
    means = [small.mean(), default.mean(), large.mean()]
    x = [np.log(b) for b in (2_500, 10_000, 40_000) for _ in per_cell[b]]
    y = [v for b in (2_500, 10_000, 40_000) for v in per_cell[b]]
    rho, rho_p = stats.spearmanr(x, y)
    band = EQUIVALENCE * means[1]
    if means[1] < MIN_DEFAULT:
        outcome = "NO VERDICT"
    elif lo > 0 and means[0] < means[1] < means[2] and rho > 0 and rho_p < 0.05:
        outcome = "TURNOVER"
    elif hi < 0 and means[0] > means[1] > means[2] and rho < 0 and rho_p < 0.05:
        outcome = "OVERFITTING"
    elif -band <= lo and hi <= band:
        outcome = "NO EFFECT"
    else:
        outcome = "INCONCLUSIVE"
    return {"diff": diff, "lo": lo, "hi": hi, "p": p, "means": means, "rho": float(rho), "rho_p": float(rho_p),
            "band": band, "outcome": outcome}


def _p(p: float) -> str:
    return "–" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def summary(src: dict) -> str:
    data = load(src)
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0  # noqa: E731
    fmt = lambda a, f: f"{np.mean(a):{f}} ± {sd(a):{f.lstrip('+')}}"  # noqa: E731
    col = lambda b, m: [r[m] for r in data[b]]  # noqa: E731
    late = f"{LATE[0] // 1000}k–{LATE[1] // 1000}k"
    lines = [f"### {ENV}: replay buffer size (copy every {PERIOD} env steps, 0.1 updates per env step)", "",
             "| Buffer | Copies per turnover | Seeds | **Peak excess, ≤ 500k (primary)** "
             f"| Excess, first {MATCHED_COPIES} copies | Peak max-Q (step) | Probes > {CEILING + MARGIN:g} "
             f"| Return, {late} | Median bias, {late} |", "|---|---|---|---|---|---|---|---|---|"]
    for b in CELLS:
        if not data[b]:
            lines.append(f"| {b:,} | {b // PERIOD} | 0 | – | – | – | – | – | – |")
            continue
        lines.append(f"| {b:,} | {b // PERIOD} | {len(data[b])} | **{fmt(col(b, 'primary'), '+.1f')}** "
                     f"| {fmt(col(b, 'excess_copies'), '+.1f')} | {fmt(col(b, 'peak'), '.1f')} "
                     f"({np.mean(col(b, 'peak_step')) / 1000:.0f}k) | {100 * np.mean(col(b, 'over')):.0f}% "
                     f"| {fmt(col(b, 'late_return'), '.1f')} | {fmt(col(b, 'late_bias'), '+.3f')} |")
    lines += ["", "### Pre-registered outcome", ""]
    if any(len(data[b]) < 2 for b in CELLS):
        lines.append("Not every cell has at least 2 finished seeds yet: no outcome.")
        return "\n".join(lines)
    v = verdict({b: col(b, "primary") for b in CELLS})
    lines += ["| Quantity | Value |", "|---|---|",
              f"| Precondition: buffer 10,000 mean ≥ +{MIN_DEFAULT:g} | {v['means'][1]:+.1f} |",
              f"| 40,000 − 2,500 | {v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}], p {_p(v['p'])} |",
              "| Means, 2,500 / 10,000 / 40,000 | " + " / ".join(f"{m:+.1f}" for m in v["means"]) + " |",
              f"| Spearman ρ(log buffer, excess), 30 runs | {v['rho']:+.2f}, p {_p(v['rho_p'])} |",
              f"| No-effect band (±{EQUIVALENCE:.0%} of the 10,000 mean) | ±{v['band']:.1f} |",
              "", f"**Outcome: {v['outcome']}.** {OUTCOME_TEXT[v['outcome']]}",
              "", f"| Other comparisons (a − b, not predicted) | Peak excess, ≤ 500k | Return, {late} |", "|---|---|---|"]
    for a, b in [(2_500, 10_000), (40_000, 10_000)]:
        cells = []
        for m in ("primary", "late_return"):
            d, lo, hi, p = week4.welch_ci(col(a, m), col(b, m))
            cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}], p {_p(p)}")
        lines.append(f"| buffer {a:,} vs {b:,} | " + " | ".join(cells) + " |")
    lines += ["", "*Envelope after n copies: 100 (1 − 0.99^(n+1)); the copy schedule is the same in every run. "
              "Replays per transition are the same at every buffer size (0.1 updates per env step × batch 128 = 12.8). "
              "The 10,000 cell is week 5's default runs (test 1, cell A).*"]
    return "\n".join(lines)


def figure(src: dict, out: Path) -> Path:
    """(a) max-Q over training, median and IQR across seeds, with the envelope; (b) each seed's primary
    measure against buffer size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    import plot_report as pr

    data = load(src)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor(pr.SURFACE)
    pr.style_axis(ax1, "a  Largest prediction on probe states", "max Q")
    pr.style_axis(ax2, "b  Peak excess per seed", "max Q − envelope (≤ 500k)")
    steps = np.arange(10_000, TOTAL + 1, 1000)
    env_line = [week4.envelope(week4.copies(s, PERIOD, 10_000), CEILING, R_MAX) for s in steps]
    (env_h,) = ax1.plot(steps, env_line, color=pr.INK_2, linewidth=1.2, linestyle="--", label="copy envelope")
    ceil_h = ax1.axhline(CEILING + MARGIN, color=pr.INK_2, linewidth=1.0, linestyle=":", label=f"ceiling + {MARGIN:g}")
    handles = []
    for b, key in CELLS.items():
        curves = [r["curve"] for r in data[b] if r["curve"]]
        if not curves:
            continue
        grid = sorted({s for cv in curves for s, _ in cv})
        table = np.full((len(curves), len(grid)), np.nan)
        pos = {s: j for j, s in enumerate(grid)}
        for i, cv in enumerate(curves):
            for s, q in cv:
                table[i, pos[s]] = q
        with np.errstate(all="ignore"), pr._quiet():
            q25, med, q75 = np.nanpercentile(table, [25, 50, 75], axis=0)
        line = pr.line_with_band(ax1, np.array(grid), med, None, key, q25, q75)
        if line is not None:
            line.set_label(f"buffer {b:,}" + (" (default)" if b == 10_000 else ""))
            handles.append(line)
    ax1.set_xlabel("environment steps (median and IQR across seeds)", color=pr.INK_2, fontsize=9)
    ax1.set_xlim(0, TOTAL)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.legend(handles=handles + [env_h, ceil_h], frameon=False, fontsize=8, loc="upper right")
    rng = np.random.default_rng(0)
    for b, key in CELLS.items():
        y = np.array([r["primary"] for r in data[b]])
        if not len(y):
            continue
        ax2.plot([b / 1.3, b * 1.3], [y.mean()] * 2, color=pr.INK_2, linewidth=1.5, zorder=2)
        ax2.scatter(b * np.exp(rng.uniform(-0.08, 0.08, len(y))), y, s=64, color=pr.COLORS[key],
                    edgecolor=pr.SURFACE, linewidth=2, zorder=3)
    ax2.axhline(0, color=pr.INK_2, linewidth=1.0)
    ax2.set_xscale("log")
    ax2.set_xticks(list(CELLS))
    ax2.set_xticklabels([f"{b:,}" for b in CELLS])
    ax2.minorticks_off()
    ax2.set_xlabel("replay buffer size (bar = mean)", color=pr.INK_2, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=pr.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", default=str(Path("runs") / "week5c"), help="where the new runs go")
    p.add_argument("--week5-dir", default=str(Path("runs") / "week5"), help="the buffer-10,000 cell")
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)
    src = sources(args.week5_dir, runs_dir)

    if not args.summary_only:
        jobs = week5c_jobs(runs_dir)
        to_run, skipped, moved = sweep.prepare(jobs, runs_dir, TOTAL, args.dry_run)
        cpu_h, wall_h = sweep.estimate_hours(to_run, TOTAL, args.max_parallel)
        print(f"{len(jobs)} new jobs: {len(skipped)} already finished, {len(to_run)} to run, "
              f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
        print(f"estimate: {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall at {args.max_parallel} at once")
        found = load(src)
        print("finished seeds found per buffer (10,000 from week 5; 2,500 and 40,000 from week 5c): "
              + ", ".join(f"{b:,} {sorted(r['seed'] for r in found[b])}" for b in CELLS))
        if args.dry_run:
            for job in to_run:
                print("  python -m " + " ".join(job["argv"]))
            return
        results = run_all([(j["name"], j["argv"]) for j in to_run], log_dir=runs_dir / "logs",
                          max_parallel=args.max_parallel)
        failed = [name for name, code in results if code != 0]
        if failed:
            raise SystemExit(f"FAILED ({len(failed)}): {failed}  -> rerun the same command to retry only these")

    if not args.no_plot:
        import plot_report
        for path in plot_report.plot([str(args.week5_dir), str(runs_dir)], str(Path("figures") / "week5c.png"),
                                     only=list(CELLS.values())):
            print(f"wrote {path}")
        print(f"wrote {figure(src, Path('figures') / 'week5c_buffer.png')}")
    text = summary(src)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    out = Path("figures") / "week5c_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
