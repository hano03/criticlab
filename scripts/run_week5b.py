"""Week 5b: does a slow target network still prevent DQN's soft divergence on CartPole when the
replay ratio is high? A 2x2: target copied every {500, 5,000} env steps x {0.1, 1.0} gradient
updates per env step.

    python scripts/run_week5b.py --dry-run            # the 15 new jobs and the time estimate
    python scripts/run_week5b.py --max-parallel 10    # run them, then draw figures and the summary
    python scripts/run_week5b.py --summary-only       # summary of existing runs only

The four cells (CartPole-v1, plain DQN, 10 seeds each, 10 probe episodes of up to 1,000 steps):
  A  copy every   500, 0.1 updates/step ( 50 per copy)  dqn              500k  runs/week5, seeds 0-9
  B  copy every   500, 1.0 updates/step (500 per copy)  dqn_g10          500k  runs/week5, seeds 0-9
  C  copy every 5,000, 0.1 updates/step (500 per copy)  dqn_tgt5000      1.5M  runs/week4, seeds 0-4
                                                                               + runs/week5b, seeds 5-9 (new)
  D  copy every 5,000, 1.0 updates/step (5,000 per copy) dqn_tgt5000_g10 1.5M  runs/week5b, seeds 0-9 (new)
The 1.5M runs use --exploration-fraction 1/6, so epsilon reaches 0.05 at 250k as in the 500k runs.
Each cell is read only from these folders and seeds (sources()); anything else in them is ignored.
runs/week4 also holds plain DQN at copy interval 500, cell A's configuration, which must not count.

Why. Updates per copy = copy interval x updates per env step, so only two of the three can vary
independently. Week 4 raised the copy interval at a fixed replay ratio: the overshoot vanished (A ->
C). Week 5 raised the replay ratio at a fixed copy interval: it more than doubled (A -> B). D is
the missing corner: a slow target at the high replay ratio.

Pre-registration (Day 31, before the new runs; also docs/preregistration_week5b.md and spec v3.5).
No direction is predicted: both outcomes are plausible, and the rule below classifies the result.
Primary measure, per seed: peak excess of max-Q over the copy envelope within the first 290 target
copies (week 4's primary measure; copies-matched because the copy interval differs between cells).
  Protection by the slow target at the high replay ratio, from the cell means:
      P = (B - D) / (B - C)      (D = C gives 1, D = B gives 0)
  PROTECTS        P >= 0.75 and the 95% Welch interval for D - B lies below 0
  DOES NOT        P <= 0.25 (D close to B, or worse)
  PARTLY          anything in between
  NO VERDICT      if C's mean exceeds +30 (week 4's slow-target result did not hold up) or
                  B - C < 100 (no high-ratio overshoot to protect against)
Reported, not predicted: the interaction (D - C) - (B - A) with a Welch interval; peak excess over
the whole run and within the first 500k steps; share of probes above 110; greedy return and median
normalised bias over the last 100k steps of each run (400k-500k or 1.4M-1.5M).

Resumable like run_week2.py. Figures: figures/week5b.png (+ _hero) for the new runs,
figures/week5b_2x2.png; tables and the outcome: figures/week5b_summary.md.
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
# cell -> (key, copy interval, updates per env step)
CELLS = {"A": ("dqn", 500, 0.1), "B": ("dqn_g10", 500, 1.0),
         "C": ("dqn_tgt5000", 5000, 0.1), "D": ("dqn_tgt5000_g10", 5000, 1.0)}
KEY_TO_CELL = {key: cell for cell, (key, _, _) in CELLS.items()}
SHORT, LONG = 500_000, week4.LONG_TOTAL                        # 500k and 1.5M steps
NEW_JOBS = [("dqn_tgt5000_g10", list(range(10))), ("dqn_tgt5000", [5, 6, 7, 8, 9])]
MATCHED_COPIES = week4.MATCHED_COPIES                          # 290
MARGIN = week4.MARGIN
PROTECTS, DOES_NOT = 0.75, 0.25
MAX_C, MIN_GAP = 30.0, 100.0
OUTCOME_TEXT = {
    "PROTECTS": "Spacing the copies 5,000 env steps apart prevents the overshoot even at 1.0 update per "
                "env step (5,000 updates per copy). The copy interval in env steps, not the replay ratio, "
                "decides whether DQN's values run away on CartPole.",
    "DOES NOT": "At 1.0 update per env step the slow target no longer prevents the overshoot. The replay "
                "ratio dominates: a slow target protects only when each transition is replayed rarely.",
    "PARTLY": "The slow target removes part of the high-ratio overshoot. Both the copy interval and the "
              "replay ratio matter; report the size of each.",
    "NO VERDICT": "A reused cell did not behave as before (see the precondition), so the comparison cannot "
                  "answer the question. Check the runs before reading anything else.",
}


def week5b_jobs(runs_dir) -> list[dict]:
    jobs = []
    for key, seeds in NEW_JOBS:
        for job in sweep.build_jobs([key], [ENV], seeds, LONG, 10, runs_dir):
            job["argv"] = list(job["argv"]) + week4.LONG_FLAGS + week4.CARTPOLE_FLAGS
            job["total"] = LONG
            jobs.append(job)
    return jobs


def finished(report: dict) -> bool:
    probes = report.get("probes") or []
    return bool(probes) and probes[-1]["step"] == report["meta"].get("total_timesteps")


def measures(report: dict) -> dict:
    total = int(report["meta"]["total_timesteps"])
    late = (total - 100_000, total)
    row = week4.per_seed(report, CEILING, R_MAX, late=late, matched_steps=SHORT)
    row["excess_whole"] = week4.per_seed(report, CEILING, R_MAX, late=late, matched_steps=total)["excess_steps"]
    row["primary"] = row["excess_copies"]
    return row


def sources(week5_dir, week4_dir, runs_dir) -> dict:
    """The pre-registered source of every cell: cell -> [(folder, seeds read from it)]. Nothing else
    is read: week 4's folder also holds plain DQN (cell A's configuration), which must not count."""
    week5_dir, week4_dir, runs_dir = Path(week5_dir), Path(week4_dir), Path(runs_dir)
    ten = frozenset(range(10))
    return {"A": [(week5_dir, ten)], "B": [(week5_dir, ten)],
            "C": [(week4_dir, frozenset(range(5))), (runs_dir, frozenset(range(5, 10)))],
            "D": [(runs_dir, ten)]}


def load(src: dict) -> dict:
    """cell -> per-seed measures, read only from that cell's designated folders and seeds (see
    sources()). Unfinished runs, runs of the wrong length and other configurations are ignored.
    The same (cell, seed) found twice within the designated sources is an error, so no seed is
    counted twice."""
    out: dict[str, list[dict]] = {cell: [] for cell in CELLS}
    seen: dict[tuple, Path] = {}
    cache: dict[Path, list[tuple[Path, dict]]] = {}
    for cell, folders in src.items():
        key, period, _ = CELLS[cell]
        expected = SHORT if period == 500 else LONG
        for folder, seeds in folders:
            if folder not in cache:
                cache[folder] = [(path, json.loads(path.read_text(encoding="utf-8")))
                                 for path in sorted(folder.glob("*/critic_report.json"))]
            for path, report in cache[folder]:
                meta = report["meta"]
                if (sweep.algo_key(meta) != key or meta.get("env_id") != ENV or meta.get("seed") not in seeds
                        or int(meta.get("total_timesteps", -1)) != expected or not finished(report)):
                    continue
                ident = (cell, meta["seed"])
                if ident in seen:
                    raise SystemExit(f"seed counted twice: cell {cell}, seed {ident[1]}: {seen[ident]} and {path}")
                seen[ident] = path
                row = measures(report)
                row["seed"] = meta["seed"]
                ls = int(meta.get("learning_starts", 0))
                row["curve"] = [(week4.copies(q["step"], period, ls), q["metrics"]["bias/q_max"])  # for the figure
                                for q in report["probes"]
                                if q["step"] > ls and q["metrics"].get("bias/q_max") is not None]
                out[cell].append(row)
    return out


def welch_interaction(groups: list[list[float]], signs: list[int], level: float = 0.95):
    """sum(sign_i * mean_i) with a Welch-Satterthwaite interval (difference of differences)."""
    est = sum(s * np.mean(g) for g, s in zip(groups, signs))
    v = [np.var(g, ddof=1) / len(g) for g in groups]
    se = float(np.sqrt(sum(v)))
    if se == 0:
        return est, est, est
    df = sum(v) ** 2 / sum(vi ** 2 / (len(g) - 1) for vi, g in zip(v, groups))
    half = stats.t.ppf(0.5 + level / 2, df) * se
    return est, est - half, est + half


def verdict(a, b, c, d) -> dict:
    """The pre-registered outcome from per-seed primary values of the four cells."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    diff, lo, hi, p = week4.welch_ci(d, b)
    gap = b.mean() - c.mean()
    protection = (b.mean() - d.mean()) / gap if gap else float("nan")
    if c.mean() > MAX_C or gap < MIN_GAP:
        outcome = "NO VERDICT"
    elif protection >= PROTECTS and hi < 0:
        outcome = "PROTECTS"
    elif protection <= DOES_NOT:
        outcome = "DOES NOT"
    else:
        outcome = "PARTLY"
    inter = welch_interaction([d, c, b, a], [1, -1, -1, 1])
    return {"diff": diff, "lo": lo, "hi": hi, "p": p, "protection": float(protection),
            "outcome": outcome, "interaction": inter}


def _p(p: float) -> str:
    return "–" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def summary(src: dict) -> str:
    data = load(src)
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0  # noqa: E731
    fmt = lambda a, f: f"{np.mean(a):{f}} ± {sd(a):{f.lstrip('+')}}"  # noqa: E731
    col = lambda cell, m: [r[m] for r in data[cell]]  # noqa: E731
    lines = [f"### {ENV}: copy interval × replay ratio", "",
             f"| Cell | Copy every | Updates per env step (per copy) | Seeds | **Peak excess, first {MATCHED_COPIES} copies "
             f"(primary)** | Peak excess, whole run | Peak excess, first 500k | Peak max-Q | Probes > {CEILING + MARGIN:g} "
             "| Return, last 100k | Median bias, last 100k |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, (key, period, ratio) in CELLS.items():
        rows = data[cell]
        if not rows:
            lines.append(f"| {cell} | {period:,} | {ratio:.1f} | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {cell} | {period:,} | {ratio:.1f} ({int(period * ratio):,}) | {len(rows)} "
            f"| **{fmt(col(cell, 'primary'), '+.1f')}** | {fmt(col(cell, 'excess_whole'), '+.1f')} "
            f"| {fmt(col(cell, 'excess_steps'), '+.1f')} | {fmt(col(cell, 'peak_whole'), '.1f')} "
            f"| {100 * np.mean(col(cell, 'over')):.0f}% | {fmt(col(cell, 'late_return'), '.1f')} "
            f"| {fmt(col(cell, 'late_bias'), '+.3f')} |")
    lines += ["", "### Pre-registered outcome", ""]
    if any(len(data[c]) < 2 for c in CELLS):
        lines.append("Not every cell has at least 2 finished seeds yet: no outcome.")
        return "\n".join(lines)
    prim = {c: col(c, "primary") for c in CELLS}
    v = verdict(prim["A"], prim["B"], prim["C"], prim["D"])
    est, lo, hi = v["interaction"]
    lines += ["| Quantity | Value |", "|---|---|",
              f"| Precondition: C mean ≤ +{MAX_C:g} and B − C ≥ {MIN_GAP:g} | C {np.mean(prim['C']):+.1f}; "
              f"B − C {np.mean(prim['B']) - np.mean(prim['C']):+.1f} |",
              f"| D − B (slow target at the high ratio) | {v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}], p {_p(v['p'])} |",
              f"| Protection P = (B − D) / (B − C) | {v['protection']:.2f} (PROTECTS ≥ {PROTECTS} with D − B < 0; "
              f"DOES NOT ≤ {DOES_NOT}) |",
              f"| Interaction (D − C) − (B − A), not predicted | {est:+.1f} [{lo:+.1f}, {hi:+.1f}] |",
              "", f"**Outcome: {v['outcome']}.** {OUTCOME_TEXT[v['outcome']]}",
              "", "| Other comparisons (a − b, not predicted) | Peak excess, first 290 copies | Return, last 100k |",
              "|---|---|---|"]
    for name, x, y in [("C − A: slow target at 0.1 (week 4's effect)", "C", "A"),
                       ("B − A: high ratio at the fast target (week 5's effect)", "B", "A"),
                       ("D − C: high ratio at the slow target", "D", "C")]:
        cells = []
        for m in ("primary", "late_return"):
            dd, l2, h2, pp = week4.welch_ci(col(x, m), col(y, m))
            cells.append(f"{dd:+.1f} [{l2:+.1f}, {h2:+.1f}], p {_p(pp)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"*Envelope after n copies: 100 (1 − 0.99^(n+1)). The primary measure uses probes within the first "
              f"{MATCHED_COPIES} copies: about 155k steps at copy interval 500, about 1.46M at 5,000, so matched copies "
              "come at different training stages (week 4's limitation). Returns are over each run's last 100k steps "
              "(400k–500k or 1.4M–1.5M). Cell C pools week-4 seeds 0–4 with new seeds 5–9; A and B are week 5's runs.*"]
    return "\n".join(lines)


def figure(src: dict, out: Path) -> Path:
    """(a) max-Q against target copies, median across seeds, with the envelope; (b) the 2x2 as an
    interaction plot of the primary measure, one dot per seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_report as pr

    data = load(src)
    colour = {500: "#e34948", 5000: "#9b2b2b"}   # copy interval; validated pair, plus labels and line style
    style = {0.1: "-", 1.0: "--"}                # replay ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(pr.SURFACE)
    pr.style_axis(ax1, "a  Largest prediction, by target copies", "max Q (median across seeds)")
    pr.style_axis(ax2, f"b  Peak excess, first {MATCHED_COPIES} copies", "max Q − envelope")
    n = np.arange(0, MATCHED_COPIES + 1)
    ax1.plot(n, [week4.envelope(int(i), CEILING, R_MAX) for i in n], color=pr.INK_2, linewidth=1.2,
             linestyle=":", label="copy envelope")
    for cell, (key, period, ratio) in CELLS.items():
        curves = [r["curve"] for r in data[cell] if r["curve"]]
        if not curves:
            continue
        grid = sorted({c for cv in curves for c, _ in cv if c <= MATCHED_COPIES})
        table = np.full((len(curves), len(grid)), np.nan)
        pos = {c: j for j, c in enumerate(grid)}
        for i, cv in enumerate(curves):
            for c, q in cv:
                if c in pos:
                    table[i, pos[c]] = q
        with np.errstate(all="ignore"), pr._quiet():
            med = np.nanmedian(table, axis=0)
        ok = np.isfinite(med)
        ax1.plot(np.array(grid)[ok], med[ok], color=colour[period], linestyle=style[ratio], linewidth=1.8,
                 label=f"{cell}: copy every {period:,}, {ratio:.1f} updates per env step")
    ax1.set_xlabel("target copies made", color=pr.INK_2, fontsize=9)
    ax1.set_xlim(0, MATCHED_COPIES)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    rng = np.random.default_rng(0)
    xs = {0.1: 0.0, 1.0: 1.0}
    for period in (500, 5000):
        means = []
        for ratio in (0.1, 1.0):
            cell = next(c for c, (_, p, r) in CELLS.items() if p == period and r == ratio)
            y = np.array([r["primary"] for r in data[cell]])
            if not len(y):
                means.append(np.nan)
                continue
            x = xs[ratio] + (0.06 if period == 5000 else -0.06) + rng.uniform(-0.03, 0.03, len(y))
            ax2.scatter(x, y, s=40, color=colour[period], edgecolor=pr.SURFACE, linewidth=1.5, zorder=3)
            means.append(y.mean())
        ax2.plot([0, 1], means, color=colour[period], linewidth=1.8, zorder=2,
                 label=f"copy every {period:,} env steps")
    ax2.axhline(0, color=pr.INK_2, linewidth=1.0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["0.1", "1.0"])
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_xlabel("gradient updates per env step (line = cell mean)", color=pr.INK_2, fontsize=9)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=pr.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", default=str(Path("runs") / "week5b"), help="where the new runs go")
    p.add_argument("--week4-dir", default=str(Path("runs") / "week4"), help="cell C, seeds 0-4")
    p.add_argument("--week5-dir", default=str(Path("runs") / "week5"), help="cells A and B")
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)
    src = sources(args.week5_dir, args.week4_dir, runs_dir)

    if not args.summary_only:
        jobs = week5b_jobs(runs_dir)
        to_run, skipped, moved = sweep.prepare(jobs, runs_dir, LONG, args.dry_run)
        cpu_h, wall_h = sweep.estimate_hours(to_run, LONG, args.max_parallel)
        print(f"{len(jobs)} new jobs: {len(skipped)} already finished, {len(to_run)} to run, "
              f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
        print(f"estimate: {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall at {args.max_parallel} at once")
        found = load(src)
        print("finished seeds found per cell (A, B from week 5; C 0-4 from week 4; C 5-9, D from week 5b): "
              + ", ".join(f"{c} {sorted(r['seed'] for r in found[c])}" for c in CELLS))
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
        for path in plot_report.plot([str(runs_dir)], str(Path("figures") / "week5b.png")):
            print(f"wrote {path}")
        print(f"wrote {figure(src, Path('figures') / 'week5b_2x2.png')}")
    text = summary(src)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    out = Path("figures") / "week5b_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
