"""Can cheap training signals stand in for the probe? (week-5 question 1)

Tests whether TD error and twin-critic disagreement, which cost nothing to log, track the critic
bias the probe measures (median normalised bias per probe), using existing critic_report.json files.

    python scripts/analyze_signals.py runs/week2

A raw correlation is not enough: both signals grow with the size of Q, and everything drifts with
training time. So the script reports four views, from naive to controlled:
  1. across runs (late-training means),
  2. within runs, on probe-to-probe changes (removes shared time trends),
  3. partial correlation controlling for log|Q| and training step, with per-run fixed effects,
  4. return collapses: did the signal rise in the probe *before* the collapse?
  5. detrended: subtract a rolling median (5 probes) from signal and bias, then correlate the
     residuals (levels), and the residual signal with the next bias change, controlling for the
     change in Q (lead). Per-run Spearman; Wilcoxon signed-rank across runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SIGNALS = {"td": "td/abs_mean", "twin": "twin/abs_diff_mean"}


def load(root: Path, min_step: int) -> dict:
    runs = defaultdict(list)
    for path in root.rglob("critic_report.json"):
        if "_incomplete" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["meta"]
        algo = meta["algo"]
        if algo == "td3" and meta.get("clipped_double_q") is False:
            algo = "td3_no_cdq"
        for p in data["probes"]:
            m = p["metrics"]
            if p["step"] < min_step or m.get("bias/norm_bias_median") is None or m.get("td/abs_mean") is None:
                continue
            runs[(meta["env_id"], algo, meta["seed"])].append({
                "step": p["step"], "bias": m["bias/norm_bias_median"], "q": m["bias/q_mean"],
                "ret": m.get("eval/deterministic_return_mean", m["probe/return_mean"]),
                **{k: m.get(v) for k, v in SIGNALS.items()},
            })
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in runs.items()}


def per_run(runs, env, fn):
    vals = [fn(v) for k, v in runs.items() if k[0] == env]
    return np.array([x for x in vals if x is not None and np.isfinite(x)])


def detrend(x, window: int = 5) -> np.ndarray:
    """Residual from a centred rolling median: removes slow training-time trends."""
    x = np.asarray(x, float)
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return x - np.array([np.median(xp[i:i + window]) for i in range(len(x))])


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after linearly removing rank(z) from both ranks."""
    if len(x) < 6:
        return float("nan")
    rx, ry, rz = (stats.rankdata(t) for t in (x, y, z))
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)  # noqa: E731
    return float(stats.pearsonr(res(rx), res(ry))[0])


def wilcoxon(rhos) -> str:
    if len(rhos) < 5:
        return "n/a"
    p = stats.wilcoxon(rhos).pvalue
    return f"median rho {np.median(rhos):+.2f}, > 0 in {np.sum(rhos > 0)}/{len(rhos)} runs, p = {p:.3f}"


def describe(rhos) -> str:
    if len(rhos) < 2:
        return "n/a"
    p = stats.ttest_1samp(rhos, 0).pvalue
    return f"mean rho {rhos.mean():+.2f}, > 0 in {np.sum(rhos > 0)}/{len(rhos)} runs, p = {p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="folder holding run folders with critic_report.json")
    ap.add_argument("--min-step", type=int, default=50_000, help="ignore early probes")
    ap.add_argument("--late-step", type=int, default=150_000, help="'late training' for view 1")
    ap.add_argument("--algo", default=None, help="only this algorithm (ddpg, td3, td3_no_cdq, sac)")
    args = ap.parse_args()
    runs = load(Path(args.root), args.min_step)
    if args.algo:
        runs = {k: v for k, v in runs.items() if k[1] == args.algo}
    envs = sorted({k[0] for k in runs})

    for env in envs:
        print(f"\n=== {env} ({sum(1 for k in runs if k[0] == env)} runs) ===")
        for sig in SIGNALS:
            have = {k: v for k, v in runs.items() if k[0] == env and v[0][sig] is not None}
            if not have:
                continue
            # 1. across runs
            late = [[r for r in v if r["step"] >= args.late_step] for v in have.values()]
            x = [np.mean([r[sig] for r in v]) for v in late if v]
            y = [np.mean([r["bias"] for r in v]) for v in late if v]
            rho, p = stats.spearmanr(x, y)
            print(f"  {sig:4s} 1 across runs (late means):        rho {rho:+.2f}, p = {p:.3f}, n = {len(x)} runs")
            # 2. within runs, changes
            def dchange(v, lead=False):
                ds, db = np.diff([r[sig] for r in v]), np.diff([r["bias"] for r in v])
                if lead:
                    ds, db = ds[:-1], db[1:]
                return stats.spearmanr(ds, db)[0] if len(ds) > 3 else None
            print(f"  {sig:4s} 2 within runs, same-probe changes:  {describe(per_run(have, env, dchange))}")
            print(f"  {sig:4s}   ...signal change -> next bias change: "
                  f"{describe(per_run(have, env, lambda v: dchange(v, lead=True)))}")
            # 3. partial correlation, run fixed effects, controls log|Q| and step
            Y, X, Z = [], [], []
            for v in have.values():
                a = np.array([[r["bias"], np.log(abs(r[sig]) + 1e-9), np.log(abs(r["q"]) + 1e-9), r["step"] / 1e5]
                              for r in v])
                a -= a.mean(0)
                Y.append(a[:, 0]); X.append(a[:, 1]); Z.append(a[:, 2:])
            Y, X, Z = np.concatenate(Y), np.concatenate(X), np.concatenate(Z)
            res = lambda t: t - Z @ np.linalg.lstsq(Z, t, rcond=None)[0]  # noqa: E731
            r, p = stats.pearsonr(res(X), res(Y))
            print(f"  {sig:4s} 3 partial r | log|Q|, step, run:     r {r:+.2f}, p = {p:.3g}, n = {len(Y)} probes")
            # 5. detrended
            def levels(v):
                return stats.spearmanr(detrend([r[sig] for r in v]), detrend([r["bias"] for r in v]))[0] \
                    if len(v) > 5 else None

            def lead(v):
                ds = detrend([r[sig] for r in v])[:-1]
                return partial_spearman(ds, np.diff([r["bias"] for r in v]), np.diff([r["q"] for r in v]))
            print(f"  {sig:4s} 5 detrended levels:                {wilcoxon(per_run(have, env, levels))}")
            print(f"  {sig:4s}   ...detrended signal -> next bias change | dQ: {wilcoxon(per_run(have, env, lead))}")
        # 4. collapses
        events = []
        for k, v in runs.items():
            if k[0] != env:
                continue
            for i in range(2, len(v)):
                if v[i - 1]["ret"] > 300 and v[i]["ret"] < 0.4 * v[i - 1]["ret"]:
                    events.append((v[i - 2], v[i - 1]))
        if events:
            td_up = sum(b["td"] > a["td"] for a, b in events)
            tw = [(a, b) for a, b in events if a["twin"] is not None]
            tw_up = sum(b["twin"] > a["twin"] for a, b in tw)
            print(f"  4 collapses (return falls > 60% from > 300): {len(events)} events; "
                  f"TD error rose just before in {td_up}/{len(events)}, twin disagreement in {tw_up}/{len(tw)}")


if __name__ == "__main__":
    main()
