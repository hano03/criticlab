"""analyze_signals.py must find a signal that really tracks bias, and see nothing in pure noise."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_signals.py"


def _write_runs(root: Path, tracks: bool, n_runs: int = 6, n_probes: int = 30) -> None:
    rng = np.random.default_rng(0)
    for seed in range(n_runs):
        bias = np.cumsum(rng.normal(0, 0.05, n_probes))          # random walk, so changes are informative
        q = 100 + 5 * np.arange(n_probes) + rng.normal(0, 1, n_probes)
        noise = rng.normal(0, 0.01, n_probes)
        twin = (np.exp(bias) + noise) if tracks else np.exp(rng.normal(0, 0.3, n_probes))
        probes = [{"step": 10_000 * (i + 1), "metrics": {
            "bias/norm_bias_median": float(bias[i]), "bias/q_mean": float(q[i]),
            "td/abs_mean": float(abs(rng.normal(1, 0.1))), "twin/abs_diff_mean": float(twin[i]),
            "probe/return_mean": 1000.0, "eval/deterministic_return_mean": 1000.0}} for i in range(n_probes)]
        run = root / f"Fake-v0__td3__{seed}"
        run.mkdir(parents=True)
        meta = {"algo": "td3", "env_id": "Fake-v0", "seed": seed, "clipped_double_q": True}
        (run / "critic_report.json").write_text(json.dumps({"meta": meta, "probes": probes}))


def _view2_twin_rho(root: Path) -> float:
    out = subprocess.run([sys.executable, str(SCRIPT), str(root), "--min-step", "0"],
                         check=True, capture_output=True, text=True).stdout
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("twin 2 within runs"))
    return float(re.search(r"mean rho ([+-][\d.]+)", line).group(1))


def test_tracking_signal_is_found(tmp_path):
    _write_runs(tmp_path, tracks=True)
    assert _view2_twin_rho(tmp_path) > 0.7


def test_noise_signal_is_not(tmp_path):
    _write_runs(tmp_path, tracks=False)
    assert abs(_view2_twin_rho(tmp_path)) < 0.3
