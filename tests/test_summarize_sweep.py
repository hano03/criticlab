"""summarize_sweep.py: per-seed final values, sample std, Welch tests, --at-step."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("summarize_sweep", _scripts / "summarize_sweep.py")
summarize_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_sweep)

FINAL = {"ppo": [1.0, 2.0, 3.0], "ppo_lam1.0": [4.0, 6.0, 8.0]}  # per-seed final returns


def _write(root: Path):
    for key, finals in FINAL.items():
        lam = 0.95 if key == "ppo" else 1.0
        for seed, final in enumerate(finals):
            probes = [{"step": 10_000, "metrics": {"eval/deterministic_return_mean": 100.0 + seed,
                                                   "bias/norm_bias_median": 0.5}},
                      {"step": 20_000, "metrics": {"eval/deterministic_return_mean": final,
                                                   "bias/norm_bias_median": 0.1 * final}}]
            run = root / f"Hopper-v5__{key}__{seed}"
            run.mkdir(parents=True)
            meta = {"algo": "ppo", "env_id": "Hopper-v5", "seed": seed, "gae_lambda": lam}
            (run / "critic_report.json").write_text(json.dumps({"meta": meta, "probes": probes}))


def test_final_probe_sample_std_and_welch(tmp_path):
    _write(tmp_path)
    out = summarize_sweep.summarize([str(tmp_path)], ref="ppo")
    assert "| PPO, λ = 0.95 | 3 | 20,000 | 2 ± 1 (med 2) |" in out        # sample std of 1, 2, 3 is 1
    assert "| PPO, λ = 1.0 | 3 | 20,000 | 6 ± 2 (med 6) |" in out
    res = stats.ttest_ind(FINAL["ppo_lam1.0"], FINAL["ppo"], equal_var=False)
    assert f"t = {res.statistic:+.2f}, p = {res.pvalue:.3f}" in out
    assert "Explained var." not in out  # columns appear only for recorded metrics


def test_at_step_uses_the_earlier_probe(tmp_path):
    _write(tmp_path)
    out = summarize_sweep.summarize([str(tmp_path)], at_step=15_000)
    assert "| 10,000 | 101 ± 1 (med 101) | +0.500 ± 0.000" in out
    assert np.isclose(np.std([100, 101, 102], ddof=1), 1.0)


def test_window_averages_each_seed_over_its_probes(tmp_path):
    _write(tmp_path)
    out = summarize_sweep.summarize([str(tmp_path)], window=(10_000, 20_000), ref="ppo")
    # seed s of ppo: mean of (100 + s) and final -> (101, 102.5, 104) -> mean 102.5
    finals = FINAL["ppo"]
    per_seed = [(100 + s + f) / 2 for s, f in enumerate(finals)]
    assert f"| PPO, λ = 0.95 | 3 | 2 | {np.mean(per_seed):.0f} ± {np.std(per_seed, ddof=1):.0f}" in out
    assert "per-seed mean over probes from 10,000 to 20,000" in out
    only_late = summarize_sweep.summarize([str(tmp_path)], window=(15_000, 20_000))
    assert "| PPO, λ = 0.95 | 3 | 1 | 2 ± 1 (med 2) |" in only_late


def test_constant_metric_has_no_welch_test(tmp_path):
    _write(tmp_path)
    for report in tmp_path.rglob("critic_report.json"):  # every run: episode length 1000 at both probes
        data = json.loads(report.read_text())
        for probe in data["probes"]:
            probe["metrics"]["probe/length_mean"] = 1000.0
        report.write_text(json.dumps(data))
    out = summarize_sweep.summarize([str(tmp_path)], ref="ppo")
    assert "– (no spread)" in out and "nan" not in out
