"""plot_report.py on synthetic reports: grouping, wildcards, one figure per environment."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

_spec = importlib.util.spec_from_file_location(
    "plot_report", Path(__file__).resolve().parents[1] / "scripts" / "plot_report.py")
plot_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_report)


def _write_report(root: Path, env_id: str, algo: str, seed: int, twin: bool, **meta) -> None:
    rng = np.random.default_rng(seed)
    probes = []
    for step in (10_000, 20_000, 30_000):
        g = rng.normal(100, 30, 200)
        q = g + 5
        metrics = {"probe/return_mean": float(step / 100), "probe/usable_states": 200,
                   "bias/norm_bias_mean": 0.05, "bias/norm_bias_median": 0.01,
                   "bias/spearman": 0.9, "td/abs_mean": 1.0,
                   "perf/sps_window": 100.0}
        if twin:
            metrics["twin/abs_diff_mean"] = 0.3
        if algo == "sac":
            metrics["eval/deterministic_return_mean"] = float(step / 90)
        probes.append({"step": step, "metrics": metrics, "raw": {"q": q.tolist(), "g": g.tolist()}})
    run = root / f"{env_id}__{algo}__{seed}"
    run.mkdir(parents=True)
    report = {"meta": {"algo": algo, "env_id": env_id, "seed": seed, **meta}, "probes": probes}
    (run / "critic_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_groups_seeds_and_ablation(tmp_path):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        _write_report(runs, "Hopper-v5", "td3", seed, twin=True, clipped_double_q=True)
        _write_report(runs, "Hopper-v5", "td3", seed + 10, twin=True, clipped_double_q=False)
        _write_report(runs, "Hopper-v5", "ddpg", seed, twin=False)
        _write_report(runs, "Hopper-v5", "sac", seed, twin=True)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in plot_report.find_reports([str(runs)])]
    keys = sorted({plot_report.group_key(r["meta"]) for r in reports})
    assert keys == ["ddpg", "sac", "td3", "td3_no_cdq"]

    written = plot_report.plot([str(runs / "Hopper-v5__*")], str(tmp_path / "fig" / "hopper.png"))
    assert [p.name for p in written] == ["hopper.png", "hopper_hero.png"]
    assert all(p.stat().st_size > 10_000 for p in written)


def test_one_figure_per_environment(tmp_path):
    runs = tmp_path / "runs"
    _write_report(runs, "Hopper-v5", "td3", 0, twin=True)
    _write_report(runs, "HalfCheetah-v5", "td3", 0, twin=True)
    written = plot_report.plot([str(runs)], str(tmp_path / "all.png"), hero=False)
    assert sorted(p.name for p in written) == ["all__HalfCheetah-v5.png", "all__Hopper-v5.png"]


def test_stack_averages_across_seeds():
    reports = [{"probes": [{"step": 1, "metrics": {"m": 1.0}}, {"step": 2, "metrics": {"m": 3.0}}]},
               {"probes": [{"step": 1, "metrics": {"m": 3.0}}, {"step": 2, "metrics": {"m": None}}]}]
    steps, mean, std = plot_report.stack(reports, "m")
    assert steps.tolist() == [1, 2]
    assert mean.tolist() == [2.0, 3.0]           # a missing value is skipped, not counted as 0
    assert np.allclose(std, [np.sqrt(2.0), 0.0])  # sample std; a single seed gets no band


def test_quantiles_ignore_one_extreme_seed():
    reports = [{"probes": [{"step": 1, "metrics": {"m": v}}]} for v in (1.0, 1.5, 2.0, 2.1, 32.3)]
    steps, med, q25, q75 = plot_report.stack_quantiles(reports, "m")
    assert med.tolist() == [2.0] and q25.tolist() == [1.5] and q75.tolist() == [2.1]
    _, mean, _ = plot_report.stack(reports, "m")
    assert mean[0] > 7  # the mean is dragged up by the one collapsed seed


def test_no_reports_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit):
        plot_report.plot([str(tmp_path / "nothing_here")], str(tmp_path / "x.png"))


def test_hero_metric_choice_and_fallback(tmp_path):
    runs = tmp_path / "runs"
    _write_report(runs, "Hopper-v5", "td3", 0, twin=True)
    for metric in ("median", "mean"):
        written = plot_report.plot([str(runs)], str(tmp_path / f"{metric}.png"), hero_metric=metric)
        assert written[-1].name == f"{metric}_hero.png"
    # a report without the median (older runs) falls back to the mean instead of drawing nothing
    old = tmp_path / "old"
    _write_report(old, "Hopper-v5", "ddpg", 0, twin=False)
    report = old / "Hopper-v5__ddpg__0" / "critic_report.json"
    report.write_text(report.read_text(encoding="utf-8").replace('"bias/norm_bias_median": 0.01, ', ""),
                      encoding="utf-8")
    assert len(plot_report.plot([str(old)], str(tmp_path / "old.png"))) == 2
