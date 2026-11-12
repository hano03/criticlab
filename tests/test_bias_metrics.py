import math

import numpy as np

from criticlab.diagnostics import NORM_FLOOR, bias_metrics, td_error_stats


def _returns(n=1000, seed=0):
    return np.random.default_rng(seed).normal(100.0, 50.0, size=n)


def test_constant_overestimation():
    g = _returns()
    m = bias_metrics(g + 5.0, g)
    assert math.isclose(m["bias_mean"], 5.0)
    assert m["bias_std"] < 1e-9
    assert math.isclose(m["bias_abs_mean"], 5.0)
    expected_norm = np.mean(5.0 / np.maximum(np.abs(g), NORM_FLOOR))
    assert math.isclose(m["norm_bias_mean"], expected_norm)
    assert math.isclose(m["pearson"], 1.0)
    assert math.isclose(m["spearman"], 1.0)


def test_median_resists_a_few_extreme_states():
    # 95 well-predicted states and 5 near-terminal states (|G| < 10) wildly overestimated
    g = np.concatenate([np.full(95, 200.0), np.full(5, 1.0)])
    q = np.concatenate([np.full(95, 200.0), np.full(5, 301.0)])
    m = bias_metrics(q, g)
    assert math.isclose(m["norm_bias_mean"], 5 * 300.0 / NORM_FLOOR / 100)  # 1.5, driven by 5 states
    assert m["norm_bias_median"] == 0.0


def test_perfect_critic():
    g = _returns()
    m = bias_metrics(g, g)
    assert m["bias_mean"] == 0.0 and m["bias_abs_mean"] == 0.0
    assert math.isclose(m["pearson"], 1.0)


def test_anti_correlated_critic():
    g = _returns()
    m = bias_metrics(-g, g)
    assert math.isclose(m["pearson"], -1.0)
    assert math.isclose(m["spearman"], -1.0)


def test_normalisation_floor_applies_to_small_returns():
    g = np.zeros(4)
    m = bias_metrics(g + 5.0, g)
    assert math.isclose(m["norm_bias_mean"], 5.0 / NORM_FLOOR)
    assert math.isnan(m["pearson"])  # zero variance: correlation undefined, not a crash


def test_spearman_handles_ties():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    assert math.isclose(bias_metrics(x, x)["spearman"], 1.0)


def test_spearman_is_rank_based():
    g = np.arange(1.0, 11.0)
    m = bias_metrics(np.exp(g), g)  # monotone but nonlinear
    assert math.isclose(m["spearman"], 1.0)
    assert m["pearson"] < 1.0


def test_empty_input_returns_nans():
    m = bias_metrics([], [])
    assert m["n"] == 0 and math.isnan(m["bias_mean"])


def test_td_error_stats():
    s = td_error_stats([1.0, 2.0], [2.0, 4.0])
    assert math.isclose(s["td/abs_mean"], 1.5)
    assert math.isclose(s["td/target_q_mean"], 3.0)
    assert math.isclose(s["td/target_q_abs_max"], 4.0)
