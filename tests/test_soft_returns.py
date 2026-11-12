import numpy as np
import pytest

from criticlab.diagnostics import discounted_returns, soft_returns


def test_alpha_zero_equals_plain_returns():
    r = np.array([1.0, -2.0, 0.5, 3.0])
    logp = np.array([-1.0, -0.3, -2.0, -0.7])
    np.testing.assert_allclose(soft_returns(r, logp, alpha=0.0, gamma=0.9), discounted_returns(r, 0.9))


def test_hand_computed_three_steps():
    # H_2 = 3
    # H_1 = 2 + 0.9 * (3    - 0.5 * -3) = 6.05
    # H_0 = 1 + 0.9 * (6.05 - 0.5 * -2) = 7.345
    out = soft_returns([1, 2, 3], [-1, -2, -3], alpha=0.5, gamma=0.9)
    np.testing.assert_allclose(out, [7.345, 6.05, 3.0])


def test_first_logp_is_never_used():
    # Q(s_0, a_0) is conditioned on a_0, so logp_0 must not enter H_0.
    a = soft_returns([1, 2, 3], [-1, -2, -3], alpha=0.5, gamma=0.9)
    b = soft_returns([1, 2, 3], [-99, -2, -3], alpha=0.5, gamma=0.9)
    np.testing.assert_allclose(a, b)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        soft_returns([1, 2, 3], [-1, -2], alpha=0.5, gamma=0.9)
