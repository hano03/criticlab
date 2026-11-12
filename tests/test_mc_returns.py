import numpy as np

from criticlab.diagnostics import discounted_returns, usable_mask


def test_discounted_returns_hand_computed():
    # G_2 = 1, G_1 = 1 + 0.5*1 = 1.5, G_0 = 1 + 0.5*1.5 = 1.75
    np.testing.assert_allclose(discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0])


def test_discounted_returns_gamma_one_is_reverse_cumsum():
    r = np.array([3.0, -1.0, 2.0, 5.0])
    np.testing.assert_allclose(discounted_returns(r, 1.0), np.cumsum(r[::-1])[::-1])


def test_discounted_returns_empty():
    assert discounted_returns([], 0.99).shape == (0,)


def test_terminated_episode_uses_every_state():
    assert usable_mask(10, truncated=False, horizon=3).all()


def test_truncated_episode_drops_the_tail():
    # length 10, horizon 3: t <= 7 has at least 3 observed rewards -> states 0..7 usable
    mask = usable_mask(10, truncated=True, horizon=3)
    assert mask.tolist() == [True] * 8 + [False] * 2


def test_truncated_episode_shorter_than_horizon_has_no_usable_states():
    assert not usable_mask(2, truncated=True, horizon=3).any()
