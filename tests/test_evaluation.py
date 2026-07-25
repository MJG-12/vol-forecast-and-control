import numpy as np

from vol_forecast.evaluation import qlike_series


def test_qlike_is_zero_for_a_perfect_forecast() -> None:
    realized = np.array([0.01, 0.04, 0.09])
    loss = qlike_series(realized, realized)
    np.testing.assert_allclose(loss, 0.0, atol=1e-14)
