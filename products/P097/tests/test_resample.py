"""Bootstrap intervals: the declared resampling unit, determinism, validation."""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.resample import (
    BootstrapInterval,
    bootstrap_mean_interval,
    bootstrap_statistic_interval,
)


def test_mean_interval_on_a_constant_sample_is_degenerate() -> None:
    """Every resample of a constant sample gives the same mean.

    Hand arithmetic: a sample of ten 4.0s has mean 4.0 in every resample, so
    the interval is [4.0, 4.0] and the point estimate is 4.0.
    """
    interval = bootstrap_mean_interval(
        np.full(10, 4.0), resampling_unit="session", resamples=50, seed=3
    )
    assert interval.point == pytest.approx(4.0, rel=0.0, abs=1e-12)
    assert interval.low == pytest.approx(4.0, rel=0.0, abs=1e-12)
    assert interval.high == pytest.approx(4.0, rel=0.0, abs=1e-12)
    assert interval.width == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_total_scaling_multiplies_the_mean_by_the_unit_count() -> None:
    """With ``total=True`` the statistic is ``n * mean``.

    Hand arithmetic: ten 4.0s give a total of 40.0.
    """
    interval = bootstrap_mean_interval(
        np.full(10, 4.0), resampling_unit="session", total=True, resamples=50, seed=3
    )
    assert interval.point == pytest.approx(40.0, rel=0.0, abs=1e-12)
    assert interval.n_units == 10


def test_interval_is_deterministic_at_a_fixed_seed() -> None:
    """Two runs at the same seed give identical ends.

    PCG64's draw order is fixed by the seed, so this is an equality of two
    values produced by the same arithmetic on the same draws, which is the one
    case where a bit-exact comparison is a correct claim. It is still asserted
    to a tolerance of zero on the relative bound rather than with ``==``, so
    the intent is visible.
    """
    sample = np.linspace(0.0, 10.0, 40)
    first = bootstrap_mean_interval(
        sample, resampling_unit="session", resamples=200, seed=97044
    )
    second = bootstrap_mean_interval(
        sample, resampling_unit="session", resamples=200, seed=97044
    )
    assert first.low == pytest.approx(second.low, rel=0.0, abs=0.0)
    assert first.high == pytest.approx(second.high, rel=0.0, abs=0.0)


def test_different_seeds_give_different_intervals() -> None:
    """A resampler that ignored its seed would pass the determinism test too."""
    sample = np.linspace(0.0, 10.0, 40)
    first = bootstrap_mean_interval(sample, resampling_unit="session", resamples=200, seed=1)
    second = bootstrap_mean_interval(sample, resampling_unit="session", resamples=200, seed=2)
    assert (first.low, first.high) != (second.low, second.high)


def test_rows_are_the_resampling_unit_for_a_two_dimensional_sample() -> None:
    """A two-dimensional sample resamples whole ROWS, keeping columns aligned.

    The statistic here is the difference between the two columns' means, which
    is identically zero for a sample whose two columns are equal. If the
    resampler drew the columns independently the statistic would be non-zero,
    so a degenerate interval is evidence that rows move together.
    """
    column = np.linspace(1.0, 20.0, 20)
    sample = np.column_stack([column, column])

    def difference(resampled: np.ndarray) -> float:
        return float(resampled[:, 0].mean() - resampled[:, 1].mean())

    interval = bootstrap_statistic_interval(
        sample, difference, resampling_unit="session", resamples=100, seed=4
    )
    assert interval.low == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert interval.high == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_require_resampling_unit_accepts_the_declared_unit() -> None:
    """The guard returns the interval so it can wrap a return value."""
    interval = bootstrap_mean_interval(
        np.arange(5.0), resampling_unit="session", resamples=20, seed=1
    )
    assert interval.require_resampling_unit("session") is interval


def test_require_resampling_unit_rejects_a_different_unit() -> None:
    """The message says why the disagreement is not cosmetic."""
    interval = bootstrap_mean_interval(
        np.arange(5.0), resampling_unit="session", resamples=20, seed=1
    )
    with pytest.raises(ValueError, match="too narrow by the within-session correlation"):
        interval.require_resampling_unit("attempt")


def test_contains_is_a_closed_test() -> None:
    """A value equal to an end is inside, because the ends are observed values."""
    interval = BootstrapInterval(
        point=1.0,
        low=0.5,
        high=1.5,
        confidence=0.95,
        resamples=10,
        resampling_unit="session",
        n_units=10,
        seed=1,
    )
    assert interval.contains(0.5)
    assert interval.contains(1.5)
    assert not interval.contains(0.4999)
    assert not interval.contains(1.5001)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, 95.0])
def test_invalid_confidence_rejected(confidence: float) -> None:
    """Coverage is a fraction strictly inside (0, 1), not a percentage."""
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        bootstrap_mean_interval(
            np.arange(5.0), resampling_unit="session", confidence=confidence, resamples=10
        )


@pytest.mark.parametrize("resamples", [0, 1, -5, 2.5, "100"])
def test_invalid_resample_count_rejected(resamples: object) -> None:
    """Two is the smallest count for which a percentile means anything."""
    with pytest.raises(ValueError, match="resamples must be an integer >= 2"):
        bootstrap_mean_interval(
            np.arange(5.0),
            resampling_unit="session",
            resamples=resamples,  # type: ignore[arg-type]
        )


def test_empty_sample_rejected() -> None:
    """A bootstrap needs at least one unit to resample."""
    with pytest.raises(ValueError, match="at least one resampling unit"):
        bootstrap_mean_interval(np.asarray([]), resampling_unit="session", resamples=10)


def test_unnamed_resampling_unit_rejected() -> None:
    """An interval whose unit is unstated cannot be checked by the caller."""
    with pytest.raises(ValueError, match="resampling_unit must be a non-empty name"):
        bootstrap_statistic_interval(
            np.arange(5.0), lambda values: float(values.mean()), resampling_unit="",
            resamples=10,
        )


def test_three_dimensional_sample_rejected() -> None:
    """Rows are the unit, so a sample has at most two dimensions."""
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        bootstrap_statistic_interval(
            np.zeros((2, 2, 2)), lambda values: 0.0, resampling_unit="session", resamples=10
        )


def test_mean_interval_rejects_a_two_dimensional_sample() -> None:
    """The mean helper is one-dimensional by contract."""
    with pytest.raises(ValueError, match="one-dimensional for a mean"):
        bootstrap_mean_interval(np.zeros((3, 3)), resampling_unit="session", resamples=10)


def test_wider_confidence_gives_a_wider_interval() -> None:
    """A 99 per cent interval contains a 90 per cent one from the same draws.

    Both use the same seed, so the resampled statistics are the same sample and
    the comparison is between two quantiles of one empirical distribution,
    where the ordering is a property of the quantile function and not a
    sampling outcome. A strict inequality is still avoided: with a discrete
    resample distribution the two quantiles can coincide.
    """
    sample = np.linspace(0.0, 10.0, 60)
    narrow = bootstrap_mean_interval(
        sample, resampling_unit="session", confidence=0.90, resamples=400, seed=8
    )
    wide = bootstrap_mean_interval(
        sample, resampling_unit="session", confidence=0.99, resamples=400, seed=8
    )
    assert wide.low <= narrow.low
    assert wide.high >= narrow.high
