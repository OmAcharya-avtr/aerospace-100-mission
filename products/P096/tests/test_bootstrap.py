"""The event-level bootstrap: the resampling unit is declared AND asserted."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.bootstrap import (
    RESAMPLING_UNIT,
    BootstrapInterval,
    event_bootstrap,
)


def test_resampling_unit_is_the_handover_event():
    assert RESAMPLING_UNIT == "handover event"


def test_interval_rejects_any_other_resampling_unit():
    # The unit is not merely documented. A caller constructing an interval with
    # the slot as its unit is refused, because resampling slots within one
    # handover would report an interval narrower than the data support.
    with pytest.raises(ValueError, match="resampling_unit must be"):
        BootstrapInterval(
            point=1.0,
            low=0.5,
            high=1.5,
            confidence=0.95,
            resampling_unit="accounting slot",
            n_events=100,
            n_resamples=1000,
        )


def test_interval_brackets_the_mean_of_a_known_sample():
    # A 95 per cent percentile interval on the mean of 400 unit-variance normal
    # draws has half-width about 1.96 / sqrt(400) = 0.098. Asserted as a band,
    # not an equality: the interval is a random quantity.
    values = np.random.default_rng(7).normal(10.0, 1.0, 400)
    interval = event_bootstrap(values, lambda v: float(v.mean()), resamples=2000)
    assert interval.low < interval.point < interval.high
    assert 0.12 < interval.width < 0.26
    assert interval.n_events == 400
    assert interval.n_resamples == 2000
    assert interval.resampling_unit == RESAMPLING_UNIT


def test_only_the_event_axis_is_resampled():
    # A two-axis input keeps its second axis intact, so a statistic may be a
    # function of a whole per-event row.
    values = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
    interval = event_bootstrap(values, lambda v: float(v[:, 1].mean()), resamples=200)
    # Every row is identical, so every resample gives the same statistic and the
    # interval collapses onto the point. Exact here because no arithmetic
    # variability exists, not because floating point promises it: the statistic
    # is the mean of 50 copies of one number.
    assert interval.point == pytest.approx(2.0, rel=1e-15, abs=0.0)
    assert interval.width == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_reproducible_with_a_supplied_generator():
    values = np.random.default_rng(1).exponential(3.0, 200)
    first = event_bootstrap(
        values, lambda v: float(v.mean()), resamples=500, rng=np.random.default_rng(9)
    )
    second = event_bootstrap(
        values, lambda v: float(v.mean()), resamples=500, rng=np.random.default_rng(9)
    )
    assert first.low == second.low
    assert first.high == second.high


def test_default_generator_is_also_reproducible():
    values = np.random.default_rng(2).exponential(3.0, 120)
    first = event_bootstrap(values, lambda v: float(v.mean()), resamples=300)
    second = event_bootstrap(values, lambda v: float(v.mean()), resamples=300)
    assert first.low == second.low


def test_excludes_and_disjoint_from():
    values = np.random.default_rng(3).normal(5.0, 0.5, 300)
    interval = event_bootstrap(values, lambda v: float(v.mean()), resamples=800)
    assert interval.excludes(0.0)
    assert not interval.excludes(interval.point)
    far = BootstrapInterval(
        point=100.0,
        low=99.0,
        high=101.0,
        confidence=0.95,
        resampling_unit=RESAMPLING_UNIT,
        n_events=300,
        n_resamples=800,
    )
    assert interval.disjoint_from(far)
    assert far.disjoint_from(interval)


def test_too_few_events_raise():
    with pytest.raises(ValueError, match="at least 5 events"):
        event_bootstrap(np.arange(4.0), lambda v: float(v.mean()))


def test_scalar_input_raises():
    with pytest.raises(ValueError, match="at least one axis"):
        event_bootstrap(np.float64(1.0), lambda v: float(v))


@pytest.mark.parametrize("resamples", [0, -1])
def test_non_positive_resamples_raise(resamples):
    with pytest.raises(ValueError, match="at least 1"):
        event_bootstrap(np.arange(20.0), lambda v: float(v.mean()), resamples=resamples)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_confidence_out_of_range_raises(confidence):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        event_bootstrap(
            np.arange(20.0), lambda v: float(v.mean()), confidence=confidence
        )


def test_interval_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="exceeds high"):
        BootstrapInterval(
            point=1.0,
            low=2.0,
            high=1.0,
            confidence=0.95,
            resampling_unit=RESAMPLING_UNIT,
            n_events=10,
            n_resamples=10,
        )


def test_interval_rejects_degenerate_counts():
    for kwargs in ({"n_events": 0}, {"n_resamples": 0}):
        base = dict(
            point=1.0,
            low=0.0,
            high=2.0,
            confidence=0.95,
            resampling_unit=RESAMPLING_UNIT,
            n_events=10,
            n_resamples=10,
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match="at least 1"):
            BootstrapInterval(**base)
