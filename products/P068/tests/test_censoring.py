"""Kaplan-Meier and inverse-probability-of-censoring weights."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.censoring import (
    censoring_summary,
    ipcw_weights,
    kaplan_meier,
)


def test_kaplan_meier_hand_calculated_product_limit():
    # durations 1..5, events at 1, 3 and 4; censored at 2 and 5.
    #   t=1: 5 at risk, 1 event -> S = 1 - 1/5         = 0.8
    #   t=3: 3 at risk, 1 event -> S = 0.8 * (1 - 1/3) = 0.533333...
    #   t=4: 2 at risk, 1 event -> S = 0.5333 * (1/2)  = 0.266667
    km = kaplan_meier(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([True, False, True, True, False]),
    )
    np.testing.assert_allclose(km.times, [1.0, 3.0, 4.0])
    np.testing.assert_allclose(km.survival, [0.8, 0.8 * 2.0 / 3.0, 0.8 * 2.0 / 3.0 * 0.5],
                               rtol=1e-12)
    np.testing.assert_array_equal(km.n_at_risk, [5, 3, 2])
    np.testing.assert_array_equal(km.n_events, [1, 1, 1])


def test_kaplan_meier_with_no_censoring_is_the_empirical_survivor():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    km = kaplan_meier(d, np.ones(4, dtype=bool))
    np.testing.assert_allclose(km.survival, [0.75, 0.5, 0.25, 0.0], rtol=1e-12)


def test_kaplan_meier_with_no_events_is_identically_one():
    km = kaplan_meier(np.array([1.0, 2.0]), np.zeros(2, dtype=bool))
    assert km.times.size == 0
    np.testing.assert_allclose(km.evaluate(np.array([0.5, 10.0])), [1.0, 1.0])


def test_evaluate_left_limit_excludes_the_event_at_that_time():
    km = kaplan_meier(np.array([1.0, 2.0, 3.0]), np.ones(3, dtype=bool))
    # S(2-) is the value after t=1, which is 2/3; S(2) is 1/3.
    assert float(km.evaluate(2.0, left_limit=True)) == pytest.approx(2.0 / 3.0)
    assert float(km.evaluate(2.0, left_limit=False)) == pytest.approx(1.0 / 3.0)


def test_ties_put_events_before_censorings():
    # One event and one censoring both recorded at t = 2. The censored unit must
    # still be counted in the risk set for that event: 3 at risk, not 2.
    km = kaplan_meier(
        np.array([1.0, 2.0, 2.0, 3.0]),
        np.array([False, True, False, True]),
    )
    assert km.n_at_risk[0] == 3


@pytest.mark.parametrize(
    ("durations", "events", "match"),
    [
        (np.array([]), np.array([]), "at least one"),
        (np.array([1.0, 2.0]), np.array([True]), "same length"),
        (np.array([0.0, 1.0]), np.array([True, True]), "strictly positive"),
        (np.array([np.inf, 1.0]), np.array([True, True]), "finite"),
    ],
)
def test_kaplan_meier_validation(durations, events, match):
    with pytest.raises(ValueError, match=match):
        kaplan_meier(durations, events)


def test_ipcw_weights_are_zero_on_censored_units():
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e = np.array([True, False, True, False, True])
    w = ipcw_weights(d, e)
    assert np.all(w[~e] == 0.0)
    assert np.all(w[e] > 0.0)


def test_ipcw_weights_increase_with_the_observed_duration():
    """Late events survived more censoring opportunities, so they weigh more."""
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    e = np.array([True, False, True, False, True, True])
    w = ipcw_weights(d, e, normalise=False)
    observed = w[e]
    assert np.all(np.diff(observed) >= -1e-12)


def test_ipcw_weights_are_all_one_when_nothing_is_censored():
    d = np.array([5.0, 6.0, 7.0, 8.0])
    e = np.ones(4, dtype=bool)
    np.testing.assert_allclose(ipcw_weights(d, e, normalise=False), np.ones(4))


def test_ipcw_weights_respect_the_cap():
    rng = np.random.default_rng(0)
    d = np.sort(rng.uniform(1.0, 100.0, 60))
    e = d < 90.0
    w = ipcw_weights(d, e, weight_cap=3.0, normalise=False)
    assert w.max() <= 3.0 + 1e-12


def test_ipcw_normalisation_leaves_the_mean_non_zero_weight_at_one():
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e = np.array([True, False, True, False, True])
    w = ipcw_weights(d, e, normalise=True)
    assert w[w > 0].mean() == pytest.approx(1.0)


def test_ipcw_rejects_a_cap_at_or_below_one():
    with pytest.raises(ValueError, match="weight_cap"):
        ipcw_weights(np.array([1.0, 2.0]), np.array([True, False]), weight_cap=1.0)


def test_censoring_summary_counts():
    d = np.array([10.0, 20.0, 30.0, 40.0])
    e = np.array([True, False, True, False])
    s = censoring_summary(d, e)
    assert s["n"] == 4.0
    assert s["n_complete"] == 2.0
    assert s["n_censored"] == 2.0
    assert s["censored_fraction"] == pytest.approx(0.5)
    assert s["median_complete_duration_days"] == pytest.approx(20.0)
