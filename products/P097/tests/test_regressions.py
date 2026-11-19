"""Pinned counterexamples and pinned behaviour, with literal inputs in the body.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is LOST
on a fresh clone. Every counterexample Hypothesis found during this product's
build is pinned here as a named test with its literal input, and every
behaviour that a draft assertion got wrong is pinned here as a statement of
what is actually true.

Each entry says what was asserted, what falsified it, and what the correct
claim is. Reachability was checked for every one: the input is inside the
declared strategy of the property test it came from, so it is a real case and
not an artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.geometric import (
    expected_attempts_to_first_pass,
    failure_probability_for_expected_attempts,
)
from flakerank.history import RunHistory
from flakerank.predictor import (
    IntermittencyCostPredictor,
    build_rows,
    make_splits,
)

#: Double-precision unit roundoff.
UNIT_ROUNDOFF = 2.0**-53


def test_regression_round_trip_conditioning_at_expectation_25929728() -> None:
    """Hypothesis counterexample: ``expectation = 25929728.0``.

    FOUND BY: ``test_identity_probability_expectation_round_trip`` in
    ``tests/test_geometric.py``, whose strategy is
    ``st.floats(min_value=1.0, max_value=1e9)``. The value is inside that
    strategy, so the counterexample is reachable and is a real case.

    WHAT WAS ASSERTED: that ``E -> p -> E`` round-trips to ``rel=1e-9``.

    WHAT HAPPENED: the round trip returned 25929728.03245014, a relative
    difference of 1.2514647096395493e-09, which is above 1e-9.

    WHAT IS ACTUALLY TRUE: the round trip's relative error grows linearly with
    ``E``, because ``p = 1 - 1/E`` carries an absolute rounding error of order
    ``eps`` once ``p`` is near 1, and recovering ``E' = 1/(1 - p)`` turns that
    into a relative error of ``eps * E``. The property test's tolerance was
    changed to ``4 * eps * E``, which follows the conditioning. A constant
    tolerance was the wrong shape, not a wrong number.
    """
    expectation = 25929728.0
    probability = failure_probability_for_expected_attempts(expectation)
    recovered = expected_attempts_to_first_pass(probability)
    # The measured value, pinned so a change in either direction is caught.
    assert recovered == pytest.approx(25929728.03245014, rel=1e-13, abs=0.0)
    relative = abs(recovered - expectation) / expectation
    assert relative == pytest.approx(1.2514647096395493e-09, rel=1e-12, abs=0.0)
    # The conditioning-following tolerance admits it; a constant 1e-9 does not.
    assert relative <= 4.0 * UNIT_ROUNDOFF * expectation
    assert relative > 1e-9


def test_regression_ensemble_mean_can_sit_outside_the_tree_quantile_interval(
    history: RunHistory,
) -> None:
    """Pinned behaviour: the confidence interval need not contain the mean.

    WHAT WAS ASSERTED: in ``tests/test_predictor.py``, that the ensemble mean
    lies inside the 90 per cent central interval of the tree predictions.

    WHAT HAPPENED: on the held-out TESTS split, 47 of 88 rows have the mean
    strictly above the upper quantile. The first such row has a mean of
    1.00719352e-05 against an interval of exactly [0.0, 0.0].

    WHAT IS ACTUALLY TRUE: for a test that almost never fails, nearly every
    tree predicts ``log1p(0) = 0`` and a handful predict a positive value, so
    the distribution across trees is a point mass at zero with a thin right
    tail. A central quantile interval of a distribution that skewed does not
    contain its own mean. A second draft then asserted that every escaping row
    has width exactly zero, and that failed too: 2 of 55 rows on the held-out
    TIME split have a width of 7.58412808e-04. The claim that survives is that
    the escape happens only where the interval is near-degenerate.

    The consequence for the reader is in ``MODEL_CARD.md`` under failure cases:
    the interval is an interval on the trees' answers, not a predictive
    interval for the next window's cost.
    """
    rows = build_rows(
        history, origins=(120, 150, 180, 210, 240, 270), lookback=120, horizon=30,
        minimum_attempts=8,
    )
    splits = make_splits(rows, train_origins=(120, 150, 180, 210), time_origins=(240, 270))
    model = IntermittencyCostPredictor().fit(splits["train"])

    prediction = model.predict(splits["heldout_tests"])
    outside = (prediction.mean < prediction.low) | (prediction.mean > prediction.high)
    assert int(np.count_nonzero(outside)) == 47
    assert prediction.mean[0] == pytest.approx(1.00719352e-05, rel=1e-6, abs=0.0)
    assert prediction.low[0] == 0.0
    assert prediction.high[0] == 0.0

    later = model.predict(splits["heldout_time"])
    outside_later = (later.mean < later.low) | (later.mean > later.high)
    assert int(np.count_nonzero(outside_later)) == 55
    widths = later.interval_width[outside_later]
    assert int(np.count_nonzero(widths > 0.0)) == 2
    assert float(widths.max()) == pytest.approx(7.58412808e-04, rel=1e-6, abs=0.0)


def test_regression_tau_b_and_scipy_disagree_only_on_a_fully_tied_vector() -> None:
    """Pinned behaviour: the one place this implementation and scipy differ.

    ``flakerank.disagreement.kendall_tau_b`` returns 0.0 when either vector is
    entirely tied, where the denominator of the tau-b formula is zero;
    ``scipy.stats.kendalltau`` returns NaN. The difference is declared in the
    module docstring and pinned here with literal inputs, so that a future
    change to either side is caught by name rather than by a property test
    that happens to draw a constant vector.
    """
    from scipy import stats

    from flakerank.disagreement import kendall_tau_b, spearman_rho

    tied = np.asarray([3.0, 3.0, 3.0, 3.0])
    untied = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau_b(tied, untied) == 0.0
    assert spearman_rho(tied, untied) == 0.0
    assert np.isnan(stats.kendalltau(tied, untied, variant="b").statistic)
    assert np.isnan(stats.spearmanr(tied, untied).statistic)


def test_regression_bracket_is_too_wide_to_catch_a_doubling_on_the_real_history(
    history: RunHistory,
) -> None:
    """Pinned limitation: the order-statistic bracket cannot catch every error.

    The bracket's width is set by the spread of the duration sample, and the
    committed history spans 0.011 s to 1848.115 s. A doubled accumulator total
    therefore still fits inside it, so the bracket catches structural errors
    (the wrong mask, a missing group, a sign) and not every multiplicative one.
    This is pinned as a test rather than left as a sentence, because it tells a
    reader exactly how much the bound check is worth. The control in
    ``tests/test_bounds.py`` uses a near-constant-duration history, where the
    bracket IS tight, to show the doubling is caught when the bound can see it.
    """
    from flakerank.baselines import accumulated_wasted_time
    from flakerank.bounds import wasted_time_bracket

    truth = accumulated_wasted_time(history, resamples=10, seed=5)
    doubled = wasted_time_bracket(history, 2.0 * truth.total_seconds)
    assert doubled.brackets, "pinned: the real history's bracket does not catch a doubling"
    tenfold = wasted_time_bracket(history, 10.0 * truth.total_seconds)
    assert not tenfold.brackets, "pinned: a tenfold error IS caught"
