"""Head-to-head harness: fallbacks, splits and the full integration path."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from degradepredict.benchmark import (
    BASELINE_FITTERS,
    FALLBACK_QUANTILES,
    baseline_forecasts,
    evaluate_forecasts,
    run_head_to_head,
)
from degradepredict.generate import DEFAULT_LIMIT, generate_population

SMALL = {"n_estimators": 40, "max_depth": 2}


def test_fallback_quantiles_are_increasing_and_span_the_declared_range():
    assert np.all(np.diff(FALLBACK_QUANTILES) > 0)
    assert FALLBACK_QUANTILES[0] > 1.0
    assert FALLBACK_QUANTILES[-1] < 10000.0


@pytest.mark.parametrize("model", sorted(BASELINE_FITTERS))
def test_every_baseline_produces_one_forecast_per_unit(model):
    pop = generate_population(25, seed=17)
    run = baseline_forecasts(pop, DEFAULT_LIMIT, model)
    assert len(run.forecasts) == len(pop)
    assert run.n_fallback >= 0
    assert run.seconds >= 0.0


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_forecasts(generate_population(3, seed=1), DEFAULT_LIMIT, "quadratic")


def test_a_window_the_baseline_cannot_fit_is_counted_as_a_fallback():
    pop = generate_population(4, seed=21)
    broken = dataclasses.replace(
        pop[0],
        values_percent=np.full(pop[0].times_days.size, 95.0)
        + 0.01 * pop[0].times_days,  # drifting upward: no crossing to report
    )
    run = baseline_forecasts([broken, *pop[1:]], DEFAULT_LIMIT, "linear")
    assert run.n_fallback == 1
    assert run.forecasts[0].model.endswith("-fallback")


def test_evaluate_forecasts_reports_the_split_keys():
    pop = generate_population(40, seed=23)
    run = baseline_forecasts(pop, DEFAULT_LIMIT, "linear")
    scores = evaluate_forecasts(run.forecasts, pop)
    for key in (
        "coverage_complete",
        "coverage_censored",
        "coverage_pooled",
        "crps_days_complete",
        "crps_days_censored",
        "pinball_days_complete",
        "pinball_days_censored",
        "falsified_fraction_censored",
        "pit_ks_statistic",
    ):
        assert key in scores, key
    assert scores["n_complete"] + scores["n_censored"] == scores["n"]


def test_run_head_to_head_covers_every_model_on_the_same_units():
    train = generate_population(220, seed=2026)
    test = generate_population(60, seed=2027)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, forecaster_kwargs=SMALL)
    assert set(results) == {
        "linear",
        "exponential",
        "powerlaw-fade",
        "learned-ipcw",
        "learned-complete-case",
    }
    for payload in results.values():
        assert len(payload["run"].forecasts) == len(test)
        assert payload["scores"]["n"] == float(len(test))


def test_head_to_head_can_omit_the_unweighted_ablation():
    train = generate_population(220, seed=2026)
    test = generate_population(30, seed=2028)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, include_unweighted=False,
                               forecaster_kwargs=SMALL)
    assert "learned-complete-case" not in results


def test_coverage_is_never_summarised_by_the_pooled_number_alone():
    """The pooled figure must sit between the two split figures, not replace them."""
    train = generate_population(220, seed=2026)
    test = generate_population(80, seed=2029)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, include_unweighted=False,
                               forecaster_kwargs=SMALL)
    s = results["learned-ipcw"]["scores"]
    lo = min(s["coverage_complete"], s["coverage_censored"])
    hi = max(s["coverage_complete"], s["coverage_censored"])
    assert lo - 1e-12 <= s["coverage_pooled"] <= hi + 1e-12
