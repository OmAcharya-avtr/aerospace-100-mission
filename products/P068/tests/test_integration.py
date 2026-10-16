"""End-to-end: generate, fit, forecast, score, with the censoring split intact."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.baselines import exponential_baseline, linear_baseline
from degradepredict.benchmark import baseline_forecasts, run_head_to_head
from degradepredict.censoring import censoring_summary
from degradepredict.generate import DEFAULT_LIMIT, generate_population
from degradepredict.metrics import coverage_by_censoring, crps
from degradepredict.physics import powerlaw_fade_baseline
from degradepredict.report import format_head_to_head

SMALL = {"n_estimators": 60, "max_depth": 2}


@pytest.fixture(scope="module")
def pipeline():
    train = generate_population(400, seed=20261016)
    test = generate_population(120, seed=20261017)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, forecaster_kwargs=SMALL)
    return train, test, results


def test_the_whole_pipeline_runs_and_scores_every_unit(pipeline):
    _, test, results = pipeline
    for name, payload in results.items():
        assert len(payload["run"].forecasts) == len(test), name
        assert np.isfinite(payload["scores"]["crps_days_complete"])
        assert np.isfinite(payload["scores"]["crps_days_censored"])


def test_censored_and_complete_coverage_are_both_reported_and_differ(pipeline):
    _, _, results = pipeline
    s = results["learned-ipcw"]["scores"]
    assert s["n_complete"] > 0 and s["n_censored"] > 0
    assert s["coverage_complete"] != s["coverage_censored"]


def test_the_report_renders_both_columns(pipeline):
    _, _, results = pipeline
    text = format_head_to_head(results)
    assert "cov cmpl" in text and "cov cens" in text
    assert "lowest CRPS on censored trajectories" in text


def test_each_baseline_is_near_nominal_on_its_own_shape_family():
    """A correctly specified baseline recovers its nominal coverage.

    Linear-family units scored by the linear baseline, exponential-family units
    by the exponential baseline. Both are required to clear 0.70 at a nominal
    0.90; the gap to nominal is the discretisation of the origin and the finite
    sample, and the measured values are in
    ``validation/validate_headtohead_output.txt``. The same baselines collapse
    off their own family, which is the finding this repository is about and is
    checked in the next test.
    """
    pop = generate_population(240, seed=4242)
    truth = np.array([tr.remaining_true_days for tr in pop])
    event = np.array([tr.event for tr in pop])
    family = np.array([tr.family for tr in pop])
    for model, own in (("linear", "linear"), ("exponential", "exponential")):
        run = baseline_forecasts(pop, DEFAULT_LIMIT, model)
        mask = family == own
        sub = [f for f, keep in zip(run.forecasts, mask, strict=True) if keep]
        cov = coverage_by_censoring(sub, truth[mask], event[mask], level=0.90)
        assert cov["coverage_pooled"] > 0.70, (model, cov["coverage_pooled"])


def test_a_misspecified_shape_destroys_the_interval():
    """The power-law fit applied to straight-line units must under-cover badly."""
    pop = generate_population(240, seed=4242)
    truth = np.array([tr.remaining_true_days for tr in pop])
    event = np.array([tr.event for tr in pop])
    mask = np.array([tr.family == "linear" for tr in pop])
    run = baseline_forecasts(pop, DEFAULT_LIMIT, "powerlaw-fade")
    sub = [f for f, keep in zip(run.forecasts, mask, strict=True) if keep]
    cov = coverage_by_censoring(sub, truth[mask], event[mask], level=0.90)
    assert cov["coverage_pooled"] < 0.30


def test_pooled_coverage_hides_the_split(pipeline):
    """The pooled figure must be far from at least one of the two halves."""
    _, _, results = pipeline
    s = results["learned-ipcw"]["scores"]
    gap = max(
        abs(s["coverage_pooled"] - s["coverage_complete"]),
        abs(s["coverage_pooled"] - s["coverage_censored"]),
    )
    assert gap > 0.10


def test_censoring_summary_of_the_test_set_matches_the_scores(pipeline):
    _, test, results = pipeline
    summary = censoring_summary(
        np.array([tr.remaining_observed_days for tr in test]),
        np.array([tr.event for tr in test]),
    )
    s = results["linear"]["scores"]
    assert summary["n_complete"] == s["n_complete"]
    assert summary["n_censored"] == s["n_censored"]


def test_all_three_baselines_fit_the_same_window_and_disagree():
    tr = generate_population(1, seed=20261016)[0]
    fits = [
        linear_baseline(tr.times_days, tr.values_percent, DEFAULT_LIMIT),
        exponential_baseline(tr.times_days, tr.values_percent, DEFAULT_LIMIT),
        powerlaw_fade_baseline(tr.times_days, tr.values_percent, DEFAULT_LIMIT),
    ]
    crossings = [f.crossing_days for f in fits]
    assert len(set(np.round(crossings, 6))) == 3
    assert all(c > tr.origin_days for c in crossings)


def test_crps_is_finite_and_positive_for_every_model(pipeline):
    _, test, results = pipeline
    truth = np.array([tr.remaining_true_days for tr in test])
    for payload in results.values():
        scores = crps(payload["run"].forecasts, truth)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0.0)
