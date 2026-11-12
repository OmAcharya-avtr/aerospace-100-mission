"""The learned predictor, its uncertainty output, and the comparison rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget.baselines import sustained_truth
from thermalbudget.bootstrap import BootstrapResult
from thermalbudget.predictor import (
    FEATURE_NAMES,
    NotEnoughRunsError,
    SteadyStatePredictor,
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
    rc_warmup_prediction,
    warmup_features,
)


def _reference(half_width: float, point: float = 60.0) -> BootstrapResult:
    return BootstrapResult(
        point=point,
        ci_low=point - half_width,
        ci_high=point + half_width,
        confidence=0.95,
        n_resamples=1000,
        standard_error=half_width / 2.0,
        bias=0.0,
        n_groups=3,
        n_samples=300,
        n_distinct_group_draws=27,
    )


def test_features_are_computed_from_the_warm_up_window_only(no_throttle_campaign):
    trace = no_throttle_campaign["soak_1.00_r1"]
    features = warmup_features(trace, 90.0)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    # The same window of a trace truncated after it gives identical features:
    # nothing in the feature vector reads beyond the window.
    truncated = trace.window(0.0, 200.0)
    assert np.allclose(features, warmup_features(truncated, 90.0), rtol=0.0, atol=0.0)


def test_features_refuse_a_window_longer_than_the_trace(no_throttle_campaign):
    with pytest.raises(ValueError, match="shorter than the"):
        warmup_features(no_throttle_campaign["burst_1.00_a"], 90.0)
    with pytest.raises(ValueError, match="must be positive"):
        warmup_features(no_throttle_campaign["soak_1.00_r1"], 0.0)


def test_features_refuse_a_window_with_too_few_finite_samples(no_throttle_campaign):
    trace = no_throttle_campaign["soak_1.00_r1"]
    broken = trace.window(0.0, 200.0)
    broken.die_temperature_c[:] = math.nan
    with pytest.raises(ValueError, match="fewer than the 20 required"):
        warmup_features(broken, 90.0)


def test_the_dataset_is_grouped_by_run_and_covers_every_duty(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    assert dataset.n_rows == 36  # 6 runs x 6 declared window lengths
    assert dataset.runs == (
        "soak_0.25",
        "soak_0.50",
        "soak_0.75",
        "soak_1.00_r1",
        "soak_1.00_r2",
        "soak_1.00_r3",
    )
    assert dataset.duties == (0.25, 0.5, 0.75, 1.0)
    assert dataset.excluded == ()
    assert np.all(np.isfinite(dataset.features))


def test_the_dataset_excludes_an_unequilibrated_run_by_name(no_throttle_campaign, short_run):
    dataset = build_dataset([*no_throttle_campaign.soaks(), short_run])
    assert short_run.label not in dataset.runs
    assert any(label == short_run.label for label, _ in dataset.excluded)


def test_the_dataset_refuses_fewer_than_two_contributing_runs(short_run):
    with pytest.raises(NotEnoughRunsError, match="fewer than 2 contributing runs"):
        build_dataset([short_run])


def test_the_predictor_returns_a_mean_and_a_positive_sigma(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    model = SteadyStatePredictor(kind="ridge", n_models=40).fit(
        dataset.features, dataset.die_target_c, dataset.run
    )
    mean, sigma = model.predict(dataset.features[:5])
    assert mean.shape == (5,)
    assert sigma.shape == (5,)
    assert np.all(sigma > 0.0)


def test_the_predictor_is_reproducible_from_its_seed(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    first = SteadyStatePredictor(n_models=30, seed=5).fit(
        dataset.features, dataset.die_target_c, dataset.run
    )
    second = SteadyStatePredictor(n_models=30, seed=5).fit(
        dataset.features, dataset.die_target_c, dataset.run
    )
    assert np.allclose(
        first.predict(dataset.features)[0],
        second.predict(dataset.features)[0],
        rtol=0.0,
        atol=0.0,
    )


def test_the_predictor_refuses_to_predict_before_fitting():
    with pytest.raises(RuntimeError, match="before fit"):
        SteadyStatePredictor().predict(np.zeros((1, len(FEATURE_NAMES))))


def test_the_predictor_validates_its_inputs(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    with pytest.raises(ValueError, match="do not agree"):
        SteadyStatePredictor().fit(dataset.features, dataset.die_target_c[:-1], dataset.run)
    with pytest.raises(NotEnoughRunsError, match="at least 2 groups"):
        SteadyStatePredictor().fit(
            dataset.features, dataset.die_target_c, ["one"] * dataset.n_rows
        )
    with pytest.raises(ValueError, match="must be finite"):
        broken = dataset.features.copy()
        broken[0, 0] = math.nan
        SteadyStatePredictor().fit(broken, dataset.die_target_c, dataset.run)
    with pytest.raises(ValueError, match="kind must be"):
        SteadyStatePredictor(kind="svm").fit(
            dataset.features, dataset.die_target_c, dataset.run
        )


def test_the_rc_baseline_sees_exactly_the_same_warm_up_window(no_throttle_campaign):
    trace = no_throttle_campaign["soak_1.00_r1"]
    prediction = rc_warmup_prediction(trace, 90.0)
    assert math.isfinite(prediction)
    # And it refuses, returning nan rather than a fallback, on a window too short
    # to identify an asymptote.
    assert math.isnan(rc_warmup_prediction(trace, 1.0))


def test_held_out_duty_evaluates_every_duty_and_names_what_it_skipped(
    no_throttle_campaign,
):
    dataset = build_dataset(no_throttle_campaign.soaks())
    result = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), n_models=40)
    assert len(result.splits) == 4
    assert {split.held_out for split in result.splits} == {
        "duty 0.25",
        "duty 0.5",
        "duty 0.75",
        "duty 1",
    }
    assert result.skipped == ()
    assert math.isfinite(result.learned_mae_c)
    assert math.isfinite(result.rc_mae_c)
    assert result.margin_c == pytest.approx(
        result.rc_mae_c - result.learned_mae_c, rel=1e-12, abs=0.0
    )


def test_the_forest_cannot_extrapolate_and_the_suite_records_that(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    forest = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), kind="forest")
    hottest = next(split for split in forest.splits if split.held_out == "duty 1")
    truths = [row[2] for row in hottest.rows]
    predictions = [row[3] for row in hottest.rows]
    # Every prediction is bounded by the training targets, so on the hottest
    # held-out duty the forest predicts something cooler than the truth.
    assert max(predictions) < min(truths)


def test_a_margin_inside_the_run_to_run_interval_is_not_an_improvement(
    no_throttle_campaign,
):
    dataset = build_dataset(no_throttle_campaign.soaks())
    result = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), n_models=40)
    # A deliberately huge reference interval must make ANY margin
    # indistinguishable, in either direction. This is the rule, asserted
    # directly rather than inferred from a particular dataset.
    verdict = compare_against_rc(result, _reference(half_width=1000.0))
    assert verdict.verdict == "NOT_DISTINGUISHABLE"
    assert verdict.ratio < 1.0
    assert "NOT DISTINGUISHABLE" in verdict.statement


def test_a_margin_outside_the_interval_is_attributed_to_whichever_side_won(
    no_throttle_campaign,
):
    dataset = build_dataset(no_throttle_campaign.soaks())
    result = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), n_models=40)
    verdict = compare_against_rc(result, _reference(half_width=1e-6))
    assert verdict.verdict in ("LEARNED_BETTER", "BASELINE_BETTER")
    assert verdict.ratio > 1.0
    # On these fixtures the data are generated FROM the RC form, so the RC
    # baseline is expected to win. That is recorded, not softened.
    assert verdict.verdict == "BASELINE_BETTER"
    assert result.margin_c < 0.0


def test_a_non_finite_reference_claims_nothing():
    class _Result:
        margin_c = 1.0

    verdict = compare_against_rc(_Result(), _reference(half_width=0.0))
    assert verdict.verdict == "NOT_DISTINGUISHABLE"
    assert "nothing is claimed" in verdict.statement


def test_the_comparison_uses_baseline_3_run_to_run_interval(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    result = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), n_models=40)
    truth = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=2000)
    verdict = compare_against_rc(result, truth.die_c)
    assert verdict.reference_half_width_c == pytest.approx(
        truth.die_c.half_width, rel=1e-12, abs=0.0
    )
    assert verdict.verdict in ("NOT_DISTINGUISHABLE", "LEARNED_BETTER", "BASELINE_BETTER")


def test_coverage_is_reported_as_a_fraction(no_throttle_campaign):
    dataset = build_dataset(no_throttle_campaign.soaks())
    result = leave_one_duty_out(dataset, no_throttle_campaign.soaks(), n_models=40)
    assert 0.0 <= result.coverage <= 1.0
    for split in result.splits:
        assert 0.0 <= split.learned_coverage <= 1.0
