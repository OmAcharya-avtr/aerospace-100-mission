"""The three baselines and the headline, in both regimes the fixtures provide."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget.baselines import (
    BURST_SKIP_S,
    burst_baseline,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from thermalbudget.provenance import Origin


def test_baseline_1_predicts_the_burst_figure_itself(no_throttle_campaign):
    base = burst_baseline(no_throttle_campaign.bursts(), n_resamples=2000)
    assert base.prediction_iters_per_s == base.throughput.point
    assert base.n_runs == 2
    assert base.throughput.n_distinct_group_draws == 4
    assert base.skip_s == BURST_SKIP_S
    assert set(base.per_run_mean) == {"burst_1.00_a", "burst_1.00_b"}
    assert base.throughput.ci_low < base.throughput.point < base.throughput.ci_high


def test_baseline_1_reports_that_a_burst_is_not_equilibrated(no_throttle_campaign):
    base = burst_baseline(no_throttle_campaign.bursts(), n_resamples=500)
    for verdict in base.equilibrium.values():
        assert not verdict.equilibrated


def test_baseline_1_refuses_one_run_and_refuses_a_starved_run(no_throttle_campaign):
    with pytest.raises(ValueError, match="at least 2 burst runs"):
        burst_baseline(no_throttle_campaign.bursts()[:1])
    short = no_throttle_campaign["burst_1.00_a"].window(0.0, 5.0)
    with pytest.raises(ValueError, match="fewer than the 10 required"):
        burst_baseline([short, no_throttle_campaign["burst_1.00_b"]])


def test_baseline_3_includes_only_equilibrated_runs(no_throttle_campaign):
    truth = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=2000)
    assert len(truth.included) == 3
    assert truth.excluded == ()
    assert truth.throughput.n_distinct_group_draws == 27
    assert set(truth.within_run_throughput) == set(truth.included)
    assert truth.die_c.point > 0.0


def test_baseline_3_excludes_a_run_that_is_not_equilibrated_and_names_it(
    no_throttle_campaign, short_run
):
    traces = [*no_throttle_campaign.soaks(1.0), short_run]
    truth = sustained_truth(traces, n_resamples=500)
    assert short_run.label not in truth.included
    excluded = dict(truth.excluded)
    assert short_run.label in excluded
    assert "NOT_EQUILIBRATED" in excluded[short_run.label]


def test_baseline_3_refuses_rather_than_averaging_unequilibrated_runs(short_run):
    with pytest.raises(ValueError, match="at least 2 equilibrated runs"):
        sustained_truth([short_run], n_resamples=100)


def test_baseline_3_within_and_between_run_levels_are_reported_separately(
    no_throttle_campaign,
):
    truth = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=2000)
    within = [result.half_width for result in truth.within_run_throughput.values()]
    assert all(value > 0.0 for value in within)
    # Both levels exist as numbers the reader can compare; which dominates is a
    # property of the data and is not asserted here.
    assert truth.throughput.half_width > 0.0


def test_baseline_2_reduces_to_baseline_1_when_the_clock_never_moves(no_throttle_campaign):
    base1 = burst_baseline(no_throttle_campaign.bursts(), n_resamples=500)
    base2 = rc_baseline(
        no_throttle_campaign.soaks(1.0),
        no_throttle_campaign.cooldowns(),
        no_throttle_campaign.bursts(),
    )
    assert base2.clock_degenerate
    assert base2.reduces_to_baseline_1
    assert base2.predicted_throughput_iters_per_s == pytest.approx(
        base1.prediction_iters_per_s, rel=0.002, abs=0.0
    )
    assert any("NOT IDENTIFIABLE" in note for note in base2.notes)


def test_baseline_2_predicts_a_throttled_steady_state_when_the_clock_does_move(
    throttled_campaign,
):
    base2 = rc_baseline(
        throttled_campaign.soaks(1.0),
        throttled_campaign.cooldowns(),
        throttled_campaign.bursts(),
    )
    assert not base2.clock_degenerate
    assert not base2.reduces_to_baseline_1
    assert base2.predicted_clock_khz < base2.anchor_clock_khz
    assert base2.predicted_throughput_iters_per_s < base2.anchor_throughput_iters_per_s


def test_baseline_2_reports_both_taus_side_by_side_without_choosing(no_throttle_campaign):
    base2 = rc_baseline(
        no_throttle_campaign.soaks(1.0),
        no_throttle_campaign.cooldowns(),
        no_throttle_campaign.bursts(),
    )
    assert len(base2.tau_rise_s) == 3
    assert len(base2.tau_cool_s) >= 3
    assert math.isfinite(base2.tau_rise_mean_s)
    assert math.isfinite(base2.tau_cool_mean_s)
    # The fixture's two time constants differ by design, so the ratio is not 1.
    assert base2.tau_ratio > 1.5
    # And at least one independent route produced a number.
    assert len(base2.independent_tau) >= 1


def test_baseline_2_anchors_on_the_burst_not_on_the_soak_equilibrium(no_throttle_campaign):
    base2 = rc_baseline(
        no_throttle_campaign.soaks(1.0),
        no_throttle_campaign.cooldowns(),
        no_throttle_campaign.bursts(),
    )
    burst_samples = np.concatenate(
        [
            trace.throughput_iters_per_s[
                np.isfinite(trace.throughput_iters_per_s) & (trace.t_rel > BURST_SKIP_S)
            ]
            for trace in no_throttle_campaign.bursts()
        ]
    )
    assert base2.anchor_throughput_iters_per_s == pytest.approx(
        float(burst_samples.mean()), rel=1e-12, abs=0.0
    )


def test_baseline_2_input_validation(no_throttle_campaign):
    with pytest.raises(ValueError, match="at least one heating run"):
        rc_baseline([], no_throttle_campaign.cooldowns(), no_throttle_campaign.bursts())
    with pytest.raises(ValueError, match="throughput anchor"):
        rc_baseline(no_throttle_campaign.soaks(1.0), no_throttle_campaign.cooldowns(), [])


def test_baseline_2_names_a_cooldown_whose_fit_refused(no_throttle_campaign):
    # cool_a follows a 20 s burst, so its decay amplitude is a couple of degrees
    # and the fit may refuse. Whether it refuses is data-dependent; what is
    # asserted is that a refusal is NAMED rather than dropped.
    base2 = rc_baseline(
        no_throttle_campaign.soaks(1.0),
        no_throttle_campaign.cooldowns(),
        no_throttle_campaign.bursts(),
    )
    refused = [note for note in base2.notes if "refused" in note]
    for note in refused:
        assert "cool" in note or "rise" in note


def test_the_headline_reports_indistinguishable_when_the_interval_contains_zero(
    no_throttle_campaign,
):
    base1 = burst_baseline(no_throttle_campaign.bursts(), n_resamples=4000)
    base3 = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=4000)
    head = overstatement(
        base1.throughput_groups, base3.throughput_groups, n_resamples=4000
    )
    assert head.verdict == "INDISTINGUISHABLE"
    assert head.overstatement.contains(0.0)
    assert head.sustained_fraction.contains(1.0)
    assert "indistinguishable" in head.statement
    assert "SYNTHETIC" in head.statement


def test_the_headline_reports_a_gap_when_the_interval_excludes_zero(throttled_campaign):
    base1 = burst_baseline(throttled_campaign.bursts(), n_resamples=4000)
    base3 = sustained_truth(throttled_campaign.soaks(1.0), n_resamples=4000)
    head = overstatement(
        base1.throughput_groups, base3.throughput_groups, n_resamples=4000
    )
    assert head.verdict == "OVERSTATES"
    assert not head.overstatement.contains(0.0)
    assert head.overstatement.ci_low > 0.0


def test_the_two_headline_forms_are_exact_transforms_of_each_other(throttled_campaign):
    base1 = burst_baseline(throttled_campaign.bursts(), n_resamples=2000)
    base3 = sustained_truth(throttled_campaign.soaks(1.0), n_resamples=2000)
    head = overstatement(
        base1.throughput_groups, base3.throughput_groups, n_resamples=2000
    )
    # Point estimates: sustained/burst and burst/sustained - 1 are reciprocals
    # shifted by one, exactly, because both come from the same two numbers.
    assert 1.0 / head.sustained_fraction.point - 1.0 == pytest.approx(
        head.overstatement.point, rel=1e-12, abs=0.0
    )
    # The interval ENDS swap, because the transform is decreasing: the high end
    # of the fraction maps to the low end of the overstatement. They agree to
    # within percentile INTERPOLATION and not exactly, and the reason is worth
    # stating rather than hiding behind a loose tolerance. numpy.percentile
    # interpolates linearly between the two order statistics that bracket the
    # requested quantile. A monotone NONLINEAR transform does not commute with
    # linear interpolation, so transforming the draws and then interpolating is
    # not the same arithmetic as interpolating and then transforming. The
    # measured disagreement on this fixture is about 1.6e-8 relative; the bound
    # below is 1e-6, which is three orders of magnitude inside the width of the
    # interval itself and two above the effect.
    assert 1.0 / head.sustained_fraction.ci_high - 1.0 == pytest.approx(
        head.overstatement.ci_low, rel=1e-6, abs=0.0
    )
    assert 1.0 / head.sustained_fraction.ci_low - 1.0 == pytest.approx(
        head.overstatement.ci_high, rel=1e-6, abs=0.0
    )


def test_the_headline_refuses_a_zero_or_negative_throughput_side(no_throttle_campaign):
    idle = no_throttle_campaign["idle_pre"].throughput_iters_per_s
    base3 = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=200)
    with pytest.raises(ValueError, match="non-positive"):
        overstatement([idle, idle], base3.throughput_groups, n_resamples=200)


def test_the_headline_carries_the_origin_it_was_given(no_throttle_campaign):
    base1 = burst_baseline(no_throttle_campaign.bursts(), n_resamples=500)
    base3 = sustained_truth(no_throttle_campaign.soaks(1.0), n_resamples=500)
    head = overstatement(
        base1.throughput_groups,
        base3.throughput_groups,
        n_resamples=500,
        origin=Origin.SYNTHETIC,
    )
    assert head.origin is Origin.SYNTHETIC
