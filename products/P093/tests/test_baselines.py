"""The three baselines, each against the synthetic closed form where there is one."""

from __future__ import annotations

import pytest

from powerenvelope.baselines import (
    RAMP_SEPARABLE_MIN_S,
    RAMP_WINDOW_S,
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    energy_per_useful_decision_from_cycle_mj,
    floor_occupancy_regression,
    floor_power,
    floor_variants,
    ground_truth,
    mean_active_power_mw,
    rail_accounting_curve,
    ramp_energy,
    run_order_rows,
    sleep_power_mw,
    two_term_cycle_energy_mj,
)
from powerenvelope.trace import FLOOR_CELL, TraceDataAbsentError

from fixtures import synthetic_campaign
from fixtures.synthetic_campaign import CLOSED_FORM_REL

# ------------------------------------------------------------------ baseline 1


def test_baseline1_measured_active_matches_the_closed_form(campaign, reference_cell):
    result = energy_per_inference(reference_cell, "measured-active")
    expected_active = synthetic_campaign.active_energy_mj(0.50, reference_cell.cycles)
    expected_power = expected_active / (reference_cell.cycles * 1.0)
    assert result.mean_active_power_mw == pytest.approx(
        expected_power, rel=CLOSED_FORM_REL, abs=0.0
    )
    assert result.mean_latency_s == pytest.approx(0.010, rel=1e-9, abs=0.0)
    assert result.energy_mj == pytest.approx(
        expected_power * 0.010, rel=CLOSED_FORM_REL, abs=0.0
    )


def test_baseline1_nominal_rail_is_the_voltage_ratio_below_measured(reference_cell):
    measured = energy_per_inference(reference_cell, "measured-active")
    nominal = energy_per_inference(reference_cell, "nominal-rail")
    assert measured.mean_active_power_mw / nominal.mean_active_power_mw == pytest.approx(
        synthetic_campaign.nominal_factor(), rel=1e-9, abs=0.0
    )


def test_baseline1_counts_inferences_from_the_file(reference_cell):
    result = energy_per_inference(reference_cell)
    assert result.n_inferences == synthetic_campaign.inferences(
        0.50, 64, reference_cell.cycles
    )


def test_baseline1_rejects_an_unknown_variant(reference_cell):
    with pytest.raises(ValueError, match="variant must be one of"):
        energy_per_inference(reference_cell, "datasheet")


def test_baseline1_refuses_the_floor_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="background floor"):
        energy_per_inference(campaign.require(FLOOR_CELL))


# ------------------------------------------------------------------ baseline 2


def test_floor_power_matches_the_closed_form(campaign):
    floor = floor_power(campaign)
    assert floor.mean_mw == pytest.approx(
        synthetic_campaign.floor_mw(), rel=CLOSED_FORM_REL, abs=0.0
    )
    assert floor.cell == FLOOR_CELL
    assert floor.sd_mw == pytest.approx(0.0, rel=0.0, abs=1e-9)


def test_floor_power_reports_the_foreign_occupancy_it_measured(campaign):
    # The synthetic monitor writes exactly SYNTHETIC_FOREIGN_BUSY for a second
    # with no inference in it, and the floor cell has no inferences at all.
    assert floor_power(campaign).foreign_busy_mean == pytest.approx(
        0.08, rel=1e-9, abs=0.0
    )


def test_floor_power_refuses_a_cell_that_ran_the_pipeline(campaign):
    with pytest.raises(TraceDataAbsentError, match="ran the pipeline"):
        floor_power(campaign, "w64_d050")


def test_ramp_is_separable_at_half_duty_and_positive(reference_cell):
    ramp = ramp_energy(reference_cell)
    assert ramp.separable
    assert ramp.n_wakes == reference_cell.cycles
    assert ramp.energy_mj > 0.0
    assert ramp.bandwidth_limited


def test_ramp_excess_is_near_the_declared_synthetic_excess(reference_cell):
    # The synthetic model holds the ramp excess for exactly SYNTHETIC_RAMP_S,
    # so the excess energy is close to excess * window. It is not equal to it:
    # the measured steady level is estimated from the samples after the window,
    # and the trapezoid straddles the step at the window's right edge.
    ramp = ramp_energy(reference_cell)
    ideal = synthetic_campaign.ramp_excess_mw() * RAMP_WINDOW_S
    assert ramp.energy_mj == pytest.approx(ideal, rel=0.15, abs=0.0)


def test_ramp_is_not_separable_at_the_lowest_duty(campaign):
    ramp = ramp_energy(campaign.require("w64_d010"))
    assert not ramp.separable
    assert ramp.energy_mj == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert ramp.n_wakes == 0


def test_ramp_separability_threshold_is_three_windows():
    assert RAMP_SEPARABLE_MIN_S == pytest.approx(3.0 * RAMP_WINDOW_S, rel=0.0, abs=0.0)


def test_ramp_rejects_a_non_positive_window(reference_cell):
    with pytest.raises(ValueError, match="must be positive"):
        ramp_energy(reference_cell, 0.0)


def test_duty_cycle_model_components_sum_to_the_cycle_energy(campaign):
    model = duty_cycle_model(campaign, "w64_d050")
    assert model.e_sleep_mj + model.e_ramp_mj + model.e_active_mj == pytest.approx(
        model.e_cycle_mj, rel=1e-12, abs=0.0
    )


def test_duty_cycle_model_uses_measured_floor_and_counted_rate(campaign):
    model = duty_cycle_model(campaign, "w64_d050")
    assert model.p_floor_mw == pytest.approx(
        floor_power(campaign).mean_mw, rel=0.0, abs=0.0
    )
    assert model.acceptance_rate == pytest.approx(
        synthetic_campaign.acceptance_rate(0.50, 64, 10), rel=1e-12, abs=0.0
    )
    assert model.n_inf_per_cycle == pytest.approx(100.0, rel=1e-12, abs=0.0)


def test_duty_cycle_model_is_within_ten_per_cent_of_ground_truth(campaign):
    # Both are measured quantities for the same cell; the analytic model drops
    # nothing except the within-active accounting, so it should be close. The
    # tolerance is declared here rather than fitted: ten per cent.
    model = duty_cycle_model(campaign, "w64_d050")
    truth = ground_truth(campaign.require("w64_d050"))
    ratio = model.energy_per_useful_decision_mj / truth.energy_per_useful_decision_mj
    assert ratio == pytest.approx(1.0, rel=0.10, abs=0.0)


def test_duty_cycle_model_refuses_the_floor_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="background floor"):
        duty_cycle_model(campaign, FLOOR_CELL)


def test_rail_accounting_is_a_fit_over_every_cell(campaign):
    rail = rail_accounting_curve(campaign)
    assert len(rail.cells) == 20
    assert len(rail.load_mw) == len(rail.ratio) == 20
    assert 0.0 < min(rail.ratio) < 1.0
    assert rail.r_squared <= 1.0


def test_rail_accounting_refuses_fewer_than_three_cells(tmp_path):
    small = synthetic_campaign.campaign(
        tmp_path / "small",
        plan=(("floor", 64, 0.0, 0.0, False), ("w64_d050", 64, 0.50, 42.5, True)),
    )
    with pytest.raises(ValueError, match="at least three"):
        rail_accounting_curve(small)


# ------------------------------------------------------------------ baseline 3


def test_ground_truth_matches_the_closed_form(campaign, reference_cell):
    truth = ground_truth(reference_cell)
    expected_energy = synthetic_campaign.total_energy_mj(0.50, reference_cell.cycles)
    expected_accepted = synthetic_campaign.accepted(0.50, 64, reference_cell.cycles)
    assert truth.energy_mj == pytest.approx(expected_energy, rel=CLOSED_FORM_REL, abs=0.0)
    assert truth.accepted == expected_accepted
    assert truth.energy_per_useful_decision_mj == pytest.approx(
        expected_energy / expected_accepted, rel=CLOSED_FORM_REL, abs=0.0
    )


def test_ground_truth_acceptance_rate_is_the_declared_rule(reference_cell):
    assert ground_truth(reference_cell).acceptance_rate == pytest.approx(
        synthetic_campaign.acceptance_rate(0.50, 64, reference_cell.cycles),
        rel=1e-12,
        abs=0.0,
    )


def test_ground_truth_reason_counts_sum_to_the_inferences(reference_cell):
    truth = ground_truth(reference_cell)
    assert sum(truth.reason_counts.values()) == truth.inferences


def test_between_run_spread_is_zero_on_identical_repeats(campaign):
    # The four repeats of the reference cell are byte-identical in the ideal
    # synthetic campaign, so the measured spread is exactly zero. That is the
    # correct answer for this input and it is asserted rather than worked round.
    spread = between_run_spread(campaign)
    assert spread.n == 4
    assert spread.sd_mj == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert spread.half_width_mj == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_between_run_spread_is_positive_once_the_runs_differ(jittered_campaign):
    spread = between_run_spread(jittered_campaign)
    assert spread.sd_mj > 0.0
    assert spread.half_width_mj > 0.0
    assert spread.low_mj < spread.mean_mj < spread.high_mj
    assert spread.range_mj > 0.0


def test_between_run_spread_t_critical_is_for_three_degrees_of_freedom(campaign):
    # Student t, 95 per cent two-sided, n = 4: 3.182446. Standard table value.
    assert between_run_spread(campaign).t_critical == pytest.approx(
        3.182446, rel=1e-6, abs=0.0
    )


def test_between_run_spread_refuses_a_single_run(campaign):
    with pytest.raises(ValueError, match="at least two repeats"):
        between_run_spread(campaign, ("w64_d050",))


def test_between_run_spread_refuses_a_bad_confidence(campaign):
    with pytest.raises(ValueError, match="confidence must be"):
        between_run_spread(campaign, confidence=1.5)


def test_permissive_control_accepts_far_more_than_the_strict_gate(campaign):
    # This is the negative control for the rejection half of the headline: at
    # the permissive tau the rejection factor should nearly vanish. The gate in
    # the synthetic generator is tau-dependent in the campaign's own order, so
    # the control has something to fire on.
    strict = ground_truth(campaign.require("w64_d050"))
    permissive = ground_truth(campaign.require("w64_d050_permissive"))
    assert permissive.acceptance_rate > 2.0 * strict.acceptance_rate
    assert 1.0 / permissive.acceptance_rate < 1.5


# ------------------------------------------- the measured-campaign machinery
#
# These exercise, on SYNTHETIC data, the helpers the measured reports need:
# the two-term cycle-energy identity, the floor-occupancy regression, the
# run-order table and the per-cell sleep power. The measured versions are in
# validation/validate_measured_baselines.py.


def test_ramp_is_charged_per_wake_and_not_per_cycle(campaign):
    # At duty 1.00 the active phases of consecutive cycles are contiguous, so
    # the run has ONE wake and not one per cycle. Charging the per-wake figure
    # to every cycle would over-count it by the cycle count. The measured data
    # found this through the two-term identity below.
    full = duty_cycle_model(campaign, "w64_d100")
    assert full.n_wakes == 1
    assert full.e_ramp_per_cycle_mj == pytest.approx(
        full.e_ramp_mj / campaign.require("w64_d100").cycles, rel=1e-12, abs=0.0
    )
    half = duty_cycle_model(campaign, "w64_d050")
    assert half.n_wakes == campaign.require("w64_d050").cycles
    assert half.e_ramp_per_cycle_mj == pytest.approx(half.e_ramp_mj, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    "label", ["w48_d010", "w48_d050", "w64_d050", "w64_d100", "w96_d075", "w96_d100"]
)
def test_two_term_and_three_term_cycle_energies_agree(campaign, label):
    # The two forms partition the same integral, so in exact arithmetic they
    # would be equal. They differ because the three-term form charges the ramp
    # per CYCLE while the two-term form carries it in a mean over the MEASURED
    # active duration, and the measured active duration is not exactly
    # ``cycles * duty * cycle_s``: the pipeline's loop overshoots or undershoots
    # the phase boundary by up to one inference. The residual is therefore that
    # overshoot times the ramp's share of the cycle energy -- measured at
    # -0.010 times the overshoot across the campaign -- and the largest value
    # observed is 3.02e-5 relative, on w96_d025. The tolerance is 1e-4.
    model = duty_cycle_model(campaign, label)
    assert model.e_cycle_mj == pytest.approx(
        model.two_term_e_cycle_mj, rel=1e-4, abs=0.0
    )
    assert abs(model.identity_residual) < 1e-4


def test_two_term_cycle_energy_known_answer():
    # duty 0.5, cycle 2.0 s: 1.0 s asleep at 4000 mW and 1.0 s active at
    # 5000 mW is 9000 mJ. Hand calculation.
    assert two_term_cycle_energy_mj(0.5, 2.0, 4000.0, 5000.0) == pytest.approx(
        9000.0, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    ("duty", "cycle_s", "match"),
    [(1.5, 2.0, "duty must be in"), (0.5, 0.0, "cycle_s must be positive")],
)
def test_two_term_cycle_energy_validates(duty, cycle_s, match):
    with pytest.raises(ValueError, match=match):
        two_term_cycle_energy_mj(duty, cycle_s, 1.0, 1.0)


def test_energy_per_useful_decision_from_cycle_uses_counted_numbers(campaign):
    cell = campaign.require("w64_d050")
    model = duty_cycle_model(campaign, "w64_d050")
    assert energy_per_useful_decision_from_cycle_mj(
        cell, model.e_cycle_mj
    ) == pytest.approx(model.energy_per_useful_decision_mj, rel=1e-12, abs=0.0)


def test_mean_active_power_is_the_time_weighted_active_mean(campaign):
    cell = campaign.require("w64_d050")
    assert mean_active_power_mw(cell) == pytest.approx(
        energy_per_inference(cell).mean_active_power_mw, rel=1e-12, abs=0.0
    )


def test_floor_regression_refuses_a_cell_that_ran_the_pipeline(campaign):
    with pytest.raises(TraceDataAbsentError, match="not foreign"):
        floor_occupancy_regression(campaign, "w64_d050")


def test_floor_regression_refuses_a_floor_with_no_occupancy_variation(campaign):
    # The synthetic floor run writes a constant occupancy, so its slope against
    # occupancy is not defined and the regression says so rather than returning
    # a meaningless number. On the measured campaign the occupancy does vary
    # and the regression is reported. The refusal tests on a computed standard
    # deviation with a relative threshold, not against an exact zero.
    with pytest.raises(ValueError, match="did not vary measurably"):
        floor_occupancy_regression(campaign)


def test_floor_variants_are_ordered_and_named(jittered_campaign):
    variants = floor_variants(jittered_campaign, include_regression=False)
    assert set(variants) == {"measured mean", "5th percentile"}
    assert variants["5th percentile"] <= variants["measured mean"]


def test_floor_variants_refuse_the_regression_row_when_it_is_undefined(campaign):
    with pytest.raises(ValueError, match="did not vary measurably"):
        floor_variants(campaign, include_regression=True)


def test_run_order_rows_are_in_execution_order(campaign):
    rows = run_order_rows(campaign)
    assert len(rows) == 20
    assert rows[0].elapsed_s == pytest.approx(0.0, rel=0.0, abs=1e-9)
    assert all(
        earlier.elapsed_s <= later.elapsed_s
        for earlier, later in zip(rows[:-1], rows[1:], strict=True)
    )
    assert {row.cell for row in rows} == set(campaign.cells)


def test_sleep_power_is_nan_at_full_duty_and_finite_otherwise(campaign):
    import math

    mean_full, min_full = sleep_power_mw(campaign.require("w64_d100"))
    assert math.isnan(mean_full) and math.isnan(min_full)
    mean_half, min_half = sleep_power_mw(campaign.require("w64_d050"))
    assert min_half <= mean_half
    assert mean_half < mean_active_power_mw(campaign.require("w64_d050"))
