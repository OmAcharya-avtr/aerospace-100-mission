"""The headline factor, and that its factors account for it rather than claim to."""

from __future__ import annotations

import numpy as np
import pytest

from powerenvelope.baselines import energy_per_inference, ground_truth
from powerenvelope.decomposition import decompose
from powerenvelope.trace import FLOOR_CELL, TraceDataAbsentError

from fixtures import synthetic_campaign


def test_factor_is_the_ratio_of_the_two_measured_quantities(reference_cell):
    result = decompose(reference_cell)
    truth = ground_truth(reference_cell)
    baseline = energy_per_inference(reference_cell)
    assert result.factor == pytest.approx(
        truth.energy_per_useful_decision_mj / baseline.energy_mj, rel=1e-14, abs=0.0
    )


def test_three_named_factors_give_the_factor_exactly(campaign):
    # The decomposition has no residual term. E_use = E_total/A and
    # E_inf = E_active/N are shares of the same two integrals, so the three
    # named factors multiply to F identically; what is left is rounding.
    for label in ("w48_d010", "w64_d050", "w96_d075", "w64_d050_permissive"):
        for variant in ("measured-active", "nominal-rail"):
            result = decompose(campaign.require(label), variant=variant)
            assert result.product_of_named_factors() == pytest.approx(
                result.factor, rel=1e-12, abs=0.0
            )
            assert abs(result.closure_error) < 1e-12


def test_the_declared_phases_partition_the_run(campaign):
    # E_active and E_sleep tile the run's cycles exactly; the run may extend a
    # few milliseconds past the last cycle and that tail is the whole residual.
    for label in ("w48_d025", "w64_d050", "w96_d100"):
        result = decompose(campaign.require(label))
        assert result.active_energy_mj + result.sleep_energy_mj == pytest.approx(
            result.total_energy_mj, rel=1e-3, abs=0.0
        )
        assert abs(result.partition_residual) < 1e-3


def test_phase_occupancy_is_a_diagnostic_and_is_one_on_synthetic_data(reference_cell):
    # The synthetic pipeline's inferences tile each declared active phase
    # exactly, so the occupancy is 1 and the shortfall is zero. On the measured
    # campaign it is 0.9997 with a 10.5 ms shortfall. Either way it is a
    # DIAGNOSTIC and not a factor of the decomposition.
    occupancy = decompose(reference_cell).occupancy
    assert occupancy.occupancy == pytest.approx(1.0, rel=1e-9, abs=0.0)
    assert occupancy.shortfall_s == pytest.approx(0.0, rel=0.0, abs=1e-9)


def test_rejection_factor_is_the_reciprocal_of_the_counted_rate(reference_cell):
    result = decompose(reference_cell)
    assert result.rejection_factor == pytest.approx(
        1.0 / synthetic_campaign.acceptance_rate(0.50, 64, reference_cell.cycles),
        rel=1e-12,
        abs=0.0,
    )


def test_rail_voltage_factor_is_exactly_one_for_the_measured_variant(reference_cell):
    assert decompose(reference_cell, "measured-active").rail_voltage_factor == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_rail_voltage_factor_is_the_voltage_ratio_for_the_nominal_variant(reference_cell):
    assert decompose(reference_cell, "nominal-rail").rail_voltage_factor == pytest.approx(
        synthetic_campaign.nominal_factor(), rel=1e-9, abs=0.0
    )


def test_duty_cycle_factor_is_one_at_full_duty(campaign):
    # At duty 1.0 there is no sleep window, so the whole-run energy and the
    # active energy are the same integral.
    assert decompose(campaign.require("w64_d100")).duty_cycle_factor == pytest.approx(
        1.0, rel=1e-12, abs=0.0
    )


def test_duty_cycle_factor_rises_as_the_duty_falls(campaign):
    factors = [
        decompose(campaign.require(f"w64_d{int(duty * 100):03d}")).duty_cycle_factor
        for duty in (0.10, 0.25, 0.50, 0.75, 1.00)
    ]
    assert all(
        earlier > later
        for earlier, later in zip(factors[:-1], factors[1:], strict=True)
    )


def test_headline_factor_at_half_duty_is_the_product_of_two_counted_ratios(
    campaign, reference_cell
):
    # F = (N/A) * (E_total/E_active) with both energies over the run. The
    # counts come from the closed-form gate rule and the energies from the
    # product's own integrator, so this checks the ASSEMBLY of F rather than
    # the integrals, which validate_baselines_synthetic.py checks separately.
    cycles = reference_cell.cycles
    accepted = synthetic_campaign.accepted(0.50, 64, cycles)
    inferences = synthetic_campaign.inferences(0.50, 64, cycles)
    result = decompose(reference_cell)
    assert result.n_inferences == inferences
    assert result.n_accepted == accepted
    assert result.factor == pytest.approx(
        (inferences / accepted) * (result.total_energy_mj / result.active_energy_mj),
        rel=1e-12,
        abs=0.0,
    )


def test_permissive_control_reduces_the_rejection_factor_and_the_headline(campaign):
    strict = decompose(campaign.require("w64_d050"))
    permissive = decompose(campaign.require("w64_d050_permissive"))
    assert permissive.rejection_factor < 0.5 * strict.rejection_factor
    assert permissive.factor < strict.factor
    # The duty-cycle factor is a property of the duty, not of the gate, so it
    # must be essentially unchanged by the control.
    assert permissive.duty_cycle_factor == pytest.approx(
        strict.duty_cycle_factor, rel=0.02, abs=0.0
    )


def test_nominal_variant_gives_a_larger_factor_than_the_measured_variant(reference_cell):
    # The nominal 5.0 V is below the synthetic bus voltage, so the quoted power
    # is too small and the factor it understates by is correspondingly larger.
    measured = decompose(reference_cell, "measured-active")
    nominal = decompose(reference_cell, "nominal-rail")
    assert nominal.factor > measured.factor


def test_lines_render_every_factor(reference_cell):
    lines = decompose(reference_cell).lines()
    text = "\n".join(lines)
    for fragment in (
        "HEADLINE understatement factor F",
        "rejection factor N/A",
        "duty-cycle factor",
        "rail-voltage factor",
        "closure error of the three",
        "DIAGNOSTIC, not a factor",
        "occupied by inferences",
    ):
        assert fragment in text


def test_decompose_rejects_an_unknown_variant(reference_cell):
    with pytest.raises(ValueError, match="variant must be one of"):
        decompose(reference_cell, "handwave")


def test_decompose_refuses_the_floor_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="background floor"):
        decompose(campaign.require(FLOOR_CELL))


def test_every_grid_cell_has_a_finite_factor(campaign):
    for cell in campaign.grid_cells():
        result = decompose(cell)
        assert np.isfinite(result.factor)
        assert result.factor > 0.0
