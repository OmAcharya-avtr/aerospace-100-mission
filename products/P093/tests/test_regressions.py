"""Pinned counterexamples and pinned numbers.

Two kinds of test live here.

**Pinned counterexamples.** ``.hypothesis/`` is gitignored, so a falsifying
example found by one run is lost on a fresh clone. Every counterexample
Hypothesis has found in this product is written below as a named test with the
literal inputs in the body.

**Pinned numbers.** The headline figures of the SYNTHETIC campaign, so that a
change to the integrator, the window derivation or the decomposition cannot pass
unnoticed. These are synthetic figures and they are not measurements of
anything; the measured figures live in ``validation/`` and move only when the
campaign is re-run.
"""

from __future__ import annotations

import numpy as np
import pytest

from powerenvelope.bounds import EPS
from powerenvelope.decomposition import decompose
from powerenvelope.energy import trapezoid_energy_mj
from powerenvelope.windows import extract_windows

from fixtures import synthetic_campaign

# --------------------------------------------------- pinned counterexamples


def test_time_shift_counterexample_t_zero_to_1e_minus_4_shift_2048():
    """Hypothesis, first run of tests/test_properties.py, 2026-11-14.

    ``t = [0.0, 1e-4]``, ``power = [1.0, 1.0]``, ``shift = 2048.0``. The naive
    claim that the trapezoidal integral is invariant under a shift of the
    timestamps is FALSE here: the shifted spacing is
    ``1.0000000020227162e-4`` rather than ``1e-4``, a relative change of
    2.0e-9, because 2048.0001 is not representable. The claim this product makes
    instead is that the change is bounded by ``4 * EPS * max|t| / min(dt)``, and
    that bound is asserted below on exactly these literals.
    """
    t = np.asarray([0.0, 1.0e-4])
    power = np.asarray([1.0, 1.0])
    shift = 2048.0
    unshifted = trapezoid_energy_mj(t, power)
    shifted = trapezoid_energy_mj(t + shift, power)
    assert unshifted == pytest.approx(1.0e-4, rel=1e-15, abs=0.0)
    # The disagreement is real and is not zero.
    assert shifted != unshifted
    observed = abs(shifted / unshifted - 1.0)
    assert observed == pytest.approx(2.0227e-9, rel=1e-3, abs=0.0)
    bound = 4.0 * EPS * float(np.max(np.abs(t + shift))) / float(np.min(np.diff(t)))
    assert observed < bound
    # And the naive 1e-9 relative claim would NOT hold, which is why it is gone.
    assert observed > 1e-9


# ------------------------------------------------------------ pinned numbers


def test_pinned_synthetic_reference_cell_counts(reference_cell):
    truth = reference_cell.require_decisions()
    assert truth.inferences == 1000
    assert truth.accepted_count == 307
    assert truth.reason_counts() == {
        "accept": 307,
        "area_out": 40,
        "border": 56,
        "low_score": 557,
        "no_candidate": 40,
    }


def test_pinned_synthetic_reference_cell_energies(reference_cell):
    result = decompose(reference_cell)
    assert result.total_energy_mj == pytest.approx(44556.6336, rel=1e-9, abs=0.0)
    assert result.active_energy_mj == pytest.approx(25338.24, rel=1e-9, abs=0.0)
    assert result.active_duration_s == pytest.approx(10.0, rel=1e-12, abs=0.0)
    assert result.energy_per_inference_mj == pytest.approx(25.33824, rel=1e-9, abs=0.0)
    assert result.energy_per_useful_decision_mj == pytest.approx(
        145.13561433, rel=1e-8, abs=0.0
    )


def test_pinned_synthetic_headline_factor_and_decomposition(reference_cell):
    result = decompose(reference_cell)
    assert result.factor == pytest.approx(5.727927998641027, rel=1e-9, abs=0.0)
    assert result.rejection_factor == pytest.approx(3.25732899, rel=1e-8, abs=0.0)
    assert result.duty_cycle_factor == pytest.approx(
        1.7584738955827954, rel=1e-9, abs=0.0
    )
    assert result.rail_voltage_factor == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert result.accounting_factor == pytest.approx(1.0, rel=1e-9, abs=0.0)
    assert abs(result.closure_error) < 1e-12


def test_pinned_synthetic_nominal_variant(reference_cell):
    result = decompose(reference_cell, "nominal-rail")
    assert result.rail_voltage_factor == pytest.approx(1.0176, rel=1e-9, abs=0.0)
    assert result.factor == pytest.approx(5.828739531417111, rel=1e-9, abs=0.0)


def test_pinned_synthetic_permissive_control(campaign):
    strict = decompose(campaign.require("w64_d050"))
    permissive = decompose(campaign.require("w64_d050_permissive"))
    assert permissive.acceptance_rate == pytest.approx(0.829, rel=1e-9, abs=0.0)
    assert permissive.rejection_factor == pytest.approx(1.206272, rel=1e-6, abs=0.0)
    assert strict.rejection_factor / permissive.rejection_factor == pytest.approx(
        2.7003257328990227, rel=1e-9, abs=0.0
    )


def test_pinned_synthetic_window_dataset(campaign):
    windows = extract_windows(campaign)
    assert len(windows) == 38
    assert windows.foreign_imputed == 12
    assert windows.floor_foreign_busy == pytest.approx(0.08, rel=1e-9, abs=0.0)


def test_pinned_synthetic_closed_form_constants():
    assert synthetic_campaign.floor_mw() == pytest.approx(1913.088, rel=1e-12, abs=0.0)
    assert synthetic_campaign.active_excess_mw() == pytest.approx(
        610.56, rel=1e-12, abs=0.0
    )
    assert synthetic_campaign.ramp_excess_mw() == pytest.approx(203.52, rel=1e-12, abs=0.0)
