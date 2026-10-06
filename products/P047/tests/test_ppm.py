"""M-ary PPM slot clock: the exact known answer, the loop, and what does not carry over."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slotsync.loop import LoopDesign, jitter_variance_closed_form, jitter_variance_coloured
from slotsync.ppm import (
    PpmConfig,
    duty_cycle_penalty_db,
    equivalent_slot_bandwidth,
    measure_ppm_slot_statistics,
    ppm_slot_autocovariance,
    ppm_slot_scurve,
    run_ppm_slot_loop,
    slot_index_error_rate,
)
from slotsync.pulses import half_sine, raised_cosine_time
from slotsync.scurve import default_offsets

ZETA = 1.0 / math.sqrt(2.0)
SLOT = half_sine(1.0)


def test_known_answer_ppm_slot_s_curve_is_exactly_a_sine() -> None:
    """S(eps) = sin(2 pi eps) and K_d = 2 pi, hand-computed.

    The slot pulse is the half sine ``p(t) = cos(pi t)`` on ``|t| <= 1/2``, which
    is zero at the slot edges, so no slot overlaps another and only the slot
    carrying the pulse contributes.  With gates at ``+-1/4`` slot,

        e(eps) = p(eps - 1/4)^2 - p(eps + 1/4)^2
               = cos^2(pi(eps - 1/4)) - cos^2(pi(eps + 1/4))
               = [cos(2 pi eps - pi/2) - cos(2 pi eps + pi/2)] / 2
               = [sin(2 pi eps) + sin(2 pi eps)] / 2
               = sin(2 pi eps)

    valid for ``|eps| <= 1/4``, where both gates remain inside the pulse.  Hence
    ``K_d = 2 pi`` exactly, the S-curve peaks at ``eps = 1/4`` and its useful
    range is a quarter of a slot.
    """
    offsets = np.linspace(-0.24, 0.24, 97)
    curve = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=offsets)
    assert np.max(np.abs(curve.values - np.sin(2.0 * np.pi * offsets))) < 1e-12
    fine = np.linspace(-0.001, 0.001, 21)
    sharp = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=fine, fit_halfwidth=0.001)
    assert sharp.gain_central_difference == pytest.approx(2.0 * math.pi, rel=1e-6)
    assert sharp.gain == pytest.approx(2.0 * math.pi, rel=1e-5)
    assert sharp.bias == pytest.approx(0.0, abs=1e-15)


def test_ppm_slot_s_curve_peaks_at_the_gate_spacing() -> None:
    curve = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=default_offsets(0.5, 201))
    assert curve.peak_offset == pytest.approx(0.25, abs=0.01)


def test_ppm_slot_s_curve_does_not_reverse_for_a_return_to_zero_slot() -> None:
    """A real difference from symbol timing: there is no adjacent stable lock point.

    With a slot pulse narrower than one slot the detector output decays towards
    zero as the pulse leaves the gates instead of crossing zero and reversing, so
    the half-slot boundary that bounds a symbol-timing cycle slip does not exist
    here.  That is why a PPM slot slip shows up as a wrong slot index rather than
    as a loop slip.
    """
    curve = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=default_offsets(0.5, 201))
    assert curve.reversal_offset is None


def test_slot_bandwidth_conversion_hand_computed() -> None:
    # One update per symbol, four slots per symbol: a loop with B_n = 0.01 per
    # update is 0.0025 cycles per slot.
    assert equivalent_slot_bandwidth(0.01, 4) == pytest.approx(0.0025)
    assert equivalent_slot_bandwidth(0.01, 16) == pytest.approx(0.000625)
    with pytest.raises(ValueError, match="at least 2"):
        equivalent_slot_bandwidth(0.01, 1)
    with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
        equivalent_slot_bandwidth(0.9, 4)


@pytest.mark.parametrize("order", [2, 4, 8])
def test_gain_does_not_depend_on_the_order_when_the_slot_is_known(order: int) -> None:
    """Only the on-slot contributes, so the gain is an order-independent slot property."""
    fine = np.linspace(-0.001, 0.001, 21)
    curve = ppm_slot_scurve(PpmConfig(order, 0.25, True), SLOT, offsets=fine, fit_halfwidth=0.001)
    assert curve.gain_central_difference == pytest.approx(2.0 * math.pi, rel=1e-6)


def test_open_loop_statistics_and_slot_errors() -> None:
    clean = measure_ppm_slot_statistics(
        PpmConfig(4, 0.25, False), SLOT, sample_snr_db=20.0, symbols=4000
    )
    noisy = measure_ppm_slot_statistics(
        PpmConfig(4, 0.25, False), SLOT, sample_snr_db=-4.0, symbols=4000
    )
    assert clean["slot_error_rate"] == 0.0
    assert noisy["slot_error_rate"] > 0.05
    assert noisy["variance"] > clean["variance"]
    assert clean["self_noise_variance"] == pytest.approx(0.0, abs=1e-12)


def test_detector_noise_variance_grows_with_the_order_at_fixed_sample_snr() -> None:
    """The duty-cycle penalty: more slots, more chances for noise to win a slot."""
    rates = [
        measure_ppm_slot_statistics(
            PpmConfig(order, 0.25, False), SLOT, sample_snr_db=2.0, symbols=4000
        )["slot_error_rate"]
        for order in (2, 4, 8)
    ]
    assert rates[0] < rates[1] < rates[2]


def test_duty_cycle_penalty_is_a_finite_number_in_decibels() -> None:
    penalty = duty_cycle_penalty_db(
        PpmConfig(4, 0.25, False),
        SLOT,
        sample_snr_db=10.0,
        reference_gain=2.0 * math.pi,
        symbols=4000,
    )
    assert math.isfinite(penalty)


def test_autocovariance_of_the_slot_detector_is_nearly_white() -> None:
    autocovariance = ppm_slot_autocovariance(
        PpmConfig(4, 0.25, False), SLOT, sample_snr_db=20.0, max_lag=6, symbols=8000
    )
    assert np.max(np.abs(autocovariance[1:] / autocovariance[0])) < 0.05


def test_closed_loop_acquires_and_matches_the_prediction() -> None:
    config = PpmConfig(4, 0.25, False)
    fine = np.linspace(-0.001, 0.001, 21)
    gain = ppm_slot_scurve(config, SLOT, offsets=fine, fit_halfwidth=0.001).gain_central_difference
    autocovariance = ppm_slot_autocovariance(
        config, SLOT, sample_snr_db=20.0, max_lag=6, symbols=12000
    )
    design = LoopDesign.from_bandwidth(0.004, ZETA, gain)
    run = run_ppm_slot_loop(
        config, SLOT, design, n_symbols=30000, sample_snr_db=20.0, true_offset=0.05
    )
    predicted = jitter_variance_coloured(design, autocovariance)
    white = jitter_variance_closed_form(0.004, gain, autocovariance[0])
    assert not run.diverged
    assert run.slip_count == 0
    # The measured-to-predicted ratio for the PPM slot loop sits in 0.7 to 1.15
    # over B_n from 0.002 to 0.02; the band is asserted rather than a tight
    # tolerance, and ``validation/validate_ppm_slot.py`` reports the sweep that
    # establishes it.  The squared-gate detector is quadratic in the noise, so its
    # output variance depends on the timing error itself and the linearised
    # prediction is not expected to be exact.
    assert 0.65 < run.jitter_variance / predicted < 1.2
    assert 0.65 < run.jitter_variance / white < 1.2
    assert slot_index_error_rate(run) == 0.0
    summary = run.ppm_summary()
    assert summary["order"] == 4
    assert summary["B_slot"] == pytest.approx(0.001)


def test_slot_jitter_creates_a_slot_error_floor_at_low_snr() -> None:
    config = PpmConfig(8, 0.25, False)
    fine = np.linspace(-0.001, 0.001, 21)
    gain = ppm_slot_scurve(config, SLOT, offsets=fine, fit_halfwidth=0.001).gain_central_difference
    design = LoopDesign.from_bandwidth(0.01, ZETA, gain)
    run = run_ppm_slot_loop(config, SLOT, design, n_symbols=6000, sample_snr_db=-2.0)
    assert run.slot_error_rate > 0.0


def test_known_slot_mode_runs_and_differs_from_max_energy_at_low_snr() -> None:
    fine = np.linspace(-0.001, 0.001, 21)
    gain = ppm_slot_scurve(
        PpmConfig(4, 0.25, True), SLOT, offsets=fine, fit_halfwidth=0.001
    ).gain_central_difference
    design = LoopDesign.from_bandwidth(0.01, ZETA, gain)
    known = run_ppm_slot_loop(
        PpmConfig(4, 0.25, True), SLOT, design, n_symbols=6000, sample_snr_db=-2.0, seed=3
    )
    blind = run_ppm_slot_loop(
        PpmConfig(4, 0.25, False), SLOT, design, n_symbols=6000, sample_snr_db=-2.0, seed=3
    )
    assert known.jitter_variance != pytest.approx(blind.jitter_variance, rel=1e-6)


def test_configuration_and_argument_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 slots"):
        PpmConfig(1)
    with pytest.raises(ValueError, match=r"\(0, 0.5\]"):
        PpmConfig(4, 0.75)
    with pytest.raises(ValueError, match="reaches beyond"):
        ppm_slot_scurve(PpmConfig(2, 0.25, True), raised_cosine_time(4.0), neighbours=1)
    with pytest.raises(ValueError, match="non-negative"):
        ppm_slot_scurve(PpmConfig(4), SLOT, neighbours=-1)
    design = LoopDesign.from_bandwidth(0.01, ZETA, 6.28)
    with pytest.raises(ValueError, match="at least 2000"):
        run_ppm_slot_loop(PpmConfig(4), SLOT, design, n_symbols=10)
    with pytest.raises(ValueError, match="at least 1000"):
        measure_ppm_slot_statistics(PpmConfig(4), SLOT, sample_snr_db=10.0, symbols=10)


def test_config_label_mentions_the_slot_decision_mode() -> None:
    assert "known slot" in PpmConfig(4, 0.25, True).label
    assert "max-energy" in PpmConfig(4, 0.25, False).label
