"""Loop design, the pole identity, the three jitter routes and the slip estimate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slotsync.loop import (
    LoopDesign,
    cycle_slip_rate_rice,
    error_autocorrelation_weights,
    jitter_variance_closed_form,
    jitter_variance_coloured,
    jitter_variance_exact,
    loop_snr_db,
    noise_bandwidth_closed_form,
    noise_bandwidth_numeric,
    slip_free_seconds,
)

DAMPINGS = [0.3, 0.5, 1.0 / math.sqrt(2.0), 1.0, 2.0]
BANDWIDTHS = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]


def test_noise_bandwidth_hand_computed() -> None:
    """B_n = (theta / 2)(zeta + 1/(4 zeta)), hand-evaluated at two points.

    At ``zeta = 1/2`` the bracket is ``0.5 + 0.5 = 1``, so ``B_n = theta / 2``.
    At ``zeta = 1`` it is ``1 + 0.25 = 1.25``, so ``B_n = 0.625 theta``.
    """
    assert noise_bandwidth_closed_form(0.02, 0.5) == pytest.approx(0.01)
    assert noise_bandwidth_closed_form(0.02, 1.0) == pytest.approx(0.0125)


@pytest.mark.parametrize("damping", DAMPINGS)
@pytest.mark.parametrize("theta", [0.005, 0.05, 0.5])
def test_noise_bandwidth_matches_the_quadrature_integral(damping: float, theta: float) -> None:
    """The closed form against int_0^inf |H(j 2 pi f)|^2 df: an independent route."""
    numeric = noise_bandwidth_numeric(theta, damping)
    assert noise_bandwidth_closed_form(theta, damping) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("damping", DAMPINGS)
@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
def test_discrete_poles_sit_exactly_at_the_analogue_poles_mapped_through_the_exponential(
    damping: float, bandwidth: float
) -> None:
    """The coefficient mapping is exact pole placement; this checks the identity."""
    design = LoopDesign.from_bandwidth(bandwidth, damping, 1.75)
    theta = design.natural_frequency
    analogue = np.roots([1.0, 2.0 * damping * theta, theta**2])
    expected = np.sort_complex(np.exp(analogue))
    actual = np.sort_complex(design.closed_loop_poles)
    assert np.max(np.abs(actual - expected)) < 1e-7
    assert design.is_stable


@pytest.mark.parametrize("damping", DAMPINGS)
@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
def test_coefficient_mapping_round_trips(damping: float, bandwidth: float) -> None:
    design = LoopDesign.from_bandwidth(bandwidth, damping, 2.5)
    recovered = LoopDesign.from_coefficients(
        design.k_proportional, design.k_integral, design.detector_gain
    )
    assert recovered.noise_bandwidth == pytest.approx(bandwidth, rel=1e-9)
    assert recovered.damping == pytest.approx(damping, rel=1e-9)
    assert recovered.natural_frequency == pytest.approx(design.natural_frequency, rel=1e-9)


def test_coefficients_scale_inversely_with_the_detector_gain() -> None:
    """The closed loop must not depend on K_d; only the raw coefficients do."""
    low = LoopDesign.from_bandwidth(0.01, 0.7, 1.0)
    high = LoopDesign.from_bandwidth(0.01, 0.7, 4.0)
    assert high.k_proportional == pytest.approx(low.k_proportional / 4.0)
    assert high.k_integral == pytest.approx(low.k_integral / 4.0)
    assert np.allclose(
        np.sort_complex(low.closed_loop_poles), np.sort_complex(high.closed_loop_poles)
    )


def test_jitter_closed_form_hand_computed() -> None:
    # sigma^2 = 2 B_n sigma_n^2 / K_d^2 = 2 * 0.01 * 0.25 / 4 = 1.25e-3.
    assert jitter_variance_closed_form(0.01, 2.0, 0.25) == pytest.approx(1.25e-3)


@pytest.mark.parametrize("bandwidth", [0.0005, 0.001, 0.005])
def test_exact_discrete_jitter_approaches_the_closed_form_for_a_narrow_loop(
    bandwidth: float,
) -> None:
    """The small-bandwidth approximation, measured rather than asserted."""
    design = LoopDesign.from_bandwidth(bandwidth, 1.0 / math.sqrt(2.0), 1.5)
    exact = jitter_variance_exact(design, 0.3)
    closed = jitter_variance_closed_form(bandwidth, 1.5, 0.3)
    assert exact / closed == pytest.approx(1.0, rel=2.0 * bandwidth + 1e-3)


def test_exact_discrete_jitter_diverges_from_the_closed_form_for_a_wide_loop() -> None:
    """At B_n = 0.05 the approximation is already 4 to 6 per cent optimistic."""
    design = LoopDesign.from_bandwidth(0.05, 1.0 / math.sqrt(2.0), 1.5)
    ratio = jitter_variance_exact(design, 0.3) / jitter_variance_closed_form(0.05, 1.5, 0.3)
    assert 1.03 < ratio < 1.10


def test_jitter_scales_as_one_over_gain_squared_and_linearly_in_noise() -> None:
    base = LoopDesign.from_bandwidth(0.01, 0.7, 1.0)
    doubled = LoopDesign.from_bandwidth(0.01, 0.7, 2.0)
    assert jitter_variance_exact(doubled, 1.0) == pytest.approx(
        jitter_variance_exact(base, 1.0) / 4.0, rel=1e-9
    )
    assert jitter_variance_exact(base, 2.0) == pytest.approx(
        2.0 * jitter_variance_exact(base, 1.0), rel=1e-12
    )


def test_coloured_route_reduces_to_the_white_route_for_white_noise() -> None:
    design = LoopDesign.from_bandwidth(0.01, 0.7, 1.3)
    white = np.zeros(17)
    white[0] = 0.4
    assert jitter_variance_coloured(design, white) == pytest.approx(
        jitter_variance_exact(design, 0.4), rel=1e-12
    )


def test_coloured_route_reduces_the_jitter_for_negatively_correlated_noise() -> None:
    """Negative correlation at lag one is what the early-late gate actually has."""
    design = LoopDesign.from_bandwidth(0.01, 0.7, 1.3)
    autocovariance = np.zeros(17)
    autocovariance[0] = 1.0
    autocovariance[1] = -0.45
    coloured = jitter_variance_coloured(design, autocovariance)
    assert 0.0 < coloured < jitter_variance_exact(design, 1.0)


def test_autocorrelation_weights_start_at_the_unit_variance_jitter() -> None:
    design = LoopDesign.from_bandwidth(0.005, 0.7, 1.0)
    weights = error_autocorrelation_weights(design, 6)
    assert weights[0] == pytest.approx(jitter_variance_exact(design, 1.0), rel=1e-12)
    assert np.all(np.abs(weights[1:]) <= weights[0] * (1.0 + 1e-12))


def test_loop_snr_hand_computed() -> None:
    # boundary 0.5, sigma^2 = 0.0025 -> 0.25/0.0025 = 100 -> 20 dB.
    assert loop_snr_db(0.0025, 0.5) == pytest.approx(20.0)


def test_slip_rate_falls_exponentially_with_loop_snr() -> None:
    design = LoopDesign.from_bandwidth(0.02, 1.0 / math.sqrt(2.0), 1.5)
    rates = [cycle_slip_rate_rice(design, variance) for variance in (2.0, 1.0, 0.5, 0.25)]
    assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))
    # The rate carries exp(-a^2 / (2 sigma^2)), so each halving of the noise
    # variance costs a *larger* factor than the one before: the successive ratios
    # must grow, which is the signature of the exponential rather than a power law.
    assert rates[1] / rates[2] > rates[0] / rates[1]
    assert rates[2] / rates[3] > rates[1] / rates[2]


def test_slip_free_seconds_hand_computed() -> None:
    # 1e-6 slips per symbol at 1e9 symbols per second -> 1 / (1e-6 * 1e9) = 1e-3 s.
    assert slip_free_seconds(1e-6, 1e9) == pytest.approx(1e-3)
    assert slip_free_seconds(0.0, 1e9) == float("inf")


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=1e-4, max_value=0.08),
    st.floats(min_value=0.25, max_value=3.0),
    st.floats(min_value=0.2, max_value=8.0),
)
def test_design_is_always_stable_and_round_trips(
    bandwidth: float, damping: float, gain: float
) -> None:
    design = LoopDesign.from_bandwidth(bandwidth, damping, gain)
    assert design.is_stable
    recovered = LoopDesign.from_coefficients(
        design.k_proportional, design.k_integral, design.detector_gain
    )
    assert recovered.noise_bandwidth == pytest.approx(bandwidth, rel=1e-6)
    assert recovered.damping == pytest.approx(damping, rel=1e-6)


def test_design_validation() -> None:
    with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
        LoopDesign.from_bandwidth(0.6, 0.7, 1.0)
    with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
        LoopDesign.from_bandwidth(0.0, 0.7, 1.0)
    with pytest.raises(ValueError, match="damping must be positive"):
        LoopDesign.from_bandwidth(0.01, 0.0, 1.0)
    with pytest.raises(ValueError, match="carries no timing information"):
        LoopDesign.from_bandwidth(0.01, 0.7, 0.0)
    with pytest.raises(ValueError, match="both coefficients must be positive"):
        LoopDesign.from_coefficients(0.0, 0.1, 1.0)
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
        LoopDesign.from_coefficients(2.0, 0.1, 1.0)


def test_other_validation() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        noise_bandwidth_closed_form(0.0, 0.7)
    with pytest.raises(ValueError, match="damping must be positive"):
        noise_bandwidth_closed_form(0.01, -1.0)
    with pytest.raises(ValueError, match="non-negative"):
        jitter_variance_closed_form(0.01, 1.0, -1.0)
    with pytest.raises(ValueError, match="must be non-zero"):
        jitter_variance_closed_form(0.01, 0.0, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        loop_snr_db(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        loop_snr_db(0.1, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        error_autocorrelation_weights(LoopDesign.from_bandwidth(0.01, 0.7, 1.0), -1)
    with pytest.raises(ValueError, match="at least one entry"):
        jitter_variance_coloured(LoopDesign.from_bandwidth(0.01, 0.7, 1.0), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must be positive"):
        slip_free_seconds(1e-6, 0.0)
