"""S-curves and the detector gain, including the hand-computable known answers.

The three known-answer families here are the foundation of the whole package:
the loop analysis depends on ``K_d``, so ``K_d`` has to be right.  Each one is
derived in the test's own comment and the derivations are repeated in
``docs/TIMING_MODEL.md`` section 2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slotsync.pulses import (
    half_sine,
    nyquist_raised_cosine,
    raised_cosine_time,
    rectangular,
    triangular,
)
from slotsync.scurve import default_offsets, scurve
from slotsync.ted import TedConfig

FINE = np.linspace(-0.02, 0.02, 41)


def test_known_answer_early_late_on_a_triangle_is_exactly_two() -> None:
    """K_d = 2 exactly, hand-computed.

    Triangular pulse ``p(t) = 1 - |t|`` (half-width one symbol), decision-directed
    early-late gate at half-spacing ``d``, isolated-symbol algebra:

        e(eps) = p(eps - d) - p(eps + d)
               = [1 - (d - eps)] - [1 - (d + eps)]        for |eps| < d
               = 2 eps

    so ``dS/d eps = 2`` for every ``d`` in ``(0, 1/2]`` as long as ``|eps| < d``.
    The neighbouring symbols contribute equal and opposite amounts to the two
    gates for a triangle, so the ensemble average does not change the slope - and
    the measurement below is the full ensemble average over all 32 data patterns,
    not the isolated-symbol case, which is why it is a real check.
    """
    for delta in (0.1, 0.25, 0.4, 0.5):
        curve = scurve(TedConfig("early-late", "antipodal", delta, "dd"), triangular(1.0), FINE)
        assert curve.exact
        assert curve.gain == pytest.approx(2.0, abs=1e-12), delta
        assert curve.gain_central_difference == pytest.approx(2.0, abs=1e-12), delta
        assert curve.bias == pytest.approx(0.0, abs=1e-14)


def test_known_answer_mueller_muller_on_a_triangle_is_the_symbol_variance() -> None:
    """K_d = E[a^2], hand-computed.

    For zero-mean independent data of variance ``s2`` and a pulse ``h``, the
    detector ``e[k] = a[k] x[k-1] - a[k-1] x[k]`` has mean

        S(eps) = s2 * (h(eps - 1) - h(eps + 1)).

    For the unit-peak triangle of half-width one symbol and ``0 < eps < 1``,
    ``h(eps - 1) = eps`` and ``h(eps + 1) = 0``, so ``S(eps) = s2 * eps`` and
    ``K_d = s2``: exactly 1 for antipodal data.

    For unipolar OOK the data has a non-zero mean and the cross terms do not
    vanish; carrying them through gives ``S(eps) = (1/4)(h(eps-1) - h(eps+1))``,
    i.e. ``K_d = 1/4`` - a factor of four less gain than antipodal data, with no
    lock-point bias.  Both are checked.
    """
    antipodal = scurve(TedConfig("mueller-muller", "antipodal"), triangular(1.0), FINE)
    assert antipodal.gain == pytest.approx(1.0, abs=1e-12)
    assert antipodal.gain_central_difference == pytest.approx(1.0, abs=1e-12)
    unipolar = scurve(TedConfig("mueller-muller", "ook"), triangular(1.0), FINE)
    assert unipolar.gain == pytest.approx(0.25, abs=1e-12)
    assert unipolar.bias == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("rolloff", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_known_answer_mueller_muller_on_a_nyquist_raised_cosine(rolloff: float) -> None:
    """K_d = 2 cos(pi a) / (1 - 4 a^2), hand-computed, with the a = 1/2 limit pi/2.

    The Nyquist raised cosine is ``h(t) = sinc(t) cos(pi a t) / (1 - (2 a t)^2)``.
    From the Mueller-Mueller mean ``S(eps) = h(eps - 1) - h(eps + 1)`` and the
    evenness of ``h``,

        K_d = S'(0) = h'(-1) - h'(1) = -2 h'(1).

    At ``t = 1`` the factor ``sinc(1) = 0``, so only the term differentiating the
    sinc survives: ``h'(1) = sinc'(1) cos(pi a) / (1 - 4 a^2)`` and
    ``sinc'(1) = d/dt [sin(pi t)/(pi t)]|_{t=1} = -1``.  Hence

        K_d = 2 cos(pi a) / (1 - 4 a^2),

    which is exactly 2 for ``a = 0`` and, by l'Hopital at the removable
    singularity ``a = 1/2``, exactly ``pi / 2``.  The measured values agree to
    better than 1e-05 relative; the residual is the central-difference truncation
    plus the pulse's truncation to four symbols.
    """
    if abs(1.0 - 4.0 * rolloff**2) < 1e-12:
        expected = math.pi / 2.0
    else:
        expected = 2.0 * math.cos(math.pi * rolloff) / (1.0 - 4.0 * rolloff**2)
    curve = scurve(
        TedConfig("mueller-muller", "antipodal"),
        nyquist_raised_cosine(rolloff, 4.0),
        FINE,
        max_exact_symbols=16,
    )
    assert curve.exact
    assert curve.gain_central_difference == pytest.approx(expected, rel=1e-5)


def test_early_late_gain_collapses_to_zero_on_a_rectangle() -> None:
    """A rectangle is flat inside the pulse, so a difference detector sees nothing."""
    curve = scurve(TedConfig("early-late", "antipodal", 0.25, "dd"), rectangular(1.0), FINE)
    assert curve.gain == 0.0
    assert curve.linear_halfwidth == 0.0


def test_mueller_muller_gain_is_zero_without_inter_symbol_interference() -> None:
    """The detector is driven by ``h(+-1)``; a pulse narrower than a symbol has none."""
    curve = scurve(TedConfig("mueller-muller", "antipodal"), half_sine(1.0), FINE)
    assert curve.gain == pytest.approx(0.0, abs=1e-14)


def test_mueller_muller_is_ill_conditioned_on_a_flat_topped_raised_cosine() -> None:
    """A documented failure: the two gain estimates disagree by more than 2x.

    ``raised_cosine_time`` has zero slope at ``t = +-1``, so the Mueller-Mueller
    gain is third order in the offset rather than first.  The least-squares and
    central-difference estimates then measure different things, which is the
    package's signal that the gain is not usable.
    """
    curve = scurve(TedConfig("mueller-muller", "antipodal"), raised_cosine_time(1.0))
    assert abs(curve.gain) < 0.2
    assert abs(curve.gain / max(abs(curve.gain_central_difference), 1e-12)) > 2.0


def test_gardner_on_a_discontinuous_pulse_is_ill_conditioned() -> None:
    """Also documented: a rectangle makes Gardner's S-curve a step at the origin.

    The two gain estimates then differ by more than a factor of five, which is how
    a user finds out that the number is meaningless rather than large.
    """
    curve = scurve(TedConfig("gardner", "antipodal"), rectangular(1.0))
    ratio = curve.gain_central_difference / curve.gain
    assert ratio > 5.0


def test_sign_convention_is_positive_gain_for_all_three_detectors() -> None:
    pulse = nyquist_raised_cosine(0.5, 4.0)
    for config in (
        TedConfig("early-late", "antipodal", 0.25, "dd"),
        TedConfig("gardner", "antipodal"),
        TedConfig("mueller-muller", "antipodal"),
    ):
        assert scurve(config, pulse, FINE, max_exact_symbols=16).gain > 0.0, config.label


def test_odd_symmetry_of_the_s_curve() -> None:
    curve = scurve(TedConfig("gardner", "antipodal"), nyquist_raised_cosine(0.5, 4.0))
    assert np.allclose(curve.values, -curve.values[::-1], atol=1e-12)


def test_linear_range_peak_and_reversal_are_ordered() -> None:
    curve = scurve(TedConfig("early-late", "antipodal", 0.25, "square"), triangular(1.0))
    reversal = curve.reversal_offset
    assert reversal is not None
    # The 10 % linearity tolerance is wide enough to reach a little past the peak
    # of this particular S-curve (0.260 against a peak at 0.250), so the ordering
    # that holds in general is linear range < reversal and peak < reversal, not
    # linear range <= peak.
    assert 0.0 < curve.linear_halfwidth < reversal
    assert 0.0 < curve.peak_offset < reversal
    # The triangular early-late S-curve is one-symbol periodic, so the reversal
    # sits at exactly half a symbol.
    assert reversal == pytest.approx(0.5, abs=1e-9)


def test_self_noise_is_reported_and_is_zero_for_mueller_muller_on_a_nyquist_pulse() -> None:
    nyquist = nyquist_raised_cosine(0.5, 4.0)
    mm = scurve(TedConfig("mueller-muller", "antipodal"), nyquist, FINE, max_exact_symbols=16)
    assert mm.self_noise_at_origin < 1e-6
    el = scurve(
        TedConfig("early-late", "antipodal", 0.25, "dd"), nyquist, FINE, max_exact_symbols=16
    )
    assert el.self_noise_at_origin > 0.1


def test_monte_carlo_fallback_is_flagged_and_close_to_the_exact_answer() -> None:
    pulse = nyquist_raised_cosine(0.5, 4.0)
    exact = scurve(TedConfig("mueller-muller", "antipodal"), pulse, FINE, max_exact_symbols=16)
    approximate = scurve(
        TedConfig("mueller-muller", "antipodal"), pulse, FINE, max_exact_symbols=4, trials=40000
    )
    assert exact.exact and not approximate.exact
    assert approximate.gain == pytest.approx(exact.gain, rel=0.05)


def test_offsets_validation() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        scurve(TedConfig("gardner"), triangular(1.0), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="odd and at least 11"):
        default_offsets(0.5, 10)
    with pytest.raises(ValueError, match=r"\(0.05, 2.0\]"):
        default_offsets(3.0, 101)


def test_default_offsets_contain_the_origin_exactly() -> None:
    offsets = default_offsets()
    assert 0.0 in offsets
    assert offsets[0] == -offsets[-1]
