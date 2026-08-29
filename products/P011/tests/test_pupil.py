"""Unit, known-answer and validation tests for the pupil grid."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from waveforge.pupil import Pupil


def test_pitch_and_area_hand_check():
    # D = 0.5 m over 64 samples -> dx = 0.5/64 = 7.8125 mm exactly.
    p = Pupil(0.5, 64)
    assert p.dx == pytest.approx(0.0078125, rel=0, abs=1e-15)
    # Clear area of an unobscured 0.5 m circle: pi/4 * 0.25 = 0.19634954 m^2.
    assert p.area == pytest.approx(0.19634954084936207, rel=1e-12)


def test_masked_sample_count_approaches_area_ratio():
    # The fraction of a square grid inside the inscribed circle -> pi/4 = 0.7854.
    p = Pupil(1.0, 512)
    assert p.n_valid / p.n_grid**2 == pytest.approx(np.pi / 4.0, abs=2e-3)


def test_obscuration_removes_the_right_area_fraction():
    # eps = 0.3 -> annulus area fraction 1 - 0.09 = 0.91 of the full disc.
    full = Pupil(1.0, 512)
    ann = Pupil(1.0, 512, obscuration=0.3)
    assert ann.n_valid / full.n_valid == pytest.approx(0.91, abs=5e-3)
    assert ann.area / full.area == pytest.approx(0.91, rel=1e-12)


def test_coords_are_centred_and_symmetric():
    p = Pupil(2.0, 32)
    x, y = p.coords()
    assert x.mean() == pytest.approx(0.0, abs=1e-15)
    assert y.mean() == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(x, -x[:, ::-1])
    assert np.allclose(y, -y[::-1, :])


def test_normalized_edge_radius_is_one():
    p = Pupil(4.0, 64)
    rho, _ = p.polar()
    # Largest normalised radius inside the mask must not exceed 1.
    assert rho[p.mask].max() <= 1.0
    assert rho[p.mask].max() > 0.98


def test_piston_removal_and_variance():
    p = Pupil(1.0, 32)
    phase = np.full((32, 32), 3.7)
    out = p.piston_removed(phase)
    assert np.allclose(out[p.mask], 0.0)
    assert p.variance(phase) == pytest.approx(0.0, abs=1e-24)
    # Known answer: a field equal to +1 on half the pupil and -1 on the other
    # half has variance 1 exactly.
    x, _ = p.coords()
    two_level = np.where(x >= 0, 1.0, -1.0)
    assert p.variance(two_level) == pytest.approx(1.0, rel=1e-12)


def test_variance_is_shift_invariant_and_scales_quadratically():
    p = Pupil(1.0, 48)
    rng = np.random.default_rng(0)
    phase = rng.standard_normal((48, 48))
    v = p.variance(phase)
    assert p.variance(phase + 5.0) == pytest.approx(v, rel=1e-12)
    assert p.variance(3.0 * phase) == pytest.approx(9.0 * v, rel=1e-12)


@given(
    diameter=st.floats(min_value=0.05, max_value=10.0),
    n=st.integers(min_value=8, max_value=64),
)
@settings(max_examples=25, deadline=None)
def test_area_matches_analytic_for_any_size(diameter, n):
    p = Pupil(diameter, n)
    assert p.area == pytest.approx(np.pi / 4.0 * diameter**2, rel=1e-12)
    assert p.dx == pytest.approx(diameter / n, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diameter": 0.0, "n_grid": 32},
        {"diameter": -1.0, "n_grid": 32},
        {"diameter": float("nan"), "n_grid": 32},
        {"diameter": 1.0, "n_grid": 7},
        {"diameter": 1.0, "n_grid": 32, "obscuration": -0.1},
        {"diameter": 1.0, "n_grid": 32, "obscuration": 0.95},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        Pupil(**kwargs)


def test_variance_rejects_wrong_shape():
    p = Pupil(1.0, 16)
    with pytest.raises(ValueError):
        p.variance(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        p.masked_mean(np.zeros((8, 8)))


def test_check_sampling_thresholds():
    p = Pupil(0.5, 64)  # dx = 7.8125 mm
    p.check_sampling(r0=0.10, actuator_pitch=0.0625)  # 12.8 and 8.0 samples
    with pytest.raises(ValueError, match="under-samples turbulence"):
        p.check_sampling(r0=0.01)  # 1.28 samples per r0
    with pytest.raises(ValueError, match="under-samples the DM"):
        p.check_sampling(actuator_pitch=0.02)  # 2.56 samples per actuator
    with pytest.raises(ValueError):
        p.check_sampling(r0=-1.0)
