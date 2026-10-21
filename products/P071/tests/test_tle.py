"""The optional TLE entry point, checked against the published SGP4 vectors.

The reference values below are read from ``tcppver.out``, the official SGP4
verification output that ships inside the installed ``sgp4`` package, for
satellite 00005 of ``SGP4-VER.TLE``. They are quoted here as literals so that
the test states what it is checking, and they are also re-read from the package
in ``validation/validate_tle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.tle import (
    ClosestApproach,
    find_closest_approach,
    state_from_tle,
    synthetic_secondary,
)

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

# tcppver.out, satellite 5, t = 0 min:
#   7022.46529266  -1400.08296755  0.03995155  1.893841015  6.405893759  4.534807250
EPOCH_POSITION_KM = (7022.46529266, -1400.08296755, 0.03995155)
EPOCH_VELOCITY_KMPS = (1.893841015, 6.405893759, 4.534807250)
# tcppver.out, satellite 5, t = 360 min:
#  -7154.03120202  -3783.17682504  -3536.19412294  4.741887409 -4.151817765 -2.093935425
LATE_POSITION_KM = (-7154.03120202, -3783.17682504, -3536.19412294)


def test_epoch_state_matches_the_published_vector():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    np.testing.assert_allclose(
        position / 1000.0, EPOCH_POSITION_KM, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        velocity / 1000.0, EPOCH_VELOCITY_KMPS, rtol=0.0, atol=1e-8
    )


def test_propagated_state_matches_the_published_vector():
    position, _ = state_from_tle(LINE1, LINE2, 360.0)
    np.testing.assert_allclose(position / 1000.0, LATE_POSITION_KM, rtol=0.0, atol=1e-7)


def test_backward_propagation_returns_to_the_epoch_state():
    forward, _ = state_from_tle(LINE1, LINE2, 100.0)
    back, _ = state_from_tle(LINE1, LINE2, -100.0)
    assert not np.allclose(forward, back)
    at_epoch, _ = state_from_tle(LINE1, LINE2, 0.0)
    np.testing.assert_allclose(at_epoch / 1000.0, EPOCH_POSITION_KM, rtol=0.0, atol=1e-7)


def test_bad_tle_raises():
    with pytest.raises(Exception):  # noqa: B017 - sgp4 raises its own parse error type
        state_from_tle("not a tle", "nor is this", 0.0)


def test_synthetic_secondary_offset_magnitude_is_as_declared():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    offset = (30.0, 110.0, 40.0)
    secondary_position, _ = synthetic_secondary(
        position, velocity, offset_rtn_m=offset, velocity_rotation_rad=0.0
    )
    assert float(np.linalg.norm(secondary_position - position)) == pytest.approx(
        float(np.linalg.norm(offset)), rel=1e-9
    )


def test_synthetic_secondary_preserves_the_speed_under_rotation():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    _, rotated = synthetic_secondary(
        position, velocity, offset_rtn_m=(0.0, 0.0, 0.0), velocity_rotation_rad=0.7
    )
    assert float(np.linalg.norm(rotated)) == pytest.approx(
        float(np.linalg.norm(velocity)), rel=1e-12
    )


def test_synthetic_secondary_zero_rotation_is_the_identity():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    _, unchanged = synthetic_secondary(
        position, velocity, offset_rtn_m=(0.0, 0.0, 0.0), velocity_rotation_rad=0.0
    )
    np.testing.assert_allclose(unchanged, velocity, rtol=1e-14)


def test_synthetic_secondary_rejects_a_bad_axis_or_offset():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    with pytest.raises(ValueError, match="rotation_axis"):
        synthetic_secondary(
            position,
            velocity,
            offset_rtn_m=(0.0, 0.0, 0.0),
            velocity_rotation_rad=0.1,
            rotation_axis="sideways",
        )
    with pytest.raises(ValueError, match="3 components"):
        synthetic_secondary(
            position, velocity, offset_rtn_m=(0.0, 0.0), velocity_rotation_rad=0.1
        )


def test_closest_approach_finds_the_declared_offset():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    secondary_position, secondary_velocity = synthetic_secondary(
        position, velocity, offset_rtn_m=(30.0, 110.0, 40.0), velocity_rotation_rad=0.7854
    )
    result = find_closest_approach(
        lambda t: state_from_tle(LINE1, LINE2, t / 60.0),
        lambda t: (secondary_position + secondary_velocity * t, secondary_velocity),
        -60.0,
        60.0,
    )
    assert isinstance(result, ClosestApproach)
    assert result.separation_m <= result.coarse_separation_m
    assert result.separation_m < 130.0
    assert result.evaluations > 0


def test_closest_approach_refinement_improves_on_the_coarse_scan():
    position, velocity = state_from_tle(LINE1, LINE2, 0.0)
    secondary_position, secondary_velocity = synthetic_secondary(
        position, velocity, offset_rtn_m=(0.0, 400.0, 0.0), velocity_rotation_rad=0.2
    )

    def state_b(t: float):
        return secondary_position + secondary_velocity * t, secondary_velocity

    coarse = find_closest_approach(
        lambda t: state_from_tle(LINE1, LINE2, t / 60.0), state_b, -30.0, 30.0,
        coarse_steps=4, golden_iterations=0,
    )
    fine = find_closest_approach(
        lambda t: state_from_tle(LINE1, LINE2, t / 60.0), state_b, -30.0, 30.0,
        coarse_steps=400, golden_iterations=60,
    )
    assert fine.separation_m <= coarse.separation_m


def test_closest_approach_rejects_a_bad_interval():
    with pytest.raises(ValueError, match="t_end_s"):
        find_closest_approach(
            lambda t: (np.zeros(3), np.ones(3)), lambda t: (np.ones(3), np.ones(3)), 1.0, 1.0
        )
    with pytest.raises(ValueError, match="coarse_steps"):
        find_closest_approach(
            lambda t: (np.zeros(3), np.ones(3)),
            lambda t: (np.ones(3), np.ones(3)),
            0.0,
            1.0,
            coarse_steps=1,
        )
