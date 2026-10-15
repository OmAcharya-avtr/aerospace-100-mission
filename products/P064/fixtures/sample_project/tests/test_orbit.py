import math

import pytest

from sampleproj import orbit


def test_circular_velocity_leo():
    # r = 7.0e6 m: v = sqrt(3.986004418e14 / 7.0e6) = 7546.05... m/s
    assert orbit.circular_velocity(7.0e6) == pytest.approx(7546.0532, rel=1e-6)


def test_period_matches_vis_viva():
    r = 7.0e6
    v = orbit.circular_velocity(r)
    assert orbit.period(r) == pytest.approx(2.0 * math.pi * r / v, rel=1e-9)


def test_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        orbit.period(0.0)
