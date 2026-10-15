import pytest

from sampleproj import filters


def test_moving_average_known_answer():
    # window 3 over [1,2,3,4,5] -> [2,3,4]
    assert filters.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx([2.0, 3.0, 4.0])


def test_moving_average_window_one_is_identity():
    xs = [1.0, 7.0, -3.0]
    assert filters.moving_average(xs, 1) == pytest.approx(xs)


def test_residual_rms_of_a_ramp_is_the_half_window_offset():
    # a moving average of a linear ramp reproduces the centre, so the trailing
    # residual is a constant offset of one half-window step, not zero.
    out = filters.residual_rms([0.0, 1.0, 2.0, 3.0, 4.0], 3)
    assert out == pytest.approx(1.0)
