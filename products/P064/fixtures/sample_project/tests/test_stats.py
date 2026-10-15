import pytest

from sampleproj import stats


def test_mean_known_answer():
    assert stats.mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_variance_bessel_corrected():
    # [1,2,3,4]: mean 2.5, SS = 2.25+0.25+0.25+2.25 = 5.0, /(4-1) = 1.6666667
    assert stats.variance([1.0, 2.0, 3.0, 4.0]) == pytest.approx(5.0 / 3.0)


def test_rms_known_answer():
    # rms([3,4]) = sqrt((9+16)/2) = sqrt(12.5)
    assert stats.rms([3.0, 4.0]) == pytest.approx(12.5**0.5)
