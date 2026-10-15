import pytest

from sampleproj import integrate


def test_trapezoid_linear_is_exact():
    # f(x) = x on [0,1] with h=0.25: exact integral 0.5
    ys = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert integrate.trapezoid(ys, 0.25) == pytest.approx(0.5)


def test_simpson_quadratic_is_exact():
    # f(x) = x^2 on [0,1], h=0.25, exact 1/3; Simpson is exact for quadratics
    ys = [0.0, 0.0625, 0.25, 0.5625, 1.0]
    assert integrate.simpson(ys, 0.25) == pytest.approx(1.0 / 3.0)


def test_simpson_rejects_even_sample_count():
    with pytest.raises(ValueError):
        integrate.simpson([0.0, 1.0, 2.0, 3.0], 0.1)
