import math

import pytest

from sampleproj import vecops


def test_dot_known_answer():
    # [1,2,3].[4,5,6] = 4 + 10 + 18 = 32
    assert vecops.dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)


def test_norm_known_answer():
    # |[3,4]| = 5
    assert vecops.norm([3.0, 4.0]) == pytest.approx(5.0)


def test_normalize_is_unit():
    u = vecops.normalize([1.0, 1.0, 1.0])
    assert vecops.norm(u) == pytest.approx(1.0)
    assert u[0] == pytest.approx(1.0 / math.sqrt(3.0))


def test_cross_right_handed():
    assert vecops.cross([1, 0, 0], [0, 1, 0]) == pytest.approx([0.0, 0.0, 1.0])
