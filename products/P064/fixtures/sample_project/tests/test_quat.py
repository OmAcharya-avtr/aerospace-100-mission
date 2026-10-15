import pytest

from sampleproj import quat


def test_identity_multiply():
    q = [0.5, 0.5, 0.5, 0.5]
    assert quat.multiply([1.0, 0.0, 0.0, 0.0], q) == pytest.approx(q)


def test_conjugate_flips_vector_part():
    assert quat.conjugate([1.0, 2.0, 3.0, 4.0]) == pytest.approx([1.0, -2.0, -3.0, -4.0])


def test_rotate_90_about_z():
    # 90 deg about z maps x -> y
    s = 0.5**0.5
    out = quat.rotate([s, 0.0, 0.0, s], [1.0, 0.0, 0.0])
    assert out == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_rotate_preserves_length():
    s = 0.5**0.5
    out = quat.rotate([s, s, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert sum(c * c for c in out) == pytest.approx(14.0)
