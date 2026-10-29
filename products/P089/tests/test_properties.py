"""Hypothesis property tests over the algebraic identities this product rests on.

Every identity here is exact in exact arithmetic, so each is asserted to a
tolerance that follows the conditioning of the expression rather than to an
absolute constant. No boolean produced by a threshold comparison is asserted
equal to anything: a threshold boolean is a step function of a rounded quantity
and is the one thing floating point cannot promise.

``.hypothesis/`` is gitignored, so any falsifying example found by a run is lost
on a fresh clone. Every counterexample this suite found during the build is
pinned by literal value in ``tests/test_regressions.py``.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aligncal.geometry import (
    rotation_from_vector,
    rotation_to_vector,
    skew,
    unskew,
)
from aligncal.observability import (
    eigenvalue_bracket,
    excitation_spectrum,
    gramian,
    noise_floor_eigenvalue,
)
from aligncal.record import AlignmentRecord, SensorSpec

PROFILE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

finite = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=64
)
small = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=64
)
vectors = st.lists(finite, min_size=3, max_size=3).map(lambda v: np.array(v, dtype=float))
small_vectors = st.lists(small, min_size=3, max_size=3).map(
    lambda v: np.array(v, dtype=float)
)
#: Misalignment-sized vectors: components in [-0.1, 0.1] rad, so the magnitude
#: is always inside the validity range of the linearisation and the property
#: below does not have to filter most of its inputs away.
misalignments = st.lists(
    st.floats(
        min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False, width=64
    ),
    min_size=3,
    max_size=3,
).map(lambda v: np.array(v, dtype=float))


@PROFILE
@given(vectors)
def test_skew_is_the_cross_product(vector):
    other = np.array([1.0, -2.0, 0.5])
    left = skew(vector) @ other
    right = np.cross(vector, other)
    scale = float(np.linalg.norm(vector)) * float(np.linalg.norm(other))
    assert np.max(np.abs(left - right)) <= 8.0 * np.finfo(float).eps * max(scale, 1.0e-300)


@PROFILE
@given(vectors)
def test_skew_unskew_round_trip(vector):
    assert np.max(np.abs(unskew(skew(vector)) - vector)) == 0.0


@PROFILE
@given(vectors)
def test_equation_three_identity(vector):
    """[a]x^T [a]x = |a|^2 I - a a^T, with a slack following the conditioning."""
    left = skew(vector).T @ skew(vector)
    right = float(vector @ vector) * np.eye(3) - np.outer(vector, vector)
    scale = float(vector @ vector)
    assert np.max(np.abs(left - right)) <= 16.0 * np.finfo(float).eps * max(scale, 1.0e-300)


@PROFILE
@given(vectors)
def test_skew_annihilates_its_own_vector(vector):
    """[a]x a = 0: the algebraic reason the rotation axis is unobservable."""
    scale = float(vector @ vector)
    assert np.max(np.abs(skew(vector) @ vector)) <= 8.0 * np.finfo(float).eps * max(
        scale, 1.0e-300
    )


@PROFILE
@given(small_vectors)
def test_rotation_is_orthogonal_with_unit_determinant(vector):
    matrix = rotation_from_vector(vector)
    assert np.max(np.abs(matrix.T @ matrix - np.eye(3))) <= 1.0e-14
    assert abs(float(np.linalg.det(matrix)) - 1.0) <= 1.0e-14


@PROFILE
@given(small_vectors)
def test_rotation_logarithm_round_trip(vector):
    """exp and log invert each other below pi, to a slack following |theta|."""
    magnitude = float(np.linalg.norm(vector))
    assume(magnitude < np.pi * 0.98)
    recovered = rotation_to_vector(rotation_from_vector(vector))
    assert np.max(np.abs(recovered - vector)) <= 1.0e-12 * max(magnitude, 1.0)


@PROFILE
@given(small_vectors)
def test_rotation_preserves_length(vector):
    probe = np.array([0.3, -0.7, 1.1])
    rotated = rotation_from_vector(vector) @ probe
    assert abs(float(np.linalg.norm(rotated)) - float(np.linalg.norm(probe))) <= 1.0e-14


@PROFILE
@given(small_vectors, small_vectors)
def test_rotation_inverse_is_the_negated_vector(first, second):
    matrix = rotation_from_vector(first)
    inverse = rotation_from_vector(-first)
    assert np.max(np.abs(matrix @ inverse - np.eye(3))) <= 1.0e-13
    del second


def _record(rates: np.ndarray, sigma: float = 1.0e-4) -> AlignmentRecord:
    times = np.arange(rates.shape[0], dtype=float) / 50.0
    return AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", sigma),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", sigma)},
        secondary_rates={"s": rates},
        provenance="tests/test_properties.py",
    )


rate_blocks = st.integers(min_value=4, max_value=40).flatmap(
    lambda n: st.lists(
        st.lists(
            st.floats(
                min_value=-0.2,
                max_value=0.2,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=3,
            max_size=3,
        ),
        min_size=n,
        max_size=n,
    ).map(lambda rows: np.array(rows, dtype=float))
)


@PROFILE
@given(rate_blocks)
def test_gramian_is_symmetric_positive_semidefinite(rates):
    assume(float(np.sum(rates * rates)) > 1.0e-12)
    matrix = gramian(_record(rates), "s")
    assert np.max(np.abs(matrix - matrix.T)) == 0.0
    scale = float(np.max(np.abs(matrix)))
    assert float(np.min(np.linalg.eigvalsh(matrix))) >= -1.0e-12 * max(scale, 1.0e-300)


@PROFILE
@given(rate_blocks)
def test_equation_ten_holds_whatever_the_motion(rates):
    """lambda_i = (W / s^2)(1 - p_i), with the eigenvalue and share orders reversed."""
    assume(float(np.sum(rates * rates)) > 1.0e-10)
    record = _record(rates)
    shares, _, energy = excitation_spectrum(record)
    scale = energy / record.pair_sigma("s") ** 2
    values = np.linalg.eigvalsh(gramian(record, "s"))
    predicted = scale * (1.0 - shares)
    assert np.max(np.abs(values - predicted)) <= 1.0e-9 * scale


@PROFILE
@given(rate_blocks)
def test_excitation_shares_sum_to_one(rates):
    assume(float(np.sum(rates * rates)) > 1.0e-12)
    shares, _, _ = excitation_spectrum(_record(rates))
    assert abs(float(np.sum(shares)) - 1.0) <= 1.0e-12
    assert float(np.min(shares)) >= -1.0e-12


@PROFILE
@given(rate_blocks)
def test_gramian_annihilates_a_single_axis_motion(rates):
    """Any motion collapsed onto one axis has that axis in the Gramian's null space."""
    assume(float(np.sum(rates * rates)) > 1.0e-10)
    axis = np.array([0.3, -0.5, 0.81])
    axis = axis / float(np.linalg.norm(axis))
    collapsed = np.outer(rates @ axis, axis)
    assume(float(np.sum(collapsed * collapsed)) > 1.0e-12)
    matrix = gramian(_record(collapsed), "s")
    scale = float(np.max(np.abs(matrix)))
    assert float(np.linalg.norm(matrix @ axis)) <= 1.0e-12 * max(scale, 1.0e-300)


@PROFILE
@given(rate_blocks)
def test_bracket_contains_the_smallest_eigenvalue(rates):
    """Equation (13): Gershgorin and Rayleigh bound what eigh computes."""
    assume(float(np.sum(rates * rates)) > 1.0e-10)
    matrix = gramian(_record(rates), "s")
    lower, upper = eigenvalue_bracket(matrix)
    smallest = float(np.min(np.linalg.eigvalsh(matrix)))
    scale = max(float(np.max(np.abs(matrix))), 1.0e-300)
    assert lower <= smallest + 1.0e-10 * scale
    assert smallest <= upper + 1.0e-10 * scale


@PROFILE
@given(
    st.integers(min_value=4, max_value=5000),
    st.floats(min_value=1.0e-6, max_value=1.0e-2, allow_nan=False, allow_infinity=False),
)
def test_noise_floor_is_linear_in_samples_and_independent_of_sigma(samples, sigma):
    """With equal sensor sigmas, s^2 = 2 sigma^2 and the floor is exactly n."""
    rates = np.tile(np.array([[0.01, 0.02, -0.03]]), (samples, 1))
    floor = noise_floor_eigenvalue(_record(rates, sigma=sigma), "s")
    assert abs(floor - samples) <= 1.0e-9 * samples


@PROFILE
@given(rate_blocks, misalignments)
def test_residual_linearisation_error_follows_the_misalignment(rates, theta):
    """|r - [w]x theta| <= |theta|^2 |w| to a constant, which is the documented bound."""
    assume(float(np.sum(rates * rates)) > 1.0e-10)
    magnitude = float(np.linalg.norm(theta))
    assume(magnitude > 1.0e-9)
    rotation = rotation_from_vector(-theta)
    exact = rates @ rotation.T - rates
    linear = skew(rates) @ theta
    worst_rate = float(np.max(np.linalg.norm(rates, axis=1)))
    assert np.max(np.abs(exact - linear)) <= magnitude * magnitude * worst_rate
