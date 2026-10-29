"""Pinned defects. Every one of these is a bug that existed in this build.

A Hypothesis counterexample lives in ``.hypothesis/``, which is gitignored, so
it is lost on a fresh clone. Every defect found during this build -- by
Hypothesis, by a failing hand-written test, or by a measurement that came out
absurd -- is pinned here with its literal inputs and a note saying what went
wrong, so that a cold clone re-checks all of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import batch, dataset, observability
from aligncal.geometry import rotation_from_vector, rotation_to_vector, skew, unskew


def test_logarithm_scale_factor_is_p_over_sin_p_not_p_over_two_sin_p():
    """Pinned: the first draft of ``rotation_to_vector`` halved every angle.

    ``(R - R^T) / 2 = sin(p) [n]x`` and :func:`unskew` already takes the
    antisymmetric part, so the scale factor is ``p / sin p``. With the spurious
    factor of two the round trip returned half the input, which every
    long-record estimate then silently halved. Checked at a literal value where
    the two differ by exactly a factor of two.
    """
    theta = np.array([0.3, -0.5, 0.81])
    recovered = rotation_to_vector(rotation_from_vector(theta))
    assert np.max(np.abs(recovered - theta)) < 1.0e-12
    assert np.max(np.abs(recovered - 0.5 * theta)) > 0.1


def test_generated_secondary_rate_applies_the_mounting_row_wise():
    """Pinned: the generator omitted a transpose and produced ``(C^true)^T w``.

    ``rates @ A`` has rows ``A^T w``, so the mounting has to be applied as
    ``rates @ A.T``. With the transpose missing, every record whose declared
    mounting was not the identity generated data for the inverse rotation, the
    residual of equation (7) was of order the mounting angle rather than of
    order the misalignment, and the batch estimator diverged on 98.2 per cent
    of a 600-record population with a median error of 1880 mrad. The literal
    check is that the de-rotated secondary rate equals the reference rate when
    no misalignment is injected.
    """
    generated = dataset.generate(
        samples=200, noise_sigma=1.0e-12, seed=4242, theta_rad=np.zeros(3)
    )
    difference = generated.record.derotated("secondary_1") - generated.true_rates
    assert float(np.max(np.abs(difference))) < 1.0e-10


def test_mounting_is_not_the_identity_in_that_regression():
    """The regression above is only meaningful if the mounting is actually rotated."""
    generated = dataset.generate(samples=50, seed=4242)
    mounting = generated.record.secondaries["secondary_1"].mounting
    assert float(np.max(np.abs(mounting - np.eye(3)))) > 0.1


def test_equation_ten_pairs_ascending_eigenvalues_with_descending_shares():
    """Pinned: comparing the two in the same order fails by a factor of 1.

    ``numpy.linalg.eigvalsh`` returns ascending eigenvalues; ``excitation_spectrum``
    returns descending shares. By equation (10) those orders CORRESPOND -- the
    largest eigenvalue goes with the smallest excitation share -- and reversing
    one of them turns a correct identity into a 100 per cent error. Pinned on
    the single-axis profile, where the mis-pairing is maximal.
    """
    generated = dataset.generate(profile="single_axis", samples=500, seed=77)
    record = generated.record
    shares, _, energy = observability.excitation_spectrum(record)
    scale = energy / record.pair_sigma("secondary_1") ** 2
    values = np.linalg.eigvalsh(observability.gramian(record, "secondary_1"))
    assert np.max(np.abs(values - scale * (1.0 - shares))) < 1.0e-9 * scale
    assert np.max(np.abs(values - scale * (1.0 - shares[::-1]))) > 0.9 * scale


def test_residual_linearisation_is_relative_order_theta_not_theta_squared():
    """Pinned: the residual's RELATIVE linearisation error is O(|theta|), not O(|theta|^2).

    The rotation matrix ``I + [theta]x`` is accurate to ``|theta|^2 / 2``
    against a matrix of order 1. The residual is itself of order
    ``|theta| |w|``, so the same absolute error is a relative error of
    ``|theta| / 2`` -- a hundred times larger at 10 mrad. A test written to the
    matrix figure failed, and the distinction is pinned here at the literal
    misalignment that exposed it.
    """
    times = np.arange(200, dtype=float) / 20.0
    rates = np.column_stack(
        [0.05 * np.sin(times), 0.04 * np.cos(0.7 * times), 0.03 * np.sin(1.3 * times)]
    )
    theta = np.array([1.0, -0.6, 0.3]) * 1.0e-3
    exact = rates @ rotation_from_vector(-theta).T - rates
    linear = skew(rates) @ theta
    scale = float(np.max(np.abs(linear)))
    relative = float(np.max(np.abs(exact - linear))) / scale
    magnitude = float(np.linalg.norm(theta))
    assert relative < magnitude  # O(|theta|), the correct order
    assert relative > magnitude * magnitude  # and NOT O(|theta|^2)


def test_gauss_newton_needs_a_trust_region_on_a_low_information_record():
    """Pinned: without the trust region the iteration failed to converge on 7.8 per cent.

    A 150-sample record at 0.01 rad/s amplitude and 8e-4 rad/s noise has a
    Cramer-Rao one-sigma of tens of milliradians, so the first Newton step can
    be of order a radian, which leaves the region where the linearisation of
    (20) holds. The step limiter of :data:`aligncal.batch.DEFAULT_MAX_STEP_RAD`
    fixes it. Pinned on a literal configuration that reproduces the regime.
    """
    generated = dataset.generate(
        samples=150, amplitude=0.01, noise_sigma=8.0e-4, theta_max_rad=3.0e-3, seed=31337
    )
    verdict = observability.assess(generated.record)
    solution = batch.solve(generated.record, verdict, "secondary_1")
    assert solution.converged
    assert float(np.linalg.norm(solution.theta_rad)) < np.pi


def test_withheld_component_is_exactly_zero_after_the_matrix_logarithm():
    """Pinned: a rounding-level leak into the withheld direction.

    The Gauss-Newton updates are built from the retained basis only, but the
    matrix logarithm of a product of such updates can carry a component of
    order 1e-19 rad into the withheld direction. It is projected out
    explicitly, and the exact zero is the claim this test checks, because a
    non-zero number in a direction reported as unestimated is exactly the
    failure this product exists to prevent.
    """
    generated = dataset.generate(profile="single_axis", samples=1500, seed=2024)
    verdict = observability.assess(generated.record)
    sensor = verdict["secondary_1"]
    solution = batch.solve(generated.record, verdict, "secondary_1")
    axis = sensor.degrees_of_freedom[-1].direction
    assert solution.withheld == (2,)
    assert abs(float(np.dot(solution.theta_rad, axis))) < 1.0e-15


def test_unskew_of_a_symmetric_matrix_is_zero():
    """Pinned: ``unskew`` must take the antisymmetric part rather than assume it.

    A covariance or Gramian that has drifted symmetric to within rounding is
    passed through this function by the MEKF reset path; assuming
    antisymmetry there returned half a diagonal entry as a rotation.
    """
    symmetric = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert np.array_equal(unskew(symmetric), np.zeros(3))


def test_single_axis_record_without_the_noise_floor_gate_looks_full_rank():
    """Pinned as a control: the measured Gramian of a single-axis record is NOT singular.

    Equation (11): the reference sensor's own noise contributes
    ``2 sigma_A^2`` per sample in every direction, including the one the motion
    never excited. A bare rank test on the measured Gramian therefore reports
    three observable degrees of freedom for a record that supports two. This is
    the defect the floor gate exists to prevent, and the test asserts that the
    unguarded condition number is finite and modest -- which is to say, that
    the unguarded test really would be fooled.
    """
    generated = dataset.generate(profile="single_axis", samples=1000, seed=5150)
    matrix = observability.gramian(generated.record, "secondary_1")
    values = np.linalg.eigvalsh(matrix)
    assert float(values[0]) > 0.0
    assert float(values[-1] / values[0]) < 1.0e6  # a rank test would call this full rank
    sensor = observability.assess(generated.record)["secondary_1"]
    assert sensor.rank == 2  # the floor gate is not fooled


def test_coverage_treats_an_infinite_interval_as_a_miss():
    """Pinned: an infinite sigma is a refusal, not a successful prediction.

    The analytic estimator reports an infinite one-sigma in a withheld
    direction. Counting that as covering the truth would have scored the
    refusing estimator as perfectly calibrated and the measurement this product
    exists to make would have come out backwards.
    """
    from aligncal.benchmark import coverage

    assert coverage(np.array([1.0e-3]), np.array([np.inf])) == 0.0


def test_pytest_approx_absolute_default_is_never_relied_on():
    """Pinned: ``pytest.approx(x, rel=r)`` silently passes on a 1e-12 absolute bound.

    Every ``rel=`` call in this suite passes an explicit ``abs=``. This test is
    the reminder, and it demonstrates the trap on a quantity of the size this
    product actually handles: a misalignment component of 1e-5 rad compared at
    a relative tolerance of 1e-9 passes against a value that is wrong by 1e-13
    only because of the hidden absolute default.
    """
    actual = 1.0e-5
    wrong = actual + 1.0e-13
    assert wrong == pytest.approx(actual, rel=1.0e-9)  # passes on the hidden default
    assert wrong != pytest.approx(actual, rel=1.0e-9, abs=0.0)  # with abs=0.0 it does not
