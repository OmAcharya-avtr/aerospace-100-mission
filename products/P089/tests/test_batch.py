"""Baseline 1: the batch solution, its subspace restriction and its SVD cross-check."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import batch, dataset, observability
from aligncal.geometry import rotation_from_vector
from aligncal.record import AlignmentRecord, SensorSpec


def _noiseless(theta: np.ndarray, profile: str = "three_axis", samples: int = 600):
    """A record with a known misalignment and no noise at all, for known-answer tests."""
    times = np.arange(samples, dtype=float) / 100.0
    rates = dataset.rate_profile(profile, times, amplitude=0.05, seed=3)
    mounting = rotation_from_vector(np.array([0.4, -0.9, 0.2]))
    true_mounting = mounting @ rotation_from_vector(-np.asarray(theta, dtype=float))
    return AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 1.0e-8),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", 1.0e-8, mounting)},
        secondary_rates={"s": rates @ true_mounting.T},
        provenance="tests/test_batch.py _noiseless",
    )


def test_normal_equation_rhs_matches_explicit_summation(rich):
    record = rich.record
    name = record.names[0]
    from aligncal.geometry import skew
    from aligncal.record import residuals

    weight = 1.0 / record.pair_sigma(name) ** 2
    accumulated = np.zeros(3)
    residual = residuals(record, name)
    for index, rate in enumerate(record.reference_rates):
        accumulated += skew(rate).T @ residual[index]
    accumulated *= weight
    computed = batch.normal_equation_rhs(record, name)
    assert np.max(np.abs(computed - accumulated)) < 1.0e-6 * float(np.max(np.abs(computed)))


def test_noiseless_recovery_is_exact_to_the_iteration_tolerance():
    """A known answer: with no noise and rich excitation the estimate is the truth."""
    theta = np.array([2.0, -1.3, 0.7]) * 1.0e-3
    record = _noiseless(theta)
    verdict = observability.assess(record)
    solution = batch.solve(record, verdict, "s")
    assert solution.converged
    assert np.max(np.abs(solution.theta_rad - theta)) < 1.0e-10


def test_noiseless_recovery_at_a_large_misalignment():
    """100 mrad, where a single linear pass would carry a 5 per cent relative error.

    The Gauss-Newton iteration re-linearises, so the recovered value is exact
    to the iteration tolerance and not to O(|theta|).
    """
    theta = np.array([60.0, -80.0, 30.0]) * 1.0e-3
    record = _noiseless(theta)
    verdict = observability.assess(record)
    solution = batch.solve(record, verdict, "s")
    assert np.max(np.abs(solution.theta_rad - theta)) < 1.0e-9


def test_single_linear_pass_does_carry_the_linearisation_error():
    """One iteration at 100 mrad is wrong by about |theta|/2 relative, as documented."""
    theta = np.array([60.0, -80.0, 30.0]) * 1.0e-3
    record = _noiseless(theta)
    verdict = observability.assess(record)
    one = batch.solve(record, verdict, "s", max_iterations=1)
    relative = float(np.linalg.norm(one.theta_rad - theta)) / float(np.linalg.norm(theta))
    assert 1.0e-4 < relative < 0.2


def test_withheld_direction_is_exactly_zero(single_axis, single_verdict):
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    assert solution.withheld == (2,)
    axis = single_verdict["secondary_1"].degrees_of_freedom[2].direction
    assert abs(float(np.dot(solution.theta_rad, axis))) < 1.0e-15


def test_nothing_is_withheld_on_a_rich_record(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    assert solution.withheld == ()


def test_covariance_is_the_pseudo_inverse_on_the_retained_subspace(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    gramian = rich_verdict["secondary_1"].gramian
    product = solution.covariance @ gramian
    assert np.max(np.abs(product - np.eye(3))) < 1.0e-8


def test_covariance_has_a_null_direction_when_one_is_withheld(single_axis, single_verdict):
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    axis = single_verdict["secondary_1"].degrees_of_freedom[2].direction
    assert float(np.linalg.norm(solution.covariance @ axis)) < 1.0e-12 * float(
        np.max(np.abs(solution.covariance))
    )


def test_sigma_in_direction_is_infinite_on_a_withheld_direction(single_axis, single_verdict):
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    axis = single_verdict["secondary_1"].degrees_of_freedom[2].direction
    assert not np.isfinite(solution.sigma_rad_in_direction(axis))
    best = single_verdict["secondary_1"].degrees_of_freedom[0].direction
    assert np.isfinite(solution.sigma_rad_in_direction(best))


def test_sigma_in_direction_matches_the_crb(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    for dof in rich_verdict["secondary_1"].degrees_of_freedom:
        assert solution.sigma_rad_in_direction(dof.direction) == pytest.approx(
            dof.crb_sigma_rad, rel=1.0e-8, abs=0.0
        )


def test_sigma_in_direction_rejects_zero(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    with pytest.raises(ValueError, match="non-zero direction"):
        solution.sigma_rad_in_direction(np.zeros(3))


def test_solution_rotation_is_a_rotation(rich, rich_verdict):
    from aligncal.geometry import is_rotation

    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    assert is_rotation(solution.rotation)


def test_theta_mrad_and_magnitude(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    assert np.allclose(solution.theta_mrad, solution.theta_rad * 1.0e3, atol=0.0, rtol=0.0)
    assert solution.magnitude_arcsec == pytest.approx(
        float(np.linalg.norm(solution.theta_rad)) * 206264.806247, rel=1.0e-10, abs=0.0
    )


def test_residual_rms_decreases(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    assert solution.residual_rms_after <= solution.residual_rms_before


def test_solve_rejects_bad_iteration_controls(rich, rich_verdict):
    with pytest.raises(ValueError, match="max_step_rad"):
        batch.solve(rich.record, rich_verdict, "secondary_1", max_step_rad=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        batch.solve(rich.record, rich_verdict, "secondary_1", max_iterations=0)


def test_solve_rejects_unknown_sensor(rich, rich_verdict):
    with pytest.raises(KeyError, match="unknown secondary"):
        batch.solve(rich.record, rich_verdict, "nope")


def test_trust_region_limits_one_step():
    limited = batch._limit(np.array([1.0, 0.0, 0.0]), 0.2)
    assert float(np.linalg.norm(limited)) == pytest.approx(0.2, rel=1.0e-14, abs=0.0)
    unchanged = np.array([0.01, 0.02, -0.03])
    assert np.array_equal(batch._limit(unchanged, 0.2), unchanged)
    assert np.array_equal(batch._limit(np.zeros(3), 0.2), np.zeros(3))


def test_trust_region_preserves_direction():
    step = np.array([3.0, -4.0, 0.0])
    limited = batch._limit(step, 0.5)
    assert np.max(np.abs(limited / np.linalg.norm(limited) - step / np.linalg.norm(step))) < 1e-14


# ---------------------------------------------------------------------------
# Equation (17): Markley's SVD solution as an independent cross-check.
# ---------------------------------------------------------------------------


def test_svd_agrees_with_the_normal_equations_on_a_rich_record(rich, rich_verdict):
    """Two routes to the same minimiser, sharing no arithmetic."""
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    svd_theta, singular = batch.wahba_svd(rich.record, "secondary_1")
    assert np.max(np.abs(svd_theta - solution.theta_rad)) < 1.0e-9
    assert np.all(np.diff(singular) <= 1.0e-9)


def test_svd_agrees_across_profiles_on_the_observable_subspace():
    """On a rank-deficient record the two agree only where the record supports them."""
    worst = 0.0
    for profile in ("two_axis", "three_axis", "isotropic"):
        generated = dataset.generate(profile=profile, samples=1200, seed=61)
        verdict = observability.assess(generated.record)
        solution = batch.solve(generated.record, verdict, "secondary_1")
        svd_theta, _ = batch.wahba_svd(generated.record, "secondary_1")
        worst = max(worst, float(np.max(np.abs(svd_theta - solution.theta_rad))))
    assert worst < 1.0e-8


def test_svd_has_no_mechanism_to_withhold(single_axis, single_verdict):
    """The SVD solution returns a number for the direction the record cannot support."""
    svd_theta, singular = batch.wahba_svd(single_axis.record, "secondary_1")
    axis = single_verdict["secondary_1"].degrees_of_freedom[2].direction
    component = abs(float(np.dot(svd_theta, axis)))
    batch_component = abs(
        float(
            np.dot(batch.solve(single_axis.record, single_verdict, "secondary_1").theta_rad, axis)
        )
    )
    assert batch_component < 1.0e-15
    assert component > 1.0e-6
    assert float(np.min(singular)) / float(np.max(singular)) < 1.0e-3


def test_svd_contamination_reaches_the_observable_subspace(single_axis, single_verdict):
    """On a rank-deficient record the SVD gauge rotation contaminates the good components.

    This is the sharper half of the result. The SVD solution of (17) picks up a
    rotation of order a radian in the direction the record cannot support, and
    because composing rotations is not a linear operation that large component
    also moves the two components the record DOES support -- by more than the
    whole magnitude of the correct estimate. An unconstrained solution on a
    rank-deficient record is not merely uninformative about the unobservable
    direction; it is wrong about the observable ones too.
    """
    sensor = single_verdict["secondary_1"]
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    svd_theta, _ = batch.wahba_svd(single_axis.record, "secondary_1")
    axis = sensor.degrees_of_freedom[2].direction
    assert abs(float(np.dot(svd_theta, axis))) > 0.1  # radians: a gauge rotation
    difference = float(np.linalg.norm(sensor.project_observable(svd_theta - solution.theta_rad)))
    assert difference > float(np.linalg.norm(solution.theta_rad))


# ---------------------------------------------------------------------------
# Equation (18): the regularised variant.
# ---------------------------------------------------------------------------


def test_map_shrinks_toward_zero():
    theta = np.array([2.0, -1.3, 0.7]) * 1.0e-3
    record = _noiseless(theta)
    verdict = observability.assess(record)
    strong = batch.solve_map(record, verdict, "s", prior_sigma_rad=1.0e-6)
    weak = batch.solve_map(record, verdict, "s", prior_sigma_rad=1.0)
    assert float(np.linalg.norm(strong.theta_rad)) < float(np.linalg.norm(weak.theta_rad))
    assert float(np.linalg.norm(weak.theta_rad - theta)) < 1.0e-8


def test_map_withholds_nothing(single_axis, single_verdict):
    solution = batch.solve_map(
        single_axis.record, single_verdict, "secondary_1", prior_sigma_rad=1.0e-3
    )
    assert solution.withheld == ()


def test_map_returns_the_prior_mean_in_an_unexcited_direction(noiseless_single):
    """With no excitation about the axis the MAP estimate there is zero, the prior mean."""
    verdict = observability.assess(noiseless_single)
    solution = batch.solve_map(
        noiseless_single, verdict, "secondary_1", prior_sigma_rad=1.0e-3
    )
    axis = verdict["secondary_1"].degrees_of_freedom[-1].direction
    assert abs(float(np.dot(solution.theta_rad, axis))) < 1.0e-12


def test_map_covariance_approaches_the_prior_in_an_unexcited_direction(noiseless_single):
    verdict = observability.assess(noiseless_single)
    prior = 1.0e-3
    solution = batch.solve_map(noiseless_single, verdict, "secondary_1", prior_sigma_rad=prior)
    axis = verdict["secondary_1"].degrees_of_freedom[-1].direction
    sigma = float(np.sqrt(axis @ solution.covariance @ axis))
    assert sigma == pytest.approx(prior, rel=1.0e-6, abs=0.0)


@pytest.mark.parametrize("prior", [0.0, -1.0, np.nan, np.inf])
def test_map_rejects_an_unusable_prior(rich, rich_verdict, prior):
    with pytest.raises(ValueError, match="prior_sigma_rad"):
        batch.solve_map(rich.record, rich_verdict, "secondary_1", prior_sigma_rad=prior)


def test_map_rejects_bad_iteration_controls(rich, rich_verdict):
    with pytest.raises(ValueError, match="max_step_rad"):
        batch.solve_map(
            rich.record, rich_verdict, "secondary_1", prior_sigma_rad=1e-3, max_step_rad=-1.0
        )
    with pytest.raises(ValueError, match="max_iterations"):
        batch.solve_map(
            rich.record, rich_verdict, "secondary_1", prior_sigma_rad=1e-3, max_iterations=0
        )


def test_predicted_residual_shape_and_value(rich):
    theta = np.array([1.0e-3, 0.0, 0.0])
    predicted = batch.predicted_residual(rich.record, theta)
    assert predicted.shape == (len(rich.record), 3)
    from aligncal.geometry import skew

    assert np.array_equal(predicted[5], skew(rich.record.reference_rates[5]) @ theta)


def test_summary_puts_the_verdict_first(single_axis, single_verdict):
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    lines = solution.summary()
    assert lines[0].startswith("sensor secondary_1")
    assert "RANK DEFICIENT" in lines[0]
    text = "\n".join(lines)
    assert "WITHHELD" in text
    assert text.index("does NOT support") < text.index("BATCH ESTIMATE")


def test_multi_secondary_solutions_are_independent(multi):
    verdict = observability.assess(multi.record)
    solutions = {
        name: batch.solve(multi.record, verdict, name) for name in multi.record.names
    }
    for name, solution in solutions.items():
        error = float(np.linalg.norm(solution.theta_rad - multi.truth[name]))
        assert error < 5.0e-4
    thetas = np.vstack([s.theta_rad for s in solutions.values()])
    assert float(np.min(np.std(thetas, axis=0))) > 0.0
