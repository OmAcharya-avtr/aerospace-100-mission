"""Baseline 2: the MEKF, its agreement with the batch solution, and its own refusal."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import batch, dataset, observability, recursive


def test_agrees_with_batch_on_a_rich_record(rich, rich_verdict):
    """The two solutions agree on every observable degree of freedom.

    The tolerance is stated rather than inherited: the MEKF carries a prior of
    one-sigma 0.1 rad, which shrinks the estimate along degree of freedom i by
    lambda_i / (lambda_i + 1 / p0^2). With lambda_i of order 1e7 that factor is
    within 1e-5 of unity, and the measured agreement in
    validation/validate_recursive_agreement.py is reported against it.
    """
    batch_solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    sensor = rich_verdict["secondary_1"]
    for dof in sensor.degrees_of_freedom:
        difference = abs(
            float(np.dot(mekf.theta_rad - batch_solution.theta_rad, dof.direction))
        )
        assert difference < 0.02 * dof.crb_sigma_rad


def test_agreement_holds_across_profiles_on_the_observable_subspace():
    worst_ratio = 0.0
    for profile in ("two_axis", "three_axis", "isotropic"):
        generated = dataset.generate(profile=profile, samples=1200, seed=818)
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        batch_solution = batch.solve(generated.record, verdict, "secondary_1")
        mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=False)
        for dof in sensor.degrees_of_freedom:
            difference = abs(
                float(np.dot(mekf.theta_rad - batch_solution.theta_rad, dof.direction))
            )
            worst_ratio = max(worst_ratio, difference / dof.crb_sigma_rad)
    assert worst_ratio < 0.1


def test_covariance_matches_the_crb_on_the_observable_subspace(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    for dof in rich_verdict["secondary_1"].degrees_of_freedom:
        filter_sigma = mekf.sigma_rad_in_direction(dof.direction)
        assert filter_sigma == pytest.approx(dof.crb_sigma_rad, rel=0.05, abs=0.0)


def test_covariance_stays_symmetric_and_positive_definite(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    assert mekf.joseph_symmetry < 1.0e-12
    assert mekf.min_eigenvalue > 0.0
    assert np.max(np.abs(mekf.covariance - mekf.covariance.T)) == 0.0


def test_prior_is_retained_in_the_unobservable_direction(single_axis, single_verdict):
    """The filter's own refusal: it never reduces the prior where there is no excitation."""
    mekf = recursive.run(single_axis.record, single_verdict, "secondary_1", store_history=False)
    retention = mekf.prior_retention()
    assert retention[2] > 20.0 * retention[0]
    assert mekf.inflation_ratio() > 20.0


def test_noiseless_single_axis_retains_the_prior_exactly(noiseless_single):
    """With no noise at all the filter learns literally nothing about the axis."""
    verdict = observability.assess(noiseless_single)
    mekf = recursive.run(noiseless_single, verdict, "secondary_1", store_history=False)
    retention = mekf.prior_retention()
    assert retention[-1] == pytest.approx(1.0, rel=1.0e-6, abs=0.0)
    axis = verdict["secondary_1"].degrees_of_freedom[-1].direction
    assert abs(float(np.dot(mekf.theta_rad, axis))) < 1.0e-9


def test_rich_record_reduces_the_prior_in_every_direction(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    assert np.all(mekf.prior_retention() < 0.01)


def test_history_shapes(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=True)
    assert mekf.history.shape == (len(rich.record), 3)
    assert mekf.sigma_history.shape == (len(rich.record), 3)
    assert np.max(np.abs(mekf.history[-1] - mekf.theta_rad)) < 1.0e-14


def test_history_is_empty_when_not_stored(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    assert mekf.history.shape == (0, 3)


def test_sigma_history_decreases_overall(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=True)
    assert float(np.max(mekf.sigma_history[-1])) < float(np.max(mekf.sigma_history[0]))


def test_process_noise_keeps_the_covariance_from_collapsing(rich, rich_verdict):
    """A thermally drifting mounting: a non-zero Q leaves a floor on the sigma."""
    rigid = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    drifting = recursive.run(
        rich.record,
        rich_verdict,
        "secondary_1",
        process_noise=1.0e-8,
        store_history=False,
    )
    best = rich_verdict["secondary_1"].degrees_of_freedom[0].direction
    assert drifting.sigma_rad_in_direction(best) > rigid.sigma_rad_in_direction(best)


def test_recursive_rejects_bad_parameters(rich, rich_verdict):
    with pytest.raises(ValueError, match="initial_sigma_rad"):
        recursive.run(rich.record, rich_verdict, "secondary_1", initial_sigma_rad=0.0)
    with pytest.raises(ValueError, match="process_noise"):
        recursive.run(rich.record, rich_verdict, "secondary_1", process_noise=-1.0)


def test_recursive_rejects_unknown_sensor(rich, rich_verdict):
    with pytest.raises(KeyError, match="unknown secondary"):
        recursive.run(rich.record, rich_verdict, "nope")


def test_sigma_in_direction_rejects_zero(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    with pytest.raises(ValueError, match="non-zero direction"):
        mekf.sigma_rad_in_direction(np.zeros(3))


def test_theta_mrad(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    assert np.allclose(mekf.theta_mrad, mekf.theta_rad * 1.0e3, atol=0.0, rtol=0.0)


def test_summary_puts_the_verdict_first(single_axis, single_verdict):
    mekf = recursive.run(single_axis.record, single_verdict, "secondary_1", store_history=False)
    lines = mekf.summary()
    text = "\n".join(lines)
    assert lines[0].startswith("sensor secondary_1")
    assert text.index("does NOT support") < text.index("RECURSIVE (MEKF) ESTIMATE")
    assert "prior retained" in text


def test_recovers_an_injected_misalignment_on_a_rich_record(rich, rich_verdict):
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    truth = rich.truth["secondary_1"]
    sensor = rich_verdict["secondary_1"]
    error = sensor.project_observable(mekf.theta_rad - truth)
    worst_crb = max(d.crb_sigma_rad for d in sensor.degrees_of_freedom)
    assert float(np.linalg.norm(error)) < 5.0 * worst_crb


def test_recursive_is_insensitive_to_a_weaker_prior(rich, rich_verdict):
    """Two priors differing by a factor of ten give the same answer to well inside the CRB."""
    tight = recursive.run(
        rich.record, rich_verdict, "secondary_1", initial_sigma_rad=0.01, store_history=False
    )
    loose = recursive.run(
        rich.record, rich_verdict, "secondary_1", initial_sigma_rad=0.1, store_history=False
    )
    crb = rich_verdict["secondary_1"].degrees_of_freedom[0].crb_sigma_rad
    assert float(np.linalg.norm(tight.theta_rad - loose.theta_rad)) < 0.05 * crb
