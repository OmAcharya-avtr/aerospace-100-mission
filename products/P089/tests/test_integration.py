"""End-to-end: the verdict gates the estimate, the three estimators are consistent.

These tests exercise the whole path -- generate, assess, batch, recursive,
features, learned, score -- on the same records, because every module above is
unit-tested in isolation and nothing there checks that the pieces agree.
"""

from __future__ import annotations

import numpy as np
import pytest

import aligncal
from aligncal import batch, benchmark, dataset, features, learned, observability, recursive


@pytest.fixture(scope="module")
def trained():
    population = dataset.training_population(count=200, samples=120, seed=61)
    matrix, targets, _ = features.build_population(population)
    model = learned.LearnedEstimator(
        seed=20261029, **learned.CONFIGURATIONS["bagged_ridge"]
    ).fit(matrix, targets)
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    return model, trivial


def test_the_verdict_is_required_to_produce_an_estimate(rich):
    """Every estimator's signature takes the verdict. Omitting it is a TypeError."""
    with pytest.raises(TypeError):
        batch.solve(rich.record, "secondary_1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        recursive.run(rich.record, "secondary_1")  # type: ignore[call-arg]


def test_result_field_order_puts_observability_first():
    """Structural, not cosmetic: the dataclass field order is part of the contract."""
    from dataclasses import fields

    assert [f.name for f in fields(aligncal.AlignmentResult)][0] == "observability"
    assert [f.name for f in fields(batch.BatchSolution)][0] == "observability"
    assert [f.name for f in fields(recursive.RecursiveSolution)][0] == "observability"


def test_full_pipeline_on_a_rich_record_recovers_the_truth(rich):
    result = aligncal.align(rich.record, with_recursive=True)
    sensor = result.observability["secondary_1"]
    truth = rich.truth["secondary_1"]
    worst_crb = max(d.crb_sigma_rad for d in sensor.degrees_of_freedom)
    for solution in (result.batch["secondary_1"], result.recursive["secondary_1"]):
        error = float(np.linalg.norm(solution.theta_rad - truth))
        assert error < 5.0 * worst_crb


def test_full_pipeline_on_a_single_axis_record_withholds_and_says_so(single_axis):
    result = aligncal.align(single_axis.record, with_recursive=True)
    sensor = result.observability["secondary_1"]
    assert sensor.rank_deficient
    batch_solution = result.batch["secondary_1"]
    assert batch_solution.withheld == (2,)
    axis = sensor.degrees_of_freedom[2].direction
    assert not np.isfinite(batch_solution.sigma_rad_in_direction(axis))
    # And the observable part is still recovered.
    truth = single_axis.truth["secondary_1"]
    observable_error, _ = benchmark.subspace_split(batch_solution.theta_rad - truth, sensor)
    best = sensor.degrees_of_freedom[0].crb_sigma_rad
    assert float(np.linalg.norm(observable_error)) < 6.0 * best


def test_scored_errors_agree_with_the_solutions():
    sensors = []
    errors = []
    for seed in range(15):
        generated = dataset.generate(samples=600, seed=7000 + seed)
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        solution = batch.solve(generated.record, verdict, "secondary_1")
        errors.append(solution.theta_rad - generated.truth["secondary_1"])
    table = benchmark.score("batch", np.array(errors), sensors)
    assert table.records == 15
    assert table.rmse_unobservable_mrad < 1.0e-6  # every record is full rank here
    assert table.rmse_observable_mrad == pytest.approx(
        table.rmse_total_mrad, rel=1.0e-6, abs=0.0
    )


def test_learned_and_analytic_disagree_about_a_withheld_direction(trained):
    """The product's headline, end to end on one record.

    The analytic estimator withholds the direction and reports an infinite
    bound; the learned estimator reports a number and a finite interval that
    does not contain the truth.
    """
    model, _ = trained
    probe = dataset.generate(profile="single_axis", samples=120, seed=313131)
    axis = observability.dominant_rotation_axis(probe.record)
    generated = dataset.generate(
        profile="single_axis", samples=120, seed=313131, theta_rad=2.0e-3 * axis
    )
    verdict = observability.assess(generated.record)
    sensor = verdict["secondary_1"]
    worst = sensor.degrees_of_freedom[-1].direction
    truth = float(np.dot(generated.truth["secondary_1"], worst))

    analytic = batch.solve(generated.record, verdict, "secondary_1")
    assert abs(float(np.dot(analytic.theta_rad, worst))) < 1.0e-15
    assert not np.isfinite(analytic.sigma_rad_in_direction(worst))

    prediction = model.predict(features.build(generated.record, "secondary_1", verdict)[None, :])
    claim = float(prediction.component_in_direction(worst[None, :])[0])
    sigma = float(prediction.sigma_in_direction(worst[None, :])[0])
    assert np.isfinite(sigma) and sigma > 0.0
    assert abs(claim - truth) > 3.0 * sigma


def test_learned_beats_the_trivial_baseline_but_not_the_map_baseline(trained):
    """Measured on a held-out short-record population. Both halves are the result.

    This is the product's benchmark conclusion as a test. If a future change
    makes the learned estimator beat the regularised analytic baseline, the
    second assertion fails and the README has to be re-measured -- which is the
    correct consequence.
    """
    model, trivial = trained
    held_out = dataset.training_population(count=100, samples=120, seed=424242)
    matrix, targets, _ = features.build_population(held_out)
    sensors = []
    map_errors = []
    for generated in held_out:
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        map_errors.append(
            batch.solve_map(
                generated.record, verdict, "secondary_1", prior_sigma_rad=1.0e-3
            ).theta_rad
            - generated.truth["secondary_1"]
        )
    learned_table = benchmark.score(
        "learned", model.predict(matrix).theta_rad - targets, sensors
    )
    trivial_table = benchmark.score(
        "trivial", trivial.predict(matrix).theta_rad - targets, sensors
    )
    map_table = benchmark.score("map", np.array(map_errors), sensors)
    assert learned_table.rmse_observable_mrad < trivial_table.rmse_observable_mrad
    assert map_table.rmse_observable_mrad < learned_table.rmse_observable_mrad


def test_unbiased_batch_loses_to_the_map_baseline_on_short_records():
    """The specified Baseline 1 is the worst estimator here on 120-sample records.

    That is not a defect in the implementation: the unbiased maximum-likelihood
    solution has a variance set by the Cramer-Rao bound, and on a record this
    short that bound is comparable to the spread of the misalignment population
    itself. It is reported as measured and the regularised variant is reported
    beside it.
    """
    held_out = dataset.training_population(count=80, samples=120, seed=515151)
    sensors = []
    plain = []
    regularised = []
    for generated in held_out:
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        truth = generated.truth["secondary_1"]
        plain.append(batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth)
        regularised.append(
            batch.solve_map(
                generated.record, verdict, "secondary_1", prior_sigma_rad=1.0e-3
            ).theta_rad
            - truth
        )
    plain_table = benchmark.score("batch", np.array(plain), sensors)
    map_table = benchmark.score("map", np.array(regularised), sensors)
    assert map_table.rmse_total_mrad < plain_table.rmse_total_mrad


def test_long_records_reverse_that_conclusion():
    """With enough excitation the unbiased solution is the better one, as it should be.

    The prior that helps on a short record hurts on a long one, because the
    likelihood is then sharp and the shrinkage is pure bias. Measured on
    2000-sample rich records.
    """
    sensors = []
    plain = []
    regularised = []
    for seed in range(12):
        generated = dataset.generate(profile="three_axis", samples=2000, seed=9100 + seed)
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        truth = generated.truth["secondary_1"]
        plain.append(batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth)
        regularised.append(
            batch.solve_map(
                generated.record, verdict, "secondary_1", prior_sigma_rad=1.0e-4
            ).theta_rad
            - truth
        )
    plain_table = benchmark.score("batch", np.array(plain), sensors)
    map_table = benchmark.score("map", np.array(regularised), sensors)
    assert plain_table.rmse_total_mrad < map_table.rmse_total_mrad


def test_recursive_matches_batch_and_the_prior_is_not_what_separates_them():
    """The two solutions agree, and the reason they do not agree exactly is measured.

    The prior-shrinkage argument predicts that along degree of freedom i the
    recursive estimate is the batch estimate scaled by
    ``lambda_i / (lambda_i + 1 / p0^2)``, so the difference should fall by a
    factor of 100 when the prior one-sigma is tightened from 0.1 rad to 0.01
    rad. **It does not.** Measured here: the worst difference is 0.00975 of the
    Cramer-Rao one-sigma at ``p0 = 0.1`` and 0.00437 at ``p0 = 0.01``, a factor
    of 2.2 rather than 100. So the residual disagreement between the batch and
    the recursive solution is dominated by the sequential re-linearisation of
    (20) about a moving estimate, not by the filter's prior, and quoting the
    shrinkage figure as the expected agreement would be wrong by three orders
    of magnitude. The agreement itself -- under 2 per cent of the bound -- is
    the number this product reports.
    """
    worst = {0.1: 0.0, 0.01: 0.0}
    for seed in range(8):
        generated = dataset.generate(profile="three_axis", samples=1200, seed=6200 + seed)
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        batch_solution = batch.solve(generated.record, verdict, "secondary_1")
        for prior in (0.1, 0.01):
            mekf = recursive.run(
                generated.record,
                verdict,
                "secondary_1",
                initial_sigma_rad=prior,
                store_history=False,
            )
            for dof in sensor.degrees_of_freedom:
                difference = abs(
                    float(np.dot(mekf.theta_rad - batch_solution.theta_rad, dof.direction))
                )
                worst[prior] = max(worst[prior], difference / dof.crb_sigma_rad)
    assert worst[0.1] < 0.02
    assert worst[0.01] < 0.02
    # If the prior were the cause, tightening it by 10 would cut the difference
    # by 100. The measured ratio is near 2, so it is not the cause.
    assert worst[0.1] / worst[0.01] < 10.0
