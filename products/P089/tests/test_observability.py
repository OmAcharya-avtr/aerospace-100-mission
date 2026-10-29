"""The observability verdict: the Gramian, equation (10), the noise floor and the bracket."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import dataset
from aligncal.geometry import angle_between, skew
from aligncal.observability import (
    DEFAULT_FLOOR_FACTOR,
    DEFAULT_REQUIRED_SIGMA_RAD,
    assess,
    dominant_rotation_axis,
    eigenvalue_bracket,
    excitation_spectrum,
    gramian,
    noise_floor_eigenvalue,
    sensitivity_block,
)
from aligncal.record import AlignmentRecord, SensorSpec


def test_gramian_matches_its_definition_by_explicit_summation(rich):
    """G = (1/s^2) sum H^T H, computed by a loop rather than by the vectorised form."""
    record = rich.record
    name = record.names[0]
    weight = 1.0 / record.pair_sigma(name) ** 2
    accumulated = np.zeros((3, 3))
    for rate in record.reference_rates:
        matrix = skew(rate)
        accumulated += matrix.T @ matrix
    accumulated *= weight
    computed = gramian(record, name)
    assert np.max(np.abs(computed - accumulated)) < 1.0e-6 * float(np.max(np.abs(computed)))


def test_gramian_is_symmetric_and_positive_semidefinite(rich):
    matrix = gramian(rich.record, rich.record.names[0])
    assert np.max(np.abs(matrix - matrix.T)) == 0.0
    assert float(np.min(np.linalg.eigvalsh(matrix))) >= -1.0e-6 * float(np.max(np.abs(matrix)))


def test_sensitivity_block_is_the_skew_of_the_reference_rate(rich):
    block = sensitivity_block(rich.record)
    assert block.shape == (len(rich.record), 3, 3)
    assert np.array_equal(block[7], skew(rich.record.reference_rates[7]))


def test_excitation_shares_sum_to_one(rich):
    shares, _, _ = excitation_spectrum(rich.record)
    assert float(np.sum(shares)) == pytest.approx(1.0, rel=1.0e-12, abs=0.0)
    assert np.all(np.diff(shares) <= 1.0e-15)


def test_excitation_spectrum_rejects_a_motionless_record():
    times = np.arange(10, dtype=float)
    rates = np.zeros((10, 3))
    record = AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 1.0e-4),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", 1.0e-4)},
        secondary_rates={"s": rates},
    )
    with pytest.raises(ValueError, match="no motion at all"):
        excitation_spectrum(record)


def test_equation_ten_exactly_on_a_noiseless_record(noiseless_single):
    """lambda_i = (W / s^2) (1 - p_i), to rounding.

    ``eigvalsh`` returns ascending eigenvalues and ``excitation_spectrum``
    returns descending shares, and by (10) those two orders correspond: the
    LARGEST eigenvalue goes with the SMALLEST excitation share. Comparing them
    in the same order was the first draft of this test and it failed by a
    factor of 1; the pairing is pinned in tests/test_regressions.py.
    """
    record = noiseless_single
    name = "secondary_1"
    shares, _, energy = excitation_spectrum(record)
    scale = energy / record.pair_sigma(name) ** 2
    values = np.linalg.eigvalsh(gramian(record, name))
    predicted = scale * (1.0 - shares)
    assert np.max(np.abs(values - predicted)) < 1.0e-6 * scale


def test_equation_ten_on_every_profile():
    """The identity holds whatever the excitation, which is why it is the product's spine."""
    worst = 0.0
    for profile in dataset.PROFILES:
        generated = dataset.generate(profile=profile, samples=800, seed=99)
        record = generated.record
        name = record.names[0]
        shares, _, energy = excitation_spectrum(record)
        scale = energy / record.pair_sigma(name) ** 2
        values = np.linalg.eigvalsh(gramian(record, name))
        worst = max(worst, float(np.max(np.abs(values - scale * (1.0 - shares)))) / scale)
    assert worst < 1.0e-9


def test_noiseless_single_axis_gramian_is_rank_two(noiseless_single):
    """The exact-arithmetic statement: a single-axis rotation gives a singular Gramian."""
    matrix = gramian(noiseless_single, "secondary_1")
    values = np.linalg.eigvalsh(matrix)
    assert values[0] / values[-1] < 1.0e-12


def test_noiseless_single_axis_null_space_is_the_rotation_axis(noiseless_single):
    axis = dominant_rotation_axis(noiseless_single)
    matrix = gramian(noiseless_single, "secondary_1")
    assert float(np.linalg.norm(matrix @ axis)) < 1.0e-9 * float(np.max(np.abs(matrix)))
    expected = np.array([1.0, 2.0, -2.0]) / 3.0
    assert min(
        angle_between(axis, expected), angle_between(axis, -expected)
    ) < 1.0e-9


def test_dominant_axis_sign_is_stable(rich):
    axis = dominant_rotation_axis(rich.record)
    assert axis[int(np.argmax(np.abs(axis)))] > 0.0
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0, rel=1.0e-12, abs=0.0)


def test_noise_floor_scales_with_samples_and_sigma():
    """2 n sigma_A^2 / s^2, hand-computed with sigma_A = sigma_j so s^2 = 2 sigma^2."""
    generated = dataset.generate(samples=1000, noise_sigma=1.0e-4, seed=1)
    floor = noise_floor_eigenvalue(generated.record, "secondary_1")
    assert floor == pytest.approx(1000.0, rel=1.0e-12, abs=0.0)
    doubled = dataset.generate(samples=2000, noise_sigma=1.0e-4, seed=1)
    assert noise_floor_eigenvalue(doubled.record, "secondary_1") == pytest.approx(
        2000.0, rel=1.0e-12, abs=0.0
    )


def test_single_axis_record_is_reported_rank_deficient(single_verdict):
    sensor = single_verdict["secondary_1"]
    assert sensor.rank == 2
    assert sensor.rank_deficient
    assert len(sensor.unobservable_directions) == 1


def test_rich_record_is_reported_full_rank(rich_verdict):
    sensor = rich_verdict["secondary_1"]
    assert sensor.rank == 3
    assert not sensor.rank_deficient
    assert sensor.unobservable_directions == ()


def test_two_axis_excitation_is_enough_for_all_three_degrees_of_freedom():
    """Rotation about two independent axes observes all three alignment components."""
    generated = dataset.generate(profile="two_axis", samples=1500, seed=31)
    sensor = assess(generated.record)["secondary_1"]
    assert sensor.rank == 3


def test_unobservable_direction_is_the_rotation_axis(single_verdict):
    sensor = single_verdict["secondary_1"]
    assert sensor.unobservable_axis_angle_deg < 0.01


def test_unobservable_direction_is_the_rotation_axis_across_seeds():
    worst = 0.0
    for seed in range(40):
        generated = dataset.generate(profile="single_axis", samples=600, seed=5000 + seed)
        sensor = assess(generated.record)["secondary_1"]
        assert sensor.rank_deficient
        worst = max(worst, sensor.unobservable_axis_angle_deg)
    assert worst < 0.5


def test_crb_is_the_inverse_square_root_of_the_eigenvalue(rich_verdict):
    for dof in rich_verdict["secondary_1"].degrees_of_freedom:
        assert dof.crb_sigma_rad == pytest.approx(
            dof.eigenvalue**-0.5, rel=1.0e-12, abs=0.0
        )


def test_crb_is_infinite_on_an_unobservable_direction(single_verdict):
    worst = single_verdict["secondary_1"].degrees_of_freedom[-1]
    assert not np.isfinite(worst.crb_sigma_rad)
    assert not np.isfinite(worst.crb_sigma_arcsec)


def test_arcsec_conversion():
    generated = dataset.generate(samples=500, seed=2)
    dof = assess(generated.record)["secondary_1"].degrees_of_freedom[0]
    assert dof.crb_sigma_arcsec == pytest.approx(
        dof.crb_sigma_rad * 206264.80624709636, rel=1.0e-10, abs=0.0
    )


def test_marginal_verdict_appears_when_the_requirement_is_tightened(rich):
    """The same record, two declared requirements, two different verdicts."""
    loose = assess(rich.record, required_sigma_rad=1.0e-2)["secondary_1"]
    tight = assess(rich.record, required_sigma_rad=1.0e-6)["secondary_1"]
    assert all(d.verdict == "observable" for d in loose.degrees_of_freedom)
    assert all(d.verdict == "marginal" for d in tight.degrees_of_freedom)
    assert tight.rank == 3  # a marginal direction still clears the noise floor
    assert len(tight.marginal_directions) == 3


def test_floor_factor_controls_the_rank_verdict(single_axis):
    """Below a factor of 1 the noise floor stops protecting anything."""
    protected = assess(single_axis.record, floor_factor=4.0)["secondary_1"]
    unprotected = assess(single_axis.record, floor_factor=1.0)["secondary_1"]
    assert protected.rank == 2
    assert unprotected.rank in (2, 3)


def test_assess_rejects_a_floor_factor_below_one(rich):
    with pytest.raises(ValueError, match="at least 1"):
        assess(rich.record, floor_factor=0.5)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_assess_rejects_an_unusable_requirement(rich, value):
    with pytest.raises(ValueError, match="required_sigma_rad"):
        assess(rich.record, required_sigma_rad=value)


def test_bracket_contains_the_smallest_eigenvalue_on_every_profile():
    """Equation (13), checked against eigh on every declared profile and both noises."""
    for profile in dataset.PROFILES:
        for sigma in (1.0e-5, 5.0e-4):
            generated = dataset.generate(
                profile=profile, samples=700, noise_sigma=sigma, seed=404
            )
            matrix = gramian(generated.record, "secondary_1")
            lower, upper = eigenvalue_bracket(matrix)
            smallest = float(np.min(np.linalg.eigvalsh(matrix)))
            assert lower <= smallest * (1.0 + 1.0e-9) + 1.0e-9
            assert smallest <= upper * (1.0 + 1.0e-9) + 1.0e-9


def test_bracket_is_reported_consistent(rich_verdict, single_verdict):
    assert rich_verdict["secondary_1"].bracket_consistent
    assert single_verdict["secondary_1"].bracket_consistent


def test_bracket_hand_computed_on_a_diagonal_matrix():
    """For a diagonal matrix Gershgorin is exact: both bounds are the smallest entry."""
    lower, upper = eigenvalue_bracket(np.diag([3.0, 7.0, 11.0]))
    assert lower == 3.0
    assert upper == 3.0


def test_bracket_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\) matrix"):
        eigenvalue_bracket(np.zeros((2, 2)))


def test_projections_sum_to_the_original_vector(single_verdict):
    sensor = single_verdict["secondary_1"]
    rng = np.random.default_rng(12)
    for _ in range(20):
        vector = rng.normal(size=3)
        total = sensor.project_observable(vector) + sensor.project_unobservable(vector)
        assert np.max(np.abs(total - vector)) < 1.0e-14


def test_unobservable_projection_is_along_the_rotation_axis(single_verdict):
    sensor = single_verdict["secondary_1"]
    rng = np.random.default_rng(13)
    axis = sensor.degrees_of_freedom[-1].direction
    for _ in range(20):
        component = sensor.project_unobservable(rng.normal(size=3))
        magnitude = float(np.linalg.norm(component))
        if magnitude > 1.0e-12:
            assert min(
                angle_between(component, axis), angle_between(component, -axis)
            ) < 1.0e-8


def test_projection_is_a_projection(rich_verdict):
    sensor = rich_verdict["secondary_1"]
    rng = np.random.default_rng(14)
    for _ in range(10):
        vector = rng.normal(size=3)
        once = sensor.project_observable(vector)
        twice = sensor.project_observable(once)
        assert np.max(np.abs(once - twice)) < 1.0e-13


def test_axis_crb_is_finite_when_everything_is_observable(rich_verdict):
    sigmas = rich_verdict["secondary_1"].axis_crb_sigma_rad()
    assert np.all(np.isfinite(sigmas))
    assert np.all(sigmas > 0.0)


def test_axis_crb_is_infinite_when_a_direction_is_withheld(single_verdict):
    """An axis with any projection on an unobservable direction has no finite bound."""
    sigmas = single_verdict["secondary_1"].axis_crb_sigma_rad()
    assert not np.all(np.isfinite(sigmas))


def test_observable_mask_and_basis_are_consistent(single_verdict):
    sensor = single_verdict["secondary_1"]
    assert sensor.observable_mask.tolist() == [True, True, False]
    basis = sensor.basis
    assert np.max(np.abs(basis.T @ basis - np.eye(3))) < 1.0e-12


def test_excitation_entropy_bounds():
    """0 bits for a single axis, log2(3) = 1.584963 bits for isotropic motion."""
    single = assess(dataset.generate(profile="single_axis", samples=900, seed=8).record)
    isotropic = assess(dataset.generate(profile="isotropic", samples=900, seed=8).record)
    assert single.excitation_entropy < 0.01
    assert isotropic.excitation_entropy > 1.4
    assert isotropic.excitation_entropy <= np.log2(3.0) + 1.0e-12


def test_report_indexing_and_errors(rich_verdict):
    assert rich_verdict["secondary_1"].name == "secondary_1"
    with pytest.raises(KeyError, match="unknown secondary"):
        rich_verdict["nope"]


def test_report_covers_every_secondary(multi):
    verdict = assess(multi.record)
    assert sorted(verdict.sensors) == ["secondary_1", "secondary_2", "secondary_3"]
    assert not verdict.any_rank_deficient


def test_any_rank_deficient_is_true_for_a_single_axis_record(single_verdict):
    assert single_verdict.any_rank_deficient


def test_summary_puts_the_unobservable_direction_before_the_estimate(single_verdict):
    lines = single_verdict.summary()
    text = "\n".join(lines)
    assert lines[0].startswith("OBSERVABILITY VERDICT")
    assert "does NOT support" in text
    assert text.index("does NOT support") < text.index("per-body-axis")


def test_describe_names_the_axis(single_verdict):
    description = single_verdict["secondary_1"].degrees_of_freedom[-1].describe()
    assert "unobservable" in description
    assert "unbounded" in description
    assert "axis" in description


def test_defaults_are_the_documented_values():
    assert DEFAULT_FLOOR_FACTOR == 4.0
    assert DEFAULT_REQUIRED_SIGMA_RAD == 1.0e-3
