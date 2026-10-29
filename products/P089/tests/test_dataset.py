"""The generator: determinism, the declared profiles, and the injected truth."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import dataset, observability
from aligncal.geometry import angle_between, rotation_from_vector


def test_profiles_are_the_declared_five():
    assert sorted(dataset.PROFILES) == [
        "isotropic",
        "near_single_axis",
        "single_axis",
        "three_axis",
        "two_axis",
    ]


def test_generation_is_deterministic():
    first = dataset.generate(samples=300, seed=4242)
    second = dataset.generate(samples=300, seed=4242)
    assert np.array_equal(first.record.reference_rates, second.record.reference_rates)
    assert np.array_equal(
        first.record.secondary_rates["secondary_1"],
        second.record.secondary_rates["secondary_1"],
    )
    assert np.array_equal(first.truth["secondary_1"], second.truth["secondary_1"])


def test_different_seeds_give_different_records():
    first = dataset.generate(samples=300, seed=1)
    second = dataset.generate(samples=300, seed=2)
    assert not np.array_equal(first.record.reference_rates, second.record.reference_rates)
    assert not np.array_equal(first.truth["secondary_1"], second.truth["secondary_1"])


def test_rate_profile_single_axis_is_rank_one():
    times = np.arange(400, dtype=float) / 50.0
    rates = dataset.rate_profile("single_axis", times, seed=3)
    moment = rates.T @ rates
    values = np.linalg.eigvalsh(moment)
    assert values[0] / values[-1] < 1.0e-15
    assert values[1] / values[-1] < 1.0e-15


def test_rate_profile_two_axis_is_rank_two():
    times = np.arange(400, dtype=float) / 50.0
    rates = dataset.rate_profile("two_axis", times, seed=3)
    values = np.linalg.eigvalsh(rates.T @ rates)
    # The third axis is unexcited in exact arithmetic. Its measured eigenvalue
    # is not exactly zero but at the rounding level of the rotation that places
    # the excitation frame, which is why the bound here is 1e-15 and not 1e-20.
    assert values[0] / values[-1] < 1.0e-15
    assert values[1] / values[-1] > 0.1


def test_rate_profile_three_axis_is_rank_three():
    times = np.arange(400, dtype=float) / 50.0
    rates = dataset.rate_profile("three_axis", times, seed=3)
    values = np.linalg.eigvalsh(rates.T @ rates)
    assert values[0] / values[-1] > 0.05


def test_rate_profile_amplitude_is_respected():
    times = np.arange(2000, dtype=float) / 100.0
    rates = dataset.rate_profile("single_axis", times, amplitude=0.07, seed=5)
    assert float(np.max(np.linalg.norm(rates, axis=1))) == pytest.approx(
        0.07, rel=0.01, abs=0.0
    )


def test_rate_profile_axes_are_not_body_aligned():
    """A rotated excitation frame, so no result depends on the motion lying along an axis."""
    times = np.arange(500, dtype=float) / 50.0
    axis = dataset.rate_profile("single_axis", times, seed=11)[np.argmax(
        np.linalg.norm(dataset.rate_profile("single_axis", times, seed=11), axis=1)
    )]
    axis = axis / np.linalg.norm(axis)
    angles = [np.degrees(angle_between(axis, column)) for column in np.eye(3)]
    assert min(min(a, 180.0 - a) for a in angles) > 5.0


def test_rate_profile_near_single_axis_leak_controls_the_second_eigenvalue():
    times = np.arange(2000, dtype=float) / 100.0
    small = dataset.rate_profile("near_single_axis", times, seed=7, leak=0.01)
    large = dataset.rate_profile("near_single_axis", times, seed=7, leak=0.2)
    small_values = np.linalg.eigvalsh(small.T @ small)
    large_values = np.linalg.eigvalsh(large.T @ large)
    assert large_values[1] / large_values[2] > small_values[1] / small_values[2]


def test_rate_profile_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown profile"):
        dataset.rate_profile("spiral", np.arange(10.0))


@pytest.mark.parametrize("amplitude", [0.0, -1.0, np.nan])
def test_rate_profile_rejects_bad_amplitude(amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        dataset.rate_profile("single_axis", np.arange(10.0), amplitude=amplitude)


@pytest.mark.parametrize("leak", [0.0, -0.1, 1.5, np.nan])
def test_rate_profile_rejects_bad_leak(leak):
    with pytest.raises(ValueError, match="leak"):
        dataset.rate_profile("near_single_axis", np.arange(10.0), leak=leak)


def test_generated_record_is_a_valid_record():
    generated = dataset.generate(samples=250, n_secondaries=2, seed=9)
    assert len(generated.record) == 250
    assert generated.record.names == ("secondary_1", "secondary_2")
    assert set(generated.truth) == {"secondary_1", "secondary_2"}


def test_generated_truth_respects_the_magnitude_bound():
    for seed in range(30):
        generated = dataset.generate(samples=60, theta_max_rad=1.5e-3, seed=seed)
        assert float(np.linalg.norm(generated.truth["secondary_1"])) <= 1.5e-3 + 1.0e-18


def test_generated_mounting_is_a_rotation():
    from aligncal.geometry import is_rotation

    generated = dataset.generate(samples=60, n_secondaries=3, seed=12)
    for name in generated.record.names:
        assert is_rotation(generated.record.secondaries[name].mounting, tolerance=1.0e-10)


def test_secondaries_have_different_mountings_and_misalignments():
    generated = dataset.generate(samples=60, n_secondaries=3, seed=13)
    mountings = [generated.record.secondaries[n].mounting for n in generated.record.names]
    assert float(np.max(np.abs(mountings[0] - mountings[1]))) > 1.0e-3
    thetas = np.vstack([generated.truth[n] for n in generated.record.names])
    assert float(np.min(np.std(thetas, axis=0))) > 0.0


def test_explicit_theta_is_injected_exactly():
    requested = np.array([1.0e-3, -2.0e-3, 0.5e-3])
    generated = dataset.generate(samples=60, n_secondaries=2, seed=14, theta_rad=requested)
    for name in generated.record.names:
        assert np.array_equal(generated.truth[name], requested)


def test_explicit_theta_is_rejected_when_malformed():
    with pytest.raises(ValueError, match="3 components"):
        dataset.generate(samples=60, theta_rad=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        dataset.generate(samples=60, theta_rad=np.array([np.nan, 0.0, 0.0]))


def test_truth_component_projects_onto_a_direction():
    requested = np.array([1.0e-3, 0.0, 0.0])
    generated = dataset.generate(samples=60, seed=15, theta_rad=requested)
    assert generated.truth_component("secondary_1", np.array([1.0, 0.0, 0.0])) == pytest.approx(
        1.0e-3, rel=1.0e-12, abs=0.0
    )
    assert abs(
        generated.truth_component("secondary_1", np.array([0.0, 1.0, 0.0]))
    ) < 1.0e-18


def test_true_rates_are_noiseless_and_the_measured_ones_are_not():
    generated = dataset.generate(samples=400, noise_sigma=1.0e-4, seed=16)
    difference = generated.record.reference_rates - generated.true_rates
    assert float(np.std(difference)) == pytest.approx(1.0e-4, rel=0.15, abs=0.0)
    assert float(np.std(generated.true_rates)) > 0.0


def test_exact_rotation_is_used_not_its_linearisation():
    """The generated secondary rate carries the full exp(-[theta]x), to rounding."""
    theta = np.array([0.05, -0.03, 0.02])
    generated = dataset.generate(
        samples=200, noise_sigma=1.0e-12, seed=17, theta_rad=theta
    )
    mounting = generated.record.secondaries["secondary_1"].mounting
    expected = generated.true_rates @ (mounting @ rotation_from_vector(-theta)).T
    assert float(np.max(np.abs(generated.record.secondary_rates["secondary_1"] - expected))) < 1e-10


def test_generate_rejects_bad_arguments():
    with pytest.raises(ValueError, match="at least 2 samples"):
        dataset.generate(samples=1)
    with pytest.raises(ValueError, match="rate_hz"):
        dataset.generate(samples=10, rate_hz=0.0)
    with pytest.raises(ValueError, match="noise_sigma"):
        dataset.generate(samples=10, noise_sigma=0.0)
    with pytest.raises(ValueError, match="theta_max_rad"):
        dataset.generate(samples=10, theta_max_rad=-1.0)
    with pytest.raises(ValueError, match="at least 1 secondary"):
        dataset.generate(samples=10, n_secondaries=0)


def test_provenance_records_the_generating_call():
    generated = dataset.generate(samples=50, seed=18)
    assert "aligncal.dataset.generate" in generated.record.provenance
    assert "seed=18" in generated.record.provenance


def test_provenance_notes_an_explicit_theta():
    generated = dataset.generate(samples=50, seed=18, theta_rad=np.zeros(3))
    assert "theta_rad=declared" in generated.record.provenance


def test_training_population_is_deterministic():
    first = dataset.training_population(count=12, samples=60, seed=55)
    second = dataset.training_population(count=12, samples=60, seed=55)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.record.reference_rates, b.record.reference_rates)
        assert np.array_equal(a.truth["secondary_1"], b.truth["secondary_1"])


def test_training_population_mixes_profiles():
    population = dataset.training_population(count=80, samples=60, seed=56)
    assert len({generated.profile for generated in population}) >= 3


def test_training_population_includes_rank_deficient_records():
    """The learned estimator is trained on data that contains unobservable directions."""
    population = dataset.training_population(count=60, samples=120, seed=57)
    deficient = sum(
        observability.assess(g.record)["secondary_1"].rank_deficient for g in population
    )
    assert deficient > 0


def test_training_population_spans_the_declared_ranges():
    population = dataset.training_population(count=120, samples=60, seed=58)
    amplitudes = np.array([g.amplitude for g in population])
    noises = np.array([g.noise_sigma for g in population])
    assert float(np.min(amplitudes)) >= 0.01 - 1.0e-12
    assert float(np.max(amplitudes)) <= 0.08 + 1.0e-12
    assert float(np.min(noises)) >= 5.0e-5 - 1.0e-18
    assert float(np.max(noises)) <= 8.0e-4 + 1.0e-18
    assert float(np.max(amplitudes) / np.min(amplitudes)) > 3.0


def test_training_population_rejects_bad_arguments():
    with pytest.raises(ValueError, match="count >= 1"):
        dataset.training_population(count=0, seed=1)
    with pytest.raises(ValueError, match="at least one profile"):
        dataset.training_population(count=3, seed=1, profiles=())
    with pytest.raises(ValueError, match="unknown profiles"):
        dataset.training_population(count=3, seed=1, profiles=("helix",))
    with pytest.raises(ValueError, match="amplitude_range"):
        dataset.training_population(count=3, seed=1, amplitude_range=(0.1, 0.01))
    with pytest.raises(ValueError, match="noise_range"):
        dataset.training_population(count=3, seed=1, noise_range=(0.0, 1.0))
