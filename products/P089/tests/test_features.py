"""The feature set, including the check that it carries the observability evidence."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import dataset, features, observability


def test_feature_count_and_names():
    assert len(features.FEATURE_NAMES) == 24
    assert len(set(features.FEATURE_NAMES)) == 24


def test_build_returns_the_declared_length(rich):
    vector = features.build(rich.record, "secondary_1")
    assert vector.shape == (24,)
    assert np.all(np.isfinite(vector))


def test_build_is_identical_with_and_without_a_precomputed_verdict(rich, rich_verdict):
    without = features.build(rich.record, "secondary_1")
    with_report = features.build(rich.record, "secondary_1", rich_verdict)
    assert np.max(np.abs(without - with_report)) < 1.0e-12


def test_features_carry_the_excitation_shares(single_axis, single_verdict):
    """The model is given the evidence that a degree of freedom is unsupported."""
    vector = features.build(single_axis.record, "secondary_1", single_verdict)
    index = features.FEATURE_NAMES.index("excitation_p0")
    assert vector[index] > 0.99
    assert vector[features.FEATURE_NAMES.index("excitation_p2")] < 0.01


def test_features_carry_the_dominant_axis(single_axis, single_verdict):
    vector = features.build(single_axis.record, "secondary_1", single_verdict)
    start = features.FEATURE_NAMES.index("dominant_axis_x")
    axis = vector[start : start + 3]
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0, rel=1.0e-9, abs=0.0)
    assert np.max(np.abs(axis - single_verdict.dominant_axis)) < 1.0e-12


def test_features_carry_the_entropy(rich, rich_verdict):
    vector = features.build(rich.record, "secondary_1", rich_verdict)
    index = features.FEATURE_NAMES.index("excitation_entropy")
    assert vector[index] == pytest.approx(rich_verdict.excitation_entropy, rel=1e-12, abs=0.0)


def test_normalised_gramian_trace_is_two(rich):
    """G / (W/s^2) = I - M and trace(M) = 1, so the trace of the normalised form is 2."""
    vector = features.build(rich.record, "secondary_1")
    diagonal = sum(
        vector[features.FEATURE_NAMES.index(name)] for name in ("gnorm_xx", "gnorm_yy", "gnorm_zz")
    )
    assert diagonal == pytest.approx(2.0, rel=1.0e-9, abs=0.0)


def test_information_scale_grows_with_record_length():
    short = features.build(dataset.generate(samples=200, seed=20).record, "secondary_1")
    long = features.build(dataset.generate(samples=2000, seed=20).record, "secondary_1")
    index = features.FEATURE_NAMES.index("log10_information")
    assert long[index] > short[index] + 0.9


def test_log10_samples_is_exact():
    generated = dataset.generate(samples=1000, seed=21)
    vector = features.build(generated.record, "secondary_1")
    assert vector[features.FEATURE_NAMES.index("log10_samples")] == pytest.approx(
        3.0, rel=1.0e-12, abs=0.0
    )


def test_duration_feature_matches_the_record():
    generated = dataset.generate(samples=500, rate_hz=50.0, seed=22)
    vector = features.build(generated.record, "secondary_1")
    assert vector[features.FEATURE_NAMES.index("duration_s")] == pytest.approx(
        generated.record.duration_s, rel=1.0e-12, abs=0.0
    )


def test_residual_statistics_are_scaled_by_the_pair_sigma(rich):
    """A record with no misalignment has a residual of about sqrt(2) pair sigmas per axis."""
    generated = dataset.generate(samples=1500, seed=23, theta_rad=np.zeros(3))
    vector = features.build(generated.record, "secondary_1")
    index = features.FEATURE_NAMES.index("residual_rms_scaled")
    # Three independent axes each at one pair sigma gives a vector RMS of sqrt(3).
    assert vector[index] == pytest.approx(np.sqrt(3.0), rel=0.1, abs=0.0)
    del rich


def test_b_scaled_responds_to_the_injected_misalignment():
    zero = features.build(
        dataset.generate(samples=1500, seed=24, theta_rad=np.zeros(3)).record, "secondary_1"
    )
    large = features.build(
        dataset.generate(
            samples=1500, seed=24, theta_rad=np.array([3.0e-3, 0.0, 0.0])
        ).record,
        "secondary_1",
    )
    index = features.FEATURE_NAMES.index("b_scaled_x")
    assert abs(large[index]) > abs(zero[index]) + 1.0e-4


def test_build_population_shapes():
    population = dataset.training_population(count=15, samples=80, seed=25)
    matrix, targets, profiles = features.build_population(population)
    assert matrix.shape == (15, 24)
    assert targets.shape == (15, 3)
    assert len(profiles) == 15
    assert set(profiles) <= set(dataset.PROFILES)


def test_build_population_rejects_empty():
    with pytest.raises(ValueError, match="empty population"):
        features.build_population([])


def test_build_population_covers_every_secondary():
    generated = dataset.generate(samples=80, n_secondaries=3, seed=26)
    matrix, targets, _ = features.build_population([generated])
    assert matrix.shape == (3, 24)
    assert targets.shape == (3, 3)


def test_features_differ_between_profiles():
    single = features.build(
        dataset.generate(profile="single_axis", samples=600, seed=27).record, "secondary_1"
    )
    isotropic = features.build(
        dataset.generate(profile="isotropic", samples=600, seed=27).record, "secondary_1"
    )
    assert float(np.max(np.abs(single - isotropic))) > 0.1


def test_features_rejects_unknown_sensor(rich):
    with pytest.raises(KeyError, match="unknown secondary"):
        features.build(rich.record, "nope")


def test_entropy_helper_matches_the_report(rich):
    shares, _, _ = observability.excitation_spectrum(rich.record)
    assert features._entropy(shares) == pytest.approx(
        observability.assess(rich.record).excitation_entropy, rel=1.0e-12, abs=0.0
    )
