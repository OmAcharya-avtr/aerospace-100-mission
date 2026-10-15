"""The synthetic observation generator."""

from __future__ import annotations

import numpy as np
import pytest

from safemodetree import NOMINAL_LABEL, ObservationSet, make_observations


def test_shape_and_labels(fmea):
    observations = make_observations(fmea, n_per_fault=40, seed=7)
    assert observations.features.shape[1] == len(fmea.observability.channels)
    counts = observations.counts()
    assert counts[NOMINAL_LABEL] == 40 * len(fmea.fault_ids)
    for fault_id in fmea.fault_ids:
        assert counts[fault_id] == 40
    assert observations.n_samples == 2 * 40 * len(fmea.fault_ids)


def test_determinism_for_a_fixed_seed(fmea):
    first = make_observations(fmea, n_per_fault=25, seed=11)
    second = make_observations(fmea, n_per_fault=25, seed=11)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


def test_different_seeds_differ(fmea):
    first = make_observations(fmea, n_per_fault=25, seed=11)
    second = make_observations(fmea, n_per_fault=25, seed=12)
    assert not np.array_equal(first.features, second.features)


def test_class_means_track_the_declared_signature(fmea):
    observations = make_observations(fmea, n_per_fault=4000, noise_scale=1.0, seed=3)
    index = observations.channel_ids.index("bus_voltage")
    nominal = observations.features[observations.labels == NOMINAL_LABEL, index]
    faulted = observations.features[observations.labels == "bat_cell_short", index]
    # Declared: nominal 28.00 V, faulted 24.20 V, noise_sd 0.40 V. With 4000
    # samples the standard error of the mean is 0.40/sqrt(4000) = 0.0063 V, so
    # 0.05 V is an eight-sigma tolerance.
    assert nominal.mean() == pytest.approx(28.00, abs=0.05)
    assert faulted.mean() == pytest.approx(24.20, abs=0.05)


def test_ambiguity_class_members_have_the_same_distribution(fmea):
    observations = make_observations(fmea, n_per_fault=6000, noise_scale=1.0, seed=5)
    blinding = observations.features[observations.labels == "st_blinding"]
    latchup = observations.features[observations.labels == "st_latchup"]
    # Identical declared signatures, so the means agree to within sampling
    # noise on every channel. Tolerance is 0.1 of the largest declared noise_sd
    # (0.60 mN*m), far above the 0.008 standard error at n = 6000.
    assert np.allclose(blinding.mean(axis=0), latchup.mean(axis=0), atol=0.06)


def test_noise_scale_multiplies_the_spread(fmea):
    narrow = make_observations(fmea, n_per_fault=3000, noise_scale=1.0, seed=9)
    wide = make_observations(fmea, n_per_fault=3000, noise_scale=3.0, seed=9)
    index = narrow.channel_ids.index("attitude_residual")
    narrow_sd = narrow.features[narrow.labels == NOMINAL_LABEL, index].std()
    wide_sd = wide.features[wide.labels == NOMINAL_LABEL, index].std()
    assert wide_sd / narrow_sd == pytest.approx(3.0, rel=0.1)


def test_nominal_fraction_zero_gives_no_nominal_class(fmea):
    observations = make_observations(fmea, n_per_fault=10, nominal_fraction=0.0, seed=1)
    assert NOMINAL_LABEL not in observations.counts()
    assert not observations.is_nominal.any()
    assert observations.class_names == fmea.fault_ids


def test_observation_set_validates_its_own_shapes(fmea):
    observations = make_observations(fmea, n_per_fault=5, seed=1)
    with pytest.raises(ValueError, match="must be 2-D"):
        ObservationSet(
            features=np.zeros(3),
            labels=observations.labels[:3],
            channel_ids=observations.channel_ids,
            class_names=observations.class_names,
            noise_scale=1.0,
        )
    with pytest.raises(ValueError, match="rows and labels"):
        ObservationSet(
            features=np.zeros((3, 7)),
            labels=observations.labels[:2],
            channel_ids=observations.channel_ids,
            class_names=observations.class_names,
            noise_scale=1.0,
        )
    with pytest.raises(ValueError, match="columns and there are"):
        ObservationSet(
            features=np.zeros((3, 4)),
            labels=observations.labels[:3],
            channel_ids=observations.channel_ids,
            class_names=observations.class_names,
            noise_scale=1.0,
        )
