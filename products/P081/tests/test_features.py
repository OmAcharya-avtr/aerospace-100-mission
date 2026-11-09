"""The residual-signature features: count, determinism, substitutions, errors."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.features import FEATURE_NAMES, extract, extract_many, feature_notes
from twindiverge.plant import NOMINAL, simulate_with_sensitivities
from twindiverge.telemetry import TELEMETRY_SETS, measure


def test_there_are_sixteen_declared_features():
    assert len(FEATURE_NAMES) == 16
    assert len(set(FEATURE_NAMES)) == 16


def test_extract_returns_one_row_of_the_declared_width(short_set, short_twin):
    observed, _ = measure(short_set, NOMINAL, seed=5)
    row = extract(short_set, observed, short_twin)
    assert row.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(row))


def test_features_are_a_function_of_the_observation_alone(short_set, short_twin):
    first, _ = measure(short_set, NOMINAL, seed=5)
    assert np.array_equal(
        extract(short_set, first, short_twin), extract(short_set, first, short_twin)
    )


def test_a_zero_residual_gives_zero_features_except_the_peak_position(
    short_set, short_twin
):
    """With the observation equal to the twin, every feature that is a moment of
    the residual is exactly zero. The peak position is the index of the maximum
    of an all-zero array, which numpy reports as 0, and the ratio of two zero
    root-mean-squares is set to 0 by the declared substitution."""
    observed = np.concatenate([short_twin.rate_rad_s])
    row = extract(short_set, observed, short_twin)
    assert row == pytest.approx(np.zeros(16), rel=0.0, abs=0.0)


def test_extract_many_stacks(short_set, short_twin):
    rows = np.vstack([measure(short_set, NOMINAL, seed=seed)[0] for seed in (1, 2, 3)])
    assert extract_many(short_set, rows, short_twin).shape == (3, 16)


def test_the_coast_campaign_substitutes_the_driven_features(coast_set):
    notes = feature_notes(coast_set)
    assert len(notes) == 1
    assert "no driven segment" in notes[0]
    twin = simulate_with_sensitivities(NOMINAL, coast_set.campaign)
    observed, _ = measure(coast_set, NOMINAL, seed=5)
    row = extract(coast_set, observed, twin)
    assert float(row[FEATURE_NAMES.index("mean_driven")]) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )
    assert float(row[FEATURE_NAMES.index("driven_coast_rms_ratio")]) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_a_campaign_with_both_segments_substitutes_nothing(full_set):
    assert feature_notes(full_set) == ()


def test_extract_validates_its_input(short_set, short_twin, torque_set):
    from twindiverge.plant import CAMPAIGNS
    from twindiverge.telemetry import TelemetrySet

    with pytest.raises(ValueError, match="rows"):
        extract(short_set, np.zeros(3), short_twin)
    torque_only = TelemetrySet("torque_only", CAMPAIGNS["short"], ("torque",))
    with pytest.raises(ValueError, match="no rate channel"):
        extract(torque_only, np.zeros(torque_only.rows), short_twin)
    _ = torque_set


def test_the_rate_block_is_located_by_channel_order(torque_set):
    """A two-channel observation stacks rate then torque, and the features must
    read the rate block and not the first rows blindly. Measured by comparing
    against the single-channel set over the same campaign and seed."""
    twin = simulate_with_sensitivities(NOMINAL, torque_set.campaign)
    observed, _ = measure(torque_set, NOMINAL, seed=9)
    single = TELEMETRY_SETS["rate_short"]
    rate_only, _ = measure(single, NOMINAL, seed=9)
    width = torque_set.campaign.samples
    assert np.array_equal(observed[:width], rate_only)
    assert extract(torque_set, observed, twin) == pytest.approx(
        extract(single, rate_only, twin), rel=1e-15, abs=0.0
    )
