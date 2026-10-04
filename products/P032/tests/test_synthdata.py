"""Synthetic dataset generator: determinism, shapes and structure."""

from __future__ import annotations

import numpy as np
import pytest

from constellink.synthdata import (
    FEATURE_NAMES,
    N_FEATURES,
    DatasetConfig,
    LinkDataset,
    default_optical_terminal,
    default_rf_terminal,
    default_stations,
    generate_dataset,
    planned_rates,
    stratify,
)


def test_feature_contract():
    assert N_FEATURES == len(FEATURE_NAMES) == 9
    assert FEATURE_NAMES[0] == "climatology_prior"
    assert FEATURE_NAMES[-2:] == ("is_optical", "is_ground")


def test_default_terminals_are_valid():
    assert default_rf_terminal().frequency_hz == pytest.approx(26.0e9)
    assert default_optical_terminal().wavelength_m == pytest.approx(1550e-9)
    assert default_optical_terminal().photons_per_bit > 0.0


def test_default_stations_are_distinct_and_valid():
    stations = default_stations()
    assert len({s.name for s in stations}) == len(stations)
    assert all(0.0 <= s.min_elevation_deg < 90.0 for s in stations)


def test_planned_rates_cover_all_four_link_classes():
    rates = planned_rates(DatasetConfig())
    assert set(rates) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(v > 0.0 for v in rates.values())
    # A ground leg is shorter than the 5000 km ISL limit, so its plan is a
    # higher rate for the same terminal.
    assert rates[(0, 1)] > rates[(0, 0)]
    assert rates[(1, 1)] > rates[(1, 0)]


def test_planned_rates_respond_to_backoff():
    a = planned_rates(DatasetConfig())
    b = planned_rates(DatasetConfig(plan_backoff_db=3.0))
    for key in a:
        assert b[key] == pytest.approx(a[key] / 10.0 ** 0.3, rel=1e-12)


def test_stratify_cross_product():
    el = np.array([90.0, 15.0, 30.0, 60.0])
    opt = np.array([0.0, 0.0, 1.0, 1.0])
    gnd = np.array([0.0, 1.0, 1.0, 1.0])
    s = stratify(el, opt, gnd)
    assert s.dtype.kind == "i"
    assert len(np.unique(s)) == 4
    # An ISL always lands in band 0 regardless of the elevation placeholder.
    assert s[0] == 0


def test_config_validation():
    for kwargs in ({"horizon_hours": 0.0}, {"step_s": 0.0},
                   {"plan_backoff_db": -1.0}, {"turbulence_sigma_db": -1.0},
                   {"obs_noise_scale": -1.0}, {"optical_fraction": 1.5}):
        with pytest.raises(ValueError):
            DatasetConfig(**kwargs)


def test_dataset_shapes_and_binary_labels(small_dataset):
    d = small_dataset
    assert d.x.shape == (len(d), N_FEATURES)
    assert set(np.unique(d.y)) <= {0, 1}
    assert d.stratum.shape == (len(d),)
    assert d.group.shape == (len(d),)
    assert d.feature_names == FEATURE_NAMES
    assert np.all(np.isfinite(d.x))


def test_dataset_is_deterministic_for_a_seed():
    cfg = DatasetConfig(horizon_hours=2.0, seed=42)
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.stratum, b.stratum)


def test_different_seeds_give_different_labels():
    a = generate_dataset(DatasetConfig(horizon_hours=2.0, seed=1))
    b = generate_dataset(DatasetConfig(horizon_hours=2.0, seed=2))
    # The geometry is identical (same constellation and horizon), so the row
    # count matches, but the latent weather draws differ.
    assert len(a) == len(b)
    assert not np.array_equal(a.y, b.y)


def test_dataset_has_both_link_kinds_and_both_terminals(small_dataset):
    d = small_dataset
    assert 0.0 < d.x[:, 8].mean() < 1.0      # some ground, some ISL
    assert 0.0 < d.x[:, 7].mean() < 1.0      # some optical, some RF


def test_dataset_label_is_not_degenerate(small_dataset):
    assert 0.05 < small_dataset.base_rate < 0.95


def test_isl_rows_carry_the_clear_sky_weather_sentinel(small_dataset):
    d = small_dataset
    isl = d.x[:, 8] == 0.0
    # ISL rows have latent tau 0 and visibility 100 km; the observations are
    # multiplicative, so tau stays exactly 0 and visibility stays large.
    assert np.all(d.x[isl, 4] == 0.0)
    assert np.all(d.x[isl, 5] > 10.0)


def test_climatology_prior_is_the_stratum_base_rate(small_dataset):
    d = small_dataset
    for sid in np.unique(d.stratum):
        m = d.stratum == sid
        assert np.allclose(d.x[m, 0], d.y[m].mean())


def test_dataset_rejects_bad_arrays():
    x = np.zeros((3, N_FEATURES))
    with pytest.raises(ValueError, match="columns"):
        LinkDataset(x=np.zeros((3, 2)), y=np.zeros(3), stratum=np.zeros(3),
                    group=np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="rows"):
        LinkDataset(x=x, y=np.zeros(2), stratum=np.zeros(3),
                    group=np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="binary"):
        LinkDataset(x=x, y=np.array([0, 1, 2]), stratum=np.zeros(3),
                    group=np.array(["a", "b", "c"]))


def test_group_labels_name_the_link(small_dataset):
    assert all("|" in g for g in small_dataset.group)
