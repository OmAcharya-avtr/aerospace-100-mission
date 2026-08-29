"""Tests for turbscope.synthetic."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turbscope.scintillometer import rytov_variance
from turbscope.synthetic import (
    LOG10_RYTOV_RANGE,
    PATH_LENGTH_RANGE_M,
    SCINT_WAVELENGTH_M,
    WAVE_TYPE,
    cn2_from_target_rytov,
    generate_scenarios,
    split_indices,
    synthesize_measurement,
)


def test_generate_scenarios_is_deterministic():
    a = generate_scenarios(50, seed=7)
    b = generate_scenarios(50, seed=7)
    assert [s.cn2_path for s in a] == [s.cn2_path for s in b]
    assert [s.path_length_m for s in a] == [s.path_length_m for s in b]


def test_generate_scenarios_different_seed_differs():
    a = generate_scenarios(50, seed=7)
    b = generate_scenarios(50, seed=8)
    assert [s.cn2_path for s in a] != [s.cn2_path for s in b]


def test_generate_scenarios_count_and_positivity():
    scenarios = generate_scenarios(30, seed=1)
    assert len(scenarios) == 30
    for s in scenarios:
        assert s.cn2_path > 0.0
        assert PATH_LENGTH_RANGE_M[0] <= s.path_length_m <= PATH_LENGTH_RANGE_M[1]
        assert 10.0 ** LOG10_RYTOV_RANGE[0] <= s.rytov_variance_true <= 10.0 ** LOG10_RYTOV_RANGE[1]


def test_generate_scenarios_rejects_non_positive_count():
    with pytest.raises(ValueError):
        generate_scenarios(0)


def test_cn2_from_target_rytov_matches_forward_model():
    target = 0.5
    length = 400.0
    cn2 = cn2_from_target_rytov(target, length)
    back = float(rytov_variance(cn2, length, SCINT_WAVELENGTH_M, WAVE_TYPE))
    assert back == pytest.approx(target, rel=1e-9)


def test_cn2_from_target_rytov_rejects_non_positive():
    with pytest.raises(ValueError):
        cn2_from_target_rytov(0.0, 400.0)
    with pytest.raises(ValueError):
        cn2_from_target_rytov(1.0, 0.0)


def test_synthesize_measurement_is_positive_and_reproducible():
    scenarios = generate_scenarios(5, seed=3)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1 = synthesize_measurement(scenarios[0], rng1)
    m2 = synthesize_measurement(scenarios[0], rng2)
    assert m1 == m2
    assert m1.sigma_i2_scint > 0.0
    assert m1.var_long_dimm > 0.0
    assert m1.var_trans_dimm > 0.0
    assert m1.path_length_m == scenarios[0].path_length_m


def test_synthesize_measurement_noise_changes_with_rng_state():
    scenarios = generate_scenarios(5, seed=3)
    rng = np.random.default_rng(0)
    m1 = synthesize_measurement(scenarios[0], rng)
    m2 = synthesize_measurement(scenarios[0], rng)  # rng advances between calls
    assert m1 != m2


def test_synthesize_measurement_mean_over_many_draws_is_near_truth():
    from turbscope.dimm import differential_variance
    from turbscope.scintillometer import scintillation_index_full

    scenarios = generate_scenarios(1, seed=11)
    sc = scenarios[0]
    rng = np.random.default_rng(123)
    draws = [synthesize_measurement(sc, rng) for _ in range(400)]
    mean_scint = np.mean([d.sigma_i2_scint for d in draws])
    r_var = rytov_variance(sc.cn2_path, sc.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE)
    true_scint = float(scintillation_index_full(r_var))
    # multiplicative zero-mean noise -> sample mean within a few % of truth at n=400
    assert mean_scint == pytest.approx(true_scint, rel=0.05)
    from turbscope.synthetic import APERTURE_DIAM_M, DIMM_WAVELENGTH_M, SEPARATION_M

    true_var_l = float(
        differential_variance(
            sc.cn2_path, sc.path_length_m, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
            "longitudinal",
        )
    )
    mean_var_l = np.mean([d.var_long_dimm for d in draws])
    assert mean_var_l == pytest.approx(true_var_l, rel=0.05)


def test_split_indices_is_disjoint_and_covers_all():
    train, test = split_indices(100, test_fraction=0.3, seed=1)
    assert set(train.tolist()) & set(test.tolist()) == set()
    assert set(train.tolist()) | set(test.tolist()) == set(range(100))
    assert len(test) == 30


def test_split_indices_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_indices(10, test_fraction=0.0)
    with pytest.raises(ValueError):
        split_indices(10, test_fraction=1.0)


def test_split_indices_rejects_too_small_n():
    with pytest.raises(ValueError):
        split_indices(1)


@given(st.integers(min_value=2, max_value=500))
def test_split_indices_sizes_sum_to_n(n):
    train, test = split_indices(n, test_fraction=0.25, seed=5)
    assert len(train) + len(test) == n
