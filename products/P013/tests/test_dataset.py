"""Tests for turbscope.dataset."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope.dataset import FEATURE_NAMES, build_table, generate_default_scenarios, grouped_split


def test_build_table_shape_and_columns():
    scenarios = generate_default_scenarios(20, seed=1)
    x, y, groups = build_table(scenarios, n_realisations=3, seed=5)
    assert x.shape == (60, len(FEATURE_NAMES))
    assert y.shape == (60,)
    assert groups.shape == (60,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_build_table_groups_repeat_per_scenario():
    scenarios = generate_default_scenarios(10, seed=1)
    _, _, groups = build_table(scenarios, n_realisations=4, seed=5)
    counts = np.bincount(groups)
    assert np.all(counts == 4)
    assert len(counts) == 10


def test_build_table_target_matches_scenario_truth():
    scenarios = generate_default_scenarios(5, seed=2)
    x, y, groups = build_table(scenarios, n_realisations=2, seed=1)
    for gi, sc in enumerate(scenarios):
        rows = np.where(groups == gi)[0]
        for r in rows:
            assert y[r] == pytest.approx(np.log10(sc.cn2_path))


def test_build_table_is_deterministic():
    scenarios = generate_default_scenarios(8, seed=3)
    x1, y1, g1 = build_table(scenarios, n_realisations=2, seed=42)
    x2, y2, g2 = build_table(scenarios, n_realisations=2, seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(g1, g2)


def test_build_table_different_noise_seed_differs():
    scenarios = generate_default_scenarios(8, seed=3)
    x1, _, _ = build_table(scenarios, n_realisations=2, seed=42)
    x2, _, _ = build_table(scenarios, n_realisations=2, seed=43)
    assert not np.allclose(x1, x2)


def test_build_table_rejects_empty_scenarios():
    with pytest.raises(ValueError):
        build_table([], n_realisations=1)


def test_build_table_rejects_zero_realisations():
    scenarios = generate_default_scenarios(3, seed=1)
    with pytest.raises(ValueError):
        build_table(scenarios, n_realisations=0)


def test_grouped_split_disjoint():
    train, test = grouped_split(50, test_fraction=0.2, seed=1)
    assert set(train.tolist()) & set(test.tolist()) == set()


def test_generate_default_scenarios_matches_module_function():
    from turbscope.synthetic import generate_scenarios

    a = generate_default_scenarios(10, seed=9)
    b = generate_scenarios(10, seed=9)
    assert [s.cn2_path for s in a] == [s.cn2_path for s in b]
