"""Unit, KAT, edge-case tests for wavelab.screens.kolmogorov_screen."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.screens import kolmogorov_screen


def test_rejects_small_grid():
    with pytest.raises(ValueError):
        kolmogorov_screen(4, 0.15, seed=0)


def test_rejects_non_integer_grid():
    with pytest.raises(TypeError):
        kolmogorov_screen(16.5, 0.15, seed=0)


def test_rejects_non_positive_r0():
    with pytest.raises(ValueError):
        kolmogorov_screen(16, 0.0, seed=0)
    with pytest.raises(ValueError):
        kolmogorov_screen(16, -0.1, seed=0)


def test_rejects_non_positive_pupil_diameter():
    with pytest.raises(ValueError):
        kolmogorov_screen(16, 0.15, seed=0, pupil_diameter=0.0)


def test_rejects_non_integer_seed():
    with pytest.raises(TypeError):
        kolmogorov_screen(16, 0.15, seed=1.5)


def test_output_shape():
    s = kolmogorov_screen(32, 0.15, seed=0)
    assert s.shape == (32, 32)
    assert np.all(np.isfinite(s))


def test_deterministic_given_same_seed():
    s1 = kolmogorov_screen(32, 0.15, seed=7)
    s2 = kolmogorov_screen(32, 0.15, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_different_seeds_give_different_screens():
    s1 = kolmogorov_screen(32, 0.15, seed=1)
    s2 = kolmogorov_screen(32, 0.15, seed=2)
    assert not np.allclose(s1, s2)


def test_stronger_turbulence_smaller_r0_gives_larger_variance():
    # Smaller r0/D -> stronger turbulence -> larger phase variance (Roddier
    # 1981 PSD scales as r0^(-5/3)).
    rng_seed = 0
    strong = kolmogorov_screen(48, 0.08, seed=rng_seed)
    weak = kolmogorov_screen(48, 0.40, seed=rng_seed)
    assert strong.std() > weak.std()


def test_variance_scales_approximately_as_r0_power():
    # Var(phi) propto r0^(-5/3): check the ratio of two r0 values against the
    # ratio of empirical variances (averaged over several seeds for stability),
    # loosely -- the FFT method's known low-frequency deficit (module
    # docstring) means this is not exact, so only the sign and rough
    # magnitude of the scaling are checked.
    r0a, r0b = 0.10, 0.20
    var_a = np.mean([kolmogorov_screen(48, r0a, seed=s).var() for s in range(5)])
    var_b = np.mean([kolmogorov_screen(48, r0b, seed=s).var() for s in range(5)])
    predicted_ratio = (r0b / r0a) ** (-5.0 / 3.0)  # var_b / var_a, predicted
    empirical_ratio = var_b / var_a
    # Loose bound: same order of magnitude and correct direction (< 1).
    assert empirical_ratio < 1.0
    assert 0.05 < empirical_ratio / predicted_ratio < 5.0
