"""Simulator sanity, reproducibility and input-validation tests."""

import numpy as np
import pytest

from scintinet import (
    SimParams,
    angular_spectrum_propagate,
    kolmogorov_phase_screen,
    rytov_variance,
    simulate_scintillation,
)

# Fast test configuration: lambda=1.55e-6, L=1000 m ->
# sqrt(lambda*L)=0.0394 m; dx=0.4/128=3.1 mm <= 9.8 mm; width 0.4 >= 0.157 m.
FAST = dict(
    wavelength=1.55e-6,
    path_length=1000.0,
    grid_size=128,
    grid_width=0.4,
    n_screens=4,
    n_realizations=3,
)


class TestVacuumAndSanity:
    def test_vacuum_energy_conserved_random_field(self):
        # Angular-spectrum propagation is unitary: sum |U|^2 is exact.
        rng = np.random.default_rng(1)
        u = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        e0 = np.sum(np.abs(u) ** 2)
        u2 = angular_spectrum_propagate(u, 1.55e-6, 0.005, 500.0)
        assert np.sum(np.abs(u2) ** 2) == pytest.approx(e0, rel=1e-12)

    def test_zero_turbulence_no_scintillation(self):
        # A plane wave in vacuum stays a plane wave: sigma_I^2 ~ 0, <I> = 1.
        p = SimParams(cn2=0.0, **FAST)
        r = simulate_scintillation(p, seed=0)
        assert abs(r.sigma_i2_point) < 1e-12
        assert r.mean_intensity == pytest.approx(1.0, abs=1e-12)

    def test_zero_distance_identity(self):
        u = np.ones((32, 32), dtype=complex)
        out = angular_spectrum_propagate(u, 1e-6, 0.01, 0.0)
        assert np.allclose(out, u)

    def test_mean_intensity_near_unity_with_turbulence(self):
        p = SimParams(cn2=5e-16, **FAST)
        r = simulate_scintillation(p, seed=3)
        # Weak scattering redistributes but does not absorb energy.
        assert r.mean_intensity == pytest.approx(1.0, abs=0.02)

    def test_turbulence_produces_scintillation(self):
        p = SimParams(cn2=1e-15, **FAST)
        r = simulate_scintillation(p, seed=5)
        assert r.sigma_i2_point > 1e-4

    def test_aperture_averaging_reduces_index(self):
        p = SimParams(cn2=1e-15, aperture_diameters=(0.02, 0.06), **FAST)
        r = simulate_scintillation(p, seed=7)
        assert r.sigma_i2_aperture[0.06] < r.sigma_i2_aperture[0.02] < r.sigma_i2_point


class TestPhaseScreen:
    def test_screen_zero_mean_and_real(self):
        rng = np.random.default_rng(2)
        s = kolmogorov_phase_screen(rng, 128, 0.003, 1e-15 * 250.0, 1.55e-6)
        assert s.dtype == np.float64
        assert abs(s.mean()) < 1e-10  # piston removed
        assert s.std() > 0.0

    def test_screen_variance_scales_with_cn2dz(self):
        # Phase variance is linear in Cn^2*dz (spectrum is linear in it).
        rng1 = np.random.default_rng(9)
        rng2 = np.random.default_rng(9)
        s1 = kolmogorov_phase_screen(rng1, 128, 0.003, 1e-15 * 100.0, 1.55e-6)
        s2 = kolmogorov_phase_screen(rng2, 128, 0.003, 4e-15 * 100.0, 1.55e-6)
        assert s2.var() / s1.var() == pytest.approx(4.0, rel=1e-9)


class TestReproducibility:
    def test_same_seed_identical(self):
        p = SimParams(cn2=5e-16, aperture_diameters=(0.03,), **FAST)
        r1 = simulate_scintillation(p, seed=42)
        r2 = simulate_scintillation(p, seed=42)
        assert r1.sigma_i2_point == r2.sigma_i2_point
        assert r1.sigma_i2_aperture == r2.sigma_i2_aperture

    def test_different_seed_differs(self):
        p = SimParams(cn2=5e-16, **FAST)
        r1 = simulate_scintillation(p, seed=42)
        r2 = simulate_scintillation(p, seed=43)
        assert r1.sigma_i2_point != r2.sigma_i2_point


class TestBenchmarkRegression:
    """Benchmark: simulated sigma_I^2 must track Rytov theory in the weak regime.

    Regression: a fixed-seed reference value guards against silent changes to
    the screen synthesis or propagator normalisation.
    """

    def test_weak_regime_agreement_with_rytov(self):
        # Benchmark vs theory: expect agreement within tens of percent for a
        # reduced grid (256^2, 8 screens, 8 realizations). Tolerance 35%.
        p = SimParams(
            cn2=1e-15,
            wavelength=1.55e-6,
            path_length=2000.0,
            grid_size=256,
            grid_width=0.5,
            n_screens=8,
            n_realizations=8,
        )
        r = simulate_scintillation(p, seed=42)
        theory = rytov_variance(1e-15, 1.55e-6, 2000.0)  # 0.070950
        assert r.sigma_i2_point == pytest.approx(theory, rel=0.35)

    def test_fixed_seed_regression_value(self):
        # Regression reference produced by this exact configuration at build
        # time (seed 42): sigma_i2_point = 0.0639720889. Any normalisation
        # change (e.g. the factor-2 screen variance bug fixed during
        # development) shifts this by far more than the tolerance.
        p = SimParams(
            cn2=1e-15,
            wavelength=1.55e-6,
            path_length=2000.0,
            grid_size=256,
            grid_width=0.5,
            n_screens=8,
            n_realizations=8,
        )
        r = simulate_scintillation(p, seed=42)
        assert r.sigma_i2_point == pytest.approx(0.0639720889, abs=1e-6)


class TestValidationErrors:
    def test_negative_cn2(self):
        with pytest.raises(ValueError, match="cn2"):
            simulate_scintillation(SimParams(cn2=-1e-15, **FAST), seed=0)

    def test_bad_seed_type(self):
        with pytest.raises(TypeError, match="seed"):
            simulate_scintillation(SimParams(cn2=1e-15, **FAST), seed=1.5)

    def test_sampling_violation_coarse_grid(self):
        # dx = 2/32 = 62.5 mm > sqrt(lambda*L)/4 = 9.8 mm -> must raise.
        p = SimParams(
            cn2=1e-15, wavelength=1.55e-6, path_length=1000.0,
            grid_size=32, grid_width=2.0, n_screens=2, n_realizations=1,
        )
        with pytest.raises(ValueError, match="sampling"):
            simulate_scintillation(p, seed=0)

    def test_sampling_violation_small_domain(self):
        # width 0.1 < 4*sqrt(lambda*L) = 0.157 -> must raise.
        p = SimParams(
            cn2=1e-15, wavelength=1.55e-6, path_length=1000.0,
            grid_size=256, grid_width=0.1, n_screens=2, n_realizations=1,
        )
        with pytest.raises(ValueError, match="sampling"):
            simulate_scintillation(p, seed=0)

    def test_aperture_too_large(self):
        p = SimParams(cn2=1e-15, aperture_diameters=(0.3,), **FAST)
        with pytest.raises(ValueError, match="aperture"):
            simulate_scintillation(p, seed=0)

    def test_nonsquare_field_raises(self):
        with pytest.raises(ValueError, match="square"):
            angular_spectrum_propagate(np.ones((8, 16)), 1e-6, 0.01, 10.0)

    def test_zero_screens_raises(self):
        p = SimParams(
            cn2=1e-15, wavelength=1.55e-6, path_length=1000.0,
            grid_size=128, grid_width=0.4, n_screens=0, n_realizations=1,
        )
        with pytest.raises(ValueError, match="n_screens"):
            simulate_scintillation(p, seed=0)
