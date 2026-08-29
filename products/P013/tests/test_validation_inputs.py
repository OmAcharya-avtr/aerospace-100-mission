"""Input-validation tests: every public entry point must reject bad physics."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import (
    PathGeometry,
    SensorSuite,
    TurbScopeModel,
    dimm_coefficient,
    dimm_variance,
    fried_from_average,
    generate_dataset,
    invert_dimm,
    invert_scintillation,
    scintillation_index,
    scintillation_weight,
    simulate_measurement,
    weight_normalisation,
    weighted_path_average,
)
from turbscope.scintillation import aperture_parameter_sq, gamma_gamma_parameters, rytov_variance


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_path_length_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError):
        PathGeometry(bad, 1.55e-6)


@pytest.mark.parametrize("bad", [1e-8, 1e-4, 0.0, -5e-7])
def test_wavelength_outside_the_stated_band_is_rejected(bad):
    with pytest.raises(ValueError, match="wavelength_m"):
        PathGeometry(1000.0, bad)


def test_geometry_label_is_validated():
    with pytest.raises(ValueError, match="geometry must be one of"):
        PathGeometry(1000.0, 1.55e-6, "cylindrical")
    with pytest.raises(TypeError):
        PathGeometry(1000.0, 1.55e-6, 3)


def test_path_samples_are_validated(path):
    z = path.uniform_grid(51)
    with pytest.raises(ValueError, match="equal length"):
        rytov_variance(z, np.ones(5), path)
    with pytest.raises(ValueError, match="at least 3"):
        rytov_variance(z[:2], np.ones(2), path)
    with pytest.raises(ValueError, match="strictly increasing"):
        rytov_variance(z[::-1], np.ones_like(z), path)
    bad = np.full_like(z, 1e-15)
    bad[3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        rytov_variance(z, bad, path)
    nan = np.full_like(z, 1e-15)
    nan[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rytov_variance(z, nan, path)


def test_weighted_average_rejects_unknown_kind(path):
    z = path.uniform_grid(11)
    with pytest.raises(ValueError, match="kind must be"):
        weighted_path_average(z, np.ones_like(z), kind="phase")
    with pytest.raises(ValueError, match="kind must be"):
        weight_normalisation("phase", "spherical")


def test_kernel_rejects_out_of_range_coordinate():
    with pytest.raises(ValueError, match=r"u must lie in \[0, 1\]"):
        scintillation_weight(np.array([-0.1, 0.5]))


def test_scintillation_index_rejects_negative_beta():
    with pytest.raises(ValueError, match="non-negative"):
        scintillation_index(np.array([-1.0]))
    with pytest.raises(ValueError, match="finite"):
        scintillation_index(np.array([np.nan]))


def test_gamma_gamma_requires_nonzero_turbulence():
    with pytest.raises(ValueError, match="beta0_sq"):
        gamma_gamma_parameters(0.0)


def test_aperture_parameter_rejects_negative_diameter(path):
    with pytest.raises(ValueError, match="receiver_diameter_m"):
        aperture_parameter_sq(-0.1, path)
    with pytest.raises(TypeError, match="PathGeometry"):
        aperture_parameter_sq(0.1, "1000 m")


def test_dimm_baseline_validity_range_is_enforced():
    with pytest.raises(ValueError, match="baseline >= "):
        dimm_coefficient(0.10, 0.15, "longitudinal")
    with pytest.raises(ValueError, match="component must be"):
        dimm_coefficient(0.06, 0.20, "radial")
    with pytest.raises(ValueError, match="r0_m"):
        dimm_variance(-0.1, 500e-9, 0.06, 0.20)


def test_fried_rejects_zero_cn2(path):
    with pytest.raises(ValueError, match="cn2_average"):
        fried_from_average(0.0, path)


def test_inversion_input_validation(path):
    with pytest.raises(ValueError, match="sigma_i2"):
        invert_scintillation(0.0, path)
    with pytest.raises(ValueError, match="method must be"):
        invert_scintillation(0.1, path, method="magic")
    with pytest.raises(ValueError, match="coverage"):
        invert_scintillation(0.1, path, coverage=1.5)
    with pytest.raises(TypeError, match="n_samples"):
        invert_scintillation(0.1, path, n_samples=100.5)
    with pytest.raises(ValueError, match="n_samples"):
        invert_scintillation(0.1, path, n_samples=3)
    with pytest.raises(ValueError, match="variance_rad2"):
        invert_dimm(-1.0, path, subaperture_m=0.06, baseline_m=0.20)


def test_sensor_suite_validation():
    with pytest.raises(ValueError, match="dimm_baseline_m"):
        SensorSuite(dimm_subaperture_m=0.10, dimm_baseline_m=0.15)
    with pytest.raises(ValueError, match="n_irradiance_samples"):
        SensorSuite(n_irradiance_samples=2)


def test_simulate_measurement_requires_a_generator(path, suite):
    z = path.uniform_grid(101)
    cn2 = np.full_like(z, 1e-15)
    with pytest.raises(TypeError, match="numpy Generator"):
        simulate_measurement(z, cn2, path, suite, 12345)


def test_generate_dataset_grid_must_be_odd():
    with pytest.raises(ValueError, match="odd integer"):
        generate_dataset(20, n_grid=200)
    with pytest.raises(ValueError, match="n_scenarios"):
        generate_dataset(2)


def test_model_rejects_wrong_feature_count():
    m = TurbScopeModel()
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        m.fit(np.zeros((10, 4)), np.zeros(10))
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict_log10(np.zeros((1, 13)))


def test_model_rejects_bad_coverage():
    with pytest.raises(ValueError, match="coverage"):
        TurbScopeModel(coverage=0.0)
