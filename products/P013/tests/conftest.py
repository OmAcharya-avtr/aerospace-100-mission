"""Shared fixtures for the TurbScope test suite."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import PathGeometry, SensorSuite


@pytest.fixture
def path() -> PathGeometry:
    """A 1 km spherical-wave path at 1550 nm."""
    return PathGeometry(1000.0, 1550e-9, "spherical")


@pytest.fixture
def suite() -> SensorSuite:
    """A modest instrument suite with plenty of samples."""
    return SensorSuite(
        receiver_diameter_m=0.10,
        dimm_subaperture_m=0.06,
        dimm_baseline_m=0.20,
        n_irradiance_samples=2000,
        n_dimm_frames=1000,
        dimm_noise_arcsec=0.05,
    )


@pytest.fixture
def uniform_profile(path: PathGeometry):
    """A uniform Cn2 = 1e-15 m^-2/3 path on a 401-point grid."""
    z = path.uniform_grid(401)
    return z, np.full_like(z, 1e-15)
