"""Shared fixtures.

Residual sets are session-scoped because generating them runs the exact
rotation model once per observation, which is the slowest thing in the suite,
and because nothing in the suite mutates them.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointmodel import synthetic
from pointmodel.terms import coefficients_from_mapping


@pytest.fixture(scope="session")
def nominal_coefficients() -> np.ndarray:
    """The declared test coefficients of :data:`synthetic.NOMINAL_COEFFICIENTS`."""
    return coefficients_from_mapping(synthetic.NOMINAL_COEFFICIENTS)


@pytest.fixture(scope="session")
def clean_full_sky() -> synthetic.ResidualSet:
    """240 observations over the full sky, 2 arcsec noise, no outliers."""
    return synthetic.generate("full_sky", 240, sigma_arcsec=2.0, seed=4242)


@pytest.fixture(scope="session")
def noiseless_full_sky() -> synthetic.ResidualSet:
    """160 observations over the full sky with no noise at all.

    Used wherever a recovery test must separate the first-order truncation of
    the design matrix from the noise, which is the only honest way to measure
    the truncation.
    """
    return synthetic.generate("full_sky", 160, sigma_arcsec=0.0, seed=99)


@pytest.fixture(scope="session")
def narrow_band() -> synthetic.ResidualSet:
    """200 observations confined to a 43-47 degree elevation band."""
    return synthetic.generate("narrow_el", 200, sigma_arcsec=2.0, seed=7)


@pytest.fixture(scope="session")
def single_azimuth() -> synthetic.ResidualSet:
    """200 observations on one azimuth: exactly rank deficient by two."""
    return synthetic.generate("single_az", 200, sigma_arcsec=2.0, seed=7)


@pytest.fixture(scope="session")
def contaminated() -> synthetic.ResidualSet:
    """200 full-sky observations with 6 per cent acquisition failures."""
    return synthetic.generate(
        "full_sky",
        200,
        sigma_arcsec=2.0,
        outlier_fraction=0.06,
        outlier_arcsec=300.0,
        seed=31337,
    )
