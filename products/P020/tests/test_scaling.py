"""Property-based tests of the analytic scaling laws (Hypothesis).

The scaling exponents are algebraic identities of the closed forms, so they
must hold for *every* profile, wavelength and zenith angle inside the validity
ranges - which is exactly what property-based testing is for.

    r0      ~ lambda^(6/5)   sec(zeta)^(-3/5)
    theta0  ~ lambda^(6/5)   sec(zeta)^(-8/5)
    f_G     ~ lambda^(-6/5)  sec(zeta)^(+3/5)
    sigma_R^2 ~ lambda^(-7/6) sec(zeta)^(+11/6)

and, in the Cn^2 amplitude,  r0 ~ A^(-3/5),  theta0 ~ A^(-3/5),
f_G ~ A^(3/5),  sigma_R^2 ~ A^(1).
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from atmoprofile import (
    EXPONENT_SEC_ZENITH,
    EXPONENT_WAVELENGTH,
    bufton_wind,
    constant_profile,
    fried_parameter,
    greenwood_frequency,
    hv57,
    isoplanatic_angle,
    rytov_variance,
    slc_night,
)

PROFILES = {"hv57": hv57(), "slc_night": slc_night(), "slab": constant_profile(1e-15, 0.0, 5000.0)}
WIND = bufton_wind(5.0)
LAM_REF = 500e-9

wavelengths = st.floats(min_value=350e-9, max_value=12e-6, allow_nan=False, allow_infinity=False)
zeniths = st.floats(min_value=0.0, max_value=math.radians(59.0))
profile_keys = st.sampled_from(sorted(PROFILES))

SETTINGS = settings(max_examples=40, deadline=None)


@SETTINGS
@given(key=profile_keys, lam=wavelengths)
def test_r0_scales_as_lambda_to_the_six_fifths(key, lam):
    profile = PROFILES[key]
    ref = fried_parameter(profile, LAM_REF)
    got = fried_parameter(profile, lam)
    expected = ref * (lam / LAM_REF) ** EXPONENT_WAVELENGTH["r0"]
    assert got == pytest.approx(expected, rel=1e-10)


@SETTINGS
@given(key=profile_keys, lam=wavelengths)
def test_theta0_scales_as_lambda_to_the_six_fifths(key, lam):
    profile = PROFILES[key]
    ref = isoplanatic_angle(profile, LAM_REF)
    got = isoplanatic_angle(profile, lam)
    assert got == pytest.approx(ref * (lam / LAM_REF) ** (6 / 5), rel=1e-10)


@SETTINGS
@given(lam=wavelengths)
def test_greenwood_scales_as_lambda_to_the_minus_six_fifths(lam):
    profile = PROFILES["hv57"]
    ref = greenwood_frequency(profile, WIND, LAM_REF)
    got = greenwood_frequency(profile, WIND, lam)
    assert got == pytest.approx(ref * (lam / LAM_REF) ** (-6 / 5), rel=1e-10)


@SETTINGS
@given(key=profile_keys, lam=wavelengths, wave=st.sampled_from(["plane", "spherical"]))
def test_rytov_scales_as_lambda_to_the_minus_seven_sixths(key, lam, wave):
    profile = PROFILES[key]
    ref = rytov_variance(profile, LAM_REF, wave=wave, warn_strong=False)
    got = rytov_variance(profile, lam, wave=wave, warn_strong=False)
    assert got == pytest.approx(ref * (lam / LAM_REF) ** (-7 / 6), rel=1e-10)


@SETTINGS
@given(key=profile_keys, zen=zeniths)
def test_zenith_exponents(key, zen):
    """Every quantity must follow its stated sec(zeta) power exactly."""
    profile = PROFILES[key]
    sec = 1.0 / math.cos(zen)
    checks = {
        "r0": (
            fried_parameter(profile, LAM_REF, zenith_rad=zen),
            fried_parameter(profile, LAM_REF),
        ),
        "theta0": (
            isoplanatic_angle(profile, LAM_REF, zenith_rad=zen),
            isoplanatic_angle(profile, LAM_REF),
        ),
        "rytov_plane": (
            rytov_variance(profile, LAM_REF, zenith_rad=zen, warn_strong=False),
            rytov_variance(profile, LAM_REF, warn_strong=False),
        ),
        "rytov_spherical": (
            rytov_variance(profile, LAM_REF, zenith_rad=zen, wave="spherical", warn_strong=False),
            rytov_variance(profile, LAM_REF, wave="spherical", warn_strong=False),
        ),
    }
    for name, (slant, vertical) in checks.items():
        assert slant == pytest.approx(
            vertical * sec ** EXPONENT_SEC_ZENITH[name], rel=1e-10
        ), name


@SETTINGS
@given(zen=zeniths)
def test_greenwood_zenith_exponent(zen):
    profile = PROFILES["hv57"]
    sec = 1.0 / math.cos(zen)
    slant = greenwood_frequency(profile, WIND, LAM_REF, zenith_rad=zen)
    vertical = greenwood_frequency(profile, WIND, LAM_REF)
    assert slant == pytest.approx(vertical * sec ** (3 / 5), rel=1e-10)


@SETTINGS
@given(
    amp=st.floats(min_value=1e-18, max_value=1e-14, allow_nan=False),
    scale=st.floats(min_value=1.5, max_value=50.0),
)
def test_cn2_amplitude_scaling(amp, scale):
    """Multiplying Cn^2 by s multiplies mu by s: r0 ~ s^(-3/5), sigma_R^2 ~ s."""
    weak = constant_profile(amp, 0.0, 2000.0)
    strong = constant_profile(amp * scale, 0.0, 2000.0)
    r0_weak = fried_parameter(weak, LAM_REF)
    r0_strong = fried_parameter(strong, LAM_REF)
    assert r0_strong == pytest.approx(r0_weak * scale ** (-3 / 5), rel=1e-10)

    th_weak = isoplanatic_angle(weak, LAM_REF)
    th_strong = isoplanatic_angle(strong, LAM_REF)
    assert th_strong == pytest.approx(th_weak * scale ** (-3 / 5), rel=1e-10)

    s2_weak = rytov_variance(weak, LAM_REF, warn_strong=False)
    s2_strong = rytov_variance(strong, LAM_REF, warn_strong=False)
    assert s2_strong == pytest.approx(s2_weak * scale, rel=1e-10)


@SETTINGS
@given(
    amp=st.floats(min_value=1e-17, max_value=1e-15, allow_nan=False),
    v=st.floats(min_value=1.0, max_value=60.0),
)
def test_greenwood_uniform_wind_is_linear_in_wind(amp, v):
    """f_G ~ [int Cn^2 v^(5/3)]^(3/5) is exactly linear in a uniform wind."""
    from atmoprofile import constant_wind

    profile = constant_profile(amp, 0.0, 2000.0)
    f1 = greenwood_frequency(profile, constant_wind(1.0), LAM_REF)
    fv = greenwood_frequency(profile, constant_wind(v), LAM_REF)
    assert fv == pytest.approx(f1 * v, rel=1e-9)
