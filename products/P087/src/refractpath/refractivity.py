"""Four independently published formulations of the refractivity of air.

Every function here returns ``n - 1``, dimensionless, for air at a stated
pressure, temperature, water vapour partial pressure and vacuum wavelength.
They are listed in order of publication and they are deliberately not wrappers
around one another: the point of the module is that four separate authors'
fits to the same physical quantity disagree by a measurable amount, and that
amount propagates linearly into every refraction angle this package computes.

Equation (5), Edlen (1966), dispersion of standard air at 15 C, 101325 Pa, dry,
300 ppm CO2:

    (n_s - 1) x 1e8 = 8342.13 + 2406030 / (130 - s^2) + 15997 / (38.9 - s^2)

with ``s = 1 / lambda`` in reciprocal micrometres.

Source: Edlen, B. (1966), "The refractive index of air", Metrologia 2(2),
71-80. Units: lambda in micrometres in vacuum. Assumptions: standard air as
defined above. Validity: the visible and near infrared.

Equation (6), Birch and Downs (1994), the same quantity for 450 ppm CO2:

    (n_s - 1) x 1e8 = 8342.54 + 2406147 / (130 - s^2) + 15998 / (38.9 - s^2)

Source: Birch, K. P. and Downs, M. J. (1993), "An updated Edlen equation for
the refractive index of air", Metrologia 30(3), 155-162, and Birch, K. P. and
Downs, M. J. (1994), "Correction to the updated Edlen equation for the
refractive index of air", Metrologia 31(4), 315-316.

Equation (7), the Edlen-family density scaling to ambient pressure and
temperature, used by both equation (5) and equation (6) so that the comparison
between them isolates the dispersion term:

    (n - 1) = (n_s - 1) (p / 96095.43) (1 + 1e-8 (0.601 - 0.00972 t) p)
              / (1 + 0.0036610 t)

Source: the Birch and Downs (1993) form, as implemented in the NIST
Engineering Metrology Toolbox under the name "modified Edlen equation". Units:
p in Pa, t in degrees Celsius.

Equation (8), the Edlen-family water vapour correction:

    (n_moist - 1) = (n_dry - 1) - f (3.7345 - 0.0401 s^2) x 1e-10

with ``f`` the water vapour partial pressure in Pa. Source: as equation (7).
Note the sign: water vapour **lowers** the refractivity of air in the optical
band. It raises it in the radio band, which this product does not cover.

Equation (9), the Explanatory Supplement / IAU form, which is the
low-precision formulation that is actually used in astronomical pointing code:

    (n - 1) = (287.604 + 1.6288 s^2 + 0.0136 s^4) x 1e-6
              x (p / 101325) x (273.15 / T)
              - (0.0624 - 0.000680 s^2) e_hPa x 1e-6 / (1 + 0.003661 t)

Source: the refraction section of the Explanatory Supplement to the
Astronomical Almanac, 3rd edition, Urban, S. E. and Seidelmann, P. K. (eds.),
University Science Books, 2013; the same expression appears in the SLALIB
documentation. Units: p in Pa on input to this module and converted
internally, T in K, e in hPa inside the published expression, t in C.
Assumptions: standard air at 0 C and 101325 Pa for the leading coefficients.
Validity: the optical band.

Equation (10), Ciddor (1996). This one is structurally different from the
others: instead of an empirical pressure and temperature scaling it computes
the densities of the dry and the moist components with the BIPM-1991 equation
of state and forms a density-weighted sum of two separate dispersion terms.

    (n_as - 1) x 1e8 = 5792105 / (238.0185 - s^2) + 167917 / (57.362 - s^2)
    (n_axs - 1)      = (n_as - 1) (1 + 0.534e-6 (x_c - 450))
    (n_ws - 1) x 1e8 = 1.022 (295.235 + 2.6422 s^2 - 0.032380 s^4
                              + 0.004028 s^6)
    (n - 1)          = (rho_a / rho_axs)(n_axs - 1) + (rho_w / rho_ws)(n_ws - 1)

Source: Ciddor, P. E. (1996), "Refractive index of air: new equations for the
visible and near infrared", Applied Optics 35(9), 1566-1573. Validity: 300 to
1690 nm, 80 to 120 kPa, -40 to +100 C, 0 to 100 per cent humidity, as stated
in that paper.

Equation (11), the BIPM-1991 compressibility used by equation (10):

    Z = 1 - (p/T)[a0 + a1 t + a2 t^2 + (b0 + b1 t) x_w + (c0 + c1 t) x_w^2]
          + (p/T)^2 (d + e x_w)

Equation (12), the saturation vapour pressure and enhancement factor used by
equation (10), as given by Ciddor (1996) from the IAPWS formulation:

    p_sv = exp(A T^2 + B T + C + D / T)
    f    = alpha + beta p + gamma t^2

A reader wanting one number should use equation (10). A reader wanting to know
how much their choice of formulation matters should use
:func:`refractivity_spread`, which is the output this product exists to
produce.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

ArrayLike = float | np.ndarray

#: Universal gas constant used by Ciddor (1996), J/(mol K). The 1996 paper uses
#: the 1986 CODATA value; it is kept here rather than updated because changing
#: it changes a published formulation into an unpublished one.
R_CIDDOR = 8.314510

#: Molar mass of water, kg/mol, as used by Ciddor (1996).
M_WATER = 0.018015


def _as_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _check_inputs(
    wavelength_um: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_c: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = _as_array(wavelength_um, "wavelength_um")
    pressure = _as_array(pressure_pa, "pressure_pa")
    temperature = _as_array(temperature_c, "temperature_c")
    if np.any(lam <= 0.0):
        raise ValueError(f"wavelength_um must be positive, got minimum {lam.min()!r}")
    if np.any(pressure <= 0.0):
        raise ValueError(f"pressure_pa must be positive, got minimum {pressure.min()!r}")
    if np.any(temperature <= -273.15):
        raise ValueError(
            f"temperature_c must be above absolute zero, got minimum {temperature.min()!r}"
        )
    sigma_sq = 1.0 / lam**2
    if np.any(sigma_sq >= 38.9):
        raise ValueError(
            "wavelength_um is at or below the 0.1603 um pole of the Edlen-family "
            "dispersion denominators; no formulation here is defined there"
        )
    if np.any(sigma_sq >= 57.362):
        raise ValueError("wavelength_um is at or below the 0.1320 um pole of equation (10)")
    return lam, pressure, temperature


def _edlen_density_factor(pressure_pa: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Equation (7)."""
    return (
        (pressure_pa / 96095.43)
        * (1.0 + 1e-8 * (0.601 - 0.00972 * temperature_c) * pressure_pa)
        / (1.0 + 0.0036610 * temperature_c)
    )


def _edlen_vapour_term(sigma_sq: np.ndarray, vapour_pressure_pa: np.ndarray) -> np.ndarray:
    """Equation (8), the quantity subtracted from the dry refractivity."""
    return vapour_pressure_pa * (3.7345 - 0.0401 * sigma_sq) * 1e-10


def edlen_1966(
    wavelength_um: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_c: ArrayLike,
    vapour_pressure_pa: ArrayLike = 0.0,
) -> np.ndarray:
    """Refractivity ``n - 1`` from equations (5), (7) and (8).

    Args:
        wavelength_um: Vacuum wavelength, micrometres.
        pressure_pa: Total air pressure, pascals.
        temperature_c: Air temperature, degrees Celsius.
        vapour_pressure_pa: Water vapour partial pressure, pascals.

    Returns:
        ``n - 1``, dimensionless, broadcast over the inputs.

    Raises:
        ValueError: On a non-finite, non-positive or out-of-domain input.
    """
    lam, pressure, temperature = _check_inputs(wavelength_um, pressure_pa, temperature_c)
    vapour = _as_array(vapour_pressure_pa, "vapour_pressure_pa")
    sigma_sq = 1.0 / lam**2
    standard = (
        8342.13 + 2406030.0 / (130.0 - sigma_sq) + 15997.0 / (38.9 - sigma_sq)
    ) * 1e-8
    dry = standard * _edlen_density_factor(pressure, temperature)
    return np.asarray(dry - _edlen_vapour_term(sigma_sq, vapour), dtype=float)


def birch_downs_1994(
    wavelength_um: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_c: ArrayLike,
    vapour_pressure_pa: ArrayLike = 0.0,
) -> np.ndarray:
    """Refractivity ``n - 1`` from equations (6), (7) and (8).

    Args:
        wavelength_um: Vacuum wavelength, micrometres.
        pressure_pa: Total air pressure, pascals.
        temperature_c: Air temperature, degrees Celsius.
        vapour_pressure_pa: Water vapour partial pressure, pascals.

    Returns:
        ``n - 1``, dimensionless, broadcast over the inputs.

    Raises:
        ValueError: On a non-finite, non-positive or out-of-domain input.
    """
    lam, pressure, temperature = _check_inputs(wavelength_um, pressure_pa, temperature_c)
    vapour = _as_array(vapour_pressure_pa, "vapour_pressure_pa")
    sigma_sq = 1.0 / lam**2
    standard = (
        8342.54 + 2406147.0 / (130.0 - sigma_sq) + 15998.0 / (38.9 - sigma_sq)
    ) * 1e-8
    dry = standard * _edlen_density_factor(pressure, temperature)
    return np.asarray(dry - _edlen_vapour_term(sigma_sq, vapour), dtype=float)


def iau_explanatory_supplement(
    wavelength_um: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_c: ArrayLike,
    vapour_pressure_pa: ArrayLike = 0.0,
) -> np.ndarray:
    """Refractivity ``n - 1`` from equation (9), the low-precision form in use.

    Args:
        wavelength_um: Vacuum wavelength, micrometres.
        pressure_pa: Total air pressure, pascals.
        temperature_c: Air temperature, degrees Celsius.
        vapour_pressure_pa: Water vapour partial pressure, pascals.

    Returns:
        ``n - 1``, dimensionless, broadcast over the inputs.

    Raises:
        ValueError: On a non-finite, non-positive or out-of-domain input.
    """
    lam, pressure, temperature = _check_inputs(wavelength_um, pressure_pa, temperature_c)
    vapour = _as_array(vapour_pressure_pa, "vapour_pressure_pa")
    sigma_sq = 1.0 / lam**2
    kelvin = temperature + 273.15
    dry = (
        (287.604 + 1.6288 * sigma_sq + 0.0136 * sigma_sq**2)
        * 1e-6
        * (pressure / 101325.0)
        * (273.15 / kelvin)
    )
    wet = (
        (0.0624 - 0.000680 * sigma_sq)
        / (1.0 + 0.003661 * temperature)
        * (vapour / 100.0)
        * 1e-6
    )
    return np.asarray(dry - wet, dtype=float)


def ciddor_saturation_pressure_pa(kelvin: ArrayLike) -> np.ndarray:
    """Saturation vapour pressure from the first line of equation (12), Pa.

    Args:
        kelvin: Absolute temperature, K.

    Returns:
        Saturation vapour pressure, Pa.
    """
    kelvin = np.asarray(kelvin, dtype=float)
    return np.exp(
        1.2378847e-5 * kelvin**2 - 1.9121316e-2 * kelvin + 33.93711047 - 6.3431645e3 / kelvin
    )


def ciddor_enhancement(pressure_pa: ArrayLike, temperature_c: ArrayLike) -> np.ndarray:
    """Water vapour enhancement factor from the second line of equation (12).

    Args:
        pressure_pa: Total air pressure, Pa.
        temperature_c: Air temperature, degrees Celsius.

    Returns:
        The dimensionless enhancement factor.
    """
    pressure_pa = np.asarray(pressure_pa, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)
    return 1.00062 + 3.14e-8 * pressure_pa + 5.6e-7 * temperature_c**2


def ciddor_compressibility(
    pressure_pa: ArrayLike, kelvin: ArrayLike, mole_fraction_water: ArrayLike
) -> np.ndarray:
    """Compressibility of moist air from equation (11), the BIPM-1991 form.

    Args:
        pressure_pa: Total air pressure, Pa.
        kelvin: Absolute temperature, K.
        mole_fraction_water: Mole fraction of water vapour, 0 to 1.

    Returns:
        The dimensionless compressibility Z.
    """
    pressure_pa = np.asarray(pressure_pa, dtype=float)
    kelvin = np.asarray(kelvin, dtype=float)
    mole_fraction_water = np.asarray(mole_fraction_water, dtype=float)
    t = kelvin - 273.15
    a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
    b0, b1 = 5.707e-6, -2.051e-8
    c0, c1 = 1.9898e-4, -2.376e-6
    d, e = 1.83e-11, -0.765e-8
    xw = mole_fraction_water
    return (
        1.0
        - (pressure_pa / kelvin)
        * (a0 + a1 * t + a2 * t**2 + (b0 + b1 * t) * xw + (c0 + c1 * t) * xw**2)
        + (pressure_pa / kelvin) ** 2 * (d + e * xw)
    )


def ciddor_molar_mass_air(co2_ppm: ArrayLike) -> np.ndarray:
    """Molar mass of dry air with the stated CO2 content, kg/mol, from Ciddor (1996).

    Args:
        co2_ppm: Carbon dioxide mole fraction, parts per million.

    Returns:
        Molar mass in kg/mol.
    """
    return np.asarray(1e-3 * (28.9635 + 12.011e-6 * (np.asarray(co2_ppm, float) - 400.0)))


def ciddor_1996(
    wavelength_um: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_c: ArrayLike,
    vapour_pressure_pa: ArrayLike = 0.0,
    co2_ppm: ArrayLike = 450.0,
) -> np.ndarray:
    """Refractivity ``n - 1`` from equations (10) to (12).

    Args:
        wavelength_um: Vacuum wavelength, micrometres.
        pressure_pa: Total air pressure, pascals.
        temperature_c: Air temperature, degrees Celsius.
        vapour_pressure_pa: Water vapour partial pressure, pascals. Converted
            internally to the mole fraction equation (10) needs.
        co2_ppm: Carbon dioxide mole fraction, parts per million.

    Returns:
        ``n - 1``, dimensionless, broadcast over the inputs.

    Raises:
        ValueError: On a non-finite, non-positive or out-of-domain input, or
            if the vapour partial pressure exceeds the total pressure.
    """
    lam, pressure, temperature = _check_inputs(wavelength_um, pressure_pa, temperature_c)
    vapour = _as_array(vapour_pressure_pa, "vapour_pressure_pa")
    if np.any(vapour > pressure):
        raise ValueError("vapour_pressure_pa must not exceed pressure_pa")
    co2 = _as_array(co2_ppm, "co2_ppm")
    sigma_sq = 1.0 / lam**2
    kelvin = temperature + 273.15

    dry_standard = (
        5792105.0 / (238.0185 - sigma_sq) + 167917.0 / (57.362 - sigma_sq)
    ) * 1e-8
    dry_standard_co2 = dry_standard * (1.0 + 0.534e-6 * (co2 - 450.0))
    wet_standard = (
        1.022
        * (
            295.235
            + 2.6422 * sigma_sq
            - 0.032380 * sigma_sq**2
            + 0.004028 * sigma_sq**3
        )
        * 1e-8
    )

    mole_fraction_water = np.asarray(vapour / pressure, dtype=float)
    molar_mass_air = ciddor_molar_mass_air(co2)
    compressibility = ciddor_compressibility(pressure, kelvin, mole_fraction_water)

    density_dry_component = (
        pressure * molar_mass_air * (1.0 - mole_fraction_water)
        / (compressibility * R_CIDDOR * kelvin)
    )
    density_wet_component = (
        pressure * M_WATER * mole_fraction_water / (compressibility * R_CIDDOR * kelvin)
    )

    # Standard-state densities for the two dispersion terms: dry air at 15 C and
    # 101325 Pa with the stated CO2 content, and pure water vapour at 20 C and
    # 1333 Pa, both as defined by Ciddor (1996).
    standard_dry_density = (
        101325.0
        * molar_mass_air
        / (ciddor_compressibility(101325.0, 288.15, 0.0) * R_CIDDOR * 288.15)
    )
    standard_wet_density = (
        1333.0 * M_WATER / (ciddor_compressibility(1333.0, 293.15, 1.0) * R_CIDDOR * 293.15)
    )

    return np.asarray(
        (density_dry_component / standard_dry_density) * dry_standard_co2
        + (density_wet_component / standard_wet_density) * wet_standard,
        dtype=float,
    )


def ciddor_vapour_pressure_pa(
    temperature_c: ArrayLike, relative_humidity: ArrayLike, pressure_pa: ArrayLike
) -> np.ndarray:
    """Water vapour partial pressure from equation (12), Pa.

    This is the Ciddor/IAPWS saturation form with the Ciddor enhancement
    factor, provided so that a caller can compare it against the Buck (1981)
    form of equation (4). They disagree, and the size of the disagreement is
    measured in ``validation/validate_dispersion.py``.

    Args:
        temperature_c: Air temperature, degrees Celsius.
        relative_humidity: Relative humidity, 0 to 1.
        pressure_pa: Total air pressure, pascals.

    Returns:
        Water vapour partial pressure, pascals.
    """
    temperature = _as_array(temperature_c, "temperature_c")
    humidity = _as_array(relative_humidity, "relative_humidity")
    pressure = _as_array(pressure_pa, "pressure_pa")
    kelvin = temperature + 273.15
    return np.asarray(
        ciddor_enhancement(pressure, temperature)
        * humidity
        * ciddor_saturation_pressure_pa(kelvin),
        dtype=float,
    )


#: The four formulations, by name. Every consumer in this package iterates over
#: this mapping rather than hard-coding a choice, so adding a fifth formulation
#: adds it to every disagreement figure at once.
REFRACTIVITY_MODELS: dict[str, Callable[..., np.ndarray]] = {
    "iau_explanatory_supplement": iau_explanatory_supplement,
    "edlen_1966": edlen_1966,
    "birch_downs_1994": birch_downs_1994,
    "ciddor_1996": ciddor_1996,
}

#: Published validity interval in micrometres for each formulation, with the
#: source of the interval. Where the source does not state an interval the
#: entry says so and the interval is this product's declaration.
REFRACTIVITY_VALIDITY_UM: dict[str, tuple[float, float, str]] = {
    "iau_explanatory_supplement": (
        0.30,
        1.00,
        (
            "declared by this product; the Explanatory Supplement form is quoted "
            "for the optical band without a stated interval"
        ),
    ),
    "edlen_1966": (0.30, 1.69, "Edlen (1966), visible and near infrared"),
    "birch_downs_1994": (
        0.35,
        0.65,
        "Birch and Downs (1993), the interval over which the fit was determined",
    ),
    "ciddor_1996": (0.30, 1.69, "Ciddor (1996), stated in that paper"),
}


def refractivity_spread(
    wavelength_um: float,
    pressure_pa: float,
    temperature_c: float,
    vapour_pressure_pa: float = 0.0,
    co2_ppm: float = 450.0,
) -> dict[str, float]:
    """Every formulation's refractivity at one state, plus the spread.

    Args:
        wavelength_um: Vacuum wavelength, micrometres.
        pressure_pa: Total air pressure, pascals.
        temperature_c: Air temperature, degrees Celsius.
        vapour_pressure_pa: Water vapour partial pressure, pascals.
        co2_ppm: Carbon dioxide mole fraction, parts per million. Used only by
            equation (10).

    Returns:
        Mapping with one entry per formulation giving ``n - 1``, plus
        ``spread`` (maximum minus minimum) and ``relative_spread``
        (``spread`` divided by the mean).
    """
    values: dict[str, float] = {}
    for name, function in REFRACTIVITY_MODELS.items():
        if name == "ciddor_1996":
            value = function(
                wavelength_um, pressure_pa, temperature_c, vapour_pressure_pa, co2_ppm
            )
        else:
            value = function(wavelength_um, pressure_pa, temperature_c, vapour_pressure_pa)
        values[name] = float(value)
    magnitudes = list(values.values())
    spread = max(magnitudes) - min(magnitudes)
    mean = sum(magnitudes) / len(magnitudes)
    values["spread"] = spread
    values["relative_spread"] = spread / mean if mean != 0.0 else math.inf
    return values
