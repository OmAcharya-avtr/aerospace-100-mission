"""Deterministic free-space optical (FSO) link budget.

All equations carry source, units, assumptions, and validity range.

Primary references
------------------
- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics*, 2nd ed.,
  Wiley, 2007 - Gaussian beam propagation (Chapter 3).
- I. I. Kim, B. McArthur, E. Korevaar, "Comparison of laser beam propagation
  at 785 nm and 1550 nm in fog and haze for optical wireless communications",
  Proc. SPIE 4214, 2001 - empirical fog/haze attenuation vs. visibility.
- A. K. Majumdar and J. C. Ricklin (eds.), *Free-Space Laser Communications:
  Principles and Advances*, Springer, 2008 - standard FSO link budget
  formulation (received power as product of gain/loss factors).

Sign convention: losses are stored as non-negative dB quantities and
subtracted from the transmit power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_LN10 = math.log(10.0)

#: Kim-model reference wavelength (Kim et al. 2001), metres.
_KIM_REF_WAVELENGTH_M = 550e-9


def _require_positive(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _require_finite(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")


def _require_fraction(name: str, value: float) -> None:
    _require_finite(name, value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def db_from_fraction(fraction: float) -> float:
    """Convert a power transmission fraction to a non-negative loss in dB.

    loss_dB = -10 log10(fraction). Units: fraction dimensionless in (0, 1],
    result in dB >= 0. Standard decibel definition (IEEE Std 100).
    """
    _require_fraction("fraction", fraction)
    return -10.0 * math.log10(fraction)


def watts_from_dbm(p_dbm: float) -> float:
    """Convert power in dBm to watts. P[W] = 1e-3 * 10^(P[dBm]/10)."""
    _require_finite("p_dbm", p_dbm)
    return 1e-3 * 10.0 ** (p_dbm / 10.0)


def dbm_from_watts(p_w: float) -> float:
    """Convert power in watts to dBm. P[dBm] = 10 log10(P[W]/1e-3)."""
    _require_positive("p_w", p_w)
    return 10.0 * math.log10(p_w / 1e-3)


def gaussian_divergence_half_angle(wavelength_m: float, waist_radius_m: float) -> float:
    """Far-field 1/e^2 divergence half-angle of a TEM00 Gaussian beam.

    theta = lambda / (pi * w0)

    Source: Saleh & Teich 2007, Eq. (3.1-21).
    Units: wavelength_m [m], waist_radius_m [m] (1/e^2 intensity radius at
    the waist), result [rad].
    Assumptions: diffraction-limited TEM00 beam, paraxial regime
    (theta << 1 rad). Validity: w0 >> lambda.
    """
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("waist_radius_m", waist_radius_m)
    if wavelength_m < 100e-9 or wavelength_m > 20e-6:
        raise ValueError(
            f"wavelength_m={wavelength_m} outside supported optical band "
            "[100 nm, 20 um]; check units (metres expected)"
        )
    return wavelength_m / (math.pi * waist_radius_m)


def rayleigh_range(wavelength_m: float, waist_radius_m: float) -> float:
    """Rayleigh range z_R = pi * w0^2 / lambda.

    Source: Saleh & Teich 2007, Eq. (3.1-11). Units: [m].
    """
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("waist_radius_m", waist_radius_m)
    return math.pi * waist_radius_m**2 / wavelength_m


def beam_radius(wavelength_m: float, waist_radius_m: float, range_m: float) -> float:
    """1/e^2 intensity radius of a Gaussian beam after propagating range_m.

    w(z) = w0 * sqrt(1 + (z / z_R)^2),  z_R = pi w0^2 / lambda

    Source: Saleh & Teich 2007, Eq. (3.1-8).
    Units: all lengths [m]. Assumptions: TEM00, vacuum/weak-turbulence
    diffraction only (turbulent beam spreading not included; see
    beamtwin.channel for stochastic effects). Validity: paraxial.
    """
    _require_positive("range_m", range_m)
    z_r = rayleigh_range(wavelength_m, waist_radius_m)
    return waist_radius_m * math.sqrt(1.0 + (range_m / z_r) ** 2)


def geometric_capture_fraction(beam_radius_m: float, aperture_radius_m: float) -> float:
    """Fraction of Gaussian-beam power captured by a centred circular aperture.

    eta = 1 - exp(-2 a^2 / w^2)

    Obtained by integrating the TEM00 intensity profile
    I(r) = I0 exp(-2 r^2 / w^2) over a disc of radius a (Saleh & Teich 2007,
    Sec. 3.1; the "power in the bucket" result, Eq. (3.1-16)).
    Units: lengths [m], result dimensionless in (0, 1).
    Assumptions: beam centred on aperture (offset handled separately by the
    pointing-loss term), aperture in the far field.
    """
    _require_positive("beam_radius_m", beam_radius_m)
    _require_positive("aperture_radius_m", aperture_radius_m)
    return 1.0 - math.exp(-2.0 * aperture_radius_m**2 / beam_radius_m**2)


def pointing_loss_fraction(
    offset_m: float,
    beam_radius_m: float,
) -> float:
    """Power transmission factor for a transverse beam-centre offset.

    L_p = exp(-2 d^2 / w^2)

    This is the on-axis intensity ratio of a Gaussian beam displaced by d,
    i.e. the point-receiver approximation (receiver aperture much smaller
    than the beam: a << w). Standard FSO pointing model; see e.g. Farid &
    Hranilovic, J. Lightwave Technol. 25(7), 2007, Eq. (8) with the
    aperture-averaging factor set to unity, and Majumdar & Ricklin 2008,
    Ch. 3.
    Units: offset_m [m] (transverse displacement at the receiver plane),
    beam_radius_m [m], result dimensionless in (0, 1].
    Assumptions: a << w (point receiver). For a/w up to ~0.3 the error in
    the captured fraction is small; document larger apertures as a
    limitation.
    """
    _require_finite("offset_m", offset_m)
    if offset_m < 0.0:
        raise ValueError(f"offset_m must be >= 0, got {offset_m}")
    _require_positive("beam_radius_m", beam_radius_m)
    return math.exp(-2.0 * offset_m**2 / beam_radius_m**2)


def kim_attenuation_db_per_km(visibility_km: float, wavelength_m: float) -> float:
    """Empirical fog/haze attenuation from visibility (Kim model).

    beta(lambda) = (3.91 / V) * (lambda / 550 nm)^(-q)   [1/km]
    alpha_dB/km  = (10 / ln 10) * beta = 4.343 * beta

    with the Kim et al. (2001) size-distribution exponent:
        q = 1.6            for V > 50 km
        q = 1.3            for 6 km < V <= 50 km
        q = 0.16 V + 0.34  for 1 km < V <= 6 km
        q = V - 0.5        for 0.5 km < V <= 1 km
        q = 0              for V <= 0.5 km

    Source: I. I. Kim, B. McArthur, E. Korevaar, Proc. SPIE 4214, 2001
    (modification of the Kruse model). Visibility defined at 550 nm with a
    2 % contrast threshold (3.91/V form).
    Units: visibility_km [km], wavelength_m [m], result [dB/km].
    Assumptions: homogeneous haze/fog along the path. Validity: empirical
    fit; least reliable for dense fog (V < 0.5 km) where attenuation becomes
    wavelength-independent.
    """
    _require_positive("visibility_km", visibility_km)
    _require_positive("wavelength_m", wavelength_m)
    v = visibility_km
    if v > 50.0:
        q = 1.6
    elif v > 6.0:
        q = 1.3
    elif v > 1.0:
        q = 0.16 * v + 0.34
    elif v > 0.5:
        q = v - 0.5
    else:
        q = 0.0
    beta_per_km = (3.91 / v) * (wavelength_m / _KIM_REF_WAVELENGTH_M) ** (-q)
    return (10.0 / _LN10) * beta_per_km


@dataclass(frozen=True)
class LinkParams:
    """Deterministic FSO link parameters.

    Attributes
    ----------
    wavelength_m : optical wavelength [m], validity [100 nm, 20 um].
    tx_power_dbm : transmit optical power [dBm].
    tx_efficiency : transmitter optics power efficiency, fraction (0, 1].
    rx_efficiency : receiver optics power efficiency, fraction (0, 1].
    beam_waist_radius_m : 1/e^2 waist radius at the transmitter [m].
    rx_aperture_radius_m : receiver aperture radius [m].
    range_m : link range [m].
    pointing_bias_rad : static pointing bias angle [rad], >= 0.
    attenuation_db_per_km : atmospheric attenuation [dB/km], >= 0
        (use kim_attenuation_db_per_km to derive from visibility).
    rx_sensitivity_dbm : receiver sensitivity threshold [dBm].
    """

    wavelength_m: float = 1550e-9
    tx_power_dbm: float = 20.0
    tx_efficiency: float = 0.8
    rx_efficiency: float = 0.8
    beam_waist_radius_m: float = 0.02
    rx_aperture_radius_m: float = 0.05
    range_m: float = 1000.0
    pointing_bias_rad: float = 0.0
    attenuation_db_per_km: float = 0.43
    rx_sensitivity_dbm: float = -30.0

    def __post_init__(self) -> None:
        _require_positive("wavelength_m", self.wavelength_m)
        if self.wavelength_m < 100e-9 or self.wavelength_m > 20e-6:
            raise ValueError(
                f"wavelength_m={self.wavelength_m} outside supported band [100 nm, 20 um]"
            )
        _require_finite("tx_power_dbm", self.tx_power_dbm)
        if self.tx_power_dbm > 50.0:
            raise ValueError(
                f"tx_power_dbm={self.tx_power_dbm} exceeds 50 dBm (100 W); "
                "implausible for the FSO terminals this model covers"
            )
        _require_fraction("tx_efficiency", self.tx_efficiency)
        _require_fraction("rx_efficiency", self.rx_efficiency)
        _require_positive("beam_waist_radius_m", self.beam_waist_radius_m)
        _require_positive("rx_aperture_radius_m", self.rx_aperture_radius_m)
        _require_positive("range_m", self.range_m)
        _require_finite("pointing_bias_rad", self.pointing_bias_rad)
        if self.pointing_bias_rad < 0.0:
            raise ValueError(f"pointing_bias_rad must be >= 0, got {self.pointing_bias_rad}")
        _require_finite("attenuation_db_per_km", self.attenuation_db_per_km)
        if self.attenuation_db_per_km < 0.0:
            raise ValueError(
                f"attenuation_db_per_km must be >= 0, got {self.attenuation_db_per_km}"
            )
        _require_finite("rx_sensitivity_dbm", self.rx_sensitivity_dbm)


@dataclass(frozen=True)
class LinkBudget:
    """Computed link budget. All losses are non-negative dB values.

    received_power_dbm = tx_power_dbm - tx_optics_loss_db
                         - geometric_loss_db - pointing_loss_db
                         - atmospheric_loss_db - rx_optics_loss_db
    margin_db = received_power_dbm - rx_sensitivity_dbm
    """

    params: LinkParams
    divergence_half_angle_rad: float
    beam_radius_at_rx_m: float
    tx_optics_loss_db: float
    geometric_loss_db: float
    pointing_loss_db: float
    atmospheric_loss_db: float
    rx_optics_loss_db: float
    received_power_dbm: float
    margin_db: float
    margin_negative: bool = field(default=False)

    def as_dict(self) -> dict[str, float | bool]:
        """Budget terms as a flat JSON-serialisable dict (dB / dBm / rad / m)."""
        return {
            "tx_power_dbm": self.params.tx_power_dbm,
            "divergence_half_angle_rad": self.divergence_half_angle_rad,
            "beam_radius_at_rx_m": self.beam_radius_at_rx_m,
            "tx_optics_loss_db": self.tx_optics_loss_db,
            "geometric_loss_db": self.geometric_loss_db,
            "pointing_loss_db": self.pointing_loss_db,
            "atmospheric_loss_db": self.atmospheric_loss_db,
            "rx_optics_loss_db": self.rx_optics_loss_db,
            "received_power_dbm": self.received_power_dbm,
            "rx_sensitivity_dbm": self.params.rx_sensitivity_dbm,
            "margin_db": self.margin_db,
            "margin_negative": self.margin_negative,
        }


def compute_budget(params: LinkParams) -> LinkBudget:
    """Compute the deterministic link budget for the given parameters.

    Combines the individual factors documented in this module:
    Gaussian-beam diffraction (Saleh & Teich 2007), geometric capture on a
    circular aperture, static pointing loss (point-receiver approximation),
    Beer-Lambert atmospheric attenuation alpha*L in dB, and fixed optics
    efficiencies. Turbulence and jitter are handled stochastically in
    beamtwin.channel.

    Returns a LinkBudget; margin_negative is True when the received power is
    below the receiver sensitivity (link fails deterministically).
    """
    w_rx = beam_radius(params.wavelength_m, params.beam_waist_radius_m, params.range_m)
    theta = gaussian_divergence_half_angle(params.wavelength_m, params.beam_waist_radius_m)
    eta_geo = geometric_capture_fraction(w_rx, params.rx_aperture_radius_m)
    offset_m = params.pointing_bias_rad * params.range_m
    l_point = pointing_loss_fraction(offset_m, w_rx)

    tx_loss_db = db_from_fraction(params.tx_efficiency)
    rx_loss_db = db_from_fraction(params.rx_efficiency)
    geo_loss_db = db_from_fraction(eta_geo)
    point_loss_db = db_from_fraction(l_point) if l_point > 0.0 else math.inf
    atm_loss_db = params.attenuation_db_per_km * params.range_m / 1000.0

    p_rx_dbm = (
        params.tx_power_dbm
        - tx_loss_db
        - geo_loss_db
        - point_loss_db
        - atm_loss_db
        - rx_loss_db
    )
    margin_db = p_rx_dbm - params.rx_sensitivity_dbm
    return LinkBudget(
        params=params,
        divergence_half_angle_rad=theta,
        beam_radius_at_rx_m=w_rx,
        tx_optics_loss_db=tx_loss_db,
        geometric_loss_db=geo_loss_db,
        pointing_loss_db=point_loss_db,
        atmospheric_loss_db=atm_loss_db,
        rx_optics_loss_db=rx_loss_db,
        received_power_dbm=p_rx_dbm,
        margin_db=margin_db,
        margin_negative=margin_db < 0.0,
    )
