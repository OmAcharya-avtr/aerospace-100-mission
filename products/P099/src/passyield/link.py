"""Achievable downlink rate as a function of elevation, from a link budget.

WHY THIS MODULE EXISTS
----------------------
The product's headline number is a shortfall in delivered bits. Bits are the
time integral of a rate, and the rate on a low-Earth downlink is not constant
across a pass: the slant range swings by a factor of three or more between the
elevation mask and the zenith, and the atmosphere is in the way for longer at
low elevation. A shortfall accounting that treats the booked rate as achievable
throughout the window attributes a geometric effect to the operator, which is
exactly the error this product measures.

THE MODEL, WITH ITS SOURCES
---------------------------
Four cited pieces, composed. Every one carries units, assumptions and a
validity range, and none of them is tuned to this product's data.

1. **Free-space path loss.**

   .. math:: L_{fs} = 20\\log_{10}\\!\\left(\\frac{4\\pi d}{\\lambda}\\right)\\ \\text{[dB]}

   Source: Friis, H. T. (1946), "A Note on a Simple Transmission Formula",
   Proceedings of the IRE 34(5), 254-256. Units: ``d`` and ``lambda`` in the
   same length unit; this module uses metres. Assumptions: isotropic
   free-space spreading between matched antennas in the far field.
   Validity: any range beyond the Fraunhofer distance of the antennas, which
   for a metre-class X-band aperture is of order 100 m and is satisfied by
   nine orders of magnitude here.

2. **Clear-air slant-path attenuation.**

   .. math:: A(\\varepsilon) = A_{zenith} / \\sin\\varepsilon\\ \\text{[dB]}

   The plane-parallel (secant-law, "airmass") scaling of a zenith attenuation
   to a slant path. Source: ITU-R Recommendation P.618 (gaseous and
   hydrometeor attenuation on an Earth-space path; the ``1/sin`` path-length
   scaling is the standard form used there and in ITU-R P.676 for gaseous
   attenuation). Units: dB in, dB out, elevation in degrees.
   **Validity range: elevation above about 10 degrees.** Below that the
   Earth's curvature makes the plane-parallel assumption fail and the
   recommendations switch to an effective-path-length formulation; ITU-R P.618
   states the 1/sin form for angles above 10 degrees for that reason. This
   module therefore **raises** below :data:`MINIMUM_VALID_ELEVATION_DEG`
   rather than extrapolating, and :class:`passyield.geometry.GroundStation`
   refuses a mask below it.

3. **Carrier-to-noise-density ratio.**

   .. math:: C/N_0 = \\text{EIRP} + G/T - L_{fs} - A - 10\\log_{10} k

   The standard Earth-space link equation (Maral, Bousquet and Sun (2020),
   "Satellite Communications Systems", 6th edition, Wiley, chapter 5 on link
   budgets; Ippolito, L. J. (2017), "Satellite Communications Systems
   Engineering", 2nd edition, Wiley, chapter 4). Units: EIRP in dBW, G/T in
   dB/K, losses in dB, result in dB-Hz. ``-10*log10(k) = 228.5991...`` dB with
   the Boltzmann constant ``k = 1.380649e-23`` J/K, which is an exact defining
   constant of the SI since the 2019 redefinition, so this term carries no
   uncertainty.

4. **Spectral efficiency.**

   .. math:: \\eta = \\min\\!\\left(\\eta_{max},\\ \\log_2(1 + \\text{SNR}/M)\\right)

   Shannon, C. E. (1948), "A Mathematical Theory of Communication", Bell
   System Technical Journal 27, 379-423 and 623-656: the capacity of a
   band-limited additive-white-Gaussian-noise channel. ``SNR = (C/N_0)/B`` in
   linear units, ``M`` an implementation margin in linear units, ``eta_max``
   the ceiling imposed by the highest modulation and coding scheme the
   hardware actually carries.

   **What this is and is not.** It is a capacity-shaped envelope, continuous
   and monotone in elevation. It is **not** a modulation-and-coding table: a
   real DVB-S2 or CCSDS downlink steps between discrete modcods, so its true
   rate-versus-elevation curve is a staircase under this envelope with
   hysteresis at the switch points. The envelope is the honest choice here
   because it introduces no unsourced threshold numbers, and the consequence
   is recorded in ``README.md`` under Limitations: a staircase would add a
   further, operator-independent component to the shortfall that this product
   does not model.

THE COMPOSITION IS MONOTONE
---------------------------
:math:`d(\\varepsilon)` decreases and :math:`A(\\varepsilon)` decreases as
elevation rises, so :math:`C/N_0` increases and :math:`\\eta` is
non-decreasing. ``tests/test_link.py`` asserts the monotonicity with
Hypothesis, and the saturation region where ``eta == eta_max`` is flat rather
than increasing, which is why the assertion is non-decreasing and not strict.

UNITS ON EVERY BOUNDARY
-----------------------
Elevation degrees, slant range km, rate **Mbit/s**, bandwidth MHz, data volume
**Mbit**. Mbit rather than bytes throughout, because a link budget is in bits
and converting twice is how a factor of eight reaches a report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Speed of light in vacuum, m/s. Exact by the SI definition of the metre.
SPEED_OF_LIGHT_M_S = 299792458.0

#: ``-10*log10(k)`` in dB with ``k = 1.380649e-23`` J/K, the exact SI defining
#: value of the Boltzmann constant since 2019. Carries no uncertainty.
MINUS_TEN_LOG_K_DB = -10.0 * math.log10(1.380649e-23)

#: Lowest elevation at which the slant-path attenuation model of ITU-R P.618
#: is declared valid, degrees. Below this the module raises rather than
#: extrapolating the plane-parallel secant law into the regime where the
#: recommendation itself abandons it.
MINIMUM_VALID_ELEVATION_DEG = 10.0


@dataclass(frozen=True)
class LinkBudget:
    """The spacecraft-and-waveform side of the downlink budget.

    One instance describes the vehicle; the station side arrives through
    :class:`passyield.geometry.GroundStation`. Splitting it that way is what
    makes a held-out-station split meaningful: the station enters the physics
    only through two numbers, so a model can transfer to a station it never
    saw.

    Attributes
    ----------
    eirp_dbw:
        Spacecraft effective isotropic radiated power toward the station, dBW.
        Taken as constant over the pass: the spacecraft antenna pattern and
        the body-pointing error are not modelled. Recorded in Limitations.
    frequency_hz:
        Downlink centre frequency, Hz.
    bandwidth_mhz:
        Occupied bandwidth, MHz.
    implementation_margin_db:
        Loss of the real receiver against the Shannon bound, dB. Absorbs
        synchronisation loss, quantisation, filter and phase-noise penalties.
    max_spectral_efficiency_bits_per_hz:
        Ceiling ``eta_max``, bit/s/Hz, set by the highest modcod carried.
    """

    eirp_dbw: float
    frequency_hz: float
    bandwidth_mhz: float
    implementation_margin_db: float
    max_spectral_efficiency_bits_per_hz: float

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError(f"frequency_hz must be positive, got {self.frequency_hz!r}")
        if self.bandwidth_mhz <= 0.0:
            raise ValueError(f"bandwidth_mhz must be positive, got {self.bandwidth_mhz!r}")
        if self.implementation_margin_db < 0.0:
            raise ValueError(
                "implementation_margin_db is a loss and must be non-negative, got "
                f"{self.implementation_margin_db!r}"
            )
        if self.max_spectral_efficiency_bits_per_hz <= 0.0:
            raise ValueError(
                "max_spectral_efficiency_bits_per_hz must be positive, got "
                f"{self.max_spectral_efficiency_bits_per_hz!r}"
            )
        if not math.isfinite(self.eirp_dbw):
            raise ValueError(f"eirp_dbw must be finite, got {self.eirp_dbw!r}")

    @property
    def wavelength_m(self) -> float:
        """Downlink wavelength, metres."""
        return SPEED_OF_LIGHT_M_S / self.frequency_hz

    @property
    def max_rate_mbit_s(self) -> float:
        """Rate ceiling ``R_max = B * eta_max``, Mbit/s.

        This is the only rate figure the independent bound in
        :mod:`passyield.bounds` is allowed to use, so it is defined here once.
        """
        return self.bandwidth_mhz * self.max_spectral_efficiency_bits_per_hz


#: The vehicle used by the committed campaign and by every documented number.
#: A 50 MHz X-band downlink with a modest transmit chain. The figures are
#: declared design values for a SYNTHETIC vehicle, not measurements of any
#: flown spacecraft, and ``DATASET_CARD.md`` repeats that. They were chosen so
#: that the rate varies across the whole elevation range rather than sitting
#: against the ceiling, because a rate-versus-elevation relationship that
#: saturates at the mask would make the geometric term of the decomposition
#: identically zero and the product would measure nothing.
#: ``validation/validate_link.py`` reports the measured spread and the fraction
#: of the elevation range that saturates.
DEFAULT_BUDGET = LinkBudget(
    eirp_dbw=6.0,
    frequency_hz=8.2e9,
    bandwidth_mhz=50.0,
    implementation_margin_db=3.0,
    max_spectral_efficiency_bits_per_hz=5.0,
)


def free_space_loss_db(slant_range_km: np.ndarray | float, wavelength_m: float) -> np.ndarray:
    """Free-space path loss, dB. Friis (1946); see the module docstring.

    Parameters
    ----------
    slant_range_km:
        Range, km. Must be positive.
    wavelength_m:
        Wavelength, metres. Must be positive.
    """
    if wavelength_m <= 0.0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m!r}")
    ranges = np.asarray(slant_range_km, dtype=float)
    if np.any(ranges <= 0.0):
        raise ValueError("slant_range_km must be positive everywhere")
    return 20.0 * np.log10(4.0 * math.pi * ranges * 1000.0 / wavelength_m)


def slant_attenuation_db(
    elevation_deg: np.ndarray | float, zenith_attenuation_db: float
) -> np.ndarray:
    """Clear-air slant-path attenuation, dB. ITU-R P.618 secant scaling.

    Raises
    ------
    ValueError
        If any elevation is below :data:`MINIMUM_VALID_ELEVATION_DEG`. The
        plane-parallel law is not extrapolated below the range in which the
        recommendation states it.
    """
    if zenith_attenuation_db < 0.0:
        raise ValueError(
            f"zenith_attenuation_db must be non-negative, got {zenith_attenuation_db!r}"
        )
    elevation = np.asarray(elevation_deg, dtype=float)
    if np.any(elevation < MINIMUM_VALID_ELEVATION_DEG) or np.any(elevation > 90.0):
        raise ValueError(
            "elevation_deg must lie in "
            f"[{MINIMUM_VALID_ELEVATION_DEG}, 90] degrees for the ITU-R P.618 secant "
            "slant-path model; the model is not extrapolated outside its stated validity range"
        )
    return zenith_attenuation_db / np.sin(np.radians(elevation))


def carrier_to_noise_density_dbhz(
    elevation_deg: np.ndarray | float,
    slant_range_km: np.ndarray | float,
    gain_over_temperature_db: float,
    zenith_attenuation_db: float,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> np.ndarray:
    """``C/N0`` in dB-Hz from the Earth-space link equation. See the docstring."""
    return (
        budget.eirp_dbw
        + gain_over_temperature_db
        - free_space_loss_db(slant_range_km, budget.wavelength_m)
        - slant_attenuation_db(elevation_deg, zenith_attenuation_db)
        + MINUS_TEN_LOG_K_DB
    )


def spectral_efficiency_bits_per_hz(
    elevation_deg: np.ndarray | float,
    slant_range_km: np.ndarray | float,
    gain_over_temperature_db: float,
    zenith_attenuation_db: float,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> np.ndarray:
    """Shannon spectral efficiency with an implementation margin, bit/s/Hz.

    ``eta = min(eta_max, log2(1 + SNR/M))`` with ``SNR = (C/N0)/B`` linear.
    Shannon (1948); see the module docstring for what the envelope is and is
    not.
    """
    cn0 = carrier_to_noise_density_dbhz(
        elevation_deg, slant_range_km, gain_over_temperature_db, zenith_attenuation_db, budget
    )
    bandwidth_dbhz = 10.0 * np.log10(budget.bandwidth_mhz * 1.0e6)
    snr_db = cn0 - bandwidth_dbhz - budget.implementation_margin_db
    snr_linear = np.power(10.0, snr_db / 10.0)
    efficiency = np.log2(1.0 + snr_linear)
    return np.minimum(efficiency, budget.max_spectral_efficiency_bits_per_hz)


def rate_mbit_s(
    elevation_deg: np.ndarray | float,
    slant_range_km: np.ndarray | float,
    gain_over_temperature_db: float,
    zenith_attenuation_db: float,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> np.ndarray:
    """Achievable downlink rate, Mbit/s.

    ``R = B * eta`` with ``B`` in MHz and ``eta`` in bit/s/Hz, so the product
    is in Mbit/s directly. Non-decreasing in elevation and bounded above by
    :attr:`LinkBudget.max_rate_mbit_s`.
    """
    efficiency = spectral_efficiency_bits_per_hz(
        elevation_deg, slant_range_km, gain_over_temperature_db, zenith_attenuation_db, budget
    )
    return budget.bandwidth_mhz * efficiency
