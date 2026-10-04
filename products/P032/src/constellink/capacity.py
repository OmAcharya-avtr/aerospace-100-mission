"""Per-link capacity models: RF and free-space optical, with a hybrid dispatcher.

Unit and sign conventions follow LinkBudgetX (P006) and are restated here
because this package does not import it: every ``L_*`` term is a POSITIVE loss
in dB, gains are positive dB, and power is carried in dBW on the RF side and
in watts on the optical side (converted internally).

RF chain
--------
Free-space path loss, Friis 1946 ("A Note on a Simple Transmission Formula",
Proc. IRE 34(5), 254-256).  Friis gives the received-to-transmitted power
ratio for matched antennas in free space as

    P_rx / P_tx = G_tx G_rx (lambda / (4 pi R))^2                        (1)

so, in dB,

    L_fs[dB] = 20 log10(4 pi R / lambda)                                 (2)

Units: ``R`` in m, ``lambda`` in m.  Validity: far field of both apertures,
free space, no multipath, polarisation matched.

Carrier-to-noise-density ratio (Maral & Bousquet 2009, "Satellite
Communications Systems: Systems, Techniques and Technology", 5th ed., Wiley,
Ch. 5 -- the standard link-budget equation):

    C/N0[dB-Hz] = EIRP[dBW] - L_fs[dB] - L_other[dB] + G/T[dB(K^-1)]
                  - k[dB(W/K/Hz)]                                        (3)

with ``EIRP = P_tx + G_tx - L_tx`` and ``k = 10 log10(1.380649e-23)
= -228.5991 dB(W/K/Hz)`` from the SI-defined Boltzmann constant (exact since
the 2019 SI revision).

Two capacities are reported and must not be confused:

* ``shannon_capacity_bps``: the information-theoretic bound for an AWGN
  channel of bandwidth ``B`` (Shannon 1948, "A Mathematical Theory of
  Communication", Bell Syst. Tech. J. 27, 379-423 and 623-656):
      C = B log2(1 + S/N),  S/N = (C/N0)_linear / B                      (4)
  No real modem reaches this.
* ``achievable_rate_bps``: the rate at which the required energy per bit is
  met with the stated margin, which is what a link plan actually uses:
      R_b = (C/N0)_linear / (Eb/N0_required)_linear / margin_linear      (5)
  Equation (5) is the dB-domain relation ``Eb/N0 = C/N0 - 10 log10(R_b)``
  rearranged (Maral & Bousquet 2009, Ch. 5).  ``Eb/N0_required`` is a modem
  property the caller supplies; this package ships no modcod table and
  invents no coding gains.

Optical chain
-------------
Far-field Gaussian beam (Saleh & Teich, "Fundamentals of Photonics", 2nd ed.,
Wiley, Ch. 3).  ``theta_half`` is the 1/e^2 HALF divergence angle; the public
input ``beam_divergence_full_rad`` is the FULL angle, so
``theta_half = theta_full / 2``.  Beam radius at range ``R``:
``w(R) = theta_half R`` (far field, ``R`` much greater than the Rayleigh
range).  A centred circular receive aperture of radius ``a`` captures

    f_geo = 1 - exp(-2 a^2 / w(R)^2)                                     (6)

and a radial pointing error ``theta_err`` attenuates by

    f_point = exp(-2 theta_err^2 / theta_half^2)                         (7)

(the standard FSO pointing-loss form; Majumdar & Ricklin, "Free-Space Laser
Communications: Principles and Advances", Springer 2008, Ch. 1).  Validity:
``a`` small compared with ``w``; for comparable sizes truncation and offset
couple and the product (6)x(7) is only approximate.

Atmospheric ground leg.  The implemented model is the plane-parallel
(cosecant) scaling of a user-supplied zenith attenuation,

    A(eps)[dB] = A_zenith[dB] / sin(eps)                                 (8)

as given for slant paths in Ippolito, "Satellite Communications Systems
Engineering", Wiley 2008, Ch. 4.  Validity: elevation above roughly 10 deg;
below that the plane-parallel assumption fails and (8) over-predicts.  The
authoritative models for Earth-space atmospheric propagation are the ITU-R
P-series recommendations, which this package does NOT implement; the ``itur``
package on PyPI does, and a reader who needs them should use it.  An optional
visibility-driven specific attenuation is available from the Kim-Kruse model
(Kim, McArthur & Korevaar, Proc. SPIE 4214, 2001, "Comparison of laser beam
propagation at 785 nm and 1550 nm in fog and haze for optical wireless
communications") applied over a user-supplied effective path length.

Achievable optical rate.  The receiver sensitivity is expressed in photons
per bit, which is the conventional figure of merit for photon-counting and
PPM optical receivers (Hemmati, ed., "Deep Space Optical Communications",
JPL Deep-Space Communications and Navigation Series, Wiley 2006, Ch. 4):

    R_b = P_rx / (E_photon * n_pb * margin),  E_photon = h c / lambda    (9)

with ``h = 6.62607015e-34 J s`` and ``c = 299 792 458 m/s`` (both SI-exact).
``n_pb`` is a required input: no default photons-per-bit figure is supplied
here, because it is a receiver-design property and a plausible-looking
default would be a fabricated number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

__all__ = [
    "BOLTZMANN_DB_W_K_HZ",
    "PLANCK_J_S",
    "SPEED_OF_LIGHT_M_S",
    "RfTerminal",
    "OpticalTerminal",
    "RfLinkResult",
    "OpticalLinkResult",
    "free_space_path_loss_db",
    "rf_link",
    "optical_link",
    "kim_specific_attenuation_db_km",
    "slant_path_attenuation_db",
    "hybrid_capacity_bps",
]

# 10 * log10(1.380649e-23); Boltzmann constant is SI-exact since 2019.
BOLTZMANN_DB_W_K_HZ = 10.0 * math.log10(1.380649e-23)
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299792458.0


def _loss_db(fraction: float) -> float:
    """Convert a transmitted power fraction in (0, 1] to a positive dB loss."""
    if fraction <= 0.0:
        raise ValueError(f"power fraction must be > 0, got {fraction}")
    if fraction >= 1.0:
        return 0.0
    return -10.0 * math.log10(fraction)


def free_space_path_loss_db(range_km: float, frequency_hz: float) -> float:
    """Friis free-space path loss [dB], Eq. (2).

    Parameters
    ----------
    range_km : slant range [km], > 0.
    frequency_hz : carrier frequency [Hz], > 0.
    """
    if range_km <= 0.0:
        raise ValueError(f"range_km must be > 0, got {range_km}")
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")
    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    return 20.0 * math.log10(4.0 * math.pi * (range_km * 1e3) / wavelength_m)


@dataclass(frozen=True)
class RfTerminal:
    """RF link terminal pair description (units in field names).

    Attributes
    ----------
    tx_power_dbw : transmit power [dBW].
    tx_gain_dbi : transmit antenna gain [dBi].
    tx_loss_db : transmit feed/pointing loss [dB], >= 0.
    rx_g_over_t_db_per_k : receive figure of merit G/T [dB(1/K)].
    frequency_hz : carrier frequency [Hz], > 0.
    bandwidth_hz : occupied bandwidth [Hz], > 0.
    required_ebn0_db : modem requirement at the target error rate [dB].
    other_loss_db : all remaining losses -- polarisation, atmosphere,
        implementation -- as a single positive dB figure, >= 0.
    margin_db : design margin held back from the achievable rate [dB], >= 0.
    """

    tx_power_dbw: float
    tx_gain_dbi: float
    rx_g_over_t_db_per_k: float
    frequency_hz: float
    bandwidth_hz: float
    required_ebn0_db: float
    tx_loss_db: float = 0.0
    other_loss_db: float = 0.0
    margin_db: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError(f"frequency_hz must be > 0, got {self.frequency_hz}")
        if self.bandwidth_hz <= 0.0:
            raise ValueError(f"bandwidth_hz must be > 0, got {self.bandwidth_hz}")
        for fname in ("tx_loss_db", "other_loss_db", "margin_db"):
            if getattr(self, fname) < 0.0:
                raise ValueError(f"{fname} must be >= 0 (losses are positive dB), "
                                 f"got {getattr(self, fname)}")

    @property
    def eirp_dbw(self) -> float:
        """Effective isotropic radiated power [dBW]."""
        return self.tx_power_dbw + self.tx_gain_dbi - self.tx_loss_db


@dataclass(frozen=True)
class RfLinkResult:
    """RF link outcome.  All dB quantities; rates in bit/s."""

    range_km: float
    eirp_dbw: float
    fspl_db: float
    other_loss_db: float
    c_over_n0_dbhz: float
    esn0_db: float
    shannon_capacity_bps: float
    achievable_rate_bps: float
    closes: bool

    def as_dict(self) -> dict[str, float | bool]:
        """All fields as a plain dict."""
        return asdict(self)


def rf_link(range_km: float, term: RfTerminal,
            extra_loss_db: float = 0.0) -> RfLinkResult:
    """Evaluate the RF link at ``range_km``, Eqs. (1)-(5).

    ``extra_loss_db`` adds a positive dB loss on top of ``term.other_loss_db``
    (used for the time-varying atmospheric term on a ground leg).

    ``closes`` is ``achievable_rate_bps >= 1`` bit/s, i.e. the link supports a
    non-trivial rate after the required Eb/N0 and the margin are honoured.
    """
    if extra_loss_db < 0.0:
        raise ValueError(f"extra_loss_db must be >= 0, got {extra_loss_db}")
    fspl = free_space_path_loss_db(range_km, term.frequency_hz)
    losses = term.other_loss_db + extra_loss_db
    c_n0 = (term.eirp_dbw - fspl - losses
            + term.rx_g_over_t_db_per_k - BOLTZMANN_DB_W_K_HZ)
    # Shannon bound, Eq. (4).
    snr_lin = 10.0 ** (c_n0 / 10.0) / term.bandwidth_hz
    shannon = term.bandwidth_hz * math.log2(1.0 + snr_lin)
    # Achievable rate from the modem requirement, Eq. (5).
    rate = 10.0 ** ((c_n0 - term.required_ebn0_db - term.margin_db) / 10.0)
    # Es/N0 at the occupied bandwidth, a diagnostic for the symbol-rate case.
    esn0 = c_n0 - 10.0 * math.log10(term.bandwidth_hz)
    return RfLinkResult(range_km=range_km, eirp_dbw=term.eirp_dbw, fspl_db=fspl,
                        other_loss_db=losses, c_over_n0_dbhz=c_n0, esn0_db=esn0,
                        shannon_capacity_bps=shannon, achievable_rate_bps=rate,
                        closes=rate >= 1.0)


@dataclass(frozen=True)
class OpticalTerminal:
    """Free-space optical terminal pair description.

    Attributes
    ----------
    tx_power_w : transmit optical power [W], > 0.
    wavelength_m : operating wavelength [m], > 0.
    beam_divergence_full_rad : FULL 1/e^2 divergence angle [rad], > 0.
    rx_aperture_diameter_m : receive clear aperture diameter [m], > 0.
    pointing_error_rad : radial pointing error [rad], >= 0.
    photons_per_bit : receiver sensitivity [photons/bit], > 0.  REQUIRED --
        no default is supplied, see the module docstring.
    tx_optics_loss_db, rx_optics_loss_db : optical train losses [dB], >= 0.
    margin_db : design margin [dB], >= 0.
    """

    tx_power_w: float
    wavelength_m: float
    beam_divergence_full_rad: float
    rx_aperture_diameter_m: float
    photons_per_bit: float
    pointing_error_rad: float = 0.0
    tx_optics_loss_db: float = 0.0
    rx_optics_loss_db: float = 0.0
    margin_db: float = 0.0

    def __post_init__(self) -> None:
        for fname in ("tx_power_w", "wavelength_m", "beam_divergence_full_rad",
                      "rx_aperture_diameter_m", "photons_per_bit"):
            if getattr(self, fname) <= 0.0:
                raise ValueError(f"{fname} must be > 0, got {getattr(self, fname)}")
        for fname in ("pointing_error_rad", "tx_optics_loss_db", "rx_optics_loss_db",
                      "margin_db"):
            if getattr(self, fname) < 0.0:
                raise ValueError(f"{fname} must be >= 0, got {getattr(self, fname)}")

    @property
    def theta_half_rad(self) -> float:
        """1/e^2 half divergence angle [rad]."""
        return self.beam_divergence_full_rad / 2.0


@dataclass(frozen=True)
class OpticalLinkResult:
    """Optical link outcome."""

    range_km: float
    beam_radius_m: float
    geometric_capture_fraction: float
    geometric_loss_db: float
    pointing_loss_db: float
    atmospheric_loss_db: float
    rx_power_w: float
    photon_energy_j: float
    achievable_rate_bps: float
    closes: bool

    def as_dict(self) -> dict[str, float | bool]:
        """All fields as a plain dict."""
        return asdict(self)


def optical_link(range_km: float, term: OpticalTerminal,
                 atmospheric_loss_db: float = 0.0) -> OpticalLinkResult:
    """Evaluate the optical link at ``range_km``, Eqs. (6)-(9)."""
    if range_km <= 0.0:
        raise ValueError(f"range_km must be > 0, got {range_km}")
    if atmospheric_loss_db < 0.0:
        raise ValueError(f"atmospheric_loss_db must be >= 0, got {atmospheric_loss_db}")
    r_m = range_km * 1e3
    w = term.theta_half_rad * r_m
    a = term.rx_aperture_diameter_m / 2.0
    f_geo = 1.0 - math.exp(-2.0 * a ** 2 / w ** 2)
    f_point = math.exp(-2.0 * (term.pointing_error_rad / term.theta_half_rad) ** 2)
    l_geo_db = _loss_db(f_geo)
    l_point_db = _loss_db(f_point)
    total_loss_db = (term.tx_optics_loss_db + l_geo_db + l_point_db
                     + atmospheric_loss_db + term.rx_optics_loss_db)
    p_rx = term.tx_power_w * 10.0 ** (-total_loss_db / 10.0)
    e_photon = PLANCK_J_S * SPEED_OF_LIGHT_M_S / term.wavelength_m
    rate = p_rx / (e_photon * term.photons_per_bit * 10.0 ** (term.margin_db / 10.0))
    return OpticalLinkResult(
        range_km=range_km, beam_radius_m=w, geometric_capture_fraction=f_geo,
        geometric_loss_db=l_geo_db, pointing_loss_db=l_point_db,
        atmospheric_loss_db=atmospheric_loss_db, rx_power_w=p_rx,
        photon_energy_j=e_photon, achievable_rate_bps=rate, closes=rate >= 1.0)


def kim_specific_attenuation_db_km(visibility_km: float, wavelength_nm: float) -> float:
    """Kim-Kruse visibility model specific attenuation [dB/km].

    Kim, McArthur & Korevaar, Proc. SPIE 4214 (2001).  The attenuation
    coefficient is

        beta[dB/km] = (3.912 / V) * (lambda / 550 nm)^(-q)

    where ``V`` is visibility [km] and ``q`` is the piecewise size-distribution
    exponent of Kim et al.::

        V > 50 km                 q = 1.6
        6 km  < V <= 50 km        q = 1.3
        1 km  < V <= 6 km         q = 0.16 V + 0.34
        0.5 km < V <= 1 km        q = V - 0.5
        V <= 0.5 km               q = 0

    Units: ``visibility_km`` > 0 [km], ``wavelength_nm`` > 0 [nm], result in
    dB/km.  Validity: fog and haze; the model is empirical and the 3.912
    numerator follows from the 2 % contrast threshold definition of
    visibility.
    """
    if visibility_km <= 0.0:
        raise ValueError(f"visibility_km must be > 0, got {visibility_km}")
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm}")
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
    return (3.912 / v) * (wavelength_nm / 550.0) ** (-q)


def slant_path_attenuation_db(zenith_attenuation_db: float, elevation_deg: float,
                              min_elevation_deg: float = 10.0) -> float:
    """Plane-parallel (cosecant) slant-path scaling, Eq. (8).

    ``A(eps) = A_zenith / sin(eps)`` (Ippolito 2008, Ch. 4).  Raises below
    ``min_elevation_deg``, where the plane-parallel assumption fails, rather
    than returning a number the model does not support.
    """
    if zenith_attenuation_db < 0.0:
        raise ValueError(
            f"zenith_attenuation_db must be >= 0, got {zenith_attenuation_db}")
    if elevation_deg < min_elevation_deg:
        raise ValueError(
            f"elevation {elevation_deg} deg is below min_elevation_deg "
            f"{min_elevation_deg} deg: the plane-parallel cosecant law of "
            f"Ippolito 2008 Ch. 4 is not valid there and over-predicts "
            f"attenuation; use an ITU-R P-series model (see the itur package)")
    if elevation_deg > 90.0:
        raise ValueError(f"elevation_deg must be <= 90, got {elevation_deg}")
    return zenith_attenuation_db / math.sin(math.radians(elevation_deg))


def hybrid_capacity_bps(range_km: float,
                        rf: RfTerminal | None = None,
                        optical: OpticalTerminal | None = None,
                        extra_rf_loss_db: float = 0.0,
                        optical_atmospheric_loss_db: float = 0.0) -> float:
    """Best achievable rate [bit/s] over whichever terminals are present.

    A hybrid terminal is modelled as selecting the better of the two legs at
    each instant: ``max`` over the available models.  This is the
    select-the-better-link policy, not a sum: the models describe independent
    radios sharing one pointing budget, and simultaneous operation of both is
    not assumed.  At least one of ``rf`` / ``optical`` must be given.
    """
    rates = []
    if rf is not None:
        rates.append(rf_link(range_km, rf, extra_loss_db=extra_rf_loss_db
                             ).achievable_rate_bps)
    if optical is not None:
        rates.append(optical_link(range_km, optical,
                                  atmospheric_loss_db=optical_atmospheric_loss_db
                                  ).achievable_rate_bps)
    if not rates:
        raise ValueError("hybrid_capacity_bps needs at least one of rf / optical")
    return max(rates)
