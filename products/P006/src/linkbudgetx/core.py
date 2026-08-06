"""Deterministic free-space optical (FSO) link-budget model.

Theory and conventions
----------------------
The link budget follows the standard FSO decomposition (Majumdar & Ricklin,
"Free-Space Laser Communications: Principles and Advances", Springer 2008,
Ch. 1; conceptually an optical adaptation of Friis 1946, Proc. IRE 34):

    P_rx[dBm] = P_tx[dBm] - L_tx_optics - L_geo - L_point - L_atm - L_rx_optics
    margin[dB] = P_rx[dBm] - S_rx[dBm]

where every L term is a positive loss in dB.

Angle convention (stated explicitly because conventions differ across texts):

- The input field ``beam_divergence_rad`` is the FULL divergence angle
  ``theta_full``. For a Gaussian beam this is the full angle between the
  1/e^2-intensity directions in the far field.
- The half angle is ``theta_half = theta_full / 2``. Gaussian far-field
  formulas below are written in terms of ``theta_half``, matching the
  standard Gaussian-beam result ``theta_half = lambda / (pi w0)``
  (Saleh & Teich, "Fundamentals of Photonics", Wiley, 2nd ed., Ch. 3,
  Gaussian beam optics).

Geometric spreading loss
------------------------
Far-field approximation (range R much greater than the Rayleigh range
``z_R = pi w0^2 / lambda``): the 1/e^2 beam radius at range R is

    w(R) ~= theta_half * R        [m]

- Gaussian profile: transverse intensity ``I(r) = I0 exp(-2 r^2 / w^2)``.
  A centred circular aperture of radius ``a`` captures the fraction

      f_geo = 1 - exp(-2 a^2 / w(R)^2)

  (Saleh & Teich, 2nd ed., Ch. 3 — power transmitted through a circular
  aperture of a Gaussian beam). For ``a << w`` this reduces to
  ``f_geo ~= 2 a^2 / w^2``.
- Flat-top profile: uniform intensity over a disc of radius
  ``theta_half * R``; a centred aperture captures

      f_geo = min(1, (a / (theta_half R))^2)

Then ``L_geo = -10 log10(f_geo)`` dB.

Pointing loss (Gaussian beam)
-----------------------------
The far-field angular intensity of a Gaussian beam is
``I(theta) = I0 exp(-2 theta^2 / theta_half^2)`` where ``theta_half`` is the
1/e^2 HALF-angle divergence (Saleh & Teich, 2nd ed., Ch. 3). A static radial
pointing offset ``theta_err`` therefore attenuates the received power by

    L_point(linear) = exp(-2 theta_err^2 / theta_half^2)
    L_point[dB]     = -10 log10(exp(-2 theta_err^2 / theta_half^2))
                    = (20 / ln 10) * (theta_err / theta_half)^2

This is the classic ``exp(-2 theta_err^2 / theta_div^2)`` form quoted in FSO
literature (e.g. Majumdar & Ricklin 2008, pointing-error treatment), in which
``theta_div`` denotes the 1/e^2 HALF angle. Because this library's input is
the FULL angle, it uses ``theta_half = beam_divergence_rad / 2`` inside the
formula. The same form is applied to the flat-top profile as a first-order
approximation (documented limitation).

Assumptions and validity
------------------------
- Far field only (R >> Rayleigh range). No near-field/defocus modelling.
- The pointing factor assumes a receiver small compared with the beam
  (``a << w``); for apertures comparable to the beam the truncation and
  offset couple and the product ``f_geo * f_point`` is only approximate.
- Atmospheric attenuation is a user-supplied specific attenuation in dB/km
  (e.g. from the Kim/Kruse visibility models, Kim, McArthur & Korevaar,
  Proc. SPIE 4214, 2001); no turbulence/scintillation model is included.
- Educational (validation Level 1); not for operational link sizing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .units import linear_to_db

__all__ = ["LinkBudget", "LinkBudgetResult", "BEAM_PROFILES"]

BEAM_PROFILES = ("gaussian", "flattop")


@dataclass(frozen=True)
class LinkBudgetResult:
    """All link-budget terms in dB, plus supporting quantities.

    Loss fields are positive dB losses. ``margin_db`` is received power minus
    receiver sensitivity; a positive margin means the link closes.
    """

    tx_power_dbm: float
    tx_optics_loss_db: float
    geometric_loss_db: float
    pointing_loss_db: float
    atmospheric_loss_db: float
    rx_optics_loss_db: float
    rx_power_dbm: float
    rx_sensitivity_dbm: float
    margin_db: float
    beam_radius_at_rx_m: float
    capture_fraction: float

    def as_dict(self) -> dict[str, float]:
        """Return all fields as a plain ``dict`` (all floats)."""
        return asdict(self)

    def format_table(self) -> str:
        """Return a fixed-width, human-readable budget table (str)."""
        rows = [
            ("Tx power", f"{self.tx_power_dbm:+.2f} dBm"),
            ("Tx optics loss", f"-{self.tx_optics_loss_db:.2f} dB"),
            ("Geometric spreading loss", f"-{self.geometric_loss_db:.2f} dB"),
            ("Pointing loss", f"-{self.pointing_loss_db:.2f} dB"),
            ("Atmospheric loss", f"-{self.atmospheric_loss_db:.2f} dB"),
            ("Rx optics loss", f"-{self.rx_optics_loss_db:.2f} dB"),
            ("Rx power", f"{self.rx_power_dbm:+.2f} dBm"),
            ("Rx sensitivity", f"{self.rx_sensitivity_dbm:+.2f} dBm"),
            ("Link margin", f"{self.margin_db:+.2f} dB"),
        ]
        info = [
            ("Beam radius at Rx (1/e^2)", f"{self.beam_radius_at_rx_m:.4g} m"),
            ("Aperture capture fraction", f"{self.capture_fraction:.4g}"),
        ]
        width = max(len(k) for k, _ in rows + info)
        lines = ["FSO link budget", "=" * (width + 16)]
        lines += [f"{k:<{width}}  {v:>12}" for k, v in rows]
        lines.append("-" * (width + 16))
        lines += [f"{k:<{width}}  {v:>12}" for k, v in info]
        return "\n".join(lines)


@dataclass
class LinkBudget:
    """Free-space optical link-budget inputs (units in field names).

    Attributes:
        tx_power_dbm: Transmit optical power [dBm].
        wavelength_nm: Operating wavelength [nm]. Must be > 0. Informational
            for the budget itself (the far-field model is parameterised by
            divergence, not wavelength); kept for documentation and future
            diffraction-limited divergence helpers. Typical FSO bands:
            780-1600 nm.
        beam_divergence_rad: FULL divergence angle [rad] (1/e^2 full angle for
            Gaussian). Must be > 0.
        range_km: Slant range transmitter->receiver [km]. Must be > 0.
        rx_aperture_diameter_m: Receiver aperture diameter [m]. Must be > 0.
        tx_optics_efficiency: Transmit-optics power throughput, in (0, 1].
        rx_optics_efficiency: Receive-optics power throughput, in (0, 1].
        pointing_error_rad: Static radial pointing offset [rad]. Must be >= 0.
        atmos_attenuation_db_per_km: Specific atmospheric attenuation
            [dB/km], >= 0 (user-supplied, e.g. from Kim et al. 2001 model).
        rx_sensitivity_dbm: Receiver sensitivity [dBm] at the required data
            rate / BER.
        beam_profile: ``"gaussian"`` or ``"flattop"``.
    """

    tx_power_dbm: float
    wavelength_nm: float
    beam_divergence_rad: float
    range_km: float
    rx_aperture_diameter_m: float
    rx_sensitivity_dbm: float
    tx_optics_efficiency: float = 1.0
    rx_optics_efficiency: float = 1.0
    pointing_error_rad: float = 0.0
    atmos_attenuation_db_per_km: float = 0.0
    beam_profile: str = "gaussian"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate all inputs; raise ``ValueError`` with an actionable message."""
        if not math.isfinite(self.tx_power_dbm):
            raise ValueError(f"tx_power_dbm must be finite, got {self.tx_power_dbm}.")
        if not (self.wavelength_nm > 0.0):
            raise ValueError(
                f"wavelength_nm must be > 0 (got {self.wavelength_nm}); typical FSO "
                "wavelengths are 780-1600 nm."
            )
        if not (self.beam_divergence_rad > 0.0):
            raise ValueError(
                f"beam_divergence_rad must be > 0 (got {self.beam_divergence_rad}); "
                "a zero-divergence beam is unphysical (diffraction limit, "
                "Saleh & Teich Ch. 3)."
            )
        if not (self.range_km > 0.0):
            raise ValueError(f"range_km must be > 0, got {self.range_km}.")
        if not (self.rx_aperture_diameter_m > 0.0):
            raise ValueError(
                f"rx_aperture_diameter_m must be > 0, got {self.rx_aperture_diameter_m}."
            )
        for name in ("tx_optics_efficiency", "rx_optics_efficiency"):
            val = getattr(self, name)
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val}.")
        if self.pointing_error_rad < 0.0:
            raise ValueError(
                f"pointing_error_rad must be >= 0 (radial offset magnitude), "
                f"got {self.pointing_error_rad}."
            )
        if self.atmos_attenuation_db_per_km < 0.0:
            raise ValueError(
                f"atmos_attenuation_db_per_km must be >= 0, "
                f"got {self.atmos_attenuation_db_per_km}."
            )
        if not math.isfinite(self.rx_sensitivity_dbm):
            raise ValueError(f"rx_sensitivity_dbm must be finite, got {self.rx_sensitivity_dbm}.")
        if self.beam_profile not in BEAM_PROFILES:
            raise ValueError(
                f"beam_profile must be one of {BEAM_PROFILES}, got {self.beam_profile!r}."
            )

    # -- individual physical terms -------------------------------------------------

    def beam_radius_at_rx_m(self) -> float:
        """1/e^2 beam radius (Gaussian) or geometric spot radius (flat-top) at Rx [m].

        Far-field: ``w(R) = theta_half * R`` with ``theta_half = full/2``
        (Saleh & Teich, 2nd ed., Ch. 3; valid for R >> Rayleigh range).
        """
        theta_half = self.beam_divergence_rad / 2.0
        return theta_half * self.range_km * 1e3

    def capture_fraction(self) -> float:
        """Fraction of transmitted power captured by the centred Rx aperture.

        Gaussian: ``1 - exp(-2 a^2 / w^2)``; flat-top: ``min(1, a^2/(theta_half R)^2)``.
        Dimensionless, in (0, 1].
        """
        a = self.rx_aperture_diameter_m / 2.0
        w = self.beam_radius_at_rx_m()
        if self.beam_profile == "gaussian":
            return 1.0 - math.exp(-2.0 * (a / w) ** 2)
        return min(1.0, (a / w) ** 2)

    def geometric_loss_db(self) -> float:
        """Geometric spreading loss ``-10 log10(capture_fraction)`` [dB, >= 0]."""
        return -linear_to_db(self.capture_fraction())

    def pointing_loss_db(self) -> float:
        """Pointing loss [dB, >= 0] for radial offset ``pointing_error_rad``.

        ``L_p = -10 log10( exp(-2 theta_err^2 / theta_half^2) )`` where
        ``theta_half = beam_divergence_rad / 2`` is the 1/e^2 HALF angle.
        Exactly 0 dB at zero pointing error. Gaussian far-field result;
        applied to flat-top as a first-order approximation.
        """
        theta_half = self.beam_divergence_rad / 2.0
        ratio2 = (self.pointing_error_rad / theta_half) ** 2
        # -10*log10(exp(-2 x)) = 20*x*log10(e); analytic form avoids underflow.
        return 20.0 * ratio2 * math.log10(math.e)

    def atmospheric_loss_db(self) -> float:
        """Atmospheric attenuation ``alpha[dB/km] * R[km]`` [dB, >= 0]."""
        return self.atmos_attenuation_db_per_km * self.range_km

    # -- full budget ----------------------------------------------------------------

    def compute(self) -> LinkBudgetResult:
        """Evaluate the full link budget.

        Returns:
            LinkBudgetResult: every intermediate term in dB and ``margin_db``.
        """
        tx_optics_loss = -linear_to_db(self.tx_optics_efficiency)
        rx_optics_loss = -linear_to_db(self.rx_optics_efficiency)
        geo = self.geometric_loss_db()
        point = self.pointing_loss_db()
        atm = self.atmospheric_loss_db()
        rx_power = self.tx_power_dbm - tx_optics_loss - geo - point - atm - rx_optics_loss
        return LinkBudgetResult(
            tx_power_dbm=self.tx_power_dbm,
            tx_optics_loss_db=tx_optics_loss,
            geometric_loss_db=geo,
            pointing_loss_db=point,
            atmospheric_loss_db=atm,
            rx_optics_loss_db=rx_optics_loss,
            rx_power_dbm=rx_power,
            rx_sensitivity_dbm=self.rx_sensitivity_dbm,
            margin_db=rx_power - self.rx_sensitivity_dbm,
            beam_radius_at_rx_m=self.beam_radius_at_rx_m(),
            capture_fraction=self.capture_fraction(),
        )

    def replace(self, **changes: float | str) -> "LinkBudget":
        """Return a validated copy with the given fields replaced."""
        params = asdict(self)
        params.update(changes)
        return LinkBudget(**params)  # type: ignore[arg-type]
