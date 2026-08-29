"""Adaptive-optics residual error budget and Strehl ratio.

The residual wavefront variance of a closed-loop AO system is conventionally
split into statistically independent terms that add in quadrature-of-variance:

    sigma^2_total = sigma^2_fit + sigma^2_temporal + sigma^2_noise + ...      (1)

Each term below carries its source, units, assumptions and validity range.
Independence is an approximation: fitting and aliasing errors in particular are
correlated in a real system (Rigaut, Veran & Lai, *Proc. SPIE* **3353**, 1038,
1998).  This package reports the terms separately and *also* measures the true
end-to-end residual by simulation, so the size of the approximation is visible
rather than assumed.

1. Fitting error
----------------
A DM with actuator pitch ``d_act`` cannot correct spatial frequencies above
roughly ``f_c = 1 / (2 d_act)``.  Integrating the Kolmogorov phase PSD above
that cut-off gives

    sigma^2_fit = a_F * (d_act / r0)^(5/3)                        [rad^2]     (2)

The ideal-spatial-filter value of ``a_F`` follows analytically from the PSD
(see :func:`ideal_filter_fitting_coefficient`) and is 0.2741.  The commonly
quoted engineering value is ``a_F = 0.28`` for a continuous face-sheet DM
(R. H. Hudgin, "Wave-front compensation error due to finite corrector-element
size", *J. Opt. Soc. Am.* **67**, 393-395, 1977; Hardy 1998 Table 6.1), with
values up to ~0.34 for other influence-function shapes.

*Validity:* ``d_act`` in the inertial subrange, ``d_act << L0``; a DM whose
influence functions extend beyond one pitch does better than the hard-cut-off
model, so the measured coefficient for a specific mirror can be lower.

2. Modal fitting error
----------------------
When correction is modal rather than zonal the residual after perfect
correction of the first ``J`` Zernike modes is Noll's ``Delta_J`` — see
:func:`waveforge.statistics.noll_residual_variance`.

3. Temporal error
-----------------
Pure delay ``tau``:

    sigma^2_delay = (tau / tau_0)^(5/3),  tau_0 = 0.314 r0 / v   [rad^2]     (3)

(D. L. Fried, *J. Opt. Soc. Am. A* **7**, 1224-1225, 1990.)  For a
continuously running first-order servo of 3 dB closed-loop bandwidth
``f_3dB``:

    sigma^2_bw = (f_G / f_3dB)^(5/3),  f_G = 0.427 v / r0        [rad^2]     (4)

(D. P. Greenwood, *J. Opt. Soc. Am.* **67**, 390-393, 1977.)  Equations (3)
and (4) describe different idealisations and must not be added together.

4. Noise error
--------------
Measurement noise enters the residual through the reconstructor and is shaped
by the loop.  For a slope noise variance ``sigma^2_s`` [rad^2/m^2], identical
and independent on every slope, and a linear reconstruction chain that maps
slopes to a pupil phase estimate through a matrix ``P`` (rows = pupil samples,
columns = slopes),

    sigma^2_noise = eta(g, d) * sigma^2_s * mean_over_pupil( diag(P P^T) )   (5)

with ``eta`` the closed-loop noise-variance amplification
:func:`waveforge.control.noise_variance_gain` (``g/(2-g)`` for a one-frame
delay).  This is exact for the stated linear model, so it is *computed*, not
approximated by a table coefficient.

5. Strehl ratio
---------------
The extended Marechal approximation:

    S ~= exp(-sigma^2_total)                                                 (6)

and the original quadratic form ``S ~= 1 - sigma^2`` (A. Marechal, *Rev. Opt.*
**26**, 257, 1947; Born & Wolf Sec. 9.1; Hardy 1998 Eq. 4.20).  Equation (6)
is exact for a zero-mean *Gaussian* phase whose variance is uniform over the
pupil, and the quadratic form is its two-term expansion, valid only for
``sigma^2 << 1``.  Conventional engineering practice restricts (6) to
``sigma^2 <~ 1 rad^2`` (``S >~ 0.37``); ``validation/`` measures where it
actually breaks for the phase statistics produced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statistics import KOLMOGOROV_PSD_CYCLIC, greenwood_frequency, greenwood_time_constant

__all__ = [
    "ErrorBudget",
    "bandwidth_error",
    "delay_error",
    "fitting_error",
    "ideal_filter_fitting_coefficient",
    "noise_error",
    "strehl_marechal",
    "strehl_marechal_quadratic",
    "variance_from_strehl",
]

#: Engineering value for a continuous face-sheet DM (Hudgin 1977; Hardy 1998).
HUDGIN_FITTING_COEFFICIENT: float = 0.28


def ideal_filter_fitting_coefficient() -> float:
    """``a_F`` for an ideal low-pass at ``f_c = 1/(2 d_act)``, derived not quoted.

    ``sigma^2 = int_{f>f_c} Phi(f) 2 pi f df = 0.0229 * 2 pi * (3/5) *
    (2 d_act / r0)^(5/3)``, i.e. ``a_F = 0.0229 * 2 pi * 0.6 * 2^(5/3)``.
    Evaluates to 0.2741, consistent with the quoted 0.28.
    """
    return float(KOLMOGOROV_PSD_CYCLIC * 2.0 * np.pi * 0.6 * 2.0 ** (5.0 / 3.0))


def fitting_error(
    actuator_pitch_m: float,
    r0_m: float,
    coefficient: float = HUDGIN_FITTING_COEFFICIENT,
) -> float:
    """DM fitting-error variance, Eq. (2) [rad^2]."""
    if not np.isfinite(actuator_pitch_m) or actuator_pitch_m <= 0.0:
        raise ValueError(f"actuator_pitch_m must be finite and > 0, got {actuator_pitch_m!r}")
    if not np.isfinite(r0_m) or r0_m <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0, got {r0_m!r}")
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError(f"coefficient must be finite and > 0, got {coefficient!r}")
    return float(coefficient * (actuator_pitch_m / r0_m) ** (5.0 / 3.0))


def delay_error(delay_s: float, r0_m: float, wind_speed_m_s: float) -> float:
    """Pure-delay temporal error, Eq. (3) [rad^2]."""
    if not np.isfinite(delay_s) or delay_s < 0.0:
        raise ValueError(f"delay_s must be finite and >= 0, got {delay_s!r}")
    tau0 = greenwood_time_constant(r0_m, wind_speed_m_s)
    return float((delay_s / tau0) ** (5.0 / 3.0))


def bandwidth_error(bandwidth_hz: float, r0_m: float, wind_speed_m_s: float) -> float:
    """Servo-bandwidth temporal error, Eq. (4) [rad^2]."""
    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth_hz must be finite and > 0, got {bandwidth_hz!r}")
    f_g = greenwood_frequency(r0_m, wind_speed_m_s)
    return float((f_g / bandwidth_hz) ** (5.0 / 3.0))


def noise_error(
    slope_noise_sigma: float,
    propagation_matrix: np.ndarray,
    noise_gain: float = 1.0,
) -> float:
    """Noise-induced residual phase variance, Eq. (5) [rad^2].

    Parameters
    ----------
    slope_noise_sigma:
        Per-slope one-sigma measurement noise [rad/m], ``>= 0``.
    propagation_matrix:
        ``P`` with shape ``(n_pupil_samples, n_slopes)``, mapping a slope
        vector to a pupil phase estimate [rad per (rad/m)].
    noise_gain:
        Closed-loop noise-variance amplification ``eta``, ``>= 0``
        (``1`` for open-loop reconstruction).
    """
    if not np.isfinite(slope_noise_sigma) or slope_noise_sigma < 0.0:
        raise ValueError(f"slope_noise_sigma must be finite and >= 0, got {slope_noise_sigma!r}")
    if not np.isfinite(noise_gain) or noise_gain < 0.0:
        raise ValueError(f"noise_gain must be finite and >= 0, got {noise_gain!r}")
    p = np.asarray(propagation_matrix, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"propagation_matrix must be 2-D, got shape {p.shape}")
    # Piston is unobservable and is removed from the residual, so remove the
    # mean of each column before computing the propagated variance.
    p = p - p.mean(axis=0, keepdims=True)
    per_sample = np.einsum("ij,ij->i", p, p)
    return float(noise_gain * slope_noise_sigma**2 * per_sample.mean())


def strehl_marechal(variance_rad2: float | np.ndarray) -> np.ndarray:
    """Extended Marechal Strehl ``exp(-sigma^2)``, Eq. (6) — dimensionless."""
    var = np.asarray(variance_rad2, dtype=float)
    if np.any(var < 0.0):
        raise ValueError("variance must be non-negative")
    return np.exp(-var)


def strehl_marechal_quadratic(variance_rad2: float | np.ndarray) -> np.ndarray:
    """Original quadratic Marechal Strehl ``1 - sigma^2`` (clipped at 0)."""
    var = np.asarray(variance_rad2, dtype=float)
    if np.any(var < 0.0):
        raise ValueError("variance must be non-negative")
    return np.maximum(1.0 - var, 0.0)


def variance_from_strehl(strehl: float | np.ndarray) -> np.ndarray:
    """Invert Eq. (6): ``sigma^2 = -ln(S)`` [rad^2]."""
    s = np.asarray(strehl, dtype=float)
    if np.any(s <= 0.0) or np.any(s > 1.0):
        raise ValueError("Strehl ratio must lie in (0, 1]")
    return -np.log(s)


@dataclass(frozen=True)
class ErrorBudget:
    """Additive AO residual error budget in rad^2, with a Strehl estimate.

    Attributes
    ----------
    fitting, temporal, noise:
        The three terms of Eq. (1) [rad^2], each ``>= 0``.
    other:
        Any additional independent term (aliasing, calibration, scintillation)
        the user wishes to include [rad^2].
    """

    fitting: float = 0.0
    temporal: float = 0.0
    noise: float = 0.0
    other: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fitting", "temporal", "noise", "other"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    @property
    def total(self) -> float:
        """Sum of all terms [rad^2]."""
        return float(self.fitting + self.temporal + self.noise + self.other)

    @property
    def rms_rad(self) -> float:
        """Total residual RMS wavefront [rad]."""
        return float(np.sqrt(self.total))

    @property
    def strehl(self) -> float:
        """Extended Marechal Strehl of the total, Eq. (6)."""
        return float(strehl_marechal(self.total))

    def as_dict(self) -> dict[str, float]:
        """Terms plus derived quantities, for reporting."""
        return {
            "fitting_rad2": float(self.fitting),
            "temporal_rad2": float(self.temporal),
            "noise_rad2": float(self.noise),
            "other_rad2": float(self.other),
            "total_rad2": self.total,
            "rms_rad": self.rms_rad,
            "strehl_marechal": self.strehl,
        }

    def dominant_term(self) -> str:
        """Name of the largest contributor."""
        terms = {
            "fitting": self.fitting,
            "temporal": self.temporal,
            "noise": self.noise,
            "other": self.other,
        }
        return max(terms, key=lambda k: terms[k])
