"""Strehl metrics and the adaptive-optics residual-error budget.

Strehl ratio
------------
Exact definition used here: the ratio of the on-axis far-field intensity of the
aberrated pupil to that of the same pupil with no aberration, for uniform
amplitude,

```
S = | (1/A) integral_A exp(i phi) dA |^2                 [-]
```

(Born, M. & Wolf, E. 1999, *Principles of Optics*, 7th ed., Cambridge
University Press, sec. 9.1.3). ``phi`` is optical phase [rad], piston-free.

The **extended Marechal approximation**

```
S ~= exp(-sigma^2)                                       [-]
```

follows from expanding the integral for small aberration and is exact in the
ensemble sense for zero-mean Gaussian phase (``<exp(i phi)> = exp(-sigma^2/2)``).
Born & Wolf give the second-order form ``S ~= 1 - sigma^2``; the exponential
form is the standard AO usage (Hardy 1998 eq. 4.6). *Validity:* commonly quoted
as ``sigma^2 <~ 1 rad^2`` / ``S >~ 0.1``. ``validation/validate_strehl.py``
measures the actual error against the exact integral over a range of
``sigma^2`` and reports where it exceeds 10 %.

Error budget
------------
Independent error terms add in variance (Hardy 1998 ch. 9):

```
sigma_total^2 = sigma_fit^2 + sigma_temp^2 + sigma_noise^2 + sigma_alias^2 + ...
```

* **Fitting error** -- spatial frequencies the DM cannot make:
  ``sigma_fit^2 = mu (d_act/r0)^(5/3)`` with ``mu = 0.34`` for a
  continuous-facesheet mirror (Hudgin, R. H. 1977, *JOSA* **67**, 393-395).
* **Temporal error** -- pure loop delay ``tau``:
  ``sigma_temp^2 = (tau/tau0)^(5/3)`` with ``tau0 = 0.314 r0/v``
  (Hardy 1998 eqs. 3.51, 9.60; Fried, D. L. 1990, "Time-delay-induced
  mean-square error in adaptive optics", *JOSA A* **7**, 1224-1225). An
  equivalent servo-bandwidth form is ``(f_G/f_3dB)^(5/3)`` with the Greenwood
  frequency ``f_G = 0.426 v/r0`` (Greenwood 1977, *JOSA* **67**, 390-393).
* **Noise error** -- sensor noise propagated through the reconstructor. There
  is no universal coefficient: it depends on the reconstructor. This module
  therefore computes it directly from the reconstruction matrix,
  ``sigma_noise^2 = sigma_s^2 * ||R||_F^2 / n_modes`` for white slope noise of
  variance ``sigma_s^2``, and reports the resulting *noise propagation
  coefficient*, which for zonal least-squares reconstructors grows like
  ``ln(n_sub)`` (Rigaut, F. & Gendron, E. 1992, "Laser guide star in adaptive
  optics: the tilt determination problem", *A&A* **261**, 677-684, discuss the
  reconstructor noise propagation; the logarithmic growth of zonal
  least-squares noise propagation is standard -- Hardy 1998 sec. 9.4).

Aliasing, scintillation, non-common-path and calibration errors are **not**
modelled; see README Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dm import HUDGIN_FITTING_COEFF
from .pupil import Pupil

__all__ = [
    "strehl_exact",
    "strehl_marechal",
    "marechal_inverse",
    "fitting_error",
    "temporal_error",
    "temporal_error_bandwidth",
    "noise_propagation_coefficient",
    "noise_error",
    "ErrorBudget",
]


def strehl_exact(phase: NDArray[np.float64], pupil: Pupil) -> float:
    """Exact on-axis Strehl ratio ``|<exp(i phi)>|^2`` [-] over the clear aperture.

    Parameters
    ----------
    phase:
        ``(n_grid, n_grid)`` optical phase [rad]. Piston is irrelevant to the
        modulus and is not removed.
    pupil:
        The pupil whose mask defines the integration area.

    Notes
    -----
    Uniform pupil amplitude is assumed (no apodisation, no scintillation).
    """
    arr = np.asarray(phase, dtype=np.float64)
    n = pupil.n_grid
    if arr.shape != (n, n):
        raise ValueError(f"phase must have shape {(n, n)}, got {arr.shape}")
    vals = arr[pupil.mask]
    return float(abs(np.mean(np.exp(1j * vals))) ** 2)


def strehl_marechal(variance: float) -> float:
    """Extended Marechal approximation ``S = exp(-sigma^2)`` [-].

    Parameters
    ----------
    variance:
        Piston-removed residual phase variance ``sigma^2`` [rad^2], >= 0.
    """
    v = float(variance)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"variance must be >= 0, got {variance!r}")
    return float(np.exp(-v))


def marechal_inverse(strehl: float) -> float:
    """Residual variance implied by a Strehl ratio, ``sigma^2 = -ln S`` [rad^2]."""
    s = float(strehl)
    if not (0.0 < s <= 1.0):
        raise ValueError(f"strehl must be in (0, 1], got {strehl!r}")
    return float(-np.log(s))


def fitting_error(actuator_pitch: float, r0: float, coefficient: float = HUDGIN_FITTING_COEFF):
    """Fitting error variance ``mu (d/r0)^(5/3)`` [rad^2] (Hudgin 1977)."""
    d = float(actuator_pitch)
    r = float(r0)
    if not np.isfinite(d) or d <= 0:
        raise ValueError(f"actuator_pitch must be > 0, got {actuator_pitch!r}")
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"r0 must be > 0, got {r0!r}")
    mu = float(coefficient)
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError(f"coefficient must be > 0, got {coefficient!r}")
    return float(mu * (d / r) ** (5.0 / 3.0))


def temporal_error(delay: float, tau0: float) -> float:
    """Pure-delay temporal error ``(tau/tau0)^(5/3)`` [rad^2] (Fried 1990)."""
    t = float(delay)
    t0 = float(tau0)
    if not np.isfinite(t) or t < 0:
        raise ValueError(f"delay must be >= 0, got {delay!r}")
    if not np.isfinite(t0) or t0 <= 0:
        raise ValueError(f"tau0 must be > 0, got {tau0!r}")
    return float((t / t0) ** (5.0 / 3.0))


def temporal_error_bandwidth(greenwood_freq: float, bandwidth_3db: float) -> float:
    """Servo-bandwidth temporal error ``(f_G/f_3dB)^(5/3)`` [rad^2] (Greenwood 1977)."""
    fg = float(greenwood_freq)
    fb = float(bandwidth_3db)
    if not np.isfinite(fg) or fg < 0:
        raise ValueError(f"greenwood_freq must be >= 0, got {greenwood_freq!r}")
    if not np.isfinite(fb) or fb <= 0:
        raise ValueError(f"bandwidth_3db must be > 0, got {bandwidth_3db!r}")
    return float((fg / fb) ** (5.0 / 3.0))


def noise_propagation_coefficient(
    reconstructor: NDArray[np.float64], influence: NDArray[np.float64], n_samples: int
) -> float:
    """Noise propagation from white slope noise to residual phase variance.

    Parameters
    ----------
    reconstructor:
        ``(n_actuators, n_slopes)`` matrix mapping slopes [rad/m] to DM
        commands [rad].
    influence:
        ``(n_samples, n_actuators)`` DM influence matrix [-].
    n_samples:
        Number of pupil samples the variance is averaged over [-].

    Returns
    -------
    float
        ``p`` such that ``sigma_noise^2 = p * sigma_s^2`` [m^2], where
        ``sigma_s^2`` is the per-slope noise variance [(rad/m)^2].

    Notes
    -----
    For independent slope noise of covariance ``sigma_s^2 I``, the command
    covariance is ``sigma_s^2 R R^T`` and the mean residual phase variance over
    the pupil is ``sigma_s^2 trace(F R R^T F^T) / n_samples`` with ``F`` the
    influence matrix. Piston is *not* removed here; the loop removes it.
    """
    r = np.asarray(reconstructor, dtype=np.float64)
    f = np.asarray(influence, dtype=np.float64)
    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if r.ndim != 2 or f.ndim != 2 or f.shape[1] != r.shape[0]:
        raise ValueError(
            f"shape mismatch: influence {f.shape} cannot multiply reconstructor {r.shape}"
        )
    fr = f @ r
    return float(np.sum(fr * fr) / n)


def noise_error(slope_variance: float, propagation: float) -> float:
    """Noise-induced residual variance ``sigma_noise^2 = p sigma_s^2`` [rad^2]."""
    v = float(slope_variance)
    p = float(propagation)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"slope_variance must be >= 0, got {slope_variance!r}")
    if not np.isfinite(p) or p < 0:
        raise ValueError(f"propagation must be >= 0, got {propagation!r}")
    return v * p


@dataclass(frozen=True)
class ErrorBudget:
    """Additive residual-variance budget and the Strehl it implies.

    All terms are in rad^2 of optical phase at the working wavelength, and are
    assumed statistically independent so that they add (Hardy 1998 ch. 9).
    """

    fitting: float
    temporal: float
    noise: float
    other: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fitting", "temporal", "noise", "other"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be a finite value >= 0, got {v!r}")
            object.__setattr__(self, name, v)

    @property
    def total(self) -> float:
        """Total residual variance [rad^2]."""
        return self.fitting + self.temporal + self.noise + self.other

    @property
    def strehl(self) -> float:
        """Marechal Strehl ratio implied by :attr:`total` [-]."""
        return strehl_marechal(self.total)

    def rms_wavefront(self, wavelength: float) -> float:
        """Residual wavefront RMS [m] = ``sqrt(total) * lambda / (2 pi)``."""
        lam = float(wavelength)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"wavelength must be > 0, got {wavelength!r}")
        return float(np.sqrt(self.total) * lam / (2.0 * np.pi))

    def as_dict(self) -> dict[str, float]:
        """Budget terms plus totals, for reporting."""
        return {
            "fitting": self.fitting,
            "temporal": self.temporal,
            "noise": self.noise,
            "other": self.other,
            "total": self.total,
            "strehl": self.strehl,
        }
