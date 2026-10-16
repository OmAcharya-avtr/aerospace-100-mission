"""Per-dwell detection and false-alarm probabilities for a photon-counting dwell.

Model
-----
A dwell of duration ``tau`` accumulates photo-events at rate
``lambda_s(rho) * I + lambda_b`` and declares a detection when the count
reaches a fixed threshold ``k``:

    lambda_s(rho) = lambda_s0 * exp(-2 rho^2 / w^2)        [s^-1]
    N ~ Poisson((I * lambda_s(rho) + lambda_b) * tau)      [counts]
    P_d = E_I[ P(N >= k) ],  P_fa = P(N >= k | lambda_s = 0)

Symbols, units and sources
--------------------------
``rho``  angular offset of the target from the dwell boresight [rad].
``w``    Gaussian-beam 1/e^2 half-width [rad]; the exp(-2 rho^2 / w^2) profile
         is the far-field TEM00 irradiance of Siegman, A. E. (1986) *Lasers*,
         University Science Books, Ch. 17.
``lambda_s0`` on-axis detected signal photo-event rate [s^-1].
``lambda_b``  background plus dark photo-event rate [s^-1].
``k``    integer count threshold [counts]; threshold detection on a Poisson
         count is the direct-detection photon-counting receiver of Gagliardi,
         R. M., & Karp, S. (1995) *Optical Communications*, 2nd ed., Wiley,
         Ch. 5. The detection/false-alarm pair as a function of a threshold on
         a count statistic is the classical radar formulation of Marcum, J. I.
         (1960), "A statistical theory of target detection by pulsed radar",
         *IRE Transactions on Information Theory* 6(2), 59-267.
``I``    normalised irradiance fluctuation, E[I] = 1, log-irradiance variance
         equal to the Rytov variance ``sigma_R^2``. The lognormal irradiance
         PDF in the weak-fluctuation regime is Andrews, L. C., & Phillips,
         R. L. (2005) *Laser Beam Propagation through Random Media*, 2nd ed.,
         SPIE Press, Ch. 9. Validity range: ``sigma_R^2 <~ 1``; beyond that the
         lognormal model is known to be wrong and this module does not pretend
         otherwise. ``sigma_R^2 = 0`` disables scintillation entirely.

The expectation over ``I`` is evaluated with Gauss-Hermite quadrature
(Abramowitz, M., & Stegun, I. A. (1964) *Handbook of Mathematical Functions*,
eq. 25.4.46), substituting ``ln I = -sigma_R^2 / 2 + sigma_R * sqrt(2) * x``
so that E[I] = 1 exactly.

Assumptions
-----------
Counts in disjoint dwells are independent; the irradiance is constant within a
dwell and independent between dwells (valid when the dwell is long compared
with the atmospheric coherence time but the dwells are separated by more than
it, which is a strong assumption and is recorded in README Limitations);
detector dead time, afterpulsing and blocking losses are not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def poisson_upper_tail(threshold: int, mean_counts: np.ndarray | float) -> np.ndarray | float:
    """P(N >= threshold) for N ~ Poisson(mean_counts).

    Exact finite sum ``1 - exp(-m) * sum_{j<k} m^j / j!`` evaluated with a
    multiplicative recurrence. Units: ``mean_counts`` in counts, result
    dimensionless. Valid for integer ``threshold >= 0`` and finite
    ``mean_counts >= 0``. Accuracy degrades for very large ``mean_counts``
    combined with large ``threshold`` because the tail is a difference of
    nearly equal quantities; the result is clipped to [0, 1].
    """
    k = int(threshold)
    if k < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold!r}")
    mean = np.asarray(mean_counts, dtype=float)
    if np.any(mean < 0.0) or not np.all(np.isfinite(mean)):
        raise ValueError("mean_counts must be finite and >= 0")
    if k == 0:
        out = np.ones_like(mean)
    else:
        term = np.exp(-mean)
        cdf = term.copy()
        for j in range(1, k):
            term = term * mean / j
            cdf = cdf + term
        out = np.clip(1.0 - cdf, 0.0, 1.0)
    if np.isscalar(mean_counts) or np.ndim(mean_counts) == 0:
        return float(out)
    return out


@dataclass(frozen=True)
class PoissonDwellDetector:
    """Photon-counting threshold detector with optional lognormal scintillation.

    Parameters
    ----------
    signal_rate_hz:
        On-axis detected signal photo-event rate ``lambda_s0`` [s^-1].
    background_rate_hz:
        Background plus dark photo-event rate ``lambda_b`` [s^-1].
    threshold_counts:
        Integer count threshold ``k`` [counts], ``k >= 1``.
    rytov_variance:
        Log-irradiance variance ``sigma_R^2`` [dimensionless], >= 0. Values
        above about 1 leave the validity range of the lognormal model.
    quadrature_nodes:
        Number of Gauss-Hermite nodes used for the irradiance average.
    """

    signal_rate_hz: float
    background_rate_hz: float
    threshold_counts: int
    rytov_variance: float = 0.0
    quadrature_nodes: int = 24

    def __post_init__(self) -> None:
        if not math.isfinite(self.signal_rate_hz) or self.signal_rate_hz < 0.0:
            raise ValueError(f"signal_rate_hz must be finite and >= 0, got {self.signal_rate_hz!r}")
        if not math.isfinite(self.background_rate_hz) or self.background_rate_hz < 0.0:
            raise ValueError("background_rate_hz must be finite and >= 0")
        if int(self.threshold_counts) != self.threshold_counts or self.threshold_counts < 1:
            raise ValueError(
                f"threshold_counts must be an integer >= 1, got {self.threshold_counts!r}"
            )
        if not math.isfinite(self.rytov_variance) or self.rytov_variance < 0.0:
            raise ValueError(f"rytov_variance must be finite and >= 0, got {self.rytov_variance!r}")
        if int(self.quadrature_nodes) < 2:
            raise ValueError("quadrature_nodes must be >= 2")

    def irradiance_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Hermite nodes ``I_h`` and normalised weights ``W_h`` for E[g(I)].

        ``sum(W) == 1`` to within rounding and ``sum(W * I) == 1`` to within the
        quadrature error of the lognormal mean. Dimensionless.
        """
        if self.rytov_variance == 0.0:
            return np.ones(1), np.ones(1)
        x, w = np.polynomial.hermite.hermgauss(int(self.quadrature_nodes))
        sigma = math.sqrt(self.rytov_variance)
        values = np.exp(-0.5 * self.rytov_variance + sigma * math.sqrt(2.0) * x)
        return values, w / math.sqrt(math.pi)

    def false_alarm_probability(self, dwell_s: float) -> float:
        """P(declare detection | no target in the beam), dimensionless.

        ``P(N >= k | Poisson(lambda_b * tau))``. Independent of the target, so
        no irradiance average is needed. Valid for ``dwell_s >= 0``.
        """
        if not math.isfinite(dwell_s) or dwell_s < 0.0:
            raise ValueError(f"dwell_s must be finite and >= 0, got {dwell_s!r}")
        return float(poisson_upper_tail(self.threshold_counts, self.background_rate_hz * dwell_s))

    def detection_probability(
        self,
        dwell_s: float,
        offset_rad: np.ndarray | float,
        beam_half_angle_rad: float,
    ) -> np.ndarray | float:
        """P(declare detection | target at angular offset ``offset_rad``).

        Units: ``dwell_s`` seconds, ``offset_rad`` and ``beam_half_angle_rad``
        radians, result dimensionless in [0, 1]. Valid for ``dwell_s >= 0`` and
        ``beam_half_angle_rad > 0``. Includes the background contribution, so
        ``detection_probability(tau, inf, w) == false_alarm_probability(tau)``.
        """
        if not math.isfinite(dwell_s) or dwell_s < 0.0:
            raise ValueError(f"dwell_s must be finite and >= 0, got {dwell_s!r}")
        if not math.isfinite(beam_half_angle_rad) or beam_half_angle_rad <= 0.0:
            raise ValueError("beam_half_angle_rad must be finite and > 0")
        offset = np.asarray(offset_rad, dtype=float)
        if np.any(offset < 0.0):
            raise ValueError("offset_rad must be >= 0")
        shape_factor = np.exp(-2.0 * (offset / beam_half_angle_rad) ** 2)
        irradiance, weights = self.irradiance_nodes()
        signal = self.signal_rate_hz * shape_factor[..., None] * irradiance
        mean = (signal + self.background_rate_hz) * dwell_s
        probability = (poisson_upper_tail(self.threshold_counts, mean) * weights).sum(axis=-1)
        if np.isscalar(offset_rad) or np.ndim(offset_rad) == 0:
            return float(probability)
        return probability

    def detection_probability_table(
        self,
        dwell_s: float,
        beam_half_angle_rad: float,
        max_offset_rad: float,
        n_points: int = 513,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tabulate ``P_d(rho)`` on a uniform grid for fast interpolation.

        Returns ``(offsets_rad, probabilities)``. The table is used by the
        continuous-geometry Monte Carlo. Linear interpolation on 1025 points
        out to five beam half-widths costs at most 1.96e-5 absolute in the
        reference configuration, measured on a 977-point probe at 1.096 beam
        half-widths off boresight; the measurement is reported in
        ``validation/validate_detection_model_output.txt`` and asserted in
        ``tests/test_detection.py``. The error is largest on the steep
        shoulder of the beam, near one half-width off boresight.
        """
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        offsets = np.linspace(0.0, float(max_offset_rad), int(n_points))
        values = np.asarray(
            self.detection_probability(dwell_s, offsets, beam_half_angle_rad), dtype=float
        )
        return offsets, values
