"""Declared covariate shift and its exact likelihood-ratio weights.

Weighted conformal prediction (Tibshirani, R.J., Barber, R.F., Candes, E.J. and
Ramdas, A., "Conformal Prediction Under Covariate Shift", *Advances in Neural
Information Processing Systems* 32, 2019) replaces exchangeability with the
weaker assumption that the test covariate density is ``dP_test/dP_cal = w(x)``
for a **known** likelihood ratio ``w``. Everything that package does rests on
that ratio being right, so this module makes it explicit and exact.

The shift this package declares moves the mean of two covariates and nothing
else: all-up mass and the headwind component. Both are Gaussian with a fixed
standard deviation under calibration and under test, so the likelihood ratio
has the closed form

    log w(x) = sum_j [ delta_j (x_j - mu_j) / sigma_j^2 - delta_j^2 / (2 sigma_j^2) ]

with ``delta_j`` the mean displacement of coordinate ``j``. The derivation is
one line of the Gaussian density ratio and is checked numerically against the
densities themselves in ``validation/validate_known_answers.py``.

The severity of a shift is reported as the Mahalanobis displacement
``sqrt(sum_j (delta_j / sigma_j)^2)``, which is the quantity that governs how
fast the weights degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CALIBRATION_MASS_MEAN = 6.00
"""Mean all-up mass under calibration [kg]."""
CALIBRATION_MASS_SD = 0.60
"""Standard deviation of all-up mass, calibration and test [kg]."""
CALIBRATION_HEADWIND_MEAN = 0.00
"""Mean headwind component under calibration [m/s]."""
CALIBRATION_HEADWIND_SD = 2.00
"""Standard deviation of the headwind component, calibration and test [m/s]."""

MASS_SHIFT_PER_SEVERITY = 0.18
"""Mass mean displacement per unit severity [kg], = 0.3 standard deviations."""
HEADWIND_SHIFT_PER_SEVERITY = 0.60
"""Headwind mean displacement per unit severity [m/s], = 0.3 standard deviations."""

SHIFT_LEVELS = (0.0, 1.0, 2.0, 3.0)
"""The in-distribution case and the three declared shifts of the audit."""


@dataclass(frozen=True)
class CovariateShift:
    """A declared Gaussian mean shift in mass and headwind.

    Attributes
    ----------
    severity:
        Dimensionless shift magnitude. ``0.0`` is the calibration
        distribution; ``1.0``, ``2.0`` and ``3.0`` are the three declared
        shifts of the audit.
    mass_sd:
        Standard deviation of mass [kg], identical under calibration and test.
    headwind_sd:
        Standard deviation of the headwind component [m/s], identical under
        calibration and test.
    """

    severity: float = 0.0
    mass_sd: float = CALIBRATION_MASS_SD
    headwind_sd: float = CALIBRATION_HEADWIND_SD

    def __post_init__(self) -> None:
        if self.severity < 0.0:
            raise ValueError(f"severity must be >= 0, got {self.severity}")
        if self.mass_sd <= 0.0:
            raise ValueError(f"mass_sd must be > 0 kg, got {self.mass_sd}")
        if self.headwind_sd <= 0.0:
            raise ValueError(f"headwind_sd must be > 0 m/s, got {self.headwind_sd}")

    @property
    def mass_delta(self) -> float:
        """Mass mean displacement ``delta_mass`` [kg]."""
        return MASS_SHIFT_PER_SEVERITY * self.severity

    @property
    def headwind_delta(self) -> float:
        """Headwind mean displacement ``delta_headwind`` [m/s]."""
        return HEADWIND_SHIFT_PER_SEVERITY * self.severity

    @property
    def mass_mean(self) -> float:
        """Test-time mean all-up mass [kg]."""
        return CALIBRATION_MASS_MEAN + self.mass_delta

    @property
    def headwind_mean(self) -> float:
        """Test-time mean headwind component [m/s]."""
        return CALIBRATION_HEADWIND_MEAN + self.headwind_delta

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis displacement ``sqrt(sum (delta_j / sigma_j)^2)`` [-]."""
        return float(
            np.hypot(self.mass_delta / self.mass_sd, self.headwind_delta / self.headwind_sd)
        )

    @property
    def is_identity(self) -> bool:
        """``True`` when the shift is the calibration distribution itself."""
        return self.mass_delta == 0.0 and self.headwind_delta == 0.0

    def scaled(self, fraction: float) -> CovariateShift:
        """Return the same shift with its severity multiplied by ``fraction``.

        Used to build a deliberately misspecified weight model: the data comes
        from ``self`` and the weights from ``self.scaled(f)`` with ``f != 1``.
        """
        if fraction < 0.0:
            raise ValueError(f"fraction must be >= 0, got {fraction}")
        return CovariateShift(
            severity=self.severity * fraction,
            mass_sd=self.mass_sd,
            headwind_sd=self.headwind_sd,
        )

    def log_likelihood_ratio(self, mass: np.ndarray, headwind: np.ndarray) -> np.ndarray:
        """Exact ``log w(x) = log dP_test/dP_cal`` [-].

        Parameters
        ----------
        mass:
            All-up mass [kg].
        headwind:
            Headwind component [m/s].

        Returns
        -------
        ndarray
            Natural logarithm of the likelihood ratio, broadcast over inputs.
        """
        m = np.asarray(mass, dtype=float)
        w = np.asarray(headwind, dtype=float)
        terms = 0.0
        for value, mean, sd, delta in (
            (m, CALIBRATION_MASS_MEAN, self.mass_sd, self.mass_delta),
            (w, CALIBRATION_HEADWIND_MEAN, self.headwind_sd, self.headwind_delta),
        ):
            terms = terms + delta * (value - mean) / sd**2 - delta**2 / (2.0 * sd**2)
        return np.asarray(terms, dtype=float)

    def likelihood_ratio(self, mass: np.ndarray, headwind: np.ndarray) -> np.ndarray:
        """Exact likelihood ratio ``w(x) >= 0`` [-]. See :meth:`log_likelihood_ratio`."""
        return np.exp(self.log_likelihood_ratio(mass, headwind))

    def describe(self) -> str:
        """One-line human-readable summary of the declared shift."""
        return (
            f"severity {self.severity:.3f}: mass {CALIBRATION_MASS_MEAN:.3f} -> "
            f"{self.mass_mean:.3f} kg (sd {self.mass_sd:.3f}), headwind "
            f"{CALIBRATION_HEADWIND_MEAN:.3f} -> {self.headwind_mean:.3f} m/s "
            f"(sd {self.headwind_sd:.3f}), Mahalanobis {self.mahalanobis:.4f}"
        )
