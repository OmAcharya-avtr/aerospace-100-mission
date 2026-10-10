"""Synthetic dataset generation for the energy-per-leg regression.

Nothing is downloaded and no data file is committed. Every array is produced by
this module from a ``numpy.random.Generator`` seeded in the caller, so each
number in ``validation/`` is reproduced exactly by re-running the script that
printed it. See ``DATASET_CARD.md``.

Observation model
-----------------
The noiseless target is :func:`conformalband.physics.leg_energy`. The observed
target is multiplicative Gaussian,

    E_obs = E_true * (1 + sigma_rel * z),     z ~ N(0, 1),

so the residual scale grows with the energy itself. That heteroscedasticity is
what makes the marginal split-conformal band the wrong width in the tails and
is the reason the Mondrian variant exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import DEFAULT_AIRFRAME, Airframe, leg_energy
from .shift import CALIBRATION_HEADWIND_SD, CALIBRATION_MASS_SD, CovariateShift

FEATURE_NAMES: tuple[str, ...] = (
    "airspeed_mps",
    "mass_kg",
    "air_density_kgm3",
    "headwind_mps",
    "distance_m",
)
"""Column order of the feature matrix, with units in the names."""

FEATURE_UNITS: tuple[str, ...] = ("m/s", "kg", "kg/m^3", "m/s", "m")

AIRSPEED_MEAN = 20.0
AIRSPEED_SD = 1.50
AIR_DENSITY_MEAN = 1.18
AIR_DENSITY_SD = 0.035
DISTANCE_MEAN = 1000.0
DISTANCE_SD = 80.0

NOISE_FRACTION = 0.06
"""Relative standard deviation of the multiplicative observation noise [-]."""

MASS_INDEX = FEATURE_NAMES.index("mass_kg")
HEADWIND_INDEX = FEATURE_NAMES.index("headwind_mps")


@dataclass(frozen=True)
class Dataset:
    """A generated sample of the energy-per-leg regression.

    Attributes
    ----------
    features:
        Shape ``(n, 5)`` float64, columns as in :data:`FEATURE_NAMES`.
    energy:
        Observed energy per leg [Wh], shape ``(n,)``.
    truth:
        Noiseless energy per leg [Wh], shape ``(n,)``.
    shift:
        The declared covariate shift the features were drawn under.
    """

    features: np.ndarray
    energy: np.ndarray
    truth: np.ndarray
    shift: CovariateShift

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {self.features.shape}"
            )
        if self.energy.shape != (self.features.shape[0],):
            raise ValueError("energy must have shape (n,) matching features")
        if self.truth.shape != self.energy.shape:
            raise ValueError("truth must have the same shape as energy")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def mass(self) -> np.ndarray:
        """All-up mass column [kg]."""
        return self.features[:, MASS_INDEX]

    @property
    def headwind(self) -> np.ndarray:
        """Headwind column [m/s]."""
        return self.features[:, HEADWIND_INDEX]


def sample_covariates(
    n_samples: int,
    rng: np.random.Generator,
    shift: CovariateShift | None = None,
) -> np.ndarray:
    """Draw ``n_samples`` covariate rows under a declared shift.

    Parameters
    ----------
    n_samples:
        Number of rows, positive.
    rng:
        Seeded generator; the only source of randomness.
    shift:
        Declared covariate shift. ``None`` means the calibration
        distribution.

    Returns
    -------
    ndarray
        Shape ``(n_samples, 5)``, columns as in :data:`FEATURE_NAMES`.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    shift = CovariateShift() if shift is None else shift
    airspeed = rng.normal(AIRSPEED_MEAN, AIRSPEED_SD, n_samples)
    mass = rng.normal(shift.mass_mean, CALIBRATION_MASS_SD, n_samples)
    density = rng.normal(AIR_DENSITY_MEAN, AIR_DENSITY_SD, n_samples)
    headwind = rng.normal(shift.headwind_mean, CALIBRATION_HEADWIND_SD, n_samples)
    distance = rng.normal(DISTANCE_MEAN, DISTANCE_SD, n_samples)
    return np.column_stack([airspeed, mass, density, headwind, distance])


def make_dataset(
    n_samples: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    severity: float = 0.0,
    shift: CovariateShift | None = None,
    airframe: Airframe = DEFAULT_AIRFRAME,
    noise_fraction: float = NOISE_FRACTION,
) -> Dataset:
    """Generate one dataset.

    Parameters
    ----------
    n_samples:
        Number of rows, positive.
    seed:
        Seed for a fresh generator. Ignored when ``rng`` is given.
    rng:
        Existing generator to draw from.
    severity:
        Shorthand for ``shift=CovariateShift(severity)``. Ignored when
        ``shift`` is given.
    shift:
        Declared covariate shift.
    airframe:
        Airframe coefficients used for the noiseless target.
    noise_fraction:
        Relative standard deviation of the multiplicative noise [-].

    Returns
    -------
    Dataset
        Features [see :data:`FEATURE_UNITS`], observed and noiseless energy
        [Wh].
    """
    if noise_fraction < 0.0:
        raise ValueError(f"noise_fraction must be >= 0, got {noise_fraction}")
    if rng is None:
        rng = np.random.default_rng(seed)
    shift = CovariateShift(severity=severity) if shift is None else shift
    features = sample_covariates(n_samples, rng, shift)
    truth = leg_energy(
        features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4], airframe
    )
    energy = truth * (1.0 + noise_fraction * rng.standard_normal(n_samples))
    return Dataset(features=features, energy=energy, truth=truth, shift=shift)


@dataclass(frozen=True)
class AuditSplit:
    """The three disjoint samples one audit replicate needs.

    ``fit`` trains the regressor, ``calibration`` is drawn from the
    calibration distribution and supplies the conformity scores, and ``test``
    is drawn under the declared shift.
    """

    fit: Dataset
    calibration: Dataset
    test: Dataset

    @property
    def shift(self) -> CovariateShift:
        """The declared shift the test sample was drawn under."""
        return self.test.shift


def make_audit_split(
    *,
    seed: int,
    n_fit: int = 1500,
    n_calibration: int = 500,
    n_test: int = 1000,
    severity: float = 0.0,
    airframe: Airframe = DEFAULT_AIRFRAME,
    noise_fraction: float = NOISE_FRACTION,
) -> AuditSplit:
    """Build one audit replicate: fit and calibration in distribution, test shifted.

    Parameters
    ----------
    seed:
        Seed for this replicate. One generator supplies all three samples, so
        the whole replicate is a deterministic function of ``seed``.
    n_fit, n_calibration, n_test:
        Sample sizes, each positive.
    severity:
        Declared shift severity applied to the test sample only.
    airframe:
        Airframe coefficients.
    noise_fraction:
        Relative observation noise [-].

    Returns
    -------
    AuditSplit
    """
    rng = np.random.default_rng(seed)
    kwargs = {"rng": rng, "airframe": airframe, "noise_fraction": noise_fraction}
    fit = make_dataset(n_fit, severity=0.0, **kwargs)
    calibration = make_dataset(n_calibration, severity=0.0, **kwargs)
    test = make_dataset(n_test, severity=severity, **kwargs)
    return AuditSplit(fit=fit, calibration=calibration, test=test)
