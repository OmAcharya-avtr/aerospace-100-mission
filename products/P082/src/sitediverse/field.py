"""The synthetic concurrent cloud record, and the ground truth it carries.

THE RECORD IS SYNTHETIC. No real site climatology is used anywhere in this
package, no real optical ground station is named as a data source, and not one
number in this repository is a measurement of the atmosphere. The record is
generated here, from declared constants and a declared integer seed, by the
model written out below. Its value is that the true spatial correlation
structure is KNOWN, so every estimator in this package can be scored against it
-- which is not possible with a real record, where the correlation length is
exactly the unknown. What that costs is stated in ``DATASET_CARD.md`` and in
README.md: nothing measured here transfers to a real network, because the
generator is the model, and a model that recovers its own generator has
demonstrated consistency and not skill.

The generator
-------------
**(4) Latent Gaussian field, space.** At each epoch the latent field over ``S``
stations is a zero-mean, unit-variance Gaussian vector with correlation matrix
``R(regime)``,

    R_ij(r) = rho_0 exp( -sqrt( (d_along,ij / L_along(r))^2
                               + (d_cross,ij / L_cross(r))^2 ) ),  i != j
    R_ii(r) = 1

with ``d_along`` and ``d_cross`` from equation (3). ``rho_0 < 1`` is a declared
nugget: it is the part of the station-to-station variability that is local to the
station and does not correlate with anything, which in a cloud field is the
sub-grid convective cell. Setting ``rho_0 = 1`` would make two co-located
stations perfectly redundant, which no measurement supports. The exponential
form is the Ornstein-Uhlenbeck / Matern-with-nu-one-half covariance; see Stein
1999 *Interpolation of Spatial Data*, chapter 2, and Gneiting & Guttorp 2006 for
the space-time separable construction used in equation (5).

**(5) Latent Gaussian field, time.** Persistence is a first-order
autoregression on the whole field,

    z_t = phi z_{t-1} + sqrt(1 - phi^2) e_t,   e_t ~ N(0, R(regime_t))

so that for stations ``i, j`` and epochs ``t, s`` in the same regime

    Corr(z_it, z_js) = R_ij phi^{|t - s|}

which is separable in space and time by construction. ``phi`` is declared per
three-hour epoch. The integral time scale implied is
``(1 + phi) / (1 - phi)`` epochs, which at ``phi = 0.72`` is 6.14 epochs or 18.4
hours -- the right order for a synoptic cloud system, and the reason the
bootstrap of ``sitediverse.empirical`` must be a BLOCK bootstrap.

**(6) The cloud indicator.** The binary cloud-cover indicator is the latent
field thresholded against a climatological probit level,

    c_it = 1  if  z_it < Phi^{-1}( q_it )
    q_it = Phi( b_i + A_i cos(2 pi t / T_year - psi_i)
                    + A_d cos(2 pi t / T_day) )

so the marginal cloud probability at station ``i`` and epoch ``t`` is EXACTLY
``q_it`` and is known. ``b_i`` sets the station's long-term level, ``A_i`` and
``psi_i`` its seasonal amplitude and phase -- shared within a regional group,
because two stations 100 km apart do not have opposite seasonal cycles -- and
``A_d`` a common diurnal term. The modulation is applied in PROBIT space, not to
``q`` directly, so that ``q_it`` stays strictly inside ``(0, 1)`` for every
amplitude and no clipping is needed. This threshold-a-Gaussian construction is
the Gaussian-copula binary field of Nelsen 2006 chapter 2 and Sklar 1959, read
in the forward direction; ``sitediverse.copula`` reads it in the reverse
direction to estimate ``R`` from the record.

**A note on what equation (6) means for a floating-point test.** ``c_it`` is a
STRICT comparison of two rounded quantities. Two builds of this generator that
agree on ``z`` and on the threshold to within one unit in the last place can
still disagree on ``c`` at any epoch where the two sit within an ulp of each
other, and the number of such epochs grows with the record. No test in this
package asserts bit-exact equality of a cloud indicator. The tests assert that
the cloud RATE agrees to a tolerance, and allow an individual indicator to
differ only where ``z - threshold`` is inside that tolerance. See
``tests/test_field.py``.

Regimes
-------
The year is divided into four declared quarters and each carries its own
``(L_along, L_cross)``. This is the single feature of the record that an
isotropic two-parameter correlation-length model fitted on great-circle
separation cannot represent, and measuring what that costs is the point of the
learned component. The regime label is observable -- it is the date -- so using
it is not leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .geometry import SITES, Site, along_cross_matrices

#: Epochs per day. A three-hour sampling interval, the cadence of a reanalysis
#: or synoptic cloud-cover product. DECLARED.
EPOCHS_PER_DAY: int = 8

#: Epochs per declared year, 365 days of 8 epochs. No leap years: the record is
#: synthetic and a 365-day year makes the seasonal harmonic exactly periodic,
#: which keeps the climatology estimator's season bins equal in width.
EPOCHS_PER_YEAR: int = 365 * EPOCHS_PER_DAY

#: Number of declared seasonal regimes. Four quarters.
REGIME_COUNT: int = 4


@dataclass(frozen=True)
class FieldSpec:
    """Declared generator constants. Every one of these is the ground truth.

    Parameters
    ----------
    nugget:
        ``rho_0`` of equation (4), dimensionless, in ``(0, 1]``.
    along_km, cross_km:
        ``L_along(r)``, ``L_cross(r)`` of equation (4) in kilometres, one per
        regime. Length ``REGIME_COUNT``.
    persistence:
        ``phi`` of equation (5), dimensionless, in ``[0, 1)``.
    base_rate_low, base_rate_high:
        The long-term cloud rate of the first and last station; the rest are
        spaced linearly between them. Dimensionless probabilities.
    seasonal_amplitude:
        ``A_i`` of equation (6) per regional group, in probit units.
    seasonal_phase:
        ``psi_i`` of equation (6) per regional group, in radians.
    diurnal_amplitude:
        ``A_d`` of equation (6), in probit units, common to all stations.
    years:
        Record length in declared years.
    seed:
        Integer seed of ``numpy.random.default_rng``. The record is a
        deterministic function of this spec and nothing else.
    """

    nugget: float
    along_km: tuple[float, ...]
    cross_km: tuple[float, ...]
    persistence: float
    base_rate_low: float
    base_rate_high: float
    seasonal_amplitude: tuple[float, ...]
    seasonal_phase: tuple[float, ...]
    diurnal_amplitude: float
    years: int
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 < self.nugget <= 1.0:
            raise ValueError(f"nugget must lie in (0, 1], got {self.nugget!r}")
        if len(self.along_km) != REGIME_COUNT or len(self.cross_km) != REGIME_COUNT:
            raise ValueError(
                f"along_km and cross_km must each have {REGIME_COUNT} entries, "
                f"got {len(self.along_km)} and {len(self.cross_km)}"
            )
        if any(length <= 0.0 for length in (*self.along_km, *self.cross_km)):
            raise ValueError("every correlation length must be strictly positive")
        if not 0.0 <= self.persistence < 1.0:
            raise ValueError(
                f"persistence must lie in [0, 1), got {self.persistence!r}"
            )
        for name, rate in (
            ("base_rate_low", self.base_rate_low),
            ("base_rate_high", self.base_rate_high),
        ):
            if not 0.0 < rate < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {rate!r}")
        if self.diurnal_amplitude < 0.0:
            raise ValueError("diurnal_amplitude must be non-negative")
        if self.years < 1:
            raise ValueError(f"years must be at least 1, got {self.years!r}")

    @property
    def epochs(self) -> int:
        """Record length in epochs."""
        return self.years * EPOCHS_PER_YEAR

    @property
    def integral_time_scale_epochs(self) -> float:
        """``(1 + phi) / (1 - phi)``: the AR(1) integral time scale, epochs."""
        return (1.0 + self.persistence) / (1.0 - self.persistence)

    def isotropic(self) -> FieldSpec:
        """The same spec with the anisotropy removed, regime by regime.

        Each regime's two lengths are replaced by their geometric mean, so the
        area of the declared correlation ellipse is preserved and the field
        becomes isotropic. This is the control record: against it an isotropic
        two-parameter model of ``sitediverse.correlation`` is CORRECTLY
        SPECIFIED, and any learned margin measured on it is a measure of the
        learned component's cost rather than of its skill.
        """
        geometric = tuple(
            float(np.sqrt(a * c))
            for a, c in zip(self.along_km, self.cross_km, strict=True)
        )
        return FieldSpec(
            nugget=self.nugget,
            along_km=geometric,
            cross_km=geometric,
            persistence=self.persistence,
            base_rate_low=self.base_rate_low,
            base_rate_high=self.base_rate_high,
            seasonal_amplitude=self.seasonal_amplitude,
            seasonal_phase=self.seasonal_phase,
            diurnal_amplitude=self.diurnal_amplitude,
            years=self.years,
            seed=self.seed + 1,
        )


#: The declared reference record: anisotropic, four seasonal regimes, 8 years at
#: a three-hour cadence. The correlation lengths run from 230 km across the axis
#: to 1250 km along it, a ratio of 5.4, which is at the strong end of what is
#: reported for cloud fields and is chosen so that the cost of an isotropic
#: model is measurable rather than marginal.
ANISOTROPIC = FieldSpec(
    nugget=0.95,
    along_km=(1250.0, 820.0, 560.0, 900.0),
    cross_km=(230.0, 300.0, 380.0, 270.0),
    persistence=0.72,
    base_rate_low=0.26,
    base_rate_high=0.54,
    seasonal_amplitude=(0.42, 0.30, 0.48, 0.36, 0.33),
    seasonal_phase=(0.0, 0.55, 2.60, 3.60, 1.80),
    diurnal_amplitude=0.17,
    years=8,
    seed=20260002,
)

#: The declared control record: the same, made isotropic regime by regime.
ISOTROPIC = ANISOTROPIC.isotropic()

#: A short record, used by the test suite wherever the claim under test does not
#: depend on record length.
SHORT = FieldSpec(
    nugget=0.95,
    along_km=(1250.0, 820.0, 560.0, 900.0),
    cross_km=(230.0, 300.0, 380.0, 270.0),
    persistence=0.72,
    base_rate_low=0.26,
    base_rate_high=0.54,
    seasonal_amplitude=(0.42, 0.30, 0.48, 0.36, 0.33),
    seasonal_phase=(0.0, 0.55, 2.60, 3.60, 1.80),
    diurnal_amplitude=0.17,
    years=2,
    seed=20260002,
)

#: The declared records, by name.
RECORD_SPECS: dict[str, FieldSpec] = {
    "anisotropic": ANISOTROPIC,
    "isotropic": ISOTROPIC,
    "short": SHORT,
}


def regime_labels(epochs: int) -> np.ndarray:
    """Declared seasonal regime of each epoch, integers in ``[0, 4)``.

    The year is cut into four equal quarters by epoch index. The label is a
    function of the calendar alone, so a model that uses it is using the date
    and not the record.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs!r}")
    day_of_year = (np.arange(epochs) % EPOCHS_PER_YEAR) / EPOCHS_PER_YEAR
    return np.minimum((day_of_year * REGIME_COUNT).astype(int), REGIME_COUNT - 1)


def true_correlation(
    spec: FieldSpec, regime: int, sites: tuple[Site, ...] = SITES
) -> np.ndarray:
    """Ground-truth correlation matrix ``R(regime)`` of equation (4).

    This is the generator's own matrix. It is available to the validation
    scripts, which score the estimators against it, and to nothing else: no
    estimator in this package may call it, and ``sitediverse.study`` does not.
    """
    if not 0 <= regime < REGIME_COUNT:
        raise ValueError(
            f"regime must lie in [0, {REGIME_COUNT}), got {regime!r}"
        )
    along, cross = along_cross_matrices(sites)
    scaled = np.sqrt(
        (along / spec.along_km[regime]) ** 2 + (cross / spec.cross_km[regime]) ** 2
    )
    matrix = spec.nugget * np.exp(-scaled)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def true_correlation_at(
    spec: FieldSpec, regime: int, along_km: np.ndarray, cross_km: np.ndarray
) -> np.ndarray:
    """Ground-truth correlation of equation (4) at arbitrary separations."""
    scaled = np.sqrt(
        (np.asarray(along_km, dtype=float) / spec.along_km[regime]) ** 2
        + (np.asarray(cross_km, dtype=float) / spec.cross_km[regime]) ** 2
    )
    return spec.nugget * np.exp(-scaled)


def true_thresholds(spec: FieldSpec, sites: tuple[Site, ...] = SITES) -> np.ndarray:
    """Ground-truth probit threshold of equation (6), shape ``(epochs, S)``.

    The marginal cloud probability is ``Phi`` of this array, exactly.
    """
    count = len(sites)
    epoch = np.arange(spec.epochs)
    base = norm.ppf(
        np.linspace(spec.base_rate_low, spec.base_rate_high, count)
    )
    cluster = np.array([site.cluster for site in sites])
    amplitude = np.asarray(spec.seasonal_amplitude, dtype=float)[cluster]
    phase = np.asarray(spec.seasonal_phase, dtype=float)[cluster]
    seasonal = amplitude * np.cos(
        2.0 * np.pi * epoch[:, None] / EPOCHS_PER_YEAR - phase
    )
    diurnal = spec.diurnal_amplitude * np.cos(
        2.0 * np.pi * epoch[:, None] / EPOCHS_PER_DAY
    )
    return base + seasonal + diurnal


@dataclass(frozen=True)
class Record:
    """A concurrent synthetic cloud record and the ground truth beside it.

    Attributes
    ----------
    spec:
        The ``FieldSpec`` that produced it.
    cloud:
        ``(epochs, S)`` boolean. ``True`` means cloud blocks the link.
    regime:
        ``(epochs,)`` integer regime label.
    latent:
        ``(epochs, S)`` float, the latent Gaussian field of equation (5).
        GROUND TRUTH: available to validation and used by no estimator.
    threshold_true:
        ``(epochs, S)`` float, the probit threshold of equation (6). GROUND
        TRUTH, same rule.
    sites:
        The station geometry the record was generated on.
    """

    spec: FieldSpec
    cloud: np.ndarray
    regime: np.ndarray
    latent: np.ndarray
    threshold_true: np.ndarray
    sites: tuple[Site, ...]

    @property
    def epochs(self) -> int:
        """Record length in epochs."""
        return int(self.cloud.shape[0])

    @property
    def site_count(self) -> int:
        """Number of stations."""
        return int(self.cloud.shape[1])

    @property
    def site_names(self) -> tuple[str, ...]:
        """Station names in column order."""
        return tuple(site.name for site in self.sites)

    def marginal_rate_true(self) -> np.ndarray:
        """Ground-truth long-term marginal cloud probability per station."""
        return norm.cdf(self.threshold_true).mean(axis=0)

    def margin_to_threshold(self) -> np.ndarray:
        """``z - threshold``: how far each indicator is from flipping.

        The quantity equation (6)'s strict comparison rounds away. A test that
        wants to allow an indicator to differ between two builds uses this to
        decide which indicators are entitled to.
        """
        return self.latent - self.threshold_true


def generate_record(
    spec: FieldSpec, sites: tuple[Site, ...] = SITES
) -> Record:
    """Generate the declared record of equations (4) to (6).

    Deterministic in ``spec.seed``: two calls with the same spec return
    bit-identical arrays in one build of numpy. Across builds, see the note in
    this module's docstring on what a thresholded Gaussian can and cannot
    promise.

    Returns
    -------
    Record
    """
    count = len(sites)
    epochs = spec.epochs
    regime = regime_labels(epochs)
    factors = [
        np.linalg.cholesky(true_correlation(spec, r, sites))
        for r in range(REGIME_COUNT)
    ]
    rng = np.random.default_rng(spec.seed)
    white = rng.standard_normal((epochs, count))
    innovation = np.empty_like(white)
    for r in range(REGIME_COUNT):
        mask = regime == r
        innovation[mask] = white[mask] @ factors[r].T
    latent = np.empty_like(innovation)
    latent[0] = innovation[0]
    scale = np.sqrt(1.0 - spec.persistence**2)
    for epoch in range(1, epochs):
        latent[epoch] = spec.persistence * latent[epoch - 1] + scale * innovation[epoch]
    threshold = true_thresholds(spec, sites)
    return Record(
        spec=spec,
        cloud=latent < threshold,
        regime=regime,
        latent=latent,
        threshold_true=threshold,
        sites=tuple(sites),
    )
