"""The measured uncertainty on the offset estimate, and its measured coverage.

Why this module is the point of the package
-------------------------------------------
A two-way time transfer implementation that reports an offset and quotes a
datasheet figure for its uncertainty has reported one number and asserted the
other. Everything here is measured from the exchange data that produced the
estimate, the one exception is explicit and labelled, and the resulting
interval's **coverage is itself measured** and published whether or not it is
good.

The budget
----------
Three independent contributions, combined in quadrature per JCGM 100:2008
(*Evaluation of measurement data -- Guide to the expression of uncertainty in
measurement*), section 5.1.2:

1. **Statistical**, ``u_stat``. The standard error of the fitted offset from
   the least-squares parameter covariance (:mod:`timetransfer.skew`),
   **inflated for serial correlation**. Queueing delay is correlated from one
   exchange to the next, so the ordinary standard error, which assumes
   independence, is optimistic. The inflation uses the effective sample size

       n_eff = n * (1 - r1) / (1 + r1)

   with ``r1`` the lag-1 autocorrelation of the fit residuals -- the standard
   AR(1) result of Bayley and Hammersley, "The effective number of independent
   observations in an autocorrelated time series", *Journal of the Royal
   Statistical Society, Supplement* 8(2):184-197 (1946), section 2. The factor
   applied is ``sqrt(n / n_eff)``, and it is reported separately so a reader
   can see how much of the final figure it is.

2. **Quantisation**, ``u_quant``. ``delta / sqrt(12)`` per exchange from
   timestamp resolution (:mod:`timetransfer.timebase`), divided by
   ``sqrt(n_eff)`` for the fitted mean. Type B in the JCGM sense: it comes
   from the measured resolution of the timebase, not from its data.

3. **Asymmetry**, ``u_asym``. The irreducible one. Equation (5) of
   :mod:`timetransfer.twoway` says the offset error is ``A / 2`` exactly, and
   the static part of ``A`` is not observable from timestamps. There are
   exactly three honest things to do with it, and
   :class:`AsymmetrySource` names which one was done:

   * ``"unaccounted"`` -- no asymmetry term at all. ``u_asym = 0`` and the
     interval is a statement about repeatability, **not** about accuracy. The
     result carries ``accounted=False`` and every renderer prints a warning.
   * ``"declared_bound"`` -- the caller declares a bound ``B`` on ``|A|`` from
     external knowledge (a calibrated cable pair, a reciprocal path).
     ``u_asym = (B / 2) / sqrt(3)``, the standard deviation of a uniform
     distribution over the bound, JCGM 100:2008 section 4.3.7.
   * ``"estimated"`` -- an asymmetry corrector was applied and its
     **measured** held-out residual standard deviation is used. This is what
     :mod:`timetransfer.asym` produces, and its number comes from a test split
     the corrector never saw.

Coverage
--------
:func:`measure_coverage` takes the realised errors and the reported
uncertainties over many seeded scenarios and returns the fraction of scenarios
whose error fell inside ``k * u``, with an exact Clopper-Pearson interval on
that fraction (Clopper and Pearson, "The use of confidence or fiducial limits
illustrated in the case of the binomial", *Biometrika* 26(4):404-413, 1934).
The nominal coverage of ``k = 1.96`` is 0.95 under a normal error
distribution.

**The measured coverage of this package's offset uncertainty is reported in
README.md and validation/VALIDATION.md, including where it falls short of
nominal.** An interval whose coverage is not measured is a decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats

from .errors import ConfigurationError, InsufficientDataError
from .skew import LeastSquaresFit
from .timebase import offset_quantisation_uncertainty

__all__ = [
    "AsymmetrySource",
    "CoverageResult",
    "OffsetUncertainty",
    "effective_sample_size",
    "measure_coverage",
    "offset_uncertainty",
]


class AsymmetrySource(StrEnum):
    """How the asymmetry contribution to the uncertainty was obtained."""

    UNACCOUNTED = "unaccounted"
    DECLARED_BOUND = "declared_bound"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class EffectiveSampleSize:
    """Effective sample size of a serially correlated residual series.

    Attributes
    ----------
    n:
        Raw sample count.
    n_effective:
        ``n * (1 - r1) / (1 + r1)``, clipped to ``[1, n]``.
    lag1_autocorrelation:
        ``r1``, the lag-1 autocorrelation of the residuals.
    inflation:
        ``sqrt(n / n_effective)``, the factor applied to the standard error.
    clipped:
        ``True`` when ``n_effective`` hit a clip, which happens for negative
        ``r1`` (anti-correlated residuals, where the AR(1) formula would claim
        *more* independent samples than there are observations).
    """

    n: int
    n_effective: float
    lag1_autocorrelation: float
    inflation: float
    clipped: bool


def effective_sample_size(residuals: np.ndarray) -> EffectiveSampleSize:
    """Effective sample size of ``residuals`` under an AR(1) correlation model.

    Parameters
    ----------
    residuals:
        Fit residuals [s], at least 4 of them.

    Returns
    -------
    EffectiveSampleSize
    """
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if r.size < 4:
        raise InsufficientDataError(
            f"effective sample size needs at least 4 residuals, got {r.size}",
            n_usable=int(r.size),
            n_required=4,
        )
    centred = r - np.mean(r)
    denom = float(np.sum(centred * centred))
    if denom <= 0.0:
        return EffectiveSampleSize(
            n=int(r.size),
            n_effective=float(r.size),
            lag1_autocorrelation=0.0,
            inflation=1.0,
            clipped=False,
        )
    r1 = float(np.sum(centred[:-1] * centred[1:]) / denom)
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    raw = float(r.size) * (1.0 - r1) / (1.0 + r1)
    clipped = not (1.0 <= raw <= float(r.size))
    n_eff = float(np.clip(raw, 1.0, float(r.size)))
    return EffectiveSampleSize(
        n=int(r.size),
        n_effective=n_eff,
        lag1_autocorrelation=r1,
        inflation=float(np.sqrt(float(r.size) / n_eff)),
        clipped=clipped,
    )


@dataclass(frozen=True)
class OffsetUncertainty:
    """An offset estimate with its measured uncertainty budget.

    Attributes
    ----------
    offset_s:
        The offset estimate [s] at ``epoch_s``.
    epoch_s:
        Local-clock epoch the estimate belongs to [s].
    u_statistical_s, u_quantisation_s, u_asymmetry_s:
        The three contributions [s], each a standard uncertainty.
    u_combined_s:
        Quadrature sum of the three [s].
    coverage_factor:
        ``k``; the expanded uncertainty is ``k * u_combined_s``.
    expanded_s:
        ``k * u_combined_s`` [s].
    n, n_effective, lag1_autocorrelation, correlation_inflation:
        The serial-correlation accounting behind ``u_statistical_s``.
    asymmetry_source:
        Which of the three honest treatments was applied.
    resolution_ns:
        Timestamp resolution the quantisation term used [ns].
    accounted:
        ``False`` when ``asymmetry_source`` is ``UNACCOUNTED``. An interval
        with ``accounted=False`` is a repeatability statement, not an accuracy
        statement, and :meth:`as_text` says so.
    """

    offset_s: float
    epoch_s: float
    u_statistical_s: float
    u_quantisation_s: float
    u_asymmetry_s: float
    u_combined_s: float
    coverage_factor: float
    expanded_s: float
    n: int
    n_effective: float
    lag1_autocorrelation: float
    correlation_inflation: float
    asymmetry_source: AsymmetrySource
    resolution_ns: int

    @property
    def accounted(self) -> bool:
        """``True`` unless the asymmetry contribution was omitted entirely."""
        return self.asymmetry_source is not AsymmetrySource.UNACCOUNTED

    @property
    def dominant_term(self) -> str:
        """Name of the largest contribution."""
        terms = {
            "statistical": self.u_statistical_s,
            "quantisation": self.u_quantisation_s,
            "asymmetry": self.u_asymmetry_s,
        }
        return max(terms, key=lambda key: terms[key])

    def interval_s(self) -> tuple[float, float]:
        """``(offset - expanded, offset + expanded)`` [s]."""
        return self.offset_s - self.expanded_s, self.offset_s + self.expanded_s

    def as_text(self) -> str:
        """Human-readable budget, with the asymmetry caveat when it applies."""
        lines = [
            f"offset              : {self.offset_s:.9e} s at t = {self.epoch_s:.6f} s",
            f"u statistical       : {self.u_statistical_s:.6e} s "
            f"(x{self.correlation_inflation:.3f} for r1 = {self.lag1_autocorrelation:+.4f})",
            f"u quantisation      : {self.u_quantisation_s:.6e} s "
            f"(delta = {self.resolution_ns} ns)",
            f"u asymmetry         : {self.u_asymmetry_s:.6e} s "
            f"({self.asymmetry_source.value})",
            f"u combined          : {self.u_combined_s:.6e} s "
            f"(dominant: {self.dominant_term})",
            f"expanded (k={self.coverage_factor:.2f})   : {self.expanded_s:.6e} s",
            f"n / n_effective     : {self.n} / {self.n_effective:.1f}",
        ]
        if not self.accounted:
            lines.append(
                "WARNING             : no asymmetry term. This interval describes "
                "repeatability, not accuracy."
            )
        return "\n".join(lines)


def offset_uncertainty(
    fit: LeastSquaresFit,
    *,
    epoch_s: float | None = None,
    resolution_ns: int = 1,
    asymmetry_source: AsymmetrySource = AsymmetrySource.UNACCOUNTED,
    asymmetry_bound_s: float = 0.0,
    asymmetry_stderr_s: float = 0.0,
    coverage_factor: float = 1.96,
) -> OffsetUncertainty:
    """Build the offset uncertainty budget from a fit and the declared asymmetry.

    Parameters
    ----------
    fit:
        A least-squares clock-polynomial fit over the per-exchange offsets.
    epoch_s:
        Epoch to report the offset at [s]; defaults to the fit's own reference
        epoch, where the statistical uncertainty is smallest.
    resolution_ns:
        Timestamp resolution [ns], >= 1.
    asymmetry_source:
        Which treatment of the asymmetry applies.
    asymmetry_bound_s:
        Bound ``B`` on ``|A|`` [s], required and used only for
        :attr:`AsymmetrySource.DECLARED_BOUND`. Contributes
        ``(B / 2) / sqrt(3)``.
    asymmetry_stderr_s:
        Measured held-out residual standard deviation of an asymmetry-bias
        estimator [s], required and used only for
        :attr:`AsymmetrySource.ESTIMATED`.
    coverage_factor:
        ``k``, > 0. ``1.96`` is nominally 95 per cent under a normal error
        distribution; whether the realised coverage matches is measured by
        :func:`measure_coverage` and is not assumed here.

    Returns
    -------
    OffsetUncertainty
    """
    if int(resolution_ns) < 1:
        raise ConfigurationError(f"resolution_ns must be >= 1, got {resolution_ns!r}")
    if not (float(coverage_factor) > 0.0):
        raise ConfigurationError(f"coverage_factor must be > 0, got {coverage_factor!r}")
    source = AsymmetrySource(asymmetry_source)
    if source is AsymmetrySource.DECLARED_BOUND and not (float(asymmetry_bound_s) > 0.0):
        raise ConfigurationError(
            "asymmetry_source=declared_bound requires asymmetry_bound_s > 0; "
            "a bound of zero is a claim that the path is exactly reciprocal"
        )
    if source is AsymmetrySource.ESTIMATED and not (float(asymmetry_stderr_s) > 0.0):
        raise ConfigurationError(
            "asymmetry_source=estimated requires asymmetry_stderr_s > 0, measured on "
            "data the estimator did not see"
        )
    epoch = fit.t_ref_s if epoch_s is None else float(epoch_s)
    _, u_fit = fit.offset_at(epoch)
    ess = effective_sample_size(fit.residuals)
    u_stat = float(u_fit * ess.inflation)
    u_q_single = offset_quantisation_uncertainty(float(resolution_ns))
    u_quant = float(u_q_single / np.sqrt(ess.n_effective))
    if source is AsymmetrySource.DECLARED_BOUND:
        u_asym = float((float(asymmetry_bound_s) / 2.0) / np.sqrt(3.0))
    elif source is AsymmetrySource.ESTIMATED:
        u_asym = float(asymmetry_stderr_s)
    else:
        u_asym = 0.0
    u_comb = float(np.sqrt(u_stat**2 + u_quant**2 + u_asym**2))
    value, _ = fit.offset_at(epoch)
    return OffsetUncertainty(
        offset_s=float(value),
        epoch_s=epoch,
        u_statistical_s=u_stat,
        u_quantisation_s=u_quant,
        u_asymmetry_s=u_asym,
        u_combined_s=u_comb,
        coverage_factor=float(coverage_factor),
        expanded_s=float(coverage_factor) * u_comb,
        n=ess.n,
        n_effective=ess.n_effective,
        lag1_autocorrelation=ess.lag1_autocorrelation,
        correlation_inflation=ess.inflation,
        asymmetry_source=source,
        resolution_ns=int(resolution_ns),
    )


@dataclass(frozen=True)
class CoverageResult:
    """Measured coverage of a reported uncertainty against its nominal value.

    Attributes
    ----------
    n_trials, n_covered:
        Scenarios run and scenarios whose error fell inside ``k * u``.
    measured:
        ``n_covered / n_trials``.
    nominal:
        The coverage the factor ``k`` implies under a normal error
        distribution, ``2 * Phi(k) - 1``.
    lower, upper:
        Exact Clopper-Pearson interval on ``measured`` at
        ``interval_confidence``.
    interval_confidence:
        Confidence of the interval on the coverage itself.
    coverage_factor:
        The ``k`` used.
    median_normalised_error:
        Median of ``|error| / u``. Under a normal error distribution this is
        0.6745; much above means the uncertainty is optimistic, much below
        means it is conservative, and it is reported because it localises the
        failure better than the coverage fraction does.
    """

    n_trials: int
    n_covered: int
    measured: float
    nominal: float
    lower: float
    upper: float
    interval_confidence: float
    coverage_factor: float
    median_normalised_error: float

    @property
    def nominal_inside_interval(self) -> bool:
        """``True`` if the nominal coverage lies inside the measured interval."""
        return self.lower <= self.nominal <= self.upper

    @property
    def verdict(self) -> str:
        """``"calibrated"``, ``"optimistic"`` or ``"conservative"``."""
        if self.nominal_inside_interval:
            return "calibrated"
        return "optimistic" if self.measured < self.nominal else "conservative"

    def as_text(self) -> str:
        """Human-readable result."""
        return (
            f"coverage factor k      : {self.coverage_factor:.3f}\n"
            f"nominal coverage       : {self.nominal:.6f}\n"
            f"measured coverage      : {self.measured:.6f} "
            f"({self.n_covered}/{self.n_trials})\n"
            f"Clopper-Pearson {self.interval_confidence:.2f} : "
            f"[{self.lower:.6f}, {self.upper:.6f}]\n"
            f"median |error| / u     : {self.median_normalised_error:.6f} "
            f"(0.674490 under a normal error law)\n"
            f"verdict                : {self.verdict}"
        )


def measure_coverage(
    errors_s: np.ndarray,
    uncertainties_s: np.ndarray,
    *,
    coverage_factor: float = 1.96,
    interval_confidence: float = 0.95,
) -> CoverageResult:
    """Measured coverage of reported uncertainties against realised errors.

    Parameters
    ----------
    errors_s:
        Realised ``estimate - truth`` [s] for each scenario.
    uncertainties_s:
        The standard uncertainty reported for each scenario [s], all > 0.
    coverage_factor:
        ``k``; an error counts as covered when ``|error| <= k * u``.
    interval_confidence:
        Confidence of the Clopper-Pearson interval on the measured coverage.

    Returns
    -------
    CoverageResult
    """
    err = np.asarray(errors_s, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainties_s, dtype=np.float64).reshape(-1)
    if err.size != unc.size:
        raise ConfigurationError(
            f"errors and uncertainties must match in length: {err.size} vs {unc.size}"
        )
    if err.size < 2:
        raise InsufficientDataError(
            f"coverage needs at least 2 trials, got {err.size}",
            n_usable=int(err.size),
            n_required=2,
        )
    if np.any(unc <= 0.0):
        raise ConfigurationError("every reported uncertainty must be > 0 to measure coverage")
    k = float(coverage_factor)
    if not (k > 0.0):
        raise ConfigurationError(f"coverage_factor must be > 0, got {k!r}")
    conf = float(interval_confidence)
    if not (0.0 < conf < 1.0):
        raise ConfigurationError(f"interval_confidence must be in (0, 1), got {conf!r}")
    covered = int(np.sum(np.abs(err) <= k * unc))
    n = int(err.size)
    alpha = 1.0 - conf
    lower = (
        0.0 if covered == 0 else float(stats.beta.ppf(alpha / 2.0, covered, n - covered + 1))
    )
    upper = (
        1.0
        if covered == n
        else float(stats.beta.ppf(1.0 - alpha / 2.0, covered + 1, n - covered))
    )
    nominal = float(2.0 * stats.norm.cdf(k) - 1.0)
    return CoverageResult(
        n_trials=n,
        n_covered=covered,
        measured=float(covered) / n,
        nominal=nominal,
        lower=lower,
        upper=upper,
        interval_confidence=conf,
        coverage_factor=k,
        median_normalised_error=float(np.median(np.abs(err) / unc)),
    )
