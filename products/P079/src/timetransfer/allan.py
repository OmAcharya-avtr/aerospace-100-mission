"""Allan deviation over tau, with a confidence interval at each tau.

Why this is implemented here
----------------------------
``allantools`` is the established Python implementation and is the right
dependency for most uses. It is **not installed in the environment this
product was measured in**, and no number in this repository is taken from it.
The implementation below is therefore written out, and verified against the
closed-form Allan deviations of three synthetic processes
(:mod:`timetransfer.noise`) rather than against another implementation.

The estimator
-------------
Overlapping Allan variance from phase data ``x_i`` [s] on a uniform grid of
spacing ``tau0``, with ``tau = m * tau0``:

    sigma_y^2(tau) = 1 / (2 (N - 2m) tau^2)
                     * sum_{i=0}^{N-2m-1} (x_{i+2m} - 2 x_{i+m} + x_i)^2

This is equation (10) of Riley, *Handbook of Frequency Stability Analysis*,
NIST Special Publication 1065 (2008), section 5.2.4, and the overlapping
estimator is preferred over the non-overlapping one because it uses every
available second difference and so has more degrees of freedom at the same
``N``. The quantity is dimensionless (a fractional frequency deviation); the
input is in seconds and ``tau`` is in seconds.

Confidence intervals
--------------------
The Allan variance estimate is treated as chi-squared distributed with
``edf`` equivalent degrees of freedom, so that a ``1 - alpha`` interval for the
variance is

    [ edf * s^2 / chi2_{1-alpha/2, edf} ,  edf * s^2 / chi2_{alpha/2, edf} ]

and the deviation interval is its square root. ``edf`` depends on the noise
type and on ``m``; the simplified formulas of Riley, NIST SP 1065 (2008),
Table 3 -- after Howe, Allan and Barnes, "Properties of signal sources and
measurement methods", *Proceedings of the 35th Annual Frequency Control
Symposium*, 1981 -- are implemented in :func:`equivalent_degrees_of_freedom`
for white PM, white FM, flicker FM and random-walk FM.

**These are approximations and this package measures how good they are.**
``validation/validate_allan.py`` runs a Monte Carlo over seeded realisations
and reports the achieved coverage of the nominal 95 per cent interval for each
noise type. Where the achieved coverage misses the nominal, that is reported
as measured, including in the README.

Noise identification
--------------------
:func:`identify_noise` implements the lag-1 autocorrelation method of Riley,
NIST SP 1065 (2008), section 5.5.3 (after Riley and Greenhall, "Power law noise
identification using the lag 1 autocorrelation", *Proceedings of the 18th
European Frequency and Time Forum*, 2004): difference the data until the lag-1
autocorrelation ``r1`` falls below 0.25, then

    delta = r1 / (1 + r1),   p = -2 (delta + d),   alpha = p + 2

for phase data differenced ``d`` times, where ``alpha`` is the exponent of
``S_y(f) ~ f**alpha``. ``alpha = 2`` is white PM, ``1`` flicker PM, ``0`` white
FM, ``-1`` flicker FM, ``-2`` random-walk FM.

It is a *heuristic* and it is measured rather than trusted:
``validation/validate_allan.py`` reports its hit rate over seeded realisations
of the three processes. :func:`allan_table` takes an explicit ``noise_kind``
whenever the caller knows it, because an identified noise type that is wrong
produces a confidence interval that is wrong without anything looking wrong.

Validity range
--------------
Uniformly spaced phase data with no gaps, no deterministic polynomial (remove
offset, skew and drift first -- see :mod:`timetransfer.skew`), and
``m <= (N - 1) // 2``. Averaging factors beyond ``N // 5`` are permitted but
the interval at those tau is wide and the estimate is dominated by a handful of
second differences; :func:`octave_taus` stops at ``N // 5`` by default and says
so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .errors import ConfigurationError

__all__ = [
    "AllanPoint",
    "AllanTable",
    "LogSlopeFit",
    "NoiseIdentification",
    "allan_table",
    "equivalent_degrees_of_freedom",
    "fit_log_slope",
    "identify_noise",
    "octave_taus",
    "overlapping_allan_variance",
]

_EDF_KINDS: tuple[str, ...] = ("white_pm", "white_fm", "flicker_fm", "random_walk_fm")


def overlapping_allan_variance(x_s: np.ndarray, tau0_s: float, m: int) -> tuple[float, int]:
    """Overlapping Allan variance at ``tau = m * tau0``.

    Parameters
    ----------
    x_s:
        Phase (time-error) samples [s], uniformly spaced, length ``N >= 3``.
    tau0_s:
        Sample spacing [s], > 0.
    m:
        Averaging factor, ``1 <= m <= (N - 1) // 2``.

    Returns
    -------
    tuple
        ``(allan_variance, n_terms)`` where ``n_terms = N - 2m`` is the number
        of second differences used. The variance is dimensionless.
    """
    x = np.asarray(x_s, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 3:
        raise ConfigurationError(f"phase data needs at least 3 samples, got {n}")
    if not np.all(np.isfinite(x)):
        raise ConfigurationError("phase data contains non-finite samples")
    if not (float(tau0_s) > 0.0):
        raise ConfigurationError(f"tau0_s must be > 0, got {tau0_s!r}")
    m = int(m)
    if m < 1 or 2 * m > n - 1:
        raise ConfigurationError(
            f"averaging factor m must satisfy 1 <= m <= (N-1)//2 = {(n - 1) // 2}, got {m}"
        )
    tau = m * float(tau0_s)
    second = x[2 * m :] - 2.0 * x[m : n - m] + x[: n - 2 * m]
    n_terms = int(second.size)
    avar = float(np.sum(second * second) / (2.0 * n_terms * tau * tau))
    return avar, n_terms


def equivalent_degrees_of_freedom(noise_kind: str, n: int, m: int) -> float:
    """Equivalent degrees of freedom of an overlapping Allan variance estimate.

    Simplified formulas from Riley, NIST SP 1065 (2008), Table 3, after Howe,
    Allan and Barnes (1981). They are approximations; their achieved coverage
    is measured in ``validation/validate_allan.py``.

    Parameters
    ----------
    noise_kind:
        ``"white_pm"``, ``"white_fm"``, ``"flicker_fm"`` or
        ``"random_walk_fm"``.
    n:
        Number of phase samples, >= 4.
    m:
        Averaging factor, >= 1 and ``2m <= n - 1``.

    Returns
    -------
    float
        Equivalent degrees of freedom, clipped below at 1.0 because a
        chi-squared interval is undefined at fewer.
    """
    if noise_kind not in _EDF_KINDS:
        raise ConfigurationError(
            f"unknown noise kind {noise_kind!r} for edf; expected one of {_EDF_KINDS}"
        )
    n = int(n)
    m = int(m)
    if n < 4:
        raise ConfigurationError(f"n must be >= 4 for an edf estimate, got {n}")
    if m < 1 or 2 * m > n - 1:
        raise ConfigurationError(f"m must satisfy 1 <= m <= (n-1)//2, got {m}")
    nf = float(n)
    mf = float(m)
    if noise_kind == "white_pm":
        edf = (nf + 1.0) * (nf - 2.0 * mf) / (2.0 * (nf - mf))
    elif noise_kind == "white_fm":
        edf = (3.0 * (nf - 1.0) / (2.0 * mf) - 2.0 * (nf - 2.0) / nf) * (
            4.0 * mf * mf / (4.0 * mf * mf + 5.0)
        )
    elif noise_kind == "flicker_fm":
        if m == 1:
            edf = 2.0 * (nf - 2.0) / (2.3 * nf - 4.9)
        else:
            edf = 5.0 * nf * nf / (4.0 * mf * (nf + 3.0 * mf))
    else:  # random_walk_fm
        edf = ((nf - 2.0) / mf) * (
            ((nf - 1.0) ** 2 - 3.0 * mf * (nf - 1.0) + 4.0 * mf * mf) / ((nf - 3.0) ** 2)
        )
    return float(max(1.0, edf))


def octave_taus(n: int, *, max_m: int | None = None) -> list[int]:
    """Octave-spaced averaging factors ``1, 2, 4, 8, ...`` for ``n`` samples.

    The default ceiling is ``n // 5``, the usual rule of thumb for an Allan
    deviation worth plotting (Riley, NIST SP 1065, section 5.2.2): beyond it the
    estimate rests on a handful of second differences. Larger factors are
    available by passing ``max_m`` explicitly, up to ``(n - 1) // 2``.

    Parameters
    ----------
    n:
        Number of phase samples, >= 4.
    max_m:
        Largest averaging factor to include.
    """
    n = int(n)
    if n < 4:
        raise ConfigurationError(f"n must be >= 4, got {n}")
    hard_ceiling = (n - 1) // 2
    ceiling = hard_ceiling if max_m is None else min(int(max_m), hard_ceiling)
    if max_m is None:
        ceiling = min(ceiling, max(1, n // 5))
    if ceiling < 1:
        raise ConfigurationError(f"no valid averaging factor for n = {n}")
    out: list[int] = []
    m = 1
    while m <= ceiling:
        out.append(m)
        m *= 2
    return out


@dataclass(frozen=True)
class AllanPoint:
    """One ``tau`` of an Allan deviation table.

    Attributes
    ----------
    m:
        Averaging factor.
    tau_s:
        Averaging time [s].
    adev:
        Overlapping Allan deviation, dimensionless.
    avar:
        Overlapping Allan variance, dimensionless.
    n_terms:
        Second differences used, ``N - 2m``.
    edf:
        Equivalent degrees of freedom used for the interval.
    lower, upper:
        Bounds of the confidence interval on the deviation.
    confidence:
        Nominal coverage of the interval, e.g. ``0.95``.
    """

    m: int
    tau_s: float
    adev: float
    avar: float
    n_terms: int
    edf: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class AllanTable:
    """An Allan deviation over tau with its intervals and provenance.

    Attributes
    ----------
    points:
        One :class:`AllanPoint` per averaging factor.
    tau0_s:
        Sample spacing [s].
    n_samples:
        Phase samples used.
    noise_kind:
        Noise type the degrees of freedom were computed for.
    noise_kind_source:
        ``"declared"`` if the caller named it, ``"identified"`` if
        :func:`identify_noise` did.
    """

    points: tuple[AllanPoint, ...]
    tau0_s: float
    n_samples: int
    noise_kind: str
    noise_kind_source: str

    @property
    def tau_s(self) -> np.ndarray:
        """Averaging times [s]."""
        return np.asarray([p.tau_s for p in self.points], dtype=np.float64)

    @property
    def adev(self) -> np.ndarray:
        """Allan deviations, dimensionless."""
        return np.asarray([p.adev for p in self.points], dtype=np.float64)

    @property
    def lower(self) -> np.ndarray:
        """Lower interval bounds."""
        return np.asarray([p.lower for p in self.points], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        """Upper interval bounds."""
        return np.asarray([p.upper for p in self.points], dtype=np.float64)

    def as_text(self) -> str:
        """Fixed-width table, suitable for a validation output file."""
        head = (
            f"noise kind {self.noise_kind} ({self.noise_kind_source}), "
            f"N = {self.n_samples}, tau0 = {self.tau0_s:.6g} s\n"
            f"{'m':>7} {'tau [s]':>12} {'adev':>13} {'lower':>13} {'upper':>13} "
            f"{'edf':>9} {'terms':>7}"
        )
        rows = [
            f"{p.m:>7d} {p.tau_s:>12.6g} {p.adev:>13.6e} {p.lower:>13.6e} "
            f"{p.upper:>13.6e} {p.edf:>9.2f} {p.n_terms:>7d}"
            for p in self.points
        ]
        return "\n".join([head, *rows])


@dataclass(frozen=True)
class NoiseIdentification:
    """Result of :func:`identify_noise`.

    Attributes
    ----------
    kind:
        Nearest noise type for which an edf formula is implemented. Flicker PM
        has no edf formula here and is reported as ``"white_pm"`` with
        ``alpha`` left at its measured value, so the caller can see that the
        mapping was approximate.
    alpha:
        Measured exponent of ``S_y(f) ~ f**alpha``.
    r1:
        Lag-1 autocorrelation at the differencing level that decided it.
    n_differences:
        How many times the data was differenced.
    """

    kind: str
    alpha: float
    r1: float
    n_differences: int


def _alpha_to_kind(alpha: float) -> str:
    """Nearest implemented edf noise kind for a measured ``alpha``."""
    if alpha >= 0.5:
        return "white_pm"
    if alpha >= -0.5:
        return "white_fm"
    if alpha >= -1.5:
        return "flicker_fm"
    return "random_walk_fm"


def identify_noise(x_s: np.ndarray, *, max_differences: int = 3) -> NoiseIdentification:
    """Identify the power-law noise type of phase data by lag-1 autocorrelation.

    Implements Riley, NIST SP 1065 (2008), section 5.5.3; see the module
    docstring for the arithmetic.

    Parameters
    ----------
    x_s:
        Phase samples [s], length >= 8.
    max_differences:
        Give up after this many differencing steps and report the steepest
        implemented kind.

    Returns
    -------
    NoiseIdentification
    """
    x = np.asarray(x_s, dtype=np.float64).reshape(-1)
    if x.size < 8:
        raise ConfigurationError(f"noise identification needs at least 8 samples, got {x.size}")
    series = x.copy()
    r1 = 0.0
    for d in range(int(max_differences) + 1):
        centred = series - np.mean(series)
        denom = float(np.sum(centred * centred))
        if denom <= 0.0:
            return NoiseIdentification(kind="white_pm", alpha=2.0, r1=0.0, n_differences=d)
        r1 = float(np.sum(centred[:-1] * centred[1:]) / denom)
        if r1 < 0.25:
            delta = r1 / (1.0 + r1)
            p = -2.0 * (delta + d)
            alpha = p + 2.0
            return NoiseIdentification(
                kind=_alpha_to_kind(alpha), alpha=float(alpha), r1=r1, n_differences=d
            )
        series = np.diff(series)
        if series.size < 4:
            break
    return NoiseIdentification(
        kind="random_walk_fm", alpha=-2.0, r1=r1, n_differences=int(max_differences)
    )


def allan_table(
    x_s: np.ndarray,
    tau0_s: float,
    *,
    m_values: list[int] | None = None,
    noise_kind: str | None = None,
    confidence: float = 0.95,
) -> AllanTable:
    """Overlapping Allan deviation over tau, with a confidence interval at each.

    Parameters
    ----------
    x_s:
        Phase samples [s], uniformly spaced.
    tau0_s:
        Sample spacing [s].
    m_values:
        Averaging factors; defaults to :func:`octave_taus`.
    noise_kind:
        Noise type for the degrees of freedom. When ``None``,
        :func:`identify_noise` chooses it and the table records that it was
        identified rather than declared.
    confidence:
        Nominal coverage of the interval, in ``(0, 1)``.

    Returns
    -------
    AllanTable
    """
    x = np.asarray(x_s, dtype=np.float64).reshape(-1)
    conf = float(confidence)
    if not (0.0 < conf < 1.0):
        raise ConfigurationError(f"confidence must be in (0, 1), got {conf!r}")
    if noise_kind is None:
        kind = identify_noise(x).kind
        source = "identified"
    else:
        if noise_kind not in _EDF_KINDS:
            raise ConfigurationError(
                f"unknown noise kind {noise_kind!r}; expected one of {_EDF_KINDS}"
            )
        kind = noise_kind
        source = "declared"
    ms = octave_taus(x.size) if m_values is None else [int(v) for v in m_values]
    alpha = 1.0 - conf
    points: list[AllanPoint] = []
    for m in ms:
        avar, n_terms = overlapping_allan_variance(x, tau0_s, m)
        edf = equivalent_degrees_of_freedom(kind, x.size, m)
        chi_hi = float(stats.chi2.ppf(1.0 - alpha / 2.0, edf))
        chi_lo = float(stats.chi2.ppf(alpha / 2.0, edf))
        var_lo = edf * avar / chi_hi
        var_hi = edf * avar / chi_lo
        points.append(
            AllanPoint(
                m=int(m),
                tau_s=float(m) * float(tau0_s),
                adev=float(np.sqrt(avar)),
                avar=float(avar),
                n_terms=n_terms,
                edf=edf,
                lower=float(np.sqrt(var_lo)),
                upper=float(np.sqrt(var_hi)),
                confidence=conf,
            )
        )
    return AllanTable(
        points=tuple(points),
        tau0_s=float(tau0_s),
        n_samples=int(x.size),
        noise_kind=kind,
        noise_kind_source=source,
    )


@dataclass(frozen=True)
class LogSlopeFit:
    """A straight-line fit in log-log space.

    Attributes
    ----------
    slope:
        Fitted exponent ``mu`` in ``y ~ x**mu``.
    slope_stderr:
        Standard error of the slope from the residual scatter.
    intercept:
        Fitted ``log(y)`` at ``log(x) = 0``.
    n:
        Points used.
    r_squared:
        Coefficient of determination of the log-log fit.
    """

    slope: float
    slope_stderr: float
    intercept: float
    n: int
    r_squared: float


def fit_log_slope(x: np.ndarray, y: np.ndarray) -> LogSlopeFit:
    """Least-squares slope of ``log(y)`` against ``log(x)``.

    Used to compare a measured Allan deviation against the theoretical
    exponent of a known noise process. Both inputs must be strictly positive.

    Parameters
    ----------
    x, y:
        Equal-length positive arrays, at least 3 points.
    """
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size != ya.size:
        raise ConfigurationError(f"x and y must match in length: {xa.size} vs {ya.size}")
    if xa.size < 3:
        raise ConfigurationError(f"need at least 3 points for a slope, got {xa.size}")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise ConfigurationError("log-log slope needs strictly positive x and y")
    lx = np.log(xa)
    ly = np.log(ya)
    design = np.column_stack([np.ones_like(lx), lx])
    coeffs, residuals, *_ = np.linalg.lstsq(design, ly, rcond=None)
    fitted = design @ coeffs
    resid = ly - fitted
    dof = xa.size - 2
    rss = float(np.sum(resid * resid))
    sigma2 = rss / dof if dof > 0 else 0.0
    cov = sigma2 * np.linalg.inv(design.T @ design)
    tss = float(np.sum((ly - np.mean(ly)) ** 2))
    return LogSlopeFit(
        slope=float(coeffs[1]),
        slope_stderr=float(np.sqrt(max(0.0, cov[1, 1]))),
        intercept=float(coeffs[0]),
        n=int(xa.size),
        r_squared=float(1.0 - rss / tss) if tss > 0.0 else float("nan"),
    )
