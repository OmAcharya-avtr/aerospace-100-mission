"""Fade-duration distribution fitting with a goodness-of-fit report.

The exponential fade-duration distribution is the textbook assumption and it
is usually wrong on a correlated fading channel. The reason is structural, not
statistical: an exponential duration is the hitting time of a memoryless
two-state process, and the excursions of a Gauss-Markov log-amplitude below a
level are not memoryless. A fade that has already lasted ten correlation
lengths is in a different part of the state space from one that has lasted one
sample, so its remaining duration is not distributed the same way.

This module therefore does three things that a bare ``scipy.stats.expon.fit``
does not:

1. It treats the durations as what they are -- **integer multiples of the
   sample period**. The natural one-parameter memoryless law on
   ``{1, 2, 3, ...}`` samples is the shifted geometric, not the exponential,
   and a chi-square test on the sample-count histogram is a valid test of it.
   A Kolmogorov-Smirnov test against a continuous law applied to heavily tied
   discrete data is not valid, and the module says so where it reports one.
2. It handles **right censoring** in the exponential maximum likelihood and in
   the empirical survival function, so a record that ends mid-fade does not
   silently bias the fit downwards.
3. It **reports the test result either way**. A rejected exponential is the
   expected outcome and is published as such.

Caveat stated once, applied everywhere
--------------------------------------
Every p-value below is computed with the distribution's parameters estimated
from the same data. That makes the Kolmogorov-Smirnov p-value
*anticonservative*: the true rejection rate under the null is lower than the
nominal one, so a KS test that already rejects would reject even more strongly
with the correct null distribution, while a KS test that does not reject is
weaker evidence than its p-value suggests. The chi-square statistic has the
estimated parameter subtracted from its degrees of freedom, which is the
standard first-order correction and is exact only asymptotically.

References
----------
S. O. Rice (1944, 1945), *Bell System Technical Journal* -- level crossings
and excursion durations.

E. L. Kaplan and P. Meier, "Nonparametric estimation from incomplete
observations", *Journal of the American Statistical Association*
53(282):457-481 (1958) -- the product-limit survival estimator implemented in
:func:`kaplan_meier_survival`.

Units
-----
Durations in seconds throughout, except :func:`fit_geometric_samples` and
:func:`geometric_chi_square` which take integer sample counts, because that is
the scale on which the discreteness lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "DistributionFit",
    "FitComparison",
    "compare_fade_duration_models",
    "exponential_mle_with_censoring",
    "fit_geometric_samples",
    "geometric_chi_square",
    "kaplan_meier_survival",
    "ks_fitted",
]


def _as_durations(durations: ArrayLike, *, name: str = "durations") -> NDArray[np.float64]:
    arr = np.asarray(durations, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty; nothing to fit")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} contains negative values; a duration cannot be negative")
    return arr


@dataclass(frozen=True)
class DistributionFit:
    """One fitted law with its goodness of fit.

    Attributes
    ----------
    name:
        Distribution name as reported.
    params:
        Fitted parameters, in the order the docstring of the fitting function
        states. Units: seconds where a scale, dimensionless where a shape.
    n_parameters:
        Number of free parameters estimated from the data (drives the AIC).
    log_likelihood:
        Log likelihood at the fitted parameters, including the censored
        contributions where the fit supports censoring.
    aic:
        ``2 k - 2 log L``.
    statistic, p_value, test:
        The goodness-of-fit statistic, its p-value and the name of the test.
    dof:
        Degrees of freedom for a chi-square test; ``None`` for KS.
    valid:
        ``False`` when the test could not be run validly (for example a
        chi-square with fewer than three usable bins). A ``False`` here means
        the p-value must not be quoted.
    notes:
        Any caveat that must travel with the number.
    """

    name: str
    params: tuple[float, ...]
    n_parameters: int
    log_likelihood: float
    aic: float
    statistic: float
    p_value: float
    test: str
    dof: int | None = None
    valid: bool = True
    notes: str = ""

    def line(self) -> str:
        """One fixed-width row for a report table."""
        dof = "-" if self.dof is None else str(self.dof)
        return (
            f"{self.name:<22s} {self.aic:>14.3f} {self.test:<14s} "
            f"{self.statistic:>12.6g} {dof:>5s} {self.p_value:>12.4g} "
            f"{'ok' if self.valid else 'INVALID':>8s}"
        )


def exponential_mle_with_censoring(
    complete_s: ArrayLike, censored_s: ArrayLike | None = None
) -> tuple[float, float]:
    """Exponential scale and log likelihood under right censoring.

    For ``n_c`` complete observations ``t_i`` and ``n_r`` right-censored lower
    bounds ``c_j``, the exponential log likelihood is

        log L = -n_c log(theta) - (sum t_i + sum c_j) / theta,

    maximised at ``theta = (sum t_i + sum c_j) / n_c``. Ignoring the censored
    observations entirely would divide the same numerator-minus-censored-time
    by the same ``n_c`` and therefore underestimate the mean.

    Parameters
    ----------
    complete_s:
        Fully observed durations, seconds.
    censored_s:
        Right-censored lower bounds, seconds. May be ``None`` or empty.

    Returns
    -------
    (theta, log_likelihood)
        ``theta`` in seconds.
    """
    t = _as_durations(complete_s, name="complete_s")
    c = (
        np.empty(0, dtype=np.float64)
        if censored_s is None
        else np.asarray(censored_s, dtype=np.float64).ravel()
    )
    if c.size and (not np.all(np.isfinite(c)) or np.any(c < 0.0)):
        raise ValueError("censored_s must be finite and non-negative")
    total = float(t.sum() + c.sum())
    n_c = int(t.size)
    if total <= 0.0:
        raise ValueError(
            "total observed time is zero; the exponential scale is not identified "
            "(this happens with duration_convention='interval_count' when every "
            "fade is a single sample)"
        )
    theta = total / n_c
    log_like = -n_c * math.log(theta) - total / theta
    return theta, log_like


def fit_geometric_samples(lengths: ArrayLike) -> tuple[float, float]:
    """Shifted-geometric MLE on fade lengths in **samples**.

    Model: ``P(L = k) = (1 - p)**(k - 1) p`` for ``k = 1, 2, 3, ...``. This is
    the exact memoryless law on a sampled record and the discrete analogue of
    the exponential; it is what a two-state Markov channel produces. The MLE is
    ``p = 1 / mean(L)``.

    Parameters
    ----------
    lengths:
        Integer fade lengths in samples, each ``>= 1``.

    Returns
    -------
    (p, log_likelihood)
    """
    arr = np.asarray(lengths).ravel()
    if arr.size == 0:
        raise ValueError("lengths is empty; nothing to fit")
    if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
        raise ValueError("lengths contains non-finite values")
    ints = np.asarray(np.rint(np.asarray(arr, dtype=np.float64)), dtype=np.int64)
    if np.any(np.abs(np.asarray(arr, dtype=np.float64) - ints) > 1e-9):
        raise ValueError("lengths must be integer sample counts")
    if np.any(ints < 1):
        raise ValueError("lengths must all be >= 1 samples")
    mean = float(ints.mean())
    p = 1.0 / mean
    log_like = float(np.sum((ints - 1) * math.log1p(-p) + math.log(p))) if p < 1.0 else 0.0
    return p, log_like


def geometric_chi_square(
    lengths: ArrayLike, p: float, *, min_expected: float = 5.0
) -> tuple[float, int, float, int, str]:
    """Pearson chi-square of fade lengths against a shifted geometric.

    Bins are the individual sample counts ``1, 2, 3, ...`` merged from the tail
    inwards until every bin has an expected count of at least
    ``min_expected``; the final bin is the open tail. Degrees of freedom are
    ``n_bins - 1 - 1``: one constraint for the total and one for the estimated
    ``p``.

    This is a *valid* test of the memoryless hypothesis on discrete data,
    unlike a Kolmogorov-Smirnov test against a continuous law, which assumes a
    continuous null and is distorted by ties.

    Returns
    -------
    (statistic, dof, p_value, n_bins, note)
        ``p_value`` is ``nan`` and ``note`` explains why when ``dof < 1``.
    """
    arr = np.asarray(np.rint(np.asarray(lengths, dtype=np.float64)), dtype=np.int64).ravel()
    if arr.size == 0:
        raise ValueError("lengths is empty")
    pp = float(p)
    if not 0.0 < pp <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p!r}")
    n = int(arr.size)
    kmax = int(arr.max())
    observed = np.bincount(arr, minlength=kmax + 1)[1:].astype(np.float64)
    k = np.arange(1, kmax + 1, dtype=np.float64)
    expected = n * ((1.0 - pp) ** (k - 1.0)) * pp
    # Open tail bin absorbs P(L > kmax).
    expected[-1] += n * (1.0 - pp) ** kmax

    # Merge from the tail inwards.
    obs_bins: list[float] = []
    exp_bins: list[float] = []
    acc_o = 0.0
    acc_e = 0.0
    for i in range(kmax - 1, -1, -1):
        acc_o += observed[i]
        acc_e += expected[i]
        if acc_e >= min_expected:
            obs_bins.append(acc_o)
            exp_bins.append(acc_e)
            acc_o = 0.0
            acc_e = 0.0
    if acc_e > 0.0:
        if obs_bins:
            obs_bins[-1] += acc_o
            exp_bins[-1] += acc_e
        else:
            obs_bins.append(acc_o)
            exp_bins.append(acc_e)
    o = np.asarray(obs_bins[::-1], dtype=np.float64)
    e = np.asarray(exp_bins[::-1], dtype=np.float64)
    n_bins = int(o.size)
    stat = float(np.sum((o - e) ** 2 / e))
    dof = n_bins - 2
    if dof < 1:
        return stat, dof, float("nan"), n_bins, (
            f"only {n_bins} bin(s) survive min_expected={min_expected}; "
            "chi-square degrees of freedom < 1, p-value not computable"
        )
    p_value = float(stats.chi2.sf(stat, dof))
    return stat, dof, p_value, n_bins, ""


def ks_fitted(
    durations: ArrayLike, dist: stats.rv_continuous, params: tuple[float, ...]
) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic via the probability-integral transform.

    The data are mapped through the fitted CDF and tested against the uniform,
    which avoids the ``scipy.stats.kstest(x, "name", args=(...))`` call form
    altogether. The p-value is **anticonservative** because the parameters were
    estimated from the same sample, and it is additionally distorted when the
    durations are tied (they are, on a sampled record). Both caveats are
    carried in the ``notes`` field of the :class:`DistributionFit` that quotes
    it.

    Returns
    -------
    (statistic, p_value)
    """
    arr = _as_durations(durations)
    u = np.asarray(dist.cdf(arr, *params), dtype=np.float64)
    u = np.clip(u, 0.0, 1.0)
    result = stats.kstest(u, "uniform")
    return float(result.statistic), float(result.pvalue)


def kaplan_meier_survival(
    complete_s: ArrayLike, censored_s: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Product-limit survival estimate with right censoring.

    Kaplan and Meier (1958). ``S(t) = prod_{t_i <= t} (1 - d_i / n_i)`` over the
    distinct event times, with ``d_i`` the number of completed fades at ``t_i``
    and ``n_i`` the number still at risk just before ``t_i``.

    Parameters
    ----------
    complete_s:
        Observed complete durations, seconds (events).
    censored_s:
        Right-censored lower bounds, seconds.

    Returns
    -------
    (time_s, survival)
        ``time_s`` ascending distinct event times; ``survival`` the estimate
        just after each. With no censoring this reduces exactly to
        ``1 - empirical CDF`` evaluated at the event times.
    """
    t = _as_durations(complete_s, name="complete_s")
    c = (
        np.empty(0, dtype=np.float64)
        if censored_s is None
        else np.asarray(censored_s, dtype=np.float64).ravel()
    )
    times = np.unique(t)
    n_total = t.size + c.size
    surv = np.ones(times.size, dtype=np.float64)
    running = 1.0
    for idx, tt in enumerate(times):
        at_risk = int(np.count_nonzero(t >= tt) + np.count_nonzero(c >= tt))
        deaths = int(np.count_nonzero(t == tt))
        if at_risk > 0:
            running *= 1.0 - deaths / at_risk
        surv[idx] = running
    del n_total
    return times, surv


@dataclass(frozen=True)
class FitComparison:
    """Every candidate law fitted to one set of fade durations.

    Attributes
    ----------
    fits:
        In the order they were fitted: the memoryless laws first, because the
        memoryless hypothesis is the one on trial.
    n_complete, n_censored:
        Sample sizes behind every number.
    fs_hz:
        Sample rate, needed to interpret the geometric fit.
    exponential_rejected:
        ``True`` when the valid memoryless test (the geometric chi-square)
        rejects at ``alpha``.
    alpha:
        Significance level used for ``exponential_rejected``.
    """

    fits: tuple[DistributionFit, ...]
    n_complete: int
    n_censored: int
    fs_hz: float
    alpha: float = 0.01
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def best_by_aic(self) -> DistributionFit:
        return min(self.fits, key=lambda f: f.aic)

    @property
    def geometric(self) -> DistributionFit:
        for f in self.fits:
            if f.name == "geometric (samples)":
                return f
        raise KeyError("no geometric fit present")

    @property
    def exponential_rejected(self) -> bool:
        g = self.geometric
        if not g.valid or not math.isfinite(g.p_value):
            return False
        return g.p_value < self.alpha

    def report(self) -> str:
        """Table plus verdict."""
        header = (
            f"{'distribution':<22s} {'AIC':>14s} {'test':<14s} "
            f"{'statistic':>12s} {'dof':>5s} {'p':>12s} {'valid':>8s}"
        )
        lines = [
            f"Fade-duration model comparison (n_complete={self.n_complete}, "
            f"n_censored={self.n_censored}, fs={self.fs_hz!r} Hz)",
            header,
            "-" * len(header),
        ]
        lines += [f.line() for f in self.fits]
        lines.append("")
        g = self.geometric
        verdict = (
            "REJECTED" if self.exponential_rejected else "not rejected"
        )
        lines.append(
            f"Memoryless (exponential/geometric) hypothesis at alpha={self.alpha}: {verdict}"
        )
        lines.append(
            f"  decided on the geometric chi-square: statistic={g.statistic!r}, "
            f"dof={g.dof}, p={g.p_value!r}"
        )
        lines.append(f"  best fit by AIC: {self.best_by_aic.name}")
        for f in self.fits:
            if f.notes:
                lines.append(f"  note [{f.name}]: {f.notes}")
        for key, value in self.extra.items():
            lines.append(f"  {key}: {value!r}")
        return "\n".join(lines)


def compare_fade_duration_models(
    complete_s: ArrayLike,
    *,
    fs_hz: float,
    censored_s: ArrayLike | None = None,
    alpha: float = 0.01,
    min_expected: float = 5.0,
) -> FitComparison:
    """Fit and test every candidate fade-duration law.

    Candidates, in order:

    ``geometric (samples)``
        Shifted geometric on sample counts. The memoryless hypothesis on the
        scale the data actually live on. Tested by Pearson chi-square, which
        is valid for discrete data. **This is the test the verdict uses.**
    ``exponential (censored MLE)``
        Continuous memoryless law, scale from
        :func:`exponential_mle_with_censoring` so the censored fades count.
        Tested by KS for comparability with the literature, with the
        discreteness and estimated-parameter caveats attached.
    ``lognormal``, ``weibull``, ``gamma``
        Two-parameter alternatives, fitted by
        :meth:`scipy.stats.rv_continuous.fit` with the location pinned at
        zero, on the complete durations only (SciPy's ``fit`` has no censoring
        support). Tested by KS with the same caveats.

    Parameters
    ----------
    complete_s:
        Complete fade durations in seconds.
    fs_hz:
        Sample rate in Hz, used to recover sample counts for the geometric.
    censored_s:
        Right-censored lower bounds in seconds.
    alpha:
        Significance level for the published verdict.
    min_expected:
        Minimum expected count per chi-square bin.

    Returns
    -------
    FitComparison
    """
    t = _as_durations(complete_s, name="complete_s")
    c = (
        np.empty(0, dtype=np.float64)
        if censored_s is None
        else np.asarray(censored_s, dtype=np.float64).ravel()
    )
    fs = float(fs_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs_hz must be positive and finite, got {fs_hz!r}")

    fits: list[DistributionFit] = []

    lengths = np.asarray(np.rint(t * fs), dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError(
            "at least one duration rounds to fewer than 1 sample; the geometric fit "
            "needs sample counts >= 1. This happens with "
            "duration_convention='interval_count'; refit on sample_count durations."
        )
    p_geom, ll_geom = fit_geometric_samples(lengths)
    stat, dof, pv, n_bins, note = geometric_chi_square(
        lengths, p_geom, min_expected=min_expected
    )
    fits.append(
        DistributionFit(
            name="geometric (samples)",
            params=(p_geom,),
            n_parameters=1,
            log_likelihood=ll_geom,
            aic=2.0 * 1 - 2.0 * ll_geom,
            statistic=stat,
            p_value=pv,
            test="chi2",
            dof=dof,
            valid=dof >= 1,
            notes=note
            or (
                f"{n_bins} bins after tail merging at min_expected={min_expected}; "
                "mean length "
                f"{1.0 / p_geom:.6g} samples"
            ),
        )
    )

    theta, ll_exp = exponential_mle_with_censoring(t, c)
    ks_stat, ks_p = ks_fitted(t, stats.expon, (0.0, theta))
    fits.append(
        DistributionFit(
            name="exponential (cens MLE)",
            params=(theta,),
            n_parameters=1,
            log_likelihood=ll_exp,
            aic=2.0 * 1 - 2.0 * ll_exp,
            statistic=ks_stat,
            p_value=ks_p,
            test="KS",
            dof=None,
            valid=True,
            notes=(
                "KS p-value is anticonservative (parameter estimated from the same "
                "sample) and distorted by ties (durations are multiples of 1/fs). "
                "Use the geometric chi-square for the memoryless verdict. "
                "AIC includes the censored contributions, so it is not comparable "
                "with the two-parameter rows when n_censored > 0."
            ),
        )
    )

    for label, dist in (
        ("lognormal", stats.lognorm),
        ("weibull", stats.weibull_min),
        ("gamma", stats.gamma),
    ):
        try:
            params = dist.fit(t, floc=0.0)
        except Exception as exc:  # pragma: no cover - SciPy fit failure path
            fits.append(
                DistributionFit(
                    name=label,
                    params=(),
                    n_parameters=2,
                    log_likelihood=float("nan"),
                    aic=float("inf"),
                    statistic=float("nan"),
                    p_value=float("nan"),
                    test="KS",
                    valid=False,
                    notes=f"scipy fit failed: {exc}",
                )
            )
            continue
        ll = float(np.sum(dist.logpdf(t, *params)))
        ks_stat, ks_p = ks_fitted(t, dist, params)
        fits.append(
            DistributionFit(
                name=label,
                params=tuple(float(v) for v in params),
                n_parameters=2,
                log_likelihood=ll,
                aic=2.0 * 2 - 2.0 * ll,
                statistic=ks_stat,
                p_value=ks_p,
                test="KS",
                dof=None,
                valid=True,
                notes=(
                    "loc pinned at 0; fitted on complete durations only (SciPy fit "
                    "has no censoring support); KS p-value anticonservative and "
                    "tie-distorted"
                ),
            )
        )

    coef_var = float(t.std(ddof=1) / t.mean()) if t.size > 1 and t.mean() > 0 else float("nan")
    return FitComparison(
        fits=tuple(fits),
        n_complete=int(t.size),
        n_censored=int(c.size),
        fs_hz=fs,
        alpha=float(alpha),
        extra={
            "coefficient_of_variation": coef_var,
            "coefficient_of_variation_if_exponential": 1.0,
            "mean_duration_s": float(t.mean()),
            "median_duration_s": float(np.median(t)),
        },
    )
