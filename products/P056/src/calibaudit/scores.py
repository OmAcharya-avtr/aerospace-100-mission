"""Proper scoring rules for binary probabilistic forecasts.

All scores are negatively oriented (lower is better) unless the name says
``skill``. Forecasts are probabilities of the positive outcome, dimensionless
and in [0, 1]; outcomes are binary indicators in {0, 1}, dimensionless.

References
----------
Brier, G. W. (1950). "Verification of forecasts expressed in terms of
probability." *Monthly Weather Review* 78(1), 1-3.
doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

Good, I. J. (1952). "Rational decision." *Journal of the Royal Statistical
Society Series B* 14(1), 107-114. (the logarithmic score)

Gneiting, T. and Raftery, A. E. (2007). "Strictly proper scoring rules,
prediction, and estimation." *Journal of the American Statistical Association*
102(477), 359-378. doi:10.1198/016214506000001437

Murphy, A. H. (1973). "A new vector partition of the probability score."
*Journal of Applied Meteorology* 12(4), 595-600.
doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "LOG_SCORE_CLIP",
    "base_rate_forecast",
    "brier_score",
    "brier_skill_score",
    "check_forecasts",
    "constant_forecast",
    "log_score",
    "log_skill_score",
    "skill_score",
]

#: Default clip applied to forecasts before the logarithmic score, so that a
#: single confident mistake gives a finite (large) score instead of ``inf``.
#: The value is the one ``sklearn.metrics.log_loss`` uses as its historical
#: ``eps``; it bounds a single term at ``-log(1e-15) = 34.54`` nats.
LOG_SCORE_CLIP = 1e-15


def check_forecasts(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return ``(forecasts, outcomes)`` as 1-D float arrays.

    Parameters
    ----------
    forecasts
        Probabilities of the positive outcome, shape (n,), dimensionless,
        each in [0, 1].
    outcomes
        Binary outcomes, shape (n,), dimensionless, each exactly 0 or 1.
    min_samples
        Reject inputs shorter than this many samples.

    Raises
    ------
    ValueError
        On shape mismatch, wrong dimensionality, non-finite values, forecasts
        outside [0, 1], outcomes that are not 0 or 1, or fewer than
        ``min_samples`` samples.
    """
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"forecasts must be 1-D, got shape {f.shape}")
    if o.ndim != 1:
        raise ValueError(f"outcomes must be 1-D, got shape {o.shape}")
    if f.shape != o.shape:
        raise ValueError(
            f"forecasts and outcomes must have the same length, got {f.size} and {o.size}"
        )
    if f.size < min_samples:
        raise ValueError(f"need at least {min_samples} samples, got {f.size}")
    if not np.all(np.isfinite(f)):
        raise ValueError("forecasts contain non-finite values (nan or inf)")
    if not np.all(np.isfinite(o)):
        raise ValueError("outcomes contain non-finite values (nan or inf)")
    if np.any(f < 0.0) or np.any(f > 1.0):
        lo, hi = float(f.min()), float(f.max())
        raise ValueError(f"forecasts must lie in [0, 1], got range [{lo!r}, {hi!r}]")
    bad = ~np.isin(o, (0.0, 1.0))
    if np.any(bad):
        raise ValueError(
            "outcomes must be binary 0 or 1; "
            f"{int(bad.sum())} of {o.size} values are neither (first offender {o[bad][0]!r})"
        )
    return f, o


def brier_score(forecasts: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error of a probability forecast (Brier 1950), dimensionless.

    ``BS = mean((f_i - o_i)**2)``, in [0, 1]; 0 is a perfect deterministic
    forecast and 0.25 is the score of a constant 0.5 forecast.
    """
    f, o = check_forecasts(forecasts, outcomes)
    return float(np.mean((f - o) ** 2))


def log_score(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    clip: float = LOG_SCORE_CLIP,
) -> float:
    """Mean negative log-likelihood in nats (Good 1952), dimensionless.

    ``LS = -mean(o_i log f_i + (1 - o_i) log(1 - f_i))`` with ``f`` clipped to
    ``[clip, 1 - clip]``. Unbounded above; a single confident mistake dominates.

    Raises
    ------
    ValueError
        If ``clip`` is not in (0, 0.5).
    """
    f, o = check_forecasts(forecasts, outcomes)
    if not 0.0 < clip < 0.5:
        raise ValueError(f"clip must lie in (0, 0.5), got {clip!r}")
    fc = np.clip(f, clip, 1.0 - clip)
    return float(-np.mean(o * np.log(fc) + (1.0 - o) * np.log1p(-fc)))


def skill_score(score: float, reference_score: float) -> float:
    """``1 - score / reference_score`` for a negatively oriented score.

    Positive means better than the reference, 0 means equal, negative worse.

    Raises
    ------
    ValueError
        If ``reference_score`` is 0, where the ratio is undefined (a reference
        forecast that is already perfect admits no skill score).
    """
    if reference_score == 0.0:
        raise ValueError(
            "reference_score is exactly 0, so the skill score is undefined; "
            "the reference forecast is already perfect on this sample"
        )
    return float(1.0 - score / reference_score)


def brier_skill_score(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    reference: ArrayLike,
) -> float:
    """Brier skill score of ``forecasts`` against an explicit ``reference``.

    The reference is a forecast array of the same length, not a scalar: a
    skill score without a stated reference is not interpretable.
    """
    f, o = check_forecasts(forecasts, outcomes)
    r, _ = check_forecasts(reference, outcomes)
    return skill_score(brier_score(f, o), brier_score(r, o))


def log_skill_score(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    reference: ArrayLike,
    *,
    clip: float = LOG_SCORE_CLIP,
) -> float:
    """Logarithmic skill score of ``forecasts`` against an explicit ``reference``."""
    f, o = check_forecasts(forecasts, outcomes)
    r, _ = check_forecasts(reference, outcomes)
    return skill_score(log_score(f, o, clip=clip), log_score(r, o, clip=clip))


def constant_forecast(value: float, n_samples: int) -> np.ndarray:
    """A constant reference forecast of ``value`` repeated ``n_samples`` times."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must lie in [0, 1], got {value!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples!r}")
    return np.full(int(n_samples), float(value))


def base_rate_forecast(outcomes: ArrayLike) -> np.ndarray:
    """The climatological reference: the sample base rate, repeated.

    This is the standard reference in forecast verification. Computing it from
    the same outcomes it is scored against gives it an in-sample advantage of
    order 1/n; state which sample it came from whenever you quote a skill score
    against it.
    """
    o = np.asarray(outcomes, dtype=float)
    _, o = check_forecasts(np.zeros_like(o), o)
    return np.full(o.size, float(o.mean()))
