"""Finite-sample coverage bounds and the diagnostics that go with them.

The split-conformal guarantee
-----------------------------
Let ``V_1, ..., V_n`` be calibration conformity scores and ``V_{n+1}`` the test
score, and assume the ``n + 1`` scores are **exchangeable**. Take
``k = ceil((n + 1)(1 - alpha))`` and let ``q`` be the ``k``-th smallest
calibration score. Then

    P(V_{n+1} <= q) >= 1 - alpha,

and if the scores are almost surely distinct (which a continuous score
distribution guarantees),

    P(V_{n+1} <= q) = k / (n + 1)  in  [1 - alpha, 1 - alpha + 1/(n + 1)].

Reference: Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. and Wasserman, L.,
"Distribution-Free Predictive Inference for Regression", *Journal of the
American Statistical Association*, Vol. 113, No. 523, 2018, Theorem 2.1 and
the surrounding discussion; the rank argument itself is older and is the
inductive confidence machine of Papadopoulos, H., Proedrou, K., Vovk, V. and
Gammerman, A., "Inductive Confidence Machines for Regression", *ECML* 2002.

The equality is the known-answer test of this package: it is an exact
statement about a rank, it does not depend on the score distribution, and it
is reproduced to Monte-Carlo accuracy in ``tests/test_known_answers.py`` and
``validation/validate_known_answers.py``.

**Exchangeability is an assumption, not a property of your data.** Under the
covariate shift this package declares it is false, which is the entire point
of the audit: the bound below still describes what split conformal would
deliver on exchangeable data, and the measured coverage shows what it actually
delivers when that assumption is withdrawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CoverageBound:
    """The finite-sample split-conformal coverage bound for one calibration size.

    Attributes
    ----------
    n_calibration:
        Number of calibration scores ``n``.
    alpha:
        Miscoverage level in (0, 1).
    rank:
        ``k = ceil((n + 1)(1 - alpha))``, the order statistic used.
    exact:
        ``k / (n + 1)``, the exact coverage for almost surely distinct scores.
    lower:
        ``1 - alpha``, the guaranteed lower bound.
    upper:
        ``1 - alpha + 1 / (n + 1)``, the matching upper bound.

    Note on floating point: ``exact`` can sit one unit in the last place
    **below** ``lower`` when ``(n + 1)(1 - alpha)`` is an exact integer,
    because ``1 - alpha`` is generally not representable in binary. Measured
    case: ``n = 1249, alpha = 0.18`` gives ``rank = 1025``,
    ``exact = 1025/1250 = 0.82`` and ``lower = 1 - 0.18 = 0.8200000000000001``.
    In exact arithmetic the two are equal and the bound is attained. Compare
    them with a tolerance of about 1e-12, never with ``<=``.
    """

    n_calibration: int
    alpha: float
    rank: int
    exact: float
    lower: float
    upper: float

    @property
    def conservatism(self) -> float:
        """``exact - (1 - alpha)``, the over-coverage the discrete rank forces [-]."""
        return self.exact - self.lower


def split_conformal_coverage_bound(n_calibration: int, alpha: float) -> CoverageBound:
    """Return the finite-sample coverage bound for split conformal.

    Parameters
    ----------
    n_calibration:
        Number of calibration points ``n``, positive.
    alpha:
        Miscoverage level in (0, 1).

    Returns
    -------
    CoverageBound

    Raises
    ------
    ValueError
        If ``ceil((n + 1)(1 - alpha)) > n``, in which case no finite
        calibration order statistic attains the level and the conformal
        interval is the whole real line. This happens exactly when
        ``n < ceil(1 / alpha) - 1``.
    """
    if n_calibration <= 0:
        raise ValueError(f"n_calibration must be > 0, got {n_calibration}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = int(n_calibration)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        needed = math.ceil(1.0 / alpha) - 1
        raise ValueError(
            f"n_calibration={n} is too small for alpha={alpha}: the required order "
            f"statistic is {rank} > {n}, so the interval is unbounded. "
            f"Use n_calibration >= {needed}."
        )
    return CoverageBound(
        n_calibration=n,
        alpha=float(alpha),
        rank=int(rank),
        exact=rank / (n + 1),
        lower=1.0 - alpha,
        upper=1.0 - alpha + 1.0 / (n + 1),
    )


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a measured coverage [-].

    Clopper, C.J. and Pearson, E.S., "The Use of Confidence or Fiducial Limits
    Illustrated in the Case of the Binomial", *Biometrika*, Vol. 26, No. 4,
    1934, pp. 404-413.

    Parameters
    ----------
    successes:
        Number of covered test points, ``0 <= successes <= trials``.
    trials:
        Number of test points, positive.
    confidence:
        Two-sided confidence level in (0, 1).

    Returns
    -------
    tuple of float
        ``(lower, upper)``, both in [0, 1].
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    return lower, upper


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2`` [-].

    Kish, L., *Survey Sampling*, Wiley, 1965, section 11.7. For ``n`` equal
    weights it returns ``n`` exactly; it falls towards 1 as one weight comes
    to dominate. It is the diagnostic that says when weighted conformal has
    run out of usable calibration points.

    Parameters
    ----------
    weights:
        Non-negative weights, at least one strictly positive.

    Returns
    -------
    float
        Effective sample size in [1, len(weights)].
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0.0:
        raise ValueError("weights must not all be zero")
    return float(total**2 / np.square(w).sum())
