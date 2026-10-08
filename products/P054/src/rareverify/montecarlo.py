"""Plain Monte-Carlo estimation of a failure probability.

The estimator is the sample proportion

    p_hat = (1 / n) * sum_i 1[g(x_i) <= 0],     x_i ~ N(0, I)

which is unbiased with variance ``p (1 - p) / n``.  Its coefficient of
variation is ``sqrt((1 - p) / (n p))``, so reaching a 10 % coefficient of
variation at ``p = 1e-6`` needs about ``1e8`` runs.  That arithmetic is the
reason the rest of this package exists; :func:`required_samples_for_cov`
reports it.

Source: the binomial variance is elementary; the coefficient-of-variation
scaling as stated is for example in Rubinstein, R. Y. and Kroese, D. P. (2016),
"Simulation and the Monte Carlo Method", 3rd edition, Wiley, chapter on rare
events.  Dimensionless throughout.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .estimate import RareEventEstimate
from .limitstates import LimitState

__all__ = ["crude_monte_carlo", "required_samples_for_cov"]


def required_samples_for_cov(p: float, target_cov: float = 0.1) -> int:
    """Runs a crude campaign needs for a given coefficient of variation.

    ``n = (1 - p) / (p * target_cov^2)``, rounded up.

    Parameters
    ----------
    p
        Failure probability, dimensionless, in ``(0, 1)``.
    target_cov
        Target coefficient of variation of the estimator, dimensionless, > 0.

    Returns
    -------
    int
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie strictly in (0, 1), got {p}")
    target_cov = float(target_cov)
    if not target_cov > 0.0:
        raise ValueError(f"target_cov must be positive, got {target_cov}")
    return max(1, math.ceil((1.0 - p) / (p * target_cov * target_cov)))


def crude_monte_carlo(
    limit_state: LimitState,
    n_samples: int,
    rng: np.random.Generator | None = None,
    batch_size: int = 250_000,
) -> RareEventEstimate:
    """Estimate ``P(g(X) <= 0)`` by independent sampling of ``X ~ N(0, I)``.

    Parameters
    ----------
    limit_state
        The limit state to evaluate.
    n_samples
        Number of runs, ``>= 1``.
    rng
        Random generator; a default-seeded one is created if omitted.
    batch_size
        Samples per evaluation batch, to bound peak memory.  The estimate does
        not depend on it.

    Returns
    -------
    RareEventEstimate
        With ``is_binomial = True``, so a Clopper-Pearson or Wilson interval is
        valid for it.  ``standard_error`` is the plug-in
        ``sqrt(p_hat (1 - p_hat) / n)``, which is zero when no failure is
        observed; in that case the zero-failure confidence bound, not the
        standard error, is the quantity to quote.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    generator = np.random.default_rng(0) if rng is None else rng
    d = limit_state.dimension
    start = time.perf_counter()
    failures = 0
    remaining = int(n_samples)
    while remaining > 0:
        take = min(batch_size, remaining)
        x = generator.standard_normal((take, d))
        failures += int(np.count_nonzero(limit_state.g(x) <= 0.0))
        remaining -= take
    wall = time.perf_counter() - start
    p_hat = failures / n_samples
    se = math.sqrt(p_hat * (1.0 - p_hat) / n_samples)
    return RareEventEstimate(
        method="crude",
        estimate=p_hat,
        standard_error=se,
        n_samples=int(n_samples),
        true_evaluations=int(n_samples),
        n_failures=failures,
        is_binomial=True,
        effective_sample_size=float(failures),
        wall_seconds=wall,
        standard_error_kind="binomial-plugin",
        contributions=np.ones(failures, dtype=float),
        diagnostics={"batch_size": float(batch_size)},
    )
