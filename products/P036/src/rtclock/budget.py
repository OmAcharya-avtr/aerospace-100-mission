"""Timing-budget composition across the stages of a control loop.

Two compositions, kept separate because they answer different questions and
have different assumptions.

**Worst-case composition** (deterministic). The worst-case latency of a chain
of stages executed in sequence is the sum of their worst cases::

    WCET_total = sum_i WCET_i

No distributional assumption; exact for sequential execution with no
interference between stages. This is the number a deadline must be compared
against. It is pessimistic by construction, because the worst cases of
different stages generally do not coincide.

**Statistical composition** (distributional). If stage latencies are
*independent* random variables with means ``mu_i`` and standard deviations
``sigma_i``, then for the sum::

    mu_total    = sum_i mu_i                      (exact, independence not needed)
    sigma_total = sqrt(sum_i sigma_i^2)           (requires independence)

The mean is additive for any dependence structure (linearity of expectation).
The quadrature rule for the standard deviation requires zero covariance
between stages, which is the assumption most often violated in practice: two
stages sharing a cache, a bus or a lock are positively correlated, and the
quadrature result then *underestimates* the spread. This module therefore
reports ``sigma_total`` with the independence assumption named in the result,
and also reports the fully-correlated upper bound ``sum_i sigma_i``, which
holds for any dependence (it is the Cauchy-Schwarz / triangle-inequality
bound on the standard deviation of a sum).

Reference for the quadrature rule
    JCGM 100:2008, *Evaluation of measurement data -- Guide to the expression
    of uncertainty in measurement* (GUM), Sec. 5.1.2 (combined standard
    uncertainty for uncorrelated inputs) and Sec. 5.2.2 (the correlated case).

**Clock-resolution error term.** Each stage boundary is a clock reading, so a
stage latency measured on a clock of step ``q`` carries a worst-case error of
``q`` and a standard uncertainty of ``q / sqrt(6)``
(:func:`rtclock.timebase.duration_uncertainty_s`). Over ``n`` stages measured
with ``n + 1`` independent readings, the standard uncertainty of the total
from quantization alone is ``q * sqrt(n + 1) / sqrt(12)``. That term is
computed and reported separately from the stage spread, so it is never
confused with real jitter.

Units: seconds throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BudgetResult", "Stage", "compose_budget"]


@dataclass(frozen=True)
class Stage:
    """One stage of a loop, with a worst case and optionally a distribution.

    Attributes:
        name: identifier, non-empty.
        wcet_s: worst-case execution time, units s, must be > 0.
        mean_s: mean execution time, units s. Defaults to ``wcet_s``. Must
            satisfy ``0 < mean_s <= wcet_s``.
        stdev_s: standard deviation of execution time, units s, >= 0.
            Defaults to 0.0.
    """

    name: str
    wcet_s: float
    mean_s: float | None = None
    stdev_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        w = float(self.wcet_s)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"wcet_s must be a finite value > 0 s, got {self.wcet_s}")
        object.__setattr__(self, "wcet_s", w)
        if self.mean_s is None:
            object.__setattr__(self, "mean_s", w)
        else:
            m = float(self.mean_s)
            if not math.isfinite(m) or m <= 0.0:
                raise ValueError(f"mean_s must be a finite value > 0 s, got {self.mean_s}")
            if m > w:
                raise ValueError(f"mean_s ({m} s) must not exceed wcet_s ({w} s)")
            object.__setattr__(self, "mean_s", m)
        s = float(self.stdev_s)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError(f"stdev_s must be a finite value >= 0 s, got {self.stdev_s}")
        object.__setattr__(self, "stdev_s", s)


@dataclass(frozen=True)
class BudgetResult:
    """Composed timing budget for a chain of stages.

    Attributes:
        budget_s: the budget the chain was compared against, units s.
        wcet_total_s: ``sum(WCET_i)``, units s. Deterministic, exact for
            sequential execution.
        mean_total_s: ``sum(mu_i)``, units s. Exact for any dependence.
        stdev_independent_s: ``sqrt(sum(sigma_i^2))``, units s. **Assumes
            independent stages.**
        stdev_fully_correlated_s: ``sum(sigma_i)``, units s. Upper bound on
            the standard deviation of the sum for any dependence structure.
        clock_quantum_s: clock step used for the quantization term, units s.
        quantization_uncertainty_s: ``q * sqrt(n+1) / sqrt(12)``, units s --
            the standard uncertainty of the measured total from clock
            quantization at ``n+1`` stage boundaries.
        margin_s: ``budget_s - wcet_total_s``, units s. Negative means the
            worst case does not fit.
        stage_fractions: ``{name: WCET_i / wcet_total_s}``, dimensionless,
            so the dominant stage is visible without arithmetic.
        n_stages: number of stages.
    """

    budget_s: float
    wcet_total_s: float
    mean_total_s: float
    stdev_independent_s: float
    stdev_fully_correlated_s: float
    clock_quantum_s: float
    quantization_uncertainty_s: float
    margin_s: float
    stage_fractions: dict[str, float]
    n_stages: int

    @property
    def fits_worst_case(self) -> bool:
        """True when ``wcet_total_s <= budget_s``."""
        return self.wcet_total_s <= self.budget_s

    @property
    def utilization(self) -> float:
        """``wcet_total_s / budget_s``, dimensionless."""
        return self.wcet_total_s / self.budget_s

    def sigma_headroom(self, which: str = "independent") -> float:
        """Headroom from the mean to the budget, in units of the composed sigma.

        ``(budget - mean_total) / sigma_total``, dimensionless. This is a
        *distance*, not a probability: converting it to an exceedance
        probability requires a distributional assumption this package does
        not make. Execution-time distributions are typically right-skewed and
        bounded below, so a Gaussian tail estimate from this number would be
        optimistic.

        Args:
            which: ``"independent"`` or ``"correlated"``, selecting which
                composed sigma to divide by.

        Returns:
            Dimensionless headroom, or ``inf`` when the composed sigma is 0.

        Raises:
            ValueError: on an unknown ``which``.
        """
        if which == "independent":
            sigma = self.stdev_independent_s
        elif which == "correlated":
            sigma = self.stdev_fully_correlated_s
        else:
            raise ValueError(f"which must be 'independent' or 'correlated', got {which!r}")
        if sigma == 0.0:
            return math.inf
        return (self.budget_s - self.mean_total_s) / sigma


def compose_budget(
    stages: list[Stage], budget_s: float, clock_quantum_s: float = 0.0
) -> BudgetResult:
    """Compose a stage chain into a worst-case and a statistical total.

    Args:
        stages: at least one :class:`Stage`, names unique.
        budget_s: the period or deadline the chain must fit in, units s, > 0.
        clock_quantum_s: measured clock step for the quantization term,
            units s, >= 0. Pass 0.0 to omit the term; pass the measured tick
            from :func:`rtclock.timebase.measure_clock_resolution` to include
            it.

    Returns:
        A :class:`BudgetResult`.

    Raises:
        ValueError: on an empty list, duplicate names, a non-positive budget
            or a negative quantum.
    """
    if not stages:
        raise ValueError("stages must be non-empty")
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"stage names must be unique; duplicated: {dupes}")
    b = float(budget_s)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(f"budget_s must be a finite value > 0 s, got {budget_s}")
    q = float(clock_quantum_s)
    if not math.isfinite(q) or q < 0.0:
        raise ValueError(f"clock_quantum_s must be a finite value >= 0 s, got {clock_quantum_s}")

    wcet_total = math.fsum(s.wcet_s for s in stages)
    mean_total = math.fsum(float(s.mean_s or s.wcet_s) for s in stages)
    var_total = math.fsum(s.stdev_s**2 for s in stages)
    n = len(stages)
    return BudgetResult(
        budget_s=b,
        wcet_total_s=wcet_total,
        mean_total_s=mean_total,
        stdev_independent_s=math.sqrt(var_total),
        stdev_fully_correlated_s=math.fsum(s.stdev_s for s in stages),
        clock_quantum_s=q,
        quantization_uncertainty_s=q * math.sqrt(n + 1) / math.sqrt(12.0),
        margin_s=b - wcet_total,
        stage_fractions={s.name: s.wcet_s / wcet_total for s in stages},
        n_stages=n,
    )
