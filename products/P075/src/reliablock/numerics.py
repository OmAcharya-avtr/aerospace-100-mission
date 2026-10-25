"""Reliability arithmetic that does not lose the answer to cancellation.

Why this module exists
----------------------

A reliability figure and an unreliability figure carry the same information in
exact arithmetic and completely different information in float64. For a
redundant group of two blocks each with unreliability ``1e-9`` the system
unreliability is ``1e-18``; the system reliability is ``1 - 1e-18``, which is
the float64 number ``1.0`` and nothing else. Recovering ``1e-18`` from it is
not possible, because it was never there. Every number this package reports
about a redundant structure is therefore computed and carried in
**unreliability space**, and the reliability is the derived quantity, not the
other way round.

Three representations are used, and the choice is per structure:

``H``, cumulative hazard
    ``H = -log R``. Series composition is the exact sum ``H = sum H_i``, so a
    series chain of a thousand blocks costs one rounding per block and no
    cancellation at all. Mission phases accumulate hazard, so a mission profile
    is a sum in this space.

``Q``, unreliability
    ``Q = 1 - R``. Parallel composition is the exact product ``Q = prod Q_i``,
    and a k-of-n group is a sum of products of ``Q`` and ``R``. This is the
    space in which a small failure probability survives.

``R``, reliability
    Reported, and used where the structure genuinely needs it, but never used
    as the carrier from which a small ``Q`` is later recovered.

The measured cost of getting this wrong is in
``validation/validate_precision_output.txt``. It is not a rounding-level
effect: the naive route returns exactly zero for cases where the correct answer
is ``1e-18``, and loses about five significant figures on a ten-block series
chain of blocks with ``Q_i = 1e-12``.

Functions here take and return plain floats or numpy arrays and hold no state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "complement",
    "hazard_from_reliability",
    "hazard_from_unreliability",
    "log1p_safe",
    "parallel_unreliability",
    "reliability_from_hazard",
    "series_hazard",
    "series_unreliability",
    "unreliability_from_hazard",
    "weighted_binomial_survival",
]

#: Largest cumulative hazard carried as a number rather than treated as certain
#: failure. ``exp(-745.2)`` is the smallest positive float64 normal-ish value;
#: beyond ``H = 745`` the reliability underflows to zero and the unreliability
#: is 1.0 to within float64, so there is nothing left to represent.
MAX_HAZARD = 745.0


def _as_float(value: float, name: str) -> float:
    """Return ``value`` as a finite float, or raise ``ValueError``/``TypeError``."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _check_probability(value: float, name: str) -> float:
    """Return ``value`` as a float in ``[0, 1]``, or raise ``ValueError``."""
    out = _as_float(value, name)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {out!r}")
    return out


def _check_hazard(value: float, name: str) -> float:
    """Return ``value`` as a non-negative float, ``inf`` allowed.

    ``inf`` is a legitimate cumulative hazard: it is what a certainly-failed
    item has, and it is what :func:`hazard_from_unreliability` returns for
    ``Q = 1``. It is accepted here and mapped to ``R = 0``, ``Q = 1`` by the
    conversions. ``nan`` is not a hazard and is rejected.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if math.isnan(out):
        raise ValueError(f"{name} is a cumulative hazard and must not be nan")
    if out < 0.0:
        raise ValueError(f"{name} is a cumulative hazard and must be >= 0, got {out!r}")
    return out


def log1p_safe(x: float) -> float:
    """``log(1 + x)`` for ``x`` in ``[-1, inf)``, returning ``-inf`` at ``x = -1``.

    ``math.log1p(-1.0)`` raises; the mathematically correct value is ``-inf``
    and that is what a cumulative hazard of a certainly-failed block is.
    """
    value = _as_float(x, "x")
    if value < -1.0:
        raise ValueError(f"log1p is undefined for x < -1, got {value!r}")
    if value == -1.0:
        return -math.inf
    return math.log1p(value)


def complement(p: float) -> float:
    """``1 - p`` for a probability ``p``, exact in float64 for ``p`` in ``[0, 1]``.

    Sterbenz's lemma gives ``1 - p`` exactly whenever ``0.5 <= p <= 1``, and for
    ``p < 0.5`` the subtraction is exact because ``1`` and ``p`` share an
    exponent range that admits the difference. The function exists so that the
    places where a complement is deliberately taken are greppable: the defect
    this module is written against is taking a complement *and then relying on
    the small side of it*.

    Units: dimensionless probability. Range: ``[0, 1]``.
    """
    return 1.0 - _check_probability(p, "p")


def reliability_from_hazard(h: float) -> float:
    """Reliability ``R = exp(-H)`` from cumulative hazard ``H`` (dimensionless).

    ``H >= MAX_HAZARD`` returns ``0.0``: the reliability has underflowed and
    there is no float64 number between it and zero.
    """
    hazard = _check_hazard(h, "h")
    if hazard >= MAX_HAZARD:
        return 0.0
    return math.exp(-hazard)


def unreliability_from_hazard(h: float) -> float:
    """Unreliability ``Q = 1 - exp(-H) = -expm1(-H)`` from cumulative hazard ``H``.

    ``-expm1(-H)`` is accurate to full relative precision as ``H -> 0``, where
    ``1 - exp(-H)`` loses every digit below ``eps``. For ``H = 1e-18`` this
    returns ``1e-18``; the naive form returns ``0.0``.
    """
    hazard = _check_hazard(h, "h")
    if hazard >= MAX_HAZARD:
        return 1.0
    return -math.expm1(-hazard)


def hazard_from_reliability(r: float) -> float:
    """Cumulative hazard ``H = -log R`` from reliability ``R`` in ``[0, 1]``.

    ``R = 0`` returns ``inf``. This is the lossy direction for a highly
    reliable block and is provided for completeness: prefer
    :func:`hazard_from_unreliability` when the small quantity is ``Q``.
    """
    reliability = _check_probability(r, "r")
    if reliability == 0.0:
        return math.inf
    return -math.log(reliability)


def hazard_from_unreliability(q: float) -> float:
    """Cumulative hazard ``H = -log1p(-Q)`` from unreliability ``Q`` in ``[0, 1]``.

    This is the accurate direction for a highly reliable block: ``Q = 1e-18``
    gives ``H = 1e-18`` to full relative precision, where routing through
    ``R = 1 - Q`` gives ``H = 0``.
    """
    unreliability = _check_probability(q, "q")
    return -log1p_safe(-unreliability)


def series_hazard(hazards: Iterable[float]) -> float:
    """Cumulative hazard of a series chain: the exact sum of the block hazards.

    A series structure survives only if every block survives, so
    ``R = prod R_i`` and ``H = sum H_i``. The sum carries one rounding per term
    and no cancellation; the product form ``prod R_i`` followed by
    ``Q = 1 - prod R_i`` loses ``log10(eps / Q)`` significant figures, which is
    about five for a chain of blocks with ``Q_i = 1e-12``.

    Returns ``inf`` if any block hazard is infinite. Empty input returns
    ``0.0``: a series structure with no blocks cannot fail.
    """
    total = 0.0
    for index, value in enumerate(hazards):
        hazard = _check_hazard(value, f"hazards[{index}]")
        total += hazard
        if math.isinf(total):
            return math.inf
    return total


def series_unreliability(unreliabilities: Sequence[float]) -> float:
    """Unreliability of a series chain, computed through hazard space.

    ``Q = 1 - prod (1 - Q_i)``, evaluated as ``-expm1(sum log1p(-Q_i))``. For
    small ``Q_i`` this is accurate to full relative precision in ``Q``, where
    the direct expression is accurate only to ``eps`` *absolute*.

    Empty input returns ``0.0``.
    """
    hazards = (hazard_from_unreliability(q) for q in unreliabilities)
    return unreliability_from_hazard(series_hazard(hazards))


def parallel_unreliability(unreliabilities: Iterable[float]) -> float:
    """Unreliability of a parallel group: the exact product of block unreliabilities.

    A parallel structure fails only if every block fails, so ``Q = prod Q_i``.
    One rounding per factor, no cancellation, and the result stays
    representable down to ``5e-324``. The complementary route
    ``Q = 1 - prod(1 - Q_i)`` returns exactly ``0.0`` for any group whose true
    unreliability is below ``eps``, which for two blocks means any block
    unreliability below about ``1.5e-8``.

    Empty input returns ``0.0``: a parallel structure with no blocks has no way
    to fail, which is a declaration error and is rejected upstream in
    :mod:`reliablock.diagram` rather than here.
    """
    product = 1.0
    seen = False
    for index, value in enumerate(unreliabilities):
        seen = True
        product *= _check_probability(value, f"unreliabilities[{index}]")
        if product == 0.0:
            return 0.0
    return product if seen else 0.0


def weighted_binomial_survival(reliabilities: Sequence[float], k: int) -> float:
    """Probability that at least ``k`` of the given independent blocks survive.

    The Poisson-binomial survival function, evaluated by the exact
    convolution recurrence over the number of survivors rather than by
    enumerating ``2**n`` states. ``n`` blocks cost ``O(n^2)`` multiplications
    and the result is a sum of non-negative terms, so there is no cancellation.

    For identical blocks this is the binomial survival
    ``sum_{i=k}^{n} C(n, i) p^i (1 - p)^(n - i)``, which is the closed form the
    k-of-n structure is validated against in
    ``validation/validate_structures.py``.

    Parameters
    ----------
    reliabilities
        Per-block reliabilities in ``[0, 1]``, dimensionless.
    k
        Minimum number of surviving blocks for system success. ``k = 0`` is
        certain success; ``k > n`` is certain failure.

    Returns
    -------
    float
        Probability in ``[0, 1]``.
    """
    probs = [_check_probability(r, f"reliabilities[{i}]") for i, r in enumerate(reliabilities)]
    n = len(probs)
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError(f"k must be an integer, got {k!r}")
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return 1.0
    if k > n:
        return 0.0
    # distribution[j] = P(exactly j of the blocks processed so far survive)
    distribution = np.zeros(n + 1, dtype=float)
    distribution[0] = 1.0
    for index, p in enumerate(probs):
        # The previous row must be copied: updating in place would feed a
        # partially updated row back into the recurrence.
        previous = distribution[: index + 1].copy()
        distribution[: index + 2] = 0.0
        distribution[: index + 1] += previous * (1.0 - p)
        distribution[1 : index + 2] += previous * p
    # Clamped into [0, 1]. The tail sum is a sum of non-negative float64 terms
    # that add to 1 in exact arithmetic, so it can land a few ulps ABOVE 1 --
    # measured at 1.0000000000000002 for the 5-block k = n case found by
    # Hypothesis at rates [0.0078125, 5e-05, 5e-05, 1.0000000000000003e-09,
    # 1e-09] over 229415.5 hours. Returning that unclamped made
    # hazard_from_unreliability raise on a perfectly valid diagram, which is a
    # crash and not a rounding question. Pinned as a regression test.
    return min(max(float(distribution[k:].sum()), 0.0), 1.0)
