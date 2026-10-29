"""Cross-source disagreement, measured in kilometres rather than in digits.

Two catalogues publish an element set for the same object. Comparing the two
texts digit by digit answers the wrong question: a difference in the last digit
of the mean motion can be worth tens of kilometres a day later, and a large
difference in the mean anomaly can be worth nothing at all because the epochs
differ. The only comparison that means anything is the one made in position.

What this module cannot do, and says so
---------------------------------------
It **cannot say which source is right.** The comparison is symmetric: swapping
the two arguments changes the sign of nothing and the magnitude of nothing.
That is returned as a field of the result, not written in a limitations
section, because an attribution without it is the error this screen exists to
prevent. The only thing that can break the symmetry is an independent
observation, which this package does not ship and does not have.

Equation
--------
(12) Source separation:

        s_k = || r_A(t_k) - r_B(t_k) ||_2,  kilometres

    with both element sets propagated to the same absolute times. This is
    equation (11) with a different interpretation of the two inputs, and it is
    implemented by the same function so the two can never drift apart.

Units
-----
Kilometres and days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .consequence import position_errors_km
from .elements import ElementSet
from .propagate import propagate, time_grid

DEFAULT_DIVERGENCE_THRESHOLD_KM = 1.0
"""Declared policy threshold on equation (12). Not a physical constant."""


@dataclass(frozen=True)
class Divergence:
    """Measured separation between two sources for the same object.

    Attributes
    ----------
    usable:
        ``False`` when fewer than two sources were supplied, or when either
        propagation was loud. ``reason`` says which.
    reason:
        Empty when ``usable``.
    max_separation_km, rms_separation_km, initial_separation_km:
        Equation (12) over the grid.
    threshold_km:
        Declared threshold, carried with the verdict.
    diverged:
        ``max_separation_km > threshold_km``.
    which_source_is_correct:
        Always ``None``. Present as a field, and documented here, because the
        answer is not determinable from two element sets and a propagator, and
        a caller who wants it needs an independent observation this package
        does not have.
    sources:
        The two source labels compared.
    """

    usable: bool
    reason: str
    max_separation_km: float
    rms_separation_km: float
    initial_separation_km: float
    threshold_km: float
    diverged: bool
    which_source_is_correct: None
    sources: tuple[str, str]


def _unusable(reason: str, threshold: float, sources: tuple[str, str]) -> Divergence:
    nan = float("nan")
    return Divergence(
        usable=False,
        reason=reason,
        max_separation_km=nan,
        rms_separation_km=nan,
        initial_separation_km=nan,
        threshold_km=threshold,
        diverged=False,
        which_source_is_correct=None,
        sources=sources,
    )


def compare(
    source_a: ElementSet,
    source_b: ElementSet,
    start_jd: float,
    horizon_days: float = 1.0,
    samples: int = 25,
    threshold_km: float = DEFAULT_DIVERGENCE_THRESHOLD_KM,
) -> Divergence:
    """Measure equation (12) between two element sets for the same object.

    Raises
    ------
    ValueError
        If the two element sets are for different objects, or
        ``threshold_km`` is not positive and finite.
    """
    if not np.isfinite(threshold_km) or threshold_km <= 0.0:
        raise ValueError(f"threshold_km must be positive and finite, got {threshold_km!r}")
    if source_a.satnum != source_b.satnum:
        raise ValueError(
            f"cross-source comparison needs one object; got {source_a.satnum} "
            f"from {source_a.source!r} and {source_b.satnum} from {source_b.source!r}"
        )
    labels = (source_a.source, source_b.source)
    grid = time_grid(start_jd, horizon_days, samples)
    first = propagate(source_a, grid)
    second = propagate(source_b, grid)
    if first.loud or second.loud:
        loud = f"{labels[0]} ({first.loud_reason})" if first.loud else ""
        if second.loud:
            loud = (loud + "; " if loud else "") + f"{labels[1]} ({second.loud_reason})"
        return _unusable(
            "at least one source propagated loudly, so equation (12) has no "
            f"finite value: {loud}",
            threshold_km,
            labels,
        )
    separations = position_errors_km(first, second)
    maximum = float(np.max(separations))
    return Divergence(
        usable=True,
        reason="",
        max_separation_km=maximum,
        rms_separation_km=float(np.sqrt(np.mean(separations * separations))),
        initial_separation_km=float(separations[0]),
        threshold_km=threshold_km,
        diverged=maximum > threshold_km,
        which_source_is_correct=None,
        sources=labels,
    )
