"""Ephemeris continuity: does the new element set agree with the previous one.

This is the strongest of the rule screens and it is the one an operations
engineer actually applies. The question it asks is not whether an element value
is plausible but whether the *ephemeris* the new element set produces agrees
with the ephemeris the object's previous element set predicts for the same
times. The answer is in kilometres, which is the unit the consequence is
measured in, so this screen is the one that closes the gap between an element
deviation and a position error.

It was added to the rule baseline **after** the learned classifier was first
benchmarked, because the first benchmark showed the rules losing by a margin
that was a property of the baseline rather than of the problem: the earlier
screens all test element values against element bands, while the label is a
position error in kilometres, and nothing in the baseline translated between
the two. The mission's build guide requires an under-strength baseline to be
strengthened and the smaller margin published, and the numbers in
``validation/validate_learned_output.txt`` are the ones measured after this
module existed. The earlier figures are in ``validation/VALIDATION.md`` beside
them, because a reader is entitled to see how much the baseline's strength was
worth.

Self-calibration, and why it is not tuning
------------------------------------------
A previous element set is itself older than the candidate, so its propagation
to the same time carries genuine prediction error. How much is a property of
the object: a decaying low-Earth object disagrees with its own predecessor by
kilometres a day later, a geostationary object by metres. Rather than declare
one threshold for both, this screen measures the object's **own** normal
element-to-element continuity from its history and tests the candidate against
that.

Equations
---------
(24) Pairwise continuity offset, for consecutive element sets ``i`` and
     ``i+1`` of the same object:

         c_i = max_k || r_i(t_k) - r_{i+1}(t_k) ||_2,   kilometres

     with the grid ``t_k`` anchored at ``epoch(set i+1) + use_offset`` and
     spanning ``horizon`` days. This is equation (11) applied to two element
     sets of the same object at different epochs.

(25) Candidate continuity z-score, with the scale of equation (7) applied to
     the pairwise offsets of equation (24):

         z = (c_candidate - median(c_i)) / (1.4826 median(|c_i - median(c_i)|))

     One-sided: only an offset larger than normal is evidence. A candidate
     that agrees with its predecessor *better* than the history normally does
     is not suspicious.

     The scale is floored at ``scale_floor_km``, a declared policy value, for
     the same reason the element scales are floored in
     :mod:`tlecheck.history`: an object whose consecutive element sets agree to
     the metre has a robust scale that is an artefact of how little the orbit
     moves, and dividing by it manufactures enormous z-scores from nothing.

(26) Excess continuity offset, in kilometres and not in scale units:

         x = max(0, c_candidate - median(c_i))

     Equation (25) normalises by the object's own scale, which answers
     "is this unusual for this object". Equation (26) stays in kilometres,
     which answers "is this larger than the error I can tolerate" — and that
     is the question the consequence class is defined by, because the silent
     threshold is an absolute distance. Subtracting the history median removes
     the genuine prediction error of the older element set, which is a bias and
     not noise: the predecessor is an epoch older than the candidate and its
     propagation is wrong by that much before any corruption.

     Equation (26) was added to the rule baseline after equation (25) alone was
     measured against the learned classifier, for the reason recorded at the
     top of this module: a baseline tested in the wrong units is an
     under-strength baseline, and the mission's build guide requires it to be
     strengthened rather than for the inflated margin to be published.

(27) Resolution of the screen, in kilometres:

         rho = 1.4826 median(|c_i - median(c_i)|),  floored at scale_floor_km

     This is the denominator of equation (25) reported under its own name,
     because it is the number that says what the screen can and cannot see.
     Equation (26) is only evaluated when ``rho <= d_silent``: an object whose
     consecutive element sets normally disagree by fifty kilometres cannot be
     screened for a one-kilometre consequence by comparing element sets, and
     saying so is more useful than returning a verdict that is noise.
     Measured on 212 uncorrupted element sets of the test split, by family:
     ``rho`` is a median of 0.3895 km for a sun-synchronous object, 0.4723 km
     for a low-inclination low-Earth object, 1.7898 km for a navigation object,
     2.1048 km for a critically inclined high-eccentricity object, 3.3303 km
     for a geostationary object and 29.4676 km for a decaying low-Earth object.
     The declared silent threshold is 1.0 km, so for four of the six families
     this screen withholds its absolute test by construction.

Units
-----
Kilometres and days throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .consequence import position_errors_km
from .elements import ElementSet
from .epoch import epoch_julian_date
from .history import MAD_CONSISTENCY, robust_scale
from .propagate import propagate, time_grid

MINIMUM_PAIRS = 3
"""Pairwise offsets needed before equation (25) has a scale."""

DEFAULT_SCALE_FLOOR_KM = 0.050
"""Declared floor on the equation (25) scale, kilometres."""


@dataclass(frozen=True)
class ContinuityVerdict:
    """Result of the continuity screen, including the case where it cannot run.

    Attributes
    ----------
    usable:
        ``False`` when there were too few history pairs, or when a required
        propagation was loud. ``reason`` says which.
    offset_km:
        Equation (24) for the candidate against the most recent history
        element set.
    normal_offsets_km:
        Equation (24) for each consecutive history pair.
    median_km, scale_km:
        Centre and floored robust scale of ``normal_offsets_km``.
    z_score:
        Equation (25), one-sided and never negative.
    excess_km:
        Equation (26), one-sided and never negative, in kilometres.
    threshold:
        Declared z threshold, carried with the verdict.
    exceeded:
        ``z_score > threshold``.
    resolution_km:
        The robust scale of equation (25), repeated under the name that says
        what it is for: the smallest position discrepancy this screen can
        distinguish from the object's own normal element-to-element variation.
    resolves_km:
        A callable test is not possible on a frozen dataclass, so this is the
        declared silent threshold the screen was asked about and
        ``resolves_threshold`` is the answer.
    resolves_threshold:
        ``resolution_km <= resolves_km``. When ``False``, the screen cannot
        resolve a discrepancy the size of the declared silent threshold for
        this object, and the absolute test of equation (26) is withheld rather
        than reported as a pass or a fail. This is a computed limit and it is
        returned, not described.
    """

    usable: bool
    reason: str
    offset_km: float
    normal_offsets_km: tuple[float, ...]
    median_km: float
    scale_km: float
    z_score: float
    excess_km: float
    threshold: float
    exceeded: bool
    resolution_km: float
    resolves_km: float
    resolves_threshold: bool


def _offset_km(
    earlier: ElementSet,
    later: ElementSet,
    use_offset_days: float,
    horizon_days: float,
    samples: int,
) -> float:
    """Equation (24) for one pair, or ``nan`` if either propagation was loud."""
    start = epoch_julian_date(later) + use_offset_days
    if not math.isfinite(start):
        return float("nan")
    grid = time_grid(start, horizon_days, samples)
    first = propagate(earlier, grid)
    second = propagate(later, grid)
    if first.loud or second.loud:
        return float("nan")
    return float(np.max(position_errors_km(first, second)))


def pairwise_offsets_km(
    history: list[ElementSet],
    use_offset_days: float = 1.0,
    horizon_days: float = 1.0,
    samples: int = 9,
) -> list[float]:
    """Equation (24) for every consecutive pair of ``history``, oldest first.

    Non-finite pairs are dropped: a pair that could not be propagated is not
    evidence about the object's normal continuity.
    """
    ordered = sorted(history, key=epoch_julian_date)
    offsets: list[float] = []
    for earlier, later in zip(ordered[:-1], ordered[1:], strict=False):
        value = _offset_km(earlier, later, use_offset_days, horizon_days, samples)
        if math.isfinite(value):
            offsets.append(value)
    return offsets


def check(
    candidate: ElementSet,
    history: list[ElementSet],
    use_jd: float,
    horizon_days: float = 1.0,
    samples: int = 9,
    threshold: float = 4.0,
    scale_floor_km: float = DEFAULT_SCALE_FLOOR_KM,
    use_offset_days: float = 1.0,
    resolves_km: float = 1.0,
) -> ContinuityVerdict:
    """Test the candidate's ephemeris continuity against the object's history.

    Parameters
    ----------
    candidate:
        The element set being screened.
    history:
        Prior element sets for the same object, any order.
    use_jd:
        First Julian date of the comparison grid, days.
    horizon_days, samples:
        Grid for equations (24) and (25).
    threshold:
        Declared z threshold for equation (25).
    scale_floor_km:
        Declared floor on the equation (25) scale.
    resolves_km:
        The declared silent threshold the screen is asked to resolve, in
        kilometres. The verdict reports whether it can.
    use_offset_days:
        Offset from each history element set's epoch at which the pairwise
        offsets of equation (24) are evaluated. Kept equal to the candidate's
        own use offset by the caller so that the candidate's offset and the
        history's offsets are comparable.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive and finite, ``scale_floor_km`` is
        negative, or a history entry belongs to a different object.
    """
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
    if not math.isfinite(scale_floor_km) or scale_floor_km < 0.0:
        raise ValueError(
            f"scale_floor_km must be non-negative and finite, got {scale_floor_km!r}"
        )
    for prior in history:
        if prior.satnum != candidate.satnum:
            raise ValueError(
                f"history entry for object {prior.satnum} supplied for candidate "
                f"object {candidate.satnum}"
            )
    nan = float("nan")
    if len(history) < MINIMUM_PAIRS + 1:
        return ContinuityVerdict(
            usable=False,
            reason=(
                f"{len(history)} prior element sets supplied; equation (25) needs at "
                f"least {MINIMUM_PAIRS + 1} to form {MINIMUM_PAIRS} pairwise offsets, "
                "so no continuity test was run"
            ),
            offset_km=nan,
            normal_offsets_km=(),
            median_km=nan,
            scale_km=nan,
            z_score=nan,
            excess_km=nan,
            threshold=threshold,
            exceeded=False,
            resolution_km=nan,
            resolves_km=resolves_km,
            resolves_threshold=False,
        )

    offsets = pairwise_offsets_km(history, use_offset_days, horizon_days, samples)
    if len(offsets) < MINIMUM_PAIRS:
        return ContinuityVerdict(
            usable=False,
            reason=(
                f"only {len(offsets)} of the history's consecutive pairs propagated "
                f"without a loud failure; equation (25) needs {MINIMUM_PAIRS}"
            ),
            offset_km=nan,
            normal_offsets_km=tuple(offsets),
            median_km=nan,
            scale_km=nan,
            z_score=nan,
            excess_km=nan,
            threshold=threshold,
            exceeded=False,
            resolution_km=nan,
            resolves_km=resolves_km,
            resolves_threshold=False,
        )

    latest = max(history, key=epoch_julian_date)
    grid = time_grid(use_jd, horizon_days, samples)
    candidate_propagation = propagate(candidate, grid)
    latest_propagation = propagate(latest, grid)
    if candidate_propagation.loud or latest_propagation.loud:
        which = "the candidate" if candidate_propagation.loud else "the latest history entry"
        return ContinuityVerdict(
            usable=False,
            reason=(
                f"{which} propagated loudly, so equation (24) has no finite value for "
                "the candidate; a loud failure is reported by the propagator itself and "
                "needs no screen"
            ),
            offset_km=nan,
            normal_offsets_km=tuple(offsets),
            median_km=float(np.median(offsets)),
            scale_km=nan,
            z_score=nan,
            excess_km=nan,
            threshold=threshold,
            exceeded=False,
            resolution_km=nan,
            resolves_km=resolves_km,
            resolves_threshold=False,
        )

    offset = float(np.max(position_errors_km(candidate_propagation, latest_propagation)))
    array = np.array(offsets, dtype=float)
    centre = float(np.median(array))
    scale = max(robust_scale(array), scale_floor_km)
    z = max(0.0, (offset - centre) / scale) if scale > 0.0 else 0.0
    return ContinuityVerdict(
        usable=True,
        reason="",
        offset_km=offset,
        normal_offsets_km=tuple(offsets),
        median_km=centre,
        scale_km=scale,
        z_score=z,
        excess_km=max(0.0, offset - centre),
        threshold=threshold,
        exceeded=z > threshold,
        resolution_km=scale,
        resolves_km=resolves_km,
        resolves_threshold=scale <= resolves_km,
    )


def consistency_constant() -> float:
    """The 1.4826 constant of equation (7), re-exported for the report."""
    return MAD_CONSISTENCY
