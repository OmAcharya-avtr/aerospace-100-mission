"""What this screen cannot determine, computed and returned before the verdict.

This module is the first-class output of the package, not a caveat attached to
one. :func:`report` is called by :func:`tlecheck.screen.screen` before any
screen runs, and its result is the first field of every verdict this package
produces. A quality verdict read without it is a confident scalar computed from
information that does not support it.

Each limit carries a ``basis``:

``structural``
    Not determinable from a two-line element set and a propagator, at any
    corruption rate, with any amount of history. More data of the kind this
    package accepts does not help. The caller needs an independent
    observation.

``computed``
    A number this module derives, by enumeration, by arithmetic on the element
    set supplied, or by propagating the supplied history against itself.

``data-dependent``
    Determinable in principle, not determinable from what the caller supplied.
    The limit says what would be needed.

Equations
---------
(13) Checksum detection fraction over a corruption family F:

        p_det(F) = |{c in F : sum_mod_10(c) != sum_mod_10(original)}| / |F|

    Three families are enumerated exhaustively on the element set supplied,
    which makes equation (13) an exact rational number for that element set and
    not an estimate:

    - every single-digit substitution: every digit column, every one of the
      nine other digits;
    - every transposition of two digit columns holding different digits;
    - every simultaneous substitution at two distinct digit columns.

    The first family is detected with probability exactly 1, because a single
    digit change alters the sum by a non-zero amount strictly between -9 and 9.
    The second is detected with probability exactly 0, because a transposition
    does not alter the sum at all. The third is the interesting one and its
    value is reported.

    The checksum rule itself is the modulo-10 digit sum of Hoots and Roehrich
    (1980), Spacetrack Report No. 3; the enumeration is this module's.

(14) Bound slack in history scale units, for element x:

        slack(x) = min(x - lower, upper - x) / s_history(x)

    with ``s_history`` the scaled median absolute deviation of equation (7).
    This is the number of history-sigma of room a corruption has *inside* the
    plausible band, which is an upper bound on what the bounds screen alone can
    be expected to catch. A large slack means the bounds screen is not the
    screen doing the work for that element.

Units
-----
Fractions are dimensionless. Slacks are in units of the element's own robust
history scale and are therefore dimensionless. Ages are days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import continuity as continuity_module
from .bounds import BoundsPolicy, mean_motion_ceiling_rev_per_day
from .elements import ElementSet, checksum
from .history import (
    LEVEL_ELEMENTS,
    MINIMUM_HISTORY,
    TREND_ELEMENTS,
    UNTESTED_ELEMENTS,
    robust_scale,
)

STRUCTURAL = "structural"
COMPUTED = "computed"
DATA_DEPENDENT = "data-dependent"


@dataclass(frozen=True)
class Limit:
    """One thing the screen cannot determine, with its basis and its number.

    Attributes
    ----------
    name:
        Stable identifier.
    determinable:
        ``True`` only when the screen can answer this question from what was
        supplied. A ``computed`` limit may be determinable and still carry a
        number that caps what a screen can see.
    basis:
        ``structural``, ``computed`` or ``data-dependent``.
    value:
        The number, or ``None`` where the limit is purely structural.
    statement:
        One sentence a reader can act on.
    """

    name: str
    determinable: bool
    basis: str
    value: float | None
    statement: str


def checksum_detection_fractions(line: str) -> dict[str, float]:
    """Equation (13) for the three enumerated families, on one line.

    Returns a mapping with keys ``single_substitution``, ``transposition`` and
    ``double_substitution``, plus the family sizes under ``*_count`` keys so
    the fractions can be checked by hand.

    Raises
    ------
    ValueError
        If ``line`` is shorter than the 69 columns a conforming line has.
    """
    if len(line) < 69:
        raise ValueError(f"line has {len(line)} columns, need at least 69")
    body = line[:68]
    original = checksum(body)
    positions = [index for index, character in enumerate(body) if character.isdigit()]
    digits = [int(body[index]) for index in positions]

    single_total = 0
    single_detected = 0
    for digit in digits:
        for replacement in range(10):
            if replacement == digit:
                continue
            single_total += 1
            if (original + replacement - digit) % 10 != original % 10:
                single_detected += 1

    transposition_total = 0
    transposition_detected = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if digits[i] == digits[j]:
                continue
            transposition_total += 1
            # The digit-sum change of swapping the two columns, written out
            # rather than asserted to be zero, so the reported fraction is
            # arithmetic on the actual digits and not a restated claim.
            delta = (digits[j] - digits[i]) + (digits[i] - digits[j])
            if delta % 10 != 0:
                transposition_detected += 1

    double_total = 0
    double_detected = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            for first in range(10):
                if first == digits[i]:
                    continue
                for second in range(10):
                    if second == digits[j]:
                        continue
                    double_total += 1
                    delta = (first - digits[i]) + (second - digits[j])
                    if delta % 10 != 0:
                        double_detected += 1

    return {
        "single_substitution": single_detected / single_total if single_total else float("nan"),
        "single_substitution_count": float(single_total),
        "transposition": (
            transposition_detected / transposition_total
            if transposition_total
            else float("nan")
        ),
        "transposition_count": float(transposition_total),
        "double_substitution": (
            double_detected / double_total if double_total else float("nan")
        ),
        "double_substitution_count": float(double_total),
        "digit_columns": float(len(positions)),
    }


def _element_bounds(policy: BoundsPolicy) -> dict[str, tuple[float, float]]:
    """The plausible interval of each history-tested element under ``policy``."""
    return {
        "inclination_deg": (0.0, 180.0),
        "eccentricity": (0.0, policy.max_eccentricity),
        "raan_deg": (0.0, 360.0),
        "mean_motion_rev_per_day": (
            policy.min_mean_motion_rev_per_day,
            mean_motion_ceiling_rev_per_day(),
        ),
        "bstar": (-policy.max_abs_bstar, policy.max_abs_bstar),
    }


def bound_slack(
    element_set: ElementSet,
    history: list[ElementSet],
    policy: BoundsPolicy | None = None,
) -> dict[str, float]:
    """Equation (14) for every history-tested element, dimensionless.

    An element whose history scale is zero has infinite slack reported, which
    is the honest answer: the bounds screen has the whole plausible band to
    play with and the history screen has no scale to measure against.
    """
    policy = policy or BoundsPolicy()
    slacks: dict[str, float] = {}
    for name, (lower, upper) in _element_bounds(policy).items():
        value = float(getattr(element_set, name))
        if not math.isfinite(value):
            slacks[name] = float("nan")
            continue
        room = min(value - lower, upper - value)
        if len(history) < MINIMUM_HISTORY:
            slacks[name] = float("inf")
            continue
        series = np.array([getattr(prior, name) for prior in history], dtype=float)
        scale = robust_scale(series)
        if not math.isfinite(scale):
            slacks[name] = float("nan")
        elif scale == 0.0:
            slacks[name] = float("inf")
        else:
            slacks[name] = room / scale
    return slacks


def report(
    element_set: ElementSet,
    history: list[ElementSet] | None = None,
    sources: int = 1,
    epoch_requirement_days: float | None = None,
    policy: BoundsPolicy | None = None,
    silent_threshold_km: float = 1.0,
) -> tuple[Limit, ...]:
    """Everything this screen cannot determine about ``element_set``.

    Called before any screen runs, and computable before them: every limit
    below is a function of the element set text, the history, the source count
    and the declared policy, and none of them needs the candidate to have been
    screened. Eleven limits are returned, always in the same order, so a report
    can be diffed between element sets.

    Parameters
    ----------
    element_set:
        The element set about to be screened.
    history:
        Prior element sets for the same object, or ``None``.
    sources:
        How many independent sources supplied an element set for this object.
    epoch_requirement_days:
        The declared maximum acceptable epoch age, or ``None`` when the caller
        has not declared one.
    policy:
        Declared bound bands.
    silent_threshold_km:
        The declared silent threshold, in kilometres, that the continuity
        resolution limit is compared against.

    Returns
    -------
    tuple[Limit, ...]
    """
    history = list(history or [])
    policy = policy or BoundsPolicy()
    fractions = (
        checksum_detection_fractions(element_set.line1)
        if len(element_set.line1) >= 69
        else {"double_substitution": float("nan"), "transposition": float("nan")}
    )
    slacks = bound_slack(element_set, history, policy)
    offsets = (
        continuity_module.pairwise_offsets_km(history)
        if len(history) > continuity_module.MINIMUM_PAIRS
        else []
    )
    continuity_resolution = (
        max(
            float(np.median(np.abs(np.array(offsets) - np.median(offsets))) * 1.4826),
            continuity_module.DEFAULT_SCALE_FLOOR_KM,
        )
        if len(offsets) >= continuity_module.MINIMUM_PAIRS
        else float("nan")
    )
    finite_slacks = [value for value in slacks.values() if math.isfinite(value)]
    worst_slack = max(finite_slacks) if finite_slacks else float("inf")
    worst_element = (
        max((name for name in slacks if math.isfinite(slacks[name])), key=lambda n: slacks[n])
        if finite_slacks
        else "none"
    )

    limits = [
        Limit(
            name="propagated_ephemeris_accuracy",
            determinable=False,
            basis=STRUCTURAL,
            value=None,
            statement=(
                "Whether the ephemeris propagated from this element set is accurate is "
                "not determinable here. An element set is a fit, the fit residuals are "
                "not distributed with it, and no independent observation is available to "
                "this package. A clean screen means no screened problem was found; it is "
                "not a statement about ephemeris accuracy."
            ),
        ),
        Limit(
            name="which_source_is_correct",
            determinable=False,
            basis=STRUCTURAL,
            value=None,
            statement=(
                "When two sources disagree, which one is right is not determinable from "
                "the element sets. Equation (12) is symmetric in its two arguments; "
                "breaking the symmetry needs an independent observation."
            ),
        ),
        Limit(
            name="element_covariance",
            determinable=False,
            basis=STRUCTURAL,
            value=None,
            statement=(
                "The two-line element format carries no covariance and no fit residuals, "
                "so the uncertainty of each element is not determinable from the record. "
                "Every uncertainty this package reports is an uncertainty about its own "
                "screening decision and never about the elements themselves."
            ),
        ),
        Limit(
            name="checksum_blindness_to_transposition",
            determinable=True,
            basis=COMPUTED,
            value=float(fractions["transposition"]),
            statement=(
                "Equation (13), enumerated exhaustively over "
                f"{int(fractions.get('transposition_count', 0))} digit-column "
                "transpositions of line 1: the checksum detects a fraction "
                f"{fractions['transposition']:.6f} of them. A transposition does not "
                "change a digit sum, so the checksum screen is blind to this family by "
                "construction and no configuration changes that."
            ),
        ),
        Limit(
            name="checksum_blindness_to_double_substitution",
            determinable=True,
            basis=COMPUTED,
            value=float(fractions["double_substitution"]),
            statement=(
                "Equation (13), enumerated exhaustively over "
                f"{int(fractions.get('double_substitution_count', 0))} two-column digit "
                "substitutions of line 1: the checksum detects a fraction "
                f"{fractions['double_substitution']:.6f}. The complement is the fraction "
                "of two-digit corruptions that reach the propagator with a valid "
                "checksum."
            ),
        ),
        Limit(
            name="bound_slack_in_history_scale",
            determinable=bool(finite_slacks),
            basis=COMPUTED,
            value=worst_slack,
            statement=(
                "Equation (14): the loosest history-tested element is "
                f"{worst_element}, which has {worst_slack:.4g} robust history scales of "
                "room inside its plausible band. A corruption smaller than that is "
                "invisible to the bounds screen and can only be seen by the history "
                "screen or by cross-source comparison."
                if finite_slacks
                else (
                    "Equation (14) has no finite value: no element has a usable history "
                    "scale, so the room a corruption has inside the plausible band "
                    "cannot be expressed in history units."
                )
            ),
        ),
        Limit(
            name="continuity_resolution_km",
            determinable=math.isfinite(continuity_resolution),
            basis=COMPUTED,
            value=continuity_resolution,
            statement=(
                "Equation (27): the object's own consecutive element sets disagree, "
                "when propagated to a common time, with a robust scale of "
                f"{continuity_resolution:.4f} km. The declared silent threshold is "
                f"{silent_threshold_km:g} km, so the continuity screen "
                + (
                    "resolves it and its absolute test was evaluated."
                    if continuity_resolution <= silent_threshold_km
                    else "does NOT resolve it: a discrepancy of that size is "
                    "indistinguishable from this object's normal element-to-element "
                    "variation, and the absolute test was withheld."
                )
                if math.isfinite(continuity_resolution)
                else (
                    "Equation (27) has no value: fewer than "
                    f"{continuity_module.MINIMUM_PAIRS} history pairs propagated, so the "
                    "resolution of the continuity screen is not determinable and the "
                    "screen did not run."
                )
            ),
        ),
        Limit(
            name="history_consistency",
            determinable=len(history) >= MINIMUM_HISTORY,
            basis=DATA_DEPENDENT,
            value=float(len(history)),
            statement=(
                f"{len(history)} prior element sets supplied. Equation (7) needs at "
                f"least {MINIMUM_HISTORY} for a scale, so the history screen "
                + ("ran." if len(history) >= MINIMUM_HISTORY else "did not run.")
                + " The elements it never tests in any case are "
                + ", ".join(UNTESTED_ELEMENTS)
                + ", because both circulate at rates a short history cannot separate "
                "from a corruption; the elements it does test are "
                + ", ".join(LEVEL_ELEMENTS + TREND_ELEMENTS)
                + "."
            ),
        ),
        Limit(
            name="cross_source_divergence",
            determinable=sources >= 2,
            basis=DATA_DEPENDENT,
            value=float(sources),
            statement=(
                f"{sources} source(s) supplied for this object. Equation (12) needs two, "
                "so cross-source divergence "
                + ("was measured." if sources >= 2 else "is not determinable here.")
            ),
        ),
        Limit(
            name="epoch_staleness_requirement",
            determinable=epoch_requirement_days is not None,
            basis=DATA_DEPENDENT,
            value=epoch_requirement_days,
            statement=(
                "Epoch age is computed from the record; whether that age is acceptable "
                + (
                    f"was decided against the declared requirement of "
                    f"{epoch_requirement_days:g} days."
                    if epoch_requirement_days is not None
                    else "is not determinable, because no requirement was declared. The "
                    "requirement follows from what the ephemeris is for and cannot be "
                    "derived from the element set."
                )
            ),
        ),
        Limit(
            name="epoch_year_window",
            determinable=True,
            basis=COMPUTED,
            value=2056.0,
            statement=(
                "The two-digit epoch year is resolved by the 1957-2056 window of Vallado "
                f"et al. (2006) section 2, giving {element_set.epoch_full_year} for the "
                f"written value {element_set.epoch_year:02d}. The convention cannot "
                "express 2057 onward and this package will decode such an epoch into the "
                "twentieth century without complaint."
            ),
        ),
    ]
    return tuple(limits)
