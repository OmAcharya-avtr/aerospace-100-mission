"""The rule baseline: a predicted consequence class and a sweepable risk score.

This is the operational default of the package. It is a set of explicit screens
with declared thresholds, it is implemented and validated before the learned
component, and it is given a continuous score as well as a decision so that it
can be compared against a learned classifier curve-against-curve rather than
point-against-curve. Scoring a rule set only at its own operating point
understates it, and understating the baseline is how a learned model is made to
look better than it is.

The decision table, in full
---------------------------
Two things are decided and they are kept apart, because conflating them is the
mistake this product exists to stop.

**Integrity** is whether the element set is what its source wrote. A checksum
mismatch settles that on its own and nothing else is needed.

**Consequence** is whether the ephemeris the propagator produces from it will be
wrong, and by how much. A checksum mismatch says nothing at all about that: the
changed character may have been the last digit of the revolution counter.

So the verdict carries ``integrity_ok`` separately from ``predicted_class``, and
only evidence that has a *magnitude* enters the consequence score:

================================================ ========================
evidence                                         consequence
================================================ ========================
a *physical* bound is violated                   ``will_fail_loudly``
equation (15) score above 1                      ``silently_wrong``
otherwise                                        ``benign``
================================================ ========================

A field this package could not read sets equation (15) to infinity rather than
predicting a loud failure. The distinction is measured, not stylistic: a line
whose characters have shifted is unreadable *by this package* and is very often
read without complaint *by the propagator*, which then returns a confident wrong
position. On the test split the rule baseline predicted a loud failure for 13
field-shifted element sets of which the propagator refused 1; 10 of the other 12
were measured silently wrong. Predicting a loud failure from this package's own
parse failure drove the baseline's operating characteristic on that corruption
family **below chance**, to an area of 0.3686, because the records most likely to
be silently wrong were given the lowest possible score. With the infinity the
same figure is in the table in ``validation/VALIDATION.md``.

================================================ ========================
evidence                                         integrity
================================================ ========================
the checksum does not match                      not intact
the checksum column is not a digit               not intact
a numeric field did not parse                    not intact
otherwise                                        intact
================================================ ========================

A physical bound violation predicts a loud failure because the SGP4 error codes
are exactly these cases: eccentricity outside the unit interval is error 1 or
3, a non-positive mean motion is error 2, a negative semilatus rectum is error
4, and a decayed orbit is error 6. The prediction is still a prediction and it
is scored against what the propagator actually did; the agreement is measured
in ``validation/validate_consequence.py`` and reported as found.

**Why the checksum indicator is not in the consequence score, with the number.**
It was, in the first version of this module, as a declared constant of 1.0. On
the 420-object test split that put 31 element sets whose only corruption was the
checksum column itself — data intact, measured position error exactly 0.0 km —
into the ``silently_wrong`` class, which is 31 of the baseline's 56 avoidable
false positives. The checksum screen was right and the consequence prediction
was wrong, and they were the same decision. Separating them is not a weakening
of the checksum screen: a checksum mismatch is still reported, still fails
``integrity_ok``, and still appears in ``problems``.

Equation
--------
(15) Silent risk score, dimensionless, 1.0 at the screens' own thresholds:

        S = max( z_hist / z_thresh,
                 z_cont / z_cont_thresh,
                 x_cont / d_silent   (only when equation (27) resolves it),
                 d_cross / d_thresh,
                 age / age_req )

    with ``S = 0`` whenever a physical bound is violated, because an element set
    the propagator will refuse is not a member of the silent class, and
    ``S = inf`` whenever a field could not be read, because an element set this
    package cannot parse is maximally suspicious and none of the magnitude
    terms above can be computed for it. Each term is
    normalised by the screen's own declared threshold, so ``S > 1`` is exactly
    the rules' own operating point and sweeping ``S`` traces the rule
    baseline's full operating characteristic.

    Every term carries a magnitude. There are no indicator constants in
    equation (15), for the reason and with the number given above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import determinability as determinability_module
from . import screens as screens_module
from .bounds import BoundsPolicy
from .consequence import (
    BENIGN,
    DEFAULT_SILENT_THRESHOLD_KM,
    SILENTLY_WRONG,
    WILL_FAIL_LOUDLY,
)
from .elements import ElementSet

UNREADABLE_FORMAT_NAMES = frozenset({"parse_fault", "checksum_column_not_a_digit"})
"""Format findings that mean this package could not read a field.

They set equation (15) to infinity. They do **not** predict a loud propagator
failure; see the decision table in the module docstring for the measurement
that settled that.
"""

DEFAULT_HISTORY_THRESHOLD = 4.0
"""Fallback normaliser for equation (15) when no history finding fired."""


@dataclass(frozen=True)
class Problem:
    """One screened problem with its predicted consequence class."""

    screen: str
    name: str
    predicted_class: str
    detail: str


@dataclass(frozen=True)
class Verdict:
    """Quality verdict for one element set.

    The first field is :attr:`determinability`. That ordering is deliberate and
    is the batch requirement this product is built against: the verdict cannot
    be read without the statement of what the screen could not determine.

    Attributes
    ----------
    determinability:
        Nine limits from :func:`tlecheck.determinability.report`, computed
        before any screen ran.
    predicted_class:
        The rule baseline's predicted consequence class, one of
        ``will_fail_loudly``, ``silently_wrong``, ``benign``.
    silent_risk_score:
        Equation (15). The rules flag the silent class at ``> 1``.
    integrity_ok:
        ``False`` when the element set is not what its source wrote: the
        checksum mismatches, the checksum column is not a digit, or a field did
        not parse. This is a separate output from
        :attr:`predicted_class` and the two can disagree, which is the point.
    problems:
        Every screened problem, each with the class it predicts.
    evidence:
        The raw screen evidence.
    thresholds:
        Every declared policy value used, so the verdict cannot be quoted
        without them.
    """

    determinability: tuple[determinability_module.Limit, ...]
    predicted_class: str
    silent_risk_score: float
    integrity_ok: bool
    problems: tuple[Problem, ...]
    evidence: screens_module.ScreenEvidence
    thresholds: dict[str, float]


def _risk_terms(
    evidence: screens_module.ScreenEvidence,
    silent_threshold_km: float = DEFAULT_SILENT_THRESHOLD_KM,
) -> dict[str, float]:
    """The individual terms of equation (15), before the maximum."""
    terms: dict[str, float] = {}
    history = evidence.history
    terms["history"] = 0.0
    if history is not None and history.usable and history.z_scores:
        threshold = history.findings[0].threshold if history.findings else DEFAULT_HISTORY_THRESHOLD
        scores = list(history.z_scores.values())
        if any(z == math.inf for z in scores):
            # A zero robust scale with a non-zero residual. Reported as the
            # infinity it is rather than clipped to a number nothing measured.
            terms["history"] = math.inf
        else:
            finite = [z / threshold for z in scores if math.isfinite(z)]
            terms["history"] = max(finite) if finite else 0.0

    continuity = evidence.continuity
    if continuity is not None and continuity.usable and continuity.threshold > 0.0:
        terms["continuity_z"] = continuity.z_score / continuity.threshold
        terms["continuity_excess_km"] = (
            continuity.excess_km / silent_threshold_km
            if continuity.resolves_threshold
            else 0.0
        )
    else:
        terms["continuity_z"] = 0.0
        terms["continuity_excess_km"] = 0.0

    cross = 0.0
    for divergence in evidence.divergences:
        if divergence.usable and divergence.threshold_km > 0.0:
            cross = max(cross, divergence.max_separation_km / divergence.threshold_km)
    terms["cross_source"] = cross

    epoch = evidence.epoch
    if epoch is not None and math.isfinite(epoch.age_days) and epoch.requirement_days > 0.0:
        terms["epoch"] = max(0.0, epoch.age_days / epoch.requirement_days)
    else:
        terms["epoch"] = 0.0

    return terms


def classify(
    evidence: screens_module.ScreenEvidence,
    silent_threshold_km: float = DEFAULT_SILENT_THRESHOLD_KM,
) -> tuple[str, float, tuple[Problem, ...], bool]:
    """Apply the decision table and equation (15) to screen evidence.

    Returns
    -------
    tuple[str, float, tuple[Problem, ...], bool]
        Predicted consequence class, equation (15) score, the problems found,
        and whether the element set's integrity is intact.
    """
    problems: list[Problem] = []
    loud = any(finding.kind == "physical" for finding in evidence.bound_findings)
    unreadable = any(
        finding.name in UNREADABLE_FORMAT_NAMES for finding in evidence.format_findings
    )
    integrity_ok = not evidence.format_findings

    terms = _risk_terms(evidence, silent_threshold_km)
    magnitude = max(terms.values()) if terms else 0.0
    score_if_not_loud = math.inf if unreadable else magnitude

    def magnitude_class() -> str:
        return SILENTLY_WRONG if score_if_not_loud > 1.0 else BENIGN

    for finding in evidence.format_findings:
        is_unreadable = finding.name in UNREADABLE_FORMAT_NAMES
        detail = finding.detail
        if is_unreadable:
            detail += (
                "; this package could not read the field, so every magnitude term of "
                "equation (15) is unavailable and the score is set to infinity. The "
                "propagator may well read something from these characters without "
                "complaint, and whatever it reads is not what the source wrote."
            )
        else:
            detail += (
                "; the element set's integrity is not intact. Equation (15) scores "
                f"{score_if_not_loud:.4f}, so the predicted consequence of the change "
                + (
                    "is a wrong ephemeris."
                    if score_if_not_loud > 1.0
                    else "is benign: no element screen with a magnitude saw anything. "
                    "Which character changed is not determinable from a checksum."
                )
            )
        problems.append(
            Problem(
                screen="format",
                name=finding.name,
                predicted_class=WILL_FAIL_LOUDLY if loud else magnitude_class(),
                detail=detail,
            )
        )
    for finding in evidence.bound_findings:
        is_loud = finding.kind == "physical"
        problems.append(
            Problem(
                screen="bounds",
                name=finding.name,
                predicted_class=WILL_FAIL_LOUDLY if is_loud else magnitude_class(),
                detail=finding.detail
                + (
                    ""
                    if is_loud
                    else f"; this is a {finding.kind} band and carries no magnitude, so "
                    f"the consequence prediction comes from equation (15), which scores "
                    f"{score_if_not_loud:.4f}"
                ),
            )
        )
    epoch = evidence.epoch
    if epoch is not None and epoch.stale:
        problems.append(
            Problem(
                screen="epoch",
                name="epoch_stale",
                predicted_class=SILENTLY_WRONG,
                detail=(
                    f"epoch age {epoch.age_days:.4f} days exceeds the declared "
                    f"requirement of {epoch.requirement_days:g} days"
                ),
            )
        )
    if epoch is not None and epoch.future_epoch:
        problems.append(
            Problem(
                screen="epoch",
                name="epoch_in_future",
                predicted_class=SILENTLY_WRONG,
                detail=(
                    f"epoch is {-epoch.age_days:.4f} days ahead of the reference time, "
                    f"beyond the declared tolerance of "
                    f"{epoch.future_tolerance_days:g} days"
                ),
            )
        )
    history = evidence.history
    if history is not None:
        for finding in history.findings:
            problems.append(
                Problem(
                    screen="history",
                    name=f"history_outlier_{finding.element}",
                    predicted_class=SILENTLY_WRONG,
                    detail=(
                        f"{finding.element} is {finding.z_score:.4g} robust scales from "
                        f"its {finding.test} prediction {finding.predicted:.8g} "
                        f"(threshold {finding.threshold:g}, value {finding.value:.8g})"
                    ),
                )
            )
    continuity = evidence.continuity
    if continuity is not None and continuity.usable and (
        continuity.exceeded
        or (continuity.resolves_threshold and continuity.excess_km > silent_threshold_km)
    ):
        problems.append(
            Problem(
                screen="continuity",
                name="ephemeris_continuity_exceeded",
                predicted_class=SILENTLY_WRONG,
                detail=(
                    f"the candidate's ephemeris departs from its predecessor's by "
                    f"{continuity.offset_km:.4f} km, against an object history whose "
                    f"consecutive element sets normally depart by "
                    f"{continuity.median_km:.4f} km with a robust scale of "
                    f"{continuity.scale_km:.4f} km; equation (25) z-score "
                    f"{continuity.z_score:.4g} against the declared threshold "
                    f"{continuity.threshold:g}, and equation (26) excess "
                    f"{continuity.excess_km:.4f} km against the declared silent "
                    f"threshold {silent_threshold_km:g} km; the screen's own "
                    f"resolution by equation (27) is {continuity.resolution_km:.4f} km, "
                    + (
                        "which resolves that threshold"
                        if continuity.resolves_threshold
                        else "which does NOT resolve that threshold, so the absolute "
                        "test was withheld"
                    )
                ),
            )
        )
    for divergence in evidence.divergences:
        if divergence.diverged:
            problems.append(
                Problem(
                    screen="cross-source",
                    name="cross_source_divergence",
                    predicted_class=SILENTLY_WRONG,
                    detail=(
                        f"sources {divergence.sources[0]!r} and {divergence.sources[1]!r} "
                        f"separate by {divergence.max_separation_km:.4f} km at most "
                        f"(threshold {divergence.threshold_km:g} km); which one is wrong "
                        "is not determinable here"
                    ),
                )
            )

    score = 0.0 if loud else score_if_not_loud
    if loud:
        predicted = WILL_FAIL_LOUDLY
    elif score > 1.0:
        predicted = SILENTLY_WRONG
    else:
        predicted = BENIGN
    return predicted, score, tuple(problems), integrity_ok


def screen(
    element_set: ElementSet,
    history: list[ElementSet] | None = None,
    other_sources: list[ElementSet] | None = None,
    reference_jd: float | None = None,
    epoch_requirement_days: float = 7.0,
    history_threshold: float = 4.0,
    divergence_threshold_km: float = 1.0,
    continuity_threshold: float = 4.0,
    silent_threshold_km: float = DEFAULT_SILENT_THRESHOLD_KM,
    policy: BoundsPolicy | None = None,
) -> Verdict:
    """Screen one element set and return its verdict, determinability first.

    This is the package's main entry point. The determinability report is
    computed *before* the screens run and is the first field of the result.

    Parameters
    ----------
    element_set:
        Parsed element set, from :func:`tlecheck.elements.parse_tle`.
    history:
        Prior element sets for the same object, or ``None`` for none offered.
    other_sources:
        Element sets for the same object from other catalogues.
    reference_jd:
        Julian date the element set is to be used at, days. ``None`` skips the
        epoch screen, which is recorded as a determinability limit rather than
        passed over.
    epoch_requirement_days:
        Declared maximum acceptable epoch age in days.
    history_threshold:
        Declared modified z-score threshold.
    divergence_threshold_km:
        Declared cross-source separation threshold in kilometres.
    continuity_threshold:
        Declared equation (25) z threshold for the continuity screen.
    silent_threshold_km:
        Declared absolute threshold in kilometres, used by equation (26) and by
        :func:`tlecheck.consequence.measure`. One number, used by both, so the
        screen and the consequence it is screening for cannot disagree.
    policy:
        Declared bound bands.

    Returns
    -------
    Verdict
    """
    policy = policy or BoundsPolicy()
    limits = determinability_module.report(
        element_set,
        history,
        sources=1 + len(other_sources or []),
        epoch_requirement_days=epoch_requirement_days if reference_jd is not None else None,
        policy=policy,
        silent_threshold_km=silent_threshold_km,
    )
    evidence = screens_module.run(
        element_set,
        history=history,
        other_sources=other_sources,
        reference_jd=reference_jd,
        epoch_requirement_days=epoch_requirement_days,
        history_threshold=history_threshold,
        divergence_threshold_km=divergence_threshold_km,
        continuity_threshold=continuity_threshold,
        silent_threshold_km=silent_threshold_km,
        policy=policy,
    )
    predicted, score, problems, integrity_ok = classify(evidence, silent_threshold_km)
    return Verdict(
        determinability=limits,
        predicted_class=predicted,
        silent_risk_score=score,
        integrity_ok=integrity_ok,
        problems=problems,
        evidence=evidence,
        thresholds={
            "epoch_requirement_days": epoch_requirement_days,
            "history_threshold": history_threshold,
            "divergence_threshold_km": divergence_threshold_km,
            "continuity_threshold": continuity_threshold,
            "silent_threshold_km": silent_threshold_km,
            "min_perigee_altitude_km": policy.min_perigee_altitude_km,
            "max_eccentricity": policy.max_eccentricity,
            "max_abs_bstar": policy.max_abs_bstar,
            "min_mean_motion_rev_per_day": policy.min_mean_motion_rev_per_day,
        },
    )
