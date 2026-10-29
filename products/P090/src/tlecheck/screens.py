"""The rule screens: evidence, with no verdict attached.

Six screens run here and each returns what it found and nothing more. The
separation matters because a screen that returns a verdict cannot be scored
against a measured consequence, and scoring the rules against measured
consequence is the whole experiment this product runs.

The screens
-----------
``format`` and ``checksum``
    Columns read by span, lines sized, checksum recomputed by the modulo-10
    digit sum of Hoots and Roehrich (1980). A failing checksum says a digit
    changed somewhere; it does not say which digit, and this package does not
    pretend otherwise.

``bounds``
    :mod:`tlecheck.bounds`. Physical, format and policy bands kept apart.

``epoch``
    :mod:`tlecheck.epoch`. Age against a declared requirement, plus the
    distinct case of an epoch in the future of the reference time.

``history``
    :mod:`tlecheck.history`. The object against itself, with an explicit
    robust outlier test and an explicit refusal when there is not enough
    history to run one.

``continuity``
    :mod:`tlecheck.continuity`. The candidate's ephemeris against the
    ephemeris its own predecessor predicts, in kilometres, with the tolerance
    measured from the object's own history. This is the screen that translates
    an element deviation into the unit the consequence is measured in.

``cross-source``
    :mod:`tlecheck.divergence`. Measured in kilometres, symmetric, and unable
    to say which source is wrong.

Units
-----
Kilometres, days, degrees, revolutions per day, as in the modules above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import bounds as bounds_module
from . import continuity as continuity_module
from . import divergence as divergence_module
from . import history as history_module
from .elements import ElementSet
from .epoch import EpochVerdict, epoch_age


@dataclass(frozen=True)
class FormatFinding:
    """One format or checksum problem, with the line it is on."""

    name: str
    line: int
    detail: str


@dataclass(frozen=True)
class ScreenEvidence:
    """Everything the five screens found, with no verdict.

    Attributes
    ----------
    format_findings:
        Column, length and checksum problems.
    bound_findings:
        :class:`tlecheck.bounds.BoundFinding` entries.
    epoch:
        :class:`tlecheck.epoch.EpochVerdict`, or ``None`` when no reference
        time was supplied.
    history:
        :class:`tlecheck.history.HistoryVerdict`, or ``None`` when no history
        argument was supplied at all. An empty history is not the same as no
        history argument, and the two are kept apart.
    continuity:
        :class:`tlecheck.continuity.ContinuityVerdict`, or ``None`` when no
        history argument was supplied.
    divergences:
        One :class:`tlecheck.divergence.Divergence` per other source.
    derived:
        Orbit geometry from equations (1) and (2).
    """

    format_findings: tuple[FormatFinding, ...] = ()
    bound_findings: tuple[bounds_module.BoundFinding, ...] = ()
    epoch: EpochVerdict | None = None
    history: history_module.HistoryVerdict | None = None
    continuity: continuity_module.ContinuityVerdict | None = None
    divergences: tuple[divergence_module.Divergence, ...] = ()
    derived: bounds_module.DerivedOrbit | None = None
    names: tuple[str, ...] = field(default_factory=tuple)


def format_screen(element_set: ElementSet) -> tuple[FormatFinding, ...]:
    """Column, length and checksum findings for one element set."""
    findings: list[FormatFinding] = []
    for fault in element_set.faults:
        line = 2 if fault.startswith("line 2") else 1
        findings.append(FormatFinding("parse_fault", line, fault))
    pairs = (
        (
            element_set.line1,
            element_set.checksum_line1_stated,
            element_set.checksum_line1_computed,
        ),
        (
            element_set.line2,
            element_set.checksum_line2_stated,
            element_set.checksum_line2_computed,
        ),
    )
    for index, (line, stated, computed) in enumerate(pairs, start=1):
        if stated is None:
            findings.append(
                FormatFinding(
                    "checksum_column_not_a_digit",
                    index,
                    f"column 69 of line {index} is {line[68:69]!r}, not a digit",
                )
            )
        elif stated != computed:
            findings.append(
                FormatFinding(
                    "checksum_mismatch",
                    index,
                    f"line {index} states checksum {stated} and computes {computed}; "
                    "at least one character of columns 1-68 is not what the source wrote",
                )
            )
    return tuple(findings)


def run(
    element_set: ElementSet,
    history: list[ElementSet] | None = None,
    other_sources: list[ElementSet] | None = None,
    reference_jd: float | None = None,
    epoch_requirement_days: float = 7.0,
    history_threshold: float = 4.0,
    divergence_threshold_km: float = divergence_module.DEFAULT_DIVERGENCE_THRESHOLD_KM,
    policy: bounds_module.BoundsPolicy | None = None,
    divergence_horizon_days: float = 1.0,
    divergence_samples: int = 25,
    continuity_threshold: float = 4.0,
    continuity_samples: int = 9,
    use_offset_days: float = 1.0,
    silent_threshold_km: float = 1.0,
) -> ScreenEvidence:
    """Run all five screens and return their evidence.

    Parameters
    ----------
    element_set:
        The element set being screened.
    history:
        Prior element sets for the same object, or ``None`` to state that no
        history was offered. An empty list states that a history was offered
        and is empty, which produces a different determinability record.
    other_sources:
        Element sets for the same object from other catalogues.
    reference_jd:
        Julian date the element set is to be used at. ``None`` skips the epoch
        screen and is recorded as such.
    epoch_requirement_days, history_threshold, divergence_threshold_km:
        Declared policy values, reported with every verdict that uses them.
    policy:
        Declared bound bands.
    divergence_horizon_days, divergence_samples:
        Grid for equation (12).
    silent_threshold_km:
        The declared silent threshold in kilometres, passed to the continuity
        screen so that it can report whether it resolves one.
    continuity_threshold, continuity_samples, use_offset_days:
        Declared values for equations (24) and (25). ``use_offset_days`` must
        match the offset between the candidate's epoch and ``reference_jd`` for
        the candidate's offset and the history's offsets to be comparable; it
        is reported with the verdict.

    Raises
    ------
    ValueError
        If a history entry or another source belongs to a different object.
    """
    policy = policy or bounds_module.BoundsPolicy()
    history_list = list(history) if history is not None else None
    sources = list(other_sources or [])

    format_findings = format_screen(element_set)
    bound_findings = tuple(bounds_module.check(element_set, policy))
    derived = bounds_module.derive(element_set)

    epoch_verdict = None
    if reference_jd is not None:
        epoch_verdict = epoch_age(element_set, reference_jd, epoch_requirement_days)

    history_verdict = None
    continuity_verdict = None
    if history_list is not None:
        history_verdict = history_module.check(element_set, history_list, history_threshold)
        candidate_use_jd = reference_jd
        if candidate_use_jd is None:
            from .epoch import epoch_julian_date

            candidate_use_jd = epoch_julian_date(element_set) + use_offset_days
        continuity_verdict = continuity_module.check(
            element_set,
            history_list,
            candidate_use_jd,
            horizon_days=divergence_horizon_days,
            samples=continuity_samples,
            threshold=continuity_threshold,
            use_offset_days=use_offset_days,
            resolves_km=silent_threshold_km,
        )

    divergences: list[divergence_module.Divergence] = []
    if sources:
        start = reference_jd
        if start is None:
            from .epoch import epoch_julian_date

            start = epoch_julian_date(element_set)
        for other in sources:
            divergences.append(
                divergence_module.compare(
                    element_set,
                    other,
                    start,
                    divergence_horizon_days,
                    divergence_samples,
                    divergence_threshold_km,
                )
            )

    return ScreenEvidence(
        format_findings=format_findings,
        bound_findings=bound_findings,
        epoch=epoch_verdict,
        history=history_verdict,
        continuity=continuity_verdict,
        divergences=tuple(divergences),
        derived=derived,
        names=("format", "bounds", "epoch", "history", "continuity", "cross-source"),
    )
