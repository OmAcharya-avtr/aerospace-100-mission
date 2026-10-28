"""The three-way integrity classification, and the decision table behind it.

Classes
-------
``missing``
    The frame was not received. Concluded from absence, which is why it is the
    only class that can be concluded deductively.
``corrupt``
    The frame was received and its bits are damaged.
``misparsed``
    The frame was received intact and the interpretation applied to it is
    wrong. The class the specification calls *correctly-received-and-misparsed*.
``undecided``
    The evidence present does not separate the above. This is a verdict, not a
    failure, and :mod:`telemgap.determinability` says why for each member.

The decision table, stated in full
----------------------------------
Each row is a screen observation. ``validity`` is the per-frame error-control
verdict, ``rescue`` the outcome of the reinterpretation test of
:mod:`telemgap.reparse`.

==========================  ==========  ========  ============  ===========
observation                 validity    rescue    verdict       confidence
==========================  ==========  ========  ============  ===========
gap                         --          --        missing       1.00
frame_validity_failed       False       --        corrupt       0.95
value_in_failed_frame       False       no        undecided     0.00
value_out_of_range          False       --        corrupt       0.90
value_out_of_range          True        yes       misparsed     0.80
value_out_of_range          True        no        undecided     0.00
value_out_of_range          absent      yes       misparsed     0.60
value_out_of_range          absent      no        undecided     0.00
value_local_outlier         False       --        corrupt       0.80
value_local_outlier         True        yes       misparsed     0.70
value_local_outlier         True/absent no        undecided     0.00
repeated_counter, identical --          --        misparsed     0.90
repeated_counter, differing False       --        corrupt       0.80
repeated_counter, differing other       --        undecided     0.00
reorder                     --          --        misparsed     0.70
clock_reset, counter fwd    --          --        misparsed     0.70
clock_reset, counter back   --          --        undecided     0.00
==========================  ==========  ========  ============  ===========

Two rows need their reasoning stated rather than assumed, because the mapping
is a judgement and a reader is entitled to disagree with it.

**A reorder is classified misparsed.** Every bit of a reordered frame is
correct. What is wrong is the reader's assumption that arrival order is time
order, and any downstream analysis that indexes by arrival order is wrong in
exactly the way the misparsed class names. The alternative view -- that a
reorder is a transport property and belongs to neither class -- is defensible;
this package takes the position that it belongs with *the data is fine and your
reader is wrong*, and reports the observation alongside the verdict so a caller
who disagrees can re-map it. The same reasoning puts a clock reset with a
forward-moving counter in the misparsed class: the stream is intact and the
time field is wrong.

**A repeated counter with a byte-identical payload is classified misparsed**,
because the frame was received correctly and delivered twice, so the fault is
in the archive or reader rather than in the data or the link.

**A failed frame error-control field makes the frame corrupt and its
unremarkable channels undecided.** The field is one number over the whole
frame, so it proves damage and localises nothing. Reporting every channel of
such a frame as corrupt would be four verdicts from one bit of evidence;
reporting them as clean would discard the evidence entirely. They are
``undecided``, with the reason stated, and that population is the largest part
of the ambiguous residual set the learned component of
:mod:`telemgap.learned` is scored on.

Confidence
----------
The ``confidence`` column is a **declared ordinal tier, not a calibrated
probability**, and it is labelled as such everywhere it is reported. Nothing in
this package estimates the probability that a rule verdict is correct; the rule
tiers only order the verdicts by how much evidence stands behind them. The one
calibrated probability in the package comes from :mod:`telemgap.learned`, which
reports a class probability and a per-member vote dispersion, and whose
calibration is measured in ``validation/validate_learned.py``.

Modes
-----
``mode="specified"``
    Runs exactly the four screens the product specification names: timestamp
    monotonicity, cadence, counter continuity and declared ranges. This is the
    specified baseline.
``mode="strengthened"``
    Adds the local-residual screen of equation (5). It is reported beside the
    specified baseline because the specified baseline is structurally blind to
    a bit error that leaves a value inside its declared range -- a blindness
    quantified by the ``in_range_corruption`` determinability limit -- and the
    build guide requires the strongest honest baseline to be the one the
    learned component is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from . import determinability as determinability_module
from . import screens as screens_module
from .determinability import DeterminabilityVerdict
from .record import TelemetryRecord
from .reparse import DEFAULT_CONSISTENCY_TAU, best_rescue, reinterpret
from .screens import ScreenFindings, robust_local_statistics


class AnomalyClass(StrEnum):
    """The four verdicts. ``UNDECIDED`` is a verdict, not a failure."""

    MISSING = "missing"
    CORRUPT = "corrupt"
    MISPARSED = "misparsed"
    UNDECIDED = "undecided"


#: Observation names emitted by the classifier, in report order.
OBSERVATIONS = (
    "gap",
    "frame_validity_failed",
    "value_out_of_range",
    "value_local_outlier",
    "value_in_failed_frame",
    "repeated_counter",
    "reorder",
    "clock_reset",
)

#: The two run modes. See the module docstring.
MODES = ("specified", "strengthened")


@dataclass(frozen=True)
class Anomaly:
    """One classified observation.

    Attributes
    ----------
    observation
        One of :data:`OBSERVATIONS`.
    index
        Received-order frame index. For a gap, the index of the frame *after*
        the gap, so that the anomaly has a position on the timeline.
    channel
        Channel column, or ``-1`` for a frame-level observation.
    verdict
        The :class:`AnomalyClass`.
    confidence
        Declared ordinal tier in ``[0, 1]``, never a probability. See the
        module docstring.
    source
        ``"rule"`` for a rule verdict, ``"learned"`` when a verdict was
        supplied by :mod:`telemgap.learned`.
    evidence
        Statements supporting the verdict.
    reasons_undecided
        Determinability statements explaining an ``UNDECIDED`` verdict. Empty
        for a decided verdict.
    detail
        Named numbers behind the verdict, in the units of the quantity named.
    """

    observation: str
    index: int
    channel: int
    verdict: AnomalyClass
    confidence: float
    source: str
    evidence: tuple[str, ...] = ()
    reasons_undecided: tuple[str, ...] = ()
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def is_undecided(self) -> bool:
        """Whether this observation was left undecided."""
        return self.verdict is AnomalyClass.UNDECIDED


@dataclass
class IntegrityReport:
    """The package's output. ``determinability`` is first, by requirement.

    Attributes
    ----------
    determinability
        The seven limits, computed before any anomaly is classified.
    findings
        The raw screen evidence.
    anomalies
        The classified observations.
    mode
        ``"specified"`` or ``"strengthened"``.
    """

    determinability: DeterminabilityVerdict
    findings: ScreenFindings
    anomalies: tuple[Anomaly, ...]
    mode: str

    @property
    def counts(self) -> dict[str, int]:
        """Number of anomalies per verdict, including zeros."""
        tally = {verdict.value: 0 for verdict in AnomalyClass}
        for anomaly in self.anomalies:
            tally[anomaly.verdict.value] += 1
        return tally

    @property
    def undecided(self) -> tuple[Anomaly, ...]:
        """The ambiguous residual set the learned component is allowed to attack."""
        return tuple(anomaly for anomaly in self.anomalies if anomaly.is_undecided)

    def by_observation(self, observation: str) -> tuple[Anomaly, ...]:
        """Anomalies of one observation type."""
        if observation not in OBSERVATIONS:
            raise ValueError(
                f"unknown observation {observation!r}; expected one of {OBSERVATIONS}"
            )
        return tuple(anomaly for anomaly in self.anomalies if anomaly.observation == observation)

    @property
    def missing_frame_total(self) -> int:
        """Total frames the gaps imply were not received, from the time domain."""
        return self.findings.total_missing_frames


def _payload_identical(record: TelemetryRecord, left: int, right: int) -> bool:
    """Whether two frames carry the same payload, preferring raw words over values."""
    if record.raw_words is not None:
        return bool(np.array_equal(record.raw_words[left], record.raw_words[right]))
    left_values, right_values = record.values[left], record.values[right]
    both_nan = np.isnan(left_values) & np.isnan(right_values)
    return bool(np.all(both_nan | (left_values == right_values)))


def classify(
    record: TelemetryRecord,
    mode: str = "strengthened",
    residual_window: int = 8,
    residual_z: float = 8.0,
    consistency_tau: float = DEFAULT_CONSISTENCY_TAU,
    findings: ScreenFindings | None = None,
    verdict: DeterminabilityVerdict | None = None,
) -> IntegrityReport:
    """Screen ``record``, assess what cannot be determined, then classify.

    The order is deliberate and is the batch's binding requirement: the
    determinability verdict is computed before any anomaly is classified, and
    the classifier reads it rather than deciding independently whether the
    evidence it needs exists.

    Parameters
    ----------
    record
        The record to classify.
    mode
        ``"specified"`` for the four screens the specification names,
        ``"strengthened"`` to add the local-residual screen of equation (5).
    residual_window, residual_z
        Parameters of the local-residual screen. Ignored in ``"specified"``
        mode, where the screen's findings are discarded rather than not
        computed, so that the two modes see identical evidence.
    consistency_tau
        Tolerance of condition (7b) of the reinterpretation test, in robust
        standard deviations.
    findings, verdict
        Precomputed screen evidence and determinability verdict. Supplied by
        the benchmark so that both modes are classified from one screening
        pass; computed here when omitted.

    Returns
    -------
    IntegrityReport
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not isinstance(record, TelemetryRecord):
        raise TypeError(f"classify expects a TelemetryRecord, got {type(record).__name__}")
    if findings is None:
        findings = screens_module.screen(
            record, residual_window=residual_window, residual_z=residual_z
        )
    if verdict is None:
        verdict = determinability_module.assess(record)

    has_validity = verdict["frame_validity"].determinable
    has_raw = verdict["raw_words"].determinable
    statistics: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def local(column: int) -> tuple[np.ndarray, np.ndarray]:
        if column not in statistics:
            channel = record.schema.channels[column]
            statistics[column] = robust_local_statistics(
                record.values[:, column], residual_window, channel.span
            )
        return statistics[column]

    def rescue_for(index: int, column: int):
        if not has_raw:
            return None
        medians, scaled = local(column)
        candidates = reinterpret(
            int(record.raw_words[index, column]),
            record.schema.channels[column],
            float(medians[index]),
            float(scaled[index]),
            tau=consistency_tau,
        )
        return best_rescue(candidates)

    anomalies: list[Anomaly] = []

    for gap in findings.gaps:
        anomalies.append(
            Anomaly(
                observation="gap",
                index=int(gap.before_index),
                channel=-1,
                verdict=AnomalyClass.MISSING,
                confidence=1.00,
                source="rule",
                evidence=(
                    f"timestamp deficit implies {gap.missing_frames_from_time} frames over "
                    f"{gap.duration_s:.6f} s at a declared cadence of "
                    f"{record.schema.cadence_s:.6f} s",
                    (
                        f"counter deficit implies {gap.missing_frames_from_counter} frames"
                        if gap.missing_frames_from_counter >= 0
                        else "no counter delta applies: the bounding frames are not adjacent "
                        "in received order"
                    ),
                ),
                detail={
                    "duration_s": gap.duration_s,
                    "unobserved_s": gap.unobserved_s,
                    "missing_frames_from_time": float(gap.missing_frames_from_time),
                    "missing_frames_from_counter": float(gap.missing_frames_from_counter),
                    "deficits_agree": float(gap.agrees),
                },
            )
        )

    observed_outliers = (
        findings.residuals if mode == "strengthened" else ()
    )
    range_keys = {(finding.index, finding.channel) for finding in findings.ranges}

    for finding in findings.ranges:
        rescue = rescue_for(finding.index, finding.channel)
        validity_ok = (
            bool(record.validity[finding.index]) if has_validity else None
        )
        evidence = [
            f"{finding.channel_name} read {finding.value:.6g}, declared range "
            f"[{finding.low:.6g}, {finding.high:.6g}], excess {finding.excess:.6g}"
        ]
        if validity_ok is not None:
            evidence.append(
                "frame error-control verdict "
                + ("passed" if validity_ok else "failed")
            )
        if rescue is not None:
            evidence.append(
                f"reinterpretation {rescue.alternative!r} of the same word gives "
                f"{rescue.value_alternative:.6g}, inside the declared range and "
                f"{rescue.deviation_sigmas:.3f} robust sigma from the local median"
            )
        if validity_ok is False:
            chosen, confidence, reasons = AnomalyClass.CORRUPT, 0.90, ()
        elif rescue is not None:
            chosen = AnomalyClass.MISPARSED
            confidence = 0.80 if validity_ok is True else 0.60
            reasons = ()
        else:
            chosen, confidence = AnomalyClass.UNDECIDED, 0.00
            reasons = verdict.reasons(("frame_validity", "raw_words"))
            if not reasons:
                reasons = (
                    "the error-control verdict passed and no declared alternative reading of "
                    "the word is both in range and consistent with its neighbours; the frame "
                    "may be corrupt after the error-control check, misparsed in a way not "
                    "modelled here, or a genuine off-nominal reading",
                )
        anomalies.append(
            Anomaly(
                observation="value_out_of_range",
                index=int(finding.index),
                channel=int(finding.channel),
                verdict=chosen,
                confidence=confidence,
                source="rule",
                evidence=tuple(evidence),
                reasons_undecided=reasons,
                detail={
                    "value": finding.value,
                    "excess": finding.excess,
                    "validity_ok": float("nan") if validity_ok is None else float(validity_ok),
                    "rescued": float(rescue is not None),
                },
            )
        )

    for finding in observed_outliers:
        if (finding.index, finding.channel) in range_keys:
            continue
        rescue = rescue_for(finding.index, finding.channel)
        validity_ok = bool(record.validity[finding.index]) if has_validity else None
        evidence = [
            f"{finding.channel_name} read {finding.value:.6g} against a local median of "
            f"{finding.local_median:.6g} and a scaled MAD of {finding.scaled_mad:.6g}: "
            f"{finding.z:.3f} robust sigma, inside the declared range",
        ]
        if validity_ok is not None:
            evidence.append(
                "frame error-control verdict " + ("passed" if validity_ok else "failed")
            )
        if validity_ok is False:
            chosen, confidence, reasons = AnomalyClass.CORRUPT, 0.80, ()
        elif rescue is not None:
            chosen, confidence, reasons = AnomalyClass.MISPARSED, 0.70, ()
            evidence.append(
                f"reinterpretation {rescue.alternative!r} gives {rescue.value_alternative:.6g}, "
                f"{rescue.deviation_sigmas:.3f} robust sigma from the local median"
            )
        else:
            chosen, confidence = AnomalyClass.UNDECIDED, 0.00
            reasons = verdict.reasons(("frame_validity", "raw_words"))
            if not reasons:
                reasons = (
                    "the value is inside its declared range and only its local neighbourhood "
                    "says otherwise; a bit error, a reader fault and a genuine transient are "
                    "not separable from this evidence",
                )
        anomalies.append(
            Anomaly(
                observation="value_local_outlier",
                index=int(finding.index),
                channel=int(finding.channel),
                verdict=chosen,
                confidence=confidence,
                source="rule",
                evidence=tuple(evidence),
                reasons_undecided=reasons,
                detail={
                    "value": finding.value,
                    "z": finding.z,
                    "validity_ok": float("nan") if validity_ok is None else float(validity_ok),
                    "rescued": float(rescue is not None),
                },
            )
        )

    flagged_frames = set(findings.validity_failed_indices)
    for finding in findings.validity_failures:
        anomalies.append(
            Anomaly(
                observation="frame_validity_failed",
                index=int(finding.index),
                channel=-1,
                verdict=AnomalyClass.CORRUPT,
                confidence=0.95,
                source="rule",
                evidence=(
                    f"the error-control field of the frame carrying counter {finding.counter} "
                    "failed, so bits in this frame are damaged",
                ),
                detail={"counter": float(finding.counter)},
            )
        )

    decided_pairs = {
        (anomaly.index, anomaly.channel)
        for anomaly in anomalies
        if anomaly.channel >= 0
    }
    for index in sorted(flagged_frames):
        for column, channel in enumerate(record.schema.channels):
            if (index, column) in decided_pairs:
                continue
            anomalies.append(
                Anomaly(
                    observation="value_in_failed_frame",
                    index=int(index),
                    channel=int(column),
                    verdict=AnomalyClass.UNDECIDED,
                    confidence=0.00,
                    source="rule",
                    evidence=(
                        f"{channel.name} reads {float(record.values[index, column]):.6g}, "
                        "inside its declared range and consistent with its neighbours, in a "
                        "frame whose error-control field failed",
                    ),
                    reasons_undecided=(
                        "the frame's error-control field is one number over the whole frame: "
                        "it proves that bits are damaged and does not say which word carries "
                        "them, so this channel can be neither cleared nor convicted",
                    ),
                    detail={"value": float(record.values[index, column])},
                )
            )

    for index in findings.duplicate_indices:
        identical = _payload_identical(record, index - 1, index)
        validity_ok = bool(record.validity[index]) if has_validity else None
        if identical:
            chosen, confidence, reasons = AnomalyClass.MISPARSED, 0.90, ()
            evidence = (
                f"counter {int(record.counters[index])} repeats the previous frame and the "
                "payload is identical: the frame was received correctly and delivered twice",
            )
        elif validity_ok is False:
            chosen, confidence, reasons = AnomalyClass.CORRUPT, 0.80, ()
            evidence = (
                f"counter {int(record.counters[index])} repeats the previous frame with a "
                "different payload and the error-control verdict failed",
            )
        else:
            chosen, confidence = AnomalyClass.UNDECIDED, 0.00
            reasons = verdict.reasons(("frame_validity",)) or (
                "two frames claim the same counter with different payloads and both pass "
                "error control; the counter field of one of them is wrong, or the declared "
                "counter modulus is, and this record does not say which",
            )
            evidence = (
                f"counter {int(record.counters[index])} repeats the previous frame with a "
                "different payload",
            )
        anomalies.append(
            Anomaly(
                observation="repeated_counter",
                index=int(index),
                channel=-1,
                verdict=chosen,
                confidence=confidence,
                source="rule",
                evidence=evidence,
                reasons_undecided=reasons,
                detail={"payload_identical": float(identical)},
            )
        )

    counter_by_index = {finding.index: finding for finding in findings.counter}
    for index in findings.reorder_indices:
        finding = counter_by_index[index]
        anomalies.append(
            Anomaly(
                observation="reorder",
                index=int(index),
                channel=-1,
                verdict=AnomalyClass.MISPARSED,
                confidence=0.70,
                source="rule",
                evidence=(
                    f"counter moved backward by {finding.frames_implied} within the reorder "
                    f"horizon {record.schema.effective_reorder_horizon}: the bits are intact and "
                    "arrival order is not time order",
                ),
                detail={"frames_backward": float(finding.frames_implied)},
            )
        )

    for finding in findings.clock:
        if finding.kind != "clock_reset":
            continue
        anomalies.append(
            Anomaly(
                observation="clock_reset",
                index=int(finding.index),
                channel=-1,
                verdict=AnomalyClass.MISPARSED,
                confidence=0.70,
                source="rule",
                evidence=(
                    f"timestamp went backward by {-finding.delta_s:.6f} s while the counter "
                    "moved forward: the stream is intact and the time field is wrong",
                ),
                detail={"delta_s": finding.delta_s},
            )
        )

    anomalies.sort(key=lambda anomaly: (anomaly.index, anomaly.channel, anomaly.observation))
    return IntegrityReport(
        determinability=verdict,
        findings=findings,
        anomalies=tuple(anomalies),
        mode=mode,
    )
