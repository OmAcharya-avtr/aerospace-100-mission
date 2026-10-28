"""What cannot be determined from this record. Computed first, returned first.

This module exists because of a requirement, and the requirement is worth
restating: an integrity verdict that does not say what it could not decide is
worse than no verdict, because it reads as evidence. Every limit below is
**computed from the record and schema actually supplied**, carries a number
where a number exists, and is attached to the observations it affects. None of
it is a prose caveat.

The seven limits
----------------
``frame_validity``
    Whether the record carries a per-frame error-control verdict. Without it,
    *corrupt* and *misparsed* cannot be separated for any frame, because both
    present as a value outside its declared range and the only field that
    distinguishes them is the one that says whether the bits survived the link.

``raw_words``
    Whether the raw words of equation (1) were retained. Without them the
    reinterpretation test of :mod:`telemgap.reparse` cannot run, so *misparsed*
    has no positive evidence available and can never be concluded.

``counter_horizon``
    A forward counter delta above ``modulus - reorder_horizon`` is read as
    backward motion. A true forward gap that long is therefore reported as a
    reorder, and no choice of horizon removes the ambiguity -- it relocates it.
    The limit reports the shortest gap, in frames and in seconds, that this
    record cannot represent.

``in_range_corruption``
    A bit error that leaves the decoded value inside its declared range is
    invisible to a range screen. The limit reports, **per channel and exactly
    by enumeration** for a 16-bit word, the fraction of single-bit flips of an
    in-range word that produce another in-range word. That fraction is an upper
    bound on what the declared-range screen must miss, and it is a property of
    the channel's calibration, not of the data.

``cadence_declaration``
    Whether the declared cadence agrees with the record's own modal inter-frame
    interval. If it does not, every frame deficit computed from the cadence is
    wrong by a factor, and no screen can detect that from inside the record.

``time_source``
    Whether the record declares what its timestamps are. A ground receipt time
    and a spacecraft clock disagree about whether a gap is a loss of signal or
    an onboard timing event, and nothing in the array distinguishes them.

``gap_interior``
    Structural, and always undeterminable: a gap contains no information. Any
    value placed inside it is an assumption, and the error bound on that
    assumption is only as good as the declared curvature bound it is computed
    from. The limit reports how many channels declared a non-zero curvature
    bound and how many left it at zero, which is where the policy silently
    becomes a straight line through a region it knows nothing about.

Returned by :func:`assess`, carried as the **first** field of
:class:`telemgap.classify.IntegrityReport`, and printed as the first section of
every report this package produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .record import ChannelSpec, TelemetryRecord

#: Keys of the seven limits, in the order they are reported.
LIMIT_KEYS = (
    "frame_validity",
    "raw_words",
    "counter_horizon",
    "in_range_corruption",
    "cadence_declaration",
    "time_source",
    "gap_interior",
)

#: Word widths small enough to enumerate exactly in the in-range corruption
#: limit. Wider words fall back to a seeded Monte Carlo estimate, and the
#: limit's ``method`` field says which was used.
EXACT_ENUMERATION_BITS = 16

#: Sample count for the Monte Carlo fallback of the in-range corruption limit.
MONTE_CARLO_SAMPLES = 200_000


@dataclass(frozen=True)
class Limit:
    """One determinability limit, with the number that quantifies it.

    Attributes
    ----------
    key
        One of :data:`LIMIT_KEYS`.
    determinable
        ``True`` when the record supplies what is needed to decide the question
        this limit is about. ``gap_interior`` is always ``False``: it is
        structural.
    statement
        What cannot be determined, in one sentence.
    consequence
        What the classification does about it.
    quantity
        The number attached to the limit, or ``float('nan')`` when the limit is
        purely structural. Units are given by :attr:`unit`.
    unit
        Unit of :attr:`quantity`, or ``""``.
    method
        How :attr:`quantity` was obtained: ``"declared"``, ``"enumerated"``,
        ``"monte_carlo"``, ``"measured"`` or ``"structural"``.
    """

    key: str
    determinable: bool
    statement: str
    consequence: str
    quantity: float = float("nan")
    unit: str = ""
    method: str = "declared"


@dataclass
class DeterminabilityVerdict:
    """The seven limits, with helpers for the classifier and the report."""

    limits: tuple[Limit, ...]

    def __getitem__(self, key: str) -> Limit:
        for limit in self.limits:
            if limit.key == key:
                return limit
        raise KeyError(f"no determinability limit named {key!r}; expected one of {LIMIT_KEYS}")

    @property
    def undeterminable_keys(self) -> tuple[str, ...]:
        """Keys whose question this record cannot answer."""
        return tuple(limit.key for limit in self.limits if not limit.determinable)

    @property
    def can_separate_corrupt_from_misparsed(self) -> bool:
        """Whether both the validity flag and the raw words are available."""
        return bool(self["frame_validity"].determinable and self["raw_words"].determinable)

    def reasons(self, keys: tuple[str, ...]) -> tuple[str, ...]:
        """Statements of the named limits that are undeterminable, for an anomaly record."""
        return tuple(
            self[key].statement for key in keys if key in LIMIT_KEYS and not self[key].determinable
        )


def _in_range_mask(channel: ChannelSpec, words: np.ndarray) -> np.ndarray:
    """Which of ``words`` decode inside ``channel``'s declared range, equation (1)."""
    span = 1 << channel.word_bits
    raw = words.astype(np.int64)
    if channel.signed:
        half = span >> 1
        raw = np.where(raw >= half, raw - span, raw)
    values = raw.astype(np.float64) * channel.scale + channel.offset
    return (values >= channel.low) & (values <= channel.high)


def in_range_single_bit_flip_fraction(
    channel: ChannelSpec, seed: int = 0
) -> tuple[float, str, float]:
    """Fraction of single-bit flips of an in-range word that stay in range.

    This is the quantity the declared-range screen is structurally unable to
    catch. It depends only on the channel's calibration and declared range.

    Returns
    -------
    (fraction, method, in_range_code_fraction)
        ``fraction`` is the undetectable share of single-bit flips, in
        ``[0, 1]``. ``method`` is ``"enumerated"`` for a 16-bit word, where
        every ``(word, bit)`` pair is evaluated, or ``"monte_carlo"`` for a
        wider word. ``in_range_code_fraction`` is the share of all word codes
        that decode inside the declared range, reported beside it because it is
        the chance-rescue probability the reinterpretation test must overcome.
    """
    bits = channel.word_bits
    if bits <= EXACT_ENUMERATION_BITS:
        words = np.arange(1 << bits, dtype=np.int64)
        inside = _in_range_mask(channel, words)
        in_range_codes = float(inside.mean())
        plausible = words[inside]
        if plausible.size == 0:
            return 0.0, "enumerated", in_range_codes
        total = 0
        stays = 0
        for bit in range(bits):
            flipped = plausible ^ np.int64(1 << bit)
            stays += int(_in_range_mask(channel, flipped).sum())
            total += int(plausible.size)
        return float(stays / total), "enumerated", in_range_codes
    rng = np.random.default_rng(seed)
    span = 1 << bits
    sample = rng.integers(0, span, size=MONTE_CARLO_SAMPLES, dtype=np.int64)
    inside = _in_range_mask(channel, sample)
    in_range_codes = float(inside.mean())
    plausible = sample[inside]
    if plausible.size == 0:
        return 0.0, "monte_carlo", in_range_codes
    bit_choice = rng.integers(0, bits, size=plausible.size)
    flipped = plausible ^ (np.int64(1) << bit_choice.astype(np.int64))
    return float(_in_range_mask(channel, flipped).mean()), "monte_carlo", in_range_codes


def modal_interval(timestamps: np.ndarray, cadence_s: float) -> float:
    """Modal positive inter-frame interval, seconds, on a cadence-relative grid.

    The mode is taken over a histogram with bin width ``cadence_s / 20``, which
    is finer than any jitter a schema may declare (the schema requires
    ``jitter_s < cadence_s / 2``) and coarse enough that floating-point noise
    does not split the mode. Returns ``nan`` for a record with no positive
    interval.
    """
    deltas = np.diff(np.sort(timestamps))
    deltas = deltas[deltas > 0.0]
    if deltas.size == 0:
        return float("nan")
    width = cadence_s / 20.0
    bins = np.floor(deltas / width).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    chosen = unique[int(np.argmax(counts))]
    selected = deltas[bins == chosen]
    return float(np.median(selected))


def assess(record: TelemetryRecord, seed: int = 0) -> DeterminabilityVerdict:
    """Compute all seven limits for ``record``.

    Parameters
    ----------
    record
        The record whose determinability is being assessed.
    seed
        Seed for the Monte Carlo fallback of the in-range corruption limit.
        Unused for 16-bit channels, which are enumerated exactly.
    """
    if not isinstance(record, TelemetryRecord):
        raise TypeError(f"assess expects a TelemetryRecord, got {type(record).__name__}")
    schema = record.schema
    limits: list[Limit] = []

    has_validity = bool(schema.has_validity_flag and record.validity is not None)
    limits.append(
        Limit(
            key="frame_validity",
            determinable=has_validity,
            statement=(
                "no per-frame error-control verdict is present, so a damaged frame cannot be "
                "separated from an intact frame the reader decoded wrongly"
                if not has_validity
                else "a per-frame error-control verdict is present for every frame"
            ),
            consequence=(
                "every out-of-range value without a rescuing reinterpretation is reported "
                "undecided, never corrupt"
                if not has_validity
                else "an out-of-range value with a failed error-control verdict is reported corrupt"
            ),
            quantity=float(0 if not has_validity else record.validity.size),
            unit="frames with a validity verdict",
            method="declared",
        )
    )

    has_raw = bool(schema.has_raw_words and record.raw_words is not None)
    limits.append(
        Limit(
            key="raw_words",
            determinable=has_raw,
            statement=(
                "the raw words of equation (1) were not retained, so no alternative reading of "
                "the same bits can be tested and a misparse has no positive evidence available"
                if not has_raw
                else "the raw words of equation (1) are retained, so the reinterpretation test "
                "of telemgap.reparse can run"
            ),
            consequence=(
                "the misparsed class is unreachable for this record and its members appear as "
                "undecided"
                if not has_raw
                else "an out-of-range value whose word is rescued by a declared alternative "
                "reading is reported misparsed"
            ),
            quantity=float(0 if not has_raw else record.raw_words.shape[1]),
            unit="channels with raw words",
            method="declared",
        )
    )

    horizon = schema.effective_reorder_horizon
    shortest_unrepresentable = int(schema.counter_modulus - horizon)
    limits.append(
        Limit(
            key="counter_horizon",
            determinable=False,
            statement=(
                f"a forward gap of {shortest_unrepresentable} frames or more produces a counter "
                f"delta above the reorder horizon {horizon} and is read as a reorder, not a gap; "
                "the ambiguity is irreducible and no horizon removes it"
            ),
            consequence=(
                "gap lengths are reported from the timestamp deficit as well as the counter "
                "deficit, and a disagreement between the two is surfaced as evidence about the "
                "clock"
            ),
            quantity=float(shortest_unrepresentable * schema.cadence_s),
            unit="s (shortest gap not representable in the counter domain)",
            method="declared",
        )
    )

    fractions = []
    code_fractions = []
    methods = set()
    for channel in schema.channels:
        fraction, method, codes = in_range_single_bit_flip_fraction(channel, seed=seed)
        fractions.append(fraction)
        code_fractions.append(codes)
        methods.add(method)
    worst = float(max(fractions)) if fractions else 0.0
    limits.append(
        Limit(
            key="in_range_corruption",
            determinable=False,
            statement=(
                f"a single-bit error stays inside the declared range for {worst:.6f} of the "
                "in-range words of the worst channel, and a range screen cannot see it; the "
                f"share of all word codes that decode in range is at most {max(code_fractions):.6f}"
            ),
            consequence=(
                "the declared-range screen's recall against bit errors is bounded above by "
                "1 minus this fraction; the strengthened local-residual screen of equation (5) "
                "is reported beside it because it is not bounded by the same quantity"
            ),
            quantity=worst,
            unit="fraction of single-bit flips that remain in range",
            method=sorted(methods)[0] if methods else "structural",
        )
    )

    measured = modal_interval(record.timestamps, schema.cadence_s)
    agrees = bool(np.isfinite(measured) and abs(measured - schema.cadence_s) <= schema.jitter_s)
    limits.append(
        Limit(
            key="cadence_declaration",
            determinable=agrees,
            statement=(
                f"the record's modal inter-frame interval is {measured:.6f} s but the schema "
                f"declares {schema.cadence_s:.6f} s, a disagreement beyond the declared jitter "
                f"{schema.jitter_s:.6f} s; every frame deficit computed from the cadence is "
                "therefore wrong by a factor and no screen inside the record can detect it"
                if not agrees
                else f"the record's modal inter-frame interval {measured:.6f} s agrees with the "
                f"declared cadence {schema.cadence_s:.6f} s to within the declared jitter"
            ),
            consequence=(
                "frame deficits and the interpolation policy are not trustworthy until the "
                "cadence declaration is corrected"
                if not agrees
                else "frame deficits are computed from the declared cadence"
            ),
            quantity=float(measured - schema.cadence_s) if np.isfinite(measured) else float("nan"),
            unit="s (modal interval minus declared cadence)",
            method="measured",
        )
    )

    time_source = str(record.provenance.get("time_source", "")).strip()
    limits.append(
        Limit(
            key="time_source",
            determinable=bool(time_source),
            statement=(
                "the record does not declare what its timestamps are, so a gap cannot be "
                "attributed to a loss of signal rather than an onboard timing event"
                if not time_source
                else f"timestamps are declared as {time_source!r}"
            ),
            consequence=(
                "gap causes are not reported; only gap extents are"
                if not time_source
                else "gap extents are reported against the declared time source"
            ),
            method="declared",
        )
    )

    declared_curvature = sum(1 for channel in schema.channels if channel.max_curvature > 0.0)
    limits.append(
        Limit(
            key="gap_interior",
            determinable=False,
            statement=(
                "a gap contains no information; any value placed inside it is an assumption, and "
                "its error bound is only as good as the declared curvature bound it is computed "
                f"from ({declared_curvature} of {len(schema.channels)} channels declared a "
                "non-zero curvature bound)"
            ),
            consequence=(
                "the interpolation policy refuses rather than interpolates whenever the declared "
                "bound exceeds the declared tolerance, and reports which bound it used"
            ),
            quantity=float(declared_curvature),
            unit=f"channels of {len(schema.channels)} with a declared curvature bound",
            method="structural",
        )
    )

    return DeterminabilityVerdict(limits=tuple(limits))
