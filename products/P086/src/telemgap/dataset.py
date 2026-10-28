"""Seeded synthetic telemetry with injected anomalies of known type and magnitude.

No flight data is used anywhere in this package, and no number in it is taken
from a real spacecraft. Everything is generated here, from closed forms printed
in this docstring, so that every validation figure can be regenerated from a
seed and so that the *true* class of every anomaly is known exactly rather than
adjudicated.

The four channels
-----------------
Each channel is a sum of sinusoids plus an optional linear ramp,

    v(t) = c + r t + sum_k A_k sin(2 pi f_k t + phi_k)                  (11)

whose exact curvature and slew bounds follow by term-by-term differentiation,

    |v''| <= sum_k A_k (2 pi f_k)^2,   |v'| <= |r| + sum_k A_k (2 pi f_k)  (12)

Equation (12) is the *analytic* bound. It is used by validation as the
reference the policy's estimate of equation (9) is scored against, and it is
declared on two of the four channels so that both the declared and the
estimated curvature paths of :mod:`telemgap.policy` are exercised by every run.

``battery_voltage`` (V, unsigned, curvature declared)
    Two slow terms. The well-behaved case.
``valve_current`` (A, unsigned, curvature not declared)
    Mean 0.030 A with a measurement noise of 0.012 A, so roughly six samples in
    a thousand have a negative true value. The channel is **declared unsigned
    and its field is signed**, which is a standing decommutation fault, not an
    injected event: every sample whose true value is negative is stored as a
    near-full-scale word and read back at about 0.64 A. This is the signedness
    misparse, and it is present because this is what that fault looks like in a
    real archive -- a sparse scatter of impossible readings on an otherwise
    quiet channel.
``gyro_rate`` (deg/s, signed, curvature not declared)
    A slow term plus a term at 0.35 of the sampling frequency, below Nyquist
    and well above the scale a second difference can see. For a single
    sinusoid, equation (9) returns ``4 A sin^2(pi f T) / T^2`` against a true
    ``A (2 pi f)^2``, a ratio of ``(sin x / x)^2`` with ``x = pi f T``; at
    ``f T = 0.35`` that is 0.648. **The estimate therefore under-predicts the
    curvature of this channel by construction, by a factor derivable in closed
    form**, and the measured consequence for the policy is reported in
    ``validation/validate_interpolation_policy.py`` rather than asserted here.
``tank_pressure`` (bar, unsigned, curvature declared)
    A slow ramp with a small oscillation. The channel the byteswap misparse is
    injected on.

Injected anomalies
------------------
Every injection is recorded in the returned ground truth, by frame and by
channel, and every one is constructed so that the stored record stays
**self-consistent**: ``values == ChannelSpec.decode(raw_words)`` holds for
every frame and channel of every generated record, including the corrupted and
misparsed ones. That consistency matters. An inconsistent record would let any
classifier separate the classes by re-decoding the words, which no real archive
permits, and the measured scores would be meaningless.

``corruption``
    One to three bits flipped in the stored word. A declared fraction
    (``flagged_probability``, default 0.6) also has its frame error-control
    verdict set to failed; the remainder keep a passing verdict, modelling
    corruption introduced in the ground processing chain *after* the frame
    error-control check. A real link-layer CRC catches essentially every bit
    error, so the silent fraction is the interesting one and is the reason the
    undecided class is not empty. Whether a corruption lands inside or outside
    the declared range is **not** chosen: it falls out of the bit flip, and the
    split is measured.
``misparse_byteswap``
    The word is stored with its bytes reversed, as an archive written on a
    little-endian host from a big-endian downlink would hold it.
``misparse_scale``
    The word is stored for a calibration scale a decade smaller than the one
    declared, the classic unit error in a hand-maintained decommutation
    database.
``misparse_signedness``
    Not injected. See ``valve_current`` above.
``missing``
    Runs of whole frames deleted. Run lengths are drawn from a declared list
    that includes one long outage, so every record contains at least one gap
    the policy must refuse.
``duplicate``
    A frame re-inserted immediately after itself, byte-identical.
``reorder``
    A frame moved later in the received order by a few positions, with its
    timestamp and counter unchanged.
``clock_reset``
    A constant subtracted from every subsequent timestamp, with the counter
    sequence untouched.

Compute budget
--------------
A 12,000-frame four-channel record generates in well under a second on the two
cores this package was measured on; the generation time is recorded in the
validation output tables, which are declared unstable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .record import ChannelSpec, RecordSchema, TelemetryRecord

#: Nominal cadence of every generated record, seconds.
CADENCE_S = 1.0

#: Declared jitter tolerance of every generated record, seconds.
JITTER_S = 0.25 * CADENCE_S

#: Counter modulus of every generated record. CCSDS space-packet width.
COUNTER_MODULUS = 16384

#: Gap run lengths drawn from, in frames. The 600 is a deliberate long outage:
#: every record contains at least one gap the policy must refuse.
GAP_RUN_LENGTHS = (1, 1, 1, 2, 2, 3, 5, 8, 20, 600)

#: Ground-truth value-level classes.
VALUE_CLASSES = ("clean", "corrupt", "misparsed")

#: Ground-truth frame-level event names.
FRAME_EVENTS = ("duplicate", "reorder", "clock_reset")


@dataclass(frozen=True)
class SignalModel:
    """Closed form of equation (11) for one channel."""

    name: str
    constant: float
    ramp: float
    terms: tuple[tuple[float, float, float], ...]
    noise_sigma: float

    def evaluate(self, times: np.ndarray) -> np.ndarray:
        """Noiseless truth of equation (11) at ``times``, channel units."""
        times = np.asarray(times, dtype=np.float64)
        out = np.full(times.shape, self.constant, dtype=np.float64) + self.ramp * times
        for amplitude, frequency, phase in self.terms:
            out += amplitude * np.sin(2.0 * np.pi * frequency * times + phase)
        return out

    def second_derivative(self, times: np.ndarray) -> np.ndarray:
        """Exact second derivative of equation (11), channel units per second squared."""
        times = np.asarray(times, dtype=np.float64)
        out = np.zeros(times.shape, dtype=np.float64)
        for amplitude, frequency, phase in self.terms:
            omega = 2.0 * np.pi * frequency
            out -= amplitude * omega**2 * np.sin(omega * times + phase)
        return out

    @property
    def curvature_bound(self) -> float:
        """Analytic ``sum A_k (2 pi f_k)^2`` of equation (12), per second squared."""
        return float(
            sum(
                amplitude * (2.0 * np.pi * frequency) ** 2
                for amplitude, frequency, _ in self.terms
            )
        )

    @property
    def slew_bound(self) -> float:
        """Analytic ``|r| + sum A_k 2 pi f_k`` of equation (12), per second."""
        return float(
            abs(self.ramp)
            + sum(amplitude * 2.0 * np.pi * frequency for amplitude, frequency, _ in self.terms)
        )

    @property
    def bandwidth_hz(self) -> float:
        """Highest frequency present, hertz. Used by the Bernstein cross-check."""
        return float(max((frequency for _, frequency, _ in self.terms), default=0.0))

    @property
    def amplitude_bound(self) -> float:
        """Bound on ``|v - c - r t|``, channel units. Used by the Bernstein cross-check."""
        return float(sum(amplitude for amplitude, _, _ in self.terms))


#: The four channel models. Frequencies are in hertz against ``CADENCE_S``.
MODELS: tuple[SignalModel, ...] = (
    SignalModel(
        name="battery_voltage",
        constant=28.5,
        ramp=0.0,
        terms=((1.2, 1.0 / 600.0, 0.0), (0.35, 1.0 / 137.0, 1.1)),
        noise_sigma=0.004,
    ),
    SignalModel(
        name="valve_current",
        constant=0.030,
        ramp=0.0,
        terms=((0.004, 1.0 / 400.0, 0.3),),
        noise_sigma=0.012,
    ),
    SignalModel(
        name="gyro_rate",
        constant=0.0,
        ramp=0.0,
        terms=((0.40, 1.0 / 240.0, 0.0), (0.08, 0.35, 0.7)),
        noise_sigma=0.003,
    ),
    SignalModel(
        name="tank_pressure",
        constant=13.0,
        ramp=-2.0e-4,
        terms=((0.05, 1.0 / 1800.0, 0.6),),
        noise_sigma=0.002,
    ),
)

#: Which channels declare a curvature bound. The other two force the policy
#: down the estimated path of equation (9), which is where its weakness is.
DECLARES_CURVATURE = {"battery_voltage": True, "tank_pressure": True}


def channel_specs(declare_curvature: bool = True) -> tuple[ChannelSpec, ...]:
    """The four :class:`~telemgap.record.ChannelSpec` of the generated records.

    Parameters
    ----------
    declare_curvature
        When ``False``, no channel declares a curvature bound, so every channel
        takes the estimated path of equation (9). Used by validation to measure
        the estimate against the analytic bound on all four channels.
    """
    return (
        ChannelSpec(
            name="battery_voltage",
            unit="V",
            scale=2.0e-4,
            offset=24.0,
            low=26.0,
            high=31.0,
            max_slew=MODELS[0].slew_bound,
            noise_sigma=MODELS[0].noise_sigma,
            max_curvature=(
                MODELS[0].curvature_bound
                if declare_curvature and DECLARES_CURVATURE.get("battery_voltage")
                else 0.0
            ),
            word_bits=16,
            signed=False,
        ),
        ChannelSpec(
            name="valve_current",
            unit="A",
            scale=1.0e-5,
            offset=0.0,
            low=-0.08,
            high=0.08,
            max_slew=MODELS[1].slew_bound,
            noise_sigma=MODELS[1].noise_sigma,
            max_curvature=0.0,
            word_bits=16,
            signed=False,
        ),
        ChannelSpec(
            name="gyro_rate",
            unit="deg/s",
            scale=5.0e-4,
            offset=0.0,
            low=-3.0,
            high=3.0,
            max_slew=MODELS[2].slew_bound,
            noise_sigma=MODELS[2].noise_sigma,
            max_curvature=0.0,
            word_bits=16,
            signed=True,
        ),
        ChannelSpec(
            name="tank_pressure",
            unit="bar",
            scale=5.0e-4,
            offset=0.0,
            low=8.0,
            high=16.0,
            max_slew=MODELS[3].slew_bound,
            noise_sigma=MODELS[3].noise_sigma,
            max_curvature=(
                MODELS[3].curvature_bound
                if declare_curvature and DECLARES_CURVATURE.get("tank_pressure")
                else 0.0
            ),
            word_bits=16,
            signed=False,
        ),
    )


def schema(
    declare_curvature: bool = True,
    has_validity_flag: bool = True,
    has_raw_words: bool = True,
) -> RecordSchema:
    """The :class:`~telemgap.record.RecordSchema` of the generated records."""
    return RecordSchema(
        cadence_s=CADENCE_S,
        jitter_s=JITTER_S,
        counter_modulus=COUNTER_MODULUS,
        channels=channel_specs(declare_curvature=declare_curvature),
        has_validity_flag=has_validity_flag,
        has_raw_words=has_raw_words,
        reorder_horizon=None,
    )


def _encode(values: np.ndarray, channel: ChannelSpec) -> np.ndarray:
    """Inverse of equation (1) with round-to-nearest, wrapped into the word width."""
    raw = np.rint((np.asarray(values, dtype=np.float64) - channel.offset) / channel.scale)
    return np.mod(raw.astype(np.int64), np.int64(1 << channel.word_bits)).astype(np.uint64)


@dataclass
class GeneratedRecord:
    """A generated record with its complete ground truth.

    Attributes
    ----------
    record
        The :class:`~telemgap.record.TelemetryRecord` as a user would receive it.
    value_truth
        ``(n, n_channels)`` of :data:`VALUE_CLASSES` indices: 0 clean,
        1 corrupt, 2 misparsed.
    misparse_kind
        ``(n, n_channels)`` of strings naming the injected misparse, empty
        where none.
    frame_truth
        ``(n,)`` of strings: empty, or one of :data:`FRAME_EVENTS`.
    truth_times, truth_values
        The full noiseless truth on the generation grid, ``(m,)`` and
        ``(m, n_channels)``. Used to measure reconstruction error inside gaps.
    missing_slots
        Grid slot indices of the frames deleted to make the gaps.
    corruption_in_range
        ``(n, n_channels)`` bool, true where an injected corruption left the
        value inside its declared range -- the class the declared-range screen
        is structurally unable to see. Measured, not chosen.
    """

    record: TelemetryRecord
    value_truth: np.ndarray
    misparse_kind: np.ndarray
    frame_truth: np.ndarray
    truth_times: np.ndarray
    truth_values: np.ndarray
    missing_slots: np.ndarray
    corruption_in_range: np.ndarray
    parameters: dict[str, float] = field(default_factory=dict)

    def truth_counts(self) -> dict[str, int]:
        """Ground-truth population of every class and event."""
        counts = {
            name: int((self.value_truth == index).sum())
            for index, name in enumerate(VALUE_CLASSES)
        }
        counts["corrupt_in_range"] = int(self.corruption_in_range.sum())
        counts["corrupt_out_of_range"] = counts["corrupt"] - counts["corrupt_in_range"]
        for event in FRAME_EVENTS:
            counts[event] = int((self.frame_truth == event).sum())
        counts["missing_frames"] = int(self.missing_slots.size)
        counts["frames"] = len(self.record)
        return counts


def generate(
    samples: int = 12_000,
    seed: int = 20261028,
    n_gaps: int = 10,
    n_duplicates: int = 12,
    n_reorders: int = 12,
    n_clock_resets: int = 2,
    n_corruptions: int = 160,
    n_byteswap: int = 40,
    n_scale: int = 40,
    flagged_probability: float = 0.6,
    jitter_fraction: float = 0.08,
    declare_curvature: bool = True,
    has_validity_flag: bool = True,
    has_raw_words: bool = True,
) -> GeneratedRecord:
    """Generate one record with ground truth. Deterministic in ``seed``.

    Parameters
    ----------
    samples
        Frames on the generation grid, before deletions and insertions.
    seed
        Seed of ``numpy.random.default_rng``. Two records with different seeds
        are independent realisations and are used as disjoint train and test
        splits.
    n_gaps, n_duplicates, n_reorders, n_clock_resets
        Counts of frame-level injections.
    n_corruptions, n_byteswap, n_scale
        Counts of value-level injections, each a ``(frame, channel)`` pair.
    flagged_probability
        Share of corruptions whose frame error-control verdict is set to
        failed. The remainder are silent; see the module docstring.
    jitter_fraction
        Timestamp jitter amplitude as a fraction of the cadence, uniform in
        ``+/- jitter_fraction * CADENCE_S``. Must stay below 0.25 so that a
        pair of adjacent frames cannot differ by half a cadence.
    declare_curvature, has_validity_flag, has_raw_words
        Passed through to :func:`schema`. Withholding the last two is how the
        determinability limits are exercised.

    Returns
    -------
    GeneratedRecord
    """
    if samples < 64:
        raise ValueError(f"samples must be at least 64, got {samples!r}")
    if not 0.0 <= jitter_fraction < 0.25:
        raise ValueError(f"jitter_fraction must lie in [0, 0.25), got {jitter_fraction!r}")
    if not 0.0 <= flagged_probability <= 1.0:
        raise ValueError(
            f"flagged_probability must lie in [0, 1], got {flagged_probability!r}"
        )
    rng = np.random.default_rng(seed)
    record_schema = schema(
        declare_curvature=declare_curvature,
        has_validity_flag=has_validity_flag,
        has_raw_words=has_raw_words,
    )
    channels = record_schema.channels
    n_channels = len(channels)

    grid = CADENCE_S * np.arange(samples, dtype=np.float64)
    truth = np.column_stack([model.evaluate(grid) for model in MODELS])
    noisy = truth + rng.normal(
        0.0, [model.noise_sigma for model in MODELS], size=(samples, n_channels)
    )

    raw = np.empty((samples, n_channels), dtype=np.uint64)
    for column, channel in enumerate(channels):
        raw[:, column] = _encode(noisy[:, column], channel)

    value_truth = np.zeros((samples, n_channels), dtype=np.int8)
    misparse_kind = np.full((samples, n_channels), "", dtype=object)
    corruption_in_range = np.zeros((samples, n_channels), dtype=bool)
    validity = np.ones(samples, dtype=bool)

    # The standing signedness fault on valve_current: every sample whose true
    # value is negative is stored as a near-full-scale unsigned word, because
    # the field is signed and the schema says it is not.
    valve = record_schema.channel_index("valve_current")
    negative = noisy[:, valve] < 0.0
    value_truth[negative, valve] = VALUE_CLASSES.index("misparsed")
    misparse_kind[negative, valve] = "signedness"

    def free_pairs(count: int, columns: tuple[int, ...]) -> list[tuple[int, int]]:
        """Pick ``count`` distinct clean ``(frame, channel)`` pairs from ``columns``."""
        chosen: list[tuple[int, int]] = []
        attempts = 0
        while len(chosen) < count and attempts < 40 * max(count, 1):
            attempts += 1
            frame = int(rng.integers(16, samples - 16))
            column = int(columns[int(rng.integers(0, len(columns)))])
            if value_truth[frame, column] != 0:
                continue
            chosen.append((frame, column))
            value_truth[frame, column] = -1
        for frame, column in chosen:
            value_truth[frame, column] = 0
        return chosen

    byteswap_column = record_schema.channel_index("tank_pressure")
    for frame, column in free_pairs(n_byteswap, (byteswap_column,)):
        channel = channels[column]
        word = int(raw[frame, column])
        swapped = ((word & 0xFF) << 8) | ((word >> 8) & 0xFF)
        raw[frame, column] = np.uint64(swapped)
        value_truth[frame, column] = VALUE_CLASSES.index("misparsed")
        misparse_kind[frame, column] = "byteswap"
        del channel

    scale_column = record_schema.channel_index("gyro_rate")
    for frame, column in free_pairs(n_scale, (scale_column,)):
        channel = channels[column]
        word = int(
            np.mod(
                int(np.rint((noisy[frame, column] - channel.offset) / (channel.scale / 10.0))),
                1 << channel.word_bits,
            )
        )
        raw[frame, column] = np.uint64(word)
        value_truth[frame, column] = VALUE_CLASSES.index("misparsed")
        misparse_kind[frame, column] = "scale_decade_down"

    for frame, column in free_pairs(n_corruptions, tuple(range(n_channels))):
        channel = channels[column]
        n_bits = int(rng.integers(1, 4))
        bits = rng.choice(channel.word_bits, size=n_bits, replace=False)
        word = int(raw[frame, column])
        for bit in bits:
            word ^= 1 << int(bit)
        raw[frame, column] = np.uint64(word)
        value_truth[frame, column] = VALUE_CLASSES.index("corrupt")
        if rng.random() < flagged_probability:
            validity[frame] = False

    values = np.empty((samples, n_channels), dtype=np.float64)
    for column, channel in enumerate(channels):
        values[:, column] = channel.decode(
            np.where(
                channel.signed,
                np.where(
                    raw[:, column].astype(np.int64) >= (1 << (channel.word_bits - 1)),
                    raw[:, column].astype(np.int64) - (1 << channel.word_bits),
                    raw[:, column].astype(np.int64),
                ),
                raw[:, column].astype(np.int64),
            )
        )
    for column, channel in enumerate(channels):
        corrupted = value_truth[:, column] == VALUE_CLASSES.index("corrupt")
        corruption_in_range[:, column] = corrupted & channel.in_range(values[:, column])

    counters = np.mod(np.arange(samples, dtype=np.int64), COUNTER_MODULUS)
    timestamps = grid + rng.uniform(
        -jitter_fraction * CADENCE_S, jitter_fraction * CADENCE_S, size=samples
    )
    frame_truth = np.full(samples, "", dtype=object)
    # Frames any frame-level injection has already touched, including the
    # unlabelled partner of a duplicate pair and the whole span a reorder
    # permutes. A later injection refuses to touch a reserved frame. Without
    # this, a reorder whose window ends on the source of a duplicate pair moves
    # a third frame between the two copies, the copies are no longer adjacent,
    # and the counter screen cannot see the duplicate at all -- measured during
    # the build as a duplicate recall of 0.9688 that was an artefact of the
    # injection order and not of the screen.
    reserved = np.zeros(samples, dtype=bool)

    keep = np.ones(samples, dtype=bool)
    missing_slots: list[int] = []
    for _ in range(n_gaps):
        length = int(GAP_RUN_LENGTHS[int(rng.integers(0, len(GAP_RUN_LENGTHS)))])
        start = int(rng.integers(32, max(33, samples - length - 32)))
        window = np.arange(start, min(start + length, samples - 32))
        if window.size == 0 or not keep[window].all():
            continue
        keep[window] = False
        missing_slots.extend(int(slot) for slot in window)

    order = np.flatnonzero(keep)
    timestamps = timestamps[order]
    counters = counters[order]
    values = values[order]
    raw = raw[order]
    validity = validity[order]
    value_truth = value_truth[order]
    misparse_kind = misparse_kind[order]
    corruption_in_range = corruption_in_range[order]
    frame_truth = frame_truth[order]
    reserved = reserved[order]

    # Every frame-level injection below refuses a position that already
    # carries a label. Without that guard a reorder can land on a frame already
    # marked as a duplicate and overwrite the label, and the ground truth then
    # under-counts one event class and over-counts another. That was measured
    # on this generator during the build as a duplicate recall of 0.8750 and a
    # clock-reset precision of 0.5000 that were artefacts of the labelling and
    # not of the screens.
    for _ in range(n_clock_resets):
        if timestamps.size < 64:
            break
        placed = False
        for _attempt in range(64):
            position = int(rng.integers(32, timestamps.size - 32))
            if reserved[position]:
                continue
            shift = float(rng.uniform(5.0, 60.0) * CADENCE_S)
            timestamps[position:] -= shift
            frame_truth[position] = "clock_reset"
            reserved[position] = True
            placed = True
            break
        if not placed:
            break

    def insert_at(position: int, source: int) -> None:
        nonlocal timestamps, counters, values, raw, validity, reserved
        nonlocal value_truth, misparse_kind, corruption_in_range, frame_truth
        timestamps = np.insert(timestamps, position, timestamps[source])
        counters = np.insert(counters, position, counters[source])
        values = np.insert(values, position, values[source], axis=0)
        raw = np.insert(raw, position, raw[source], axis=0)
        validity = np.insert(validity, position, validity[source])
        value_truth = np.insert(value_truth, position, value_truth[source], axis=0)
        misparse_kind = np.insert(misparse_kind, position, misparse_kind[source], axis=0)
        corruption_in_range = np.insert(
            corruption_in_range, position, corruption_in_range[source], axis=0
        )
        frame_truth = np.insert(frame_truth, position, "duplicate")
        reserved = np.insert(reserved, position, True)
        reserved[source] = True

    for _ in range(n_duplicates):
        if timestamps.size < 64:
            break
        for _attempt in range(64):
            source = int(rng.integers(32, timestamps.size - 32))
            if reserved[source] or reserved[source + 1]:
                continue
            insert_at(source + 1, source)
            break

    for _ in range(n_reorders):
        if timestamps.size < 64:
            break
        position = -1
        target = -1
        for _attempt in range(64):
            candidate = int(rng.integers(32, timestamps.size - 40))
            offset = int(rng.integers(2, 6))
            end = min(candidate + offset, timestamps.size - 1)
            window = slice(max(0, candidate - 1), min(reserved.size, end + 2))
            if np.any(reserved[window]):
                continue
            position, target = candidate, end
            break
        if position < 0:
            break
        permutation = np.arange(timestamps.size)
        permutation = np.concatenate(
            (
                permutation[:position],
                permutation[position + 1 : target + 1],
                permutation[position : position + 1],
                permutation[target + 1 :],
            )
        )
        timestamps = timestamps[permutation]
        counters = counters[permutation]
        values = values[permutation]
        raw = raw[permutation]
        validity = validity[permutation]
        value_truth = value_truth[permutation]
        misparse_kind = misparse_kind[permutation]
        corruption_in_range = corruption_in_range[permutation]
        frame_truth = frame_truth[permutation]
        reserved = reserved[permutation]
        frame_truth[target] = "reorder"
        reserved[max(0, position - 1) : min(reserved.size, target + 2)] = True

    built = TelemetryRecord(
        schema=record_schema,
        timestamps=timestamps,
        counters=counters,
        values=values,
        raw_words=raw if has_raw_words else None,
        validity=validity if has_validity_flag else None,
        provenance={
            "generator": "telemgap.dataset.generate",
            "seed": int(seed),
            "time_source": "generated spacecraft elapsed time, seconds from an arbitrary epoch",
        },
    )
    return GeneratedRecord(
        record=built,
        value_truth=value_truth,
        misparse_kind=misparse_kind,
        frame_truth=frame_truth,
        truth_times=grid,
        truth_values=truth,
        missing_slots=np.asarray(sorted(missing_slots), dtype=np.int64),
        corruption_in_range=corruption_in_range,
        parameters={
            "samples": float(samples),
            "seed": float(seed),
            "jitter_fraction": float(jitter_fraction),
            "flagged_probability": float(flagged_probability),
            "n_corruptions": float(n_corruptions),
            "n_byteswap": float(n_byteswap),
            "n_scale": float(n_scale),
            "n_gaps": float(n_gaps),
        },
    )


#: Seeds of the disjoint splits. Fixed, so that every number in this repository
#: is reproducible, and distinct, so that no frame of the training record
#: appears in the test record.
TRAINING_SEED = 20261028
TEST_SEED = 71120261
CALIBRATION_SEED = 44520261


def training_record(**kwargs: object) -> GeneratedRecord:
    """The training split. Seed :data:`TRAINING_SEED`."""
    kwargs.setdefault("seed", TRAINING_SEED)
    return generate(**kwargs)  # type: ignore[arg-type]


def test_record(**kwargs: object) -> GeneratedRecord:
    """The held-out test split. Seed :data:`TEST_SEED`."""
    kwargs.setdefault("seed", TEST_SEED)
    return generate(**kwargs)  # type: ignore[arg-type]


def calibration_record(**kwargs: object) -> GeneratedRecord:
    """A third independent split, used only for probability calibration."""
    kwargs.setdefault("seed", CALIBRATION_SEED)
    return generate(**kwargs)  # type: ignore[arg-type]


def long_outage_record(
    samples: int = 600,
    outage_frames: int = 21_600,
    seed: int = 913,
) -> GeneratedRecord:
    """A constructed record with one outage the policy must refuse.

    Half the frames, then a gap of ``outage_frames`` cadence intervals -- six
    hours at the default one-second cadence -- then the other half. No other
    anomaly is injected, so the refusal cannot be attributed to anything else.
    """
    if outage_frames < 2:
        raise ValueError(f"outage_frames must be at least 2, got {outage_frames!r}")
    generated = generate(
        samples=samples,
        seed=seed,
        n_gaps=0,
        n_duplicates=0,
        n_reorders=0,
        n_clock_resets=0,
        n_corruptions=0,
        n_byteswap=0,
        n_scale=0,
        jitter_fraction=0.0,
    )
    record = generated.record
    half = len(record) // 2
    timestamps = record.timestamps.copy()
    timestamps[half:] += outage_frames * CADENCE_S
    counters = record.counters.copy()
    counters[half:] = np.mod(counters[half:] + outage_frames, COUNTER_MODULUS)
    shifted = TelemetryRecord(
        schema=record.schema,
        timestamps=timestamps,
        counters=counters,
        values=record.values.copy(),
        raw_words=None if record.raw_words is None else record.raw_words.copy(),
        validity=None if record.validity is None else record.validity.copy(),
        provenance={
            "generator": "telemgap.dataset.long_outage_record",
            "seed": int(seed),
            "outage_frames": int(outage_frames),
            "time_source": "generated spacecraft elapsed time, seconds from an arbitrary epoch",
        },
    )
    truth_times = generated.truth_times.copy()
    truth_times[half:] += outage_frames * CADENCE_S
    generated.record = shifted
    generated.truth_times = truth_times
    generated.parameters["outage_frames"] = float(outage_frames)
    return generated
