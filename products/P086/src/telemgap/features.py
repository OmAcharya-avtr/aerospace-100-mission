"""Frame-and-channel level features for the learned classifier.

One row per ``(frame, channel)`` pair. Fifteen features, listed in
:data:`FEATURE_NAMES` and documented one per line below.

Two design decisions matter for the honesty of the benchmark and are stated
here rather than buried.

**The channel identity is deliberately excluded.** The synthetic generator
injects the byteswap misparse on one named channel and the scale misparse on
another, so a feature carrying the channel index would let the model recover
the generator's construction rather than the physics. Any score obtained that
way would be an artefact of the dataset. There is no channel one-hot, no
channel index, and no channel-specific scaling constant in the feature set;
every feature is normalised by a declared property of its own channel so that
the rows are comparable across channels without naming them.

**The model is given everything the rules use.** Features 8 to 10 and 13 are
the outputs of the reinterpretation test of :mod:`telemgap.reparse`, which is
the rules' discriminator between corrupt and misparsed. The learned model is
therefore not handicapped: if it loses to the rules, it loses with the rules'
own evidence in hand, and the loss cannot be attributed to withheld
information. That is the point of the comparison.

The features
------------
1.  ``range_excess_norm`` -- signed distance outside the declared range, as a
    fraction of the range span. Negative inside.
2.  ``inside_range`` -- 1 inside the declared range, 0 outside.
3.  ``z_robust`` -- equation (5), capped at :data:`Z_CAP`.
4.  ``diff_prev_norm`` -- ``|v_i - v_{i-1}|`` over the declared one-cadence
    slew allowance.
5.  ``diff_next_norm`` -- the same, forward.
6.  ``validity_ok`` -- 1 passed, 0 failed, 0.5 when no validity field exists.
7.  ``frame_out_of_range_fraction`` -- share of the frame's channels outside
    their declared range. A bit error in one word affects one channel; a
    structurally mangled frame affects many.
8.  ``any_rescue`` -- 1 if some declared alternative reading satisfies both
    (7a) and (7b).
9.  ``best_rescue_deviation`` -- robust sigmas from the local median of the
    best rescuing alternative, capped at :data:`Z_CAP`; :data:`Z_CAP` when
    nothing rescues.
10. ``rescue_fraction`` -- share of the five alternatives that rescue.
11. ``hamming_word_norm`` -- population count of ``word XOR local_median_word``
    over the word width. A one-to-three-bit error gives a small value; a
    byteswap of a word with asymmetric bytes gives a large one.
12. ``hamming_low_byte_norm`` -- the same over the low byte only, which is
    where a quantisation-level difference lives and a byteswap does not.
13. ``byteswap_closer`` -- 1 if the byte-reversed word lands closer to the
    local median than the word as read.
14. ``interval_residual_norm`` -- ``|dt - T| / T`` for the interval ending at
    this frame, capped at :data:`INTERVAL_CAP`.
15. ``counter_delta_norm`` -- forward counter delta of equation (2), capped at
    :data:`COUNTER_CAP` and divided by it.

Units: every feature is dimensionless by construction.
"""

from __future__ import annotations

import numpy as np

from .record import TelemetryRecord
from .reparse import (
    ALTERNATIVES,
    DEFAULT_CONSISTENCY_TAU,
    byte_reverse_array,
    reinterpret_array,
)
from .screens import counter_delta, robust_local_statistics

#: Cap on the robust z-score features. A corrupt word can sit thousands of
#: robust sigmas out, and an uncapped feature of that range dominates a tree
#: split threshold for no information gain.
Z_CAP = 50.0

#: Cap on the interval residual feature, in cadences.
INTERVAL_CAP = 8.0

#: Cap on the counter delta feature, in frames.
COUNTER_CAP = 8.0

#: Feature names in column order.
FEATURE_NAMES = (
    "range_excess_norm",
    "inside_range",
    "z_robust",
    "diff_prev_norm",
    "diff_next_norm",
    "validity_ok",
    "frame_out_of_range_fraction",
    "any_rescue",
    "best_rescue_deviation",
    "rescue_fraction",
    "hamming_word_norm",
    "hamming_low_byte_norm",
    "byteswap_closer",
    "interval_residual_norm",
    "counter_delta_norm",
)


def _interval_residual(record: TelemetryRecord) -> np.ndarray:
    """``|dt - T| / T`` for the interval ending at each frame, first frame zero."""
    cadence = float(record.schema.cadence_s)
    deltas = np.diff(record.timestamps, prepend=record.timestamps[0] - cadence)
    return np.minimum(np.abs(deltas - cadence) / cadence, INTERVAL_CAP) / INTERVAL_CAP


def _counter_deltas(record: TelemetryRecord) -> np.ndarray:
    """Forward counter delta of equation (2) for each frame, first frame one."""
    modulus = int(record.schema.counter_modulus)
    counters = record.counters
    deltas = np.empty(counters.size, dtype=np.float64)
    deltas[0] = 1.0
    for index in range(1, counters.size):
        deltas[index] = counter_delta(int(counters[index - 1]), int(counters[index]), modulus)
    return np.minimum(deltas, COUNTER_CAP) / COUNTER_CAP


def build(
    record: TelemetryRecord,
    window: int = 8,
    tau: float = DEFAULT_CONSISTENCY_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and index pairs for every ``(frame, channel)`` of ``record``.

    Parameters
    ----------
    record
        The record. ``raw_words`` are required: without them the
        reinterpretation features cannot be computed, and the learned model is
        not defined for a record whose misparsed class is unreachable. The
        determinability verdict says so for such a record, and this function
        raises rather than silently producing a degraded feature set.
    window
        Half-width of the local window of equation (5).
    tau
        Consistency tolerance of (7b).

    Returns
    -------
    (features, pairs)
        ``features`` is ``(n * n_channels, len(FEATURE_NAMES))`` float64 in
        row-major ``(frame, channel)`` order. ``pairs`` is
        ``(n * n_channels, 2)`` int64 of ``(frame_index, channel_index)``.
    """
    if record.raw_words is None:
        raise ValueError(
            "feature extraction requires raw_words; without them the reinterpretation "
            "features of telemgap.reparse cannot be computed and the misparsed class is "
            "unreachable. See telemgap.determinability limit 'raw_words'."
        )
    n = len(record)
    n_channels = record.n_channels
    cadence = float(record.schema.cadence_s)
    interval = _interval_residual(record)
    counters = _counter_deltas(record)
    if record.validity is None:
        validity = np.full(n, 0.5, dtype=np.float64)
    else:
        validity = record.validity.astype(np.float64)

    outside = np.zeros((n, n_channels), dtype=bool)
    for column, channel in enumerate(record.schema.channels):
        outside[:, column] = ~channel.in_range(record.values[:, column])
    frame_fraction = outside.mean(axis=1)

    columns: list[np.ndarray] = []
    for column, channel in enumerate(record.schema.channels):
        values = record.values[:, column]
        words = record.raw_words[:, column]
        medians, scaled = robust_local_statistics(values, window, channel.span)
        excess = np.maximum(channel.low - values, values - channel.high) / channel.span
        z = np.minimum(np.abs(values - medians) / scaled, Z_CAP) / Z_CAP
        slew_allowance = channel.max_slew * cadence + 1e-6 * channel.span
        diff_prev = np.abs(np.diff(values, prepend=values[0])) / slew_allowance
        diff_next = np.abs(np.diff(values, append=values[-1])) / slew_allowance
        rescue = reinterpret_array(words, channel, medians, scaled, tau=tau)
        word_medians, _ = robust_local_statistics(
            words.astype(np.float64), window, float(1 << channel.word_bits)
        )
        median_word = np.rint(word_medians).astype(np.int64)
        median_word = np.clip(median_word, 0, (1 << channel.word_bits) - 1)
        word_int = words.astype(np.int64)
        hamming = np.bitwise_count(np.bitwise_xor(word_int, median_word)).astype(np.float64)
        low_byte = np.bitwise_count(
            np.bitwise_xor(word_int & 0xFF, median_word & 0xFF)
        ).astype(np.float64)
        swapped = byte_reverse_array(words, channel.word_bits)
        swapped_value = (
            np.where(
                channel.signed,
                np.where(
                    swapped >= (1 << (channel.word_bits - 1)),
                    swapped - (1 << channel.word_bits),
                    swapped,
                ),
                swapped,
            ).astype(np.float64)
            * channel.scale
            + channel.offset
        )
        closer = (np.abs(swapped_value - medians) < np.abs(values - medians)).astype(np.float64)
        columns.append(
            np.column_stack(
                (
                    np.clip(excess, -1.0, 1.0),
                    (~outside[:, column]).astype(np.float64),
                    z,
                    np.minimum(diff_prev, 10.0) / 10.0,
                    np.minimum(diff_next, 10.0) / 10.0,
                    validity,
                    frame_fraction,
                    rescue["any_rescue"].astype(np.float64),
                    np.minimum(rescue["best_deviation"], Z_CAP) / Z_CAP,
                    rescue["rescues"].sum(axis=1).astype(np.float64) / len(ALTERNATIVES),
                    hamming / channel.word_bits,
                    low_byte / 8.0,
                    closer,
                    interval,
                    counters,
                )
            )
        )

    stacked = np.stack(columns, axis=1)
    features = stacked.reshape(n * n_channels, len(FEATURE_NAMES))
    frames = np.repeat(np.arange(n, dtype=np.int64), n_channels)
    channels = np.tile(np.arange(n_channels, dtype=np.int64), n)
    pairs = np.column_stack((frames, channels))
    if not np.all(np.isfinite(features)):
        raise ValueError(
            "feature extraction produced a non-finite value; a non-finite channel value "
            "reached a normalisation it cannot survive"
        )
    return features, pairs


def truth_labels(value_truth: np.ndarray) -> np.ndarray:
    """Flatten a ``(n, n_channels)`` ground-truth class array to feature-row order."""
    return np.asarray(value_truth, dtype=np.int64).reshape(-1)
