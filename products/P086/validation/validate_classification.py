"""The three-way classification, and what withholding a field costs.

Three measurements, in order.

**1. Per-class precision, recall and surfaced rate** of the specified rule
baseline and of the strengthened one, on the held-out split, against the
generator's value-level ground truth. Equations (13) to (15b) of
:mod:`telemgap.benchmark`.

**1b. The ceiling on any declared-range screen**, enumerated over every word
code of every channel: the share of single-bit flips of an in-range word that
produce another in-range word, which no range screen can see.

**2. The chance-rescue rate of the reinterpretation test.** Condition (7a) of
:mod:`telemgap.reparse` -- an alternative reading that lands inside the
declared range -- is not on its own evidence of a misparse, because a damaged
word can land in range by accident. The rate at which it does so is measured
here by enumerating the whole 16-bit word space of each channel, and the rate
at which condition (7b) additionally holds is measured beside it. The ratio of
the two is what condition (7b) is worth.

**3. What each withheld field costs.** The same record is classified four
times: complete, without the frame validity flag, without the raw words, and
without both. The movement of the per-class recall and of the undecided count
is the measured value of each field, and it is exactly what the
``frame_validity`` and ``raw_words`` determinability limits assert in prose.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from telemgap import benchmark, classify, dataset, determinability  # noqa: E402
from telemgap.learned import CLASS_NAMES  # noqa: E402
from telemgap.reparse import reinterpret_array  # noqa: E402
from telemgap.report import format_table, render_scores  # noqa: E402

#: Frames in the record every number below is measured on.
SAMPLES = 12_000


def main() -> int:
    out = Tee("validate_classification_output.txt")
    out.rule("telemgap -- three-way integrity classification on the held-out split")
    for line in environment_lines():
        out(line)
    out("")
    out(f"frames            : {SAMPLES}")
    out(f"split seed        : {dataset.TEST_SEED} (test); training seed {dataset.TRAINING_SEED}")
    out("classes           : " + ", ".join(CLASS_NAMES))

    generated = dataset.test_record(samples=SAMPLES)
    record = generated.record
    truth = generated.value_truth.reshape(-1)
    out("")
    out(
        format_table(
            ["quantity", "value"],
            [[key, str(value)] for key, value in generated.truth_counts().items()],
        )
    )

    out.rule("1. per-class scores of the two rule baselines")
    tables = {}
    for mode in classify.MODES:
        report = classify.classify(record, mode=mode)
        predicted = benchmark.rule_labels(record, report).reshape(-1)
        table = benchmark.score(truth, predicted, f"rules/{mode}")
        tables[mode] = table
        out(render_scores(table))
        out("")
    out(
        "The strengthened baseline adds only the local-residual screen of equation (5). "
        "Its effect on the corrupt class is the measured value of that screen, and the "
        "reason the specified baseline is not the one the learned component is scored "
        "against."
    )
    gain = tables["strengthened"]["corrupt"].recall - tables["specified"]["corrupt"].recall
    out(f"corrupt recall gained by the local-residual screen: {gain:+.4f}")
    out.check(
        "the strengthened baseline has at least the corrupt recall of the specified one",
        gain >= 0.0,
        f"{tables['specified']['corrupt'].recall:.4f} -> "
        f"{tables['strengthened']['corrupt'].recall:.4f}",
    )
    out.check(
        "neither baseline exceeds a corrupt recall of 1 minus the in-range flip fraction "
        "plus the residual screen's reach",
        tables["specified"]["corrupt"].recall
        <= 1.0 - 0.5 * determinability.assess(record)["in_range_corruption"].quantity,
        "the specified baseline cannot see an in-range bit error, which is the "
        "in_range_corruption determinability limit",
    )

    out.rule("1b. the ceiling on any declared-range screen, enumerated")
    rows = []
    fractions = []
    for channel in record.schema.channels:
        fraction, method, codes = determinability.in_range_single_bit_flip_fraction(channel)
        fractions.append(fraction)
        rows.append(
            [
                channel.name,
                str(1 << channel.word_bits),
                method,
                f"{codes:.6f}",
                f"{fraction:.6f}",
                f"{1.0 - fraction:.6f}",
            ]
        )
    out(
        format_table(
            [
                "channel",
                "word codes",
                "method",
                "codes decoding in range",
                "single-bit flips staying in range",
                "max achievable range-screen recall",
            ],
            rows,
        )
    )
    # From the raw floats, never from the formatted strings in the table: a
    # six-decimal rendering is not the number.
    worst = max(fractions)
    out("")
    out(
        f"Worst channel: {worst:.6f} of single-bit flips of an in-range word produce "
        "another in-range word. That is an upper bound on what a declared-range screen "
        "must miss, it is a property of the channel's calibration rather than of the "
        "data, and it is what the in_range_corruption determinability limit reports on "
        "every call."
    )
    out.check(
        "the enumerated in-range flip fraction agrees with the determinability limit",
        abs(worst - determinability.assess(record)["in_range_corruption"].quantity) < 1e-12,
        f"{worst:.9f}",
    )

    out.rule("2. what condition (7b) of the reinterpretation test is worth")
    rows = []
    for channel in record.schema.channels:
        words = np.arange(1 << channel.word_bits, dtype=np.uint64)
        median = np.full(words.size, channel.mid, dtype=np.float64)
        scaled = np.full(words.size, 0.02 * channel.span, dtype=np.float64)
        bulk = reinterpret_array(words, channel, median, scaled)
        in_range_only = float(bulk["in_range"].any(axis=1).mean())
        both = float(bulk["any_rescue"].mean())
        rows.append(
            [
                channel.name,
                str(words.size),
                f"{in_range_only:.6f}",
                f"{both:.6f}",
                f"{in_range_only / both:.2f}" if both > 0.0 else "inf",
            ]
        )
        out.check(
            f"condition (7b) tightens the rescue rate on {channel.name}",
            both <= in_range_only,
            f"(7a) alone {in_range_only:.6f}, (7a) and (7b) {both:.6f}",
        )
    out(
        format_table(
            [
                "channel",
                "word codes",
                "(7a) alone",
                "(7a) and (7b)",
                "ratio",
            ],
            rows,
        )
    )
    out("")
    out(
        "Enumerated over every word code, with the local median placed at the channel's "
        "range midpoint and the robust sigma at 2 per cent of its span. A chance rescue "
        "under condition (7a) alone is common; requiring consistency with the "
        "neighbourhood is what makes a rescue evidence."
    )

    out.rule("3. the measured cost of withholding a field")
    variants = {
        "complete": record,
        "no validity flag": record.without_validity(),
        "no raw words": record.without_raw_words(),
        "neither": record.without_validity().without_raw_words(),
    }
    rows = []
    for label, variant in variants.items():
        report = classify.classify(variant)
        predicted = benchmark.rule_labels(variant, report).reshape(-1)
        table = benchmark.score(truth, predicted, label)
        verdict = report.determinability
        rows.append(
            [
                label,
                "yes" if verdict["frame_validity"].determinable else "NO",
                "yes" if verdict["raw_words"].determinable else "NO",
                str(report.counts["corrupt"]),
                str(report.counts["misparsed"]),
                str(report.counts["undecided"]),
                f"{table['corrupt'].recall:.4f}",
                f"{table['misparsed'].recall:.4f}",
                f"{table['corrupt'].surfaced_rate:.4f}",
                f"{table['misparsed'].surfaced_rate:.4f}",
            ]
        )
    out(
        format_table(
            [
                "record",
                "validity",
                "raw words",
                "corrupt verdicts",
                "misparsed verdicts",
                "undecided verdicts",
                "corrupt R",
                "misparsed R",
                "corrupt surfaced",
                "misparsed surfaced",
            ],
            rows,
        )
    )
    complete = classify.classify(record)
    no_raw = classify.classify(record.without_raw_words())
    no_validity = classify.classify(record.without_validity())
    out("")
    out.check(
        "withholding the raw words empties the misparsed class at the value level",
        all(
            anomaly.verdict is not classify.AnomalyClass.MISPARSED
            for anomaly in no_raw.anomalies
            if anomaly.observation in benchmark.VALUE_OBSERVATIONS
        ),
        "the raw_words determinability limit asserts this; it is measured here",
    )
    out.check(
        "withholding the raw words raises the undecided count",
        no_raw.counts["undecided"] > complete.counts["undecided"],
        f"{complete.counts['undecided']} -> {no_raw.counts['undecided']}",
    )
    out.check(
        "withholding the validity flag lowers the corrupt verdict count",
        no_validity.counts["corrupt"] < complete.counts["corrupt"],
        f"{complete.counts['corrupt']} -> {no_validity.counts['corrupt']}",
    )
    out.check(
        "every undecided verdict carries at least one stated reason",
        all(anomaly.reasons_undecided for anomaly in complete.undecided),
        f"{len(complete.undecided)} undecided verdicts",
    )
    out.check(
        "no decided verdict carries an undecided reason",
        all(
            not anomaly.reasons_undecided
            for anomaly in complete.anomalies
            if not anomaly.is_undecided
        ),
    )

    out.rule("the ambiguous residual set the learned component is scored on")
    predicted = benchmark.rule_labels(record, complete).reshape(-1)
    ambiguous = predicted == benchmark.UNDECIDED
    counts = np.bincount(truth[ambiguous], minlength=len(CLASS_NAMES))
    out(
        format_table(
            ["quantity", "value"],
            [
                ["rows in the record", str(truth.size)],
                ["rows the rules left undecided", str(int(ambiguous.sum()))],
                ["share of all rows", f"{float(ambiguous.mean()):.6f}"],
                *[
                    [f"true {name} among them", str(int(counts[index]))]
                    for index, name in enumerate(CLASS_NAMES)
                ],
                [
                    "majority class share",
                    f"{float(counts.max() / max(counts.sum(), 1)):.6f}",
                ],
            ],
        )
    )
    out.check(
        "the ambiguous residual set is non-empty and contains more than one class",
        int(ambiguous.sum()) > 0 and int((counts > 0).sum()) > 1,
        f"{int(ambiguous.sum())} rows, {int((counts > 0).sum())} classes present",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
