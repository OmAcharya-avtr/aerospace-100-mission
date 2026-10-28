"""Per-event-class precision and recall of the rule screens, over a swept grid.

What is measured
----------------
Each of the four timing and ordering screens is scored against the generator's
ground truth, which records every injected gap run, duplicate, reorder and
clock reset by position:

* **gaps** -- a gap is a true positive when its time-domain frame deficit
  equals the length of an injected deletion run that it covers.
* **duplicates, reorders, clock resets** -- scored by frame index.

The grid sweeps anomaly rate against cadence jitter, because both change what
the screens can see: a higher anomaly rate makes anomalies collide, and jitter
eats the margin the cadence screen works in. The grid is swept in full and
every cell is reported, including the cells where a screen does badly.

Clock resets are the deliberate exception and are reported separately. A clock
reset moves part of the record onto a different time base, so after one the
time-sorted view interleaves frames from before and after it. **Gap detection
in the presence of a clock reset is therefore expected to degrade**, and the
grid is swept twice -- with clock resets and without -- so that the size of
that degradation is a measured number rather than a caveat.

The jitter declaration
----------------------
The clean-record section below measures a second thing, and it is a rule a user
has to know. ``RecordSchema.jitter_s`` is a tolerance on the departure of an
**interval** from the cadence. A per-sample timestamp jitter of amplitude ``a``
produces interval departures of up to ``2 a``, because an interval is a
difference of two independently jittered timestamps. So a record whose
timestamps carry jitter ``a`` must declare ``jitter_s >= 2 a`` or its cadence
screen will fire on nothing but jitter. Measured below: at a per-sample jitter
of 0.08 cadences against a declared 0.25 the cadence screen is silent, and at
0.16 and 0.24 it is not.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

from telemgap import dataset, screens  # noqa: E402
from telemgap.report import format_table  # noqa: E402

#: Swept anomaly rate, as a multiplier on the generator's default injection
#: counts, and swept cadence jitter as a fraction of the cadence.
ANOMALY_RATES = (0.5, 1.0, 2.0, 4.0)
JITTER_FRACTIONS = (0.0, 0.08, 0.16, 0.24)

#: Frames per swept cell. Sized so the whole sweep finishes in seconds.
SWEEP_SAMPLES = 3000


def score_events(found: set[int], truth: set[int]) -> tuple[float, float, int, int, int]:
    """Precision, recall and the three counts behind them, for an index set."""
    true_positive = len(found & truth)
    precision = true_positive / len(found) if found else float("nan")
    recall = true_positive / len(truth) if truth else float("nan")
    return precision, recall, true_positive, len(found), len(truth)


def truth_indices(generated: dataset.GeneratedRecord, event: str) -> set[int]:
    """Frame indices carrying one injected frame-level event."""
    return {int(index) for index in np.flatnonzero(generated.frame_truth == event)}


def gap_scores(
    generated: dataset.GeneratedRecord, findings: screens.ScreenFindings
) -> tuple[int, int, int, int]:
    """Gap counts: matched runs, found gaps, injected runs, frame-deficit error.

    An injected deletion run is matched when some reported gap has a
    time-domain frame deficit equal to that run's length. The frame-deficit
    error is the absolute difference between the total frames the gaps imply
    and the total frames actually deleted.
    """
    slots = generated.missing_slots
    runs: list[int] = []
    if slots.size:
        breaks = np.flatnonzero(np.diff(slots) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [slots.size - 1]))
        runs = [int(ends[index] - starts[index] + 1) for index in range(starts.size)]
    reported = sorted(gap.missing_frames_from_time for gap in findings.gaps)
    available = list(reported)
    matched = 0
    for length in sorted(runs):
        if length in available:
            available.remove(length)
            matched += 1
    deficit_error = abs(int(sum(reported)) - int(slots.size))
    return matched, len(reported), len(runs), deficit_error


def main() -> int:
    out = Tee("validate_rule_screens_output.txt")
    out.rule("telemgap -- rule screens, per event class, over a swept grid")
    for line in environment_lines():
        out(line)
    out("")
    out(f"frames per cell        : {SWEEP_SAMPLES}")
    out(f"anomaly rate multipliers: {list(ANOMALY_RATES)}")
    out(f"cadence jitter fractions: {list(JITTER_FRACTIONS)}")
    out("")
    out("Equations (13) and (14) of telemgap.benchmark, applied to frame indices.")

    timer = Timer()
    for with_resets in (False, True):
        label = "with clock resets" if with_resets else "no clock resets"
        out.rule(f"swept grid -- {label}")
        rows = []
        worst = {"duplicate": 1.0, "reorder": 1.0, "clock_reset": 1.0}
        gap_recalls = []
        for rate in ANOMALY_RATES:
            for jitter in JITTER_FRACTIONS:
                generated = dataset.generate(
                    samples=SWEEP_SAMPLES,
                    seed=int(100_000 + 1000 * rate + 100 * jitter + with_resets),
                    n_gaps=max(1, int(round(6 * rate))),
                    n_duplicates=max(1, int(round(8 * rate))),
                    n_reorders=max(1, int(round(8 * rate))),
                    n_clock_resets=(2 if with_resets else 0),
                    n_corruptions=max(1, int(round(40 * rate))),
                    n_byteswap=max(1, int(round(10 * rate))),
                    n_scale=max(1, int(round(10 * rate))),
                    jitter_fraction=jitter,
                )
                with timer:
                    findings = screens.screen(generated.record)
                matched, reported, injected, deficit = gap_scores(generated, findings)
                gap_recall = matched / injected if injected else float("nan")
                gap_recalls.append(gap_recall)
                cells = [f"{rate:g}", f"{jitter:g}", f"{matched}/{injected}", f"{reported}"]
                cells.append(f"{gap_recall:.4f}")
                cells.append(str(deficit))
                for event, found in (
                    ("duplicate", set(findings.duplicate_indices)),
                    ("reorder", set(findings.reorder_indices)),
                    ("clock_reset", set(findings.clock_reset_indices)),
                ):
                    precision, recall, _, _, support = score_events(
                        found, truth_indices(generated, event)
                    )
                    cells.extend([f"{precision:.4f}", f"{recall:.4f}", str(support)])
                    if np.isfinite(recall):
                        worst[event] = min(worst[event], recall)
                rows.append(cells)
        header = [
            "rate",
            "jitter",
            "gaps matched",
            "gaps found",
            "gap recall",
            "deficit err",
            "dup P",
            "dup R",
            "dup n",
            "reord P",
            "reord R",
            "reord n",
            "reset P",
            "reset R",
            "reset n",
        ]
        out(format_table(header, rows))
        out("")
        finite = [value for value in gap_recalls if np.isfinite(value)]
        out(f"gap recall over the grid : min {min(finite):.4f}  mean {np.mean(finite):.4f}")
        for event, value in worst.items():
            out(f"worst {event:<12s} recall: {value:.4f}")
        if not with_resets:
            out.check(
                "every injected duplicate is found on every cell of a record with no clock reset",
                worst["duplicate"] >= 1.0 - 1e-12,
                f"worst recall {worst['duplicate']:.4f}",
            )
            out.check(
                "every injected reorder is found on every cell of a record with no clock reset",
                worst["reorder"] >= 1.0 - 1e-12,
                f"worst recall {worst['reorder']:.4f}",
            )
            out.check(
                "every injected gap run is matched on every cell of a record with no clock reset",
                min(finite) >= 1.0 - 1e-12,
                f"min gap recall {min(finite):.4f}",
            )
            baseline_gap = float(np.mean(finite))
        else:
            out.check(
                "every injected clock reset is found",
                worst["clock_reset"] >= 1.0 - 1e-12,
                f"worst recall {worst['clock_reset']:.4f}",
            )
            out(f"mean gap recall degradation from the clock resets: {baseline_gap:.4f} -> "
                f"{np.mean(finite):.4f}")

    out.rule("false positives on a record with nothing injected")
    clean_rows = []
    total_false = 0
    cadence_false = 0
    for jitter in JITTER_FRACTIONS:
        generated = dataset.generate(
            samples=SWEEP_SAMPLES,
            seed=int(900_000 + 100 * jitter),
            n_gaps=0,
            n_duplicates=0,
            n_reorders=0,
            n_clock_resets=0,
            n_corruptions=0,
            n_byteswap=0,
            n_scale=0,
            jitter_fraction=jitter,
        )
        findings = screens.screen(generated.record)
        summary = findings.summary()
        false_positives = (
            summary["gaps"]
            + summary["duplicates"]
            + summary["reorders"]
            + summary["clock_resets"]
        )
        total_false += false_positives
        if 2.0 * jitter <= dataset.JITTER_S / dataset.CADENCE_S:
            cadence_false += summary["cadence_violations"]
        clean_rows.append(
            [
                f"{jitter:g}",
                str(summary["gaps"]),
                str(summary["duplicates"]),
                str(summary["reorders"]),
                str(summary["clock_resets"]),
                str(summary["cadence_violations"]),
                str(summary["range_violations"]),
                str(summary["residual_violations"]),
            ]
        )
    out(
        format_table(
            [
                "jitter",
                "gaps",
                "dups",
                "reorders",
                "resets",
                "cadence",
                "range",
                "residual",
            ],
            clean_rows,
        )
    )
    out("")
    out(
        "range and residual findings on a record with nothing injected are NOT false "
        "positives: the valve_current channel carries a standing signedness fault by "
        "construction, which is a real anomaly the screens are right to report. The "
        "timing screens are the ones that must be silent."
    )
    out.check(
        "no gap, duplicate, reorder or clock-reset false positive on any clean cell",
        total_false == 0,
        f"{total_false} across {len(JITTER_FRACTIONS)} cells",
    )
    out(
        "The cadence screen fires on jitter alone once the per-sample jitter exceeds half "
        f"the declared jitter_s of {dataset.JITTER_S:g} s, which is the 2a rule in this "
        "script's docstring. Cells satisfying 2a <= jitter_s must be silent."
    )
    out.check(
        "the cadence screen is silent on every clean cell satisfying 2a <= jitter_s",
        cadence_false == 0,
        f"{cadence_false} cadence findings on compliant cells",
    )

    out.rule("screening cost -- UNSTABLE, two cores shared with other work")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["records screened", str(timer.calls)],
                ["frames per record", str(SWEEP_SAMPLES)],
                ["mean screen time (ms)", f"{timer.mean_ms:.1f}"],
                ["total screen time (s)", f"{timer.elapsed_s:.2f}"],
            ],
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
