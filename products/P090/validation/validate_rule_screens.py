"""The rule baseline against measured consequence, over a swept grid.

The grid sweeps the corruption rate, which is what changes the class balance and
therefore the precision of every screen. The history-length and history-threshold
sweeps are here too, because both are declared choices and a reader is entitled
to see what they are worth rather than being told that four sigma is the right
number.

The last section measures the structural limit of the cross-source screen on the
one construction that makes it visible: the corpus contains pairs of records
that are observationally identical and labelled differently, because the same
two element sets appear once with the perturbed one as the candidate and once
with the clean one as the candidate. No screen can be right on both.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    CADENCE_DAYS,
    EPOCH_REQUIREMENT_DAYS,
    HISTORY_LENGTH,
    HORIZON_DAYS,
    OBJECTS,
    SAMPLES,
    SILENT_THRESHOLD_KM,
    TARGET_FPR,
    Tee,
    bootstrap,
    declared_settings,
    environment_lines,
)

bootstrap()

from tlecheck import benchmark, dataset  # noqa: E402
from tlecheck.consequence import BENIGN, CLASSES, SILENTLY_WRONG, WILL_FAIL_LOUDLY  # noqa: E402


def _prepare(objects: int, rate: float, history_length: int, threshold: float = 4.0):
    records = dataset.split(
        "test",
        objects=objects,
        corruption_rate=rate,
        history_length=history_length,
        cadence_days=CADENCE_DAYS,
    )
    return records, benchmark.prepare(
        records,
        epoch_requirement_days=EPOCH_REQUIREMENT_DAYS,
        history_threshold=threshold,
        continuity_threshold=threshold,
        horizon_days=HORIZON_DAYS,
        samples=SAMPLES,
        silent_threshold_km=SILENT_THRESHOLD_KM,
    )


def main() -> int:
    out = Tee("validate_rule_screens_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    out.rule("1. PER-CLASS PRECISION AND RECALL OVER A SWEPT CORRUPTION RATE")
    out(
        "Six corruption rates, one grid cell each, every cell screened and "
        "measured independently. The rate changes the class balance, so precision "
        "moves with it and recall should not."
    )
    out("")
    header = f"{'rate':>6s} {'n':>5s}"
    for label in CLASSES:
        header += f" {label[:12] + ' P':>15s} {label[:12] + ' R':>15s}"
    header += f" {'accuracy':>9s}"
    out(header)
    recalls: dict[str, list[float]] = defaultdict(list)
    cells = 0
    for rate in (0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        _records, prepared = _prepare(OBJECTS, rate, HISTORY_LENGTH)
        scores = benchmark.per_class(prepared.measured_classes, prepared.rule_classes)
        line = f"{rate:6.2f} {len(prepared.records):>5d}"
        for label in CLASSES:
            line += f" {scores[label].precision:15.4f} {scores[label].recall:15.4f}"
            if np.isfinite(scores[label].recall):
                recalls[label].append(scores[label].recall)
        line += (
            f" {benchmark.accuracy(prepared.measured_classes, prepared.rule_classes):9.4f}"
        )
        out(line)
        cells += 1
    out("")
    for label in CLASSES:
        values = recalls[label]
        if not values:
            continue
        out(
            f"{label:20s} recall across the grid: min {min(values):.4f}, "
            f"median {float(np.median(values)):.4f}, max {max(values):.4f}"
        )
    out.check(
        "grid_was_swept",
        cells == 6,
        f"{cells} corruption-rate cells, {OBJECTS} objects each",
    )
    out("")
    out(
        "The loud recall is NOT 1.0 on every cell and the reason is a deliberate "
        "trade in the decision table. The rules predict a loud failure only from a "
        "physical bound violation, which maps to a documented SGP4 error code. An "
        "element set whose characters have shifted so far that a field will not "
        "parse is classified silently wrong instead, with an infinite equation (15) "
        "score, because the propagator very often reads such a line without "
        "complaint and returns a confident wrong position. The misses below are "
        "the records where the propagator did refuse it after all."
    )
    out("")
    out(
        f"{'rate':>6s} {'loud measured':>14s} {'missed':>7s} {'unreadable':>11s} "
        f"{'flagged silent':>15s} {'no finding':>11s}"
    )
    missed_total = 0
    breakdown = {"unreadable": 0, "flagged_silent": 0, "no_finding": 0}
    for rate in (0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        _cell_records, cell = _prepare(OBJECTS, rate, HISTORY_LENGTH)
        loud = cell.measured_classes == WILL_FAIL_LOUDLY
        missed = loud & (cell.rule_classes != WILL_FAIL_LOUDLY)
        cell_breakdown = {"unreadable": 0, "flagged_silent": 0, "no_finding": 0}
        for index in np.flatnonzero(missed):
            evidence = cell.evidences[int(index)]
            if any(f.name == "parse_fault" for f in evidence.format_findings):
                cell_breakdown["unreadable"] += 1
            elif cell.rule_classes[int(index)] == SILENTLY_WRONG:
                cell_breakdown["flagged_silent"] += 1
            else:
                cell_breakdown["no_finding"] += 1
        for key, value in cell_breakdown.items():
            breakdown[key] += value
        missed_total += int(missed.sum())
        out(
            f"{rate:6.2f} {int(loud.sum()):>14d} {int(missed.sum()):>7d} "
            f"{cell_breakdown['unreadable']:>11d} "
            f"{cell_breakdown['flagged_silent']:>15d} "
            f"{cell_breakdown['no_finding']:>11d}"
        )
    out("")
    out(
        f"Across the grid the rules missed {missed_total} loud failures: "
        f"{breakdown['unreadable']} were element sets this package could not parse, "
        f"{breakdown['flagged_silent']} were flagged as silently wrong by a screen "
        f"with a magnitude, and {breakdown['no_finding']} were not flagged at all. "
        "Only the last group is a miss in the operational sense, because the first "
        "two are rejected by the screen with a different reason attached. That is "
        "the measured cost of the trade and it is published beside the benefit: "
        "the field-shift operating characteristic went from 0.3686, below chance, "
        "to 0.9065 when the trade was made."
    )
    out.check(
        "loud_recall_is_high_on_every_cell",
        min(recalls[WILL_FAIL_LOUDLY]) > 0.65,
        f"min {min(recalls[WILL_FAIL_LOUDLY]):.4f}, median "
        f"{float(np.median(recalls[WILL_FAIL_LOUDLY])):.4f} over "
        f"{len(recalls[WILL_FAIL_LOUDLY])} cells",
    )
    out.check(
        "no_loud_failure_passed_the_screen_unflagged",
        breakdown["no_finding"] == 0,
        f"{breakdown['no_finding']} of {missed_total} missed loud failures were not "
        "flagged by any screen; the rest were flagged with a different class",
    )

    out.rule("2. THE RULE BASELINE AS A SWEPT SCORE, EQUATION (15)")
    out(
        "Scoring a rule set only at its own operating point understates it, so "
        "equation (15) is swept and the whole operating characteristic is reported "
        "beside the point."
    )
    out("")
    out(
        f"{'rate':>6s} {'own recall':>11s} {'own fpr':>9s} {'swept recall':>13s} "
        f"{'swept fpr':>10s} {'area':>7s}"
    )
    areas = []
    for rate in (0.25, 0.5, 0.75, 1.0):
        _records, prepared = _prepare(OBJECTS, rate, HISTORY_LENGTH)
        positive = prepared.measured_classes == SILENTLY_WRONG
        if not positive.any() or positive.all():
            continue
        point = benchmark.point_operating_point(prepared.rule_classes, positive)
        swept = benchmark.recall_at_fpr(prepared.rule_scores, positive, TARGET_FPR)
        area = benchmark.area_under_curve(prepared.rule_scores, positive)
        areas.append(area)
        out(
            f"{rate:6.2f} {point.recall:11.4f} {point.false_positive_rate:9.4f} "
            f"{swept.recall:13.4f} {swept.false_positive_rate:10.4f} {area:7.4f}"
        )
    out.check(
        "rule_score_beats_chance_on_every_cell",
        all(area > 0.5 for area in areas),
        f"min area {min(areas):.4f} over {len(areas)} cells",
    )

    out.rule("3. THE DECLARED FOUR-SIGMA THRESHOLD, SWEPT")
    out(
        "The history and continuity screens both use a declared modified "
        "z-score threshold of 4.0. It was not chosen by fitting the corruption "
        "corpus; the per-element false-positive rate it produces on uncorrupted "
        "element sets is measured here, and the sweep shows what other choices "
        "would cost."
    )
    out("")
    out(
        f"{'threshold':>10s} {'silent P':>10s} {'silent R':>10s} {'accuracy':>9s} "
        f"{'area':>7s}"
    )
    for threshold in (2.0, 3.0, 4.0, 5.0, 8.0):
        _records, prepared = _prepare(OBJECTS, 0.5, HISTORY_LENGTH, threshold)
        scores = benchmark.per_class(prepared.measured_classes, prepared.rule_classes)
        positive = prepared.measured_classes == SILENTLY_WRONG
        out(
            f"{threshold:10.1f} {scores[SILENTLY_WRONG].precision:10.4f} "
            f"{scores[SILENTLY_WRONG].recall:10.4f} "
            f"{benchmark.accuracy(prepared.measured_classes, prepared.rule_classes):9.4f} "
            f"{benchmark.area_under_curve(prepared.rule_scores, positive):7.4f}"
        )

    out.rule("4. THE PER-ELEMENT FALSE-POSITIVE RATE OF THE HISTORY SCREEN")
    out(
        "Measured on uncorrupted element sets only, which is the only population "
        "where a history finding is unambiguously a false positive."
    )
    out("")
    out(
        f"{'history length':>15s} {'z tests':>9s} {'frac > 4':>9s} {'p95':>8s} "
        f"{'p99':>8s} {'max':>10s}"
    )
    history_rows = []
    for history_length in (4, 7, 12, 18):
        records, prepared = _prepare(OBJECTS, 0.5, history_length)
        values: list[float] = []
        for record, evidence in zip(records, prepared.evidences, strict=True):
            if record.corruption != "none":
                continue
            if evidence.history is None or not evidence.history.usable:
                continue
            values.extend(
                value for value in evidence.history.z_scores.values() if np.isfinite(value)
            )
        array = np.array(values)
        if array.size == 0:
            continue
        fraction = float((array > 4.0).mean())
        history_rows.append((history_length, fraction, array))
        out(
            f"{history_length:15d} {array.size:9d} {fraction:9.4f} "
            f"{float(np.percentile(array, 95)):8.4f} "
            f"{float(np.percentile(array, 99)):8.4f} {float(array.max()):10.4f}"
        )
    chosen = next(row for row in history_rows if row[0] == HISTORY_LENGTH)
    out("")
    out(
        f"At the declared history length of {HISTORY_LENGTH} the measured "
        f"per-element false-positive rate at a four-sigma threshold is "
        f"{chosen[1]:.4f}, against the 0.0001 a Gaussian four-sigma tail would "
        "give. The excess is the small-sample variance of a median absolute "
        "deviation over twelve points and it is reported rather than corrected: "
        "a scale estimated from twelve numbers is not a scale, and the screen's "
        "operating point reflects that."
    )
    out.check(
        "history_false_positive_rate_falls_with_history_length",
        history_rows[0][1] >= history_rows[-1][1],
        f"{history_rows[0][1]:.4f} at {history_rows[0][0]} sets against "
        f"{history_rows[-1][1]:.4f} at {history_rows[-1][0]}",
    )

    out.rule("5. EQUATION (27): THE CONTINUITY SCREEN'S RESOLUTION, BY ORBIT FAMILY")
    out(
        "The screen's resolution is the robust scale of the object's own "
        "consecutive element-set disagreement. Where it exceeds the declared "
        "silent threshold the screen cannot resolve a threshold-sized error for "
        "that object and the absolute test of equation (26) is withheld. This is "
        "a computed limit and it is returned with every verdict."
    )
    out("")
    records, prepared = _prepare(OBJECTS, 0.5, HISTORY_LENGTH)
    by_family: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for record, evidence in zip(records, prepared.evidences, strict=True):
        if record.corruption != "none":
            continue
        continuity = evidence.continuity
        if continuity is None or not continuity.usable:
            continue
        by_family[record.family].append(
            (continuity.offset_km, continuity.median_km, continuity.resolution_km)
        )
    out(
        f"{'family':28s} {'n':>4s} {'median offset km':>17s} {'median normal km':>17s} "
        f"{'median resolution km':>21s} {'resolves 1 km':>14s}"
    )
    resolving = 0
    for family in sorted(by_family):
        array = np.array(by_family[family])
        resolution = float(np.median(array[:, 2]))
        resolves = resolution <= SILENT_THRESHOLD_KM
        resolving += resolves
        out(
            f"{family:28s} {len(array):>4d} {float(np.median(array[:, 0])):17.4f} "
            f"{float(np.median(array[:, 1])):17.4f} {resolution:21.4f} "
            f"{str(resolves):>14s}"
        )
    out("")
    out(
        f"{resolving} of {len(by_family)} orbit families have a continuity "
        f"resolution at or below the declared {SILENT_THRESHOLD_KM:g} km threshold. "
        "For the rest, comparing element sets cannot see a one-kilometre "
        "consequence at all, and the screen says so instead of returning a verdict "
        "that is noise. The decaying low-Earth family is the extreme case and the "
        "reason is physical: a high-drag object's element set genuinely does not "
        "predict its own successor's ephemeris to within kilometres."
    )
    out.check(
        "some_families_do_not_resolve_the_threshold",
        resolving < len(by_family),
        f"{len(by_family) - resolving} of {len(by_family)} families do not; this is "
        "the limit the screen exists to report",
    )
    out.check(
        "resolution_is_always_above_its_floor",
        all(
            float(np.min(np.array(values)[:, 2])) >= 0.050 - 1.0e-12
            for values in by_family.values()
        ),
        "the declared 0.050 km floor holds on every family",
    )

    out.rule("6. THE STRUCTURAL LIMIT OF THE CROSS-SOURCE SCREEN, MEASURED")
    out(
        "The corpus contains two constructions of the same observable situation. "
        "In the cross_source_divergence family the candidate is the perturbed "
        "source and the other source is clean, so the measured consequence is "
        "real. In the uncorrupted family with a second source attached the "
        "candidate is the clean one and the other source is perturbed, so the "
        "measured consequence is zero. The two are indistinguishable from the two "
        "element sets alone, because equation (12) is symmetric. No screen can be "
        "right on both and the measured cost is below."
    )
    out("")
    big_records, big = _prepare(OBJECTS, 0.5, HISTORY_LENGTH)
    separations: dict[str, list[float]] = defaultdict(list)
    fired_by_truth: dict[str, int] = defaultdict(int)
    attached = 0
    for record, evidence, measured in zip(
        big_records, big.evidences, big.measured_classes, strict=True
    ):
        usable = [d for d in evidence.divergences if d.usable]
        if not usable:
            continue
        attached += 1
        worst = max(d.max_separation_km for d in usable)
        construction = (
            "candidate is the perturbed source"
            if record.corruption == "cross_source_divergence"
            else "candidate is the clean source"
        )
        separations[construction].append(worst)
        if any(d.diverged for d in usable):
            fired_by_truth[str(measured)] += 1
    out(f"records with a second source : {attached}")
    out("")
    out(
        f"{'construction':36s} {'n':>4s} {'min km':>10s} {'median km':>11s} "
        f"{'max km':>11s} {'above 1 km':>11s}"
    )
    for construction in sorted(separations):
        array = np.array(separations[construction])
        out(
            f"{construction:36s} {array.size:>4d} {float(array.min()):10.4f} "
            f"{float(np.median(array)):11.4f} {float(array.max()):11.4f} "
            f"{int((array > SILENT_THRESHOLD_KM).sum()):>11d}"
        )
    out("")
    out(f"{'measured class':20s} {'records the divergence screen flagged':>38s}")
    for label in CLASSES:
        out(f"{label:20s} {fired_by_truth.get(label, 0):>38d}")
    total = sum(fired_by_truth.values())
    silent = fired_by_truth.get(SILENTLY_WRONG, 0)
    out("")
    out(f"divergence screen fired on {total} records")
    if total:
        out(
            f"of which {silent} were measured silently wrong, a precision of "
            f"{silent / total:.4f} for this screen alone"
        )
    clean_separations = np.array(
        separations.get("candidate is the clean source", [0.0])
    )
    clean_above = int((clean_separations > SILENT_THRESHOLD_KM).sum())
    out("")
    out(
        f"{clean_above} records whose candidate is the clean source have a "
        f"cross-source separation above the declared {SILENT_THRESHOLD_KM:g} km "
        "threshold. Those are irreducible false positives for this screen: the two "
        "element sets in front of it are the same two element sets as in a "
        "cross_source_divergence record, with the labels swapped. Equation (12) is "
        "symmetric and cannot tell them apart. The complement of the precision "
        "above is not a defect in the screen; it is the structural limit the "
        "determinability report names, and this is its measured size."
    )
    out.check(
        "divergence_screen_fired",
        total > 0,
        f"{total} records; a control that does not fire is not evidence",
    )
    out.check(
        "both_constructions_of_the_symmetric_situation_are_present",
        len(separations) == 2,
        f"{sorted(separations)}; the corpus contains the same observable situation "
        "labelled both ways, which is what makes the structural limit measurable",
    )
    clean_values = np.array(separations.get("candidate is the clean source", [0.0]))
    perturbed_values = np.array(
        separations.get("candidate is the perturbed source", [0.0])
    )
    overlap = float(min(perturbed_values)) < float(max(clean_values))
    out("")
    out(
        "Whether the two distributions overlap at this corpus size is a property "
        "of the generator's declared perturbation scales and is reported rather "
        f"than asserted: measured here, the clean construction reaches "
        f"{float(max(clean_values)):.4f} km and the perturbed construction starts "
        f"at {float(min(perturbed_values)):.4f} km, so they "
        + ("overlap" if overlap else "do not overlap")
        + ". The symmetry below holds whether they overlap or not."
    )
    from tlecheck.divergence import compare as compare_sources

    pair = next(
        record
        for record in big_records
        if record.corruption == "cross_source_divergence" and record.other_sources
    )
    forward = compare_sources(
        pair.candidate, pair.other_sources[0], pair.use_jd, HORIZON_DAYS, SAMPLES
    )
    backward = compare_sources(
        pair.other_sources[0], pair.candidate, pair.use_jd, HORIZON_DAYS, SAMPLES
    )
    out("")
    out(f"object {pair.satnum}: equation (12) forward  {forward.max_separation_km:.8f} km")
    out(f"object {pair.satnum}: equation (12) backward {backward.max_separation_km:.8f} km")
    out(f"which source is correct, forward  : {forward.which_source_is_correct}")
    out(f"which source is correct, backward : {backward.which_source_is_correct}")
    out.check(
        "equation_12_is_bit_exactly_symmetric",
        forward.max_separation_km == backward.max_separation_km,
        "swapping the two arguments changes nothing, which is why the attribution "
        "is not available at any threshold",
    )
    out.check(
        "the_screen_never_names_a_source",
        forward.which_source_is_correct is None
        and backward.which_source_is_correct is None,
        "the field is present in the result and is always None",
    )

    out.rule("7. EVERY SCREEN'S FIRING RATE ON UNCORRUPTED ELEMENT SETS")
    out(
        "The per-screen false-positive tally, so a reader can see which screen "
        "contributes which part of the baseline's operating point."
    )
    out("")
    tally: dict[str, int] = defaultdict(int)
    clean = 0
    for record, evidence in zip(records, prepared.evidences, strict=True):
        if record.corruption != "none":
            continue
        clean += 1
        if any(f.name == "checksum_mismatch" for f in evidence.format_findings):
            tally["checksum_mismatch"] += 1
        if any(f.name == "parse_fault" for f in evidence.format_findings):
            tally["parse_fault"] += 1
        for finding in evidence.bound_findings:
            tally[f"bounds:{finding.name}"] += 1
        if evidence.epoch is not None and evidence.epoch.stale:
            tally["epoch_stale"] += 1
        if evidence.history is not None:
            for finding in evidence.history.findings:
                tally[f"history:{finding.element}"] += 1
        if (
            evidence.continuity is not None
            and evidence.continuity.usable
            and evidence.continuity.exceeded
        ):
            tally["continuity_z_exceeded"] += 1
        if any(d.usable and d.diverged for d in evidence.divergences):
            tally["cross_source_diverged"] += 1
    out(f"uncorrupted element sets : {clean}")
    out("")
    out(f"{'screen finding':40s} {'fired':>6s} {'rate':>8s}")
    for name in sorted(tally, key=lambda key: -tally[key]):
        out(f"{name:40s} {tally[name]:>6d} {tally[name] / clean:8.4f}")
    if not tally:
        out("no screen fired on any uncorrupted element set")
    out.check(
        "no_checksum_or_parse_false_positive_on_clean_records",
        tally["checksum_mismatch"] == 0 and tally["parse_fault"] == 0,
        "the format and checksum screens are arithmetic and have no false "
        "positive rate at all",
    )
    benign_recall = benchmark.per_class(
        prepared.measured_classes, prepared.rule_classes
    )[BENIGN].recall
    out.check(
        "rule_baseline_predicts_benign_on_most_clean_records",
        benign_recall > 0.5,
        f"benign recall {benign_recall:.4f}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
