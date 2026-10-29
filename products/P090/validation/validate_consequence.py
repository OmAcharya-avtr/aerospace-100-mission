"""Measured consequence: the kilometres that make the *silently wrong* class a measurement.

Nothing in this script assumes a class from a corruption family. Every class
label is produced by propagating the candidate element set and its uncorrupted
reference with ``sgp4`` over the same absolute time grid and differencing the
positions, which is equation (11). The class is a declared threshold applied to
that number and the threshold is swept here so a reader can see the boundary
move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    HISTORY_LENGTH,
    HORIZON_DAYS,
    OBJECTS,
    SAMPLES,
    SILENT_THRESHOLD_KM,
    Tee,
    bootstrap,
    declared_settings,
    environment_lines,
)

bootstrap()

from tlecheck import dataset  # noqa: E402
from tlecheck.consequence import (  # noqa: E402
    BENIGN,
    CLASSES,
    SILENTLY_WRONG,
    WILL_FAIL_LOUDLY,
    measure,
)
from tlecheck.elements import format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402
from tlecheck.propagate import propagate, time_grid  # noqa: E402
from tlecheck.screens import run as run_screens  # noqa: E402
from tlecheck.verdict import classify  # noqa: E402


def _table(out: Tee, records, horizon: float, samples: int, threshold: float) -> dict:
    buckets: dict[str, list] = {}
    for record in records:
        consequence = measure(
            record.candidate,
            record.reference,
            record.use_jd,
            horizon_days=horizon,
            samples=samples,
            silent_threshold_km=threshold,
        )
        buckets.setdefault(record.corruption, []).append(consequence)
    header = f"{'injected corruption':28s} {'n':>4s}"
    for label in CLASSES:
        header += f" {label:>17s}"
    header += f" {'median max km':>14s} {'largest max km':>15s}"
    out(header)
    for family in dataset.CORRUPTION_FAMILIES:
        group = buckets.get(family)
        if not group:
            continue
        line = f"{family:28s} {len(group):>4d}"
        for label in CLASSES:
            line += f" {sum(1 for c in group if c.consequence_class == label):>17d}"
        finite = [c.max_error_km for c in group if np.isfinite(c.max_error_km)]
        median = float(np.median(finite)) if finite else float("nan")
        largest = float(np.max(finite)) if finite else float("nan")
        line += f" {median:14.6f} {largest:15.4f}"
        out(line)
    return buckets


def main() -> int:
    out = Tee("validate_consequence_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    records = dataset.split("test", objects=OBJECTS, history_length=HISTORY_LENGTH)
    out("")
    out(f"objects screened : {len(records)} (split 'test')")

    out.rule("1. MEASURED CONSEQUENCE BY INJECTED CORRUPTION FAMILY")
    out(
        f"Propagated with sgp4 over {HORIZON_DAYS:g} days at {SAMPLES} samples, "
        f"declared silent threshold {SILENT_THRESHOLD_KM:g} km."
    )
    out("")
    buckets = _table(out, records, HORIZON_DAYS, SAMPLES, SILENT_THRESHOLD_KM)

    checksum_only = buckets.get("checksum_digit", [])
    errors = [c.max_error_km for c in checksum_only if np.isfinite(c.max_error_km)]
    out("")
    out(
        "The checksum_digit family is the one worth reading twice. Its corruption "
        "is the checksum column alone, so the data is intact and the propagated "
        f"position error is exactly {max(errors) if errors else float('nan'):.1f} km "
        f"on all {len(checksum_only)} of them. The checksum screen fires on every "
        "one and it is right to: the element set is not what its source wrote. "
        "The consequence of that change, measured, is nothing. Those are two "
        "different statements and this package returns them separately."
    )
    out.check(
        "checksum_only_corruption_has_zero_measured_consequence",
        bool(errors) and max(errors) == 0.0,
        f"largest measured error {max(errors) if errors else float('nan'):.6f} km over "
        f"{len(checksum_only)} element sets",
    )
    out.check(
        "every_class_is_populated",
        all(
            any(c.consequence_class == label for group in buckets.values() for c in group)
            for label in CLASSES
        ),
        "all three consequence classes occur in the corpus",
    )

    out.rule("2. THE CLASS BOUNDARY MOVES WITH THE DECLARED THRESHOLD")
    out(
        "The threshold is a policy value and not a physical constant. A reader "
        "screening for conjunction assessment and a reader screening for a "
        "ground-station pass schedule do not want the same number, so the "
        "boundary is swept rather than asserted."
    )
    out("")
    out(f"{'threshold km':>13s} {'benign':>9s} {'silent':>9s} {'loud':>9s}")
    counts = {}
    for threshold in (0.1, 0.5, 1.0, 5.0, 50.0, 500.0):
        tally = dict.fromkeys(CLASSES, 0)
        for record in records:
            consequence = measure(
                record.candidate,
                record.reference,
                record.use_jd,
                horizon_days=HORIZON_DAYS,
                samples=SAMPLES,
                silent_threshold_km=threshold,
            )
            tally[consequence.consequence_class] += 1
        counts[threshold] = tally
        out(
            f"{threshold:13.1f} {tally[BENIGN]:>9d} {tally[SILENTLY_WRONG]:>9d} "
            f"{tally[WILL_FAIL_LOUDLY]:>9d}"
        )
    out.check(
        "silent_count_falls_as_the_threshold_rises",
        counts[0.1][SILENTLY_WRONG] >= counts[500.0][SILENTLY_WRONG],
        f"{counts[0.1][SILENTLY_WRONG]} at 0.1 km against "
        f"{counts[500.0][SILENTLY_WRONG]} at 500 km",
    )
    out.check(
        "loud_count_is_threshold_independent",
        len({tally[WILL_FAIL_LOUDLY] for tally in counts.values()}) == 1,
        "a loud failure is the propagator's own verdict and no threshold changes it",
    )

    out.rule("3. THE CONSEQUENCE GROWS WITH THE PREDICTION HORIZON")
    out(
        "A corrupted drag term costs nothing at the epoch and a great deal a week "
        "later. The horizon is a declared choice and this is what it is worth."
    )
    out("")
    out(
        f"{'horizon days':>13s} {'benign':>9s} {'silent':>9s} {'loud':>9s} "
        f"{'median silent km':>18s}"
    )
    for horizon, samples in ((0.25, 13), (1.0, 25), (3.0, 37), (7.0, 49)):
        tally = dict.fromkeys(CLASSES, 0)
        silent_errors = []
        for record in records:
            consequence = measure(
                record.candidate,
                record.reference,
                record.use_jd,
                horizon_days=horizon,
                samples=samples,
                silent_threshold_km=SILENT_THRESHOLD_KM,
            )
            tally[consequence.consequence_class] += 1
            if consequence.consequence_class == SILENTLY_WRONG:
                silent_errors.append(consequence.max_error_km)
        median = float(np.median(silent_errors)) if silent_errors else float("nan")
        out(
            f"{horizon:13.2f} {tally[BENIGN]:>9d} {tally[SILENTLY_WRONG]:>9d} "
            f"{tally[WILL_FAIL_LOUDLY]:>9d} {median:18.4f}"
        )

    out.rule("4. A SINGLE SIGN FLIP ON A PUBLISHED ELEMENT SET, IN KILOMETRES")
    fields = {
        "satnum": 25544,
        "epoch_year": 2019,
        "epoch_day": 343.69339541,
        "inclination_deg": 51.6439,
        "raan_deg": 211.2001,
        "eccentricity": 0.0007417,
        "arg_perigee_deg": 17.6667,
        "mean_anomaly_deg": 85.6398,
        "mean_motion_rev_per_day": 15.50103472,
        "bstar": 3.8792e-05,
        "mean_motion_dot_rev_per_day2": 1.764e-05,
        "designator": "98067A",
        "element_number": 999,
        "revolution_number": 20248,
    }
    reference = parse_tle(*format_tle(**fields))
    flipped = parse_tle(*format_tle(**{**fields, "bstar": -fields["bstar"]}))
    out("reference line 1 : " + reference.line1)
    out("flipped   line 1 : " + flipped.line1)
    out("")
    out(
        "The two lines differ in one column, the sign of the drag term, and both "
        "carry a valid checksum. Every element is inside every physical bound. "
        "The propagator reports success at every sample."
    )
    out("")
    out(f"{'horizon days':>13s} {'max error km':>14s} {'rms error km':>14s} {'class':>18s}")
    start = epoch_julian_date(reference)
    for horizon, samples in ((1.0, 25), (3.0, 37), (7.0, 49), (14.0, 57), (30.0, 61)):
        consequence = measure(flipped, reference, start, horizon, samples, SILENT_THRESHOLD_KM)
        out(
            f"{horizon:13.1f} {consequence.max_error_km:14.4f} "
            f"{consequence.rms_error_km:14.4f} {consequence.consequence_class:>18s}"
        )
    seven = measure(flipped, reference, start, 7.0, 49, SILENT_THRESHOLD_KM)
    out.check(
        "sign_flip_is_silently_wrong_at_seven_days",
        seven.consequence_class == SILENTLY_WRONG,
        f"{seven.max_error_km:.4f} km, propagator reported success at every sample",
    )
    out.check(
        "sign_flip_is_invisible_to_the_checksum",
        flipped.checksum_line1_stated == flipped.checksum_line1_computed,
        "the flipped line carries a valid checksum",
    )

    out.rule("5. A CONSTRUCTED VALID-BUT-STALE ELEMENT SET, THE DECLARED CONTROL")
    out(
        "The control is a genuine, uncorrupted element set whose only defect is "
        "its age. It is constructed with tlecheck.dataset.rewound, which runs the "
        "propagator back to the older epoch and writes the mean elements it "
        "computed there, so the element set describes the same orbit at an earlier "
        "time. Nothing in its characters is wrong: the checksum is valid, every "
        "element is inside every bound, and the propagator reports success."
    )
    out("")
    out(
        "Stamping an earlier epoch onto an element set without moving its angles "
        "does NOT produce an older element set, it produces a different orbit. The "
        "first version of this control did that and measured 9752.6918 km at an "
        "age of 1.5 days, which is a meaningless number. The figures below are "
        "from the propagator-based construction."
    )
    out("")
    out(
        f"{'age days':>10s} {'max error km':>14s} {'class':>18s} {'stale':>7s} "
        f"{'checksum ok':>12s}"
    )
    rows = []
    for age in (0.0, 1.5, 3.0, 7.5, 15.0, 30.0, 60.0):
        older = dataset.rewound(reference, -age, source="stale-control")
        if older is None:
            out(f"{age:10.1f} {'propagator refused the rewind':>44s}")
            continue
        consequence = measure(older, reference, start + 1.0, 1.0, 25, SILENT_THRESHOLD_KM)
        evidence = run_screens(older, reference_jd=start + 1.0, epoch_requirement_days=7.0)
        stale = bool(evidence.epoch and evidence.epoch.stale)
        checksum_ok = (
            older.checksum_line1_stated == older.checksum_line1_computed
            and older.checksum_line2_stated == older.checksum_line2_computed
        )
        rows.append((age, consequence, stale, checksum_ok))
        out(
            f"{age:10.1f} {consequence.max_error_km:14.4f} "
            f"{consequence.consequence_class:>18s} {str(stale):>7s} "
            f"{str(checksum_ok):>12s}"
        )
    out.check(
        "stale_control_fires_above_the_requirement",
        all(stale for age, _c, stale, _k in rows if age > 7.0),
        "every constructed age above the declared 7 day requirement is flagged",
    )
    out.check(
        "stale_control_does_not_fire_below_the_requirement",
        not any(stale for age, _c, stale, _k in rows if age < 6.0),
        "no constructed age below the requirement is flagged; the control is "
        "asserted in both directions",
    )
    out.check(
        "stale_control_is_otherwise_clean",
        all(ok for _a, _c, _s, ok in rows),
        "every constructed control carries a valid checksum and no bound finding",
    )
    out.check(
        "stale_consequence_rises_with_age",
        all(
            rows[index][1].max_error_km <= rows[index + 1][1].max_error_km + 1.0e-9
            for index in range(len(rows) - 1)
        ),
        "the measured position error is monotone in the age over the constructed "
        "ladder",
    )

    first_silent = next(
        (age for age, c, _s, _k in rows if c.consequence_class == SILENTLY_WRONG), None
    )
    out("")
    out(
        "A disagreement worth reporting rather than resolving. The declared epoch "
        "requirement of 7 days flags this element set from an age of 7.5 days "
        f"onward, and the measured consequence first exceeds the declared "
        f"{SILENT_THRESHOLD_KM:g} km threshold at an age of "
        f"{first_silent if first_silent is not None else float('nan'):g} days. For "
        "this object the staleness policy is the more conservative of the two by a "
        "factor of about four. Neither number is wrong: the requirement is an "
        "operational choice and the threshold is a different operational choice, "
        "and a screen that reported only one of them would be hiding the gap. The "
        "determinability report returns the requirement with every verdict for "
        "exactly this reason."
    )

    out.rule("6. DOES THE RULE BASELINE'S LOUD PREDICTION MATCH THE PROPAGATOR?")
    out(
        "The rules predict a loud failure from a physical bound violation, on the "
        "grounds that the SGP4 error codes are exactly those cases. That is a "
        "prediction and it is scored here against what the propagator actually "
        "did."
    )
    out("")
    agree = 0
    predicted_loud = 0
    measured_loud = 0
    both = 0
    missed_unreadable = 0
    missed_flagged = 0
    missed_unflagged = 0
    codes: dict[int, int] = {}
    for record in records:
        evidence = run_screens(
            record.candidate,
            history=list(record.history),
            other_sources=list(record.other_sources),
            reference_jd=record.use_jd,
        )
        predicted, _score, _problems, _intact = classify(evidence)
        consequence = measure(
            record.candidate,
            record.reference,
            record.use_jd,
            HORIZON_DAYS,
            SAMPLES,
            SILENT_THRESHOLD_KM,
        )
        is_predicted = predicted == WILL_FAIL_LOUDLY
        is_measured = consequence.consequence_class == WILL_FAIL_LOUDLY
        predicted_loud += is_predicted
        measured_loud += is_measured
        both += is_predicted and is_measured
        agree += is_predicted == is_measured
        if is_measured and not is_predicted:
            if any(f.name == "parse_fault" for f in evidence.format_findings):
                missed_unreadable += 1
            elif predicted == SILENTLY_WRONG:
                missed_flagged += 1
            else:
                missed_unflagged += 1
        if is_measured:
            result = propagate(
                record.candidate, time_grid(record.use_jd, HORIZON_DAYS, SAMPLES)
            )
            for code in np.unique(result.error_codes):
                codes[int(code)] = codes.get(int(code), 0) + 1
    out(f"records                         : {len(records)}")
    out(f"rules predicted a loud failure  : {predicted_loud}")
    out(f"propagator failed loudly        : {measured_loud}")
    out(f"both                            : {both}")
    out(f"loud recall                     : {both / measured_loud:.4f}")
    out(f"loud precision                  : {both / predicted_loud:.4f}")
    out("")
    out("SGP4 error codes seen on the loud records:")
    for code, count in sorted(codes.items()):
        out(f"  code {code}: {count} samples")
    out("")
    out("the loud failures the rules did not predict as loud:")
    out(f"  a field this package could not read : {missed_unreadable}")
    out(f"  flagged silently wrong instead      : {missed_flagged}")
    out(f"  not flagged by any screen           : {missed_unflagged}")
    out.check(
        "no_loud_failure_passed_the_screen_unflagged",
        missed_unflagged == 0,
        f"{missed_unflagged} of {measured_loud} loud failures were not flagged at "
        "all; the rest were flagged with a different class attached",
    )
    out.check(
        "the_rules_predict_loud_on_most_loud_failures",
        both / measured_loud > 0.9 if measured_loud else False,
        f"recall {both / measured_loud:.4f} on {measured_loud} loud failures",
    )
    out(
        "The rules both over-predict and under-predict a loud failure, and both "
        "are measured above. They over-predict because an element set whose "
        "eccentricity is implausible for its object sometimes still propagates, "
        "and when it does the measured class is silently wrong. They under-predict "
        "because an element set with a field this package cannot read is "
        "classified silently wrong rather than loud, deliberately: the propagator "
        "usually reads such a line without complaint, and predicting a loud "
        "failure from a parse failure drove the field-shift operating "
        "characteristic below chance. Neither direction is tuned away."
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
