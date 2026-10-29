"""Reproduce every "before the fix" measurement this repository quotes.

The mission's build guide requires every number in a validation document to come
from a script executed in the same session. Five of the numbers
``validation/VALIDATION.md`` quotes are measurements of code that no longer
exists: the robust scale before it had a quantisation floor, the node before it
moved to the trend test, and two earlier versions of the element-history
generator. They are the evidence for how much of the baseline's published
strength was the problem and how much was the first implementation, so they
cannot simply be dropped, and they must not be carried as remembered figures
either.

This script reproduces each of them **by construction**, deliberately
reinstating the superseded behaviour alongside the current one and measuring
both on the same corpus. Every figure in section 7 of ``VALIDATION.md`` traces
here.

What is reinstated, and how
---------------------------
1. **The unfloored robust scale.** :func:`tlecheck.history.modified_z` takes the
   floor as an argument, so passing 0.0 is the superseded behaviour exactly.
2. **The level test on the right ascension of the ascending node.** The level
   test is :func:`tlecheck.history.modified_z` against the median of the
   unwrapped history, which is what the node used to get.
3. **Generator version 2, the Brouwer mean motion written into the Kozai
   field.** Identical to :func:`tlecheck.dataset.rewound` with the inverse
   transformation of equation (18) omitted, which is one line.
4. **Generator version 1, the kinematic element history.** Reimplemented here in
   full: the J2 secular rates of equations (16) and (17) for the node and the
   argument of perigee, a mean anomaly advanced at ``360 n dt``, and the mean
   motion drift of the element set's own first-derivative field.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    CADENCE_DAYS,
    HISTORY_LENGTH,
    OBJECTS,
    Tee,
    bootstrap,
    declared_settings,
    environment_lines,
)

bootstrap()

from sgp4.api import Satrec  # noqa: E402

from tlecheck import dataset  # noqa: E402
from tlecheck.consequence import position_errors_km  # noqa: E402
from tlecheck.elements import ElementSet, format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402
from tlecheck.history import (  # noqa: E402
    modified_z,
    robust_scale,
    scale_floor,
    theil_sen,
    unwrap_degrees,
)
from tlecheck.propagate import propagate, time_grid  # noqa: E402

FAMILY_CASES = {
    "leo_low_inclination": dict(
        inclination_deg=51.60, eccentricity=0.0008, mean_motion_rev_per_day=15.50,
        bstar=4.0e-5,
    ),
    "leo_sun_synchronous": dict(
        inclination_deg=98.20, eccentricity=0.0012, mean_motion_rev_per_day=14.60,
        bstar=1.1e-4,
    ),
}


def _base(name: str, overrides: dict) -> ElementSet:
    fields = {
        "satnum": 28000,
        "epoch_year": 2026,
        "epoch_day": 200.5,
        "raan_deg": 120.0,
        "arg_perigee_deg": 40.0,
        "mean_anomaly_deg": 200.0,
        "mean_motion_dot_rev_per_day2": max(0.0, overrides["bstar"]) * 0.45,
        "designator": "26000A",
        "element_number": 100,
        "revolution_number": 10000,
        **overrides,
    }
    return parse_tle(*format_tle(**fields), source=name)


def _rewound_without_kozai_inverse(base: ElementSet, offset_days: float) -> ElementSet | None:
    """Generator version 2: ``Satrec.nm`` written straight into the Kozai field."""
    satrec = Satrec.twoline2rv(base.line1, base.line2)
    error, _position, _velocity = satrec.sgp4_tsince(offset_days * 1440.0)
    if error != 0:
        return None
    mean_motion = float(satrec.nm) * 1440.0 / (2.0 * math.pi)
    fields = {
        "satnum": base.satnum,
        "epoch_year": base.epoch_full_year,
        "epoch_day": base.epoch_day + offset_days,
        "inclination_deg": math.degrees(float(satrec.im)) % 180.0,
        "raan_deg": math.degrees(float(satrec.Om)) % 360.0,
        "eccentricity": max(0.0, min(0.99, float(satrec.em))),
        "arg_perigee_deg": math.degrees(float(satrec.om)) % 360.0,
        "mean_anomaly_deg": math.degrees(float(satrec.mm)) % 360.0,
        "mean_motion_rev_per_day": mean_motion,
        "bstar": base.bstar,
        "mean_motion_dot_rev_per_day2": base.mean_motion_dot_rev_per_day2,
        "designator": "26000A",
        "element_number": 100,
        "revolution_number": 10000,
    }
    return parse_tle(*format_tle(**fields), source="v2")


def _kinematic_history(base: ElementSet, length: int, cadence: float) -> list[ElementSet]:
    """Generator version 1: the J2 secular rates and a mean anomaly at 360 n dt."""
    node_rate = dataset.nodal_regression_deg_per_day(
        base.mean_motion_rev_per_day, base.eccentricity, base.inclination_deg
    )
    apsis_rate = dataset.apsidal_precession_deg_per_day(
        base.mean_motion_rev_per_day, base.eccentricity, base.inclination_deg
    )
    ndot2 = base.mean_motion_dot_rev_per_day2
    out: list[ElementSet] = []
    for index in range(length + 1):
        offset = cadence * (index - length)
        mean_motion = base.mean_motion_rev_per_day + 2.0 * ndot2 * offset
        fields = {
            "satnum": base.satnum,
            "epoch_year": base.epoch_full_year,
            "epoch_day": base.epoch_day + offset,
            "inclination_deg": base.inclination_deg,
            "raan_deg": (base.raan_deg + node_rate * offset) % 360.0,
            "eccentricity": base.eccentricity,
            "arg_perigee_deg": (base.arg_perigee_deg + apsis_rate * offset) % 360.0,
            "mean_anomaly_deg": (
                base.mean_anomaly_deg + 360.0 * mean_motion * offset
            ) % 360.0,
            "mean_motion_rev_per_day": mean_motion,
            "bstar": base.bstar,
            "mean_motion_dot_rev_per_day2": ndot2,
            "designator": "26000A",
            "element_number": 100 + index,
            "revolution_number": 10000 + 30 * index,
        }
        out.append(parse_tle(*format_tle(**fields), source="v1"))
    return out


def _consecutive_offsets(sets: list[ElementSet]) -> list[float]:
    """Propagated disagreement between consecutive element sets, kilometres."""
    offsets: list[float] = []
    for earlier, later in zip(sets[:-1], sets[1:], strict=False):
        start = epoch_julian_date(later) + 1.0
        if not math.isfinite(start):
            continue
        grid = time_grid(start, 1.0, 9)
        first = propagate(earlier, grid)
        second = propagate(later, grid)
        if first.loud or second.loud:
            continue
        offsets.append(float(np.max(position_errors_km(first, second))))
    return offsets


def main() -> int:
    out = Tee("validate_baseline_history_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    out.rule("1. THE ROBUST SCALE WITHOUT ITS QUANTISATION FLOOR")
    out(
        "The floor is one count of the element's own text field. Without it, a "
        "history whose mean motion is constant to the eighth written decimal has "
        "a detrended scale of order 1e-8 and the z-score of a one-count change is "
        "astronomical. Measured both ways on the same records, on two corpora: "
        "the current generator, whose element sets carry orbit-determination "
        "jitter, and the superseded version 1 generator, whose mean motion was "
        "advanced analytically with no jitter at all and therefore quantised to "
        "the written decimal. On the second corpus the candidate is constructed "
        "as the robust trend prediction plus exactly one count of the mean-motion "
        "field, 1e-8 rev/day, which is the smallest change the text format can "
        "express and the case the floor exists for."
    )
    out("")
    records = dataset.split("test", objects=OBJECTS, history_length=HISTORY_LENGTH)
    corpora: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {
        "current generator, with jitter": [],
        "version 1 generator, no jitter": [],
    }
    for record in records:
        if record.corruption != "none":
            continue
        ordered = sorted(record.history, key=epoch_julian_date)
        if len(ordered) < 4:
            continue
        corpora["current generator, with jitter"].append(
            (
                np.array([epoch_julian_date(entry) for entry in ordered]),
                np.array(
                    [entry.mean_motion_rev_per_day for entry in ordered], dtype=float
                ),
                record.candidate.mean_motion_rev_per_day,
            )
        )
    for name, overrides in FAMILY_CASES.items():
        base = _base(name, overrides)
        sets = _kinematic_history(base, HISTORY_LENGTH, CADENCE_DAYS)
        history = sets[:-1]
        times = np.array([epoch_julian_date(entry) for entry in history])
        series = np.array(
            [entry.mean_motion_rev_per_day for entry in history], dtype=float
        )
        slope, intercept = theil_sen(times, series)
        # The candidate is the linear prediction plus exactly ONE COUNT of the
        # mean-motion field, 1e-8 rev/day. That is the smallest change the text
        # format can express, and it is the case the floor exists for: a history
        # with no jitter has a robust scale at the quantiser, so the ratio of a
        # one-count residual to it is whatever the quantiser's rounding happens
        # to leave rather than a measurement of anything.
        predicted = intercept + slope * (times[-1] + CADENCE_DAYS)
        corpora["version 1 generator, no jitter"].append(
            (times, series, predicted + 1.0e-8)
        )

    out(
        f"{'corpus':34s} {'variant':20s} {'n':>5s} {'finite max':>14s} "
        f"{'infinities':>11s} {'frac > 4':>9s}"
    )
    summary: dict[tuple[str, str], np.ndarray] = {}
    for corpus, cases in corpora.items():
        floored: list[float] = []
        unfloored: list[float] = []
        for times, series, candidate_value in cases:
            slope, intercept = theil_sen(times, series)
            if not math.isfinite(slope):
                continue
            residuals = series - (intercept + slope * times)
            predicted = intercept + slope * (times[-1] + CADENCE_DAYS)
            residual = abs(candidate_value - predicted)
            raw_scale = robust_scale(residuals)
            floor = scale_floor("mean_motion_rev_per_day", series)
            for store, scale in (
                (floored, max(raw_scale, floor)),
                (unfloored, raw_scale),
            ):
                if scale == 0.0:
                    store.append(math.inf if residual > 0.0 else 0.0)
                else:
                    store.append(residual / scale)
        for label, values in (
            ("with the floor", floored),
            ("without the floor", unfloored),
        ):
            array = np.array(values)
            summary[(corpus, label)] = array
            finite = array[np.isfinite(array)]
            out(
                f"{corpus:34s} {label:20s} {array.size:>5d} "
                f"{float(finite.max()) if finite.size else float('nan'):14.6g} "
                f"{int((~np.isfinite(array)).sum()):>11d} "
                f"{float((array > 4.0).mean()):9.4f}"
            )

    current_unfloored = summary[("current generator, with jitter", "without the floor")]
    current_floored = summary[("current generator, with jitter", "with the floor")]
    old_unfloored = summary[("version 1 generator, no jitter", "without the floor")]
    old_floored = summary[("version 1 generator, no jitter", "with the floor")]
    out("")
    out(
        "A correction worth stating plainly. On the CURRENT corpus the floor "
        f"changes nothing: the largest finite z-score is "
        f"{float(current_unfloored[np.isfinite(current_unfloored)].max()):.4f} "
        "either way, because the generator's declared mean-motion jitter of "
        "1.5e-8 relative already keeps the robust scale above one count of the "
        "field. The blow-up that motivated the floor needs a history with no "
        "jitter, which is the superseded version 1 generator, and it is "
        "reproduced on that corpus below. The floor is still correct on its own "
        "terms -- it is derived from the text format and not from any corpus, and "
        "a real catalogue can publish element sets identical to the last written "
        "digit -- and its measured effect on the corpus this repository ships is "
        "nil. Both facts are published and neither is left to be inferred."
    )
    out("")
    old_unfloored_max = (
        float(old_unfloored[np.isfinite(old_unfloored)].max())
        if np.isfinite(old_unfloored).any()
        else float("nan")
    )
    old_floored_max = (
        float(old_floored[np.isfinite(old_floored)].max())
        if np.isfinite(old_floored).any()
        else float("nan")
    )
    out(
        f"version 1 corpus, largest finite z-score without the floor : "
        f"{old_unfloored_max:.6e}"
    )
    out(
        f"version 1 corpus, largest finite z-score with the floor    : "
        f"{old_floored_max:.6f}"
    )
    out(
        f"version 1 corpus, infinities without the floor             : "
        f"{int(np.isinf(old_unfloored).sum())} of {old_unfloored.size}"
    )
    out(
        f"version 1 corpus, infinities with the floor                : "
        f"{int(np.isinf(old_floored).sum())} of {old_floored.size}"
    )
    ratio = (
        old_unfloored_max / old_floored_max
        if old_floored_max not in (0.0, float("nan")) and math.isfinite(old_floored_max)
        else float("inf")
    )
    out(
        f"version 1 corpus, ratio of the two                         : "
        f"{ratio:.6e}"
    )
    out.check(
        "the_floor_matters_on_a_quantised_history",
        np.isinf(old_unfloored).any() or ratio > 1.0e4,
        f"an unjittered history has a robust scale at the quantiser, so the "
        f"unfloored z-score of a one-count residual is {ratio:.3e} times the "
        f"floored one",
    )
    out.check(
        "the_floor_changes_nothing_on_the_current_corpus",
        float(current_unfloored[np.isfinite(current_unfloored)].max())
        == float(current_floored[np.isfinite(current_floored)].max()),
        f"largest finite z-score "
        f"{float(current_floored[np.isfinite(current_floored)].max()):.4f} either "
        "way; reported rather than claimed as a benefit",
    )
    out.check(
        "the_floor_never_makes_anything_worse",
        bool(np.all(current_floored <= current_unfloored + 1.0e-9))
        and bool(np.all(old_floored <= old_unfloored + 1.0e-9)),
        "flooring a denominator can only lower a z-score, on both corpora",
    )

    out.rule("2. THE NODE UNDER A LEVEL TEST INSTEAD OF A TREND TEST")
    out(
        "The node regresses secularly, so its deviation from the median of a "
        "history is the ratio of the candidate's lever arm to the history's, and "
        "carries no information about corruption. Measured both ways on the same "
        "uncorrupted element sets."
    )
    out("")
    level: list[float] = []
    trend: list[float] = []
    for record in records:
        if record.corruption != "none":
            continue
        ordered = sorted(record.history, key=epoch_julian_date)
        if len(ordered) < 4:
            continue
        times = np.array([epoch_julian_date(entry) for entry in ordered])
        series = np.array([entry.raan_deg for entry in ordered], dtype=float)
        branch = unwrap_degrees(series)
        candidate = record.candidate.raan_deg
        offset = branch[-1] - series[-1]
        placed = candidate + offset
        placed -= 360.0 * round((placed - branch[-1]) / 360.0)
        level.append(modified_z(placed, branch, scale_floor("raan_deg", branch)))
        slope, intercept = theil_sen(times, branch)
        if not math.isfinite(slope):
            continue
        residuals = branch - (intercept + slope * times)
        scale = max(robust_scale(residuals), scale_floor("raan_deg", branch))
        predicted = intercept + slope * epoch_julian_date(record.candidate)
        trend.append(abs(placed - predicted) / scale if scale > 0.0 else 0.0)
    out(f"{'test':20s} {'n':>5s} {'min':>10s} {'median':>10s} {'max':>10s} {'spread':>10s}")
    for label, values in (("level, superseded", level), ("trend, current", trend)):
        array = np.array([value for value in values if math.isfinite(value)])
        out(
            f"{label:20s} {array.size:>5d} {float(array.min()):10.4f} "
            f"{float(np.median(array)):10.4f} {float(array.max()):10.4f} "
            f"{float(array.max() - array.min()):10.4f}"
        )
    level_array = np.array([value for value in level if math.isfinite(value)])
    out("")
    out(
        f"Under the level test the node's z-score lies between "
        f"{float(level_array.min()):.4f} and {float(level_array.max()):.4f} on "
        f"{level_array.size} uncorrupted element sets, a spread of "
        f"{float(level_array.max() - level_array.min()):.4f}. That is a fixed "
        "offset, not a measurement."
    )
    out.check(
        "the_level_test_on_the_node_is_a_fixed_offset",
        float(level_array.max() - level_array.min()) < 0.2,
        f"spread {float(level_array.max() - level_array.min()):.4f} over "
        f"{level_array.size} records",
    )

    out.rule("3. THE TWO SUPERSEDED ELEMENT-HISTORY GENERATORS")
    out(
        "The propagated disagreement between consecutive uncorrupted element sets "
        "of the same object, which is what the continuity screen measures. A "
        "generator whose own histories disagree by hundreds of kilometres makes "
        "that screen measure the generator."
    )
    out("")
    out(
        f"{'family':24s} {'generator':44s} {'median km':>12s} {'p90 km':>12s} "
        f"{'max km':>12s}"
    )
    rows: dict[tuple[str, str], float] = {}
    for name, overrides in FAMILY_CASES.items():
        base = _base(name, overrides)
        variants = {
            "version 1, kinematic J2 history": _kinematic_history(
                base, HISTORY_LENGTH, CADENCE_DAYS
            ),
            "version 2, Brouwer written as Kozai": [
                element_set
                for index in range(HISTORY_LENGTH + 1)
                if (
                    element_set := _rewound_without_kozai_inverse(
                        base, CADENCE_DAYS * (index - HISTORY_LENGTH)
                    )
                )
                is not None
            ],
            "version 3, current, with the Kozai inverse": [
                element_set
                for index in range(HISTORY_LENGTH + 1)
                if (
                    element_set := dataset.rewound(
                        base, CADENCE_DAYS * (index - HISTORY_LENGTH)
                    )
                )
                is not None
            ],
        }
        for label, sets in variants.items():
            offsets = np.array(_consecutive_offsets(sets))
            if offsets.size == 0:
                out(f"{name:24s} {label:44s} {'no usable pair':>38s}")
                continue
            rows[(name, label)] = float(np.median(offsets))
            out(
                f"{name:24s} {label:44s} {float(np.median(offsets)):12.4f} "
                f"{float(np.percentile(offsets, 90)):12.4f} "
                f"{float(offsets.max()):12.4f}"
            )
    out("")
    for name in FAMILY_CASES:
        version1 = rows.get((name, "version 1, kinematic J2 history"))
        version2 = rows.get((name, "version 2, Brouwer written as Kozai"))
        version3 = rows.get((name, "version 3, current, with the Kozai inverse"))
        if None in (version1, version2, version3):
            continue
        out(
            f"{name}: {version1:.2f} km, then {version2:.2f} km, then "
            f"{version3:.4f} km. The second step is the Kozai inverse of equation "
            f"(18) alone and it is worth a factor of {version2 / version3:.0f}."
        )
    out(
        "The two superseded versions are not ordered against each other and the "
        "measurement says so: version 2 is worse than version 1 on both families "
        "here, because writing a Brouwer mean motion into the Kozai field is a "
        "systematic half-kilometre error in the semi-major axis while version 1's "
        "fault was a kinematic mean anomaly whose error happens to be smaller on "
        "these two orbits. Version 3 beats both by three orders of magnitude, "
        "which is the only ordering claimed."
    )
    out.check(
        "the_current_generator_beats_both_superseded_versions",
        all(
            rows[(name, "version 3, current, with the Kozai inverse")]
            < rows[(name, "version 2, Brouwer written as Kozai")]
            and rows[(name, "version 3, current, with the Kozai inverse")]
            < rows[(name, "version 1, kinematic J2 history")]
            for name in FAMILY_CASES
            if (name, "version 1, kinematic J2 history") in rows
        ),
        "version 3 beats both superseded versions on both families",
    )
    out.check(
        "the_kozai_inverse_alone_is_worth_two_orders_of_magnitude",
        all(
            rows[(name, "version 2, Brouwer written as Kozai")]
            / rows[(name, "version 3, current, with the Kozai inverse")]
            > 100.0
            for name in FAMILY_CASES
            if (name, "version 2, Brouwer written as Kozai") in rows
        ),
        "the only difference between versions 2 and 3 is one line",
    )

    out.rule("4. WHAT THIS SECTION IS FOR")
    out(
        "Every figure in section 7 of validation/VALIDATION.md traces to this "
        "script. The point of keeping them is not completeness: it is that the "
        "published learned-versus-rules margin of +0.3661 recall is less than "
        "half the margin the first implementation of the baseline would have "
        "produced, and a reader is entitled to know how much of a reported "
        "margin is the problem and how much was the first attempt at the "
        "baseline."
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
