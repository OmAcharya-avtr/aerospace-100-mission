"""The worked example quoted in README.md, run so the quoted output is real.

Every line of output beneath the worked example in ``README.md`` is produced by
this script. It is re-run as the last step before hand-off and the README text
is replaced with what it printed, because two products of this mission caught
their own wrong README numbers that way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import Tee, bootstrap, declared_settings, environment_lines

bootstrap()

from tlecheck import dataset, screen  # noqa: E402
from tlecheck.consequence import measure  # noqa: E402
from tlecheck.elements import format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402

FIELDS = {
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


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    out.rule("THE WORKED EXAMPLE OF README.md")

    # The element set an operator has, and the twelve that came before it.
    truth = parse_tle(*format_tle(**FIELDS), source="catalogue")
    rng = np.random.default_rng(7)
    history = [
        element_set
        for index in range(12)
        if (
            element_set := dataset.rewound(
                truth,
                -1.5 * (12 - index),
                source="catalogue",
                rng=rng,
                element_number=100 + index,
            )
        )
        is not None
    ]

    # The same element set with the sign of its drag term flipped, and the
    # checksum recomputed so the line is internally consistent.
    flipped = parse_tle(*format_tle(**{**FIELDS, "bstar": -FIELDS["bstar"]}), source="catalogue")

    use_jd = epoch_julian_date(truth) + 1.0
    verdict = screen(flipped, history=history, reference_jd=use_jd)

    out("")
    out("the element set as received:")
    out(flipped.line1)
    out(flipped.line2)
    out("")
    out(f"checksum line 1 stated {flipped.checksum_line1_stated}, computed "
        f"{flipped.checksum_line1_computed} -- the line is internally consistent")
    out(f"integrity intact            : {verdict.integrity_ok}")
    out(f"predicted consequence class : {verdict.predicted_class}")
    out(f"equation (15) risk score    : {verdict.silent_risk_score:.6f}")
    out(f"problems found              : {len(verdict.problems)}")
    for problem in verdict.problems:
        out(f"  [{problem.screen}] {problem.name} -> {problem.predicted_class}")
    out("")
    out("what this screen reports as not determinable, as returned:")
    for limit in verdict.determinability:
        if limit.determinable:
            continue
        out(f"  {limit.name} ({limit.basis})")
    out("")
    out("two of the computed limits, with their numbers:")
    for limit in verdict.determinability:
        if limit.name in {
            "checksum_blindness_to_double_substitution",
            "continuity_resolution_km",
        }:
            out(f"  {limit.name} = {limit.value:.6f}")

    consequence = measure(flipped, truth, use_jd, horizon_days=7.0, samples=49)
    out("")
    out("measured consequence of using it, against the uncorrupted element set:")
    out(f"  class            {consequence.consequence_class}")
    out(f"  max error        {consequence.max_error_km:.4f} km over 7 days")
    out(f"  rms error        {consequence.rms_error_km:.4f} km")
    out(f"  threshold        {consequence.threshold_km:g} km (declared policy)")

    out.check(
        "the_worked_example_flags_the_sign_flip",
        verdict.predicted_class == "silently_wrong",
        f"predicted {verdict.predicted_class!r}",
    )
    out.check(
        "the_worked_example_keeps_a_valid_checksum",
        verdict.integrity_ok,
        "the corruption is invisible to the checksum, which is the point",
    )
    out.check(
        "the_worked_example_consequence_is_measured",
        consequence.max_error_km > 1.0,
        f"{consequence.max_error_km:.4f} km, propagated with sgp4",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
