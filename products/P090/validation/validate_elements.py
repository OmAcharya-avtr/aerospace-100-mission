"""Element set parsing, formatting and the exact checksum blindness.

What this script measures
------------------------
1. The checksum of a published two-line element set, against the digits that
   element set carries in column 69. This is the only known answer in this
   repository that did not come out of its own generator.
2. A character-for-character formatting round trip of that element set.
3. A field-by-field round trip over every element set of the generated corpus.
4. Equation (13), the exact checksum detection fraction of three exhaustively
   enumerated corruption families, against a closed form derived by a
   congruence argument that performs no enumeration. The second is mathematics
   the first does not use, which is the point of running both.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import OBJECTS, Tee, bootstrap, declared_settings, environment_lines

bootstrap()

from tlecheck import dataset  # noqa: E402
from tlecheck.determinability import checksum_detection_fractions  # noqa: E402
from tlecheck.elements import checksum, format_tle, parse_tle  # noqa: E402

PUBLISHED_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
PUBLISHED_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
PUBLISHED_FIELDS = {
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
    out = Tee("validate_elements_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    out.rule("1. CHECKSUM AGAINST A PUBLISHED ELEMENT SET")
    out(
        "The element set below is distributed in the sgp4 project's own "
        "documentation for object 25544 at epoch 2019-343.69339541. It is the only "
        "external known answer in this repository."
    )
    out("")
    out(PUBLISHED_LINE1)
    out(PUBLISHED_LINE2)
    out("")
    for index, line in enumerate((PUBLISHED_LINE1, PUBLISHED_LINE2), start=1):
        stated = int(line[68])
        computed = checksum(line)
        out(f"line {index}: stated {stated}, computed {computed}")
        out.check(
            f"checksum_line{index}",
            stated == computed,
            f"stated {stated} equals computed {computed}",
        )

    out.rule("2. FORMATTING ROUND TRIP, CHARACTER FOR CHARACTER")
    line1, line2 = format_tle(**PUBLISHED_FIELDS)
    out(f"formatted line 1 == published : {line1 == PUBLISHED_LINE1}")
    out(f"formatted line 2 == published : {line2 == PUBLISHED_LINE2}")
    out.check(
        "format_reproduces_published_line1",
        line1 == PUBLISHED_LINE1,
        "all 69 columns identical",
    )
    out.check(
        "format_reproduces_published_line2",
        line2 == PUBLISHED_LINE2,
        "all 69 columns identical",
    )

    out.rule("3. FIELD ROUND TRIP OVER THE GENERATED CORPUS")
    records = dataset.split("test", objects=OBJECTS, history_length=12)
    sets = [
        element_set
        for record in records
        for element_set in (*record.history, record.reference)
    ]
    out(f"element sets examined : {len(sets)}")
    worst = {
        "inclination_deg": 0.0,
        "raan_deg": 0.0,
        "eccentricity": 0.0,
        "arg_perigee_deg": 0.0,
        "mean_anomaly_deg": 0.0,
        "mean_motion_rev_per_day": 0.0,
    }
    faults = 0
    bad_checksums = 0
    for element_set in sets:
        if element_set.faults:
            faults += 1
            continue
        for line in (element_set.line1, element_set.line2):
            if int(line[68]) != checksum(line):
                bad_checksums += 1
        reparsed = parse_tle(element_set.line1, element_set.line2)
        for name in worst:
            worst[name] = max(
                worst[name], abs(getattr(reparsed, name) - getattr(element_set, name))
            )
    out(f"element sets with a parse fault : {faults}")
    out(f"lines with a bad checksum       : {bad_checksums}")
    out("")
    out(f"{'field':28s} {'worst reparse difference':>26s}")
    for name, value in worst.items():
        out(f"{name:28s} {value:26.3e}")
    out.check("corpus_has_no_parse_faults", faults == 0, f"{faults} of {len(sets)}")
    out.check(
        "corpus_checksums_all_valid", bad_checksums == 0, f"{bad_checksums} bad lines"
    )
    out.check(
        "reparse_is_bit_identical",
        max(worst.values()) == 0.0,
        f"worst difference {max(worst.values()):.3e}",
    )

    out.rule("4. EQUATION (13): EXACT CHECKSUM BLINDNESS, TWO WAYS")
    out(
        "The enumeration loops over every digit column and every replacement "
        "digit. The closed form counts the same sets by a congruence argument: "
        "for one column the nine replacements give the nine distinct non-zero "
        "residues modulo 10, so a single substitution always changes the sum; a "
        "transposition permutes the digits and changes nothing; and for a pair of "
        "columns exactly one of the nine residues at the second column cancels "
        "each residue at the first, so 9 of every 81 pairs are undetected."
    )
    out("")
    closed = {
        "single_substitution": Fraction(1, 1),
        "transposition": Fraction(0, 1),
        "double_substitution": Fraction(72, 81),
    }
    out(
        f"{'family':24s} {'line':>5s} {'enumerated':>14s} {'closed form':>14s} "
        f"{'cases':>12s}"
    )
    for index, line in enumerate((PUBLISHED_LINE1, PUBLISHED_LINE2), start=1):
        fractions = checksum_detection_fractions(line)
        for family, expected in closed.items():
            measured = fractions[family]
            count = int(fractions[f"{family}_count"])
            out(
                f"{family:24s} {index:>5d} {measured:14.10f} "
                f"{float(expected):14.10f} {count:12d}"
            )
            out.check(
                f"{family}_line{index}_matches_closed_form",
                abs(measured - float(expected)) < 1.0e-12,
                f"enumerated {measured:.10f}, closed form {float(expected):.10f}",
            )

    fractions = checksum_detection_fractions(PUBLISHED_LINE1)
    out("")
    out(
        "So the fraction of two-digit corruptions that reach the propagator with a "
        f"valid checksum is exactly {1.0 - fractions['double_substitution']:.10f}, "
        "one in nine, and the fraction of digit transpositions that do is exactly "
        "1.0000000000. Neither depends on the digits of the line."
    )
    altered = PUBLISHED_LINE1[:20] + "999999999999" + PUBLISHED_LINE1[32:]
    other = checksum_detection_fractions(altered)
    out.check(
        "double_substitution_is_digit_independent",
        other["double_substitution"] == fractions["double_substitution"],
        "a line with twelve digits replaced gives the same fraction",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
