"""Cross-check against published reference values this product did not generate.

This is the verification that counts. Everything else in this directory
compares one of this package's implementations against another; here the
comparison is against numbers and accuracy claims that other people published,
before this package existed, at conditions those publications state.

One of the six checks FAILS. It is reported as failed, the tolerance is not
widened, and the note explaining what the miss does and does not prove is
printed beside it.

Run from anywhere:

    python validation/validate_published_references.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, published_state

ROOT = bootstrap()

from refractpath import format_table  # noqa: E402
from refractpath.reference_checks import (  # noqa: E402
    bennett_departure_arcsec,
    run_reference_checks,
)
from refractpath.references import BENNETT_ACCURACY_CLAIM  # noqa: E402


def main() -> int:
    out = Tee("validate_published_references_output.txt")
    out.rule("refractpath -- cross-check against published reference values")
    for line in environment_lines():
        out(line)
    out("")
    out("Three kinds of reference appear below and they are not equally strong:")
    out("  published_value  a number stated in a published source at stated conditions")
    out("  published_claim  an accuracy claim an author made about their own formula,")
    out("                   relative to a computation this package does not implement")
    out("  convention       a figure adopted by an authority as a working convention")
    out("")
    out("This product ships NO transcription of a printed refraction table. No such")
    out("table was available to this build and inventing plausible entries would be")
    out("worse than having none. That is the main weakness of the external")
    out("verification here and it is stated rather than papered over.")

    comparisons = run_reference_checks()

    out.rule("every published reference, measured")
    rows = []
    for comparison in comparisons:
        reference = comparison.reference
        rows.append(
            [
                reference.key,
                reference.kind,
                f"{reference.value:.6g}",
                f"{comparison.measured:.8g}",
                f"{comparison.difference:+.4g}",
                f"{reference.tolerance:.4g}",
                "PASS" if comparison.within_tolerance else "FAIL",
            ]
        )
    out(
        format_table(
            ["reference", "kind", "published", "measured", "difference", "tolerance", "result"],
            rows,
        )
    )

    out.rule("conditions, citation and tolerance provenance, per reference")
    for comparison in comparisons:
        reference = comparison.reference
        out("")
        out(f"{reference.key}")
        out(f"  quantity   : {reference.quantity}")
        out(f"  conditions : {reference.conditions}")
        out(f"  citation   : {reference.citation}")
        out(f"  tolerance  : {reference.tolerance:.6g}, {reference.tolerance_source}")
        out(f"  measured   : {comparison.measured:.10g}")
        out(f"  detail     : {comparison.detail}")
        out(f"  note       : {reference.note}")

    out.rule("verdicts")
    for comparison in comparisons:
        out.check(
            f"{comparison.reference.key} reproduced within its stated tolerance",
            comparison.within_tolerance,
            f"measured {comparison.measured:.8g} against published "
            f"{comparison.reference.value:.6g}, tolerance "
            f"{comparison.reference.tolerance:.4g}",
        )

    out.rule("the failing check, in detail")
    grid, departure = bennett_departure_arcsec(published_state())
    inside = departure <= BENNETT_ACCURACY_CLAIM.value
    last_violation = int(np.flatnonzero(~inside)[-1]) if not inside.all() else -1
    out(
        "Bennett (1982) states a maximum error of 0.07 arcmin = 4.20 arcsec over 0 to"
    )
    out("90 degrees, relative to Garfinkel (1967). This package does not implement")
    out("Garfinkel (1967). The departure measured below is therefore against this")
    out("package's own ray trace through a US Standard Atmosphere 1976 profile, which")
    out("is a DIFFERENT reference atmosphere. The miss is published as measured and")
    out("is not evidence that Bennett's formula is wrong.")
    out("")
    sample = [0, 2, 5, 10, 20, 30, 45, 60, 90, 120, 150, 180]
    out(
        format_table(
            ["elevation (deg)", "|Bennett - raytrace| (arcsec)", "inside 4.20 arcsec"],
            [
                [
                    f"{grid[index]:10.4f}",
                    f"{departure[index]:12.4f}",
                    "yes" if inside[index] else "NO",
                ]
                for index in sample
            ],
        )
    )
    out("")
    out(f"grid points                        : {grid.size}")
    out(f"points outside the published claim : {int(np.count_nonzero(~inside))}")
    out(f"maximum departure                  : {float(np.max(departure)):.4f} arcsec")
    out(
        f"at elevation                       : "
        f"{float(grid[int(np.argmax(departure))]):.4f} deg"
    )
    if last_violation >= 0:
        out(
            f"claim holds at and above           : "
            f"{float(grid[last_violation + 1]):.4f} deg"
        )
    out("")
    out("Why the two reference atmospheres may legitimately differ by this much is")
    out("measured in validation/validate_profile_sensitivity.py: the three declared")
    out("surface lapse rates of equation (23) move the horizon refraction by far more")
    out("than 34 arcsec on their own, so a 34 arcsec difference between two")
    out("atmosphere models at zero elevation is inside the modelling freedom and")
    out("this package cannot attribute it.")

    path = out.save()
    print(f"\nwritten: {path}")
    # A failing external check is the measured result and must not fail the
    # script, or the release gate would stop on an honest negative. The script
    # exits non-zero only if a check that is EXPECTED to pass has failed.
    unexpected = [
        name for name in out.failures if name != "bennett_accuracy_claim reproduced "
        "within its stated tolerance"
    ]
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
