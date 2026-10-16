"""Hand-calculated known answers on the shipped cases.

Every expected number in this script was computed by hand from the YAML
before the code was run on it, and the arithmetic is printed beside the
result so the reader can check it without opening a calculator. Comparisons
use an absolute tolerance on both sides.
"""

from __future__ import annotations

import math

from _support import CASES, Emitter

from budgetroll import load, rollup, run_checks, severity_counts

TOL = 1.0e-9


def main() -> int:
    """Check every hand-computed value and report."""
    out = Emitter(__file__)
    out("budgetroll validation: hand-calculated known answers")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use.")
    out()

    out("1. clean_mass.yaml, the negative control")
    out("   nominal    2.00 + 1.50 + 0.80 + 1.20                    = 5.500 kg")
    out("   allowance  5 % of each, all four 'weighed'              = 0.275 kg")
    out("   predicted                                               = 5.775 kg")
    out("   1 sigma    sqrt(0.0100 + 0.0064 + 0.0025 + 0.0036)")
    out("              = sqrt(0.0225)                               = 0.150 kg")
    out("   margin     10.000 - 5.775                               = 4.225 kg")
    out("   margin@1s  10.000 - 5.925                               = 4.075 kg")
    clean = load(CASES / "clean_mass.yaml")
    root = rollup(clean).root
    for label, value, expected in (
        ("nominal", root.nominal, 5.500),
        ("allowance", root.allowance, 0.275),
        ("predicted", root.predicted, 5.775),
        ("1 sigma", root.uncertainty, 0.150),
        ("margin", root.margin_absolute, 4.225),
        ("margin at 1 sigma", root.margin_at_k, 4.075),
    ):
        out.check(
            f"clean_mass {label} is {expected} kg",
            out.close(value, expected, TOL),
            f"computed {value:.12f}",
        )
    out.check(
        "clean_mass raises no finding of any severity",
        run_checks(clean) == [],
        str(severity_counts(run_checks(clean))),
    )
    out()

    out("2. planted_findings.yaml, the positive control")
    out("   structure nominal 6.0 + 3.0 + 2.0                       = 11.00 kg")
    out("   structure allowance 0.30 + 0.30 + 0.40                  =  1.00 kg")
    out("   structure predicted                                     = 12.00 kg")
    out("   structure allocation 11.50, so the overrun is              0.50 kg")
    out("   structure rss sqrt(4 + 1 + 9) = sqrt(14)                =  3.741657 kg")
    out("   structure exact var = 14 + 2*0.9*2*1 + 2*0.9*2*3")
    out("                       = 14 + 3.6 + 10.8 = 28.4")
    out("                       sqrt(28.4)                          =  5.329165 kg")
    out("   root nominal 11.0 + 3.5 + 1.8 + 1.8                     = 18.10 kg")
    out("   root allowance 1.00 + 0.60 + 0.18 + 0.13                =  1.91 kg")
    out("   root predicted                                          = 20.01 kg")
    out("   root linear 3.741657 + 0.5 + 0.4 + 0.5                  =  5.141657 kg")
    out("   root exact var 28.4 + 0.25 + 0 + 0.16 + 0.04 + 0.09")
    out("                  = 28.94, sqrt                            =  5.379591 kg")
    planted = load(CASES / "planted_findings.yaml")
    roll = rollup(planted)
    structure = roll.by_id("structure")
    for label, value, expected in (
        ("structure nominal", structure.nominal, 11.00),
        ("structure allowance", structure.allowance, 1.00),
        ("structure predicted", structure.predicted, 12.00),
        ("structure margin", structure.margin_absolute, -0.50),
        ("structure declared 1 sigma", structure.uncertainty, math.sqrt(14.0)),
        ("structure exact 1 sigma", structure.exact_uncertainty, math.sqrt(28.4)),
        ("root nominal", roll.root.nominal, 18.10),
        ("root allowance", roll.root.allowance, 1.91),
        ("root predicted", roll.root.predicted, 20.01),
        ("root declared 1 sigma", roll.root.uncertainty, math.sqrt(14.0) + 1.4),
        ("root exact 1 sigma", roll.root.exact_uncertainty, math.sqrt(28.94)),
    ):
        out.check(
            f"planted {label} is {expected:.6f} kg",
            out.close(value, expected, TOL),
            f"computed {value:.12f}",
        )
    out()

    out("3. planted_findings.yaml raises every check, with these counts")
    counts: dict[str, int] = {}
    for finding in run_checks(planted):
        counts[finding.code] = counts.get(finding.code, 0) + 1
    expected_counts = {
        "BR001": 1,
        "BR002": 3,
        "BR003": 1,
        "BR004": 1,
        "BR005": 1,
        "BR006": 1,
        "BR007": 1,
        "BR008": 1,
        "BR009": 1,
        "BR010": 1,
    }
    for code in sorted(expected_counts):
        out(f"   {code}  expected {expected_counts[code]}  got {counts.get(code, 0)}")
    out.check(
        "the per-code finding counts are exactly as declared",
        counts == expected_counts,
        f"{counts}",
    )
    out.check(
        "the severity counts are 6 error, 4 warning, 2 info",
        severity_counts(run_checks(planted)) == {"error": 6, "warning": 4, "info": 2},
        str(severity_counts(run_checks(planted))),
    )
    out()

    out("4. cubesat_power.yaml")
    out("   nominal 1.20+0.90+1.10+0.35+0.55+0.40+1.60+0.45+0.60    = 7.1500 W")
    out("   allowance 0.2400+0.0900+0.0550+0.0175+0.0275")
    out("             +0.0200+0.1600+0.0225+0.1200                  = 0.7525 W")
    out("   predicted                                               = 7.9025 W")
    out("   margin    8.4000 - 7.9025                               = 0.4975 W")
    power = load(CASES / "cubesat_power.yaml")
    root = rollup(power).root
    for label, value, expected in (
        ("nominal", root.nominal, 7.1500),
        ("allowance", root.allowance, 0.7525),
        ("predicted", root.predicted, 7.9025),
        ("margin", root.margin_absolute, 0.4975),
    ):
        out.check(
            f"cubesat_power {label} is {expected} W",
            out.close(value, expected, TOL),
            f"computed {value:.12f}",
        )
    out()

    out("5. cubesat_link.yaml")
    out("   nominal 1.50+0.50+1.00+0.30+0.50+0.40+0.80+0.70+0.60    = 6.3000 dB")
    out("   allowance 0.0750+0.0500+0.2000+0.0150+0.0250")
    out("             +0.0200+0.0800+0.0700+0.0300                  = 0.5650 dB")
    out("   predicted                                               = 6.8650 dB")
    out("   margin    17.6655 - 6.8650                              = 10.8005 dB")
    link = load(CASES / "cubesat_link.yaml")
    root = rollup(link).root
    for label, value, expected in (
        ("nominal", root.nominal, 6.3000),
        ("allowance", root.allowance, 0.5650),
        ("predicted", root.predicted, 6.8650),
        ("margin", root.margin_absolute, 10.8005),
    ):
        out.check(
            f"cubesat_link {label} is {expected} dB",
            out.close(value, expected, TOL),
            f"computed {value:.12f}",
        )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
