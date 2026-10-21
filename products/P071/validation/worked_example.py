"""The worked example quoted in README.md, run end to end.

Everything the README prints under "A worked example" is produced here, so the
two cannot drift. The script loads the shipped ``leo-crossing`` case, assesses
it by every method with a Monte Carlo reference, prints the report, and then
prints the two things the report exists for: the disagreement between the
methods, and the dilution diagnosis that says the small number is small for the
wrong reason.
"""

from __future__ import annotations

import warnings

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob import __version__  # noqa: E402
from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

CASE = "examples/cases/leo_crossing.yaml"
MONTE_CARLO_SAMPLES = 3_000_000
SEED = 20261021


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule(f"conjprob {__version__} worked example")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("input: a declared conjunction case")
    case = load_case(CASE)
    out(f"case file   : {CASE}")
    out(f"case name   : {case.name}")
    out(f"hard body   : {case.radius_m:.4f} m combined radius")
    for state in (case.primary, case.secondary):
        out(f"{state.name}:")
        out(f"  position [m]   : {', '.join(f'{v:.4f}' for v in state.position_m)}")
        out(f"  velocity [m/s] : {', '.join(f'{v:.6f}' for v in state.velocity_mps)}")
        out(
            "  covariance eigenvalues [m^2] : "
            + ", ".join(
                f"{v:.4f}"
                for v in sorted(
                    __import__("numpy").linalg.eigvalsh(state.covariance_m2)
                )
            )
        )
    out("")
    out("every value above is a declared input of a synthetic scenario; none was")
    out("identified from any vehicle, catalogue or operator product")

    out.rule("output: the report")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = assess_conjunction(
            case.primary,
            case.secondary,
            case.radius_m,
            monte_carlo_samples=MONTE_CARLO_SAMPLES,
            monte_carlo_seed=SEED,
        )
    for line in report.summary().splitlines():
        out(line)

    out.rule("the warning the library raised by itself")
    for warning in caught:
        out(f"{warning.category.__name__}: {warning.message}")
    if not any(issubclass(w.category, DilutionWarning) for w in caught):
        out("no DilutionWarning was raised")
        failures += 1

    out.rule("the two numbers this product exists to produce")
    out(
        f"1. disagreement: the four analytic methods span "
        f"{report.disagreement.max_absolute_spread:.6e} in absolute probability and "
        f"{report.disagreement.max_relative_spread:.6e} relative to the quadrature "
        f"reference, with {report.disagreement.worst_method} furthest out."
    )
    out(
        f"2. dilution: the reported Pc of {report.dilution.pc_at_unit_scale:.6e} is "
        f"{report.dilution.dilution_factor:.4f} times smaller than the "
        f"{report.dilution.pc_max_numerical:.6e} attainable at covariance scale "
        f"{report.dilution.scale_at_max_numerical:.4f}. The Mahalanobis distance is "
        f"{report.dilution.mahalanobis:.6f}, below sqrt(2), so a larger covariance "
        f"would make this number smaller rather than larger."
    )
    out("")
    out("neither number is an input to a manoeuvre decision and no output of this")
    out("package is.")

    out.rule("the same report for all three shipped cases, side by side")
    out("this is the table quoted in README.md under Validation evidence")
    out(
        f"{'case':<30s} {'R/sigma_eff':>12s} {'aspect':>8s} {'M':>9s} "
        f"{'quadrature Pc':>15s} {'foster':>11s} {'chan':>11s} {'alfano':>11s} "
        f"{'small_body':>11s} {'worst':>11s}"
    )
    for name in (
        "examples/cases/leo_crossing.yaml",
        "examples/cases/large_body_tight_covariance.yaml",
        "examples/cases/far_miss_tight_covariance.yaml",
    ):
        other = load_case(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DilutionWarning)
            row = assess_conjunction(other.primary, other.secondary, other.radius_m)
        errors = row.disagreement.relative_errors
        out(
            f"{other.name:<30s} {row.regime.radius_over_sigma:>12.4f} "
            f"{row.geometry.frame.aspect_ratio:>8.4f} {row.geometry.mahalanobis:>9.5f} "
            f"{row.quadrature_pc:>15.6e} {errors['foster']:>+11.4e} "
            f"{errors['chan']:>+11.4e} {errors['alfano']:>+11.4e} "
            f"{errors['small_body']:>+11.4e} {row.disagreement.worst_method:>11s}"
        )
    out("")
    out("the four columns after the quadrature Pc are signed relative errors against it")
    out("Chan is wrong by a factor of 2.2 on the far-miss case and the leading-order")
    out("approximation by a factor of 1.4 on the large-body case; both are regimes a")
    out("real conjunction can be in, and both are flagged in the report's regime notes")

    out.rule("consistency checks on the quoted numbers")
    checks = [
        (
            "Foster-Estes agrees with the independent 2-D quadrature",
            abs(report.disagreement.relative_errors["foster"]) <= 1e-8,
            f"{report.disagreement.relative_errors['foster']:+.3e} against 1e-8",
        ),
        (
            "the Monte Carlo interval contains the Foster-Estes value",
            "foster" in report.monte_carlo_consistent,
            f"CI [{report.monte_carlo.ci_low:.6e}, {report.monte_carlo.ci_high:.6e}]",
        ),
        (
            "the case is on the dilution branch",
            report.dilution.on_dilution_branch,
            f"dPc/dln(scale) = {report.dilution.log_scale_sensitivity:+.6e}",
        ),
        (
            "the small-hard-body approximation is the furthest-out method",
            report.disagreement.worst_method == "small_body",
            f"worst method {report.disagreement.worst_method}",
        ),
    ]
    for label, ok, detail in checks:
        out(f"{'PASS' if ok else 'FAIL'}  {label}  ({detail})")
        failures += 0 if ok else 1

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
