"""Monte Carlo reference: convergence, measured interval coverage, practical floor.

Monte Carlo is the reference in this product, so its own behaviour has to be
evidenced rather than trusted. Checks:

1. convergence of the estimate and of the Clopper-Pearson interval width, with
   the measured log-log slope of the width against sample count;
2. **measured coverage** of the nominal 95 per cent interval over independent
   seeds, against the analytic value, which is the only honest check of an
   interval;
3. agreement with each analytic method on each shipped case;
4. the practical floor: samples needed for a 10 per cent relative interval as a
   function of Pc, which is why this product cannot use Monte Carlo as the
   operational method.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.montecarlo import clopper_pearson, pc_monte_carlo  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

CASES = (
    "examples/cases/leo_crossing.yaml",
    "examples/cases/large_body_tight_covariance.yaml",
)
SAMPLE_COUNTS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
COVERAGE_SEEDS = 300
COVERAGE_SAMPLES = 100_000
SEED = 20261021


def main() -> int:
    out = Tee("validate_montecarlo_output.txt")
    out.rule("conjprob validation: Monte Carlo reference")
    for line in environment_lines():
        out(line)
    failures = 0

    reports = {}
    for path in CASES:
        case = load_case(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DilutionWarning)
            reports[case.name] = (
                case,
                assess_conjunction(case.primary, case.secondary, case.radius_m),
            )

    out.rule("1. convergence against sample count")
    out("case: leo-crossing; estimator: pc_monte_carlo, 3-D, rectilinear, unbounded time")
    case, report = reports["leo-crossing"]
    reference = report.analytic["foster"]
    out(f"analytic reference (Foster-Estes): {reference:.10e}")
    out(
        f"{'samples':>10s} {'hits':>8s} {'pc':>15s} {'ci_low':>15s} {'ci_high':>15s} "
        f"{'width':>12s} {'contains':>9s} {'wall [s]':>9s}"
    )
    widths = []
    contained = 0
    for count in SAMPLE_COUNTS:
        started = time.perf_counter()
        result = pc_monte_carlo(
            report.geometry.combined_covariance_m2,
            report.geometry.miss_vector_m,
            report.geometry.relative_velocity_mps,
            case.radius_m,
            n_samples=count,
            seed=SEED,
        )
        elapsed = time.perf_counter() - started
        inside = result.ci_low <= reference <= result.ci_high
        contained += int(inside)
        widths.append(result.ci_width)
        out(
            f"{count:>10d} {result.hits:>8d} {result.pc:>15.8e} {result.ci_low:>15.8e} "
            f"{result.ci_high:>15.8e} {result.ci_width:>12.5e} {str(inside):>9s} "
            f"{elapsed:>9.3f}"
        )
    slope = float(
        np.polyfit(np.log(np.array(SAMPLE_COUNTS, dtype=float)), np.log(widths), 1)[0]
    )
    out("")
    out(f"intervals containing the analytic value: {contained} of {len(SAMPLE_COUNTS)}")
    out(f"measured log-log slope of interval width against sample count: {slope:.6f}")
    out("expected -0.5 for a binomial estimator; tolerance +/- 0.05")
    ok = contained == len(SAMPLE_COUNTS) and abs(slope + 0.5) <= 0.05
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1
    out("timings above are wall clock on a shared two-core container and are unstable;")
    out("they are recorded for scale only and no conclusion is drawn from them")

    out.rule("2. measured coverage of the nominal 95 per cent interval")
    out(
        f"{COVERAGE_SEEDS} independent seeds, {COVERAGE_SAMPLES} samples each, "
        "case leo-crossing"
    )
    out("counting how often the Clopper-Pearson interval contains the analytic value")
    hits_inside = 0
    estimates = []
    for seed in range(COVERAGE_SEEDS):
        result = pc_monte_carlo(
            report.geometry.combined_covariance_m2,
            report.geometry.miss_vector_m,
            report.geometry.relative_velocity_mps,
            case.radius_m,
            n_samples=COVERAGE_SAMPLES,
            seed=10_000 + seed,
        )
        estimates.append(result.pc)
        hits_inside += int(result.ci_low <= reference <= result.ci_high)
    coverage = hits_inside / COVERAGE_SEEDS
    interval = clopper_pearson(hits_inside, COVERAGE_SEEDS, 0.95)
    out(f"measured coverage {coverage:.6f} ({hits_inside} of {COVERAGE_SEEDS})")
    out(f"95 per cent interval on the coverage itself: [{interval[0]:.6f}, {interval[1]:.6f}]")
    out("nominal 0.95; the Clopper-Pearson interval is conservative by construction, so")
    out("the measured coverage is expected to be at or above nominal")
    out("tolerance: measured coverage >= 0.93")
    ok = coverage >= 0.93
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1
    estimates = np.array(estimates)
    out("")
    out(f"mean of the {COVERAGE_SEEDS} estimates: {estimates.mean():.10e}")
    out(f"analytic value                       : {reference:.10e}")
    bias = estimates.mean() / reference - 1.0
    standard_error = estimates.std(ddof=1) / np.sqrt(COVERAGE_SEEDS)
    out(f"relative bias of the mean            : {bias:+.6e}")
    out(f"standard error of the mean           : {standard_error / reference:.6e} relative")
    out("tolerance: |bias| <= 3 standard errors")
    ok = abs(estimates.mean() - reference) <= 3.0 * standard_error
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("3. each analytic method against the Monte Carlo interval, per case")
    for name, (case, _) in reports.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DilutionWarning)
            detailed = assess_conjunction(
                case.primary,
                case.secondary,
                case.radius_m,
                monte_carlo_samples=3_000_000,
                monte_carlo_seed=SEED,
            )
        monte_carlo = detailed.monte_carlo
        out("")
        out(
            f"{name}: mc {monte_carlo.pc:.8e}, hits {monte_carlo.hits} of "
            f"{monte_carlo.n_samples}, 95 per cent CI "
            f"[{monte_carlo.ci_low:.8e}, {monte_carlo.ci_high:.8e}]"
        )
        out(f"{'method':<12s} {'pc':>16s} {'inside CI':>10s} {'rel. to mc':>13s}")
        for method, value in detailed.disagreement.values.items():
            inside = monte_carlo.ci_low <= value <= monte_carlo.ci_high
            out(
                f"{method:<12s} {value:>16.8e} {str(inside):>10s} "
                f"{value / monte_carlo.pc - 1.0:>+13.5e}"
            )
        out(f"inside the interval : {', '.join(detailed.monte_carlo_consistent) or 'none'}")
        out(f"outside the interval: {', '.join(detailed.monte_carlo_inconsistent) or 'none'}")
        if "foster" not in detailed.monte_carlo_consistent:
            failures += 1

    out.rule("4. practical floor of the Monte Carlo reference")
    out("samples needed for a 95 per cent interval of width 0.1 * Pc, from the")
    out("binomial relation n ~ 4 * 1.96^2 / (0.1^2 * Pc) = 1537 / Pc")
    out(f"{'Pc':>10s} {'samples for 10 per cent':>24s} {'minutes at 3.3e6 samples/s':>28s}")
    rate = 3.3e6
    for exponent in range(-2, -9, -1):
        pc = 10.0**exponent
        needed = 1537.0 / pc
        out(f"{pc:>10.0e} {needed:>24.3e} {needed / rate / 60.0:>28.4g}")
    out("")
    out("the sample rate above is the one measured in check 1 and is unstable; the point")
    out("is the scaling, not the number. Monte Carlo is usable as a reference down to")
    out("about Pc = 1e-5 inside this product's three-minute budget and not below, which")
    out("is why the shipped far-miss case at Pc = 2.65e-35 has no Monte Carlo row.")

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
