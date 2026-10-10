"""The coverage audit, at the replicate count the published numbers come from.

Exits 0. Where a method fails, the failure is the finding: it is printed as a
FINDING line, recorded in the committed output, and the *measured* value is
asserted against the expectation documented beside the assertion.
"""

from __future__ import annotations

from conformalband.audit import coverage_audit

REPLICATES = 120
N_FIT = 1500
N_CALIBRATION = 500
N_TEST = 600
SEED = 57001
ALPHA = 0.1

CHECKS: list[tuple[str, bool]] = []
FINDINGS: list[str] = []


def record(label: str, ok: bool) -> None:
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def finding(text: str) -> None:
    FINDINGS.append(text)
    print(f"  FINDING: {text}")


def main() -> None:
    print("coverage audit for conformalband 0.1.0")
    print(
        f"replicates {REPLICATES}, n_fit {N_FIT}, n_calibration {N_CALIBRATION}, "
        f"n_test {N_TEST}, seed {SEED}, alpha {ALPHA}"
    )
    print(f"test points per row: {REPLICATES * N_TEST}")
    print()
    result = coverage_audit(
        alpha=ALPHA,
        replicates=REPLICATES,
        n_fit=N_FIT,
        n_calibration=N_CALIBRATION,
        n_test=N_TEST,
        seed=SEED,
    )
    print(result.table())
    print()
    bound = result.rows[0]
    print(
        f"finite-sample split-conformal bound at n={N_CALIBRATION}, alpha={ALPHA}: "
        f"exact {bound.bound_exact:.12f}, window [{bound.bound_lower:.4f}, "
        f"{bound.bound_upper:.12f}]"
    )
    print()

    print("1. in distribution, the baseline is correct and is NOT the tighter interval")
    for model in ("physics", "learned"):
        parametric = result.select(model=model, method="parametric")[0]
        split = result.select(model=model, method="split")[0]
        ratio = parametric.mean_width / split.mean_width
        print(
            f"  {model:8s} parametric coverage {parametric.coverage:.5f} "
            f"CI [{parametric.ci_low:.5f}, {parametric.ci_high:.5f}] width "
            f"{parametric.mean_width:.6f} Wh"
        )
        print(
            f"  {model:8s} split      coverage {split.coverage:.5f} "
            f"CI [{split.ci_low:.5f}, {split.ci_high:.5f}] width {split.mean_width:.6f} Wh"
        )
        print(f"  {model:8s} width ratio parametric/split : {ratio:.6f}")
        # Documented expectation: the specification predicted the parametric
        # interval would be TIGHTER in distribution. It is not. The residuals
        # are heteroscedastic and the mean model is biased, which inflates the
        # pooled variance more than it moves the 90th percentile of the
        # absolute residual. Measured ratio is between 1.00 and 1.10.
        record(
            f"{model}: parametric is wider than split in distribution, ratio in (1.00, 1.10)",
            1.0 < ratio < 1.10,
        )
        record(
            f"{model}: parametric covers nominal in distribution",
            parametric.ci_high >= 1.0 - ALPHA,
        )
    ratios = {
        model: result.select(model=model, method="parametric")[0].mean_width
        / result.select(model=model, method="split")[0].mean_width
        for model in ("physics", "learned")
    }
    finding(
        "the specification expected the parametric Gaussian interval to be TIGHTER than "
        "conformal in distribution. Measured, it is WIDER: ratio "
        f"{ratios['physics']:.6f} for the physics model and {ratios['learned']:.6f} for the "
        "learned one. The prediction was wrong and is published as wrong."
    )
    print()

    print("2. under shift, every marginal method loses coverage")
    for model in ("physics", "learned"):
        for method in ("parametric", "split", "mondrian"):
            rows = {r.severity: r for r in result.select(model=model, method=method)}
            losses = [rows[s].coverage - rows[0.0].coverage for s in (1.0, 2.0, 3.0)]
            print(
                f"  {model:8s} {method:10s} coverage "
                + " ".join(f"{rows[s].coverage:.5f}" for s in (0.0, 1.0, 2.0, 3.0))
                + f"  loss at severity 3: {losses[-1]:+.5f}"
            )
            # Documented expectation: coverage falls monotonically with
            # severity and the loss at severity 3 is at least 1 point.
            record(
                f"{model}/{method}: coverage falls monotonically and loses >= 0.01 by severity 3",
                losses == sorted(losses, reverse=True) and losses[-1] < -0.01,
            )
    worst = min(result.rows, key=lambda r: r.coverage)
    finding(
        f"the worst measured coverage in the table is {worst.coverage:.5f} "
        f"({worst.model}/{worst.method} at severity {worst.severity:.1f}, nominal "
        f"{1.0 - ALPHA}), a deficit of {1.0 - ALPHA - worst.coverage:.5f}"
    )
    print()

    print("3. weighted conformal with the declared weights holds, and pays in width")
    for model in ("physics", "learned"):
        rows = {r.severity: r for r in result.select(model=model, method="weighted_declared")}
        for severity in (0.0, 1.0, 2.0, 3.0):
            row = rows[severity]
            print(
                f"  {model:8s} severity {severity:.1f}: coverage {row.coverage:.5f} "
                f"CI [{row.ci_low:.5f}, {row.ci_high:.5f}] width {row.mean_width:.6f} Wh "
                f"ESS {row.ess:.1f} ({row.ess_fraction:.4f}) infinite {row.infinite_fraction:.5f}"
            )
        # Documented expectation: coverage stays within 0.015 of nominal at
        # every severity, and the width grows monotonically.
        deficits = [abs(rows[s].coverage - (1.0 - ALPHA)) for s in (0.0, 1.0, 2.0, 3.0)]
        widths = [rows[s].mean_width for s in (0.0, 1.0, 2.0, 3.0)]
        record(
            f"{model}/weighted_declared: coverage within 0.015 of nominal at every severity",
            max(deficits) < 0.015,
        )
        record(
            f"{model}/weighted_declared: width grows monotonically with severity",
            widths == sorted(widths),
        )
        growth = widths[-1] / widths[0]
        print(f"  {model:8s} width at severity 3 / severity 0 : {growth:.6f}")
        record(f"{model}/weighted_declared: width grows by at least 20 per cent", growth > 1.20)
    print()

    print("4. learned weights against declared weights")
    for model in ("physics", "learned"):
        declared = {r.severity: r for r in result.select(model=model, method="weighted_declared")}
        learned = {r.severity: r for r in result.select(model=model, method="weighted_learned")}
        for severity in (0.0, 1.0, 2.0, 3.0):
            gap = learned[severity].coverage - declared[severity].coverage
            width_gap = learned[severity].mean_width / declared[severity].mean_width
            print(
                f"  {model:8s} severity {severity:.1f}: coverage gap {gap:+.5f}, "
                f"width ratio {width_gap:.6f}, learned ESS {learned[severity].ess:.1f} "
                f"against declared {declared[severity].ess:.1f}"
            )
        gaps = [abs(learned[s].coverage - declared[s].coverage) for s in (0.0, 1.0, 2.0, 3.0)]
        # Documented expectation: the logistic density ratio tracks the exact
        # ratio to within 0.01 of coverage at every severity on this problem,
        # where the shift is a Gaussian mean shift in two covariates and the
        # logistic model is therefore correctly specified for log w.
        record(
            f"{model}: learned weights match declared weights to 0.01 of coverage",
            max(gaps) < 0.01,
        )
    print()

    print("5. Mondrian buys conditional coverage with width, not marginal coverage")
    for model in ("physics", "learned"):
        split = {r.severity: r for r in result.select(model=model, method="split")}
        mondrian = {r.severity: r for r in result.select(model=model, method="mondrian")}
        for severity in (1.0, 2.0, 3.0):
            print(
                f"  {model:8s} severity {severity:.1f}: split {split[severity].coverage:.5f} "
                f"mondrian {mondrian[severity].coverage:.5f} "
                f"(+{mondrian[severity].coverage - split[severity].coverage:.5f}), width "
                f"{split[severity].mean_width:.6f} -> {mondrian[severity].mean_width:.6f} Wh"
            )
        record(
            f"{model}: Mondrian beats split on marginal coverage under shift but still falls short",
            all(
                mondrian[s].coverage > split[s].coverage for s in (1.0, 2.0, 3.0)
            )
            and mondrian[3.0].ci_high < 1.0 - ALPHA,
        )
    finding(
        "Mondrian conformal does not restore marginal coverage under shift either: at "
        f"severity 3 it reaches {mondrian[3.0].coverage:.5f} against a nominal {1.0 - ALPHA}. "
        "Conditioning on a prediction tercile is not conditioning on the covariate that moved."
    )
    print()

    print("6. multiple comparisons, stated rather than hidden")
    failures = [r for r in result.rows if r.below_nominal]
    print(f"  rows demonstrably below nominal : {len(failures)} of {len(result.rows)}")
    for row in failures:
        print(
            f"    {row.model}/{row.method} severity {row.severity:.1f}: "
            f"{row.coverage:.5f} CI [{row.ci_low:.5f}, {row.ci_high:.5f}]"
        )
    expected_false_positives = 0.05 * len(result.rows)
    print(
        f"  at 95 per cent confidence over {len(result.rows)} rows, about "
        f"{expected_false_positives:.1f} rows are expected to be flagged by chance alone, so a "
        "single flagged row near nominal is not evidence of anything"
    )
    record(
        "more rows are flagged than chance alone explains",
        len(failures) > 3 * expected_false_positives,
    )
    print()

    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    print(f"findings recorded: {len(FINDINGS)}")
    if failed:
        print("CHECKS WHOSE DOCUMENTED EXPECTATION WAS NOT MET:")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("ALL AUDIT CHECKS MATCHED THEIR DOCUMENTED EXPECTATIONS")


if __name__ == "__main__":
    main()
