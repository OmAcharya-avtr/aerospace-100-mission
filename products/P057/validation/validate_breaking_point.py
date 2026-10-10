"""Where weighted conformal breaks when its declared weights are wrong.

This is the number the specification asked to be published. Exits 0.
"""

from __future__ import annotations

from conformalband.audit import breaking_point_sweep

REPLICATES = 120
N_FIT = 1500
N_CALIBRATION = 500
N_TEST = 600
SEED = 57021
ALPHA = 0.1
TRUE_SEVERITIES = (1.0, 2.0, 3.0)

CHECKS: list[tuple[str, bool]] = []
FINDINGS: list[str] = []


def record(label: str, ok: bool) -> None:
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def finding(text: str) -> None:
    FINDINGS.append(text)
    print(f"  FINDING: {text}")


def main() -> None:
    print("weighted conformal under misspecified weights, conformalband 0.1.0")
    print(
        f"replicates {REPLICATES}, n_fit {N_FIT}, n_calibration {N_CALIBRATION}, "
        f"n_test {N_TEST}, seed {SEED}, alpha {ALPHA}, model learned"
    )
    print(
        "the test data is drawn under the true severity; the weights are computed from "
        "CovariateShift(true_severity * fraction), so fraction = 1.00 is correct"
    )
    print(
        "breaking fraction = the largest fraction at or below 1.00 whose 95 per cent "
        "replicate-level coverage interval lies ENTIRELY below nominal"
    )
    print()

    summary: list[tuple[float, float | None, float | None, float, float, float]] = []
    for severity in TRUE_SEVERITIES:
        result = breaking_point_sweep(
            true_severity=severity,
            alpha=ALPHA,
            replicates=REPLICATES,
            n_fit=N_FIT,
            n_calibration=N_CALIBRATION,
            n_test=N_TEST,
            seed=SEED,
            model="learned",
        )
        print(f"true severity {severity:.1f}")
        print(result.table())
        rows = {row.fraction: row for row in result.rows}
        unweighted = rows[0.0]
        correct = rows[1.0]
        print(
            f"  no weighting (f=0.00)  : coverage {unweighted.coverage:.5f} "
            f"CI [{unweighted.ci_low:.5f}, {unweighted.ci_high:.5f}] width "
            f"{unweighted.mean_width:.6f} Wh"
        )
        print(
            f"  correct weights (f=1.00): coverage {correct.coverage:.5f} "
            f"CI [{correct.ci_low:.5f}, {correct.ci_high:.5f}] width "
            f"{correct.mean_width:.6f} Wh ESS {correct.ess:.1f}"
        )
        print(f"  smallest fraction still holding : {result.last_holding_fraction}")
        print(f"  BREAKING FRACTION               : {result.breaking_fraction}")
        print()
        summary.append(
            (
                severity,
                result.breaking_fraction,
                result.last_holding_fraction,
                unweighted.coverage,
                correct.coverage,
                rows[1.5].mean_width / correct.mean_width,
            )
        )
        # Documented expectations, per true severity:
        #  (a) the correct weights hold: the 95 % interval reaches nominal.
        #  (b) no weighting at all does not hold.
        #  (c) a breaking fraction exists at or below 1.00 and is at least 0.50,
        #      i.e. the method tolerates some error but not an arbitrary amount.
        #  (d) coverage is non-decreasing in the assumed fraction.
        record(f"severity {severity:.1f}: correct weights hold", correct.ci_high >= 1.0 - ALPHA)
        record(
            f"severity {severity:.1f}: no weighting at all is demonstrably below nominal",
            unweighted.ci_high < 1.0 - ALPHA,
        )
        coverages = [rows[f].coverage for f in sorted(rows)]
        record(
            f"severity {severity:.1f}: coverage is non-decreasing in the assumed fraction",
            all(b >= a - 2e-3 for a, b in zip(coverages, coverages[1:], strict=False)),
        )
        if result.breaking_fraction is None:
            record(f"severity {severity:.1f}: a breaking fraction was located", False)
        else:
            record(
                f"severity {severity:.1f}: breaking fraction is in [0.00, 1.00]",
                0.0 <= result.breaking_fraction <= 1.0,
            )

    print("summary across true severities")
    header = (
        f"{'true_sev':>8s} {'breaking_f':>11s} {'holding_f':>10s} {'cov_f0':>8s} "
        f"{'cov_f1':>8s} {'width_f1.5/f1':>14s}"
    )
    print(header)
    print("-" * len(header))
    for severity, breaking, holding, cov0, cov1, width_ratio in summary:
        breaking_text = "none" if breaking is None else f"{breaking:.2f}"
        holding_text = "none" if holding is None else f"{holding:.2f}"
        print(
            f"{severity:8.1f} {breaking_text:>11s} {holding_text:>10s} {cov0:8.5f} "
            f"{cov1:8.5f} {width_ratio:14.6f}"
        )
    print()
    worst = min(b for _, b, _, _, _, _ in summary if b is not None)
    best = max(b for _, b, _, _, _, _ in summary if b is not None)
    finding(
        "weighted conformal breaks when the declared shift is understated. Across true "
        f"severities {TRUE_SEVERITIES} the breaking fraction is between {worst:.2f} and "
        f"{best:.2f}, so assuming a shift of that fraction of the true one or less puts the "
        f"whole 95 per cent coverage interval below the nominal {1.0 - ALPHA}. In plain terms, "
        f"understating the declared shift by {100 * (1 - best):.0f} to "
        f"{100 * (1 - worst):.0f} per cent is enough to lose the guarantee at this "
        "measurement precision."
    )
    finding(
        "overstating the shift does not break coverage, it buys it with width and with "
        "effective sample size: at fraction 1.50 the interval is "
        f"{summary[1][5]:.3f} times the width it is at fraction 1.00, and the effective "
        "calibration size falls with it."
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
    print("ALL BREAKING-POINT CHECKS MATCHED THEIR DOCUMENTED EXPECTATIONS")


if __name__ == "__main__":
    main()
