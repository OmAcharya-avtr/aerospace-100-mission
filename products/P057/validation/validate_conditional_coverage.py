"""Conditional coverage: what a marginal band hides and Mondrian exposes.

Exits 0. The failure of marginal split conformal in the top tercile IS the
finding here, and it is asserted against a documented expectation.
"""

from __future__ import annotations

from conformalband.audit import stratified_coverage

REPLICATES = 30
N_FIT = 1500
N_CALIBRATION = 500
N_TEST = 1000
SEED = 57031
ALPHA = 0.1
SEVERITIES = (0.0, 2.0)

CHECKS: list[tuple[str, bool]] = []
FINDINGS: list[str] = []


def record(label: str, ok: bool) -> None:
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def finding(text: str) -> None:
    FINDINGS.append(text)
    print(f"  FINDING: {text}")


def main() -> None:
    print("conditional coverage by tercile of predicted energy, conformalband 0.1.0")
    print(
        f"replicates {REPLICATES}, n_fit {N_FIT}, n_calibration {N_CALIBRATION}, "
        f"n_test {N_TEST}, seed {SEED}, alpha {ALPHA}, model learned"
    )
    print(
        "the tercile edges are declared on the CALIBRATION predictions and applied unchanged "
        "to the test predictions, so under shift the test terciles are not equally populated; "
        "that imbalance is itself a symptom of the shift and is reported"
    )
    print()
    for severity in SEVERITIES:
        tally = stratified_coverage(
            alpha=ALPHA,
            replicates=REPLICATES,
            n_fit=N_FIT,
            n_calibration=N_CALIBRATION,
            n_test=N_TEST,
            seed=SEED,
            severity=severity,
            model="learned",
        )
        print(f"severity {severity:.1f}")
        header = (
            f"{'method':10s} {'tercile':>8s} {'coverage':>9s} {'width_Wh':>9s} {'points':>9s}"
        )
        print(header)
        print("-" * len(header))
        for method in ("split", "mondrian"):
            for index in sorted(tally[method]):
                coverage, width, _, count = tally[method][index]
                print(f"{method:10s} {index + 1:8d} {coverage:9.5f} {width:9.5f} {count:9d}")
        spreads = {}
        for method in ("split", "mondrian"):
            values = [tally[method][b][0] for b in sorted(tally[method])]
            spreads[method] = max(values) - min(values)
            print(
                f"  {method:10s} coverage spread across terciles {spreads[method]:.5f} "
                f"(min {min(values):.5f}, max {max(values):.5f})"
            )
        widths = {
            method: [tally[method][b][1] for b in sorted(tally[method])]
            for method in ("split", "mondrian")
        }
        print(
            f"  mondrian width across terciles: {widths['mondrian'][0]:.5f} -> "
            f"{widths['mondrian'][-1]:.5f} Wh, a factor of "
            f"{widths['mondrian'][-1] / widths['mondrian'][0]:.5f}"
        )
        # Documented expectation at both severities: the Mondrian spread is at
        # most half the marginal spread, and the marginal band under-covers in
        # the top tercile relative to nominal.
        record(
            f"severity {severity:.1f}: Mondrian spread is at most half the marginal spread",
            spreads["mondrian"] <= 0.5 * spreads["split"],
        )
        top = max(sorted(tally["split"]))
        record(
            f"severity {severity:.1f}: marginal split under-covers in the top tercile",
            tally["split"][top][0] < 1.0 - ALPHA,
        )
        record(
            f"severity {severity:.1f}: Mondrian widths increase with the tercile",
            widths["mondrian"] == sorted(widths["mondrian"]),
        )
        finding(
            f"severity {severity:.1f}: marginal split conformal covers "
            f"{tally['split'][min(tally['split'])][0]:.5f} in the lowest tercile and "
            f"{tally['split'][top][0]:.5f} in the highest, against a nominal {1.0 - ALPHA}. "
            "The marginal guarantee is met on average and violated where it matters, which "
            "is the heaviest, windiest flights."
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
    print("ALL CONDITIONAL-COVERAGE CHECKS MATCHED THEIR DOCUMENTED EXPECTATIONS")


if __name__ == "__main__":
    main()
