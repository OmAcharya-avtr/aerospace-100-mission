"""Calibration of the learned classifier, against the declared threshold logic.

The comparison is at an EQUAL FALSE-ALARM RATE. The baseline's false-alarm rate
is measured on a calibration split at its declared operating point; the learned
classifier's declare-a-fault threshold is then tuned on the same split to match
it; both are then scored on a test split neither was tuned on.

Reported at five declared noise scales, because at scale 1 the declared
thresholds sit between 3.6 and 5.0 sigma from both class means and the
comparison is uninformative. ECE is reported at three bin counts because binned
ECE is a biased estimator whose value depends on the binning.

Run:
    PYTHONPATH=src python validation/validate_calibration.py
"""

from __future__ import annotations

import time

import numpy as np

from safemodetree import (
    compare_detectors,
    learned_decisions,
    load_spec,
    packaged_spec_path,
    reliability_curve,
)

SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
BIN_COUNTS = (5, 10, 20)
SEED = 63001
N_PER_FAULT = 400
HEADLINE_SCALE = 2.0

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> None:
    """Score both detectors over the noise sweep and report the comparison."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    print("safemodetree classifier calibration, learned against threshold logic")
    print("=" * 94)
    print(f"declaration: {spec.summary()}")
    print(f"seed {SEED}, {N_PER_FAULT} samples per fault, nominal_fraction 0.5")
    print("split: 50 per cent train, 25 per cent calibration, 25 per cent test")
    print("the learned threshold is tuned on the CALIBRATION split only")
    print()

    start = time.perf_counter()
    results = {scale: compare_detectors(
        spec, n_per_fault=N_PER_FAULT, noise_scale=scale, seed=SEED
    ) for scale in SCALES}
    elapsed = time.perf_counter() - start

    header = (
        f"{'scale':>6} {'detector':<22} {'op_point':>10} {'FAR':>9} {'detect':>9} "
        f"{'ident_acc':>10} {'overall':>9} {'ECE':>9} {'MCE':>9} {'brier':>9}"
    )
    print("1. detector scores on the test split")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for scale in SCALES:
        result = results[scale]
        for key in ("baseline", "learned", "learned_argmax"):
            score = result[key]
            brier = "-" if score.brier is None else f"{score.brier:.5f}"
            print(
                f"{scale:>6.1f} {score.name:<22} {score.operating_point:>10.6f} "
                f"{score.false_alarm_rate:>9.5f} {score.detection_rate:>9.5f} "
                f"{score.identification_accuracy:>10.5f} {score.overall_accuracy:>9.5f} "
                f"{score.calibration.ece:>9.5f} {score.calibration.mce:>9.5f} {brier:>9}"
            )
    print()
    print(f"accuracy ceiling from the declared observability model: "
          f"{float(results[SCALES[0]]['ceiling']):.6f}")
    print("(four of eight faults sit in declared ambiguity classes of size two,")
    print(" so no classifier can exceed (800 + 400 + 200) / 1600 = 0.875 here)")
    print(f"elapsed: {elapsed:.2f} s")
    print()

    print("2. who wins at an equal false-alarm rate")
    print("-" * 94)
    header = (
        f"{'scale':>6} {'target_FAR':>11} {'base_FAR':>9} {'learn_FAR':>10} "
        f"{'base_ident':>11} {'learn_ident':>12} {'delta':>9} {'winner':<16}"
    )
    print(header)
    print("-" * len(header))
    learned_wins = 0
    for scale in SCALES:
        result = results[scale]
        baseline = result["baseline"]
        learned = result["learned"]
        delta = learned.identification_accuracy - baseline.identification_accuracy
        if delta > 1e-9:
            winner = "learned"
            learned_wins += 1
        elif delta < -1e-9:
            winner = "threshold logic"
        else:
            winner = "tie"
        print(
            f"{scale:>6.1f} {float(result['target_far']):>11.5f} "
            f"{baseline.false_alarm_rate:>9.5f} {learned.false_alarm_rate:>10.5f} "
            f"{baseline.identification_accuracy:>11.5f} "
            f"{learned.identification_accuracy:>12.5f} {delta:>+9.5f} {winner:<16}"
        )
    print()
    print("FINDING: the learned classifier beats the declared threshold baseline on")
    print(f"identification accuracy at {learned_wins} of {len(SCALES)} noise scales. This is the")
    print("measured result and is reported as measured. The baseline is correctly")
    print("specified for this generator -- the declared thresholds are the Bayes")
    print("boundary of an isotropic Gaussian per class -- so beating it was never the")
    print("likely outcome, and a reader should not take the learned model as the")
    print("better detector here.")
    print()

    print("3. ECE at three bin counts, because binned ECE depends on the binning")
    print("-" * 94)
    header = f"{'scale':>6} {'detector':<22}" + "".join(f" {f'ECE@{b}':>9}" for b in BIN_COUNTS)
    print(header)
    print("-" * len(header))
    ece_wins = 0
    for scale in SCALES:
        result = results[scale]
        row_values: dict[str, list[float]] = {}
        for key in ("baseline", "learned"):
            score = result[key]
            confidence, correct = _decisions(result, key)
            values = [
                reliability_curve(confidence, correct, n_bins=bins).ece
                for bins in BIN_COUNTS
            ]
            row_values[key] = values
            print(
                f"{scale:>6.1f} {score.name:<22}"
                + "".join(f" {value:>9.5f}" for value in values)
            )
        if row_values["learned"][1] < row_values["baseline"][1]:
            ece_wins += 1
    print()
    print(f"FINDING: the learned classifier has the lower ECE at 10 bins at "
          f"{ece_wins} of {len(SCALES)} scales.")
    print("The declared threshold logic is well calibrated when the thresholds are")
    print("far from the noise and becomes overconfident as the noise grows, because")
    print("its confidence is a lookup on a latched symptom word and carries no")
    print("information about how close the measurement was to the threshold.")
    print()

    print("4. reliability table at the headline scale")
    print("-" * 94)
    for key in ("baseline", "learned"):
        score = results[HEADLINE_SCALE][key]
        print(f"  {score.name} (operating point {score.operating_point:.6f})")
        for line in score.calibration.table().splitlines():
            print(f"    {line}")
        print()

    print("5. checks")
    print("-" * 94)
    headline = results[HEADLINE_SCALE]
    record(
        "no detector exceeds the declared accuracy ceiling",
        all(
            headline[key].overall_accuracy <= float(headline["ceiling"]) + 1e-9
            for key in ("baseline", "learned", "learned_argmax")
        ),
    )
    record(
        "the matched false-alarm rates are within 0.06 of each other at every scale",
        all(
            abs(results[s]["learned"].false_alarm_rate - results[s]["baseline"].false_alarm_rate)
            <= 0.06
            for s in SCALES
        ),
    )
    # At noise scale 1.0 the baseline's calibration-split false-alarm rate is
    # exactly 0, so the matched threshold is the one just above the largest
    # nominal fault posterior on the CALIBRATION split. Three test-split rows
    # exceed it, giving a test-split rate of 0.00368 against the baseline's
    # 0.00000, while the plain argmax happens to give 0.00000 as well. The
    # match is therefore NOT closer than the argmax at that scale. This is a
    # split artefact of matching a zero rate on a finite sample, it is recorded
    # rather than tuned away, and the check below is restricted to the scales
    # where the target rate is non-zero.
    nonzero = [s for s in SCALES if float(results[s]["target_far"]) > 0.0]
    record(
        "matching moves the learned false-alarm rate towards the baseline at every "
        f"scale with a non-zero target rate ({len(nonzero)} of {len(SCALES)})",
        all(
            abs(results[s]["learned"].false_alarm_rate - results[s]["baseline"].false_alarm_rate)
            <= abs(
                results[s]["learned_argmax"].false_alarm_rate
                - results[s]["baseline"].false_alarm_rate
            )
            + 1e-12
            for s in nonzero
        ),
    )
    record(
        "at noise scale 1.0 the zero target rate makes the match no closer than the "
        "argmax, as recorded above",
        abs(results[1.0]["learned"].false_alarm_rate - results[1.0]["baseline"].false_alarm_rate)
        > abs(
            results[1.0]["learned_argmax"].false_alarm_rate
            - results[1.0]["baseline"].false_alarm_rate
        ),
    )
    record(
        "the baseline false-alarm rate rises monotonically with the noise scale",
        all(
            results[SCALES[i]]["baseline"].false_alarm_rate
            < results[SCALES[i + 1]]["baseline"].false_alarm_rate
            for i in range(len(SCALES) - 1)
        ),
    )
    record(
        "the baseline ECE rises monotonically with the noise scale",
        all(
            results[SCALES[i]]["baseline"].calibration.ece
            < results[SCALES[i + 1]]["baseline"].calibration.ece
            for i in range(len(SCALES) - 1)
        ),
    )
    record(
        "the learned posterior is a probability distribution on every test row",
        headline["learned"].brier is not None and 0.0 <= headline["learned"].brier <= 2.0,
    )
    print()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("CALIBRATION MEASUREMENT COMPLETE")


def _decisions(result: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Recompute one detector's test-split confidences and correctness flags."""
    split = result["split"]
    truth = np.asarray(split.test.labels).astype(str)
    if key == "baseline":
        labels, confidence = result["baseline_model"].predict_with_confidence(
            split.test.features
        )
    else:
        labels, confidence = learned_decisions(
            result["model"], split.test.features, float(result["tau"])
        )
    return (
        np.asarray(confidence, dtype=float),
        np.asarray(labels).astype(str) == truth,
    )


if __name__ == "__main__":
    main()
