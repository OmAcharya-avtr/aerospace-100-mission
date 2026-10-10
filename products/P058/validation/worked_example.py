"""The worked example reproduced verbatim in README.md.

If this script's output and the README's output block ever differ, the README is
wrong. validate_cli.py re-runs every command the README quotes for the same
reason.
"""

from __future__ import annotations

from _harness import run


def body(report) -> None:
    import numpy as np

    from telemdrift import (
        CUSUM,
        STANDARD,
        ChangeSpec,
        analytic_factory,
        calibrate_threshold,
        change_stream,
        measure_arl0,
        measure_arl1,
        stationary,
    )

    report.section("The README worked example, run")
    print("  # 1. A telemetry channel, standardised, with a mean step at sample 1000.")
    spec = ChangeSpec("mean_step", 1.0)
    stream, change_at = change_stream(1_000, 1_500, spec, seed=58_601)
    print(f"  stream length {stream.size}, change at {change_at}, "
          f"pre-change mean {stream[:change_at].mean():+.4f}, "
          f"post-change mean {stream[change_at:].mean():+.4f}")
    print()

    print("  # 2. A CUSUM at its textbook default. What false-alarm rate is that?")
    default = CUSUM.default_threshold()
    at_default = measure_arl0(
        lambda: CUSUM(h=default),
        lambda length, seed: stationary(length, seed),
        STANDARD.eval_seeds,
        STANDARD.eval_length,
    )
    print(f"  h = {default:g}  ->  {at_default.summary()}")
    print()

    print("  # 3. Not the one we wanted. Calibrate it to a declared target instead.")
    cal = calibrate_threshold(
        name="CUSUM",
        factory_from_threshold=lambda h: CUSUM(h=h),
        default_threshold=default,
        stream_fn=lambda length, seed: stationary(length, seed),
        target_arl0=STANDARD.target_arl0,
        calibration_seeds=STANDARD.cal_seeds,
        evaluation_seeds=STANDARD.eval_seeds,
        calibration_length=STANDARD.cal_length,
        evaluation_length=STANDARD.eval_length,
    )
    print(f"  target ARL0 {STANDARD.target_arl0:.0f} samples")
    print(f"  h = {cal.threshold:.4f}")
    print(f"  fitted   ARL0 {cal.calibration_arl0:.1f} samples")
    print(f"  held-out {cal.achieved.summary()}")
    print(f"  target error {100 * cal.target_error:+.1f} % "
          "(on seeds not used for fitting)")
    print()

    print("  # 4. Now the delay is a number that means something.")
    arl1 = measure_arl1(
        analytic_factory("cusum", cal.threshold),
        lambda pre, post, seed: change_stream(pre, post, spec, seed),
        STANDARD.arl1_seeds(0),
        STANDARD.pre_length,
        STANDARD.arl1_budget,
    )
    print(f"  {arl1.summary()}")
    print(f"  median delay {np.median(arl1.delays):.0f} samples, "
          f"90th percentile {np.quantile(arl1.delays, 0.9):.0f} samples")
    print()

    print("  # 5. And the transient that must NOT count as a detection.")
    transient = ChangeSpec("transient", 4.0, 20)
    t_arl1 = measure_arl1(
        analytic_factory("cusum", cal.threshold),
        lambda pre, post, seed: change_stream(pre, post, transient, seed),
        STANDARD.arl1_seeds(2),
        STANDARD.pre_length,
        STANDARD.arl1_budget,
    )
    print(f"  transient +4.0 sigma for 20 samples: {t_arl1.summary()}")
    print("  Every alarm there is a false alarm. CUSUM does not distinguish a")
    print("  transient from a change, and neither does any other detector in this")
    print("  package (validate_transient.py).")

    report.check("the calibrated threshold is above the default, as a longer ARL0 "
                 "requires", cal.threshold > default,
                 f"{cal.threshold:.4f} > {default:g}")
    report.check("the calibrated held-out ARL0 is within 25 % of target",
                 abs(cal.target_error) < 0.25, f"{100 * cal.target_error:+.1f} %")
    report.check("the detection delay is at least 20x shorter than the ARL0 it "
                 "was bought with", cal.achieved.arl0 / arl1.arl1 > 20.0,
                 f"{cal.achieved.arl0 / arl1.arl1:.0f}x")
    report.check("CUSUM responds to the transient faster than to the real change, "
                 "which is the finding not the feature",
                 t_arl1.arl1 < arl1.arl1,
                 f"transient {t_arl1.arl1:.1f} vs change {arl1.arl1:.1f} samples")


if __name__ == "__main__":
    raise SystemExit(run("worked_example",
                         "telemdrift 0.1.0 - the README worked example",
                         body))
