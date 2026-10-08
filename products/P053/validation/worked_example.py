"""The worked example reproduced verbatim in the README.

Declare a twin, calibrate a CUSUM to a declared false-alarm target on
in-control residuals alone, inject a 1 % actuator-gain loss, and read off the
detection delay with its false-alarm cost.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from twininvalidate import (  # noqa: E402
    AssetChange,
    DetectorSpec,
    InvalidationMonitor,
    StreamSpec,
    changed_streams,
    in_control_streams,
    reference_twin,
    simulate_residuals,
)


def main() -> int:
    # 1. The declared twin and its fixed-gain residual generator.
    twin = reference_twin()
    filt = twin.steady_state()
    print(f"innovation sd sqrt(S)      {np.sqrt(filt.S):.6e} rad")

    # 2. Calibrate a CUSUM to a declared in-control ARL0 of 1000 samples
    #    (50 s at 20 Hz) using in-control residuals ONLY.
    in_control = in_control_streams(n_runs=200, n_samples=2000, seed=53001)
    monitor = InvalidationMonitor.from_target(DetectorSpec("cusum"), in_control, 1000.0)
    print(f"CUSUM threshold            {monitor.threshold:.4f}")
    print(f"achieved in-control ARL0   {monitor.calibration.achieved_arl0:.1f} samples")
    print(f"false alarms per 1000 h    "
          f"{monitor.false_alarms_per_1000h(monitor.calibration.achieved_arl0):.0f}")

    # 3. Inject the declared 1 % actuator-gain loss and measure the delay.
    changed = changed_streams("parameter_step", n_runs=200, n_samples=2000)
    delay = monitor.arl1(changed)
    print(f"detection delay            {delay.value:.1f} +- {delay.stderr:.1f} samples "
          f"= {delay.value * twin.dt:.2f} s")
    print(f"runs detected              {delay.detection_fraction * 100:.0f} %")

    # 4. One stream, for the record.
    single = simulate_residuals(
        StreamSpec(
            change=AssetChange("parameter_step", onset=300, magnitude=-0.01),
            n_runs=1,
            n_samples=900,
            seed=7,
        )
    )
    verdict = monitor.run(single)
    print(f"single stream, change at 300: {verdict.describe()}")

    # 5. What the alarm does not tell you.
    print("the alarm says the twin and the asset disagree; it does not say which")
    print("one moved. See twininvalidate.ambiguity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
