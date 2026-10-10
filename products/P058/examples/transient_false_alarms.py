"""Every detector's behaviour on the transient negative control.

The transient is a 4-sigma excursion lasting 20 samples, after which the channel
recovers. There is no change, so every alarm is a false alarm. A raw firing rate
would be meaningless on its own -- a detector at ARL0 = 500 false-alarms inside
any 70-sample window about 13 % of the time -- so the matched stationary baseline
is measured on the same seeds and the excess is what the transient caused.

Writes ../screenshots/transient_false_alarms.png.
"""

from __future__ import annotations

from pathlib import Path

from telemdrift.benchmark import (
    DETECTOR_LABELS,
    STANDARD,
    calibrate_all_analytic,
    transient_response,
)
from telemdrift.detectors import ANALYTIC_DETECTORS
from telemdrift.plotting import plot_transient_response

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "transient_false_alarms.png"
AMPLITUDE = 4.0
DURATION = 20
REPLICATES = 150


def main() -> None:
    cals = calibrate_all_analytic(STANDARD)
    labels, rates, cis = [], [], []
    print(f"transient: +{AMPLITUDE:g} sigma for {DURATION} samples, then recovery")
    print(f"all detectors at target ARL0 {STANDARD.target_arl0:.0f}; "
          f"{REPLICATES} seeded replicates; alarm window = duration + 50 = "
          f"{DURATION + 50} samples")
    print()
    print("detector        transient   baseline   excess      Wilson 95 % CI (transient)")
    for key in ANALYTIC_DETECTORS:
        r = transient_response(key, cals[key].threshold, AMPLITUDE, DURATION,
                               STANDARD, replicates=REPLICATES)
        labels.append(DETECTOR_LABELS[key])
        rates.append(r["transient_rate"])
        cis.append((r["transient_lo"], r["transient_hi"]))
        print(f"{DETECTOR_LABELS[key]:15s} {100 * r['transient_rate']:8.1f}% "
              f"{100 * r['baseline_rate']:9.1f}% {100 * r['excess']:+8.1f} pp   "
              f"[{100 * r['transient_lo']:5.1f}%, {100 * r['transient_hi']:5.1f}%]")
    print()
    print("Every alarm counted here is a false alarm: the channel recovers.")
    path = plot_transient_response(
        labels, rates, cis, OUT,
        f"Firing rate on a +{AMPLITUDE:g} sigma transient that is not a change",
    )
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
