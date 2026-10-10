"""Alarm-ratio traces for the five analytic detectors on one seeded episode.

All five are first calibrated to the same measured ARL0, then run over one short
mean-step episode. Each statistic is divided by its own threshold so the five
incomparable scales share one axis and 1.0 is the alarm line for all of them.

On the seed selection, stated because it matters
------------------------------------------------
At an ARL0 of about 500 samples, a typical 500-sample episode contains one or
two false alarms before the change, and a figure showing five detectors each
resetting at different points is unreadable. The seed used here is chosen by a
stated deterministic rule -- the first seed in a fixed range on which the five
detectors raise no more than ONE pre-change false alarm between them -- and the
search is printed. This figure is therefore an illustration of the mechanism,
not a sample of typical behaviour. Typical behaviour is in the tables: see
validation/outputs/validate_change_types.txt, where 88 to 95 % of replicates
contain a pre-change false alarm.

Writes ../screenshots/detector_traces.png.
"""

from __future__ import annotations

from pathlib import Path

from telemdrift.benchmark import DETECTOR_LABELS, STANDARD, analytic_factory, calibrate_all_analytic
from telemdrift.detectors import alarm_ratio_trace, first_alarm_at_or_after
from telemdrift.plotting import plot_detector_traces
from telemdrift.streams import ChangeSpec, change_stream

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "detector_traces.png"
SEED_RANGE = range(58_601, 58_661)
PRE, POST = 250, 250
MAX_PRE_ALARMS = 1


def main() -> None:
    cals = calibrate_all_analytic(STANDARD)
    spec = ChangeSpec("mean_step", 1.0)
    print("Selecting an episode. Rule: the first seed in "
          f"{SEED_RANGE.start}..{SEED_RANGE.stop - 1} on which the five detectors")
    print(f"raise at most {MAX_PRE_ALARMS} pre-change false alarm between them. A "
          "typical episode has")
    print("one or two per detector and the figure would be unreadable; the tables "
          "carry the")
    print("typical behaviour.")
    print()
    chosen = None
    for seed in SEED_RANGE:
        stream, idx = change_stream(PRE, POST, spec, seed)
        total_pre = 0
        for key, cal in cals.items():
            _, fired = alarm_ratio_trace(analytic_factory(key, cal.threshold)(), stream)
            total_pre += int((fired < idx).sum())
        if total_pre <= MAX_PRE_ALARMS:
            chosen = (seed, stream, idx, total_pre)
            break
        print(f"  seed {seed}: {total_pre} pre-change false alarms, rejected")
    if chosen is None:
        seed = SEED_RANGE.start
        stream, idx = change_stream(PRE, POST, spec, seed)
        chosen = (seed, stream, idx, -1)
        print(f"  no seed met the rule; falling back to {seed}")
    seed, stream, idx, total_pre = chosen
    print(f"  seed {seed}: {total_pre} pre-change false alarms, accepted")
    print()
    print(f"Episode: seed {seed}, mean step +{spec.magnitude:g} sigma at sample {idx}, "
          f"{stream.size} samples.")
    print("Alarms before the change index are false alarms; the detector is reset "
          "on each.")
    print()
    ratios, alarms = {}, {}
    print("detector        threshold      held-out ARL0  pre-alarms  first after  delay")
    for key, cal in cals.items():
        label = DETECTOR_LABELS[key]
        series, fired = alarm_ratio_trace(analytic_factory(key, cal.threshold)(), stream)
        first = first_alarm_at_or_after(fired, idx)
        ratios[label] = series
        alarms[label] = first
        delay = "n/a" if first < 0 else str(first - idx)
        print(f"{label:15s} {cal.threshold:<14.6g} {cal.achieved.arl0:13.1f} "
              f"{int((fired < idx).sum()):11d} {first:12d} {delay:>6s}")
    path = plot_detector_traces(
        stream, idx, ratios, OUT,
        f"Five detectors at one operating point (ARL0 ~ "
        f"{STANDARD.target_arl0:.0f}), seed {seed}", alarms,
    )
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
