"""Clock against die temperature, decomposed within duty cycle and between it.

**This is the figure most likely to be misread, so it is drawn to prevent the
misreading.** A single cloud of (die temperature, clock) samples pooled over the
whole campaign fits a steep slope, and a reader who sees only that will conclude
that the clock was measured responding to temperature. It was not. Duty cycle
sets the clock, through the schedutil governor responding to load, and it sets
the die temperature, through dissipated power. A pooled fit crosses that common
cause and measures nothing about throttling.

So the left panel colours every sample by its duty cycle and fits each duty
group separately. The slope that bears on thermal throttling is the WITHIN-duty
one. On the measured board, at duty 1.0, it is exactly zero: the die rose
10.94 degC and the mean CPU clock never left 1728000 kHz. The pooled line is
drawn too, dashed and labelled as confounded, because leaving it out would hide
the thing the reader needs to be warned about.

The right panel is the clock trace of every soak against time, which is where a
governor step down under heat would be visible if there were one.

The full decomposition, with its own checks, is
``validation/validate_clock_confound.py``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _common  # noqa: E402

ROOT = _common.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget.clock_throughput import (  # noqa: E402
    clock_temperature_relation,
    proportionality_exponent,
)
from thermalbudget.equilibrium import fit_linear  # noqa: E402

#: One colour per commanded duty cycle, coolest to hottest.
DUTY_COLOURS = {0.25: "#4878a8", 0.5: "#5aa06a", 0.75: "#d08a2a", 1.0: "#b03030"}

#: Seconds dropped from the start of every run, matching
#: ``thermalbudget.baselines.BURST_SKIP_S`` and ``validate_clock_confound.py``.
#: The skip is applied here so that this figure and that script report the SAME
#: within-duty slopes; two sources quoting slightly different numbers for the
#: same quantity is a defect in its own right.
SKIP_S = 2.0


def _group_samples(group):
    """Finite (die, clock) pairs of a duty group, after the declared skip."""
    die, clock = [], []
    for trace in group:
        mask = (
            (trace.t_rel > SKIP_S)
            & np.isfinite(trace.die_temperature_c)
            & np.isfinite(trace.mean_cpu_khz)
        )
        die.append(trace.die_temperature_c[mask])
        clock.append(trace.mean_cpu_khz[mask])
    return np.concatenate(die), np.concatenate(clock)


def _within_duty(traces):
    """(duty, n, die range, clock range, slope or None) for each commanded duty."""
    rows = []
    for duty in sorted({trace.duty for trace in traces}):
        group = [trace for trace in traces if trace.duty == duty]
        die, clock = _group_samples(group)
        if die.size < 3:
            continue
        if float(np.ptp(clock)) == 0.0:
            rows.append((duty, die.size, (die.min(), die.max()),
                         (clock.min(), clock.max()), None))
            continue
        rows.append((duty, die.size, (die.min(), die.max()),
                     (clock.min(), clock.max()), fit_linear(die, clock).slope))
    return rows


def main() -> int:
    plt = _common.pyplot()
    campaign, is_measured = _common.load_for_example("no_throttle")
    pooled = campaign.soaks() + campaign.bursts()
    relation = clock_temperature_relation(pooled)
    exponent = proportionality_exponent(campaign.soaks(1.0) + campaign.bursts(), duty=1.0)
    rows = _within_duty(pooled)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.9))

    for duty, _n, _die_range, _clock_range, _slope in rows:
        die, clock = _group_samples([trace for trace in pooled if trace.duty == duty])
        left.scatter(
            die,
            clock / 1000.0,
            s=5,
            alpha=0.45,
            color=DUTY_COLOURS.get(duty, "#777777"),
            edgecolors="none",
            label=f"duty {duty:g}",
        )
    for duty, _n, die_range, _clock_range, slope in rows:
        if slope is None:
            continue
        grid = np.linspace(*die_range, 20)
        die, clock = _group_samples([trace for trace in pooled if trace.duty == duty])
        intercept = float(clock.mean()) - slope * float(die.mean())
        left.plot(grid, (intercept + slope * grid) / 1000.0,
                  color=DUTY_COLOURS.get(duty, "#777777"), lw=1.6)
    if relation.identifiable:
        grid = np.linspace(*relation.temperature_range_c, 60)
        left.plot(
            grid,
            (relation.intercept_khz + relation.slope_khz_per_c * grid) / 1000.0,
            color="#222222",
            lw=1.8,
            ls="--",
            label=f"POOLED fit {relation.slope_khz_per_c / 1000.0:+.1f} MHz/degC"
                  " -- CONFOUNDED",
        )
    left.set_xlabel("die temperature, degC")
    left.set_ylabel("mean CPU clock across six cores, MHz")
    flat = [f"{duty:g}" for duty, _n, _d, _c, slope in rows if slope is None]
    left.set_title(
        "Clock against die temperature, coloured BY DUTY CYCLE\n"
        "the pooled dashed fit crosses a common cause\n"
        + (f"at duty {', '.join(flat)} the clock is EXACTLY constant"
           if flat else "no duty group has a constant clock"),
        fontsize=8.4,
    )
    left.legend(loc="lower right", fontsize=7.2, ncols=2)

    for trace in campaign.soaks():
        clock = trace.mean_cpu_khz
        keep = np.isfinite(clock)
        right.plot(
            trace.t_rel[keep],
            clock[keep] / 1000.0,
            lw=0.9,
            color=DUTY_COLOURS.get(trace.duty, "#777777"),
            label=f"{trace.label} (duty {trace.duty:g})",
        )
    right.set_xlabel("time since run start, s")
    right.set_ylabel("mean CPU clock, MHz")
    right.set_title(
        "Clock against time, every soak run\n"
        "a governor step down under heat would show\n"
        "here as a visible drop; none does",
        fontsize=8.4,
    )
    right.legend(fontsize=6.6, loc="lower right", ncols=2)

    detail = (
        "within-duty slopes, kHz/degC: "
        + "; ".join(
            f"{duty:g}: " + ("EXACTLY ZERO" if slope is None else f"{slope:+.1f}")
            for duty, _n, _d, _c, slope in rows
        )
        + f".  Pooled: {relation.slope_khz_per_c:+.1f}."
    )
    _common.stamp(figure, [left, right], is_measured, detail=detail)
    figure.subplots_adjust(bottom=0.22, wspace=0.26)
    path = _common.save(figure, "clock_relation.png")
    plt.close(figure)
    print(f"wrote {path}")
    print(f"  pooled relation: {'IDENTIFIABLE' if relation.identifiable else 'DEGENERATE'}"
          " -- CONFOUNDED BY DUTY CYCLE, see validation/validate_clock_confound.py")
    print(f"  {relation.reason}")
    for duty, n, die_range, clock_range, slope in rows:
        rendered = "EXACTLY ZERO, the clock is constant" if slope is None else f"{slope:+.1f}"
        print(
            f"  within duty {duty:4.2f}  n={n:5d}  die {die_range[0]:6.2f}--{die_range[1]:6.2f}"
            f" degC  clock {clock_range[0]:9.1f}--{clock_range[1]:9.1f} kHz  slope {rendered}"
        )
    print(
        "  NOTE: the pooled slope printed above is over ALL samples, because that is what"
        " the package API thermalbudget.clock_temperature_relation pools. "
        "validate_clock_confound.py pools the SKIP-APPLIED samples and reports a slightly"
        " different pooled figure. Both are confounded by duty cycle and neither is a"
        " throttling measurement; the within-duty slopes above are identical between the"
        " two because this example applies the same skip."
    )
    print(f"  exponent: {exponent.reason}")
    print(f"  origin: {'MEASURED' if is_measured else 'SYNTHETIC'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
