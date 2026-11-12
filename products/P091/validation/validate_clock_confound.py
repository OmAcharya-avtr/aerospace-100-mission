"""Decompose the clock-versus-die-temperature relation within and between duty cycles.

Why this script exists. ``examples/clock_relation.py`` fits a single line to the
mean CPU clock against die temperature over every measured run and reports the
relation as IDENTIFIABLE with a slope near +46000 kHz/degC. Read without this
decomposition that figure invites a wrong reading: that the board's clock was
measured falling as the die warmed, which is what thermal throttling looks like.

It is not what happened. Duty cycle drives BOTH the clock, through the
``schedutil`` governor responding to load, and the die temperature, through the
power the payload dissipates. Pooling duty cycles therefore fits a relation
across a common cause. This script separates the two contributions and reports
them side by side. The within-duty slope is the one that bears on throttling,
and at the only duty cycle where throttling could act it is exactly zero.

Every number below comes from this script's own pass over the measured traces.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from thermalbudget.trace import Origin, load_campaign  # noqa: E402

#: Seconds dropped from the start of every run, matching baselines.BURST_SKIP_S.
SKIP_S = 2.0

#: Duty groups, declared. The duty-1.0 group pools the soaks and the bursts
#: because both ran the payload at full duty; they differ only in length.
GROUPS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("duty 1.00", 1.0, ("soak_1.00_r1", "soak_1.00_r2", "soak_1.00_r3",
                        "burst_1.00_a", "burst_1.00_b")),
    ("duty 0.75", 0.75, ("soak_0.75",)),
    ("duty 0.50", 0.50, ("soak_0.50",)),
    ("duty 0.25", 0.25, ("soak_0.25",)),
)


def _samples(campaign, labels):
    die, clock = [], []
    for label in labels:
        trace = campaign.traces[label]
        mask = ((trace.t_rel > SKIP_S)
                & np.isfinite(trace.die_temperature_c)
                & np.isfinite(trace.mean_cpu_khz))
        die.append(trace.die_temperature_c[mask])
        clock.append(trace.mean_cpu_khz[mask])
    return np.concatenate(die), np.concatenate(clock)


def _slope(die, clock):
    """Least-squares slope in kHz/degC, or None when the abscissa is degenerate."""
    if die.size < 3:
        return None
    centred = die - die.mean()
    den = float(centred @ centred)
    if den == 0.0:
        return None
    return float(centred @ (clock - clock.mean()) / den)


def main() -> int:
    out = _bootstrap.Tee("validate_clock_confound_output.txt")
    out.rule("CLOCK-VERSUS-DIE RELATION -- within duty against between duty")
    out("INPUT ORIGIN: MEASURED on the board, at power mode MAXN_SUPER, one ambient.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")

    campaign = load_campaign(
        _bootstrap.measured_trace_directory(),
        origin=Origin.MEASURED,
        require=tuple(label for _, _, labels in GROUPS for label in labels),
    )

    out.rule("1  within each duty cycle")
    out("  The slope that bears on thermal throttling. Duty is held fixed, so the")
    out("  governor's load response is held fixed with it, and what remains is the")
    out("  clock's response to temperature alone.")
    out("")
    rows = []
    pooled_die, pooled_clock = [], []
    for name, _duty, labels in GROUPS:
        die, clock = _samples(campaign, labels)
        pooled_die.append(die)
        pooled_clock.append(clock)
        s = _slope(die, clock)
        constant = bool(clock.min() == clock.max())
        rows.append((name, die.size, die.min(), die.max(), clock.min(), clock.max(), s, constant))
        slope_text = ("EXACTLY ZERO, the clock is constant" if constant
                      else "undefined, degenerate abscissa" if s is None
                      else f"{s:+.1f} kHz/degC")
        out(f"  {name:10s} n={die.size:5d}  die {die.min():6.2f}--{die.max():6.2f} degC"
            f"  clock {clock.min():9.1f}--{clock.max():9.1f} kHz  slope {slope_text}")

    out("")
    out.rule("2  pooled across duty cycles -- the confounded fit")
    die_all = np.concatenate(pooled_die)
    clock_all = np.concatenate(pooled_clock)
    pooled_slope = _slope(die_all, clock_all)
    out(f"  pooled n={die_all.size}  slope {pooled_slope:+.1f} kHz/degC")
    out("  Duty cycle sets both the clock, through the governor, and the die")
    out("  temperature, through dissipated power. This slope is a relation across")
    out("  that common cause and is NOT a measurement of the clock's response to")
    out("  temperature.")

    out("")
    out.rule("3  the verdict")
    full = rows[0]
    _, n_full, die_lo, die_hi, c_lo, c_hi, slope_full, constant_full = full
    out(f"  At duty 1.00 the die rose {die_hi - die_lo:.2f} degC over {n_full} samples")
    out(f"  and the mean CPU clock stayed at {c_lo:.1f} kHz throughout.")
    out("")
    out("  Consequences, both of which the product states where the figures appear:")
    out("  - This is the most direct evidence in the campaign that the board did not")
    out("    thermally throttle under the declared payload at MAXN_SUPER.")
    out("  - Baseline 2's clock-to-throughput chain therefore has no thermal signal")
    out("    to act on, which is why baseline 2 reduces to baseline 1 rather than")
    out("    improving on it. That reduction is a measured property of this board")
    out("    under this payload, not a defect in the implementation.")

    ok_zero = out.check(
        "at duty 1.00 the clock-vs-die slope is exactly zero over the measured rise",
        constant_full and (slope_full is None or slope_full == 0.0),
        f"n={n_full}, die rise {die_hi - die_lo:.2f} degC, clock {c_lo:.1f}--{c_hi:.1f} kHz",
    )
    ok_sign = out.check(
        "no duty group shows the clock falling as the die warms by more than the "
        "governor's smallest step",
        all(r[6] is None or r[6] >= -8640.0 for r in rows),
        "; ".join(f"{r[0]} {'constant' if r[7] else format(r[6], '+.1f')}" for r in rows),
    )
    ok_conf = out.check(
        "the pooled slope exceeds every within-duty slope, confirming it is driven "
        "between groups rather than within them",
        pooled_slope is not None
        and all(r[6] is None or abs(r[6]) < abs(pooled_slope) for r in rows),
        f"pooled {pooled_slope:+.1f} kHz/degC against within-duty "
        + ", ".join("constant" if r[7] else f"{r[6]:+.1f}" for r in rows),
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 0 if (ok_zero and ok_sign and ok_conf) else 2


if __name__ == "__main__":
    raise SystemExit(main())
