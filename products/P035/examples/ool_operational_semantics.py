"""Example: the operational out-of-limit semantics on one synthetic channel.

Shows, on a single trace, every piece of bookkeeping the limit check has to get
right: soft and hard limits that change with spacecraft mode, a validity mask
that holds the counters rather than resetting or breaching, a persistence count
that delays the raise, and a clear count that steps the latch down one level at
a time.

Writes ``../screenshots/ool_operational_semantics.png`` and prints the alarm
transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telemetryool.limits import (  # noqa: E402
    AlarmState,
    ChannelSpec,
    InvalidPolicy,
    LimitSet,
    OolChecker,
)

SAFE = LimitSet(soft_low=25.0, soft_high=31.0, hard_low=23.0, hard_high=33.0)
SCIENCE = LimitSet(soft_low=27.5, soft_high=29.5, hard_low=26.5, hard_high=30.5)
MODE_SEQUENCE = [("SAFE", 0, 40), ("SCIENCE", 40, 110), ("SAFE", 110, 160)]


def build_trace() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A 160-sample bus-voltage trace in volts, with validity flags and modes."""
    rng = np.random.default_rng(4)
    n = 160
    values = 28.0 + 0.25 * rng.standard_normal(n)
    # A slow excursion that stays inside the SAFE soft band but leaves the
    # tighter SCIENCE soft band, so the same values alarm or not depending on
    # mode.
    values[45:70] += np.linspace(0.0, 2.0, 25)
    values[70:100] += 2.0
    values[100:110] += np.linspace(2.0, 0.0, 10)
    # A short hard excursion, in SAFE mode, where the hard limit is 33.0 V.
    values[118:124] += 5.5
    valid = np.ones(n, dtype=bool)
    valid[76:82] = False  # a stale-data gap in the middle of the excursion
    modes = []
    for name, start, stop in MODE_SEQUENCE:
        modes.extend([name] * (stop - start))
    return values, valid, modes


def main() -> int:
    values, valid, modes = build_trace()
    spec = ChannelSpec(
        name="BUS_V",
        units="V",
        limits={"SAFE": SAFE, "SCIENCE": SCIENCE},
        persistence_soft=5,
        persistence_hard=2,
        clear_persistence=4,
        invalid_policy=InvalidPolicy.HOLD,
        latch_across_mode_change=True,
    )
    checker = OolChecker(spec, mode="SAFE")
    samples = checker.update_series(values, valid, modes)

    print(f"channel {spec.name} [{spec.units}], {len(samples)} samples")
    print(f"SAFE    limits: soft {SAFE.soft_low}-{SAFE.soft_high} V, "
          f"hard {SAFE.hard_low}-{SAFE.hard_high} V")
    print(f"SCIENCE limits: soft {SCIENCE.soft_low}-{SCIENCE.soft_high} V, "
          f"hard {SCIENCE.hard_low}-{SCIENCE.hard_high} V")
    print(f"persistence_soft={spec.persistence_soft} "
          f"persistence_hard={spec.persistence_hard} "
          f"clear_persistence={spec.clear_persistence} "
          f"invalid_policy={spec.invalid_policy.name}")
    print(f"invalid samples: indices 76-81 inclusive ({int((~valid).sum())} samples)")
    print()
    print(f"{'idx':>5s} {'V':>8s} {'valid':>6s} {'mode':>8s} {'level':>9s} "
          f"{'soft':>5s} {'hard':>5s} {'below':>6s} {'state':>12s} {'event':>6s}")
    for s in samples:
        if not (s.raised or s.cleared):
            continue
        print(
            f"{s.index:5d} {s.value:8.3f} {int(s.valid):6d} {s.mode:>8s} "
            f"{('-' if s.level is None else s.level.name):>9s} {s.soft_count:5d} "
            f"{s.hard_count:5d} {s.below_count:6d} {s.state.name:>12s} "
            f"{'RAISE' if s.raised else 'CLEAR':>6s}"
        )
    print()
    occupancy = {
        state.name: sum(1 for s in samples if s.state is state) for state in AlarmState
    }
    print(f"samples per latched state: {occupancy}")
    print(f"final latched state: {checker.state.name}")

    states = np.array([int(s.state) for s in samples])
    soft_counts = np.array([s.soft_count for s in samples])
    hard_counts = np.array([s.hard_count for s in samples])
    idx = np.arange(len(samples))

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, height_ratios=[2.4, 1.0, 1.0]
    )
    ax = axes[0]
    for name, start, stop in MODE_SEQUENCE:
        limits = SAFE if name == "SAFE" else SCIENCE
        ax.fill_between([start, stop - 1], limits.soft_low, limits.soft_high,
                        color="#2e7d32", alpha=0.12, linewidth=0)
        ax.fill_between([start, stop - 1], limits.hard_low, limits.soft_low,
                        color="#f9a825", alpha=0.14, linewidth=0)
        ax.fill_between([start, stop - 1], limits.soft_high, limits.hard_high,
                        color="#f9a825", alpha=0.14, linewidth=0)
        for level, style in ((limits.soft_low, ":"), (limits.soft_high, ":"),
                            (limits.hard_low, "--"), (limits.hard_high, "--")):
            ax.plot([start, stop - 1], [level, level], style, color="#444444", linewidth=1.0)
        ax.axvline(start, color="#1565c0", linewidth=1.0, alpha=0.6)
        ax.text(start + 1, 34.6, f"mode {name}", fontsize=9, color="#1565c0")
    ax.plot(idx[valid], values[valid], ".-", color="#111111", linewidth=0.9,
            markersize=3, label="valid sample")
    ax.plot(idx[~valid], values[~valid], "x", color="#c62828", markersize=7,
            label="invalid sample (counters held)")
    ax.set_ylabel("BUS_V [V]")
    ax.set_ylim(22.0, 35.6)
    ax.set_title(
        "Out-of-limit semantics: mode-dependent limits, validity mask, persistence, debounce"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.step(idx, soft_counts, where="post", color="#f9a825", label="soft counter")
    ax.step(idx, hard_counts, where="post", color="#c62828", label="hard counter")
    ax.axhline(spec.persistence_soft, color="#f9a825", linestyle=":", linewidth=1.0,
               label=f"persistence_soft = {spec.persistence_soft}")
    ax.axhline(spec.persistence_hard, color="#c62828", linestyle=":", linewidth=1.0,
               label=f"persistence_hard = {spec.persistence_hard}")
    ax.set_ylabel("raise counters")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.step(idx, states, where="post", color="#1565c0", linewidth=1.6)
    ax.fill_between(idx, 0, states, step="post", color="#1565c0", alpha=0.15)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([s.name for s in AlarmState])
    ax.set_ylabel("latched state")
    ax.set_xlabel("sample index")
    ax.grid(alpha=0.2)

    out = Path(__file__).resolve().parent.parent / "screenshots"
    out.mkdir(exist_ok=True)
    path = out / "ool_operational_semantics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
