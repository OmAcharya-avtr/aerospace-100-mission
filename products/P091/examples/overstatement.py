"""The headline: burst against sustained, with the interval that decides it.

What to notice: the two bars are the point estimates and the whiskers are
two-level bootstrap intervals. The panel on the right is the quantity that
matters -- the cold-burst overstatement with its interval -- and the vertical
line at zero is the decision. An interval that straddles zero is the finding
"sustained throughput is statistically indistinguishable from burst throughput",
and this product reports that as a result rather than as a missing gap.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _common  # noqa: E402

ROOT = _common.bootstrap()

from thermalbudget.baselines import (  # noqa: E402
    burst_baseline,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from thermalbudget.provenance import meet  # noqa: E402


def main() -> int:
    plt = _common.pyplot()
    campaign, is_measured = _common.load_for_example("no_throttle")
    colour = _common.MEASURED_COLOUR if is_measured else _common.SYNTHETIC_COLOUR

    base1 = burst_baseline(
        campaign.bursts(), seed=_common.SEED, n_resamples=_common.RESAMPLES
    )
    base3 = sustained_truth(
        campaign.soaks(1.0),
        window_s=_common.WINDOW_S,
        threshold_c_per_s=_common.THRESHOLD_C_PER_S,
        seed=_common.SEED,
        n_resamples=_common.RESAMPLES,
    )
    base2 = rc_baseline(campaign.soaks(1.0), campaign.cooldowns(), campaign.bursts())
    head = overstatement(
        base1.throughput_groups,
        base3.throughput_groups,
        seed=_common.SEED,
        n_resamples=_common.RESAMPLES,
        origin=meet(base1.origin, base3.origin),
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.8), width_ratios=[1.25, 1.0])

    labels = ["baseline 1\ncold burst", "baseline 2\nRC network", "baseline 3\nmeasured"]
    points = [
        base1.throughput.point,
        base2.predicted_throughput_iters_per_s,
        base3.throughput.point,
    ]
    lows = [
        base1.throughput.point - base1.throughput.ci_low,
        0.0,
        base3.throughput.point - base3.throughput.ci_low,
    ]
    highs = [
        base1.throughput.ci_high - base1.throughput.point,
        0.0,
        base3.throughput.ci_high - base3.throughput.point,
    ]
    low = min(points) * 0.9
    bars = left.bar(labels, points, color=[colour, "#8a8a8a", _common.ACCENT], width=0.6)
    # Baselines 1 and 3 only. Baseline 2 is an analytic prediction and has no
    # resampled interval; drawing a zero-length bar for it renders as a cap mark
    # and reads as a very narrow interval, which is the opposite of the truth.
    resampled = [0, 2]
    left.errorbar(
        [labels[index] for index in resampled],
        [points[index] for index in resampled],
        yerr=[[lows[index] for index in resampled], [highs[index] for index in resampled]],
        fmt="none",
        ecolor="#222222",
        capsize=5,
        lw=1.1,
    )
    for bar, value in zip(bars, points, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2.0,
            low + 0.06 * (value - low),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#ffffff",
        )
    left.text(
        1.0,
        low + 0.55 * (points[1] - low),
        "analytic\nprediction:\nno interval",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#ffffff",
    )
    left.set_ylim(low, max(points) * 1.06)
    left.set_ylabel("throughput, iterations per wall second")
    left.set_title(
        "Three baselines on the same campaign\n"
        "baseline 2 carries no interval: it is an analytic prediction, not a resampled "
        "measurement",
        fontsize=9,
    )

    centre = head.overstatement.point
    right.errorbar(
        [100.0 * centre],
        [0.0],
        xerr=[[100.0 * (centre - head.overstatement.ci_low)],
              [100.0 * (head.overstatement.ci_high - centre)]],
        fmt="o",
        color=colour,
        ecolor="#222222",
        capsize=6,
        markersize=7,
        lw=1.3,
    )
    right.axvline(0.0, color=_common.ACCENT, lw=1.3)
    right.text(
        0.0,
        0.42,
        " no overstatement",
        color=_common.ACCENT,
        fontsize=8,
        rotation=90,
        va="center",
    )
    span = max(
        abs(100.0 * head.overstatement.ci_low), abs(100.0 * head.overstatement.ci_high), 1.0
    )
    right.set_xlim(-1.6 * span, 1.6 * span)
    right.set_ylim(-1.0, 1.0)
    right.set_yticks([])
    right.set_xlabel("cold-burst overstatement of sustained throughput, per cent")
    right.set_title(
        f"VERDICT {head.verdict}\n"
        f"{100.0 * centre:+.3f} per cent "
        f"[{100.0 * head.overstatement.ci_low:+.3f}, "
        f"{100.0 * head.overstatement.ci_high:+.3f}] per cent",
        fontsize=9.5,
    )
    right.text(
        0.5,
        -0.62,
        f"two-level bootstrap, {head.n_resamples} resamples, "
        f"{head.overstatement.n_distinct_group_draws} distinct run-level draws",
        transform=right.transData,
        ha="center",
        fontsize=7.5,
        color="#555555",
    )

    _common.stamp(
        figure,
        [left, right],
        is_measured,
        detail=f"baseline 3 included {len(base3.included)} equilibrated run(s).",
    )
    figure.subplots_adjust(bottom=0.2, wspace=0.26)
    path = _common.save(figure, "overstatement.png")
    plt.close(figure)
    print(f"wrote {path}")
    print(f"  burst     {base1.throughput.point:.2f} iters/s")
    print(f"  RC        {base2.predicted_throughput_iters_per_s:.2f} iters/s"
          + ("  [reduces to baseline 1]" if base2.reduces_to_baseline_1 else ""))
    print(f"  sustained {base3.throughput.point:.2f} iters/s")
    print(f"  overstatement {100.0 * centre:+.3f} per cent "
          f"[{100.0 * head.overstatement.ci_low:+.3f}, "
          f"{100.0 * head.overstatement.ci_high:+.3f}] -> {head.verdict}")
    print(f"  origin: {'MEASURED' if is_measured else 'SYNTHETIC'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
