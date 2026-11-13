"""The measured contention descriptors, including the foreign load on a shared board.

Draws, from the MEASURED campaign:

- top left: the per-second measured FOREIGN CPU occupancy on every run. The
  board was shared throughout and this is the contamination, measured per
  second and reported rather than assumed absent;
- top right: measured counted miss rate against the declared stressor count,
  with each run labelled;
- bottom left: mean and minimum core clock per run, against the die
  temperature, so a reader can see whether the clock moved;
- bottom right: the measured response-time distribution per CPU-stressor level,
  as a box plot, with the deadline marked.

Writes ``screenshots/contention_descriptors.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True


import _common  # noqa: E402

ROOT = _common.bootstrap()

from deadlinerisk.accounting import account  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402


def main() -> int:
    try:
        campaign = _common.measured_campaign()
    except TraceDataAbsentError as error:
        print(f"REFUSED: {error}")
        return 1

    runs = sorted(campaign.jobs)
    handle, save = _common.figure("contention_descriptors")

    top_left = handle.add_subplot(2, 2, 1)
    worst = 0.0
    for run in runs:
        monitor = campaign.monitors.get(run)
        if monitor is None or len(monitor) == 0:
            continue
        top_left.plot(monitor.t_s, monitor.foreign_pct, lw=0.9, label=run)
        worst = max(worst, float(monitor.foreign_pct.max()))
    top_left.set_xlabel("time within run (s)")
    top_left.set_ylabel("foreign CPU occupancy (% of total capacity)")
    top_left.set_title("measured foreign load: the board was SHARED")
    top_left.grid(alpha=0.3)
    top_left.legend(fontsize=6, ncol=2, loc="upper right")

    top_right = handle.add_subplot(2, 2, 2)
    for run in runs:
        n_cpu, n_mem = campaign.stressors(run)
        totals = account(campaign.jobs[run])
        marker = "o" if n_mem == 0 else "D"
        top_right.semilogy(
            n_cpu + n_mem, max(totals.miss_rate, 1e-7), marker, ms=7, alpha=0.85
        )
        top_right.annotate(
            run, xy=(n_cpu + n_mem, max(totals.miss_rate, 1e-7)),
            xytext=(5, 2), textcoords="offset points", fontsize=7,
        )
    top_right.set_xlabel("declared co-resident workers (CPU + memory stressors)")
    top_right.set_ylabel("counted miss rate over ALL releases")
    top_right.set_title("measured miss rate against declared contention")
    top_right.grid(alpha=0.3, which="both")

    bottom_left = handle.add_subplot(2, 2, 3)
    for run in runs:
        monitor = campaign.monitors.get(run)
        if monitor is None or len(monitor) == 0:
            continue
        bottom_left.plot(
            float(monitor.tj_degC.mean()), float(monitor.clk_mean_khz.mean()) / 1e3,
            "o", ms=7, alpha=0.85,
        )
        bottom_left.plot(
            float(monitor.tj_degC.mean()), float(monitor.clk_min_khz.min()) / 1e3,
            "x", ms=7, color="grey",
        )
        bottom_left.annotate(
            run, xy=(float(monitor.tj_degC.mean()),
                     float(monitor.clk_mean_khz.mean()) / 1e3),
            xytext=(5, 2), textcoords="offset points", fontsize=7,
        )
    bottom_left.set_xlabel("mean die temperature over the run (degrees C)")
    bottom_left.set_ylabel("core clock (MHz)")
    bottom_left.set_title("circles: mean clock. crosses: minimum single sample")
    bottom_left.grid(alpha=0.3)

    bottom_right = handle.add_subplot(2, 2, 4)
    cpu_runs = [run for run in runs if run.startswith("mix_c") and "mem" not in run]
    data = [campaign.jobs[run].response_s * 1e3 for run in cpu_runs]
    if data:
        bottom_right.boxplot(data, tick_labels=cpu_runs, showfliers=False)
    bottom_right.axhline(
        campaign.deadline_s * 1e3, color="crimson", lw=1.2,
        label=f"deadline D = {campaign.deadline_s * 1e3:.2f} ms",
    )
    bottom_right.set_ylabel("measured response time (ms)")
    bottom_right.set_title("response distribution per CPU-stressor level (outliers hidden)")
    bottom_right.tick_params(axis="x", rotation=45, labelsize=8)
    bottom_right.grid(alpha=0.3, axis="y")
    bottom_right.legend(fontsize=8)

    path = save()
    print(f"runs                     {len(runs)}")
    print("")
    print(f"{'run':<12}{'foreign_mean_%':>16}{'foreign_max_%':>15}{'tj_mean_C':>11}"
          f"{'clk_mean_MHz':>14}{'counted_rate':>14}")
    for run in runs:
        monitor = campaign.monitors.get(run)
        totals = account(campaign.jobs[run])
        if monitor is None or len(monitor) == 0:
            print(f"{run:<12}{'no monitor':>16}")
            continue
        print(
            f"{run:<12}{float(monitor.foreign_pct.mean()):>16.3f}"
            f"{float(monitor.foreign_pct.max()):>15.3f}"
            f"{float(monitor.tj_degC.mean()):>11.2f}"
            f"{float(monitor.clk_mean_khz.mean()) / 1e3:>14.1f}"
            f"{totals.miss_rate:>14.6e}"
        )
    print("")
    print(f"worst single-second foreign occupancy  {worst:.3f} %")
    print(f"figure written to        {path}")
    print(
        "What to notice: the top-left panel is the measured contamination. The "
        "campaign cannot attribute any part of a miss rate to foreign load rather "
        "than to the declared stressors, because the foreign load was not controlled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
