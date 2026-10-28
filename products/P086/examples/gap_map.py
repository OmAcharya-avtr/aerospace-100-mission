"""Every gap on the timeline, coloured by what the policy decided for each channel."""

from __future__ import annotations

from _common import PERMIT, REFUSE, bootstrap, save, style, tolerances

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from telemgap import classify, dataset, policy  # noqa: E402


def main() -> int:
    style()
    generated = dataset.generate(samples=4000, seed=515, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    plan = policy.plan(record, report, tolerances())

    figure, axes = plt.subplots(
        2, 1, figsize=(9.5, 5.4), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    names = list(record.schema.channel_names)
    # A gap is seconds wide on an axis thousands of seconds long, so a bar at
    # its true width is invisible. Each bar is widened to at least 1.5 per cent
    # of the axis span; the bars are a map of where the decisions are, and the
    # durations are in the table printed below the figure.
    span = float(record.timestamps.max() - record.timestamps.min())
    minimum_width = 0.015 * span
    for decision in plan.decisions:
        gap = report.findings.gaps[decision.gap_index]
        row = names.index(decision.channel_name)
        axes[0].barh(
            row,
            width=max(gap.duration_s, minimum_width),
            left=gap.start_time,
            height=0.72,
            color=REFUSE if decision.refused else PERMIT,
            edgecolor="none",
        )
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("timestamp [s]")
    axes[0].set_xlim(float(record.timestamps.min()), float(record.timestamps.max()))
    axes[0].set_ylim(-0.6, len(names) + 0.3)
    axes[0].legend(
        handles=[
            Patch(color=PERMIT, label=f"interpolation permitted ({len(plan.permissions)})"),
            Patch(color=REFUSE, label=f"refused ({len(plan.refusals)})"),
        ],
        loc="upper right",
        ncol=2,
    )
    axes[0].set_title(
        "telemgap.policy: the gap-handling policy, one bar per gap and channel "
        "(bars widened to be visible)\n"
        f"{len(report.findings.gaps)} gaps, {report.missing_frame_total} frames implied lost, "
        f"{len(plan.refusals)} of {len(plan.decisions)} decisions refused"
    )

    gates = plan.summary()
    labels = [key.removeprefix("failed_") for key in gates if key.startswith("failed_")]
    values = [gates[f"failed_{label}"] for label in labels]
    axes[1].bar(range(len(labels)), values, color=REFUSE, width=0.6)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    axes[1].set_ylabel("decisions failing")
    axes[1].set_title("which gate refused, counted over all decisions", fontsize=9)
    figure.subplots_adjust(hspace=0.75)
    path = save(figure, "gap_map")
    print(f"written: {path}")
    print(f"policy summary: {gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
