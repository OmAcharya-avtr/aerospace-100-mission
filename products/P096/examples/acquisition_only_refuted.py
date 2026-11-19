"""Baseline 1 against the ground truth, over the whole committed corpus.

Baseline 1 is the assumption that acquisition time is the whole story,
implemented at full strength: it credits the buffer with its entire free space
and it is the exact restriction of the closed-form model to its first phase. It
is still wrong, and this figure is how wrong, over 1536 handovers.

Two panels. The left is baseline 1's predicted total against the ground truth on
log axes, with the identity line; the right is the distribution of the ratio,
split by which mechanism actually dominated, because the assumption fails in a
different way depending on what it omitted.

Writes ``screenshots/acquisition_only_refuted.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from handoverloss.baselines import fluid_decomposition_array  # noqa: E402
from handoverloss.sequence import MECHANISMS  # noqa: E402


def main() -> int:
    corpus = _common.corpus()
    table = corpus.table
    truth = corpus.targets
    total_truth = truth.sum(axis=1)

    # Baseline 1 at the REALISED acquisition time, which is the most generous
    # setting it has: it is not being penalised for forecasting error here, only
    # for omitting two mechanisms.
    headroom = (
        np.asarray(table["buffer_capacity_bits"], dtype=float)
        - np.asarray(table["initial_occupancy_bits"], dtype=float)
    )
    source = np.asarray(table["source_rate_bps"], dtype=float)
    baseline_one = np.maximum(
        0.0,
        source * np.asarray(table["realised_acquisition_s"], dtype=float) - headroom,
    )
    closed = fluid_decomposition_array(
        source,
        table["downlink_rate_bps"],
        table["buffer_capacity_bits"],
        table["initial_occupancy_bits"],
        table["realised_acquisition_s"],
        table["realised_restart_s"],
        table["declared_ramp_s"],
        table["horizon_s"],
    ).sum(axis=1)

    handle, save = _common.figure("acquisition_only_refuted", size=(12.5, 5.4))
    axes = handle.subplots(1, 2)

    live = total_truth > 0.0
    axes[0].loglog(
        np.maximum(total_truth[live], 1.0),
        np.maximum(baseline_one[live], 1.0),
        ".", markersize=3, alpha=0.4, color="#b3261e", label="baseline 1 (acquisition only)",
    )
    axes[0].loglog(
        np.maximum(total_truth[live], 1.0),
        np.maximum(closed[live], 1.0),
        ".", markersize=3, alpha=0.4, color="#36618e", label="baseline 2 (analytic fluid)",
    )
    limits = (
        float(np.maximum(total_truth[live], 1.0).min()),
        float(np.maximum(total_truth[live], 1.0).max()),
    )
    axes[0].plot(limits, limits, "k-", linewidth=1.0, label="identity")
    axes[0].set_xlabel("ground-truth total loss [bits]")
    axes[0].set_ylabel("predicted total loss [bits]")
    axes[0].set_title(
        "Both baselines against the ground truth\n"
        "at the REALISED durations, so no forecasting error is included"
    )
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3, which="both")

    dominant = np.argmax(truth, axis=1)
    data = []
    labels = []
    for column, name in enumerate(MECHANISMS):
        mask = live & (dominant == column)
        if mask.sum() < 5:
            continue
        data.append(np.log10(np.maximum(baseline_one[mask], 1.0) / total_truth[mask]))
        labels.append(f"{name}\nn={int(mask.sum())}")
    axes[1].boxplot(data, tick_labels=labels, showfliers=False)
    axes[1].axhline(0.0, color="k", linewidth=1.0)
    axes[1].set_ylabel(r"$\log_{10}$(baseline 1 / ground truth)")
    axes[1].set_xlabel("mechanism that actually dominated")
    axes[1].set_title(
        "How baseline 1 fails, by what it omitted\n"
        "zero is correct; below zero is understatement"
    )
    axes[1].grid(axis="y", alpha=0.3)

    path = save()
    print("WHAT TO NOTICE")
    print(
        "  In the left panel the blue cloud sits on the identity line and the red cloud\n"
        "  sits below it, often by orders of magnitude. The right panel says why: where\n"
        "  the protocol restart or the buffer drain dominated, baseline 1 has no term for\n"
        "  the mechanism that did the damage, so it understates without bound. Where\n"
        "  acquisition genuinely dominated it is close to right, which is exactly what\n"
        "  makes the assumption so durable in practice."
    )
    print("")
    print(f"  corpus sha256                {corpus.sha256}")
    print(f"  events with non-zero loss    {int(live.sum())} of {len(corpus)}")
    for column, name in enumerate(MECHANISMS):
        mask = live & (dominant == column)
        if mask.sum() < 5:
            print(f"  {name:<14} n={int(mask.sum())}: fewer than 5 events, not summarised")
            continue
        ratio = np.maximum(baseline_one[mask], 1.0) / total_truth[mask]
        print(
            f"  {name:<14} n={int(mask.sum()):<5} median baseline1/truth="
            f"{float(np.median(ratio)):.6e}  "
            f"10th pct={float(np.percentile(ratio, 10)):.6e}"
        )
    overall = np.maximum(baseline_one[live], 1.0) / total_truth[live]
    print(f"  all events     median baseline1/truth={float(np.median(overall)):.6e}")
    closed_ratio = np.maximum(closed[live], 1.0) / total_truth[live]
    print(f"  all events     median baseline2/truth={float(np.median(closed_ratio)):.6e}")
    print("")
    print(f"figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
