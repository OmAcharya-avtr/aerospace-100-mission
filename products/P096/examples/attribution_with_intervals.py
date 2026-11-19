"""The headline figure: one handover decomposed, with an interval on each mechanism.

Draws the three mechanism volumes as bars with their bootstrap intervals, and
beside them the single number a link budget would give. The point of the figure
is that the error bars are narrow enough to decide the ordering while the
single number is not in the picture at all.

Writes ``screenshots/attribution_with_intervals.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from handoverloss import (  # noqa: E402
    HandoverSequence,
    acquisition_only_loss_bits,
    attribute,
    fluid_decomposition,
)
from handoverloss.sequence import MECHANISMS  # noqa: E402
from handoverloss.trace import EventParameters, simulate  # noqa: E402

REALISATIONS = 96 if _common.FAST else 512


def main() -> int:
    sequence = HandoverSequence(
        source_rate_bps=2.0e8,
        downlink_rate_bps=1.0e9,
        buffer_capacity_bits=3.2e10,
        initial_occupancy_bits=2.9e10,
        acquisition_s=40.0,
        restart_s=12.0,
        ramp_s=60.0,
        horizon_s=700.0,
    )
    closed = fluid_decomposition(sequence)
    only = acquisition_only_loss_bits(sequence)
    parameters = EventParameters.from_sequences([sequence] * REALISATIONS)
    trace = simulate(parameters, np.random.default_rng(96_096_2026))
    report = attribute(trace, resamples=200 if _common.FAST else 4000)

    handle, save = _common.figure("attribution_with_intervals")
    axes = handle.subplots(1, 2, width_ratios=(2.0, 1.0))

    positions = np.arange(3)
    truth = np.array([report.share(name).loss_gbit for name in MECHANISMS])
    low = np.array([report.share(name).low_gbit for name in MECHANISMS])
    high = np.array([report.share(name).high_gbit for name in MECHANISMS])
    analytic = np.array([closed.mechanisms[name] / 1e9 for name in MECHANISMS])

    axes[0].bar(
        positions - 0.19, truth, width=0.36,
        yerr=np.vstack([truth - low, high - truth]),
        capsize=5, color="#36618e", label="baseline 3: trace, 95% event bootstrap",
    )
    axes[0].bar(
        positions + 0.19, analytic, width=0.36,
        color="#9cc2e5", label="baseline 2: analytic fluid closed form",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(MECHANISMS)
    axes[0].set_ylabel("loss across the handover [Gbit]")
    axes[0].set_title(
        "Decomposition with an interval on each mechanism\n"
        f"dominant: {report.dominant}; ordering decided: {report.dominance_is_decided}"
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar([0], [only / 1e9], width=0.5, color="#b3261e")
    axes[1].bar([1], [report.total_loss_gbit], width=0.5, color="#36618e")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(
        ["baseline 1\n(acquisition only)", "ground truth\ntotal"], fontsize=8
    )
    axes[1].set_ylabel("total loss [Gbit]")
    axes[1].set_title(
        "What a link budget implies,\nagainst what the handover costs\n"
        f"understated by {report.total_loss_bits / only:.3f}x"
    )
    axes[1].grid(axis="y", alpha=0.3)

    path = save()
    print("WHAT TO NOTICE")
    print(
        "  The left panel's error bars are narrower than the gaps between the bars, so\n"
        "  the ordering of the three mechanisms is decided and not merely ranked. The\n"
        "  right panel is the whole reason the left one exists: the single number a link\n"
        f"  budget gives is {only / 1e9:.6f} Gbit against a measured total of\n"
        f"  {report.total_loss_gbit:.6f} Gbit, and it cannot say which mechanism to spend\n"
        "  money on because it contains only one of them."
    )
    print("")
    print(f"  baseline 1 (acquisition only)  {only / 1e9:.6f} Gbit")
    print(f"  baseline 2 total               {closed.total_bits / 1e9:.6f} Gbit")
    print(f"  baseline 3 total               {report.total_loss_gbit:.6f} Gbit")
    for name in MECHANISMS:
        share = report.share(name)
        print(
            f"  {name:<14} {share.loss_gbit:.6f} Gbit  "
            f"[{share.low_gbit:.6f}, {share.high_gbit:.6f}]  share {share.share:.4f}"
        )
    print(f"  resampling unit                {report.resampling_unit}")
    print(f"  decomposition residual         {report.residual_bits:.6e} bits "
          f"(tolerance {report.residual_tolerance_bits:.6e})")
    print("")
    print(f"figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
