"""The product's argument in one figure: cost against rate, over the real history.

Writes ``screenshots/cost_versus_rate.png``.

What to notice in the figure: the points are spread vertically at every flake
rate, so the rate does not determine the cost, and the most expensive test in
the suite is not the flakiest. The motivating case -- a 1-in-44 failure in a
600-second test -- is marked, and it sits far to the left and high up: a
dashboard sorted on the horizontal axis puts it near the bottom of the list.

Runs from any working directory. Reads no clock.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.history import load_history  # noqa: E402
from flakerank.ranking import rank_tests  # noqa: E402

PINNED = "soak/test_validation_probe_1_in_44"


def main() -> int:
    """Plot accumulated wasted time against flake rate, and print the headline."""
    _bootstrap.style()
    import matplotlib.pyplot as plt

    history = load_history(ROOT / _bootstrap.HISTORY)
    report = rank_tests(
        history, resamples=_bootstrap.resamples(), seed=_bootstrap.SEED, top_k=10
    )
    names = report.truth.ranking.names
    wasted = report.truth.ranking.scores
    rate = report.rate.scores
    analytic = report.analytic.scores
    index = names.index(PINNED)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    visible = rate > 0.0
    left.scatter(
        rate[visible],
        np.maximum(wasted[visible], 1e-3),
        s=22,
        alpha=0.75,
        edgecolor="none",
        label=f"{int(np.count_nonzero(visible))} tests with an observed failure",
    )
    left.scatter(
        [rate[index]],
        [max(wasted[index], 1e-3)],
        s=120,
        facecolor="none",
        edgecolor="crimson",
        linewidth=1.6,
        label="the 1-in-44 case the mission shipped",
    )
    cost_rank = int(np.count_nonzero(wasted > wasted[index])) + 1
    rate_rank = int(np.count_nonzero(rate > rate[index])) + 1
    left.annotate(
        f"cost rank {cost_rank}, rate rank {rate_rank}",
        (rate[index], max(wasted[index], 1e-3)),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
        color="crimson",
    )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("measured flake rate (dimensionless)")
    left.set_ylabel("accumulated wasted time (s)")
    left.set_title(
        "cost is not determined by rate\n"
        f"Kendall tau-b {report.truth_vs_rate.tau_b:.4f}, "
        f"top-10 overlap {report.truth_vs_rate.top_k_overlap} of 10"
    )
    left.legend(loc="upper left", fontsize=8)

    order = np.argsort(-wasted)[:12]
    positions = np.arange(order.size)
    width = 0.38
    right.barh(
        positions - width / 2,
        wasted[order],
        height=width,
        label="accumulated wasted time (ground truth)",
    )
    right.barh(
        positions + width / 2,
        analytic[order],
        height=width,
        label="analytic expected re-run cost",
    )
    right.set_yticks(positions)
    right.set_yticklabels(
        [names[position].split("/")[-1] for position in order], fontsize=7
    )
    right.invert_yaxis()
    right.set_xlabel("wasted time over 330 sessions (s)")
    right.set_title(
        "the analytic cost model against the truth\n"
        f"Kendall tau-b {report.truth_vs_analytic.tau_b:.6f}, "
        f"max rank move {report.truth_vs_analytic.max_displacement}"
    )
    right.legend(loc="lower right", fontsize=8)

    figure.tight_layout()
    destination = ROOT / _bootstrap.SCREENSHOTS / "cost_versus_rate.png"
    figure.savefig(destination)
    plt.close(figure)

    print("flakerank -- cost against rate over the committed run history")
    print("")
    print(f"tests ranked                        : {report.n_tests}")
    print(f"sessions                            : {report.n_sessions}")
    print(f"accumulated wasted time, s          : {report.truth.total_seconds!r}")
    print(f"  interval, s                       : "
          f"[{report.truth.total_interval.low!r}, {report.truth.total_interval.high!r}]")
    print(f"  resampling unit                   : {report.truth.resampling_unit}, "
          f"{report.truth.total_interval.n_units} of them")
    print("")
    print(f"Kendall tau-b, truth against rate   : {report.truth_vs_rate.tau_b!r}")
    print(f"  interval                          : "
          f"[{report.truth_vs_rate.tau_b_interval.low!r}, "
          f"{report.truth_vs_rate.tau_b_interval.high!r}]")
    print(f"Spearman rho, truth against rate    : {report.truth_vs_rate.rho!r}")
    print(f"top-10 overlap                      : "
          f"{report.truth_vs_rate.top_k_overlap} of 10")
    print(f"largest rank move                   : "
          f"{report.truth_vs_rate.max_displacement} positions")
    print("")
    print(f"the flake-rate top 10 holds this much of the waste : "
          f"{report.capture_by_rate!r}")
    print(f"the analytic top 10 holds                          : "
          f"{report.capture_by_analytic!r}")
    print("")
    print(f"the 1-in-44 case, {PINNED}:")
    print(f"  measured flake rate     : {float(rate[index])!r}")
    print(f"  accumulated wasted time : {float(wasted[index])!r} s")
    print("")
    print(f"figure written to {_bootstrap.relative(destination)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
