"""Print the closed-form detection-power table and plot it.

The figure is the product's central claim in one picture: how many independent
repeats a reproducibility check needs before agreement means anything, and what
the honest upper bound on the escaped rate is after n agreeing runs.

Writes ``screenshots/detection_power.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from seedaudit.power import (  # noqa: E402
    MISSION_RATE,
    detection_probability_pairwise,
    detection_probability_reference,
    escaped_rate_upper_bound,
    repeats_for_detection,
)
from seedaudit.report import format_power_table  # noqa: E402

#: Rates the left panel draws a curve for.
CURVE_RATES = (0.5, 0.1, MISSION_RATE, 0.01, 0.001)


def main() -> int:
    """Print the table and write the figure."""
    print(format_power_table())
    print()
    print(f"p = 1/44 = {MISSION_RATE!r} is the rate this mission actually shipped.")
    pairwise = detection_probability_pairwise(MISSION_RATE, 2)
    reference = detection_probability_reference(MISSION_RATE, 2)
    print(f"run-twice-and-diff detection power     : {pairwise!r}")
    print(f"run-twice against a committed reference: {reference!r}")
    print(f"repeats needed for 95 per cent detection: {repeats_for_detection(MISSION_RATE)}")
    print(f"escaped-rate upper bound after 2 agreeing runs: {escaped_rate_upper_bound(2)!r}")

    repeats = np.arange(1, 401)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    left = axes[0]
    for rate in CURVE_RATES:
        curve = [detection_probability_reference(rate, int(n)) for n in repeats]
        label = "p = 1/44 (shipped)" if rate == MISSION_RATE else f"p = {rate:g}"
        left.plot(repeats, curve, lw=1.6, label=label)
    left.axhline(0.95, color="0.3", ls="--", lw=1.0)
    left.annotate("0.95", xy=(400, 0.95), xytext=(-24, 6), textcoords="offset points",
                  color="0.3", fontsize=8)
    needed = repeats_for_detection(MISSION_RATE)
    left.axvline(needed, color="0.3", ls=":", lw=1.0)
    left.annotate(f"n = {needed}", xy=(needed, 0.60), xytext=(6, 0), textcoords="offset points",
                  color="0.3", fontsize=8)
    left.set_xscale("log")
    left.set_xlabel("independent repeats n")
    left.set_ylabel("detection probability")
    left.set_title("What agreement is worth: 1 - (1-p)$^n$")
    left.set_ylim(-0.02, 1.02)
    left.legend(loc="upper left", fontsize=8)

    right = axes[1]
    bound = [escaped_rate_upper_bound(int(n)) for n in repeats]
    right.plot(repeats, bound, lw=1.8, color="#b2182b", label=r"$p_{max} = 1 - \alpha^{1/n}$")
    right.plot(repeats, 3.0 / repeats, lw=1.1, ls="--", color="0.45", label="rule of three, 3/n")
    right.axhline(MISSION_RATE, color="#2166ac", lw=1.0, ls=":")
    right.annotate("1/44, the rate shipped", xy=(1.2, MISSION_RATE), xytext=(0, 6),
                   textcoords="offset points", color="#2166ac", fontsize=8)
    right.scatter([2], [escaped_rate_upper_bound(2)], s=26, color="#b2182b", zorder=5)
    right.annotate(f"n = 2: {escaped_rate_upper_bound(2):.4f}", xy=(2, escaped_rate_upper_bound(2)),
                   xytext=(12, 12), textcoords="offset points", fontsize=8)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("agreeing repeats n")
    right.set_ylabel("upper bound on the escaped rate")
    right.set_title(r"What a clean audit has earned ($\alpha$ = 0.05)")
    right.legend(loc="lower left", fontsize=8)

    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "detection_power.png"
    figure.savefig(target)
    plt.close(figure)
    print()
    print(f"figure written to {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
