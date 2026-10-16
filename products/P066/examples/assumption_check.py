"""What the assumption check finds, and how wrong RSS is when it is wrong.

Writes ``screenshots/assumption_check.png``. The left panel sweeps the
declared correlation between the two payload items of the CubeSat power
budget and plots the exact root uncertainty against the RSS value the
declaration asserts. The right panel counts the findings raised on each
shipped case by severity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from _bootstrap import CASES, SCREENSHOTS
from _plotstyle import EXACT, MUTED, OVERRUN, RSS, finish, plt

from budgetroll import Correlation, load, rollup, run_checks, severity_counts

CASE_FILES = [
    "clean_mass.yaml",
    "cubesat_link.yaml",
    "cubesat_power.yaml",
    "planted_findings.yaml",
]


def main() -> Path:
    """Build the figure and return the path written."""
    budget = load(CASES / "cubesat_power.yaml")
    rhos = np.linspace(-1.0, 1.0, 81)
    exact = []
    for rho in rhos:
        altered = replace(
            budget,
            correlations=(
                Correlation(
                    "payload_detector",
                    "payload_fpga",
                    float(rho),
                    "swept by examples/assumption_check.py",
                ),
            ),
        )
        exact.append(rollup(altered).root.exact_uncertainty)
    declared = rollup(budget).root.uncertainty

    fig, (ax_sweep, ax_counts) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax_sweep.plot(rhos, exact, color=EXACT, lw=1.8, label="exact form with the declared rho")
    ax_sweep.axhline(
        declared, color=RSS, lw=1.5, ls="--",
        label=f"RSS the declaration asserts = {declared:.6f} {budget.unit}",
    )
    ax_sweep.axvline(0.6, color=MUTED, lw=1.0, ls=":")
    at_declared = rollup(budget).root.exact_uncertainty
    ax_sweep.plot([0.6], [at_declared], "o", color=OVERRUN, ms=6)
    ax_sweep.annotate(
        f"declared rho = 0.6\nexact {at_declared:.6f} {budget.unit}\n"
        f"RSS understates by "
        f"{100.0 * (at_declared - declared) / at_declared:.2f} %",
        xy=(0.6, at_declared),
        xytext=(-0.92, at_declared * 1.0),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": OVERRUN, "lw": 1.0},
    )
    ax_sweep.set_xlabel("declared correlation between the two payload items")
    ax_sweep.set_ylabel(f"root 1 sigma [{budget.unit}]")
    ax_sweep.set_title("What RSS costs when independence is not true", fontsize=10)
    ax_sweep.legend(fontsize=7, loc="lower right")
    ax_sweep.grid(alpha=0.25)

    names, errors, warnings, infos = [], [], [], []
    for name in CASE_FILES:
        counts = severity_counts(run_checks(load(CASES / name)))
        names.append(name.replace(".yaml", ""))
        errors.append(counts["error"])
        warnings.append(counts["warning"])
        infos.append(counts["info"])
    x = np.arange(len(names))
    width = 0.26
    ax_counts.bar(x - width, errors, width, color=OVERRUN, label="error")
    ax_counts.bar(x, warnings, width, color="#c08a2e", label="warning")
    ax_counts.bar(x + width, infos, width, color=MUTED, label="info")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(names, fontsize=7, rotation=12)
    ax_counts.set_ylabel("findings")
    ax_counts.set_title("Findings raised on each shipped case", fontsize=10)
    ax_counts.legend(fontsize=7)
    ax_counts.grid(axis="y", alpha=0.25)
    for pos, (e, w, i) in enumerate(zip(errors, warnings, infos, strict=True)):
        for offset, value in ((-width, e), (0.0, w), (width, i)):
            ax_counts.text(pos + offset, value + 0.08, str(value), ha="center", fontsize=7)

    fig.suptitle(
        "The independence assumption RSS requires, checked against the declared "
        "correlation structure.",
        fontsize=10,
    )
    return finish(fig, SCREENSHOTS / "assumption_check.png")


if __name__ == "__main__":
    print(f"wrote {main()}")
