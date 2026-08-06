"""Example 1: full BeamTwin report for a 10 km terrestrial FSO link.

Prints the text report, writes the JSON report next to it, and saves a
budget-waterfall PNG to ../screenshots/.

Run: python examples/run_full_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beamtwin.scenario import (  # noqa: E402
    format_report_text,
    load_scenario,
    report_to_json,
    run_twin,
)
from beamtwin.surrogate import FadeSurrogate, default_model_path  # noqa: E402

SHOTS = ROOT / "screenshots"


def main() -> None:
    scenario = load_scenario(ROOT / "examples" / "link_10km.yaml")
    surrogate = None
    if default_model_path().exists():
        surrogate = FadeSurrogate.load(default_model_path())
    report = run_twin(scenario, surrogate=surrogate)
    print(format_report_text(report))

    out_json = ROOT / "examples" / "link_10km_report.json"
    out_json.write_text(report_to_json(report), encoding="utf-8")
    print(f"\nJSON report -> {out_json}")

    # Budget waterfall
    b = report["budget"]
    labels = [
        "Tx power",
        "Tx optics",
        "Geometric",
        "Pointing\n(bias)",
        "Atmosphere",
        "Rx optics",
        "Rx power",
    ]
    deltas = [
        b["tx_power_dbm"],
        -b["tx_optics_loss_db"],
        -b["geometric_loss_db"],
        -b["pointing_loss_db"],
        -b["atmospheric_loss_db"],
        -b["rx_optics_loss_db"],
    ]
    SHOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    running = 0.0
    for i, d in enumerate(deltas):
        colour = "#2b7bba" if i == 0 else "#c0392b"
        bottom = 0.0 if i == 0 else min(running, running + d)
        ax.bar(i, abs(d), bottom=bottom, color=colour, edgecolor="black", linewidth=0.5)
        ax.text(
            i,
            bottom + abs(d) + 0.6,
            f"{d:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        running += d
    ax.bar(
        len(deltas),
        abs(running),
        bottom=min(0.0, running),
        color="#27ae60",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.text(
        len(deltas),
        min(0.0, running) + abs(running) + 0.6,
        f"{running:+.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.axhline(
        b["rx_sensitivity_dbm"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Rx sensitivity {b['rx_sensitivity_dbm']:.0f} dBm",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("power [dBm] / loss [dB]")
    ax.set_title(
        f"BeamTwin link budget — {report['name']}  "
        f"(margin {b['margin_db']:+.2f} dB)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = SHOTS / "budget_waterfall_10km.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Waterfall plot -> {out}")


if __name__ == "__main__":
    main()
