"""Example 4: the two kinds of mis-selection, which a single number would hide.

Writes ``../screenshots/misselection_split.png``.

Four panels:

1. Selection outcome composition against feedback delay for the tuned
   fixed-margin policy: exact, too conservative, avoidably too aggressive, and
   unavoidable (the channel supported nothing). Notice that the conservative
   share is the large one at every delay: a margin that keeps outage tolerable
   necessarily spends most slots below capacity.
2. The same for hysteresis. Notice the composition is different at the same
   goodput -- hysteresis trades conservatism for aggression -- which is exactly
   why reporting one "mis-selection rate" for both policies would be useless.
3. The cost of each kind, in bit/symbol: wasted (chose below capacity) against
   lost (chose above it). Notice the two are the same order of magnitude, so
   neither can be ignored, and that they move in opposite directions as the
   margin changes.
4. The margin sweep in cost space at tau = 10 ms: wasted on one axis, lost on
   the other, goodput as colour. Notice the frontier -- every point on it is a
   different answer to "how much outage will you accept", and the goodput maximum
   is not at either end.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acmpilot.channel import ChannelConfig  # noqa: E402
from acmpilot.modcod import ModcodTable, measure_thresholds  # noqa: E402
from acmpilot.policy import FixedMargin, ThresholdHysteresis  # noqa: E402
from acmpilot.simulate import sweep_delay  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "screenshots" / "misselection_split.png"
OUT_REL = "screenshots/misselection_split.png"
TABLE_JSON = HERE.parent / "validation" / "modcod_thresholds.json"
TAUS_MS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
N_SLOTS = 12_000
SEEDS = (1, 2, 3)
KINDS = (
    ("exact_fraction", "exact: chose the best supported MODCOD", "C2"),
    ("conservative_fraction", "too conservative: chose below capacity", "C0"),
    ("avoidable_outage_fraction", "too aggressive, avoidable: outage", "C3"),
    ("unavoidable_outage_fraction", "unavoidable: channel supported nothing", "0.5"),
)


def _table() -> ModcodTable:
    if TABLE_JSON.exists():
        return ModcodTable.load_json(TABLE_JSON)
    table, _ = measure_thresholds()
    return table


def _series(table: ModcodTable, config: ChannelConfig, policy) -> list[dict]:
    return sweep_delay(
        table, config, (policy,), tau_list_s=[t * 1e-3 for t in TAUS_MS],
        n_slots=N_SLOTS, seeds=SEEDS,
    )


def _stack(ax, series: list[dict], title: str) -> None:
    taus = np.array([float(r["tau_s"]) * 1e3 for r in series])
    bottom = np.zeros_like(taus)
    width = np.full_like(taus, 2.2)
    for key, label, colour in KINDS:
        values = np.array([float(r[key]) for r in series])
        ax.bar(taus, values, bottom=bottom, width=width, color=colour, label=label)
        bottom += values
    ax.set_xlabel("round-trip feedback delay tau, ms")
    ax.set_ylabel("fraction of slots")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25, axis="y")


def main() -> int:
    table = _table()
    config = ChannelConfig(slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0)
    fixed_series = _series(table, config, FixedMargin(margin_db=2.0))
    hyst_series = _series(
        table, config, ThresholdHysteresis(up_margin_db=4.0, down_margin_db=0.5)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    _stack(axes[0, 0], fixed_series, "fixed margin 2 dB (tuned): selection outcomes")
    _stack(
        axes[0, 1], hyst_series,
        "hysteresis +4/-0.5 dB (tuned): selection outcomes\nsame goodput, different composition",
    )

    ax = axes[1, 0]
    taus = np.array([float(r["tau_s"]) * 1e3 for r in fixed_series])
    for series, label, marker in (
        (fixed_series, "fixed margin 2 dB", "o"),
        (hyst_series, "hysteresis +4/-0.5 dB", "s"),
    ):
        ax.plot(
            taus, [float(r["wasted_bit_per_symbol"]) for r in series],
            marker + "-", lw=1.5, label=f"{label}: wasted (too conservative)",
        )
        ax.plot(
            taus, [float(r["lost_bit_per_symbol"]) for r in series],
            marker + "--", lw=1.5, label=f"{label}: lost (too aggressive)",
        )
    ax.set_xlabel("round-trip feedback delay tau, ms")
    ax.set_ylabel("spectral efficiency forgone, bit/symbol")
    ax.set_title(
        "The cost of each kind of mis-selection\n"
        "both matter; one number for both would hide this"
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    margins = np.arange(0.0, 6.01, 0.5)
    wasted, lost, good = [], [], []
    for margin in margins:
        res = sweep_delay(
            table, config, (FixedMargin(margin_db=float(margin)),),
            tau_list_s=[10e-3], n_slots=N_SLOTS, seeds=SEEDS,
        )[0]
        wasted.append(float(res["wasted_bit_per_symbol"]))
        lost.append(float(res["lost_bit_per_symbol"]))
        good.append(float(res["goodput_bit_per_symbol"]))
    sc = ax.scatter(wasted, lost, c=good, cmap="viridis", s=70, zorder=3)
    ax.plot(wasted, lost, "-", color="0.6", lw=1.0, zorder=2)
    best = int(np.argmax(good))
    ax.plot(
        wasted[best], lost[best], "r*", ms=18, zorder=4,
        label=f"goodput maximum at margin {margins[best]:g} dB",
    )
    for i in (0, best, len(margins) - 1):
        ax.annotate(
            f"{margins[i]:g} dB", (wasted[i], lost[i]), textcoords="offset points",
            xytext=(7, 5), fontsize=7,
        )
    fig.colorbar(sc, ax=ax, label="goodput, bit/symbol")
    ax.set_xlabel("wasted, bit/symbol (too conservative)")
    ax.set_ylabel("lost, bit/symbol (too aggressive)")
    ax.set_title(
        "Margin sweep in cost space, tau = 10 ms\n"
        "the optimum is interior, not at either end"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "acmpilot: mis-selection split into too-aggressive and too-conservative, "
        "with their separate costs. Research-grade, simulated channel.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    for label, series in (
        ("fixed margin 2 dB", fixed_series),
        ("hysteresis +4/-0.5 dB", hyst_series),
    ):
        last = series[-1]
        print(
            f"{label:22s} at tau=40 ms: exact={float(last['exact_fraction']):.4f} "
            f"conservative={float(last['conservative_fraction']):.4f} "
            f"aggressive_avoidable={float(last['avoidable_outage_fraction']):.4f} "
            f"wasted={float(last['wasted_bit_per_symbol']):.4f} "
            f"lost={float(last['lost_bit_per_symbol']):.4f}"
        )
    print(f"margin sweep goodput maximum at {margins[best]:g} dB: {good[best]:.4f} bit/symbol")
    print(f"written: {OUT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
