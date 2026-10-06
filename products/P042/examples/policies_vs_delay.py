"""Example 3: the three non-learned policies against feedback delay.

Writes ``../screenshots/policies_vs_delay.png``.

Four panels:

1. Goodput against feedback delay, with the clairvoyant bound drawn as a flat
   dashed line. Notice the bound is flat: it never reads the feedback, so the
   delay cannot touch it. Notice also how much of the gap to it is already open
   at zero delay -- that part is the granularity of the MODCOD ladder and no
   predictor can recover it.
2. Outage probability against delay. Notice hysteresis is worse than fixed
   margin on outage at every delay, because holding a mode through a dead band
   means holding it into a fade.
3. The dead-band trade at fixed delay: switch rate against outage as the dead
   band widens. Notice the knee -- widening the band past about 4 dB buys almost
   no further outage reduction and costs goodput.
4. Goodput against fixed margin at several delays, with the maximum of each
   curve marked. Notice the maxima are at different margins: there is no single
   margin to publish, which is the whole reason adaptive rate under delay is a
   problem rather than a lookup.

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
from acmpilot.policy import (  # noqa: E402
    CLAIRVOYANT_LABEL,
    FixedMargin,
    ThresholdHysteresis,
    baseline_policies,
)
from acmpilot.simulate import sweep_delay  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "screenshots" / "policies_vs_delay.png"
OUT_REL = "screenshots/policies_vs_delay.png"
TABLE_JSON = HERE.parent / "validation" / "modcod_thresholds.json"
TAUS_MS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0)
N_SLOTS = 12_000
SEEDS = (1, 2, 3)


def _table() -> ModcodTable:
    if TABLE_JSON.exists():
        return ModcodTable.load_json(TABLE_JSON)
    table, _ = measure_thresholds()
    return table


def main() -> int:
    table = _table()
    config = ChannelConfig(slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0)
    policies = baseline_policies()
    rows = sweep_delay(
        table, config, policies, tau_list_s=[t * 1e-3 for t in TAUS_MS],
        n_slots=N_SLOTS, seeds=SEEDS,
    )
    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))

    ax = axes[0, 0]
    for i, (name, series) in enumerate(by_policy.items()):
        taus = [float(r["tau_s"]) * 1e3 for r in series]
        good = [float(r["goodput_bit_per_symbol"]) for r in series]
        err = [float(r["goodput_sem"]) for r in series]
        acausal = name == CLAIRVOYANT_LABEL
        ax.errorbar(
            taus, good, yerr=err, fmt="--" if acausal else "o-", color=f"C{i}",
            lw=1.6, capsize=2, ms=4,
            label="clairvoyant: UPPER BOUND, no causal policy can achieve"
            if acausal else name,
        )
    ax.set_xlabel("round-trip feedback delay tau, ms")
    ax.set_ylabel("goodput, bit/symbol")
    ax.set_title("Goodput against feedback delay, tau_c = 10 ms")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, (name, series) in enumerate(by_policy.items()):
        taus = [float(r["tau_s"]) * 1e3 for r in series]
        out = [float(r["outage_fraction"]) for r in series]
        acausal = name == CLAIRVOYANT_LABEL
        ax.semilogy(
            taus, np.maximum(out, 1e-5), "--" if acausal else "o-", color=f"C{i}",
            lw=1.6, ms=4,
            label="clairvoyant: UPPER BOUND, not achievable" if acausal else name,
        )
    ax.set_xlabel("round-trip feedback delay tau, ms")
    ax.set_ylabel("outage probability")
    ax.set_title("Outage against feedback delay\n(floor is the unavoidable outage of the channel)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    bands = ((0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5), (4.0, 0.5), (6.0, 0.5), (8.0, 0.5))
    sw, outs, goods, labels = [], [], [], []
    for up, down in bands:
        res = sweep_delay(
            table, config, (ThresholdHysteresis(up_margin_db=up, down_margin_db=down),),
            tau_list_s=[10e-3], n_slots=N_SLOTS, seeds=SEEDS,
        )[0]
        sw.append(float(res["switch_rate_per_slot"]))
        outs.append(float(res["outage_fraction"]))
        goods.append(float(res["goodput_bit_per_symbol"]))
        labels.append(f"{up - down:g}")
    sc = ax.scatter(outs, sw, c=goods, cmap="viridis", s=70, zorder=3)
    ax.plot(outs, sw, "-", color="0.6", lw=1.0, zorder=2)
    for x, y, lab in zip(outs, sw, labels, strict=True):
        ax.annotate(
            f"{lab} dB", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7
        )
    fig.colorbar(sc, ax=ax, label="goodput, bit/symbol")
    ax.set_xlabel("outage probability")
    ax.set_ylabel("MODCOD switches per slot")
    ax.set_title(
        "Hysteresis dead band: chatter against outage, tau = 10 ms\n"
        "(labels are dead-band width)"
    )
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    margins = np.arange(0.0, 7.01, 0.5)
    for i, tau_ms in enumerate((0.0, 2.0, 5.0, 10.0, 20.0, 40.0)):
        good = [
            float(
                sweep_delay(
                    table, config, (FixedMargin(margin_db=float(m)),),
                    tau_list_s=[tau_ms * 1e-3], n_slots=N_SLOTS, seeds=SEEDS,
                )[0]["goodput_bit_per_symbol"]
            )
            for m in margins
        ]
        ax.plot(margins, good, "-", color=f"C{i}", lw=1.4, label=f"tau = {tau_ms:g} ms")
        best = int(np.argmax(good))
        ax.plot(margins[best], good[best], "*", color=f"C{i}", ms=13)
    ax.set_xlabel("fixed margin, dB")
    ax.set_ylabel("goodput, bit/symbol")
    ax.set_title("No single margin is right: the optimum (stars) moves with delay")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "acmpilot: three non-learned rate-adaptation policies under round-trip "
        "feedback delay. Research-grade, simulated channel.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    for name, series in by_policy.items():
        first = float(series[0]["goodput_bit_per_symbol"])
        last = float(series[-1]["goodput_bit_per_symbol"])
        print(
            f"{name[:50]:52s} goodput {first:.4f} at tau=0 ms -> {last:.4f} at "
            f"tau={TAUS_MS[-1]:g} ms"
        )
    print(f"written: {OUT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
