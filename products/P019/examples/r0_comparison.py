"""Example 2: Fried parameter derived from predicted profiles vs truth and HV 5/7.

For every held-out scenario the r0 implied by the predicted median profile is
compared with the r0 implied by the synthetic truth profile and with the single
fixed value HV 5/7 gives (which cannot vary at all).  The right-hand panel shows
the diurnal behaviour that a fixed climatology cannot represent.

    python examples/r0_comparison.py

Writes ``screenshots/r0_comparison.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cncast.baselines import hv57  # noqa: E402
from cncast.dataset import default_altitude_grid, profile_cn2  # noqa: E402
from cncast.model import train_default_model  # noqa: E402
from cncast.seeing import fried_parameter  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "r0_comparison.png"
LAM = 500e-9


def main() -> int:
    model, art = train_default_model()
    grid = default_altitude_grid(48)
    scenarios = art["test_scenarios"]

    r0_pred, r0_true, r0_lo, r0_hi, hours = [], [], [], [], []
    for sc in scenarios:
        pred = model.predict_scenario(sc, grid)
        r0_pred.append(fried_parameter(grid, pred.cn2, LAM) * 100.0)
        r0_true.append(fried_parameter(grid, profile_cn2(sc, grid), LAM) * 100.0)
        # a stronger profile gives a smaller r0, so bounds swap
        r0_lo.append(fried_parameter(grid, pred.cn2_upper, LAM) * 100.0)
        r0_hi.append(fried_parameter(grid, pred.cn2_lower, LAM) * 100.0)
        hours.append(sc.hour_of_day)
    r0_pred = np.array(r0_pred)
    r0_true = np.array(r0_true)
    r0_lo = np.array(r0_lo)
    r0_hi = np.array(r0_hi)
    hours = np.array(hours)
    r0_hv = fried_parameter(grid, hv57(grid), LAM) * 100.0

    inside = np.mean((r0_true >= r0_lo) & (r0_true <= r0_hi))
    err_ml = np.sqrt(np.mean((r0_pred - r0_true) ** 2))
    err_hv = np.sqrt(np.mean((r0_hv - r0_true) ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    lim = [0.9 * min(r0_true.min(), r0_lo.min()), 1.1 * max(r0_true.max(), r0_hi.max())]
    ax1.plot(lim, lim, "k--", lw=1.0, label="perfect agreement")
    ax1.vlines(r0_true, r0_lo, r0_hi, color="tab:blue", alpha=0.25, lw=1.0,
               label="r$_0$ from the 90 % $C_n^2$ band")
    ax1.plot(r0_true, r0_pred, "o", ms=4, color="tab:blue", label="CnCast median")
    ax1.axhline(r0_hv, color="tab:red", lw=1.5,
                label=f"HV 5/7 (fixed, {r0_hv:.2f} cm)")
    ax1.set_xlabel("r$_0$ from synthetic truth profile [cm]")
    ax1.set_ylabel("r$_0$ from predicted profile [cm]")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(
        f"Derived r$_0$ at 500 nm, zenith, {len(scenarios)} held-out scenarios\n"
        f"RMSE: CnCast {err_ml:.2f} cm vs HV 5/7 {err_hv:.2f} cm; "
        f"band contains truth {inside:.0%} of the time",
        fontsize=10,
    )

    ax2.plot(hours, r0_true, "o", ms=3.5, color="0.55", alpha=0.7, label="synthetic truth")
    ax2.plot(hours, r0_pred, "x", ms=4, color="tab:blue", alpha=0.6, label="CnCast median")
    edges = np.arange(0.0, 25.0, 3.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    med_true, med_pred = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (hours >= lo) & (hours < hi)
        med_true.append(np.median(r0_true[m]) if m.any() else np.nan)
        med_pred.append(np.median(r0_pred[m]) if m.any() else np.nan)
    ax2.plot(centres, med_true, "-o", color="k", lw=2.0, label="truth, 3 h median")
    ax2.plot(centres, med_pred, "-s", color="tab:blue", lw=2.0, label="CnCast, 3 h median")
    ax2.axhline(r0_hv, color="tab:red", lw=1.5, label="HV 5/7 (no diurnal cycle)")
    ax2.set_xlabel("hour of day")
    ax2.set_ylabel("r$_0$ [cm]")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 3))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title(
        "r$_0$ against hour of day: the 3 h medians show the diurnal cycle\n"
        "a single fixed climatology curve cannot express",
        fontsize=10,
    )

    fig.suptitle(
        "Integrated seeing derived from predicted profiles — synthetic data, "
        "not measurements (DATASET_CARD.md)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    sys.stdout.write(
        f"wrote {OUT}\n"
        f"r0 RMSE  CnCast {err_ml:.3f} cm   HV 5/7 {err_hv:.3f} cm\n"
        f"r0 truth inside the derived band: {inside:.3f}\n"
        f"r0 truth range {r0_true.min():.2f}-{r0_true.max():.2f} cm\n"
        f"r0 mean bias (pred - truth): {float(np.mean(r0_pred - r0_true)):+.3f} cm\n"
        f"3 h median r0 (truth) by hour bin: "
        + ", ".join(f"{c:.1f}h={v:.2f}" for c, v in zip(centres, med_true, strict=True))
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
