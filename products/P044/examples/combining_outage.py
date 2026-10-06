"""Outage against mean SNR for 1 to 4 apertures, independent and correlated.

Panel 1: outage probability against branch mean SNR for MRC with L = 1..4,
drawn twice -- once for the independent-aperture idealisation and once for a
line array whose apertures are correlated by the geometry. Panel 2: the
measured finite-window diversity order against L for the three combiners,
both correlation cases, with the asymptotic L*min(alpha,beta) marked.

What to notice: the correlated curves are not parallel to the independent
ones, they are shallower. At L = 4 the measured slope falls from about 7.5
to about 5.6, and the asymptotic textbook value for this channel is 10.3 --
so quoting either the asymptote or the independent case overstates what a
real array delivers in the outage range a link is designed to.

Writes ../screenshots/combining_outage.png. Runtime about 60 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aperturediv.channel import gamma_gamma_params_from_rytov  # noqa: E402
from aperturediv.combining import (  # noqa: E402
    COMBINERS,
    combined_gain,
    diversity_order,
    outage_probability,
)
from aperturediv.correlation import (  # noqa: E402
    correlation_matrix,
    equispaced_positions,
    sample_correlated_gamma_gamma,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "combining_outage.png"
N = 1_000_000
SEED = 44044
L_MAX = 4
RYTOV = 1.0
SPACING_M = 0.05
RHO_C_M = 0.10
THRESHOLD_DB = 5.0
SNR_GRID = np.arange(0.0, 50.0, 0.25)
WINDOW = (3e-5, 3e-3)
COLOURS = ("#1f3f7a", "#2f8f4f", "#8a5a00", "#a01f1f")


def main() -> int:
    alpha, beta = gamma_gamma_params_from_rytov(RYTOV)
    tail = min(alpha, beta)
    pos = equispaced_positions(L_MAX, SPACING_M)
    r_corr = correlation_matrix(pos, RHO_C_M, "gaussian")
    cases = {
        "independent": np.eye(L_MAX),
        "correlated": r_corr,
    }
    irr = {
        name: sample_correlated_gamma_gamma(
            N, alpha, beta, matrix, np.random.default_rng(SEED)
        )
        for name, matrix in cases.items()
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    orders: dict[str, dict[str, list[float]]] = {}
    print(f"gamma-gamma, Rytov variance {RYTOV}: alpha {alpha:.6f}, beta {beta:.6f}")
    print(f"tail index min(alpha,beta) {tail:.6f}; outage threshold {THRESHOLD_DB} dB")
    print(f"n = {N} realisations per case, seed {SEED}, SNR grid step 0.25 dB")
    print(f"line array pitch {SPACING_M} m, correlation scale {RHO_C_M} m, "
          f"adjacent log correlation {r_corr[0, 1]:.6f}")
    print()
    for name, arr in irr.items():
        ls = "-" if name == "independent" else "--"
        for n_ap in range(1, L_MAX + 1):
            gain = combined_gain(arr[:, :n_ap], "mrc")
            p = outage_probability(gain, SNR_GRID, THRESHOLD_DB)
            visible = p > 3.0 / N
            ax1.semilogy(
                SNR_GRID[visible],
                p[visible],
                ls=ls,
                lw=2.0 if ls == "-" else 1.8,
                color=COLOURS[n_ap - 1],
                label=f"L={n_ap}, {name}",
            )
    ax1.axhline(1.0 / N, color="k", lw=0.8, alpha=0.4)
    ax1.text(0.5, 1.4 / N, f"Monte Carlo floor 1/{N}", fontsize=8)
    ax1.set_xlabel("branch mean SNR (dB)")
    ax1.set_ylabel(f"outage probability, threshold {THRESHOLD_DB:.0f} dB")
    ax1.set_ylim(5e-7, 1.2)
    ax1.set_xlim(0, 40)
    ax1.set_title("MRC outage: independent (solid) vs correlated (dashed)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.5, ncol=2, loc="lower left")

    print("measured finite-window diversity order "
          f"(window {WINDOW[0]:.0e} to {WINDOW[1]:.0e} outage)")
    print("  scheme   case          L=1     L=2     L=3     L=4")
    for scheme in COMBINERS:
        orders[scheme] = {}
        for name, arr in irr.items():
            row: list[float] = []
            for n_ap in range(1, L_MAX + 1):
                gain = combined_gain(arr[:, :n_ap], scheme)
                p = outage_probability(gain, SNR_GRID, THRESHOLD_DB)
                try:
                    row.append(diversity_order(SNR_GRID, p, window=WINDOW).order)
                except ValueError:
                    row.append(float("nan"))
            orders[scheme][name] = row
            print(f"  {scheme:6s}   {name:12s}" + "".join(f"{v:8.3f}" for v in row))
    print()

    ells = np.arange(1, L_MAX + 1)
    markers = {"mrc": "o", "egc": "s", "sc": "^"}
    for scheme in COMBINERS:
        ax2.plot(
            ells,
            orders[scheme]["independent"],
            marker=markers[scheme],
            lw=2.0,
            color=COLOURS[COMBINERS.index(scheme)],
            label=f"{scheme.upper()}, independent",
        )
        ax2.plot(
            ells,
            orders[scheme]["correlated"],
            marker=markers[scheme],
            lw=1.8,
            ls="--",
            mfc="none",
            color=COLOURS[COMBINERS.index(scheme)],
            label=f"{scheme.upper()}, correlated",
        )
    ax2.plot(ells, ells * tail, color="k", ls=":", lw=1.8,
             label=rf"asymptotic $L\,\min(\alpha,\beta)$ = {tail:.2f}L")
    ax2.set_xticks(ells)
    ax2.set_xlabel("number of apertures L")
    ax2.set_ylabel("measured log-log slope of outage vs mean SNR")
    ax2.set_title(f"Diversity order, window {WINDOW[0]:.0e}-{WINDOW[1]:.0e} outage")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print("MRC slope shortfall at L=4:")
    mi = orders["mrc"]["independent"][-1]
    mc = orders["mrc"]["correlated"][-1]
    print(f"  asymptotic             {4 * tail:.3f}")
    print(f"  measured, independent  {mi:.3f}  (shortfall {4 * tail - mi:.3f})")
    print(f"  measured, correlated   {mc:.3f}  (shortfall {4 * tail - mc:.3f})")
    print(f"  cost of assuming independence: {mi - mc:.3f} in slope")
    print()
    print(f"wrote {OUT.parent.name}/{OUT.name}")
    print("research-grade output; not flight-qualified, not certified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
