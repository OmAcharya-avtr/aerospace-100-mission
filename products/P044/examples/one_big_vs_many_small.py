"""One big aperture against several small ones, total glass held fixed.

The question this package exists for. A receiver has a fixed total
collecting area. Splitting it into n apertures of equal total area
(d = D/sqrt(n)) buys partial decorrelation between branches, and pays for it
twice: each small aperture averages the scintillation less (its own
scintillation index rises from A(D)si0 towards A(d)si0), and each branch
receives only 1/n of the mean power.

Panel 1: outage against total mean SNR for n = 1..4 at a fixed total
diameter, with the per-aperture scintillation index annotated. Panel 2: the
total mean SNR needed for 1e-3 outage against n, for three array pitches, so
the reader can see that the answer depends on how far apart the apertures
can physically go.

What to notice: splitting helps, but only when the pitch is a sizeable
fraction of the correlation scale. At the tightest pitch the apertures are
still strongly correlated and most of the diversity gain does not appear.

Writes ../screenshots/one_big_vs_many_small.png. Runtime about 70 s on one
core.
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

from aperturediv.aperture import (  # noqa: E402
    effective_scintillation_index,
    equal_area_diameter,
    fresnel_scale,
)
from aperturediv.combining import mrc_gain, outage_probability  # noqa: E402
from aperturediv.correlation import (  # noqa: E402
    correlation_matrix,
    equispaced_positions,
    sample_correlated_lognormal,
)

OUT = (
    pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "one_big_vs_many_small.png"
)
N = 800_000
SEED = 44044
TOTAL_DIAMETER_M = 0.20
WAVELENGTH_M = 1.55e-6
PATH_LENGTH_M = 10_000.0
POINT_SI = 0.6
THRESHOLD_DB = 5.0
TARGET_OUTAGE = 1e-3
SNR_GRID = np.arange(0.0, 45.0, 0.25)
PITCH_FACTORS = (1.05, 1.5, 3.0)  # array pitch as a multiple of the small-aperture diameter
COLOURS = ("#1f3f7a", "#2f8f4f", "#8a5a00", "#a01f1f")


def _snr_for_outage(gain: np.ndarray, target: float) -> float:
    return float(THRESHOLD_DB - 10.0 * np.log10(float(np.quantile(gain, target))))


def main() -> int:
    rho_c = fresnel_scale(WAVELENGTH_M, PATH_LENGTH_M)
    print(f"lambda {WAVELENGTH_M:.3e} m, path {PATH_LENGTH_M:.0f} m")
    print(f"correlation scale rho_c = sqrt(lambda L) = {rho_c:.6f} m")
    print(f"point scintillation index si(0) = {POINT_SI}")
    print(f"total aperture diameter held at {TOTAL_DIAMETER_M} m")
    print(f"n = {N} realisations per configuration, seed {SEED}")
    print(f"outage threshold {THRESHOLD_DB} dB, target outage {TARGET_OUTAGE:.0e}")
    print()
    print("  n   d each (m)     A(d)   si per aperture   branch share of mean power")
    for n_ap in range(1, 5):
        d_each = equal_area_diameter(TOTAL_DIAMETER_M, n_ap)
        si_each = effective_scintillation_index(POINT_SI, d_each, rho_c)
        print(f"{n_ap:3d} {d_each:12.5f} "
              f"{si_each / POINT_SI:9.6f} {si_each:16.6f} {1.0 / n_ap:27.4f}")
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    pitch_mid = PITCH_FACTORS[1]
    print(f"-- panel 1, array pitch = {pitch_mid:.1f} x the small-aperture diameter")
    print("  n   pitch (m)  pitch/rho_c   adj log corr   SNR for 1e-3 outage (dB)"
          "   gain over n=1 (dB)")
    ref = None
    for n_ap in range(1, 5):
        d_each = equal_area_diameter(TOTAL_DIAMETER_M, n_ap)
        si_each = effective_scintillation_index(POINT_SI, d_each, rho_c)
        pitch = pitch_mid * d_each
        pos = equispaced_positions(n_ap, pitch)
        r_log = correlation_matrix(pos, rho_c, "gaussian")
        arr = sample_correlated_lognormal(N, si_each, r_log, np.random.default_rng(SEED))
        # fixed total power: branch mean SNR is total / n, so the combining gain
        # relative to the TOTAL mean SNR is sum(I)/n
        gain_total = mrc_gain(arr) / n_ap
        p = outage_probability(gain_total, SNR_GRID, THRESHOLD_DB)
        visible = p > 3.0 / N
        ax1.semilogy(
            SNR_GRID[visible],
            p[visible],
            lw=2.0,
            color=COLOURS[n_ap - 1],
            label=f"n={n_ap}, d={d_each * 100:.1f} cm, si={si_each:.3f}",
        )
        snr_needed = _snr_for_outage(gain_total, TARGET_OUTAGE)
        if n_ap == 1:
            ref = snr_needed
        adj = r_log[0, 1] if n_ap > 1 else float("nan")
        print(f"{n_ap:3d} {pitch:11.5f} {pitch / rho_c:12.4f} {adj:14.6f} "
              f"{snr_needed:26.3f} {(ref or 0.0) - snr_needed:20.3f}")
    ax1.set_xlabel("total mean SNR (dB), summed over all apertures")
    ax1.set_ylabel(f"outage probability, threshold {THRESHOLD_DB:.0f} dB")
    ax1.set_ylim(5e-7, 1.2)
    ax1.set_xlim(0, 40)
    ax1.set_title(
        f"Fixed total glass D={TOTAL_DIAMETER_M} m, pitch {pitch_mid:.0f}d, MRC"
    )
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, loc="lower left")

    print()
    print("-- panel 2, the answer depends on the pitch")
    print("  pitch factor   n=1 (dB)   n=2 (dB)   n=3 (dB)   n=4 (dB)   best n")
    for j, factor in enumerate(PITCH_FACTORS):
        needed = []
        for n_ap in range(1, 5):
            d_each = equal_area_diameter(TOTAL_DIAMETER_M, n_ap)
            si_each = effective_scintillation_index(POINT_SI, d_each, rho_c)
            pos = equispaced_positions(n_ap, factor * d_each)
            r_log = correlation_matrix(pos, rho_c, "gaussian")
            arr = sample_correlated_lognormal(
                N, si_each, r_log, np.random.default_rng(SEED)
            )
            needed.append(_snr_for_outage(mrc_gain(arr) / n_ap, TARGET_OUTAGE))
        best = int(np.argmin(needed)) + 1
        print(f"{factor:14.2f}" + "".join(f"{v:11.3f}" for v in needed) + f"{best:9d}")
        ax2.plot(
            np.arange(1, 5),
            needed,
            marker="o",
            lw=2.0,
            color=COLOURS[j],
            label=f"pitch = {factor:.2f} x d",
        )
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xlabel("number of apertures n, total area held fixed")
    ax2.set_ylabel(f"total mean SNR for {TARGET_OUTAGE:.0e} outage (dB)")
    ax2.set_title("Splitting the glass pays only if the pitch is wide enough")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print()
    print(f"wrote {OUT.parent.name}/{OUT.name}")
    print("research-grade output; not flight-qualified, not certified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
