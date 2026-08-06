"""Level-2 evidence: simulated sigma_I^2 vs Rytov theory across the campaign.

Reads validation/dataset.csv (produced by run_campaign.py) and compares
simulated scintillation indices against the analytic weak-fluctuation
values (Andrews & Phillips 2005; Andrews 1992 aperture factor). Writes
validation/sim_vs_theory.txt with per-point ratios and summary statistics.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scintinet import rytov_variance  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader((HERE / "dataset.csv").open()))
    if not rows:
        raise SystemExit("dataset.csv is empty; run run_campaign.py first")

    lines = [
        "Simulator validation: split-step sigma_I^2 vs Rytov weak-fluctuation theory",
        "Theory: plane-wave sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6) (Andrews & Phillips 2005)",
        "Aperture factor: A = [1+1.062 kD^2/(4L)]^(-7/6) (Andrews 1992)",
        "",
        f"{'Cn2':>10} {'L_m':>7} {'lam_m':>9} {'D_m':>6} "
        f"{'sim':>10} {'theory':>10} {'ratio':>6}",
    ]
    point_ratios, ap_ratios = [], []
    seen_points = set()
    for r in rows:
        cn2 = float(r["cn2"])
        ell = float(r["path_length_m"])
        lam = float(r["wavelength_m"])
        dia = float(r["aperture_d_m"])
        sim = float(r["sigma_i2_sim"])
        theory = float(r["sigma_i2_rytov"])
        ratio = sim / theory
        lines.append(
            f"{cn2:>10.2e} {ell:>7.0f} {lam:>9.2e} {dia:>6.3f} "
            f"{sim:>10.4e} {theory:>10.4e} {ratio:>6.2f}"
        )
        if dia <= 0.005:
            ap_ratios_target = point_ratios
        else:
            ap_ratios_target = ap_ratios
        ap_ratios_target.append(ratio)

        key = (cn2, ell, lam)
        if key not in seen_points:
            seen_points.add(key)
            sr2 = rytov_variance(cn2, lam, ell)
            pt_ratio = float(r["sigma_i2_sim_point"]) / sr2
            lines.append(
                f"{'':>10} {'':>7} {'':>9} {'point':>6} "
                f"{float(r['sigma_i2_sim_point']):>10.4e} {sr2:>10.4e} {pt_ratio:>6.2f}"
            )

    pr = np.array(point_ratios)
    ar = np.array(ap_ratios)
    lines += [
        "",
        "Summary (sim/theory ratio):",
        f"  point-like aperture (D=2 mm), n={pr.size}: "
        f"mean={pr.mean():.3f}, min={pr.min():.3f}, max={pr.max():.3f}",
        f"  finite apertures (D=50,100 mm), n={ar.size}: "
        f"mean={ar.mean():.3f}, min={ar.min():.3f}, max={ar.max():.3f}",
        "",
        "PASS criterion (stated in advance): point-like sim/theory mean within",
        "[0.6, 1.4] and every point within [0.5, 1.6] (weak-regime, reduced grid).",
        f"RESULT: mean {'PASS' if 0.6 <= pr.mean() <= 1.4 else 'FAIL'}, "
        f"all-points {'PASS' if (pr.min() >= 0.5 and pr.max() <= 1.6) else 'FAIL'}",
        "",
        "Known bias: FFT screens lack sub-fundamental spatial frequencies (no",
        "subharmonics), so large-aperture indices fall below the Andrews",
        "approximation; the Andrews factor is itself approximate. Reported as-is.",
    ]
    (HERE / "sim_vs_theory.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-9:]))


if __name__ == "__main__":
    main()
