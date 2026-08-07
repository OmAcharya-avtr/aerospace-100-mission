"""Fit a synthetic turbulent wavefront to Zernike coefficients and show the residual.

The truth wavefront is a seeded random Zernike expansion with per-mode
variances taken from :func:`zernkit.coefficient_variance` (Kolmogorov, Noll
1976) at ``D/r0 = 8``, carried out to Noll ``j = 120`` so that there is genuine
content the fitting basis cannot represent. The fit uses only the first 21
modes, so the residual is dominated by unfitted high-order turbulence -- the
"fitting error" of adaptive-optics error budgets.

The measured residual variance is compared against the analytic Noll
``Delta_21``, which is the point of the figure: the two should agree to within
the sampling and finite-series error, and any large gap means an indexing or
normalisation mistake.

Run from the product root:

    python examples/wavefront_fit.py

Writes ``screenshots/wavefront_fit_residual.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import (  # noqa: E402
    coefficient_variance_noll,
    fit_wavefront,
    mode_list,
    residual_variance,
    unit_disc_grid,
    zernike_design_matrix,
)

SEED = 20260807
D_OVER_R0 = 8.0
J_TRUTH = 120  # Noll modes used to synthesise the truth wavefront
J_FIT = 21  # Noll modes used in the fit
N_PIX = 192


def main() -> Path:
    rng = np.random.default_rng(SEED)
    x, y, mask = unit_disc_grid(N_PIX)
    xm, ym = x[mask], y[mask]

    # Truth: j = 2..J_TRUTH with Kolmogorov variances (piston excluded).
    truth_indices = mode_list(J_TRUTH)[1:]
    sigmas = np.sqrt(
        [coefficient_variance_noll(j, d_over_r0=D_OVER_R0) for j in range(2, J_TRUTH + 1)]
    )
    truth_coeffs = rng.normal(scale=sigmas)
    truth = zernike_design_matrix(truth_indices, xm, ym) @ truth_coeffs

    fit = fit_wavefront(xm, ym, truth, J_FIT)
    fitted = truth - fit.residual

    measured_residual_var = float(np.mean(fit.residual**2))
    analytic_residual_var = residual_variance(J_FIT, d_over_r0=D_OVER_R0)
    truncated_tail = sum(
        coefficient_variance_noll(j, d_over_r0=D_OVER_R0) for j in range(J_TRUTH + 1, 100_000)
    )

    # Exact ensemble mean of the residual variance for this sampling and this
    # truncated truth basis: project every truth column out of the fit basis
    # once, then E[||M c||^2]/N = sum_i sigma_i^2 * mean(M[:, i]^2) because the
    # coefficients are independent zero-mean Gaussians.
    truth_matrix = zernike_design_matrix(truth_indices, xm, ym)
    fit_matrix = zernike_design_matrix(mode_list(J_FIT), xm, ym)
    proj, *_ = np.linalg.lstsq(fit_matrix, truth_matrix, rcond=None)
    leftover = truth_matrix - fit_matrix @ proj
    ensemble_residual_var = float(np.sum(sigmas**2 * np.mean(leftover**2, axis=0)))

    def panel(ax, field_flat, title, vlim):  # type: ignore[no-untyped-def]
        img = np.full(x.shape, np.nan)
        img[mask] = field_flat
        handle = ax.imshow(
            img,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        return handle

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.22)

    vlim = float(np.max(np.abs(truth)))
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    h0 = panel(
        ax0, truth, f"truth: Noll j=2..{J_TRUTH}\nKolmogorov, $D/r_0$={D_OVER_R0:g}", vlim
    )
    panel(ax1, fitted, f"least-squares fit, j=1..{J_FIT}", vlim)
    panel(
        ax2,
        fit.residual,
        f"residual, RMS = {fit.residual_rms:.3f} rad\n"
        f"({100 * fit.variance_explained:.1f} % of variance removed)",
        vlim,
    )
    cbar = fig.colorbar(h0, ax=[ax0, ax1, ax2], fraction=0.024, pad=0.015)
    cbar.set_label("phase [rad]", fontsize=8)

    ax3 = fig.add_subplot(gs[1, :2])
    j_axis = np.arange(2, J_FIT + 1)
    ax3.bar(
        j_axis - 0.2,
        truth_coeffs[: J_FIT - 1],
        width=0.4,
        label="injected",
        color="#3b6ea5",
    )
    ax3.bar(
        j_axis + 0.2,
        fit.coefficients[1:],
        width=0.4,
        label="recovered",
        color="#c1553b",
    )
    ax3.set_xticks(j_axis)
    ax3.set_xticklabels([str(j) for j in j_axis], fontsize=7)
    ax3.set_xlabel("Noll index $j$ (piston omitted)", fontsize=9)
    ax3.set_ylabel("coefficient [rad]", fontsize=9)
    ax3.legend(fontsize=8, frameon=False)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_title(
        "Injected vs recovered coefficients "
        f"(max |diff| = {np.max(np.abs(truth_coeffs[: J_FIT - 1] - fit.coefficients[1:])):.2e})",
        fontsize=9,
    )

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    text = (
        "Residual variance check\n"
        "-------------------------------------\n"
        f"this realisation       {measured_residual_var:8.3f} rad$^2$\n"
        f"ensemble mean (exact)  {ensemble_residual_var:8.3f} rad$^2$\n"
        f"Noll $\\Delta_{{{J_FIT}}}$ analytic     {analytic_residual_var:8.3f} rad$^2$\n"
        f"  less tail j>{J_TRUTH}      {analytic_residual_var - truncated_tail:8.3f} rad$^2$\n\n"
        f"condition number  {fit.condition_number:.3f}\n"
        f"samples used      {fit.n_used}\n"
        f"seed              {SEED}\n\n"
        "A single realisation scatters about the\n"
        "ensemble mean; the ensemble mean is the\n"
        "quantity that must match Noll."
    )
    ax4.text(0.0, 0.98, text, fontsize=8, family="monospace", va="top")

    fig.suptitle(
        "ZernKit -- least-squares Zernike fit of a synthetic Kolmogorov wavefront "
        "and its residual",
        fontsize=11,
    )

    out_dir = Path(__file__).resolve().parents[1] / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "wavefront_fit_residual.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"residual RMS            : {fit.residual_rms:.4f} rad")
    print(f"measured residual var   : {measured_residual_var:.4f} rad^2")
    print(f"ensemble mean (exact)   : {ensemble_residual_var:.4f} rad^2")
    print(f"analytic Noll Delta_{J_FIT}  : {analytic_residual_var:.4f} rad^2")
    print(f"tail beyond j={J_TRUTH}      : {truncated_tail:.4f} rad^2")
    print(f"Delta_{J_FIT} minus tail      : {analytic_residual_var - truncated_tail:.4f} rad^2")
    print(f"condition number        : {fit.condition_number:.4f}")
    return out


if __name__ == "__main__":
    print(f"wrote {main()}")
