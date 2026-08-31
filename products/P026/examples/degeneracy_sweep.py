"""Attitude uncertainty as two observations become parallel, and where the gate sits.

Sweeps the separation angle between two observations from 90 deg down to
0.01 deg and plots, on the same axes, the analytic 1-sigma attitude uncertainty
from Eq. V1 (optimal) and Eq. V2 (TRIAD), a seeded Monte Carlo estimate of the
same quantity, and the ``lambda_min`` of Eq. O4 that the solvers gate on.  The
vertical line is the default gate, ``lambda_min = 1e-6``, i.e. 0.1146 deg.

Writes ``../screenshots/degeneracy_sweep.png``.

Run:  python examples/degeneracy_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wahbakit import (  # noqa: E402
    VectorObservations,
    attitude_error_vector,
    dcm_from_quat,
    optimal_covariance,
    q_method,
    triad_covariance,
)

SEED = 20260831
TRIALS = 400
SIGMA = 1e-4
GATE_DEG = float(np.degrees(np.arccos(1.0 - 2.0e-6)))
SEPARATIONS = np.logspace(np.log10(90.0), np.log10(0.01), 25)
DCM_TRUE = dcm_from_quat([0.6, 0.4, -0.5, 0.3])


def geometry(separation_deg: float) -> np.ndarray:
    eta = np.radians(separation_deg)
    return np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(SEED)
    sigmas = np.full(2, SIGMA)
    optimal_sigma = []
    triad_sigma = []
    monte_carlo_sigma = []
    lambda_min = []

    for separation_deg in SEPARATIONS:
        reference = geometry(separation_deg)
        true_body = reference @ DCM_TRUE.T
        clean = VectorObservations(true_body, reference, sigmas=sigmas)
        # The sweep deliberately runs past the default gate, so it is lowered
        # here to 1e-14; this is the one place in the repository that does so,
        # and the whole point of the figure is what happens on the far side.
        optimal_sigma.append(
            float(np.sqrt(np.trace(optimal_covariance(clean, degeneracy_tol=1e-14))))
        )
        triad_sigma.append(
            float(np.sqrt(np.trace(triad_covariance(clean, degeneracy_tol=1e-14))))
        )
        lambda_min.append(clean.observability().lambda_min)

        squared = 0.0
        for _ in range(TRIALS):
            body = np.empty_like(true_body)
            for i, v in enumerate(true_body):
                seed = np.array([0.0, 0.0, 1.0])
                e1 = np.cross(v, seed)
                e1 /= np.linalg.norm(e1)
                body[i] = v + SIGMA * (rng.normal() * e1 + rng.normal() * np.cross(v, e1))
            body /= np.linalg.norm(body, axis=1)[:, None]
            obs = VectorObservations(body, reference, sigmas=sigmas)
            error = attitude_error_vector(q_method(obs, check_degeneracy=False).dcm, DCM_TRUE)
            squared += float(error @ error)
        monte_carlo_sigma.append(float(np.sqrt(squared / TRIALS)))

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].loglog(SEPARATIONS, np.degrees(optimal_sigma), "-", color="tab:blue", lw=1.6,
                   label="Eq. V1, optimal")
    axes[0].loglog(SEPARATIONS, np.degrees(triad_sigma), "-", color="tab:red", lw=1.6,
                   label="Eq. V2, TRIAD")
    axes[0].loglog(SEPARATIONS, np.degrees(monte_carlo_sigma), "o", color="k", ms=3.5,
                   label=f"Monte Carlo, q-method ({TRIALS} trials)")
    axes[0].axvline(GATE_DEG, color="tab:green", ls="--", lw=1.4,
                    label=f"default gate, {GATE_DEG:.4f} deg")
    axes[0].axhline(180.0, color="0.6", ls=":", lw=1.0, label="180 deg: no information")
    axes[0].set_xlabel("separation between the two observations [deg]")
    axes[0].set_ylabel(r"$\sqrt{\mathrm{tr}\,P}$ [deg]")
    axes[0].set_title(f"Attitude uncertainty vs geometry, sigma = {SIGMA:g} rad")
    axes[0].invert_xaxis()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].loglog(SEPARATIONS, lambda_min, "-", color="tab:purple", lw=1.6,
                   label=r"$\lambda_{\min}$ of Eq. O4")
    axes[1].loglog(SEPARATIONS, np.sin(np.radians(SEPARATIONS) / 2.0) ** 2, "k--", lw=1.0,
                   label=r"closed form $\sin^2(\eta/2)$")
    axes[1].axhline(1e-6, color="tab:green", ls="--", lw=1.4, label="gate = 1e-6")
    axes[1].axvline(GATE_DEG, color="tab:green", ls="--", lw=1.4)
    axes[1].set_xlabel("separation between the two observations [deg]")
    axes[1].set_ylabel(r"$\lambda_{\min}$")
    axes[1].set_title("The observability metric the solvers gate on")
    axes[1].invert_xaxis()
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")

    figure.suptitle(
        "wahbakit: near-parallel observations. Left of the green line every solver raises "
        "DegenerateObservationsError.",
        fontsize=10,
    )
    figure.tight_layout()
    output = Path(__file__).resolve().parents[1] / "screenshots" / "degeneracy_sweep.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)

    header = f"{'sep [deg]':>11} {'lambda_min':>13} {'Eq.V1 [deg]':>13} "
    print(header + f"{'Eq.V2 [deg]':>13} {'MC [deg]':>11}")
    for index in range(0, len(SEPARATIONS), 3):
        print(
            f"{SEPARATIONS[index]:11.4f} {lambda_min[index]:13.4e} "
            f"{np.degrees(optimal_sigma[index]):13.5f} {np.degrees(triad_sigma[index]):13.5f} "
            f"{np.degrees(monte_carlo_sigma[index]):11.5f}"
        )
    print(f"\nwrote {output}")
    print(f"elapsed {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
