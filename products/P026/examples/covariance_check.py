"""Analytic attitude covariance against a seeded Monte Carlo cloud.

Two geometries, side by side: the optimal (q-method) covariance of Eq. V1 over
four observations, and the TRIAD covariance of Eq. V2 over the first two.  Each
panel shows the Monte Carlo ``delta_theta`` samples projected onto a pair of
body axes, the analytic 1-sigma and 3-sigma ellipses from the closed form, and
the ellipse fitted to the samples themselves.

Writes ``../screenshots/covariance_vs_montecarlo.png``.

Run:  python examples/covariance_check.py
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
    covariance_axis_sigmas_deg,
    dcm_from_quat,
    optimal_covariance,
    q_method,
    triad,
    triad_covariance,
)

SEED = 20260831
TRIALS = 4000
SIGMAS = np.array([1.0e-3, 2.0e-3, 5.0e-3, 1.0e-3])
REFERENCE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
) / np.array([[1.0], [1.0], [1.0], [np.sqrt(3.0)]])
DCM_TRUE = dcm_from_quat([0.3, -0.7, 1.1, 0.2])


def sample_all(true_body, sigmas, rng, trials):
    """(trials, N, 3) draws from Eq. O1, re-normalised."""
    out = np.empty((trials, *true_body.shape))
    for i, v in enumerate(true_body):
        seed = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(v, seed)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        noise = rng.normal(size=(trials, 1)) * e1 + rng.normal(size=(trials, 1)) * e2
        out[:, i, :] = v + sigmas[i] * noise
    return out / np.linalg.norm(out, axis=2)[:, :, None]


def ellipse(covariance_2x2, scale):
    """Points on the ``scale``-sigma ellipse of a 2x2 covariance, in rad."""
    values, vectors = np.linalg.eigh(covariance_2x2)
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.vstack((np.cos(angle), np.sin(angle)))
    return (vectors @ np.diag(scale * np.sqrt(np.maximum(values, 0.0))) @ circle)


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(SEED)
    true_body = REFERENCE @ DCM_TRUE.T
    samples = sample_all(true_body, SIGMAS, rng, TRIALS)

    cases = []
    for label, solver, indices in (
        ("q-method, 4 observations (Eq. V1)", q_method, [0, 1, 2, 3]),
        ("TRIAD, observations 0 and 1 (Eq. V2)", triad, [0, 1]),
    ):
        errors = np.empty((TRIALS, 3))
        for k in range(TRIALS):
            obs = VectorObservations(
                samples[k][indices], REFERENCE[indices], sigmas=SIGMAS[indices]
            )
            errors[k] = attitude_error_vector(solver(obs).dcm, DCM_TRUE)
        clean = VectorObservations(
            true_body[indices], REFERENCE[indices], sigmas=SIGMAS[indices]
        )
        analytic = optimal_covariance(clean) if solver is q_method else triad_covariance(clean)
        cases.append((label, errors, analytic, errors.T @ errors / TRIALS))

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    for axis, (label, errors, analytic, empirical) in zip(axes, cases, strict=True):
        scale = np.degrees(1.0)
        axis.plot(
            errors[:, 0] * scale, errors[:, 1] * scale, ".", ms=1.4, alpha=0.35,
            color="0.45", label=f"Monte Carlo ({TRIALS} trials)"
        )
        for level, style in ((1.0, "-"), (3.0, ":")):
            curve = ellipse(analytic[:2, :2], level) * scale
            axis.plot(curve[0], curve[1], style, color="tab:blue", lw=1.8,
                      label=f"analytic {level:.0f}$\\sigma$")
            curve = ellipse(empirical[:2, :2], level) * scale
            axis.plot(curve[0], curve[1], style, color="tab:orange", lw=1.2,
                      label=f"sample {level:.0f}$\\sigma$")
        axis.set_xlabel(r"$\delta\theta_x$ [deg]")
        axis.set_ylabel(r"$\delta\theta_y$ [deg]")
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
        deviation = np.max(np.abs(empirical - analytic)) / np.max(np.abs(analytic))
        axis.set_title(f"{label}\nworst entry deviation {100 * deviation:.2f} %", fontsize=10)
        axis.legend(fontsize=7, loc="upper right")

    figure.suptitle(
        "wahbakit: analytic attitude covariance vs Monte Carlo, "
        f"sigmas = {np.array2string(SIGMAS, precision=4)} rad, seed {SEED}",
        fontsize=10,
    )
    figure.tight_layout()
    output = Path(__file__).resolve().parents[1] / "screenshots" / "covariance_vs_montecarlo.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)

    for label, _, analytic, empirical in cases:
        print(label)
        print(f"  analytic per-axis 1-sigma [deg]: {covariance_axis_sigmas_deg(analytic)}")
        print(f"  sample   per-axis 1-sigma [deg]: {covariance_axis_sigmas_deg(empirical)}")
        print(
            f"  worst entry deviation: "
            f"{100 * np.max(np.abs(empirical - analytic)) / np.max(np.abs(analytic)):.3f} %"
        )
    print(f"\nwrote {output}")
    print(f"elapsed {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
