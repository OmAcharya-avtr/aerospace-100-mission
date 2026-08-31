"""Monte Carlo attitude error of all four methods against the Cramer-Rao bound.

Geometry: four reference directions (three body axes plus the body diagonal),
smallest pairwise separation 54.74 deg.  TRIAD sees only the first two, because
that is all TRIAD is defined for; the other three see all four.

Writes ``../screenshots/method_accuracy_vs_noise.png``.

Run:  python examples/method_comparison.py
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
    angle_between_dcm,
    dcm_from_quat,
    olae,
    optimal_covariance,
    q_method,
    quest,
    triad,
    triad_covariance,
)

SEED = 20260831
TRIALS = 600
SIGMAS = np.logspace(-5, -1, 9)
REFERENCE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
) / np.array([[1.0], [1.0], [1.0], [np.sqrt(3.0)]])
DCM_TRUE = dcm_from_quat([0.3, -0.7, 1.1, 0.2])


def sample(true_body: np.ndarray, sigmas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw from the Eq. O1 measurement model, re-normalised to unit length."""
    out = np.empty_like(true_body)
    for i, v in enumerate(true_body):
        seed = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(v, seed)
        e1 /= np.linalg.norm(e1)
        out[i] = v + sigmas[i] * (rng.normal() * e1 + rng.normal() * np.cross(v, e1))
    return out / np.linalg.norm(out, axis=1)[:, None]


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(SEED)
    true_body = REFERENCE @ DCM_TRUE.T
    methods = {"TRIAD (2 obs)": triad, "q-method": q_method, "QUEST": quest, "OLAE": olae}
    rms = {name: [] for name in methods}
    bound_optimal = []
    bound_triad = []

    for sigma in SIGMAS:
        sigmas = np.full(4, sigma)
        errors = {name: [] for name in methods}
        for _ in range(TRIALS):
            obs = VectorObservations(sample(true_body, sigmas, rng), REFERENCE, sigmas=sigmas)
            pair = obs.subset([0, 1])
            for name, solver in methods.items():
                target = pair if name.startswith("TRIAD") else obs
                errors[name].append(angle_between_dcm(solver(target).dcm, DCM_TRUE))
        for name in methods:
            rms[name].append(float(np.sqrt(np.mean(np.square(errors[name])))))
        clean = VectorObservations(true_body, REFERENCE, sigmas=sigmas)
        bound_optimal.append(float(np.sqrt(np.trace(optimal_covariance(clean)))))
        bound_triad.append(float(np.sqrt(np.trace(triad_covariance(clean.subset([0, 1]))))))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    styles = {
        "TRIAD (2 obs)": ("tab:red", "o"),
        "q-method": ("tab:blue", "s"),
        "QUEST": ("tab:cyan", "^"),
        "OLAE": ("tab:green", "v"),
    }
    for name, (colour, marker) in styles.items():
        axes[0].loglog(
            SIGMAS, np.degrees(rms[name]), marker=marker, color=colour, label=name, lw=1.4, ms=5
        )
    axes[0].loglog(
        SIGMAS, np.degrees(bound_optimal), "k--", lw=1.2, label="Eq. V1 bound (4 obs)"
    )
    axes[0].loglog(
        SIGMAS, np.degrees(bound_triad), "--", color="tab:red", lw=1.2, label="Eq. V2 (TRIAD)"
    )
    axes[0].set_xlabel("sensor sigma [rad]")
    axes[0].set_ylabel(r"RMS $|\delta\theta|$ [deg]")
    axes[0].set_title(f"Attitude error vs sensor noise ({TRIALS} trials per point)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    for name, (colour, marker) in styles.items():
        ratio = np.array(rms[name]) / np.array(bound_optimal)
        axes[1].semilogx(SIGMAS, ratio, marker=marker, color=colour, label=name, lw=1.4, ms=5)
    axes[1].semilogx(SIGMAS, np.array(bound_triad) / np.array(bound_optimal), "--",
                     color="tab:red", lw=1.2, label="Eq. V2 / Eq. V1")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("sensor sigma [rad]")
    axes[1].set_ylabel("RMS error / Eq. V1 bound")
    axes[1].set_title("Efficiency relative to the Cramer-Rao bound")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "wahbakit: four solutions of Wahba's problem, seed "
        f"{SEED}, reference = 3 axes + body diagonal",
        fontsize=10,
    )
    figure.tight_layout()
    output = Path(__file__).resolve().parents[1] / "screenshots" / "method_accuracy_vs_noise.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)

    header = f"{'sigma [rad]':>12} " + " ".join(f"{name:>14}" for name in methods)
    print(header + f" {'Eq. V1':>12}")
    for index, sigma in enumerate(SIGMAS):
        row = " ".join(f"{np.degrees(rms[name][index]):14.5e}" for name in methods)
        print(f"{sigma:12.3e} {row} {np.degrees(bound_optimal[index]):12.5e}")
    print(f"\nwrote {output}")
    print(f"elapsed {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
