"""Learned allocator versus the exact QP, as a picture.

A reduced-size rerun of ``validation/validate_ml_vs_qp.py`` (2000 training
samples instead of 4000, a smaller network, fewer test samples) so the example
finishes in about a minute. Read the *shape*; the numbers of record are in
``validation/ml_vs_qp_output.txt``.

Saves ``screenshots/learned_vs_qp.png``.

Run: ``python examples/ml_benchmark_demo.py``
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from alloclab.dataset import generate_dataset, reference_thruster_cluster  # noqa: E402
from alloclab.ml import LearnedAllocator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "learned_vs_qp.png"

N_TRAIN = 2000
N_TEST = 800
BOUND_TOL = 1e-9


def main() -> None:
    eset = reference_thruster_cluster(max_thrust=1.0, arm=0.5)
    train = generate_dataset(eset, N_TRAIN, seed=1234)
    test = generate_dataset(eset, N_TEST, seed=5678)

    t0 = time.perf_counter()
    model = LearnedAllocator(
        eset, n_estimators=5, hidden_layer_sizes=(64, 48), max_iter=200, random_state=0
    ).fit(train.torques, train.health, train.commands)
    t_fit = time.perf_counter() - t0

    out = model.predict(test.torques, test.health)
    err_ml = np.linalg.norm(test.torques - out.commands @ eset.matrix.T, axis=1)
    err_qp = test.residual_norm
    viol_ml = eset.bound_violation(out.commands)
    viol_qp = eset.bound_violation(test.commands)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    bins = np.logspace(-14, 0, 60)
    ax.hist(np.clip(err_qp, 1e-14, None), bins=bins, color="#3b6ea5", alpha=0.75, label="exact QP")
    ax.hist(np.clip(err_ml, 1e-14, None), bins=bins, color="#a63a3a", alpha=0.7, label="learned")
    ax.set_xscale("log")
    ax.set_xlabel("allocation error $\\|\\tau - Bu\\|$ [N$\\cdot$m]")
    ax.set_ylabel("test samples")
    ax.set_title(
        f"Allocation error\nmean QP {err_qp.mean():.2e}, learned {err_ml.mean():.2e}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(viol_ml, bins=50, color="#a63a3a", alpha=0.8, label="learned")
    ax.axvline(BOUND_TOL, color="#3b6ea5", lw=2.0, label="exact QP (all at 0)")
    ax.set_yscale("log")
    ax.set_xlabel("max actuator bound violation [N], limit is 1 N")
    ax.set_ylabel("test samples")
    ok_ml = 100.0 * float(np.mean(viol_ml <= BOUND_TOL))
    ok_qp = 100.0 * float(np.mean(viol_qp <= BOUND_TOL))
    ax.set_title(
        f"Constraint satisfaction\nin bounds: learned {ok_ml:.1f}%, QP {ok_qp:.1f}%",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(
        out.confidence, err_ml, c=viol_ml, s=8, cmap="magma_r", alpha=0.8
    )
    ax.set_xlabel("confidence output (1 = ensemble agrees as in training)")
    ax.set_ylabel("allocation error [N$\\cdot$m]")
    r = float(np.corrcoef(out.confidence, err_ml)[0, 1])
    ax.set_title(f"Confidence vs error\nPearson r = {r:+.3f}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="bound violation [N]")

    fig.suptitle(
        "Learned allocator against the QP it was trained to imitate "
        f"({N_TRAIN} train / {N_TEST} test, reduced-size rerun of validation 5)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"saved {OUT}")
    print(f"fit time                 : {t_fit:.1f} s on 2 cores")
    print(f"mean error, QP / learned : {err_qp.mean():.6e} / {err_ml.mean():.6e} N*m "
          f"({err_ml.mean() / err_qp.mean():.1f}x)")
    print(f"in bounds, QP / learned  : {ok_qp:.2f}% / {ok_ml:.2f}%")
    print(f"max learned violation    : {viol_ml.max():.6e} N of a 1 N limit")
    print(f"Pearson r(confidence,err): {r:+.4f}")


if __name__ == "__main__":
    main()
