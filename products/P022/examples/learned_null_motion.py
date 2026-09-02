"""Reduced-size rerun of validation 5: the learned null-motion policy vs the classics.

Writes ``screenshots/learned_vs_classical.png``.  Runtime: about one minute.
The sizes are smaller than the validation run, so read the shape of these
panels, not their exact values; ``validation/nullmotion_ml_output.txt`` has the
numbers of record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.dataset import generate_policy_dataset, manoeuvre_suite  # noqa: E402
from cmgsteer.ml import LearnedNullMotion  # noqa: E402
from cmgsteer.nullmotion import GradientNullMotion, NoNullMotion  # noqa: E402
from cmgsteer.simulate import run_steering  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "learned_vs_classical.png"
N_BOOTSTRAP = 4000


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, values.size, size=(N_BOOTSTRAP, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    array = pyramid_array()
    train = generate_policy_dataset(
        array, 400, seed=1234, horizon=25, n_candidates=9, stride=17, n_manoeuvres=12
    )
    test = generate_policy_dataset(
        array, 150, seed=5678, horizon=25, n_candidates=9, stride=17, n_manoeuvres=6
    )
    policy = LearnedNullMotion(
        max_null_rate=0.5, n_estimators=5, hidden_layer_sizes=(64, 32), max_iter=400
    ).fit(train.features, train.coefficients)
    pred, spread = policy.predict(test.features)
    conf = policy.confidence(spread)

    suite = manoeuvre_suite(array, 10, seed=9012, n_segments=2, segment_duration=6.0, dt=0.02)
    configs = [
        ("pinv", "pinv", NoNullMotion(), "#b2182b"),
        ("sr", "sr", NoNullMotion(), "#2166ac"),
        ("sr+grad", "sr", GradientNullMotion(gain=1.0, max_rate=0.5), "#4d9221"),
        ("sr+learned", "sr", policy, "#762a83"),
    ]
    path_errors = {}
    for label, method, null_policy, _ in configs:
        errors = []
        for profile, start in suite:
            history = run_steering(
                array,
                start,
                profile,
                method=method,
                null_policy=null_policy,
                max_gimbal_rate=2.0,
            )
            errors.append(history.total_momentum_error_path)
        path_errors[label] = np.array(errors)

    rng = np.random.default_rng(4242)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    labels = [c[0] for c in configs]
    means = [path_errors[label].mean() for label in labels]
    los, his = zip(*[bootstrap_ci(path_errors[label], rng) for label in labels])
    err = np.array([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    ax.bar(labels, means, yerr=err, capsize=5, color=[c[3] for c in configs], alpha=0.85)
    ax.set_ylabel("path momentum error [N m s]")
    ax.set_title(
        f"{len(suite)} held-out manoeuvres\nerror bars: bootstrap 95% CI of the mean"
    )
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot([-1.05, 1.05], [-1.05, 1.05], color="0.5", lw=1.0, ls="--", label="perfect")
    scatter = ax.scatter(test.coefficients, pred, c=conf, s=16, cmap="viridis")
    ax.set_xlabel("oracle null-motion coefficient $k^*$")
    ax.set_ylabel("predicted $k$")
    resid = pred - test.coefficients
    r2 = 1.0 - float(np.sum(resid**2)) / float(
        np.sum((test.coefficients - test.coefficients.mean()) ** 2)
    )
    ax.set_title(f"held-out label regression, $R^2$ = {r2:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="confidence")

    ax = axes[2]
    order = np.argsort(conf)
    deciles = np.array_split(order, 10)
    centres = [conf[idx].mean() for idx in deciles]
    label_err = [np.mean(np.abs(resid[idx])) for idx in deciles]
    scores_at_pred = np.array(
        [np.interp(p, test.candidates, row) for p, row in zip(pred, test.candidate_scores)]
    )
    gap = (scores_at_pred - test.best_scores) / np.maximum(test.zero_scores, 1e-30)
    gap_by_decile = [gap[idx].mean() for idx in deciles]
    ax.plot(centres, label_err, "o-", color="#2166ac", label="mean |label error|")
    ax.plot(centres, gap_by_decile, "s-", color="#b2182b", label="normalised oracle gap")
    ax.set_xlabel("mean confidence in the decile")
    ax.set_ylabel("error")
    ax.set_title("does the confidence rank trustworthiness?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"saved {OUT}")
    print(f"{'configuration':>12} {'mean path err':>16} {'95% CI':>32}")
    for label, lo, hi, mean in zip(labels, los, his, means):
        print(f"{label:>12} {mean:>16.6e} {f'[{lo:.4e}, {hi:.4e}]':>32}")
    print(f"held-out label R^2 {r2:.4f}, Pearson r(confidence, |label error|) "
          f"{np.corrcoef(conf, np.abs(resid))[0, 1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
