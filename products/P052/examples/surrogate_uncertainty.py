"""What the learned component's uncertainty output is actually worth.

Three panels:

1. Predicted robustness against held-out truth, with the reported tree spread as
   an error bar. The spread is a ranking signal, not a confidence interval.
2. A reliability curve: nominal coverage against measured coverage for intervals
   from 0.5 to 3 standard deviations. A calibrated spread would lie on the
   diagonal.
3. Absolute held-out error binned by reported spread, which is the property the
   acquisition rule actually uses -- that bigger spread means bigger error, in
   order, whatever the magnitudes.

Saves ``../screenshots/surrogate_uncertainty.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.surrogate import DEFAULT_TREES, ForestSurrogate  # noqa: E402

N_TRAIN = 120
N_TEST = 200
TRAIN_SEED = 4100
TEST_SEED = 9400


def main() -> int:
    means, spreads, truths, labels = [], [], [], []
    for inst in suite():
        train_x = inst.sample(np.random.default_rng(TRAIN_SEED), N_TRAIN)
        test_x = inst.sample(np.random.default_rng(TEST_SEED), N_TEST)
        train_y = np.fromiter(
            (inst.evaluate(p) for p in train_x), dtype=float, count=N_TRAIN
        )
        test_y = np.fromiter((inst.evaluate(p) for p in test_x), dtype=float, count=N_TEST)
        model = ForestSurrogate(n_estimators=DEFAULT_TREES, random_state=0).fit(
            train_x, train_y
        )
        mean, spread = model.predict(test_x)
        means.append(mean)
        spreads.append(spread)
        truths.append(test_y)
        labels.extend([inst.identifier] * test_y.size)

    mean = np.concatenate(means)
    spread = np.concatenate(spreads)
    truth = np.concatenate(truths)
    error = np.abs(mean - truth)

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    show = np.random.default_rng(0).choice(mean.size, size=400, replace=False)
    left.errorbar(
        truth[show],
        mean[show],
        yerr=spread[show],
        fmt="o",
        markersize=3,
        alpha=0.45,
        elinewidth=0.8,
        color="#d62728",
    )
    lim = [float(min(truth.min(), mean.min())), float(max(truth.max(), mean.max()))]
    left.plot(lim, lim, "k--", lw=1.2)
    left.axhline(0.0, color="#555555", lw=0.8)
    left.axvline(0.0, color="#555555", lw=0.8)
    left.set_xlabel("held-out robustness (truth)")
    left.set_ylabel("surrogate prediction +- reported spread")
    left.set_title(
        f"Prediction against truth, {mean.size} held-out points\n"
        f"RMSE {np.sqrt(np.mean((mean - truth) ** 2)):.4f} (dimensionless)",
        fontsize=10,
    )
    left.grid(alpha=0.25)

    kappas = np.linspace(0.25, 3.0, 24)
    nominal = 2.0 * norm.cdf(kappas) - 1.0
    measured = np.array([float(np.mean(error <= k * spread)) for k in kappas])
    middle.plot(nominal, measured, "o-", color="#d62728", lw=1.6, markersize=4)
    middle.plot([0, 1], [0, 1], "k--", lw=1.2, label="calibrated")
    middle.set_xlabel("nominal coverage of mean +- k*spread")
    middle.set_ylabel("measured coverage")
    middle.set_xlim(0, 1)
    middle.set_ylim(0, 1)
    middle.legend(frameon=False, fontsize=9)
    middle.set_title(
        "Reliability of the reported spread.\n"
        "Above the diagonal = conservative, below = over-confident.",
        fontsize=10,
    )
    middle.grid(alpha=0.25)

    order = np.argsort(spread)
    n_bins = 10
    chunks = np.array_split(order, n_bins)
    bin_spread = [float(spread[c].mean()) for c in chunks]
    bin_error = [float(error[c].mean()) for c in chunks]
    right.plot(bin_spread, bin_error, "s-", color="#1f77b4", lw=1.6, markersize=5)
    top = max(max(bin_spread), max(bin_error))
    right.plot([0, top], [0, top], "k--", lw=1.2, label="spread = mean |error|")
    right.set_xlabel("reported spread, binned (deciles)")
    right.set_ylabel("mean |held-out error| in the bin")
    right.legend(frameon=False, fontsize=9)
    right.set_title(
        "Does a larger reported spread mean a larger error?\n"
        "This monotone ordering is all the acquisition rule needs.",
        fontsize=10,
    )
    right.grid(alpha=0.25)

    fig.suptitle(
        "The surrogate's uncertainty output: a useful ranking signal, not a "
        "confidence interval. This model is not certified for operational flight use.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "screenshots" / "surrogate_uncertainty.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    print("")
    print(f"held-out points pooled over the suite : {mean.size}")
    print(f"RMSE                                  : {np.sqrt(np.mean((mean - truth) ** 2)):.6f}")
    print(f"coverage of mean +- 1.96*spread       : {np.mean(error <= 1.96 * spread):.4f}")
    print(f"coverage of mean +- 1.00*spread       : {np.mean(error <= spread):.4f}")
    print("  (a Gaussian would give 0.9500 and 0.6827)")
    ranks_a = np.argsort(np.argsort(spread)).astype(float)
    ranks_b = np.argsort(np.argsort(error)).astype(float)
    print(f"Spearman(spread, |error|)             : {np.corrcoef(ranks_a, ranks_b)[0, 1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
