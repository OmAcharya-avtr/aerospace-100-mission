"""Learned combiner against its four non-learned references.

MRC with the true channel state is optimal by Cauchy-Schwarz, so the plot
cannot show a learned combiner beating it -- the reference line sits at
0 dB by construction. What the plot shows is the only thing that is
undetermined: what to do when the state estimate is wrong.

Panel 1: mean post-combining SNR penalty against MRC-with-the-truth, as a
function of the log-domain channel-estimation error, for MRC with the
estimate, equal gain, a one-parameter analytic shrinkage rule, and the
learned combiner. Panel 2: the uncertainty output -- predicted 10th, 50th
and 90th percentile of the penalty against the realised distribution.

What to notice: MRC-with-the-estimate is exactly optimal at zero error and
degrades without limit; equal gain is flat, because it never looks at the
estimate; and the two cross. Above that crossing the correct engineering
answer is to stop estimating the channel. The learned combiner tracks the
better of the two everywhere but its margin over the analytic shrinkage rule
is a fraction of a decibel, which is the honest size of the result.

Writes ../screenshots/learned_combiner_vs_baselines.png. Runtime about 25 s
on an idle pair of cores; the build container is heavily contended, so expect
several times that there.
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

from aperturediv.correlation import (  # noqa: E402
    correlation_matrix,
    equispaced_positions,
    sample_correlated_lognormal,
)
from aperturediv.datasets import build_features, make_combiner_dataset, split_dataset  # noqa: E402
from aperturediv.estimation import estimate_from_log_error, log_error_sigma_db  # noqa: E402
from aperturediv.learned import (  # noqa: E402
    LearnedCombiner,
    egc_weights,
    fit_shrinkage_exponent,
    mrc_estimated_weights,
    penalty_db,
    shrinkage_weights,
)

OUT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "screenshots"
    / "learned_combiner_vs_baselines.png"
)
N_ROWS = 120_000
N_TRAIN = 70_000
N_VAL = 25_000
L = 4
SI = 0.9
SPACING_M = 0.15
RHO_C_M = 0.10
SIGMA_E_MAX = 3.0
SEED = 44044
EVAL_ROWS = 20_000
LEVELS = np.linspace(0.0, SIGMA_E_MAX, 11)


def main() -> int:
    print(f"L={L}, si={SI}, pitch={SPACING_M} m, rho_c={RHO_C_M} m, seed={SEED}")
    print(f"training rows {N_TRAIN}, validation {N_VAL}, "
          f"held-out {N_ROWS - N_TRAIN - N_VAL}")
    pos = equispaced_positions(L, SPACING_M)
    r_log = correlation_matrix(pos, RHO_C_M, "gaussian")
    print(f"adjacent-aperture log correlation {r_log[0, 1]:.6f}")
    data = make_combiner_dataset(
        N_ROWS,
        n_apertures=L,
        si=SI,
        aperture_spacing_m=SPACING_M,
        correlation_scale_m=RHO_C_M,
        sigma_e_range=(0.0, SIGMA_E_MAX),
        seed=SEED,
    )
    train, val, test = split_dataset(data, n_train=N_TRAIN, n_validation=N_VAL)
    p_star = fit_shrinkage_exponent(val)
    model = LearnedCombiner(random_state=SEED).fit(train)
    print(f"analytic shrinkage exponent fitted on the validation split: p* = {p_star:.4f}")
    print("  (p=1 is MRC with the estimate, p=0 is equal gain)")
    print()

    curves = {"mrc_estimated": [], "egc": [], f"shrinkage p={p_star:.2f}": [], "learned": []}
    print("  sigma_e  sigma_e dB   mrc_est    egc   shrinkage  learned")
    for j, se in enumerate(LEVELS):
        rng = np.random.default_rng(SEED + 2000 + j)
        i_true = sample_correlated_lognormal(EVAL_ROWS, SI, r_log, rng)
        est = estimate_from_log_error(i_true, float(se), rng)
        h = np.sqrt(i_true)
        feats = build_features(est.irradiance_estimated, np.full(EVAL_ROWS, float(se)))
        vals = {
            "mrc_estimated": float(
                penalty_db(mrc_estimated_weights(est.irradiance_estimated), h).mean()
            ),
            "egc": float(penalty_db(egc_weights(EVAL_ROWS, L), h).mean()),
            f"shrinkage p={p_star:.2f}": float(
                penalty_db(shrinkage_weights(est.irradiance_estimated, p_star), h).mean()
            ),
            "learned": float(penalty_db(model.predict_weights(feats), h).mean()),
        }
        for k, v in vals.items():
            curves[k].append(v)
        print(f"{se:9.3f} {log_error_sigma_db(se):10.3f}" + "".join(
            f"{vals[k]:9.4f}" for k in curves
        ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    styles = {
        "mrc_estimated": ("#a01f1f", "-", "o"),
        "egc": ("#2f8f4f", "-", "s"),
        f"shrinkage p={p_star:.2f}": ("#8a5a00", "--", "^"),
        "learned": ("#1f3f7a", "-", "D"),
    }
    db_levels = np.array([log_error_sigma_db(s) for s in LEVELS])
    for name, series in curves.items():
        colour, ls, marker = styles[name]
        ax1.plot(db_levels, series, color=colour, ls=ls, marker=marker, lw=2.0,
                 ms=4.5, label=name)
    ax1.axhline(0.0, color="k", lw=1.4)
    ax1.text(0.15, 0.03, "MRC with the true state: 0 dB by Cauchy-Schwarz", fontsize=8)

    mrc_arr = np.array(curves["mrc_estimated"])
    egc_arr = np.array(curves["egc"])
    cross = np.nonzero(np.diff(np.sign(mrc_arr - egc_arr)))[0]
    if cross.size:
        lo, hi = db_levels[cross[0]], db_levels[cross[0] + 1]
        mid = 0.5 * (lo + hi)
        ax1.axvline(mid, color="k", ls=":", lw=1.2)
        ax1.text(mid + 0.15, max(mrc_arr) * 0.75,
                 f"EGC takes over\nnear {mid:.2f} dB", fontsize=8.5)
        print()
        print(f"MRC-with-the-estimate and EGC cross between {lo:.3f} and {hi:.3f} dB "
              "of irradiance estimation error")
    ax1.set_xlabel(r"channel-estimation error, dB of irradiance ($\sigma_e$)")
    ax1.set_ylabel("mean post-combining SNR penalty (dB)")
    ax1.set_title(f"Penalty against MRC-with-the-truth, L={L}, si={SI}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.5, loc="upper left")

    _, quant = model.combine(test.irradiance_estimated, test.sigma_e)
    realised = penalty_db(model.combine(test.irradiance_estimated, test.sigma_e)[0],
                          test.amplitude_true)
    order = np.argsort(test.sigma_e)
    se_sorted = test.sigma_e_db[order]
    block = 500
    n_blocks = len(order) // block
    centres, emp_q10, emp_q50, emp_q90, pred = [], [], [], [], [[], [], []]
    for b in range(n_blocks):
        idx = order[b * block : (b + 1) * block]
        centres.append(float(np.mean(se_sorted[b * block : (b + 1) * block])))
        emp_q10.append(float(np.quantile(realised[idx], 0.1)))
        emp_q50.append(float(np.quantile(realised[idx], 0.5)))
        emp_q90.append(float(np.quantile(realised[idx], 0.9)))
        for j in range(3):
            pred[j].append(float(np.mean(quant[idx, j])))
    ax2.plot(centres, emp_q10, color="#1f3f7a", lw=1.6, label="realised 10th percentile")
    ax2.plot(centres, emp_q50, color="#2f8f4f", lw=1.6, label="realised median")
    ax2.plot(centres, emp_q90, color="#a01f1f", lw=1.6, label="realised 90th percentile")
    ax2.plot(centres, pred[0], color="#1f3f7a", lw=1.6, ls="--", label="predicted 10th")
    ax2.plot(centres, pred[1], color="#2f8f4f", lw=1.6, ls="--", label="predicted median")
    ax2.plot(centres, pred[2], color="#a01f1f", lw=1.6, ls="--", label="predicted 90th")
    ax2.set_xlabel(r"channel-estimation error, dB of irradiance ($\sigma_e$)")
    ax2.set_ylabel("learned-combiner penalty (dB)")
    ax2.set_title("Uncertainty output: predicted (dashed) vs realised (solid)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print()
    print("pooled coverage of the uncertainty output on the held-out rows")
    for j, q in enumerate(model.quantiles):
        print(f"  nominal {q:.2f} -> empirical {float(np.mean(realised <= quant[:, j])):.6f}")
    print()
    best_nl = np.minimum.reduce([
        np.array(curves["mrc_estimated"]),
        np.array(curves["egc"]),
        np.array(curves[f"shrinkage p={p_star:.2f}"]),
    ])
    margin = np.array(curves["learned"]) - best_nl
    print(f"learned minus best non-learned, over the error range: "
          f"worst {margin.max():+.4f} dB, best {margin.min():+.4f} dB")
    print("a negative number means the learned combiner is ahead")
    print()
    print(f"wrote {OUT.parent.name}/{OUT.name}")
    print("research-grade output; not flight-qualified, not certified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
