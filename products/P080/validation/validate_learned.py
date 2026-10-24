"""Learned predictor: fitted parameters, held-out behaviour and the uncertainty output.

Every number reported in ``MODEL_CARD.md`` comes from this script. It records
the fit itself -- rows, parameters, convergence -- the measured effect of the
robust clipping, the correlation between the ensemble dispersion and the
realised absolute residual, and the two generalisation questions worth asking
of a predictor trained on one set of noise realisations and used on another.

The dispersion is a dispersion measure and not a confidence interval. Wager,
Hastie & Efron 2014 (JMLR 15) is the reference for why an ensemble spread does
not give coverage, and the number reported here is a Spearman rank correlation
against the realised absolute residual: a measurement of whether the
uncertainty output is informative, not an assertion that it is.
"""

from __future__ import annotations

from _bootstrap import (  # noqa: I001
    Tee,
    Timer,
    bootstrap,
    environment_lines,
    training_symbol_arrays,
)

ROOT = bootstrap()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.learned import KINDS, WINDOW, LearnedPredictor, design_matrix  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096


def main() -> int:
    out = Tee("validate_learned_output.txt")
    out.rule("boundcompress -- the learned residual predictor, measured")
    for line in environment_lines():
        out(line)

    arrays = training_symbol_arrays()
    features, gains, targets = design_matrix(np.concatenate(arrays[:1]))
    out("")
    out(f"training symbol arrays                     : {len(arrays)}")
    out(f"feature window (symbols)                   : {WINDOW}")
    out(f"features per row                           : {WINDOW - 1}")
    out(f"rows available before subsampling          : "
        f"{sum(max(0, int(np.asarray(a).size) - WINDOW) for a in arrays)}")

    out.rule("the normalised target has a heavy tail, which is why training clips")
    stacked = [design_matrix(array) for array in arrays]
    all_targets = np.concatenate([block[2] for block in stacked])
    all_features = np.vstack([block[0] for block in stacked])
    percentiles = [0.1, 1.0, 50.0, 99.0, 99.9]
    out(
        format_table(
            ["quantity", *[f"p{value}" for value in percentiles], "abs_max"],
            [
                [
                    "target d_i/s",
                    *[f"{float(np.percentile(all_targets, value)):.4f}"
                      for value in percentiles],
                    f"{float(np.abs(all_targets).max()):.1f}",
                ],
                [
                    "feature d_j/s",
                    *[f"{float(np.percentile(all_features, value)):.4f}"
                      for value in percentiles],
                    f"{float(np.abs(all_features).max()):.1f}",
                ],
            ],
        )
    )

    out.rule("the fit, with and without the clip")
    fit_rows = []
    models: dict[str, LearnedPredictor] = {}
    for kind in KINDS:
        for clip in (8.0, 0.0):
            timer = Timer()
            with timer:
                model = LearnedPredictor(kind=kind, members=3, clip=clip).fit(arrays)
            if clip == 8.0:
                models[kind] = model
            probe = [0, 10, 25, 45, 70]
            fit_rows.append(
                [
                    kind,
                    f"{clip:.1f}" if clip else "none",
                    str(model.train_rows),
                    str(model.parameter_count),
                    str(model.model_bytes),
                    str(sum(1 for flag in model.converged if flag)) + f"/{model.members}",
                    str(model.predict_one(probe)),
                    f"{timer.elapsed_s:.2f}",
                ]
            )
    out("")
    out(
        format_table(
            ["kind", "clip", "train_rows", "params", "model_bytes", "converged",
             "predict_one([0,10,25,45,70])", "fit_s"],
            fit_rows,
        )
    )
    out("")
    out("The clipped and unclipped models give different predictions on the same")
    out("history, which is the point: an unclipped least-squares fit is dominated by")
    out("the tail above, and the resulting coefficients are indistinguishable from")
    out("predicting nothing.")

    out.rule("the training loss the clip is responsible for")
    out("Fitted directly with scikit-learn on the same rows the ensemble members")
    out("see, so the effect of the clip and of the iteration cap are measured here")
    out("rather than asserted elsewhere. loss_ is the optimiser's own final value.")
    import warnings as _warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    rows_rng = np.random.default_rng(20261024)
    pool_features, pool_targets = all_features, all_targets
    if pool_features.shape[0] > 20000:
        keep = rows_rng.choice(pool_features.shape[0], size=20000, replace=False)
        keep.sort()
        pool_features, pool_targets = pool_features[keep], pool_targets[keep]

    loss_rows = []
    coefficient_probe = []
    for clip in (8.0, 0.0):
        if clip > 0.0:
            x_fit = np.clip(pool_features, -clip, clip)
            y_fit = np.clip(pool_targets, -clip, clip)
        else:
            x_fit, y_fit = pool_features, pool_targets
        ridge = Ridge(alpha=1.0).fit(x_fit, y_fit)
        coefficient_probe.append(
            [
                f"{clip:.1f}" if clip else "none",
                *[f"{float(value):+.4f}" for value in np.asarray(ridge.coef_).ravel()],
                f"{float(ridge.intercept_):+.4f}",
            ]
        )
        for cap in (400, 800):
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", ConvergenceWarning)
                perceptron = MLPRegressor(
                    hidden_layer_sizes=(16,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1.0,
                    max_iter=cap,
                    random_state=20261024,
                    tol=1e-7,
                ).fit(x_fit, y_fit)
            loss_rows.append(
                [
                    f"{clip:.1f}" if clip else "none",
                    str(cap),
                    str(int(perceptron.n_iter_)),
                    f"{float(perceptron.loss_):.6f}",
                    "yes" if int(perceptron.n_iter_) < cap else "no",
                ]
            )
    out("")
    out(format_table(["clip", "max_iter", "n_iter", "loss_", "converged"], loss_rows))
    out("")
    clipped_400 = float(loss_rows[0][3])
    clipped_800 = float(loss_rows[1][3])
    unclipped_400 = float(loss_rows[2][3])
    out(f"loss at clip=8, 400 iterations   : {clipped_400:.6f}")
    out(f"loss at clip=8, 800 iterations   : {clipped_800:.6f}")
    out(f"relative change from doubling cap: "
        f"{100.0 * (clipped_400 - clipped_800) / clipped_400:.2f} per cent")
    out(f"loss with no clip, 400 iterations: {unclipped_400:.6f}")
    out("")
    out("ridge coefficients on the most recent four first differences:")
    out(
        format_table(
            ["clip", "w(d1)", "w(d2)", "w(d3)", "w(d4)", "intercept"],
            coefficient_probe,
        )
    )
    out("")
    out("The unclipped coefficients are indistinguishable from predicting nothing,")
    out("which is what makes the clip load-bearing rather than cosmetic.")
    out.check(
        "the unclipped perceptron loss is far above the clipped one",
        unclipped_400 > 100.0 * clipped_400,
        f"{unclipped_400:.4f} against {clipped_400:.4f}",
    )

    out.rule("fitted coefficients of the learned linear predictor")
    out("One row per ensemble member. These are the weights on the normalised first")
    out("differences, most recent first, so a weight vector of [1, 0, 0, 0] would")
    out("reproduce the classical 'linear' predictor exactly and [0, 0, 0, 0] would")
    out("reproduce 'previous'.")
    coefficient_rows = []
    for index, member in enumerate(models["linear"]._members):  # noqa: SLF001
        coefficient_rows.append(
            [
                str(index),
                *[f"{float(value):+.4f}" for value in member.weights2],
                f"{float(member.bias2):+.4f}",
            ]
        )
    out("")
    out(
        format_table(
            ["member", "w(d1)", "w(d2)", "w(d3)", "w(d4)", "intercept"],
            coefficient_rows,
        )
    )
    mean_first = float(np.mean([m.weights2[0] for m in models["linear"]._members]))  # noqa: SLF001
    out("")
    out(f"mean weight on the most recent difference  : {mean_first:+.4f}")
    out("A negative weight here says the quantised first differences of this signal")
    out("set are anti-correlated on average, so the least-squares optimum sits")
    out("between 'previous' (weight 0) and the reflection of 'linear' (weight 1).")
    out("That is a statement about this generator's noise, not about telemetry.")

    out.rule("held-out behaviour: residual scale against the classical predictors")
    held_out = dataset.generate_all(samples=SAMPLES)
    residual_rows = []
    for name, signal in held_out.items():
        span = float(signal.values.max() - signal.values.min())
        spec = make_spec(signal.values, "abs", 1e-4 * span)
        symbols, _, _ = quantise(signal.values, spec)
        from boundcompress.predictors import get_predictor

        row = [name]
        for predictor in ("previous", "linear"):
            residuals = get_predictor(predictor).residuals(symbols)
            row.append(f"{float(np.mean(np.abs(residuals))):.3f}")
        for kind in KINDS:
            residuals = models[kind].residuals(symbols)
            row.append(f"{float(np.mean(np.abs(residuals))):.3f}")
        residual_rows.append(row)
    out("")
    out(
        format_table(
            ["signal", "previous_MAE", "linear_MAE", "learned_linear_MAE",
             "learned_mlp_MAE"],
            residual_rows,
        )
    )
    out("")
    out("Mean absolute residual in integer-symbol units at a bound of 1e-4 of each")
    out("signal's dynamic range. Lower is better, and it is a proxy: the rate is")
    out("what matters and is measured in validate_rate_distortion.py and")
    out("validate_predictors.py. A predictor can lower the mean residual and raise")
    out("the rate, by spreading the residual distribution.")

    out.rule("the uncertainty output: dispersion against the realised residual")
    out("Spearman rank correlation between the ensemble dispersion and |residual|.")
    out("A dispersion measure, not a confidence interval (Wager, Hastie & Efron")
    out("2014, JMLR 15). Reported because an uncertainty output that does not rank")
    out("errors is decoration.")
    dispersion_rows = []
    for kind in KINDS:
        model = models[kind]
        for name in ("battery_voltage", "gyro_rate", "wheel_rate", "white_noise"):
            signal = held_out[name]
            span = float(signal.values.max() - signal.values.min())
            spec = make_spec(signal.values, "abs", 1e-4 * span)
            symbols, _, _ = quantise(signal.values, spec)
            predictions, dispersion = model.predict_batch(symbols)
            residuals = np.abs(symbols[WINDOW:] - predictions).astype(np.float64)
            if float(np.std(dispersion)) == 0.0:
                correlation = float("nan")
                note = "dispersion is constant"
            else:
                correlation = float(spearmanr(dispersion, residuals).statistic)
                note = ""
            dispersion_rows.append(
                [
                    f"learned_{kind}",
                    name,
                    f"{float(np.mean(dispersion)):.4f}",
                    f"{float(np.max(dispersion)):.4f}",
                    f"{float(np.mean(residuals)):.4f}",
                    f"{correlation:.4f}",
                    note,
                ]
            )
    out("")
    out(
        format_table(
            ["model", "signal", "mean_dispersion", "max_dispersion", "mean_abs_residual",
             "spearman", "note"],
            dispersion_rows,
        )
    )
    finite = [
        float(row[5]) for row in dispersion_rows if not np.isnan(float(row[5]))
    ]
    out("")
    out(f"correlations measured  : {len(finite)}")
    if finite:
        out(f"median spearman        : {float(np.median(finite)):.4f}")
        out(f"range                  : {min(finite):.4f} to {max(finite):.4f}")
    out.check(
        "every reported dispersion is non-negative",
        all(float(row[2]) >= 0.0 for row in dispersion_rows),
    )

    out.rule("determinism of the fit")
    first = LearnedPredictor(kind="mlp", members=2, clip=8.0).fit(arrays)
    second = LearnedPredictor(kind="mlp", members=2, clip=8.0).fit(arrays)
    probe = [0, 7, 15, 26, 40]
    out(f"two independent fits, same seed : {first.predict_one(probe)} and "
        f"{second.predict_one(probe)}")
    out.check(
        "refitting the same configuration gives the same prediction",
        first.predict_one(probe) == second.predict_one(probe),
    )

    out.rule("the model is not in the bitstream")
    signal = held_out["gyro_rate"]
    result = encode(signal.values, "abs", 1e-6, predictor="learned", model=models["mlp"])
    out(f"container bytes                 : {result.certificate.encoded_bytes}")
    out(f"model bytes the decoder must hold: {models['mlp'].model_bytes}")
    out(f"model bytes as a share of one container, %: "
        f"{100.0 * models['mlp'].model_bytes / result.certificate.encoded_bytes:.2f}")
    out("")
    out("The parameters are a shared decoder asset and are charged separately in")
    out("MODEL_CARD.md rather than folded into the per-array rate, because folding")
    out("them in would make the rate depend on how many arrays the mission sends.")
    out("The break-even array count is in validate_predictors_output.txt.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
