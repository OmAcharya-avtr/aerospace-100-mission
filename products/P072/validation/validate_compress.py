"""The learned compressibility predictor against the entropy floor and a constant table.

Three estimators of the compression ratio feed the size estimate the selection
is made against:

* the **memoryless entropy floor** ``min(H / B, 1)``, the Shannon source-coding
  bound for an i.i.d. source. It needs no training data, which is its whole
  attraction, and it is biased high because a real codec exploits correlation
  the first-order entropy ignores;
* the **per-instrument constant**, the mean realised ratio of each instrument's
  products. This is what a mission database holds and it is a strong baseline;
* a **random forest** over seven features including the lag-one autocorrelation
  proxy, which the entropy floor cannot use.

A fourth arm is the forest's **conservative** estimate, ``ratio + 2 sigma`` from
the ensemble dispersion. It is the one place in this package where the
uncertainty output changes a decision rather than annotating one, and it is
measured both ways: what it saves in overflows and what it costs in capacity
left unspent.

Checks:

1. regression error and bias of each estimator on held-out products;
2. the entropy floor as a *bound*: the fraction of products whose realised ratio
   falls below it, which is the fraction for which the bound is not a bound,
   with the explanation;
3. end-to-end, charging the true size at transmission time: a window planned
   with an under-estimated size overflows, the pass runs out of time, and the
   product that did not fit and everything after it in that window is lost;
4. the conservative arm against the mean arm, in overflows and in unspent
   capacity.

Everything is synthetic. No codec is run anywhere in this repository, and the
realised ratios come from equation (7) of :mod:`downlinkpick.dataset`, so these
numbers are statements about that generator. ``numcodecs`` is the package to
reach for to measure a real ratio.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from downlinkpick.benchmark import benchmark_ratio_sources  # noqa: E402
from downlinkpick.compress import (  # noqa: E402
    CompressibilityModel,
    entropy_floor_ratios,
    global_mean_ratios,
    instrument_constant_ratios,
    instrument_constant_table,
)
from downlinkpick.dataset import (  # noqa: E402
    COMPRESSION_FEATURES,
    generate_instance,
    generate_training_table,
)
from downlinkpick.exact import exact_select  # noqa: E402

TRAIN_ROWS = 6000
TRAIN_SEED = 1002
TRAIN_FRACTION = 0.7
BENCHMARK_SEEDS = tuple(range(6001, 6031))
MODEL_SEED = 20261021


def main() -> int:
    out = Tee("validate_compress_output.txt")
    out.rule("downlinkpick validation: the learned compressibility predictor")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("training")
    out(f"table rows     : {TRAIN_ROWS}, seed {TRAIN_SEED}")
    out(f"train fraction : {TRAIN_FRACTION}")
    out(f"features       : {', '.join(COMPRESSION_FEATURES)}")
    out(f"model          : RandomForestRegressor, 200 trees, min_samples_leaf 2, "
        f"random_state {MODEL_SEED}, n_jobs 1")
    out("target         : realised ratio, equation (7) of downlinkpick.dataset")

    table = generate_training_table(TRAIN_ROWS, TRAIN_SEED)
    split = int(TRAIN_FRACTION * len(table))
    train_x, test_x = table.compression_features[:split], table.compression_features[split:]
    train_y, test_y = table.compression_target[:split], table.compression_target[split:]
    floor_test = table.entropy_floor[split:]
    out(f"train / test   : {train_x.shape[0]} / {test_x.shape[0]} rows")

    fit_timer = Timer()
    model = CompressibilityModel(seed=MODEL_SEED)
    with fit_timer:
        model.fit(train_x, train_y)
    constant_table = instrument_constant_table(
        table.instrument[:split], table.compression_target[:split]
    )
    global_mean = float(train_y.mean())

    out.rule("1. regression error and bias on held-out products")
    out("bias is the signed mean error: positive means the estimator over-reserves")
    out("capacity, negative means it under-reserves and the pass can overflow")
    out("")
    predicted, dispersion = model.predict_with_dispersion(test_x)
    conservative = np.clip(predicted + 2.0 * dispersion, 1.0e-3, 1.0)
    instrument_predicted = np.array(
        [constant_table[name] for name in table.instrument[split:]]
    )
    out(f"{'estimator':>26s} {'MAE':>10s} {'RMSE':>10s} {'bias':>11s} {'max |err|':>10s}")
    for label, values in (
        ("learned forest (mean)", predicted),
        ("learned forest (+2 sigma)", conservative),
        ("per-instrument constant", instrument_predicted),
        ("global mean", np.full(test_y.shape, global_mean)),
        ("entropy floor H/B", floor_test),
    ):
        error = values - test_y
        out(
            f"{label:>26s} {np.abs(error).mean():10.6f} "
            f"{np.sqrt((error**2).mean()):10.6f} {error.mean():11.3e} "
            f"{np.abs(error).max():10.6f}"
        )
    out("")
    out("per-instrument constant table from the training split")
    for name, ratio in sorted(constant_table.items()):
        out(f"  {name:>6s} {ratio:.6f}")
    out(f"  {'global':>6s} {global_mean:.6f}")
    out("")
    importances = model.feature_importances()
    out("impurity-based feature importances, biased towards high-cardinality features")
    for name, weight in sorted(importances.items(), key=lambda item: -item[1]):
        out(f"  {name:>26s} {weight:.6f}")

    out.rule("2. the entropy floor is not a bound for a correlated source")
    out("Shannon's source-coding theorem bounds the ratio below by H/B for a MEMORYLESS")
    out("source. The generator's equation (7) includes a correlation gain, as a real")
    out("codec does, so the realised ratio is allowed to fall below the floor. The")
    out("fraction below is therefore the fraction for which the analytic bound does not")
    out("apply, and is a statement about the generator and about real codecs alike.")
    out("")
    below = float(np.mean(test_y < floor_test))
    out(f"held-out products with realised ratio below the floor : {below:.6f}")
    out(f"mean floor                                            : {floor_test.mean():.6f}")
    out(f"mean realised ratio                                   : {test_y.mean():.6f}")
    out(f"mean floor / mean realised                            : "
        f"{floor_test.mean() / test_y.mean():.6f}")
    out(f"worst over-estimate by the floor                      : "
        f"{float((floor_test - test_y).max()):.6f}")

    out.rule("3. the uncertainty output")
    out("per-tree dispersion is a dispersion measure over the ensemble, not a calibrated")
    out("interval (Wager, Hastie and Efron 2014)")
    out("")
    absolute = np.abs(predicted - test_y)
    out(f"{'band':>12s} {'nominal normal':>15s} {'measured coverage':>18s}")
    for multiple, nominal in ((1.0, 0.6827), (1.96, 0.9500), (2.0, 0.9545)):
        out(f"{f'+-{multiple:g} sigma':>12s} {nominal:15.4f} "
            f"{float(np.mean(absolute <= multiple * dispersion)):18.6f}")
    out("")
    out(f"mean dispersion            : {dispersion.mean():.6f}")
    out(f"mean absolute error        : {absolute.mean():.6f}")
    from downlinkpick.valuemodel import spearman  # noqa: PLC0415

    out(f"Spearman(dispersion, |err|): {spearman(dispersion, absolute):.6f}")

    out.rule("4 and 5. end-to-end: true sizes charged at transmission time")
    out(f"instances: {len(BENCHMARK_SEEDS)}, seeds {BENCHMARK_SEEDS[0]}.."
        f"{BENCHMARK_SEEDS[-1]}, 24 products, 3 windows,")
    out("           capacity 45 per cent of the queue, ground-truth values on every arm")
    out("a window planned with an under-estimated size overflows: the pass runs out of")
    out("time, and the product that did not fit and everything after it in that window")
    out("is lost. delivered_fraction is the value that actually arrived, against the")
    out("oracle's. unspent_fraction is the capacity that went unused, which is what a")
    out("conservative estimator costs.")
    out("")

    def learned_arm(instance, conservative_arm: bool = False):
        prediction = model.predict_inventory(instance.inventory, instance.autocorrelation)
        return (
            prediction.conservative_mapping() if conservative_arm else prediction.as_mapping()
        )

    arms = {
        "learned-mean": learned_arm,
        "learned+2sigma": lambda instance: learned_arm(instance, conservative_arm=True),
        "instrument-constant": lambda instance: instrument_constant_ratios(
            instance.inventory, constant_table
        ),
        "global-mean": lambda instance: global_mean_ratios(
            instance.inventory, np.array([global_mean])
        ),
        "entropy-floor": lambda instance: entropy_floor_ratios(instance.inventory),
    }

    instances = [
        generate_instance(seed, n_products=24, n_windows=3, capacity_fraction=0.45)
        for seed in BENCHMARK_SEEDS
    ]
    bench_timer = Timer()
    with bench_timer:
        results = benchmark_ratio_sources(instances, arms, selector=exact_select)
    out(
        f"{'arm':>21s} {'MAE':>9s} {'bias':>11s} {'overflow':>9s} "
        f"{'mean ovf [Gbit]':>16s} {'delivered':>10s} {'unspent':>9s}"
    )
    for result in results:
        out(
            f"{result.name:>21s} {result.mae:9.6f} {result.bias:11.3e} "
            f"{result.overflow_rate:9.6f} {result.mean_overflow_bits / 1.0e9:16.6f} "
            f"{result.delivered_fraction:10.6f} {result.unspent_fraction:9.6f}"
        )
    ranked = sorted(
        (r for r in results if r.name != "oracle"), key=lambda r: -r.delivered_fraction
    )
    out("")
    out(f"best non-oracle arm by delivered value: {ranked[0].name} at "
        f"{ranked[0].delivered_fraction:.6f}")
    out("ranking by delivered value: " + ", ".join(r.name for r in ranked))
    lookup = {r.name: r for r in results}
    out("")
    out("the conservative arm against the mean arm")
    out(f"  overflow rate   : {lookup['learned+2sigma'].overflow_rate:.6f} against "
        f"{lookup['learned-mean'].overflow_rate:.6f}")
    out(f"  unspent capacity: {lookup['learned+2sigma'].unspent_fraction:.6f} against "
        f"{lookup['learned-mean'].unspent_fraction:.6f}")
    out(f"  delivered value : {lookup['learned+2sigma'].delivered_fraction:.6f} against "
        f"{lookup['learned-mean'].delivered_fraction:.6f}")
    if lookup["learned+2sigma"].overflow_rate > lookup["learned-mean"].overflow_rate:
        failures += 1
        out("  FAIL the conservative estimate overflows more often than the mean estimate,")
        out("       which would make the uncertainty output actively harmful")

    out.rule("compute budget")
    out("timings are wall clock on two shared cores and are UNSTABLE between runs")
    out(f"{'quantity':>30s} {'value':>12s}")
    out(f"{'forest fit [s]':>30s} {fit_timer.elapsed_s:12.3f}")
    out(f"{'benchmark wall clock [s]':>30s} {bench_timer.elapsed_s:12.3f}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
