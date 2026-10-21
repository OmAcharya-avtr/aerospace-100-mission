"""The learned value model against the declared passthrough and a size/age heuristic.

The batch specification predicts in writing that *the exact solver on declared
values beats everything and the learned value model helps only where
declarations are absent*. This script is written to measure that, not to confirm
it, and the result is published as measured.

Two kinds of measurement, because they answer different questions:

* **Regression error** on held-out products: mean absolute error, root mean
  square error and Spearman rank correlation against the realised value of
  equation (3) of :mod:`downlinkpick.dataset`. This says how well each estimator
  estimates.
* **End-to-end regret**, equation (8) of :mod:`downlinkpick.benchmark`: each
  estimator's values go through the same solver on the same instances, the
  resulting selection is scored against the ground-truth values, and the
  shortfall against the oracle's selection is the regret. This says how much the
  estimation error costs, which is the only question an operator has. An
  estimator with a large error and no regret is good enough; an estimator with a
  small error that crosses selection boundaries is not.

Five scenarios, because the answer depends entirely on the state of the
declarations:

1. complete and fresh declarations;
2. complete but stale declarations -- written at acquisition and never revised,
   so they ignore the decay since;
3. 30 per cent of declarations missing;
4. all declarations missing;
5. 40 per cent missing and half the remainder stale, which is the shipped
   ``missing_declarations`` case.

Test-split strategy. The models are trained on a flat table of 6000 synthetic
products generated from seed 1001. The benchmark instances are generated from
seeds 5001 upwards through a separate seed stream, so no product in the
benchmark appears in the training table. Everything is synthetic: these numbers
are statements about the generator in :mod:`downlinkpick.dataset` and not about
any spacecraft.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from downlinkpick.benchmark import benchmark_value_sources  # noqa: E402
from downlinkpick.dataset import (  # noqa: E402
    VALUE_FEATURES,
    generate_instance,
    generate_training_table,
    value_feature_row,
)
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.valuemodel import (  # noqa: E402
    ValueModel,
    declared_values,
    fit_size_age_heuristic,
    hybrid_values,
    spearman,
)

TRAIN_ROWS = 6000
TRAIN_SEED = 1001
TRAIN_FRACTION = 0.7
BENCHMARK_SEEDS = tuple(range(5001, 5031))
MODEL_SEED = 20261021

SCENARIOS = (
    ("1 complete and fresh", dict(missing_declared_fraction=0.0, stale_declared_fraction=0.0)),
    ("2 complete but stale", dict(missing_declared_fraction=0.0, stale_declared_fraction=1.0)),
    ("3 30 per cent missing", dict(missing_declared_fraction=0.3, stale_declared_fraction=0.0)),
    ("4 all missing", dict(missing_declared_fraction=1.0, stale_declared_fraction=0.0)),
    (
        "5 40 per cent missing, half of the rest stale",
        dict(missing_declared_fraction=0.4, stale_declared_fraction=0.5),
    ),
)


def main() -> int:
    out = Tee("validate_valuemodel_output.txt")
    out.rule("downlinkpick validation: the learned value model against its baselines")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("training")
    out(f"table rows        : {TRAIN_ROWS}, seed {TRAIN_SEED}")
    out(f"train fraction    : {TRAIN_FRACTION}")
    out(f"features          : {', '.join(VALUE_FEATURES)}")
    out(f"model             : RandomForestRegressor, 300 trees, min_samples_leaf 2, "
        f"random_state {MODEL_SEED}, n_jobs 1")
    out("target            : realised value, equation (3) of downlinkpick.dataset")
    out("latent            : the site weight omega of equation (3) is NOT a feature, so it")
    out("                    is the irreducible error floor of every estimator here")

    table = generate_training_table(TRAIN_ROWS, TRAIN_SEED)
    split = int(TRAIN_FRACTION * len(table))
    train_x, test_x = table.value_features[:split], table.value_features[split:]
    train_y, test_y = table.value_target[:split], table.value_target[split:]
    out(f"train / test rows : {train_x.shape[0]} / {test_x.shape[0]}")

    fit_timer = Timer()
    model = ValueModel(seed=MODEL_SEED)
    with fit_timer:
        model.fit(train_x, train_y)
    heuristic = fit_size_age_heuristic(train_x, train_y)
    out(f"size/age fit      : value = {heuristic.intercept:.6f} "
        f"+ {heuristic.log_bytes_coefficient:.6f} * log10(bytes) "
        f"+ {heuristic.age_coefficient:.6e} * age_s")

    out.rule("1. regression error on held-out products")
    out("tolerances: none. These are measurements, not assertions.")
    out("")
    predicted, dispersion = model.predict_with_dispersion(test_x)
    heuristic_predicted = heuristic.predict(test_x[:, 0], test_x[:, 1])
    constant = np.full(test_y.shape, float(train_y.mean()))
    out(f"{'estimator':>20s} {'MAE':>10s} {'RMSE':>10s} {'Spearman':>10s} {'bias':>11s}")
    for label, values in (
        ("learned forest", predicted),
        ("size/age fitted", heuristic_predicted),
        ("train mean", constant),
    ):
        error = values - test_y
        out(
            f"{label:>20s} {np.abs(error).mean():10.6f} "
            f"{np.sqrt((error**2).mean()):10.6f} {spearman(values, test_y):10.6f} "
            f"{error.mean():11.3e}"
        )
    out("")
    importances = model.feature_importances()
    out("impurity-based feature importances, which are biased towards high-cardinality")
    out("features and are reported with that caveat")
    for name, weight in sorted(importances.items(), key=lambda item: -item[1]):
        out(f"  {name:>26s} {weight:.6f}")

    out.rule("2. the uncertainty output")
    out("the forest reports the standard deviation of its per-tree predictions. This is a")
    out("dispersion measure over the ensemble and NOT a calibrated confidence interval:")
    out("the trees share training data, so the spread understates predictive uncertainty")
    out("(Wager, Hastie and Efron 2014). The measured coverage below is what that costs.")
    out("")
    absolute = np.abs(predicted - test_y)
    out(f"{'band':>12s} {'nominal normal':>15s} {'measured coverage':>18s}")
    for multiple, nominal in ((1.0, 0.6827), (1.96, 0.9500), (2.0, 0.9545)):
        coverage = float(np.mean(absolute <= multiple * dispersion))
        out(f"{f'+-{multiple:g} sigma':>12s} {nominal:15.4f} {coverage:18.6f}")
    out("")
    out(f"mean dispersion                       : {dispersion.mean():.6f} value units")
    out(f"mean absolute error                   : {absolute.mean():.6f} value units")
    out(f"Spearman(dispersion, |error|)          : "
        f"{spearman(dispersion, absolute):.6f}")
    out("a positive rank correlation means the dispersion is informative about where the")
    out("model is wrong even though its band is not calibrated, which is the useful")
    out("property: a product the ensemble disagrees about is a product to check by hand")

    out.rule("3. end-to-end regret through the exact solver")
    out(f"instances per scenario: {len(BENCHMARK_SEEDS)}, seeds "
        f"{BENCHMARK_SEEDS[0]}..{BENCHMARK_SEEDS[-1]}, 24 products, 3 windows,")
    out("                        capacity 45 per cent of the queue")
    out("ground-truth compression ratios are used for every arm, so the only thing that")
    out("varies between arms is the value estimate")
    out("regret = oracle selection value - arm selection value, both scored on the truth")
    out("")

    def learned_mapping(instance):
        features = np.array(
            [value_feature_row(product, instance.inventory) for product in instance.inventory],
            dtype=float,
        )
        prediction = model.predict(features)
        return {
            name: float(value)
            for name, value in zip(instance.inventory.names, prediction, strict=True)
        }

    arms = {
        "declared-median": lambda instance: declared_values(
            instance.inventory, fallback="median"
        ),
        "declared-zero": lambda instance: declared_values(instance.inventory, fallback="zero"),
        "size-age": lambda instance: heuristic.predict_inventory(instance.inventory),
        "learned": learned_mapping,
        "hybrid": lambda instance: hybrid_values(
            instance.inventory, learned_mapping(instance)
        ),
    }

    bench_timer = Timer()
    scenario_results: dict[str, tuple] = {}
    for label, kwargs in SCENARIOS:
        instances = [
            generate_instance(
                seed,
                n_products=24,
                n_windows=3,
                capacity_fraction=0.45,
                **kwargs,
            )
            for seed in BENCHMARK_SEEDS
        ]
        with bench_timer:
            results = benchmark_value_sources(instances, arms, selector=exact_select)
        scenario_results[label] = results
        out(f"scenario {label}")
        out(
            f"{'arm':>18s} {'MAE':>10s} {'Spearman':>10s} {'realised':>10s} "
            f"{'oracle':>10s} {'regret':>10s} {'regret frac':>12s} {'worst frac':>11s}"
        )
        for result in results:
            out(
                f"{result.name:>18s} {result.mae:10.6f} {result.spearman:10.6f} "
                f"{result.realised_value:10.5f} {result.oracle_value:10.5f} "
                f"{result.regret:10.5f} {result.regret_fraction:12.6f} "
                f"{result.worst_regret_fraction:11.6f}"
            )
        ranked = sorted(
            (r for r in results if r.name != "oracle"), key=lambda r: r.regret_fraction
        )
        out(f"  best non-oracle arm: {ranked[0].name} at regret fraction "
            f"{ranked[0].regret_fraction:.6f}")
        if any(not np.isfinite(r.spearman) for r in results):
            out("  Spearman is nan for an arm that emits one constant value for every")
            out("  product: a constant vector has no ranks. That is the finding, not a")
            out("  reporting artefact -- with no declaration anywhere, the passthrough has")
            out("  no information and the solver is choosing by size and reliability alone.")
        if abs(results[0].regret_fraction - results[1].regret_fraction) < 1.0e-12:
            undeclared = sum(
                len(instance.inventory.missing_declarations()) for instance in instances
            )
            if undeclared == 0:
                out("  declared-median and declared-zero coincide here because every value")
                out("  is declared, so neither fallback is ever used.")
            else:
                out("  declared-median and declared-zero coincide here because there is no")
                out("  declaration to take a median of, so both fall back to zero.")
        out("")

    out.rule("4. the specification's prediction, against what was measured")
    out("prediction: the exact solver on declared values beats everything, and the")
    out("            learned value model helps only where declarations are absent")
    out("")
    out(f"{'scenario':>46s} {'best arm':>18s} {'declared frac':>14s} {'learned frac':>13s}")
    prediction_holds: list[str] = []
    prediction_fails: list[str] = []
    for label, _kwargs in SCENARIOS:
        results = {r.name: r for r in scenario_results[label]}
        candidates = {name: r for name, r in results.items() if name != "oracle"}
        best = min(candidates.values(), key=lambda r: r.regret_fraction)
        out(
            f"{label:>46s} {best.name:>18s} "
            f"{results['declared-median'].regret_fraction:14.6f} "
            f"{results['learned'].regret_fraction:13.6f}"
        )
        if best.name.startswith("declared"):
            prediction_holds.append(label)
        else:
            prediction_fails.append(label)
    out("")
    out("scenarios in which a declared-value arm is the best non-oracle arm:")
    for label in prediction_holds or ["  none"]:
        out(f"  {label}")
    out("scenarios in which it is not:")
    for label in prediction_fails or ["  none"]:
        out(f"  {label}")

    out.rule("5. the same comparison through the greedy selector")
    out("if the ranking of value sources changed with the selector, the result would be a")
    out("statement about the solver and not about the estimators")
    out("")
    instances = [
        generate_instance(
            seed,
            n_products=24,
            n_windows=3,
            capacity_fraction=0.45,
            missing_declared_fraction=0.4,
            stale_declared_fraction=0.5,
        )
        for seed in BENCHMARK_SEEDS
    ]
    greedy_results = benchmark_value_sources(instances, arms, selector=greedy_select)
    exact_results = {r.name: r for r in scenario_results[SCENARIOS[4][0]]}
    out(f"{'arm':>18s} {'greedy regret frac':>19s} {'exact regret frac':>18s}")
    for result in greedy_results:
        out(
            f"{result.name:>18s} {result.regret_fraction:19.6f} "
            f"{exact_results[result.name].regret_fraction:18.6f}"
        )
    greedy_order = [
        r.name
        for r in sorted(
            (r for r in greedy_results if r.name != "oracle"),
            key=lambda r: r.regret_fraction,
        )
    ]
    exact_order = [
        r.name
        for r in sorted(
            (r for r in scenario_results[SCENARIOS[4][0]] if r.name != "oracle"),
            key=lambda r: r.regret_fraction,
        )
    ]
    out("")
    out(f"greedy ranking: {', '.join(greedy_order)}")
    out(f"exact ranking : {', '.join(exact_order)}")
    out(f"rankings agree: {greedy_order == exact_order}")

    out.rule("compute budget")
    out("timings are wall clock on two shared cores and are UNSTABLE between runs")
    out(f"{'quantity':>30s} {'value':>12s}")
    out(f"{'forest fit [s]':>30s} {fit_timer.elapsed_s:12.3f}")
    out(f"{'benchmark scenarios':>30s} {bench_timer.calls:12d}")
    out(f"{'benchmark wall clock [s]':>30s} {bench_timer.elapsed_s:12.3f}")

    out.rule("summary")
    out(f"failures: {failures}")
    out("this script has no pass/fail assertions on the model's accuracy by design: the")
    out("measured ranking is the deliverable, and a baseline win is evidence, not a bug")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
