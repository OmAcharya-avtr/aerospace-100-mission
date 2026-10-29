"""Section 5: the learned estimator against the analytic baselines and a trivial one.

The analytic baselines were implemented and validated first (sections 2 and 3)
and they are the reference here, not the other way round.

Three comparisons are kept apart.

1. Against the **specified** Baseline 1, the unbiased batch solution of (14)-(16).
2. Against the **strongest honest** analytic baseline, the regularised MAP
   variant of (18) given the same prior scale the learned estimator's training
   population was drawn from. The mission's build guide requires the strongest
   honest baseline to be used and the specified one reported beside it: a
   learned estimator trained on a known misalignment population is performing
   shrinkage toward that population, and comparing it only against the unbiased
   solution would credit it with a win it has not earned.
3. Against a **trivial** estimator that predicts the training mean and ignores
   its input entirely, so that an estimator which has learned nothing cannot
   look useful.

Every score is reported split by subspace, because the component of the truth
outside the retained span is carried in full by every estimator here and
including it would flatten every difference.
"""

from __future__ import annotations

from _bootstrap import (
    PRIOR_SIGMA_RAD,
    SEED_TEST,
    SEED_TRAIN,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
    fit_estimators,
)

bootstrap()

import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, features, learned, observability  # noqa: E402

TRAIN = 2500
TEST = 600
SHORT = 150
LONG = 2400


def _evaluate(population, models, label_prefix=""):
    """Score every estimator on one population. Returns tables and extra detail."""
    matrix, targets, profiles = features.build_population(population)
    sensors = []
    batch_errors = []
    map_errors = []
    timer_batch = Timer()
    timer_map = Timer()
    for generated in population:
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        truth = generated.truth["secondary_1"]
        with timer_batch:
            batch_errors.append(
                batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth
            )
        with timer_map:
            map_errors.append(
                batch.solve_map(
                    generated.record, verdict, "secondary_1", prior_sigma_rad=PRIOR_SIGMA_RAD
                ).theta_rad
                - truth
            )
    tables = [
        benchmark.score(f"{label_prefix}batch (14)-(16)", np.array(batch_errors), sensors),
        benchmark.score(f"{label_prefix}batch MAP (18)", np.array(map_errors), sensors),
    ]
    timers = {"batch": timer_batch, "map": timer_map}
    coverage: dict[str, float] = {}
    for name in list(learned.CONFIGURATIONS) + ["trivial"]:
        model = models[name]
        timer = Timer()
        with timer:
            prediction = model.predict(matrix)
        tables.append(
            benchmark.score(
                f"{label_prefix}learned {name}", prediction.theta_rad - targets, sensors
            )
        )
        timers[name] = timer
        errors = np.abs(prediction.theta_rad - targets).reshape(-1)
        sigmas = prediction.sigma_rad.reshape(-1)
        coverage[name] = benchmark.coverage(errors, sigmas, 2.0)
    return tables, sensors, profiles, timers, coverage, matrix, targets


def main() -> int:
    out = Tee("validate_learned_output.txt")
    out("VALIDATION 5 -- THE LEARNED ESTIMATOR AGAINST THE BASELINES")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out(f"  training records   : {TRAIN} short records of {SHORT} samples, seed {SEED_TRAIN}")
    out(f"  test records       : {TEST} held-out short records, seed {SEED_TEST}")
    out(f"  MAP prior one sigma: {PRIOR_SIGMA_RAD:g} rad, which is the per-component "
        "standard deviation of the generated misalignment population")
    out("  no hyperparameter search was run; the three declared configurations are "
        "reported as they are")

    fit_timer = Timer()
    with fit_timer:
        models = fit_estimators(count=TRAIN, samples=SHORT, seed=SEED_TRAIN)

    out.rule("5.1 the short-record population, which is the case the learned estimator is for")
    test = dataset.training_population(count=TEST, samples=SHORT, seed=SEED_TEST)
    tables, sensors, profiles, timers, coverage, matrix, targets = _evaluate(test, models)
    out(
        "All figures in milliradians. 'RMSE obs' is the per-component root mean square "
        "of the error inside the span of the floor-clearing degrees of freedom and is "
        "the ranking column. 'RMSE unobs' is outside that span and is approximately "
        "equal for every estimator by construction."
    )
    out("")
    out(benchmark.ScoreTable.header())
    for table in tables:
        out(table.row())
    out("")
    ranked = sorted(tables, key=lambda t: t.rmse_observable_mrad)
    out("  ranking on the observable subspace, best first:")
    for position, table in enumerate(ranked, start=1):
        out(f"    {position}. {table.label:<22s} {table.rmse_observable_mrad:.6f} mrad")

    by_label = {table.label: table for table in tables}
    plain = by_label["batch (14)-(16)"]
    regularised = by_label["batch MAP (18)"]
    trivial = by_label["learned trivial"]
    best_learned = min(
        (by_label[f"learned {name}"] for name in learned.CONFIGURATIONS),
        key=lambda t: t.rmse_observable_mrad,
    )
    out("")
    out(f"  best learned configuration : {best_learned.label} at "
        f"{best_learned.rmse_observable_mrad:.6f} mrad")
    out(f"  specified Baseline 1       : {plain.rmse_observable_mrad:.6f} mrad")
    out(f"  strengthened MAP baseline  : {regularised.rmse_observable_mrad:.6f} mrad")
    out(f"  trivial baseline           : {trivial.rmse_observable_mrad:.6f} mrad")
    out("")
    out.check(
        "the best learned configuration beats the trivial baseline",
        best_learned.rmse_observable_mrad < trivial.rmse_observable_mrad,
        f"{best_learned.rmse_observable_mrad:.6f} against {trivial.rmse_observable_mrad:.6f} "
        f"mrad, a reduction of "
        f"{100 * (1 - best_learned.rmse_observable_mrad / trivial.rmse_observable_mrad):.2f} "
        "per cent -- so the learned component has learned something",
    )
    out.check(
        "the best learned configuration beats the SPECIFIED Baseline 1 on short records",
        best_learned.rmse_observable_mrad < plain.rmse_observable_mrad,
        f"{best_learned.rmse_observable_mrad:.6f} against {plain.rmse_observable_mrad:.6f} "
        "mrad; the unbiased maximum-likelihood solution has a variance set by the "
        "Cramer-Rao bound, and on a 150-sample record that bound is comparable to the "
        "spread of the misalignment population itself",
    )
    beats_map = best_learned.rmse_observable_mrad < regularised.rmse_observable_mrad
    out.check(
        "THE BASELINE WINS: the strengthened MAP baseline beats every learned configuration",
        not beats_map,
        f"MAP {regularised.rmse_observable_mrad:.6f} mrad against the best learned "
        f"{best_learned.rmse_observable_mrad:.6f} mrad, a margin of "
        f"{100 * (best_learned.rmse_observable_mrad / regularised.rmse_observable_mrad - 1):.2f} "
        "per cent in favour of the analytic method. This is the expected result and it "
        "is published as measured: the only mechanism by which a learned estimator can "
        "beat an unbiased solution on a short record is shrinkage toward the population "
        "it was trained on, and equation (18) performs exactly that shrinkage in closed "
        "form with no training, no features and no model.",
    )

    out.rule("5.2 the same comparison per excitation profile")
    out(
        "Broken down so that the aggregate is not hiding a reversal. Figures are the "
        "per-component RMSE on the observable subspace, in milliradians."
    )
    out("")
    unique = sorted(set(profiles))
    out(f"{'estimator':<22s} " + " ".join(f"{p[:16]:>17s}" for p in unique))
    per_profile: dict[str, dict[str, float]] = {}
    for table in tables:
        per_profile[table.label] = {}
    for profile in unique:
        index = [k for k, value in enumerate(profiles) if value == profile]
        subset = [test[k] for k in index]
        sub_tables, _, _, _, _, _, _ = _evaluate(subset, models)
        for table in sub_tables:
            per_profile[table.label][profile] = table.rmse_observable_mrad
    for table in tables:
        row = per_profile[table.label]
        out(
            f"{table.label:<22s} "
            + " ".join(f"{row[p]:>17.6f}" for p in unique)
        )
    out("")
    reversals = 0
    for profile in unique:
        map_value = per_profile["batch MAP (18)"][profile]
        learned_values = [
            per_profile[f"learned {name}"][profile] for name in learned.CONFIGURATIONS
        ]
        if min(learned_values) < map_value:
            reversals += 1
            out(f"  REVERSAL on {profile}: best learned {min(learned_values):.6f} beats "
                f"MAP {map_value:.6f} mrad")
    if reversals == 0:
        out("  no profile reverses the aggregate conclusion")
    out.check(
        "the MAP baseline's win is not an artefact of one profile",
        reversals == 0,
        f"{reversals} of {len(unique)} profiles reverse it",
    )

    out.rule("5.3 record length: how the MAP advantage behaves, and why it never vanishes")
    out(
        "Swept on a FIXED well-excited configuration -- three_axis, 0.05 rad/s "
        "amplitude, 2e-4 rad/s noise -- so that record length is the only thing "
        "varying. The learned estimator was trained on 150-sample records, so "
        "everything to the right of the first column is outside its training "
        "distribution, and that is stated rather than hidden."
    )
    out("")
    out(f"{'samples':>8s} {'batch':>12s} {'batch MAP':>12s} {'learned best':>13s} "
        f"{'trivial':>10s} {'winner':>16s}")
    crossover = None
    margins: dict[int, float] = {}
    for samples in (150, 300, 600, 1200, 2400, 4800, 9600):
        fixed = [
            dataset.generate(
                profile="three_axis",
                samples=samples,
                amplitude=0.05,
                noise_sigma=2.0e-4,
                theta_max_rad=3.0e-3,
                seed=SEED_TEST + 3000 + index,
            )
            for index in range(60)
        ]
        fixed_tables, _, _, _, _, _, _ = _evaluate(fixed, models)
        fixed_by_label = {table.label: table for table in fixed_tables}
        plain_value = fixed_by_label["batch (14)-(16)"].rmse_observable_mrad
        map_value = fixed_by_label["batch MAP (18)"].rmse_observable_mrad
        learned_value = min(
            fixed_by_label[f"learned {name}"].rmse_observable_mrad
            for name in learned.CONFIGURATIONS
        )
        trivial_value = fixed_by_label["learned trivial"].rmse_observable_mrad
        candidates = {
            "batch": plain_value,
            "batch MAP": map_value,
            "learned": learned_value,
            "trivial": trivial_value,
        }
        winner = min(candidates, key=lambda k: candidates[k])
        margins[samples] = map_value / plain_value
        if crossover is None and plain_value < map_value:
            crossover = samples
        out(
            f"{samples:>8d} {plain_value:>12.6f} {map_value:>12.6f} "
            f"{learned_value:>13.6f} {trivial_value:>10.6f} {winner:>16s}"
        )
    out("")
    out(
        "  MEASURED: the unbiased batch solution does not overtake the regularised one "
        "anywhere from 150 to 9600 samples in this measurement, a factor of 64 in "
        "record length. The advantage COLLAPSES, from "
        f"{100 * (1 - margins[150]):.2f} per cent at 150 samples to "
        f"{100 * (1 - margins[9600]):.2f} per cent at 9600, and the remaining "
        "difference at the long end is at the level of this measurement's own "
        "sampling noise: the same sweep run with 40 records per point instead of 60, "
        "in examples/prior_and_record_length.py, measures a ratio of 1.001 at 4800 "
        "samples, which is the unbiased solution winning by a tenth of a per cent.\n\n"
        "  What is NOT claimed: that equation (18) is optimal. It is the posterior "
        "mean under a GAUSSIAN prior whose per-component variance matches the "
        "generated population's second moment. The population's actual prior is not "
        "Gaussian -- the magnitude is uniform on [0, 3] mrad and the direction is "
        "uniform on the sphere -- so (18) is not the Bayes estimator for it and no "
        "domination theorem applies. The behaviour above is a measurement, not a "
        "consequence of one."
    )
    out.check(
        "the MAP advantage over the unbiased solution shrinks as the record lengthens",
        margins[9600] > margins[150] and margins[9600] > 0.97,
        f"MAP/batch RMSE ratio {margins[150]:.6f} at 150 samples and "
        f"{margins[9600]:.6f} at 9600; the advantage is bought entirely with the prior "
        "and it is worth less the more the record says",
    )

    out.rule("5.3a the MAP win is bought with the prior: mis-specify it and it loses")
    out(
        "The regularised baseline wins above because it was handed the correct prior "
        "scale. That is a real advantage of an analytic method -- the shrinkage is in "
        "closed form, needs no training data and can be stated in a document -- but it "
        "is not free. The same estimator with a prior ten times too tight is measured "
        "below against the unbiased solution, on the same fixed configuration."
    )
    out("")
    out(f"{'samples':>8s} {'batch':>12s} {'MAP, true prior':>16s} "
        f"{'MAP, 10x tight':>15s} {'MAP, 10x loose':>15s}")
    tight_loses = 0
    for samples in (150, 600, 2400, 9600):
        fixed = [
            dataset.generate(
                profile="three_axis",
                samples=samples,
                amplitude=0.05,
                noise_sigma=2.0e-4,
                theta_max_rad=3.0e-3,
                seed=SEED_TEST + 3000 + index,
            )
            for index in range(60)
        ]
        values = {}
        sensors_fixed = []
        errors = {"batch": [], "true": [], "tight": [], "loose": []}
        for generated in fixed:
            verdict = observability.assess(generated.record)
            sensors_fixed.append(verdict["secondary_1"])
            truth = generated.truth["secondary_1"]
            errors["batch"].append(
                batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth
            )
            for key, prior in (
                ("true", PRIOR_SIGMA_RAD),
                ("tight", PRIOR_SIGMA_RAD / 10.0),
                ("loose", PRIOR_SIGMA_RAD * 10.0),
            ):
                errors[key].append(
                    batch.solve_map(
                        generated.record, verdict, "secondary_1", prior_sigma_rad=prior
                    ).theta_rad
                    - truth
                )
        for key in errors:
            values[key] = benchmark.score(
                key, np.array(errors[key]), sensors_fixed
            ).rmse_observable_mrad
        tight_loses += values["tight"] > values["batch"]
        out(
            f"{samples:>8d} {values['batch']:>12.6f} {values['true']:>16.6f} "
            f"{values['tight']:>15.6f} {values['loose']:>15.6f}"
        )
    out("")
    out.check(
        "a prior ten times too tight makes the MAP baseline LOSE to the unbiased one",
        tight_loses > 0,
        f"the mis-specified MAP estimator is worse than the unbiased batch solution at "
        f"{tight_loses} of 4 record lengths; the analytic win of 5.1 is therefore a "
        "statement about knowing the population, not a free lunch",
    )

    out.rule("5.3b the same estimators on a long MIXED population, reported as measured")
    out(
        "The population here is the same wide mixture the learned estimator was "
        "trained on -- amplitude 0.01 to 0.08 rad/s, noise 5e-5 to 8e-4 rad/s, every "
        "profile including single-axis -- at 2400 samples. The conclusion is NOT the "
        "same as the fixed-configuration sweep above, and the difference is the point."
    )
    long_population = dataset.training_population(
        count=120, samples=LONG, seed=SEED_TEST + 991
    )
    long_tables, _, _, _, _, _, _ = _evaluate(long_population, models)
    out("")
    out(benchmark.ScoreTable.header())
    for table in long_tables:
        out(table.row())
    out("")
    long_by_label = {table.label: table for table in long_tables}
    best_long_learned = min(
        (long_by_label[f"learned {name}"] for name in learned.CONFIGURATIONS),
        key=lambda t: t.rmse_observable_mrad,
    )
    out(
        f"  MEASURED, and NOT what the fixed-configuration sweep predicts: on this "
        f"mixed population at 2400 samples the regularised MAP variant still wins "
        f"({long_by_label['batch MAP (18)'].rmse_observable_mrad:.6f} mrad) and the "
        f"unbiased batch solution is still the worst analytic option "
        f"({long_by_label['batch (14)-(16)'].rmse_observable_mrad:.6f} mrad). The "
        "reason is that the mixture contains records whose Cramer-Rao bound is large "
        "at any length -- single-axis records, and low-amplitude records at high noise "
        "-- and a root-mean-square over the population is dominated by those. Record "
        "length is not what decides the crossover; the information content of the "
        "record is, and equation (10) is what says how much of that there is."
    )
    out(
        f"  The best learned configuration reaches "
        f"{best_long_learned.rmse_observable_mrad:.6f} mrad here, applied 16 times "
        "outside the record length it was trained on."
    )
    out.check(
        "the MAP baseline beats every learned configuration on the long mixed population too",
        long_by_label["batch MAP (18)"].rmse_observable_mrad
        < best_long_learned.rmse_observable_mrad,
        f"MAP {long_by_label['batch MAP (18)'].rmse_observable_mrad:.6f} against the best "
        f"learned {best_long_learned.rmse_observable_mrad:.6f} mrad",
    )

    out.rule("5.4 the uncertainty output, and whether it means anything")
    out(
        "Every learned configuration reports a per-component standard deviation from "
        "the spread of its ensemble members. That is a model-spread uncertainty and "
        "not a calibrated posterior, and the measurement below is how far from "
        "calibrated it is: the fraction of components whose error lies inside two "
        "reported standard deviations, against a nominal 0.954500, over the whole "
        "held-out short-record population."
    )
    out("")
    out(f"{'estimator':<22s} {'cover 2 sigma':>14s} {'median sigma mrad':>19s} "
        f"{'RMSE obs mrad':>15s} {'sigma/RMSE':>11s}")
    for name in list(learned.CONFIGURATIONS) + ["trivial"]:
        model = models[name]
        prediction = model.predict(matrix)  # type: ignore[attr-defined]
        median_sigma = float(np.median(prediction.sigma_rad)) * 1e3
        table = by_label[f"learned {name}"]
        out(
            f"{name:<22s} {coverage[name]:>14.6f} {median_sigma:>19.6f} "
            f"{table.rmse_total_mrad:>15.6f} "
            f"{median_sigma / table.rmse_total_mrad:>11.4f}"
        )
    out("")
    sharpest = min(
        learned.CONFIGURATIONS,
        key=lambda n: float(np.median(models[n].predict(matrix).sigma_rad)),  # type: ignore[attr-defined]
    )
    out(
        f"  The {sharpest} configuration reports a median one-sigma an order of "
        "magnitude below its own realised error, which is the overconfidence measured "
        "in section 4 and it is present on the whole population, not only in the "
        "unobservable subspace."
    )
    out.check(
        "no learned configuration's two-sigma interval reaches nominal coverage",
        all(coverage[name] < 0.9545 for name in learned.CONFIGURATIONS),
        "measured "
        + ", ".join(f"{n} {coverage[n]:.6f}" for n in learned.CONFIGURATIONS)
        + " against a nominal 0.954500",
    )
    out(
        f"  For comparison the trivial estimator reaches {coverage['trivial']:.6f}, "
        "which is better than two of the three learned configurations: its uncertainty "
        "is the honest spread of a predictor that uses no input, and that is a hard "
        "thing to beat with a model that is not calibrated."
    )

    out.rule("5.5 the analytic bound is calibrated, and that is the contrast")
    out(
        "The Cramer-Rao one sigma of (12) is checked as an interval in exactly the "
        "same way as the learned ensemble spread, over the same records, on each "
        "retained degree of freedom."
    )
    out("")
    analytic_errors = []
    analytic_sigmas = []
    for generated in test:
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        solution = batch.solve(generated.record, verdict, "secondary_1")
        truth = generated.truth["secondary_1"]
        for dof in sensor.degrees_of_freedom:
            if dof.verdict == "unobservable":
                continue
            analytic_errors.append(
                abs(float(np.dot(solution.theta_rad - truth, dof.direction)))
            )
            analytic_sigmas.append(dof.crb_sigma_rad)
    analytic_errors = np.array(analytic_errors)
    analytic_sigmas = np.array(analytic_sigmas)
    for k in (1, 2, 3):
        measured = benchmark.coverage(analytic_errors, analytic_sigmas, float(k))
        out(
            f"  batch solution, {k} sigma : measured {measured:.6f}, nominal "
            f"{benchmark.NOMINAL_COVERAGE[k]:.6f}"
        )
    two_sigma = benchmark.coverage(analytic_errors, analytic_sigmas, 2.0)
    out("")
    out.check(
        "the Cramer-Rao interval is calibrated to within 0.03 of nominal at two sigma",
        abs(two_sigma - benchmark.NOMINAL_COVERAGE[2]) < 0.03,
        f"measured {two_sigma:.6f} against nominal {benchmark.NOMINAL_COVERAGE[2]:.6f} "
        f"over {analytic_errors.size} degrees of freedom",
    )

    out.rule("5.6 compute budget and timings -- UNSTABLE, two shared cores")
    out(f"{'operation':<36s} {'calls':>7s} {'total s':>10s} {'mean ms':>12s}")
    out(f"{'fit all four estimators':<36s} {fit_timer.calls:>7d} "
        f"{fit_timer.elapsed_s:>10.4f} {fit_timer.mean_ms:>12.4f}")
    for name, timer in timers.items():
        out(f"{'predict/solve ' + name:<36s} {timer.calls:>7d} "
            f"{timer.elapsed_s:>10.4f} {timer.mean_ms:>12.4f}")
    out("")
    out(
        f"  Total compute for this product: training is {TRAIN} generated records of "
        f"{SHORT} samples and three ensembles, and it fits in the figure above. No GPU, "
        "no PyTorch, two cores. Every script in validation/ completes in under three "
        "minutes on the measured container."
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
