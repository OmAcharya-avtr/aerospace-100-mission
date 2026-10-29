"""Section 3: the recursive estimator against the batch solution, which is the reference.

Checked here: agreement on the observable subspace to a stated tolerance, swept
over excitation profile, noise level and record length; what the filter does in
a direction the record does not support; and whether the prior is what separates
the two solutions, which is measured rather than assumed.
"""

from __future__ import annotations

from _bootstrap import SEED_GRID, Tee, Timer, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from aligncal import batch, dataset, observability, recursive  # noqa: E402

PROFILES = ("single_axis", "near_single_axis", "two_axis", "three_axis", "isotropic")
NOISES = (5.0e-5, 2.0e-4, 8.0e-4)
LENGTHS = (150, 600, 2400)
SEEDS_PER_CELL = 4

#: The declared agreement tolerances, as fractions of the Cramer-Rao one sigma
#: of (12), stated per regime because the measurement says the regime matters.
#:
#: ``well_conditioned``
#:     Full-rank records with a Gramian condition number below 10. This is the
#:     regime in which the claim "the recursive estimator gives the same answer
#:     online" is a tight statement, and 0.1 of the bound is a tenth of the
#:     noise both solutions are subject to.
#: ``full_rank``
#:     Every record where all three degrees of freedom clear the noise floor of
#:     (11), including badly conditioned ones.
#: ``rank_deficient``
#:     Records with a degree of freedom below the floor. No tolerance is
#:     declared here and the disagreement is reported as measured, because the
#:     recursive estimator does not withhold the unsupported direction and the
#:     large rotation it accumulates there leaks into the supported ones
#:     through the non-linearity of rotation composition. That leakage is a
#:     result, not a tolerance.
TOLERANCES = {"well_conditioned": 0.1, "full_rank": 0.5}

#: Gramian condition number below which a record counts as well conditioned.
WELL_CONDITIONED = 10.0


def main() -> int:
    out = Tee("validate_recursive_agreement_output.txt")
    out("VALIDATION 3 -- BATCH AGAINST RECURSIVE ON THE OBSERVABLE SUBSPACE")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out(
        "  declared tolerances, as fractions of the Cramer-Rao one sigma of (12), "
        "stated per regime:"
    )
    out(f"    well conditioned (rank 3, Gramian condition number < {WELL_CONDITIONED:g}): "
        f"{TOLERANCES['well_conditioned']:g}")
    out(f"    full rank, any conditioning                                 : "
        f"{TOLERANCES['full_rank']:g}")
    out("    rank deficient                                              : none declared; "
        "reported as measured")
    out(
        f"  grid: {len(PROFILES)} profiles x {len(NOISES)} noise levels x "
        f"{len(LENGTHS)} lengths x {SEEDS_PER_CELL} seeds = "
        f"{len(PROFILES) * len(NOISES) * len(LENGTHS) * SEEDS_PER_CELL} records"
    )

    timer_batch = Timer()
    timer_mekf = Timer()
    rows = []
    buckets: dict[str, list[float]] = {
        "well_conditioned": [],
        "full_rank": [],
        "rank_deficient": [],
    }
    retentions_unobservable: list[float] = []
    retentions_observable: list[float] = []
    inflations: list[float] = []
    mekf_unobservable_z: list[float] = []
    excursions: list[float] = []
    leakage: list[float] = []

    for profile in PROFILES:
        for noise in NOISES:
            for length in LENGTHS:
                worst = 0.0
                compared = 0
                deficient = 0
                for index in range(SEEDS_PER_CELL):
                    seed = SEED_GRID + 500 * index + 7 * length
                    generated = dataset.generate(
                        profile=profile, samples=length, noise_sigma=noise, seed=seed
                    )
                    verdict = observability.assess(generated.record)
                    sensor = verdict["secondary_1"]
                    deficient += sensor.rank_deficient
                    with timer_batch:
                        batch_solution = batch.solve(generated.record, verdict, "secondary_1")
                    with timer_mekf:
                        mekf = recursive.run(
                            generated.record, verdict, "secondary_1", store_history=False
                        )
                    retention = mekf.prior_retention()
                    inflations.append(mekf.inflation_ratio())
                    record_worst = 0.0
                    for dof in sensor.degrees_of_freedom:
                        if dof.verdict == "unobservable":
                            retentions_unobservable.append(float(retention[dof.index]))
                            component = float(np.dot(mekf.theta_rad, dof.direction))
                            truth = float(
                                np.dot(generated.truth["secondary_1"], dof.direction)
                            )
                            sigma = mekf.sigma_rad_in_direction(dof.direction)
                            mekf_unobservable_z.append(abs(component - truth) / sigma)
                            excursions.append(abs(component))
                            continue
                        retentions_observable.append(float(retention[dof.index]))
                        difference = abs(
                            float(
                                np.dot(
                                    mekf.theta_rad - batch_solution.theta_rad, dof.direction
                                )
                            )
                        )
                        record_worst = max(record_worst, difference / dof.crb_sigma_rad)
                        compared += 1
                    worst = max(worst, record_worst)
                    if sensor.rank_deficient:
                        buckets["rank_deficient"].append(record_worst)
                        leakage.append(record_worst)
                    else:
                        buckets["full_rank"].append(record_worst)
                        if sensor.condition_number < WELL_CONDITIONED:
                            buckets["well_conditioned"].append(record_worst)
                rows.append(
                    {
                        "profile": profile,
                        "noise": noise,
                        "length": length,
                        "worst": worst,
                        "compared": compared,
                        "deficient": deficient,
                    }
                )

    out.rule("3.1 agreement on the observable subspace, per cell")
    out(
        "'worst diff/CRB' is the largest absolute difference between the two "
        "estimates along any degree of freedom that clears the noise floor, divided "
        "by that degree of freedom's Cramer-Rao one sigma."
    )
    out("")
    out(f"{'profile':<17s} {'sigma':>8s} {'n':>5s} {'dofs':>5s} {'deficient':>10s} "
        f"{'worst diff/CRB':>15s}")
    for row in rows:
        out(
            f"{row['profile']:<17s} {row['noise']:>8.1e} {row['length']:>5d} "
            f"{row['compared']:>5d} {row['deficient']:>10d} {row['worst']:>15.6f}"
        )
    out("")
    out(f"{'regime':<20s} {'records':>8s} {'median':>12s} {'max':>12s} {'declared':>10s}")
    for name in ("well_conditioned", "full_rank", "rank_deficient"):
        values = np.array(buckets[name])
        declared = TOLERANCES.get(name)
        out(
            f"{name:<20s} {values.size:>8d} {float(np.median(values)):>12.6f} "
            f"{float(np.max(values)):>12.6f} "
            f"{(format(declared, 'g') if declared is not None else 'none'):>10s}"
        )
    out("")
    out(f"  degrees of freedom compared: {sum(row['compared'] for row in rows)}")
    for name, declared in TOLERANCES.items():
        values = np.array(buckets[name])
        out.check(
            f"batch and recursive agree to {declared:g} of the bound on {name} records",
            bool(np.max(values) < declared),
            f"worst {float(np.max(values)):.6f} of the Cramer-Rao one sigma over "
            f"{values.size} records, median {float(np.median(values)):.6e}",
        )

    out.rule("3.1b the rank-deficient records, reported as measured")
    deficient_values = np.array(buckets["rank_deficient"])
    excursion = np.array(excursions)
    out(
        "On a record with a degree of freedom below the noise floor the recursive "
        "estimator does not withhold anything: it accumulates a large rotation in the "
        "direction the record cannot support, and because composing rotations is not "
        "a linear operation that rotation moves the components the record DOES "
        "support. This is the same mechanism that makes the unconstrained SVD "
        "solution of (17) wrong about the observable subspace, measured independently "
        "in tests/test_batch.py, and it is the strongest argument in this repository "
        "for computing the observability verdict first."
    )
    out("")
    out(f"  rank-deficient records                  : {deficient_values.size}")
    out(f"  MEKF excursion in the unsupported dir   : median "
        f"{float(np.median(excursion)) * 1e3:.4f} mrad, max "
        f"{float(np.max(excursion)) * 1e3:.4f} mrad")
    out("  injected misalignment magnitude bound   : 3.0000 mrad")
    out(f"  leakage into the supported directions   : median "
        f"{float(np.median(deficient_values)):.6f}, max "
        f"{float(np.max(deficient_values)):.6f} of the Cramer-Rao one sigma")
    out(f"  full-rank records, same quantity        : median "
        f"{float(np.median(np.array(buckets['full_rank']))):.6e}, max "
        f"{float(np.max(np.array(buckets['full_rank']))):.6f}")
    correlation = float(np.corrcoef(excursion, np.array(leakage))[0, 1])
    out(f"  correlation(excursion, leakage)         : {correlation:+.4f}")
    out("")
    out.check(
        "the leakage is larger on rank-deficient records than on full-rank ones",
        float(np.max(deficient_values)) > 4.0 * float(np.max(np.array(buckets["full_rank"]))),
        f"rank-deficient max {float(np.max(deficient_values)):.6f} against full-rank max "
        f"{float(np.max(np.array(buckets['full_rank']))):.6f}",
    )
    out.check(
        "the leakage is explained by the size of the unconstrained excursion",
        correlation > 0.2,
        f"correlation {correlation:+.4f} between the MEKF's rotation in the unsupported "
        "direction and its disagreement with the batch solution on the supported ones",
    )

    out.rule("3.2 what the filter does where the record supports nothing")
    out(
        "The MEKF carries a prior of one sigma 0.1 rad and solves over the full "
        "space, so it does not withhold anything. In a direction the record does not "
        "excite, the measurement Jacobian of (20) has no component, the gain is "
        "near zero and the covariance stays near the prior. 'prior retained' is the "
        "final one-sigma divided by the prior one-sigma: 1 means the record taught "
        "the filter nothing."
    )
    out("")
    unobservable = np.array(retentions_unobservable)
    observable = np.array(retentions_observable)
    inflation = np.array(inflations)
    mekf_z = np.array(mekf_unobservable_z)
    out(f"  degrees of freedom below the floor      : {unobservable.size}")
    out(f"  prior retained there, median            : {float(np.median(unobservable)):.6f}")
    out(f"  prior retained there, min and max       : {float(np.min(unobservable)):.6f}, "
        f"{float(np.max(unobservable)):.6f}")
    out(f"  prior retained on observable dofs, median: {float(np.median(observable)):.6e}")
    out(f"  worst/best sigma ratio, median          : {float(np.median(inflation)):.4f}")
    out(f"  worst/best sigma ratio, max             : {float(np.max(inflation)):.4f}")
    out("")
    out(
        "The filter is therefore honest about what it does not know, through its "
        "covariance rather than through a refusal. The next figure is the test of "
        "that honesty: its error in the unsupported direction, divided by the "
        "one-sigma it reports there."
    )
    out(f"  |error| / reported sigma, median        : {float(np.median(mekf_z)):.6f}")
    out(f"  |error| / reported sigma, 90th centile  : "
        f"{float(np.percentile(mekf_z, 90)):.6f}")
    out(f"  fraction inside 2 reported sigma        : {float(np.mean(mekf_z <= 2.0)):.6f}")
    out.check(
        "the filter retains at least a tenth of its prior where the record supports nothing",
        bool(np.median(unobservable) > 0.1),
        f"median retention {float(np.median(unobservable)):.6f}",
    )
    out.check(
        "the filter reduces its prior by 100x where the record does support a dof",
        bool(np.median(observable) < 0.01),
        f"median retention {float(np.median(observable)):.6e}",
    )
    out("")
    out(
        "  MEASURED AND UNFAVOURABLE, published as found: the filter's two-sigma "
        f"interval contains the truth on only {float(np.mean(mekf_z <= 2.0)):.6f} of "
        "the records with an unsupported degree of freedom, against a nominal "
        "0.954500. The recursive estimator is therefore overconfident in the "
        "unsupported direction too -- less so than the learned estimator of section "
        "5, which is measured there, but overconfident. The mechanism is equation "
        "(11): the filter forms its Jacobian from the MEASURED reference rate, so the "
        "reference sensor's own noise supplies apparent information in the "
        "unsupported direction and the filter reduces its prior there on the strength "
        "of it. Split by how much prior the filter kept:"
    )
    retained = np.array(retentions_unobservable)
    median_retention = float(np.median(retained))
    low = retained < median_retention
    out("")
    out(f"    prior retained below the median ({median_retention:.4f}): "
        f"{float(np.mean(mekf_z[low] <= 2.0)):.6f} inside 2 sigma ({int(low.sum())} records)")
    out(f"    prior retained above the median            : "
        f"{float(np.mean(mekf_z[~low] <= 2.0)):.6f} inside 2 sigma "
        f"({int((~low).sum())} records)")
    correlation_z = float(np.corrcoef(retained, mekf_z)[0, 1])
    out(f"    correlation(prior retained, |error|/sigma) : {correlation_z:+.4f}")
    out("")
    out.check(
        "the under-coverage is explained by the filter reducing its prior on noise alone",
        correlation_z < -0.2
        and float(np.mean(mekf_z[~low] <= 2.0)) > float(np.mean(mekf_z[low] <= 2.0)),
        f"correlation {correlation_z:+.4f}; coverage "
        f"{float(np.mean(mekf_z[~low] <= 2.0)):.6f} where the prior was kept against "
        f"{float(np.mean(mekf_z[low] <= 2.0)):.6f} where it was not. The only estimator "
        "in this repository that is not overconfident here is the batch solution of "
        "(14)-(16), which withholds the direction and reports an infinite bound.",
    )

    out.rule("3.3 is the prior what separates the two solutions? Measured, not assumed.")
    out(
        "The shrinkage argument predicts the recursive estimate is the batch estimate "
        "scaled by lambda_i / (lambda_i + 1 / p0^2) along degree of freedom i, so "
        "tightening the prior from 0.1 rad to 0.01 rad should cut the disagreement by "
        "a factor of 100. Measured below on rich 1200-sample records."
    )
    out("")
    out(f"{'prior p0 rad':>13s} {'worst diff/CRB':>15s} {'predicted shrink':>17s}")
    measured = {}
    for prior in (0.1, 0.03, 0.01, 0.003):
        worst = 0.0
        predicted = 0.0
        for index in range(6):
            generated = dataset.generate(
                profile="three_axis", samples=1200, seed=SEED_GRID + 9000 + index
            )
            verdict = observability.assess(generated.record)
            sensor = verdict["secondary_1"]
            batch_solution = batch.solve(generated.record, verdict, "secondary_1")
            mekf = recursive.run(
                generated.record,
                verdict,
                "secondary_1",
                initial_sigma_rad=prior,
                store_history=False,
            )
            for dof in sensor.degrees_of_freedom:
                difference = abs(
                    float(np.dot(mekf.theta_rad - batch_solution.theta_rad, dof.direction))
                )
                worst = max(worst, difference / dof.crb_sigma_rad)
                component = abs(float(np.dot(batch_solution.theta_rad, dof.direction)))
                predicted = max(
                    predicted,
                    component / (dof.eigenvalue * prior * prior) / dof.crb_sigma_rad,
                )
        measured[prior] = worst
        out(f"{prior:>13g} {worst:>15.6e} {predicted:>17.6e}")
    out("")
    ratio = measured[0.1] / measured[0.003]
    out(
        f"  The prior changed by a factor of {0.1 / 0.003:.1f}, so the shrinkage "
        f"argument predicts the disagreement to change by {(0.1 / 0.003) ** 2:.0f}. "
        f"It changed by {ratio:.4f}."
    )
    out.check(
        "the disagreement does NOT scale as the prior-shrinkage argument predicts",
        ratio < 10.0,
        f"measured factor {ratio:.4f} against a predicted {(0.1 / 0.003) ** 2:.0f}; the "
        "residual disagreement is dominated by the sequential re-linearisation of (20) "
        "about a moving estimate, not by the filter's prior, so quoting the shrinkage "
        "figure as the expected agreement would be wrong by three orders of magnitude",
    )

    out.rule("3.4 the Joseph form of (21), and whether it was needed")
    out(
        "The short-form covariance update loses symmetry and positive definiteness "
        "over the thousands of updates a rate record contains. The Joseph form is "
        "used instead, and its measured asymmetry and smallest eigenvalue are "
        "reported here so the claim is a number rather than an assertion."
    )
    out("")
    out(f"{'n':>7s} {'worst P asymmetry':>19s} {'smallest P eigenvalue':>23s}")
    for length in (150, 600, 2400, 6000):
        generated = dataset.generate(profile="three_axis", samples=length, seed=SEED_GRID + 11)
        verdict = observability.assess(generated.record)
        mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=False)
        out(f"{length:>7d} {mekf.joseph_symmetry:>19.6e} {mekf.min_eigenvalue:>23.6e}")
    generated = dataset.generate(profile="three_axis", samples=6000, seed=SEED_GRID + 11)
    verdict = observability.assess(generated.record)
    mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=False)
    out("")
    out.check(
        "the covariance stays positive definite over 6000 updates",
        mekf.min_eigenvalue > 0.0,
        f"smallest eigenvalue seen {mekf.min_eigenvalue:.6e} rad^2",
    )
    out.check(
        "the covariance stays symmetric to rounding over 6000 updates",
        mekf.joseph_symmetry < 1.0e-12,
        f"worst asymmetry {mekf.joseph_symmetry:.6e} rad^2",
    )

    out.rule("3.5 timings -- UNSTABLE, two shared cores, never quoted as a ratio in prose")
    out(f"{'operation':<34s} {'calls':>7s} {'total s':>10s} {'mean ms':>10s}")
    out(f"{'batch.solve, eq (14)-(16)':<34s} {timer_batch.calls:>7d} "
        f"{timer_batch.elapsed_s:>10.4f} {timer_batch.mean_ms:>10.4f}")
    out(f"{'recursive.run, eq (19)-(22)':<34s} {timer_mekf.calls:>7d} "
        f"{timer_mekf.elapsed_s:>10.4f} {timer_mekf.mean_ms:>10.4f}")
    out("")
    out(
        "  The recursive estimator processes one sample at a time in Python and is "
        "slower than the batch solve here by an amount that is not reported as a "
        "ratio, because these figures move between runs on a shared two-core "
        "container. The recursive estimator exists to give an answer online, not to "
        "be faster."
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
