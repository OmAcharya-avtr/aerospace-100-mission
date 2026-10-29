"""Section 2: recovery of known misalignments across the Level 2 sweep.

The grid is excitation richness x noise level x record length. Known
misalignments are injected and recovered, and the error is reported against the
Cramer-Rao bound of equation (12) in two ways: as a root-mean-square ratio, and
as the standardised score of equation (24), whose distribution over the whole
grid is compared against a standard normal.

Two things are kept apart throughout. The error INSIDE the span of the
floor-clearing degrees of freedom is what the record can be held responsible
for, and it is what the bound applies to. The error OUTSIDE that span is the
component of the truth the record cannot support; every estimator carries it in
full, including the one that correctly refuses to guess at it, so including it
in a comparison would make the refusal look like a defect.
"""

from __future__ import annotations

from _bootstrap import (
    PRIOR_SIGMA_RAD,
    SEED_GRID,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

bootstrap()

import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, observability  # noqa: E402

PROFILES = ("single_axis", "near_single_axis", "two_axis", "three_axis")
NOISES = (5.0e-5, 2.0e-4, 8.0e-4)
LENGTHS = (150, 600, 2400)
SEEDS_PER_CELL = 24


def main() -> int:
    out = Tee("validate_recovery_grid_output.txt")
    out("VALIDATION 2 -- RECOVERY ACROSS THE SWEPT GRID, AGAINST THE CRAMER-RAO BOUND")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out(
        f"  grid: {len(PROFILES)} excitation profiles x {len(NOISES)} noise levels x "
        f"{len(LENGTHS)} record lengths x {SEEDS_PER_CELL} seeds = "
        f"{len(PROFILES) * len(NOISES) * len(LENGTHS) * SEEDS_PER_CELL} records"
    )
    out("  injected misalignment: magnitude uniform on [0, 3] mrad, direction "
        "uniform on the sphere")
    out("  body rate amplitude: 0.05 rad/s; sampling 100 Hz")
    out(f"  MAP prior one sigma (eq 18): {PRIOR_SIGMA_RAD:g} rad")

    timer_batch = Timer()
    timer_map = Timer()
    rows = []
    all_scores: list[float] = []
    all_map_scores: list[float] = []

    for profile in PROFILES:
        for noise in NOISES:
            for length in LENGTHS:
                errors_batch = []
                errors_map = []
                sensors = []
                crb_total = []
                scores = []
                map_scores = []
                ranks = []
                converged = 0
                for index in range(SEEDS_PER_CELL):
                    seed = SEED_GRID + 1000 * index + length
                    generated = dataset.generate(
                        profile=profile,
                        samples=length,
                        noise_sigma=noise,
                        amplitude=0.05,
                        theta_max_rad=3.0e-3,
                        seed=seed,
                    )
                    verdict = observability.assess(generated.record)
                    sensor = verdict["secondary_1"]
                    sensors.append(sensor)
                    ranks.append(sensor.rank)
                    truth = generated.truth["secondary_1"]
                    with timer_batch:
                        solution = batch.solve(generated.record, verdict, "secondary_1")
                    converged += solution.converged
                    with timer_map:
                        regularised = batch.solve_map(
                            generated.record,
                            verdict,
                            "secondary_1",
                            prior_sigma_rad=PRIOR_SIGMA_RAD,
                        )
                    errors_batch.append(solution.theta_rad - truth)
                    errors_map.append(regularised.theta_rad - truth)
                    values = np.array(
                        [d.eigenvalue for d in sensor.degrees_of_freedom], dtype=float
                    )
                    mask = sensor.observable_mask
                    crb_total.append(
                        float(np.sqrt(np.sum(1.0 / values[mask]))) if mask.any() else np.inf
                    )
                    scores.extend(
                        float(z)
                        for z in benchmark.crb_z_scores(solution.theta_rad - truth, sensor)
                        if np.isfinite(z)
                    )
                    map_scores.extend(
                        float(z)
                        for z in benchmark.crb_z_scores(regularised.theta_rad - truth, sensor)
                        if np.isfinite(z)
                    )
                table = benchmark.score("batch", np.array(errors_batch), sensors)
                map_table = benchmark.score("map", np.array(errors_map), sensors)
                crb_rms = float(np.sqrt(np.mean(np.square(crb_total)))) * 1.0e3
                # The bound is on the NORM of the error over the retained
                # subspace, so it is compared against the root mean square of
                # that norm -- not against the per-component RMSE of the score
                # table, which divides by three components regardless of how
                # many were retained and is therefore smaller by sqrt(3 / rank).
                observable_norms = np.array(
                    [
                        float(np.linalg.norm(sensors[k].project_observable(errors_batch[k])))
                        for k in range(len(sensors))
                    ]
                )
                map_norms = np.array(
                    [
                        float(np.linalg.norm(sensors[k].project_observable(errors_map[k])))
                        for k in range(len(sensors))
                    ]
                )
                obs_norm_rms = float(np.sqrt(np.mean(np.square(observable_norms)))) * 1.0e3
                map_norm_rms = float(np.sqrt(np.mean(np.square(map_norms)))) * 1.0e3
                # Per-RECORD ratios, then the median. A ratio of two
                # root-mean-squares is the wrong statistic here: within one cell
                # the Cramer-Rao bound varies by orders of magnitude between
                # seeds, because the excitation shares of (9) do, so an RMS
                # ratio is decided by the single worst-conditioned seed and
                # ranged from 0.78 to 1.69 across cells while every per-
                # direction standardised score of (24) stayed inside 2.4 sigma.
                # The per-record ratio is well behaved and is what is reported.
                crb_array = np.array(crb_total)
                finite = np.isfinite(crb_array) & (crb_array > 0.0)
                per_record = observable_norms[finite] / crb_array[finite]
                per_record_map = map_norms[finite] / crb_array[finite]
                all_scores.extend(scores)
                all_map_scores.extend(map_scores)
                rows.append(
                    {
                        "profile": profile,
                        "noise": noise,
                        "length": length,
                        "rank": float(np.mean(ranks)),
                        "crb": crb_rms,
                        "obs": table.rmse_observable_mrad,
                        "unobs": table.rmse_unobservable_mrad,
                        "obs_norm": obs_norm_rms,
                        "rms_ratio": obs_norm_rms / crb_rms if crb_rms > 0 else np.inf,
                        "ratio": float(np.median(per_record)),
                        "map_obs": map_table.rmse_observable_mrad,
                        "map_norm": map_norm_rms,
                        "map_ratio": float(np.median(per_record_map)),
                        "crb_median": float(np.median(crb_array[finite])) * 1.0e3,
                        "z_std": float(np.std(scores)) if scores else np.nan,
                        "z_mean": float(np.mean(scores)) if scores else np.nan,
                        "z_max": float(np.max(np.abs(scores))) if scores else np.nan,
                        "z_n": len(scores),
                        "converged": converged,
                    }
                )

    out.rule("2.1 the grid")
    out(
        "All error figures are in milliradians. 'obs RMSE' is the root mean square "
        "over the cell of the NORM of the error inside the span of the floor-clearing "
        "degrees of freedom; 'CRB' is the MEDIAN over the cell of the total "
        "Cramer-Rao one sigma on that same subspace, sqrt(sum 1/lambda_i). 'obs/CRB' "
        "and 'MAP/CRB' are the MEDIAN of the PER-RECORD ratio of those two "
        "quantities, not a ratio of the two columns beside them: within one cell the "
        "bound varies by orders of magnitude between seeds because the excitation "
        "shares of (9) do, so a ratio of root-mean-squares is decided by the single "
        "worst-conditioned seed. For an efficient unbiased estimator the median "
        "per-record ratio is the median of a chi variate divided by sqrt of its "
        "degrees of freedom: 0.8326 on a rank-2 record and 0.8881 on a rank-3 one. "
        "'unobs RMSE' is the per-component root mean square outside the retained "
        "span. 'z std' is the standard deviation of the standardised score (24) over "
        "the cell.\n\n"
        "The per-record ratio is reported for information and is NOT the check. Its "
        "expected value depends on how the bound's mass is spread across directions: "
        "when all three contribute equally it is median(chi_3)/sqrt(3) = 0.8881, and "
        "when one ill-conditioned direction dominates the sum it falls to "
        "median(chi_1) = 0.6745. The near_single_axis cells are the second case. The "
        "check is on the standardised score of (24), one per retained degree of "
        "freedom, which is standard normal regardless of conditioning."
    )
    out("")
    out(
        f"{'profile':<17s} {'sigma':>8s} {'n':>5s} {'rank':>5s} {'CRB mrad':>10s} "
        f"{'obs RMSE':>10s} {'obs/CRB':>8s} {'MAP obs':>10s} {'MAP/CRB':>8s} "
        f"{'unobs RMSE':>11s} {'z std':>7s}"
    )
    for row in rows:
        out(
            f"{row['profile']:<17s} {row['noise']:>8.1e} {row['length']:>5d} "
            f"{row['rank']:>5.2f} {row['crb_median']:>10.4f} {row['obs_norm']:>10.4f} "
            f"{row['ratio']:>8.4f} {row['map_norm']:>10.4f} {row['map_ratio']:>8.4f} "
            f"{row['unobs']:>11.4f} {row['z_std']:>7.4f}"
        )

    out.rule("2.2 the unbiased batch solution against the bound")
    ratios = np.array([row["ratio"] for row in rows])
    scores = np.array(all_scores)
    out(f"  cells                             : {len(rows)}")
    out(f"  records                           : {len(rows) * SEEDS_PER_CELL}")
    out(f"  standardised scores, eq (24)      : {scores.size}")
    rms_ratios = np.array([row["rms_ratio"] for row in rows])
    out(f"  per-record |e_obs|/CRB, median over all cells : {float(np.median(ratios)):.6f}")
    out(f"  per-record |e_obs|/CRB, min and max cell      : {float(np.min(ratios)):.6f}, "
        f"{float(np.max(ratios)):.6f}")
    out("  predicted for an efficient estimator          : 0.8326 (rank 2), "
        "0.8881 (rank 3)")
    out(f"  RMS-of-norm / RMS-of-CRB, min and max cell    : "
        f"{float(np.min(rms_ratios)):.6f}, {float(np.max(rms_ratios)):.6f}  "
        f"<- the badly behaved statistic, shown for comparison")
    out(f"  z mean                            : {float(np.mean(scores)):+.6f}")
    out(f"  z standard deviation              : {float(np.std(scores)):.6f}")
    out(f"  |z| > 3 fraction                  : {float(np.mean(np.abs(scores) > 3.0)):.6f} "
        f"against a standard-normal 0.002700")
    out(f"  Gauss-Newton convergence          : "
        f"{sum(row['converged'] for row in rows)} of {len(rows) * SEEDS_PER_CELL}")
    out("")
    out(
        "The bound applies to UNBIASED estimators. A standard deviation of the "
        "standardised score materially below 1 would mean the batch solution is "
        "biased; materially above 1 would mean the bound is wrong or the estimator is "
        "inefficient. Both directions are checked."
    )
    out.check(
        "the standardised score of (24) has unit spread over the whole grid",
        0.85 < float(np.std(scores)) < 1.20,
        f"measured {float(np.std(scores)):.6f}, declared window [0.85, 1.20]",
    )
    out.check(
        "the standardised score of (24) is unbiased over the whole grid",
        abs(float(np.mean(scores))) < 0.1,
        f"measured {float(np.mean(scores)):+.6f}, declared bound 0.1",
    )
    cell_z_std = np.array([row["z_std"] for row in rows])
    cell_z_mean = np.array([row["z_mean"] for row in rows])
    out("")
    out(
        f"  per-cell z std, min and max       : {float(np.min(cell_z_std)):.6f}, "
        f"{float(np.max(cell_z_std)):.6f}"
    )
    out(
        f"  per-cell z mean, min and max      : {float(np.min(cell_z_mean)):+.6f}, "
        f"{float(np.max(cell_z_mean)):+.6f}"
    )
    out(
        f"  each cell holds about {rows[0]['z_n']} scores, so the standard error of a "
        f"cell mean is about {1.0 / np.sqrt(rows[0]['z_n']):.4f} and of a cell "
        f"standard deviation about {1.0 / np.sqrt(2 * rows[0]['z_n']):.4f}; the "
        "windows below are set from those."
    )
    out.check(
        "every cell's standardised score has a standard deviation in [0.5, 1.6]",
        bool(np.min(cell_z_std) > 0.5 and np.max(cell_z_std) < 1.6),
        f"cell range [{float(np.min(cell_z_std)):.6f}, {float(np.max(cell_z_std)):.6f}]",
    )
    out.check(
        "every cell's standardised score has a mean inside 4 standard errors of zero",
        bool(np.max(np.abs(cell_z_mean)) < 4.0 / np.sqrt(rows[0]["z_n"])),
        f"worst cell mean {float(cell_z_mean[np.argmax(np.abs(cell_z_mean))]):+.6f} "
        f"against a bound of {4.0 / np.sqrt(rows[0]['z_n']):.6f}",
    )
    out.check(
        "no standardised score anywhere in the grid exceeds 5 sigma",
        bool(np.max(np.abs(scores)) < 5.0),
        f"worst |z| {float(np.max(np.abs(scores))):.4f} over {scores.size} scores",
    )
    out.check(
        "the Gauss-Newton iteration converged on every record in the grid",
        sum(row["converged"] for row in rows) == len(rows) * SEEDS_PER_CELL,
        f"{sum(row['converged'] for row in rows)} of {len(rows) * SEEDS_PER_CELL}",
    )

    out.rule("2.3 the regularised MAP solution falls BELOW the bound, and why that is allowed")
    map_scores = np.array(all_map_scores)
    map_ratios = np.array([row["map_ratio"] for row in rows])
    below = int(np.sum(map_ratios < 1.0))
    out(f"  MAP obs RMSE / CRB, median        : {float(np.median(map_ratios)):.6f}")
    out(f"  cells below the bound             : {below} of {len(rows)}")
    out(f"  MAP standardised z, mean          : {float(np.mean(map_scores)):+.6f}")
    out(f"  MAP standardised z, std           : {float(np.std(map_scores)):.6f}")
    out("")
    out(
        "The Cramer-Rao bound of (12) constrains unbiased estimators only. The MAP "
        "estimator of (18) shrinks toward zero and is therefore biased, so a ratio "
        "below 1 is not a violation of anything. The price is the bias, measured "
        "below per profile as the mean projection of the error onto the truth "
        "direction, which is negative when the estimate is systematically short."
    )
    out("")
    out(f"{'profile':<17s} {'n':>5s} {'batch bias':>12s} {'MAP bias':>12s} {'truth RMS':>11s}")
    for profile in PROFILES:
        for length in LENGTHS:
            batch_projection = []
            map_projection = []
            magnitudes = []
            for index in range(SEEDS_PER_CELL):
                seed = SEED_GRID + 1000 * index + length
                generated = dataset.generate(
                    profile=profile,
                    samples=length,
                    noise_sigma=2.0e-4,
                    amplitude=0.05,
                    theta_max_rad=3.0e-3,
                    seed=seed,
                )
                verdict = observability.assess(generated.record)
                sensor = verdict["secondary_1"]
                truth = generated.truth["secondary_1"]
                observable_truth = sensor.project_observable(truth)
                magnitude = float(np.linalg.norm(observable_truth))
                if magnitude < 1.0e-9:
                    continue
                unit = observable_truth / magnitude
                magnitudes.append(magnitude)
                batch_projection.append(
                    float(
                        np.dot(
                            batch.solve(generated.record, verdict, "secondary_1").theta_rad
                            - truth,
                            unit,
                        )
                    )
                )
                map_projection.append(
                    float(
                        np.dot(
                            batch.solve_map(
                                generated.record,
                                verdict,
                                "secondary_1",
                                prior_sigma_rad=PRIOR_SIGMA_RAD,
                            ).theta_rad
                            - truth,
                            unit,
                        )
                    )
                )
            out(
                f"{profile:<17s} {length:>5d} "
                f"{float(np.mean(batch_projection)) * 1e3:>12.6f} "
                f"{float(np.mean(map_projection)) * 1e3:>12.6f} "
                f"{float(np.sqrt(np.mean(np.square(magnitudes)))) * 1e3:>11.6f}"
            )
    out("")
    out("  (bias figures in mrad, projected onto the observable part of the truth)")

    out.rule("2.4 the unobservable component is carried in full by every estimator")
    out(
        "The 'unobs RMSE' column of 2.1 is identical in structure for every estimator "
        "in this repository, because none of them can recover a component the record "
        "does not contain. For the single-axis profile it is about the per-component "
        "spread of the injected population, as it must be."
    )
    out("")
    out(f"{'profile':<17s} {'unobs RMSE mrad':>17s} {'mean rank':>10s}")
    for profile in PROFILES:
        subset = [row for row in rows if row["profile"] == profile]
        out(
            f"{profile:<17s} "
            f"{float(np.sqrt(np.mean([r['unobs'] ** 2 for r in subset]))):>17.6f} "
            f"{float(np.mean([r['rank'] for r in subset])):>10.4f}"
        )
    single = [row for row in rows if row["profile"] == "single_axis"]
    expected = 1.0  # per-component standard deviation of the injected population, mrad
    measured = float(np.sqrt(np.mean([r["unobs"] ** 2 for r in single])))
    out("")
    out.check(
        "the single-axis unobservable error matches the injected population spread",
        0.5 * expected < measured < 1.5 * expected,
        f"measured {measured:.6f} mrad against a predicted {expected:.6f} mrad "
        "(one component of a population with magnitude uniform on [0, 3] mrad)",
    )

    out.rule("2.5 timings -- UNSTABLE, two shared cores, never quoted as a ratio in prose")
    out(f"{'operation':<34s} {'calls':>7s} {'total s':>10s} {'mean ms':>10s}")
    out(f"{'batch.solve, eq (14)-(16)':<34s} {timer_batch.calls:>7d} "
        f"{timer_batch.elapsed_s:>10.4f} {timer_batch.mean_ms:>10.4f}")
    out(f"{'batch.solve_map, eq (18)':<34s} {timer_map.calls:>7d} "
        f"{timer_map.elapsed_s:>10.4f} {timer_map.mean_ms:>10.4f}")
    out("")
    out("  These figures move between runs on a shared two-core container.")

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
