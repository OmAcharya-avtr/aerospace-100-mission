"""The gap policy: the bound as a theorem, the coverage as a measurement, and two gates.

Six measurements, in order.

**1. The curvature table.** For each channel: the analytic bound of equation
(12), the Bernstein bound of equation (8), what the policy acts on, the noise
floor of equation (9), the measurability and resolvability verdicts, and the
ratio of the estimate to the analytic truth. The last column is where the
headline finding lives.

**2. Equation (3) as a theorem.** Linear interpolation across a gap of ``k``
cadences is evaluated against the **noiseless** truth, so that the only error
present is the interpolation error. The measured maximum must not exceed
``M2 h^2 / 8`` with ``M2`` the analytic bound of equation (12), and is checked
against the Bernstein bound of equation (8) as well -- a bound derived from
complex analysis rather than from a Taylor remainder, so it shares no
mathematics with the quantity it is checking. The slack follows the
conditioning of the two expressions and is not an absolute constant: it is
``eps * (4 * bound + amplitude)``, four roundings in evaluating the bound plus
the cancellation of a difference of two numbers of order the signal amplitude.

**3. Equation (3') as a coverage statement, and the rate at which it fails.**
The policy is run on real records with real noisy endpoints, over several
seeds, and the realised error is compared against what the policy predicted.
This is a coverage statement, so the exceedance count is a measurement and not
a pass criterion with a tuned tolerance -- and it is **not zero**. Equations
(3) and (4) bound an interpolation anchored on correct endpoints, and the
endpoint gate can only refuse an anomaly the classification *detected*. The
measured corrupt recall of the strengthened rule baseline is 0.4052
(``validate_classification_output.txt`` §1), so a bit error that stays inside
its declared range and passes the frame error-control check is invisible to
every screen and can sit at a gap endpoint. The check below therefore asserts
the **mechanism**: every exceedance must trace to a gap endpoint whose
ground-truth class is not clean and which no screen flagged. The exceedance
rate itself is reported as a measured limitation.

**4. What the resolvability gate is worth.** The policy is run twice, with the
gate and without it, and the realised error on the near-Nyquist channel is
measured both ways. Without the gate the policy permits an interpolation whose
error exceeds the declared tolerance; with it, it refuses.

**5. What the endpoint gate is worth.** The same comparison for the gate that
refuses a gap bounded by a frame with an unresolved integrity verdict.

**6. The noise-scale estimator's confounding.** Equation (16) is compared
against the generator's true noise sigma per channel. On the channel with
near-Nyquist content the estimator reads the signal as noise, and by how much
is measured.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, default_tolerances, environment_lines

ROOT = bootstrap()

from telemgap import classify, dataset, interpolate, policy  # noqa: E402
from telemgap.report import format_table  # noqa: E402

EPS = float(np.finfo(np.float64).eps)

#: Gap lengths swept in the theorem check, in cadence intervals.
GAP_SLOTS = (1, 2, 3, 4, 6, 8, 12, 16)

#: Positions sampled per gap length in the theorem check.
POSITIONS = 400

#: Seeds swept in the coverage measurement.
COVERAGE_SEEDS = (909, 515, 77, 4242, 1, 2, 31337, 20261028)

#: Frames per record in the coverage measurement.
COVERAGE_SAMPLES = 2500


def bound_slack(bound: float, amplitude: float) -> float:
    """Absolute slack following the conditioning of the two expressions compared.

    Four roundings in evaluating ``M2 h^2 / 8`` give ``4 eps * bound``; the
    measured error is a difference of two numbers of order ``amplitude``, whose
    cancellation contributes ``eps * amplitude``. Neither term is a constant
    and neither is a tuned tolerance.
    """
    return EPS * (4.0 * abs(bound) + abs(amplitude))


def main() -> int:
    out = Tee("validate_interpolation_policy_output.txt")
    out.rule("telemgap -- the gap policy: bound, coverage, and two gates")
    for line in environment_lines():
        out(line)
    tolerances = default_tolerances()
    out("")
    out(f"declared tolerances : {tolerances}")
    out(f"coverage factor k   : {policy.DEFAULT_COVERAGE_K:g}")
    out(f"max gap slots       : {policy.DEFAULT_MAX_GAP_SLOTS:g}")
    out(f"estimator half window: {policy.DEFAULT_HALF_WINDOW}")
    out(
        f"estimator bandwidth : "
        f"{policy.estimator_bandwidth_fraction(policy.DEFAULT_HALF_WINDOW):.6f} of the "
        "sampling frequency (half-power point, computed)"
    )

    generated = dataset.generate(samples=12_000, seed=dataset.TEST_SEED, n_clock_resets=0)
    record = generated.record
    estimates = policy.channel_curvatures(record)

    out.rule("1. curvature per channel: analytic, Bernstein, declared, estimated")
    rows = []
    for model, channel in zip(dataset.MODELS, record.schema.channels, strict=True):
        estimate = estimates[channel.name]
        bernstein = policy.bernstein_curvature(model.amplitude_bound, model.bandwidth_hz)
        bare = policy.estimate_curvature(
            record.values[:, record.schema.channel_index(channel.name)],
            record.schema.cadence_s,
            channel.span,
            declared_sigma=channel.noise_sigma,
        )
        ratio = bare.value / model.curvature_bound if model.curvature_bound else float("nan")
        rows.append(
            [
                channel.name,
                f"{model.curvature_bound:.6g}",
                f"{bernstein:.6g}",
                f"{channel.max_curvature:.6g}" if channel.max_curvature else "not declared",
                f"{bare.value:.6g}",
                f"{bare.noise_floor:.6g}",
                "yes" if bare.measurable else "NO",
                f"{bare.high_frequency_fraction:.6f}",
                "yes" if bare.resolvable else "NO",
                f"{ratio:.6f}",
                estimate.source,
            ]
        )
    out(
        format_table(
            [
                "channel",
                "analytic eq12",
                "bernstein eq8",
                "declared",
                "estimated eq9",
                "noise floor",
                "measurable",
                "hf fraction",
                "resolvable",
                "est/analytic",
                "policy uses",
            ],
            rows,
        )
    )
    out("")
    fast = dataset.MODELS[2]
    transfer = policy.estimator_transfer(policy.DEFAULT_HALF_WINDOW, fast.terms[1][1])
    out(
        f"The gyro_rate channel carries a term at {fast.terms[1][1]:g} of the sampling "
        f"frequency. The closed-form transfer of the equation (9) estimator there is "
        f"{transfer:.6g}, so the estimator cannot see that term's curvature, and the "
        "measured est/analytic ratio above is the consequence. This is a bias, not noise: "
        "the estimate sits above its own noise floor and the measurability gate passes it."
    )
    out.check(
        "the equation (9) estimate of the near-Nyquist channel is below one fiftieth of "
        "its analytic curvature",
        estimates["gyro_rate"].value < dataset.MODELS[2].curvature_bound / 50.0,
        f"{estimates['gyro_rate'].value:.6g} against {dataset.MODELS[2].curvature_bound:.6g}",
    )
    out.check(
        "the resolvability gate catches the near-Nyquist channel that the measurability "
        "gate passes",
        estimates["gyro_rate"].measurable and not estimates["gyro_rate"].resolvable,
        f"measurable {estimates['gyro_rate'].measurable}, "
        f"resolvable {estimates['gyro_rate'].resolvable}",
    )
    out.check(
        "the Bernstein bound of equation (8) dominates the analytic bound of equation (12) "
        "on every single-term channel",
        all(
            policy.bernstein_curvature(model.amplitude_bound, model.bandwidth_hz)
            >= model.curvature_bound * (1.0 - 1e-12)
            for model in dataset.MODELS
            if len(model.terms) == 1
        ),
    )

    out.rule("2. equation (3) as a theorem, against the noiseless truth")
    grid = dataset.CADENCE_S * np.arange(8000, dtype=np.float64)
    rows = []
    violations = 0
    comparisons = 0
    for model in dataset.MODELS:
        truth = model.evaluate(grid)
        analytic = model.curvature_bound
        bernstein = policy.bernstein_curvature(model.amplitude_bound, model.bandwidth_hz)
        for slots in GAP_SLOTS:
            duration = (slots + 1) * dataset.CADENCE_S
            bound = policy.linear_error_bound(analytic, duration)
            bound_bernstein = policy.linear_error_bound(bernstein, duration)
            starts = np.linspace(0, grid.size - slots - 3, POSITIONS).astype(np.int64)
            worst = 0.0
            for start in starts:
                stop = start + slots + 1
                fraction = (grid[start : stop + 1] - grid[start]) / (grid[stop] - grid[start])
                chord = truth[start] + fraction * (truth[stop] - truth[start])
                worst = max(worst, float(np.max(np.abs(chord - truth[start : stop + 1]))))
                comparisons += 1
            slack = bound_slack(bound, model.amplitude_bound)
            ok = worst <= bound + slack
            if not ok:
                violations += 1
            rows.append(
                [
                    model.name,
                    str(slots),
                    f"{duration:.1f}",
                    f"{worst:.6g}",
                    f"{bound:.6g}",
                    f"{worst / bound:.6f}" if bound > 0 else "inf",
                    f"{slack:.3g}",
                    "ok" if ok else "VIOLATED",
                    "ok" if worst <= bound_bernstein + bound_slack(
                        bound_bernstein, model.amplitude_bound
                    ) else "VIOLATED",
                ]
            )
    out(
        format_table(
            [
                "channel",
                "slots",
                "h (s)",
                "measured max |err|",
                "eq3 bound",
                "ratio",
                "slack",
                "vs eq12",
                "vs eq8",
            ],
            rows,
        )
    )
    out("")
    out(f"individual chord-versus-truth comparisons: {comparisons}")
    out.check(
        "equation (3) holds on every swept gap length and channel against the analytic "
        "curvature of equation (12)",
        violations == 0,
        f"{violations} violations in {len(rows)} cells, {comparisons} comparisons",
    )
    out.check(
        "equation (3) holds against the independently derived Bernstein bound of "
        "equation (8)",
        all(row[8] == "ok" for row in rows),
        "the Bernstein bound rests on complex analysis, not on a Taylor remainder",
    )

    out.rule("3. equation (3') as a coverage statement, on real noisy records")
    rows = []
    exceedances = 0
    explained = 0
    compared = 0
    diagnoses: list[str] = []
    for seed in COVERAGE_SEEDS:
        sample = dataset.generate(samples=COVERAGE_SAMPLES, seed=seed, n_clock_resets=0)
        report = classify.classify(sample.record)
        plan = policy.plan(sample.record, report, tolerances)
        built = interpolate.reconstruct(sample.record, report.findings.gaps, plan)
        flagged = policy.endpoint_verdicts(
            report, len(sample.record), sample.record.n_channels
        )
        for column, channel in enumerate(sample.record.schema.channels):
            error, count = interpolate.measured_error(
                built, sample.truth_times, sample.truth_values[:, column], column
            )
            permitted = [
                decision
                for decision in plan.decisions
                if decision.channel == column and not decision.refused
            ]
            if count == 0 or not permitted:
                continue
            worst = max(decision.predicted_error for decision in permitted)
            compared += 1
            ok = error <= worst
            if not ok:
                exceedances += 1
                # Diagnose: is some permitted gap bounded by a frame whose
                # ground-truth class is not clean and which no screen flagged?
                culprits = []
                for decision in permitted:
                    gap = report.findings.gaps[decision.gap_index]
                    for endpoint in (gap.after_index, gap.before_index):
                        truth_class = int(sample.value_truth[endpoint, column])
                        if truth_class != 0 and (endpoint, column) not in flagged:
                            culprits.append((endpoint, truth_class))
                if culprits:
                    explained += 1
                    endpoint, truth_class = culprits[0]
                    diagnoses.append(
                        f"seed {seed} {channel.name}: frame {endpoint} is truly "
                        f"{('corrupt', 'misparsed')[truth_class - 1]} at this channel, no "
                        "screen flagged it, and it anchors a permitted gap. In-range "
                        f"corruption: {bool(sample.corruption_in_range[endpoint, column])}; "
                        f"frame error control passed: "
                        f"{bool(sample.record.validity[endpoint])}"
                    )
                else:
                    diagnoses.append(
                        f"seed {seed} {channel.name}: EXCEEDED with no undetected endpoint "
                        "anomaly found; this exceedance is NOT explained by the known "
                        "mechanism and is a defect"
                    )
            rows.append(
                [
                    str(seed),
                    channel.name,
                    str(count),
                    f"{error:.6g}",
                    f"{worst:.6g}",
                    f"{tolerances[channel.name]:g}",
                    f"{error / worst:.4f}",
                    "ok" if ok else "EXCEEDED",
                ]
            )
    out(
        format_table(
            [
                "seed",
                "channel",
                "slots",
                "measured max |err|",
                "worst predicted",
                "tolerance",
                "ratio",
                "verdict",
            ],
            rows,
        )
    )
    out("")
    out(f"channel-records compared : {compared}")
    out(f"exceedances              : {exceedances}")
    out(
        f"measured exceedance rate : "
        f"{exceedances / compared:.6f}" if compared else "measured exceedance rate : n/a"
    )
    for line in diagnoses:
        out(f"  {line}")
    out("")
    out(
        "This is the product's main unfavourable result and it is published as measured. "
        "The policy's prediction is sound; the classification feeding its endpoint gate "
        "is not complete. The two numbers multiply: a gap endpoint is anomalous with "
        "roughly the record's anomaly rate, and the strengthened rule baseline misses "
        "about three fifths of corruptions because they stay inside the declared range "
        "and pass the frame error-control check. Declaring a tighter range, or supplying "
        "per-word rather than per-frame error control, is what removes it; nothing in "
        "this package can."
    )
    out.check(
        "every exceedance of equation (3') traces to a gap endpoint whose ground-truth "
        "class is not clean and which no screen flagged",
        explained == exceedances,
        f"{explained} of {exceedances} exceedances explained by the known mechanism; "
        f"{exceedances} of {compared} channel-records exceeded",
    )

    out.rule("4. what the curvature resolvability gate is worth")
    rows = []
    without_exceed = 0
    for seed in COVERAGE_SEEDS:
        sample = dataset.generate(samples=COVERAGE_SAMPLES, seed=seed, n_clock_resets=0)
        report = classify.classify(sample.record)
        column = sample.record.schema.channel_index("gyro_rate")
        for label, threshold in (("gate on", policy.DEFAULT_RESOLVABILITY_THRESHOLD),
                                 ("gate off", None)):
            plan = policy.plan(
                sample.record, report, tolerances, resolvability_threshold=threshold
            )
            built = interpolate.reconstruct(sample.record, report.findings.gaps, plan)
            error, count = interpolate.measured_error(
                built, sample.truth_times, sample.truth_values[:, column], column
            )
            permitted = [
                decision.predicted_error
                for decision in plan.decisions
                if decision.channel == column and not decision.refused
            ]
            worst = max(permitted) if permitted else float("nan")
            exceeded = bool(count and np.isfinite(worst) and error > worst)
            if label == "gate off" and exceeded:
                without_exceed += 1
            rows.append(
                [
                    str(seed),
                    label,
                    str(len(permitted)),
                    str(count),
                    f"{error:.6g}" if count else "-",
                    f"{worst:.6g}" if np.isfinite(worst) else "-",
                    f"{tolerances['gyro_rate']:g}",
                    "EXCEEDED" if exceeded else "ok",
                ]
            )
    out(
        format_table(
            [
                "seed",
                "configuration",
                "permitted gaps",
                "slots",
                "measured max |err|",
                "worst predicted",
                "tolerance",
                "verdict",
            ],
            rows,
        )
    )
    out("")
    out.check(
        "with the resolvability gate disabled, the near-Nyquist channel exceeds its "
        "prediction on at least one seed",
        without_exceed > 0,
        f"{without_exceed} of {len(COVERAGE_SEEDS)} seeds exceeded with the gate off",
    )
    out.check(
        "with the gate enabled, the near-Nyquist channel is refused on every seed",
        all(row[2] == "0" for row in rows if row[1] == "gate on"),
    )

    out.rule("5. what the endpoint integrity gate is worth")
    import dataclasses

    rows = []
    without_endpoint_exceed = 0
    for seed in COVERAGE_SEEDS:
        sample = dataset.generate(samples=COVERAGE_SAMPLES, seed=seed, n_clock_resets=0)
        report = classify.classify(sample.record)
        blind = dataclasses.replace(report, anomalies=())
        for label, used in (("gate on", report), ("gate off", blind)):
            plan = policy.plan(sample.record, used, tolerances)
            built = interpolate.reconstruct(sample.record, report.findings.gaps, plan)
            worst_ratio = 0.0
            worst_channel = "-"
            exceeded = False
            for column, channel in enumerate(sample.record.schema.channels):
                error, count = interpolate.measured_error(
                    built, sample.truth_times, sample.truth_values[:, column], column
                )
                permitted = [
                    decision.predicted_error
                    for decision in plan.decisions
                    if decision.channel == column and not decision.refused
                ]
                if count == 0 or not permitted:
                    continue
                ratio = error / max(permitted)
                if ratio > worst_ratio:
                    worst_ratio, worst_channel = ratio, channel.name
                if ratio > 1.0:
                    exceeded = True
            if label == "gate off" and exceeded:
                without_endpoint_exceed += 1
            rows.append(
                [
                    str(seed),
                    label,
                    str(len(plan.permissions)),
                    str(plan.summary()["failed_endpoints_clean"]),
                    worst_channel,
                    f"{worst_ratio:.4f}",
                    "EXCEEDED" if exceeded else "ok",
                ]
            )
    out(
        format_table(
            [
                "seed",
                "configuration",
                "permitted",
                "gaps failing the gate",
                "worst channel",
                "worst measured/predicted",
                "verdict",
            ],
            rows,
        )
    )
    out("")
    out(
        "The 'gate off' rows are produced by handing the policy an integrity report with "
        "its anomaly list emptied, which is exactly what a caller does who treats the "
        "integrity report as documentation rather than as an input."
    )
    out.check(
        "with the endpoint gate blinded, at least one seed exceeds its prediction",
        without_endpoint_exceed > 0,
        f"{without_endpoint_exceed} of {len(COVERAGE_SEEDS)} seeds",
    )

    out.rule("6. the noise-scale estimator of equation (16) against the true sigma")
    rows = []
    for model, channel in zip(dataset.MODELS, record.schema.channels, strict=True):
        column = record.schema.channel_index(channel.name)
        estimated = policy.estimate_noise_sigma(record.values[:, column])
        rows.append(
            [
                channel.name,
                f"{model.noise_sigma:.6g}",
                f"{estimated:.6g}",
                f"{estimated / model.noise_sigma:.4f}",
                f"{model.bandwidth_hz:.6g}",
            ]
        )
    out(
        format_table(
            ["channel", "true sigma", "eq16 estimate", "ratio", "top frequency Hz"],
            rows,
        )
    )
    out("")
    out(
        "Equation (16) reads a second difference as noise. A sinusoid near the sampling "
        "frequency produces large second differences, so on gyro_rate the estimator reads "
        "signal as noise by the ratio above. This is why the policy prefers a declared "
        "noise_sigma, and why the three channels without near-Nyquist content are "
        "recovered to within a few per cent."
    )
    out.check(
        "equation (16) recovers the true sigma within 15 per cent on every channel with "
        "no near-Nyquist content",
        all(
            abs(policy.estimate_noise_sigma(record.values[:, record.schema.channel_index(c.name)])
                / m.noise_sigma - 1.0) < 0.15
            for m, c in zip(dataset.MODELS, record.schema.channels, strict=True)
            if m.bandwidth_hz < 0.1
        ),
    )
    out.check(
        "equation (16) overstates the noise of the near-Nyquist channel by more than a "
        "factor of ten",
        policy.estimate_noise_sigma(
            record.values[:, record.schema.channel_index("gyro_rate")]
        )
        > 10.0 * dataset.MODELS[2].noise_sigma,
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
