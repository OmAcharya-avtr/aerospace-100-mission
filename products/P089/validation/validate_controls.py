"""Section 6: controls. Every one anchored on a truth, and every one verified to fire.

A control that does not fire is worse than no control, because it reads as
evidence. Each control below declares in advance whether it MUST fire or MUST
NOT fire, and both directions are asserted. None of them is anchored on an
estimator's output: each is built from a chosen truth, a declared
transformation, or an algebraic invariance.

The controls are grouped:

* 6.1 invariances the estimator must have, which MUST NOT fire;
* 6.2 injections and corruptions, which MUST fire, by a predicted amount where
  one can be predicted;
* 6.3 what removing the observability verdict costs, which MUST fire;
* 6.4 an audit of the test suite's own floating-point tolerances.
"""

from __future__ import annotations

from _bootstrap import (
    ROOT,
    SEED_CONTROL,
    Tee,
    axis_aligned_record,
    bootstrap,
    environment_lines,
)

bootstrap()

import numpy as np  # noqa: E402

from aligncal import batch, dataset, observability, recursive  # noqa: E402
from aligncal.geometry import rotation_from_vector  # noqa: E402
from aligncal.record import AlignmentRecord, SensorSpec  # noqa: E402

SAMPLES = 1200


def _base(seed: int, profile: str = "three_axis", **kwargs):
    return dataset.generate(profile=profile, samples=SAMPLES, seed=seed, **kwargs)


def _estimate(generated, floor_factor=observability.DEFAULT_FLOOR_FACTOR):
    verdict = observability.assess(generated.record, floor_factor=floor_factor)
    return verdict, batch.solve(generated.record, verdict, "secondary_1")


def main() -> int:
    out = Tee("validate_controls_output.txt")
    out("VALIDATION 6 -- CONTROLS")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out(f"  base record: {SAMPLES} samples, three_axis excitation, 0.05 rad/s, "
        "2e-4 rad/s noise unless stated")

    fired: list[tuple[str, bool, bool]] = []

    out.rule("6.1 invariances. These MUST NOT fire.")

    # --- C1: zero injected misalignment ---------------------------------
    out("")
    out("C1  zero injected misalignment: the estimate must be inside its own bound.")
    worst_z = 0.0
    for index in range(20):
        generated = _base(SEED_CONTROL + index, theta_rad=np.zeros(3))
        verdict, solution = _estimate(generated)
        for dof in verdict["secondary_1"].degrees_of_freedom:
            worst_z = max(
                worst_z,
                abs(float(np.dot(solution.theta_rad, dof.direction))) / dof.crb_sigma_rad,
            )
    passed = out.check(
        "C1 must NOT fire: a zero misalignment is estimated as zero to within 4 sigma",
        worst_z < 4.0,
        f"worst standardised estimate {worst_z:.4f} over 20 records and 60 degrees of "
        "freedom",
    )
    fired.append(("C1 zero injection", False, not passed))

    # --- C2: time order ------------------------------------------------
    out("")
    out(
        "C2  time order: the batch solution of a time-invariant state is a sum over "
        "samples, so permuting the record must not change it. The permutation keeps "
        "the timestamps increasing and permutes the rate rows."
    )
    generated = _base(SEED_CONTROL + 100)
    rng = np.random.default_rng(11)
    order = rng.permutation(len(generated.record))
    shuffled = AlignmentRecord(
        times=generated.record.times,
        reference=generated.record.reference,
        reference_rates=generated.record.reference_rates[order],
        secondaries=generated.record.secondaries,
        secondary_rates={
            name: block[order] for name, block in generated.record.secondary_rates.items()
        },
        provenance="controls C2 permuted",
    )
    verdict, solution = _estimate(generated)
    shuffled_verdict = observability.assess(shuffled)
    shuffled_solution = batch.solve(shuffled, shuffled_verdict, "secondary_1")
    difference = float(np.linalg.norm(shuffled_solution.theta_rad - solution.theta_rad))
    crb = verdict["secondary_1"].degrees_of_freedom[0].crb_sigma_rad
    out(f"    difference {difference * 206264.80624709636:.6e} arcsec, "
        f"{difference / crb:.3e} of the Cramer-Rao one sigma")
    passed = out.check(
        "C2 must NOT fire: the batch estimate is invariant under permuting the record",
        difference < 1.0e-6 * crb,
        f"{difference / crb:.3e} of the bound, against a declared 1e-6",
    )
    fired.append(("C2 time permutation", False, not passed))

    # --- C3: rate sign --------------------------------------------------
    out("")
    out(
        "C3  rate sign: negating every rate leaves the Gramian of (8) unchanged -- it "
        "is quadratic -- and negates the residual of (7), so the estimate must be "
        "unchanged."
    )
    negated = AlignmentRecord(
        times=generated.record.times,
        reference=generated.record.reference,
        reference_rates=-generated.record.reference_rates,
        secondaries=generated.record.secondaries,
        secondary_rates={
            name: -block for name, block in generated.record.secondary_rates.items()
        },
        provenance="controls C3 negated rates",
    )
    negated_verdict = observability.assess(negated)
    negated_solution = batch.solve(negated, negated_verdict, "secondary_1")
    difference = float(np.linalg.norm(negated_solution.theta_rad - solution.theta_rad))
    out(f"    difference {difference / crb:.3e} of the Cramer-Rao one sigma")
    passed = out.check(
        "C3 must NOT fire: the estimate is invariant under negating every rate",
        difference < 1.0e-6 * crb,
        f"{difference / crb:.3e} of the bound",
    )
    fired.append(("C3 rate sign", False, not passed))

    # --- C4: frame equivariance ----------------------------------------
    out("")
    out(
        "C4  frame equivariance: rotating the reference frame by Q must rotate the "
        "estimate by Q. The record is rebuilt with reference rates Q w, the secondary "
        "rates untouched, and the declared mounting C Q^T, which is exactly the same "
        "physical situation described in a rotated frame."
    )
    rotation = rotation_from_vector(np.array([0.7, -1.1, 0.4]))
    name = "secondary_1"
    rotated = AlignmentRecord(
        times=generated.record.times,
        reference=generated.record.reference,
        reference_rates=generated.record.reference_rates @ rotation.T,
        secondaries={
            name: SensorSpec(
                name,
                generated.record.secondaries[name].noise_sigma,
                generated.record.secondaries[name].mounting @ rotation.T,
            )
        },
        secondary_rates={name: generated.record.secondary_rates[name]},
        provenance="controls C4 rotated frame",
    )
    rotated_verdict = observability.assess(rotated)
    rotated_solution = batch.solve(rotated, rotated_verdict, name)
    expected = rotation @ solution.theta_rad
    difference = float(np.linalg.norm(rotated_solution.theta_rad - expected))
    out(f"    estimate in the rotated frame vs Q times the original: "
        f"{difference / crb:.3e} of the Cramer-Rao one sigma")
    passed = out.check(
        "C4 must NOT fire: the estimate is equivariant under a frame rotation",
        difference < 1.0e-4 * crb,
        f"{difference / crb:.3e} of the bound, against a declared 1e-4",
    )
    fired.append(("C4 frame equivariance", False, not passed))

    out("")
    out(
        "C5  declared noise: scaling every declared sigma by 10 scales the Gramian by "
        "1/100 and therefore the Cramer-Rao one sigma by 10, and must leave the "
        "estimate itself unchanged, because the weight is common to both sides of the "
        "normal equations."
    )
    loud = AlignmentRecord(
        times=generated.record.times,
        reference=SensorSpec("reference", generated.record.reference.noise_sigma * 10.0),
        reference_rates=generated.record.reference_rates,
        secondaries={
            name: SensorSpec(
                name,
                generated.record.secondaries[name].noise_sigma * 10.0,
                generated.record.secondaries[name].mounting,
            )
        },
        secondary_rates={name: generated.record.secondary_rates[name]},
        provenance="controls C5 ten times the declared noise",
    )
    loud_verdict = observability.assess(loud)
    loud_solution = batch.solve(loud, loud_verdict, name)
    estimate_difference = float(np.linalg.norm(loud_solution.theta_rad - solution.theta_rad))
    ratio = (
        loud_verdict[name].degrees_of_freedom[0].crb_sigma_rad
        / verdict[name].degrees_of_freedom[0].crb_sigma_rad
    )
    out(f"    estimate difference {estimate_difference / crb:.3e} of the original bound")
    out(f"    bound ratio {ratio:.6f}, predicted 10.000000")
    passed = out.check(
        "C5 must NOT fire on the estimate: it is invariant to a common noise rescaling",
        estimate_difference < 1.0e-6 * crb,
        f"{estimate_difference / crb:.3e} of the bound",
    )
    fired.append(("C5a estimate under noise rescaling", False, not passed))
    passed = out.check(
        "C5 MUST fire on the bound: the Cramer-Rao one sigma scales by exactly 10",
        abs(ratio - 10.0) < 1.0e-9,
        f"measured ratio {ratio:.9f}",
    )
    fired.append(("C5b bound under noise rescaling", True, passed))

    out.rule("6.2 injections and corruptions. These MUST fire, by a predicted amount.")

    out("")
    out(
        "C6  a known misalignment injected into the observable subspace must be "
        "recovered. The injection is placed along the BEST-observed direction of the "
        "record, which is anchored on the Gramian and not on any estimate."
    )
    recovered = []
    standardised = []
    for index in range(20):
        probe = _base(SEED_CONTROL + 200 + index)
        sensor = observability.assess(probe.record)["secondary_1"]
        direction = sensor.degrees_of_freedom[0].direction
        injected = 2.0e-3 * direction
        generated_c6 = _base(SEED_CONTROL + 200 + index, theta_rad=injected)
        verdict_c6, solution_c6 = _estimate(generated_c6)
        estimated = float(np.dot(solution_c6.theta_rad, direction))
        truth = float(np.dot(injected, direction))
        recovered.append(estimated / truth)
        standardised.append(
            (estimated - truth) / verdict_c6["secondary_1"].degrees_of_freedom[0].crb_sigma_rad
        )
    recovered = np.array(recovered)
    standardised = np.array(standardised)
    out(f"    recovered fraction of the injection: median {float(np.median(recovered)):.6f}, "
        f"min {float(np.min(recovered)):.6f}, max {float(np.max(recovered)):.6f}")
    out(
        "    The tolerance below is in units of the Cramer-Rao one sigma and not as a "
        "percentage of the injection. At 1200 samples the bound on this direction is "
        "about 0.23 mrad against a 2 mrad injection, so a 1.3-sigma miss looks like a "
        "16 per cent error and is the expected noise; a 5 per cent tolerance would be "
        "a tolerance on the noise rather than on the estimator."
    )
    out(f"    standardised recovery error: mean {float(np.mean(standardised)):+.4f}, "
        f"worst |z| {float(np.max(np.abs(standardised))):.4f}")
    passed = out.check(
        "C6 MUST fire: an injection in the observable subspace is recovered to within 4 sigma",
        bool(np.max(np.abs(standardised)) < 4.0),
        f"worst standardised error {float(np.max(np.abs(standardised))):.4f} over 20 records, "
        f"median recovered fraction {float(np.median(recovered)):.6f}",
    )
    fired.append(("C6 observable injection recovered", True, passed))

    out("")
    out(
        "C7  the same magnitude injected along the record's own rotation axis must NOT "
        "be recovered -- it must be withheld, exactly, with an infinite bound. The "
        "control is anchored on the chosen truth and the record is verified rank "
        "deficient first."
    )
    withheld = 0
    claims = []
    for index in range(20):
        generated_c7 = axis_aligned_record(
            seed=SEED_CONTROL + 300 + index, samples=SAMPLES, magnitude_rad=2.0e-3
        )
        verdict_c7, solution_c7 = _estimate(generated_c7)
        sensor = verdict_c7["secondary_1"]
        if not sensor.rank_deficient:
            continue
        axis = sensor.degrees_of_freedom[-1].direction
        withheld += 2 in solution_c7.withheld
        claims.append(abs(float(np.dot(solution_c7.theta_rad, axis))))
    out(f"    rank-deficient records: {len(claims)} of 20; withheld on {withheld}")
    out(f"    largest absolute claim along the axis: {float(np.max(claims)):.3e} rad")
    passed = out.check(
        "C7 MUST fire: the unsupported direction is withheld on every rank-deficient record",
        withheld == len(claims) and len(claims) == 20 and float(np.max(claims)) < 1.0e-15,
        f"{withheld} of {len(claims)} withheld, largest claim {float(np.max(claims)):.3e} rad",
    )
    fired.append(("C7 unobservable injection withheld", True, passed))

    out("")
    out(
        "C8  a wrong declared mounting must be absorbed into the estimate. A known "
        "extra rotation of 5 mrad is applied to the DECLARED mounting only, leaving "
        "the data untouched, and the estimate must move by that rotation."
    )
    extra = np.array([3.0, -4.0, 0.0]) * 1.0e-3  # magnitude exactly 5 mrad, by hand
    assert abs(float(np.linalg.norm(extra)) - 5.0e-3) < 1.0e-18
    mis_declared = AlignmentRecord(
        times=generated.record.times,
        reference=generated.record.reference,
        reference_rates=generated.record.reference_rates,
        secondaries={
            name: SensorSpec(
                name,
                generated.record.secondaries[name].noise_sigma,
                generated.record.secondaries[name].mounting @ rotation_from_vector(-extra),
            )
        },
        secondary_rates={name: generated.record.secondary_rates[name]},
        provenance="controls C8 mis-declared mounting",
    )
    mis_verdict = observability.assess(mis_declared)
    mis_solution = batch.solve(mis_declared, mis_verdict, name)
    shift = float(np.linalg.norm(mis_solution.theta_rad - solution.theta_rad))
    out(f"    estimate moved {shift * 1e3:.6f} mrad, predicted "
        f"{float(np.linalg.norm(extra)) * 1e3:.6f} mrad")
    passed = out.check(
        "C8 MUST fire: a 5 mrad error in the declared mounting moves the estimate by 5 mrad",
        abs(shift - float(np.linalg.norm(extra))) < 0.02 * float(np.linalg.norm(extra)),
        f"measured {shift * 1e3:.6f} mrad against a predicted "
        f"{float(np.linalg.norm(extra)) * 1e3:.6f} mrad; a mis-declared mounting is "
        "indistinguishable from a misalignment and this product cannot separate them",
    )
    fired.append(("C8 mis-declared mounting", True, passed))

    out("")
    out(
        "C9  negating the injected misalignment must negate the estimate. Anchored on "
        "the injection, not on the first estimate."
    )
    positive = _base(SEED_CONTROL + 400, theta_rad=np.array([1.5e-3, -0.9e-3, 0.6e-3]))
    negative = _base(SEED_CONTROL + 400, theta_rad=-np.array([1.5e-3, -0.9e-3, 0.6e-3]))
    _, positive_solution = _estimate(positive)
    _, negative_solution = _estimate(negative)
    total = float(np.linalg.norm(positive_solution.theta_rad + negative_solution.theta_rad))
    positive_verdict = observability.assess(positive.record)["secondary_1"]
    bound = float(
        np.sqrt(
            sum(
                1.0 / dof.eigenvalue
                for dof in positive_verdict.degrees_of_freedom
                if np.isfinite(dof.crb_sigma_rad)
            )
        )
    )
    out(f"    sum of the two estimates: {total * 1e3:.6f} mrad against individual "
        f"magnitudes of {float(np.linalg.norm(positive_solution.theta_rad)) * 1e3:.6f} mrad")
    out(
        "    The two records share a seed, so they share the same rate samples, but "
        "their noise draws differ because the generator draws the misalignment before "
        "the measurement noise. The residual sum is therefore the sum of two "
        "independent estimation errors and its expected magnitude is sqrt(2) times the "
        f"total Cramer-Rao one sigma of {bound * 1e3:.6f} mrad, which is "
        f"{bound * np.sqrt(2.0) * 1e3:.6f} mrad."
    )
    passed = out.check(
        "C9 MUST fire: negating the injection negates the estimate",
        total < 4.0 * np.sqrt(2.0) * bound,
        f"residual sum {total * 1e3:.6f} mrad against a predicted "
        f"{bound * np.sqrt(2.0) * 1e3:.6f} mrad and a declared bound of 4 times that",
    )
    fired.append(("C9 sign of the injection", True, passed))

    out("")
    out(
        "C10 replacing the secondary's measurements with pure noise must leave the "
        "OBSERVABILITY verdict unchanged -- it depends only on the reference motion -- "
        "and must leave the estimate near zero with a residual at the noise level."
    )
    noise_rng = np.random.default_rng(77)
    pure_noise = AlignmentRecord(
        times=generated.record.times,
        reference=generated.record.reference,
        reference_rates=generated.record.reference_rates,
        secondaries=generated.record.secondaries,
        secondary_rates={
            name: noise_rng.normal(
                scale=generated.record.secondaries[name].noise_sigma,
                size=generated.record.secondary_rates[name].shape,
            )
        },
        provenance="controls C10 secondary replaced by noise",
    )
    noise_verdict = observability.assess(pure_noise)
    noise_solution = batch.solve(pure_noise, noise_verdict, name)
    verdict_unchanged = all(
        abs(
            noise_verdict[name].degrees_of_freedom[k].eigenvalue
            - verdict[name].degrees_of_freedom[k].eigenvalue
        )
        < 1.0e-9 * verdict[name].degrees_of_freedom[k].eigenvalue
        for k in range(3)
    )
    out(f"    observability eigenvalues unchanged: {verdict_unchanged}")
    out(f"    weighted residual RMS {noise_solution.residual_rms_after:.6e} rad/s against "
        f"{solution.residual_rms_after:.6e} rad/s on the intact record")
    out(f"    estimate magnitude {float(np.linalg.norm(noise_solution.theta_rad)) * 1e3:.6f} mrad")
    passed = out.check(
        "C10 MUST fire on the residual: a noise-only secondary raises it by a factor of 10",
        noise_solution.residual_rms_after > 10.0 * solution.residual_rms_after,
        f"{noise_solution.residual_rms_after / solution.residual_rms_after:.4f} times the "
        "intact residual",
    )
    fired.append(("C10a residual on a noise-only secondary", True, passed))
    passed = out.check(
        "C10 must NOT fire on the verdict: observability depends only on the reference motion",
        verdict_unchanged,
        "every Gramian eigenvalue is unchanged to 1e-9 relative, which is the reason the "
        "verdict can be computed before the estimate at all",
    )
    fired.append(("C10b verdict on a noise-only secondary", False, not passed))

    out.rule("6.3 what the observability verdict is worth. These MUST fire.")

    out("")
    out(
        "C11 the noise-floor gate of (11) removed. With floor_factor at its minimum of "
        "1, a single-axis record's smallest eigenvalue -- which is the reference "
        "sensor's own noise -- is no longer refused, and the estimator is asked for a "
        "component the record does not contain."
    )
    unguarded_claims = []
    guarded_claims = []
    declared_rank = []
    for index in range(20):
        generated_c11 = axis_aligned_record(
            seed=SEED_CONTROL + 500 + index, samples=SAMPLES, magnitude_rad=2.0e-3
        )
        guarded = observability.assess(generated_c11.record, floor_factor=4.0)["secondary_1"]
        unguarded = observability.assess(
            generated_c11.record, floor_factor=1.0
        )["secondary_1"]
        declared_rank.append(unguarded.rank)
        axis = guarded.degrees_of_freedom[-1].direction
        guarded_claims.append(
            abs(
                float(
                    np.dot(
                        batch.solve(
                            generated_c11.record,
                            observability.assess(generated_c11.record, floor_factor=4.0),
                            "secondary_1",
                        ).theta_rad,
                        axis,
                    )
                )
            )
        )
        unguarded_claims.append(
            abs(
                float(
                    np.dot(
                        batch.solve(
                            generated_c11.record,
                            observability.assess(generated_c11.record, floor_factor=1.0),
                            "secondary_1",
                        ).theta_rad,
                        axis,
                    )
                )
            )
        )
    unguarded_claims = np.array(unguarded_claims)
    guarded_claims = np.array(guarded_claims)
    out(f"    ranks declared with the gate at 1.0: {sorted(set(declared_rank))}, "
        f"rank 3 on {int(np.sum(np.array(declared_rank) == 3))} of 20")
    out(f"    claim along the unsupported axis, gate at 4.0: max "
        f"{float(np.max(guarded_claims)):.3e} rad")
    out(f"    claim along the unsupported axis, gate at 1.0: median "
        f"{float(np.median(unguarded_claims)) * 1e3:.4f} mrad, max "
        f"{float(np.max(unguarded_claims)) * 1e3:.4f} mrad")
    out("    injected truth there: 2.0000 mrad")
    passed = out.check(
        "C11 MUST fire: without the floor gate the estimator answers where it cannot",
        bool(np.max(unguarded_claims) > 1.0e-3) and float(np.max(guarded_claims)) < 1.0e-15,
        f"unguarded claims reach {float(np.max(unguarded_claims)) * 1e3:.4f} mrad while the "
        "gated solution reports exactly nothing",
    )
    fired.append(("C11 floor gate removed", True, passed))

    out("")
    out(
        "C12 the recursive estimator has no withholding mechanism at all. On the same "
        "records it returns a number in the unsupported direction whatever the gate is "
        "set to, which is why the batch solution is the reference and the filter's "
        "covariance is the only thing that says so."
    )
    mekf_claims = []
    for index in range(10):
        generated_c12 = axis_aligned_record(
            seed=SEED_CONTROL + 600 + index, samples=SAMPLES, magnitude_rad=2.0e-3
        )
        verdict_c12 = observability.assess(generated_c12.record)
        axis = verdict_c12["secondary_1"].degrees_of_freedom[-1].direction
        mekf = recursive.run(
            generated_c12.record, verdict_c12, "secondary_1", store_history=False
        )
        mekf_claims.append(abs(float(np.dot(mekf.theta_rad, axis))))
    mekf_claims = np.array(mekf_claims)
    out(f"    MEKF claim along the unsupported axis: median "
        f"{float(np.median(mekf_claims)) * 1e3:.4f} mrad, max "
        f"{float(np.max(mekf_claims)) * 1e3:.4f} mrad")
    passed = out.check(
        "C12 MUST fire: the recursive estimator answers in the unsupported direction",
        bool(np.median(mekf_claims) > 1.0e-3),
        f"median claim {float(np.median(mekf_claims)) * 1e3:.4f} mrad against an injected "
        "2.0000 mrad; the filter is honest only through its covariance",
    )
    fired.append(("C12 recursive has no refusal", True, passed))

    out("")
    out(
        "C13 the unconstrained SVD solution of (17) is wrong about the OBSERVABLE "
        "directions on a rank-deficient record, not only about the unobservable one. "
        "The gauge rotation it picks up in the null direction contaminates the rest "
        "through the non-linearity of rotation composition."
    )
    contamination = []
    gauge = []
    bounds_c13 = []
    for index in range(10):
        generated_c13 = axis_aligned_record(
            seed=SEED_CONTROL + 700 + index, samples=SAMPLES, magnitude_rad=2.0e-3
        )
        verdict_c13 = observability.assess(generated_c13.record)
        sensor = verdict_c13["secondary_1"]
        solution_c13 = batch.solve(generated_c13.record, verdict_c13, "secondary_1")
        svd_theta, _ = batch.wahba_svd(generated_c13.record, "secondary_1")
        axis = sensor.degrees_of_freedom[-1].direction
        gauge.append(abs(float(np.dot(svd_theta, axis))))
        contamination.append(
            float(np.linalg.norm(sensor.project_observable(svd_theta - solution_c13.theta_rad)))
        )
        bounds_c13.append(sensor.degrees_of_freedom[0].crb_sigma_rad)
    gauge = np.array(gauge)
    contamination = np.array(contamination)
    bounds = np.array(bounds_c13)
    out(f"    SVD gauge rotation in the null direction: median "
        f"{float(np.median(gauge)):.4f} rad, min {float(np.min(gauge)):.4f} rad, "
        "against an injected 0.0020 rad")
    out(f"    contamination of the observable subspace : median "
        f"{float(np.median(contamination)) * 1e3:.4f} mrad, which is "
        f"{float(np.median(contamination / bounds)):.4f} of the Cramer-Rao one sigma "
        f"({float(np.median(bounds)) * 1e3:.4f} mrad)")
    out(
        "    The contamination is reported in units of the bound and not as a "
        "pass/fail, because its size depends on where the truth sits relative to the "
        "null direction. Measured on 2000-sample single-axis records with a RANDOM "
        "injected misalignment rather than an axis-aligned one, the same quantity is "
        "2.54 of the bound; that figure is in tests/test_batch.py, where it is "
        "asserted to exceed the whole magnitude of the correct estimate."
    )
    passed = out.check(
        "C13 MUST fire: the SVD solution picks up a gauge rotation of order a radian",
        bool(np.min(gauge) > 0.1),
        f"smallest gauge rotation {float(np.min(gauge)):.4f} rad against an injected "
        f"0.0020 rad, a factor of {float(np.min(gauge)) / 0.002:.0f}; equation (17) has no "
        "mechanism to withhold the direction and reports whatever the decomposition's "
        "gauge gives",
    )
    fired.append(("C13 SVD contamination", True, passed))

    out.rule("6.4 audit: does any assertion in the suite rest on pytest.approx's default?")
    out(
        "pytest.approx(expected, rel=r) passes if EITHER the relative or an absolute "
        "bound is met, and the absolute default is 1e-12. For a misalignment component "
        "of order 1e-5 rad, every relative tolerance tighter than 1e-7 is therefore "
        "silently replaced by the default and the assertion stops testing what it "
        "says it tests.\n\n"
        "The audit is not a source scan. The whole suite is re-run in a subprocess "
        "with pytest.approx PATCHED so that a call carrying rel= and no abs= raises, "
        "which is the only way to find out what actually depends on the default."
    )
    out("")
    import subprocess
    import tempfile
    import xml.etree.ElementTree as ElementTree

    from _bootstrap import subprocess_environment

    plugin = """import pytest

_original = pytest.approx


def approx(expected, rel=None, abs=None, nan_ok=False):
    if rel is not None and abs is None:
        raise AssertionError(
            "this assertion passes rel= with no abs= and therefore rests on "
            "pytest.approx's hidden 1e-12 absolute default"
        )
    return _original(expected, rel=rel, abs=abs, nan_ok=nan_ok)


pytest.approx = approx
"""
    scratch = tempfile.mkdtemp(prefix="aligncal-approx-audit-")
    (__import__("pathlib").Path(scratch) / "approx_audit_plugin.py").write_text(plugin)
    environment = subprocess_environment()
    environment["PYTHONPATH"] = scratch + ":" + environment["PYTHONPATH"]
    junit = __import__("pathlib").Path(scratch) / "audit.xml"
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "approx_audit_plugin",
            f"--junitxml={junit}",
        ],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    root = ElementTree.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    declared = int(suite.get("tests", "0"))
    counted = len(suite.findall(".//testcase"))
    failures = int(suite.get("failures", "0")) + int(suite.get("errors", "0"))
    dependent = [
        case.get("name", "?")
        for case in suite.findall(".//testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    out(f"    suite re-run with the default forced off, exit {completed.returncode}")
    out(f"    tests declared {declared}, testcase elements {counted}")
    out(f"    assertions depending on the hidden default: {failures}")
    for entry in dependent:
        out(f"      {entry}")
    out("")
    out(
        "    The single dependent test is the deliberate demonstration in "
        "tests/test_regressions.py, which asserts BOTH branches: that a bare rel= "
        "passes on a value wrong by 1e-13 and that the same comparison with abs=0.0 "
        "does not. It is supposed to fail under this audit. No measurement in this "
        "repository rests on the default."
    )
    passed = out.check(
        "exactly one assertion in the suite depends on pytest.approx's absolute default",
        failures == 1 and declared == counted and declared > 0,
        f"{failures} of {declared} tests, and the junit declaration agrees with its own "
        f"{counted} testcase elements",
    )
    fired.append(("audit of pytest.approx tolerances", True, passed))

    out.rule("6.5 the control tally")
    must_fire = [entry for entry in fired if entry[1]]
    must_not = [entry for entry in fired if not entry[1]]
    out(f"  controls that MUST fire     : {len(must_fire)}; fired: "
        f"{sum(1 for entry in must_fire if entry[2])}")
    out(f"  controls that must NOT fire : {len(must_not)}; fired: "
        f"{sum(1 for entry in must_not if entry[2])}")
    out("")
    out(f"{'control':<42s} {'direction':>12s} {'fired':>7s} {'verdict':>9s}")
    for label, expected_fire, did_fire in fired:
        verdict_text = "ok" if expected_fire == did_fire else "WRONG"
        out(
            f"{label:<42s} {('must fire' if expected_fire else 'must not'):>12s} "
            f"{str(did_fire):>7s} {verdict_text:>9s}"
        )
    out("")
    out.check(
        "every control behaved in its declared direction",
        all(expected == actual for _, expected, actual in fired),
        f"{sum(1 for _, e, a in fired if e == a)} of {len(fired)}",
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
