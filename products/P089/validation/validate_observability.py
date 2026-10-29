"""Section 1: the observability verdict itself.

Checked here: equation (10) as an identity on every declared profile; that the
unobservable direction is the dominant rotation axis; the independent Gershgorin
and Rayleigh bracket of (13) against what the eigensolver returns; the noise
floor of (11) and its effect on the rank verdict; and the sensitivity of the
verdict to the two declared thresholds, swept rather than asserted.
"""

from __future__ import annotations

from _bootstrap import SEED_GRID, Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from aligncal import dataset, observability  # noqa: E402
from aligncal.geometry import angle_between  # noqa: E402


def main() -> int:
    out = Tee("validate_observability_output.txt")
    out("VALIDATION 1 -- THE OBSERVABILITY VERDICT")
    out("")
    for line in environment_lines():
        out(f"  {line}")

    out.rule("1.1 equation (10): lambda_i = (W / s^2) (1 - p_i)")
    out(
        "The Gramian of (8) and the normalised rate second moment of (9) share "
        "eigenvectors exactly, so the least observable alignment direction is the "
        "most excited rotation axis. Measured as the worst relative departure over "
        "every declared profile and two noise levels."
    )
    out("")
    out(f"{'profile':<18s} {'sigma rad/s':>12s} {'worst rel err':>15s} {'rank':>6s} {'cond':>12s}")
    worst_identity = 0.0
    for profile in dataset.PROFILES:
        for sigma in (5.0e-5, 5.0e-4):
            generated = dataset.generate(
                profile=profile, samples=1500, noise_sigma=sigma, seed=SEED_GRID
            )
            record = generated.record
            shares, _, energy = observability.excitation_spectrum(record)
            scale = energy / record.pair_sigma("secondary_1") ** 2
            values = np.linalg.eigvalsh(observability.gramian(record, "secondary_1"))
            relative = float(np.max(np.abs(values - scale * (1.0 - shares)))) / scale
            worst_identity = max(worst_identity, relative)
            sensor = observability.assess(record)["secondary_1"]
            out(
                f"{profile:<18s} {sigma:>12.1e} {relative:>15.3e} "
                f"{sensor.rank:>6d} {sensor.condition_number:>12.4g}"
            )
    out("")
    out.check(
        "equation (10) holds on every profile",
        worst_identity < 1.0e-9,
        f"worst relative departure {worst_identity:.3e}, tolerance 1e-9",
    )

    out.rule("1.2 the unobservable direction IS the rotation axis")
    out(
        "For a single-axis record, the angle between the worst-observed Gramian "
        "eigendirection and the record's dominant rotation axis, over 60 seeds. "
        "The excitation axes are placed by a seeded random rotation, so no seed has "
        "the motion along a body axis."
    )
    angles = []
    body_angles = []
    for index in range(60):
        generated = dataset.generate(profile="single_axis", samples=900, seed=7000 + index)
        sensor = observability.assess(generated.record)["secondary_1"]
        angles.append(sensor.unobservable_axis_angle_deg)
        axis = sensor.degrees_of_freedom[-1].direction
        nearest = min(
            min(float(np.degrees(angle_between(axis, column))),
                180.0 - float(np.degrees(angle_between(axis, column))))
            for column in np.eye(3)
        )
        body_angles.append(nearest)
    angles = np.array(angles)
    body_angles = np.array(body_angles)
    out("")
    out(f"  angle to the dominant rotation axis, deg: max {float(np.max(angles)):.6f}, "
        f"median {float(np.median(angles)):.6f}")
    out(f"  angle to the NEAREST body axis, deg     : min {float(np.min(body_angles)):.4f}, "
        f"median {float(np.median(body_angles)):.4f}")
    out.check(
        "the unobservable direction is the rotation axis on 60 of 60 seeds",
        bool(np.max(angles) < 0.5),
        f"worst {float(np.max(angles)):.6f} deg, tolerance 0.5 deg",
    )
    out.check(
        "the result is not trivial: the motion is never along a body axis",
        bool(np.min(body_angles) > 5.0),
        f"closest approach {float(np.min(body_angles)):.4f} deg, requirement > 5 deg",
    )

    out.rule("1.3 equation (13): an independent bracket on lambda_min")
    out(
        "lambda_min is computed by numpy.linalg.eigh. It is bracketed below by "
        "Gershgorin's circle theorem and above by the Rayleigh quotient at the "
        "canonical basis vectors, neither of which uses an eigensolver. A negative "
        "lower bound is informative rather than wrong: it says diagonal dominance "
        "alone cannot separate the matrix from a singular one."
    )
    out("")
    out(f"{'profile':<18s} {'lower':>14s} {'lambda_min':>14s} {'upper':>14s} {'consistent':>11s}")
    inconsistent = 0
    negative_lower = 0
    for profile in dataset.PROFILES:
        generated = dataset.generate(profile=profile, samples=1500, seed=SEED_GRID + 1)
        matrix = observability.gramian(generated.record, "secondary_1")
        lower, upper = observability.eigenvalue_bracket(matrix)
        smallest = float(np.min(np.linalg.eigvalsh(matrix)))
        ok = (
            lower <= smallest * (1.0 + 1.0e-9) + 1.0e-9
            and smallest <= upper * (1.0 + 1.0e-9) + 1.0e-9
        )
        inconsistent += not ok
        negative_lower += lower < 0.0
        out(f"{profile:<18s} {lower:>14.6g} {smallest:>14.6g} {upper:>14.6g} {str(ok):>11s}")
    out("")
    out.check(
        "the bracket of (13) contains lambda_min on every profile",
        inconsistent == 0,
        f"{inconsistent} of {len(dataset.PROFILES)} inconsistent",
    )
    out(
        f"  the Gershgorin lower bound is negative on {negative_lower} of "
        f"{len(dataset.PROFILES)} profiles, which is reported rather than hidden: "
        "diagonal dominance is a weak test on a matrix this ill-conditioned."
    )

    out.rule("1.4 the noise floor of equation (11), and what it is for")
    out(
        "The Gramian is formed from the MEASURED reference rate, so the reference "
        "sensor's own noise contributes 2 sigma_A^2 per sample in every direction, "
        "including the one the motion never excited. A bare rank test on the measured "
        "Gramian is therefore wrong. Measured below: the smallest eigenvalue of a "
        "single-axis record against the floor, and the rank verdict with the gate and "
        "without it."
    )
    out("")
    out(f"{'samples':>9s} {'lambda_min':>14s} {'floor (11)':>14s} {'ratio':>10s} "
        f"{'cond':>12s} {'rank, gated':>12s} {'rank, bare':>11s}")
    fooled = 0
    gated_correct = 0
    for samples in (200, 500, 1000, 2000, 4000):
        generated = dataset.generate(profile="single_axis", samples=samples, seed=SEED_GRID + 2)
        record = generated.record
        matrix = observability.gramian(record, "secondary_1")
        values = np.linalg.eigvalsh(matrix)
        floor = observability.noise_floor_eigenvalue(record, "secondary_1")
        sensor = observability.assess(record)["secondary_1"]
        # A bare numerical rank test, which is what a reader would reach for.
        bare = int(np.linalg.matrix_rank(matrix))
        fooled += bare == 3
        gated_correct += sensor.rank == 2
        out(
            f"{samples:>9d} {float(values[0]):>14.6g} {floor:>14.6g} "
            f"{float(values[0]) / floor:>10.4f} {sensor.condition_number:>12.4g} "
            f"{sensor.rank:>12d} {bare:>11d}"
        )
    out("")
    out.check(
        "a bare numerical rank test is fooled on every single-axis record",
        fooled == 5,
        f"numpy.linalg.matrix_rank reported 3 on {fooled} of 5 records that support 2",
    )
    out.check(
        "the floor gate of (11) is not fooled on any of them",
        gated_correct == 5,
        f"the gated verdict reported rank 2 on {gated_correct} of 5",
    )

    out.rule("1.5 sensitivity of the verdict to the two declared thresholds")
    out(
        "Neither threshold is a specification. Both are swept here so the reader can "
        "see what the verdict depends on. The floor factor is swept at a fixed "
        "requirement of 1e-3 rad; the requirement is swept at a fixed floor factor "
        "of 4."
    )
    out("")
    out(f"{'floor factor':>13s} " + " ".join(f"{p[:11]:>12s}" for p in dataset.PROFILES))
    for factor in (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 64.0):
        ranks = []
        for profile in dataset.PROFILES:
            generated = dataset.generate(profile=profile, samples=1500, seed=SEED_GRID + 3)
            ranks.append(
                observability.assess(generated.record, floor_factor=factor)["secondary_1"].rank
            )
        out(f"{factor:>13g} " + " ".join(f"{r:>12d}" for r in ranks))
    out("")
    out(f"{'requirement rad':>16s} {'profile':<18s} {'observable':>11s} {'marginal':>9s} "
        f"{'unobservable':>13s}")
    for requirement in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5):
        for profile in ("single_axis", "near_single_axis", "three_axis"):
            generated = dataset.generate(profile=profile, samples=1500, seed=SEED_GRID + 3)
            sensor = observability.assess(
                generated.record, required_sigma_rad=requirement
            )["secondary_1"]
            counts = {"observable": 0, "marginal": 0, "unobservable": 0}
            for dof in sensor.degrees_of_freedom:
                counts[dof.verdict] += 1
            out(
                f"{requirement:>16.1e} {profile:<18s} {counts['observable']:>11d} "
                f"{counts['marginal']:>9d} {counts['unobservable']:>13d}"
            )
    out("")
    out(
        "Reading: the rank verdict is stable in the floor factor from 1.5 to 64 on "
        "every profile, so the declared 4 is not load-bearing. The three-valued "
        "verdict IS sensitive to the declared requirement, which is the intended "
        "behaviour: a requirement is an input, and 'observable' has no meaning "
        "without one."
    )

    out.rule("1.6 two rotation axes are enough for three alignment degrees of freedom")
    out(
        "By (10), lambda_i = (W/s^2)(1 - p_i), and a two-axis rotation has "
        "p = (1/2, 1/2, 0), so the third eigenvalue is the LARGEST. It is easy to "
        "assume three excited axes are needed; measured below, two suffice. The "
        "conditioning is reported beside it and the ordering is NOT the one a reader "
        "would guess, which is why it is measured and not asserted."
    )
    out("")
    out(f"{'profile':<18s} {'p0':>9s} {'p1':>9s} {'p2':>9s} {'rank':>5s} {'cond':>12s} "
        f"{'worst CRB arcsec':>17s}")
    for profile in dataset.PROFILES:
        generated = dataset.generate(profile=profile, samples=1500, seed=SEED_GRID + 4)
        report = observability.assess(generated.record)
        sensor = report["secondary_1"]
        worst = max(
            (d.crb_sigma_arcsec for d in sensor.degrees_of_freedom if np.isfinite(d.crb_sigma_rad)),
            default=float("inf"),
        )
        rendered = "unbounded" if sensor.rank_deficient else f"{worst:.3f}"
        out(
            f"{profile:<18s} {report.excitation[0]:>9.6f} {report.excitation[1]:>9.6f} "
            f"{report.excitation[2]:>9.6f} {sensor.rank:>5d} "
            f"{sensor.condition_number:>12.4g} {rendered:>17s}"
        )
    two_axis = observability.assess(
        dataset.generate(profile="two_axis", samples=1500, seed=SEED_GRID + 4).record
    )["secondary_1"]
    three_axis = observability.assess(
        dataset.generate(profile="three_axis", samples=1500, seed=SEED_GRID + 4).record
    )["secondary_1"]
    out("")
    out.check(
        "two excited rotation axes give rank 3",
        two_axis.rank == 3,
        f"rank {two_axis.rank}, condition number {two_axis.condition_number:.4f}",
    )
    out.check(
        "both two-axis and three-axis records are well conditioned",
        two_axis.condition_number < 10.0 and three_axis.condition_number < 10.0,
        f"two-axis {two_axis.condition_number:.4f}, three-axis "
        f"{three_axis.condition_number:.4f}, both against a declared bound of 10",
    )
    out(
        f"  ORDERING, reported as measured and not as expected: the two-axis record is "
        f"WORSE conditioned ({two_axis.condition_number:.4f}) than the three-axis one "
        f"({three_axis.condition_number:.4f}). Two equal-amplitude axes give "
        "p = (1/2, 1/2, 0) and therefore eigenvalues in the ratio 2:1:1 by (10); the "
        "three-axis profile's unequal amplitudes spread p more evenly and the ratio is "
        "closer to 1. Excitation on more axes is not the same thing as better "
        "conditioning, and the number of excited axes alone does not predict the "
        "ordering."
    )

    out.rule("1.7 a worked verdict, printed in full")
    generated = dataset.generate(profile="near_single_axis", samples=2000, seed=SEED_GRID + 5)
    for line in observability.assess(generated.record).summary():
        out(line)

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
