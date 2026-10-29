"""The worked example quoted in README.md, executed so the quoted output is real.

This file is the source of the README's worked-example block. The code between
the two markers below is what appears there, with ``out`` spelled ``print``, and
the output in README.md is copied from ``worked_example_output.txt``, which this
script writes.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()


def main() -> int:
    out = Tee("worked_example_output.txt")
    out("WORKED EXAMPLE -- the block quoted in README.md, executed")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out("=" * 78)
    out("")

    # --- README BLOCK BEGINS -------------------------------------------
    import numpy as np

    from aligncal import batch, dataset, observability

    generated = dataset.generate(profile="single_axis", samples=1500, seed=9091)
    record = generated.record

    # The verdict is computed BEFORE any estimate, and it is the first field of
    # the result. Nothing in this package returns an alignment without it.
    verdict = observability.assess(record, required_sigma_rad=1.0e-3)
    sensor = verdict["secondary_1"]

    out(f"samples                  : {len(record)}")
    out(f"excitation shares (eq 9)  : {np.array2string(verdict.excitation, precision=6)}")
    out(f"excitation entropy        : {verdict.excitation_entropy:.6f} of 1.584963 bits")
    out(f"Gramian rank (eq 8, 11)   : {sensor.rank} of 3")
    out(f"dominant rotation axis    : {np.array2string(verdict.dominant_axis, precision=4)}")
    out("")

    out("per degree of freedom, worst last:")
    for dof in sensor.degrees_of_freedom:
        bound = (
            "unbounded"
            if not np.isfinite(dof.crb_sigma_rad)
            else f"{dof.crb_sigma_arcsec:9.3f} arcsec"
        )
        out(
            f"  dof {dof.index}  {dof.verdict:<13s} CRB {bound:>18s}  "
            f"excitation share {dof.excitation_share:.6f}"
        )
    out("")

    unsupported = sensor.unobservable_directions[0]
    out(f"the record does not support dof {unsupported.index}, and that direction is")
    out(f"  {np.array2string(unsupported.direction, precision=4)}")
    out(f"which is {sensor.unobservable_axis_angle_deg:.4f} deg from the dominant rotation axis.")
    out("No volume of this motion would change that: by equation (10) the Gramian")
    out("eigenvalue along a rotation axis is (W / s^2) (1 - p) with p = 1.")
    out("")

    # Only now the estimate, which takes the verdict as a required argument.
    solution = batch.solve(record, verdict, "secondary_1")
    out(f"batch estimate (eq 14-16) : "
        f"{np.array2string(solution.theta_mrad, precision=6)} mrad")
    out(f"withheld degrees of freedom: {solution.withheld}")
    for index in solution.withheld:
        direction = sensor.degrees_of_freedom[index].direction
        out(f"  component along dof {index}   : "
            f"{float(np.dot(solution.theta_rad, direction)):+.3e} rad, projected out")
        out(f"  bound reported there      : {solution.sigma_rad_in_direction(direction)}")
    out("")

    truth = generated.truth["secondary_1"]
    out(f"injected truth            : {np.array2string(truth * 1e3, precision=6)} mrad")
    observable_error = sensor.project_observable(solution.theta_rad - truth)
    unobservable_error = sensor.project_unobservable(solution.theta_rad - truth)
    arcsec = 206264.806247
    out(f"error, observable part    : "
        f"{np.linalg.norm(observable_error) * arcsec:.4f} arcsec")
    out(f"error, unobservable part  : "
        f"{np.linalg.norm(unobservable_error) * arcsec:.4f} arcsec")
    out("The second figure is the component of the truth the record cannot support.")
    out("Every estimator carries it in full, including this one, which declines to")
    out("guess at it rather than reporting a number for it.")
    # --- README BLOCK ENDS ---------------------------------------------
    out("")

    # The front door does the same thing in one call, verdict first.
    import aligncal

    result = aligncal.align(record, with_recursive=True)
    mekf = result.recursive["secondary_1"]
    out(f"recursive estimate (19-22): {np.array2string(mekf.theta_mrad, precision=6)} mrad")
    out(f"prior retained per dof    : "
        f"{np.array2string(mekf.prior_retention(), precision=6)}")
    out(f"worst/best sigma ratio    : {mekf.inflation_ratio():.4f}")
    out("The recursive estimator does not withhold anything. It reports what it does")
    out("not know through its covariance: the prior it started with is still there in")
    out("the direction the record never excited.")

    out("")
    out("=" * 78)
    out.check(
        "the record used by the README example is rank deficient",
        sensor.rank_deficient,
        f"rank {sensor.rank} of 3",
    )
    out.check(
        "the withheld component is zero to within 1e-15 rad",
        abs(float(np.dot(solution.theta_rad, unsupported.direction))) < 1.0e-15,
        f"{float(np.dot(solution.theta_rad, unsupported.direction)):+.3e} rad",
    )
    out.check(
        "the bound reported in the withheld direction is infinite",
        not np.isfinite(solution.sigma_rad_in_direction(unsupported.direction)),
        "inf",
    )
    out.check(
        "the recursive estimator keeps most of its prior in the withheld direction",
        float(mekf.prior_retention()[unsupported.index]) > 0.05,
        f"retained {float(mekf.prior_retention()[unsupported.index]):.6f}",
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
