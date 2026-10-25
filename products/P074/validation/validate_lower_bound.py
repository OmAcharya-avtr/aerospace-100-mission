"""The triangle inequality the product's only rigorous claim rests on.

This package measures the separation ``D`` between two propagations. The only
rigorous statement that supports is

    ``|e1| + |e2| >= D``, so ``max(|e1|, |e2|) >= D/2``,

where ``e1`` and ``e2`` are the two propagators' unknown errors against truth.
``D/2`` is therefore a lower bound on **the larger** of the two errors, and
nothing more. It is not a lower bound on either error individually and it is
not an estimate of either.

The check is against a case where truth is known in closed form: a two-body
orbit propagated by :mod:`orbitdrift.kepler`. Two deliberately wrong
propagations stand in for the two models, their errors against truth are
computable, and the inequality is verified directly. This is a check against an
independently derived bound rather than a comparison of two implementations,
and it is the class of check that catches an inequality written the wrong way
round -- which a property test sharing the expression could not.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift.constants import WGS72  # noqa: E402
from orbitdrift.divergence import larger_error_lower_bound_km  # noqa: E402
from orbitdrift.kepler import (  # noqa: E402
    ClassicalElements,
    elements_to_rv,
    orbital_period_s,
    propagate_two_body,
)


def main() -> int:
    out = Tee("validate_lower_bound_output.txt")
    out.rule("orbitdrift -- the lower bound, against a case where truth is known")
    for line in environment_lines():
        out(line)

    out.rule("the statement being checked")
    out("  D        = |r_model_a(t) - r_model_b(t)|, the separation this package measures")
    out("  e1, e2   = the two models' unknown errors against truth")
    out("  claim    = max(|e1|, |e2|) >= D/2, from |e1| + |e2| >= D")
    out("  NOT      = a bound on |e1| alone, or on |e2| alone, or an estimate of either")

    elements = ClassicalElements(
        WGS72.radius_km + 700.0, 0.01, float(np.deg2rad(51.6)), 0.3, 0.7, 0.2
    )
    r0, v0 = elements_to_rv(elements)
    period = orbital_period_s(r0, v0)
    times = np.linspace(0.0, 20.0 * period, 401)
    truth, _ = propagate_two_body(r0, v0, times)

    out.rule("the inequality over a grid of deliberately wrong model pairs")
    out("  Each pair is two closed-form two-body propagations from perturbed initial")
    out("  conditions. Truth is the unperturbed one, so e1 and e2 are computable.")
    out("")
    out("  dr_a km   dv_b scale   worst D km   worst bound km   margin km   violations")
    violations_total = 0
    rows = 0
    worst_margin = float("inf")
    largest_bound = 0.0
    for offset_km in (0.0, 0.05, 0.5, 5.0, 50.0):
        for velocity_scale in (1.0, 1.000002, 1.00002, 1.0002):
            model_a, _ = propagate_two_body(r0 + np.array([offset_km, 0.0, 0.0]), v0, times)
            model_b, _ = propagate_two_body(r0, v0 * velocity_scale, times)
            error_a = np.linalg.norm(model_a - truth, axis=1)
            error_b = np.linalg.norm(model_b - truth, axis=1)
            separation = np.linalg.norm(model_a - model_b, axis=1)
            bound = larger_error_lower_bound_km(separation)
            larger = np.maximum(error_a, error_b)
            margin = larger - bound
            violations = int(np.count_nonzero(margin < -1e-9))
            violations_total += violations
            rows += 1
            worst_margin = min(worst_margin, float(margin.min()))
            largest_bound = max(largest_bound, float(bound.max()))
            out(
                f"  {offset_km:7.2f}   {velocity_scale:10.6f}   "
                f"{float(separation.max()):10.4f}   {float(bound.max()):14.4f}   "
                f"{float(margin.min()):9.4f}   {violations:10d}"
            )

    out("")
    out(f"  pairs tested                  : {rows}")
    out(f"  sample points per pair        : {times.size}")
    out(f"  total inequality violations   : {violations_total}")
    out(f"  worst margin (larger - bound) : {worst_margin:.6e} km")
    out.check(
        "the inequality holds at every sample of every pair",
        violations_total == 0,
        f"{violations_total} violations, worst margin {worst_margin:.3e} km",
    )
    out.check(
        "the bound is not vacuous: somewhere it is a real constraint",
        largest_bound > 1.0,
        f"largest bound {largest_bound:.4f} km",
    )

    out.rule("the bound says nothing about either model individually")
    out("  A perfect model compared against a bad one gives a large D and an error")
    out("  of exactly zero. The bound still holds -- it bounds the LARGER error --")
    out("  but the smaller error is below it, which is why README.md does not claim")
    out("  the bound applies to the element set the user holds.")
    bad, _ = propagate_two_body(r0 + np.array([5.0, 0.0, 0.0]), v0, times)
    separation = np.linalg.norm(truth - bad, axis=1)
    bound = larger_error_lower_bound_km(separation)
    perfect_error = np.zeros_like(separation)
    bad_error = np.linalg.norm(bad - truth, axis=1)
    out("")
    out(f"  worst D                       : {float(separation.max()):.4f} km")
    out(f"  worst bound D/2               : {float(bound.max()):.4f} km")
    out(f"  perfect model's error         : {float(perfect_error.max()):.4f} km")
    out(f"  bad model's error             : {float(bad_error.max()):.4f} km")
    out.check(
        "the perfect model's error is below the bound at every sample",
        bool(np.all(perfect_error < bound + 1e-12)),
        "so the bound is not a bound on an individual model",
    )
    out.check(
        "the larger of the two errors is above the bound at every sample",
        bool(np.all(np.maximum(perfect_error, bad_error) >= bound - 1e-9)),
        f"worst margin {float(np.min(np.maximum(perfect_error, bad_error) - bound)):.3e} km",
    )

    out.rule("negative control: the inequality written the wrong way round")
    out("  If the bound were D rather than D/2, it would claim max(|e1|, |e2|) >= D,")
    out("  which is false whenever the two errors are comparable. The control shows")
    out("  it failing, which is what makes the passing result above evidence.")
    model_a, _ = propagate_two_body(r0 + np.array([1.0, 0.0, 0.0]), v0, times)
    model_b, _ = propagate_two_body(r0 - np.array([1.0, 0.0, 0.0]), v0, times)
    error_a = np.linalg.norm(model_a - truth, axis=1)
    error_b = np.linalg.norm(model_b - truth, axis=1)
    separation = np.linalg.norm(model_a - model_b, axis=1)
    wrong_violations = int(np.count_nonzero(np.maximum(error_a, error_b) < separation - 1e-9))
    correct_violations = int(
        np.count_nonzero(
            np.maximum(error_a, error_b) < larger_error_lower_bound_km(separation) - 1e-9
        )
    )
    out("")
    out(f"  samples                       : {times.size}")
    out(f"  violations with the bound D   : {wrong_violations}")
    out(f"  violations with the bound D/2 : {correct_violations}")
    out.check(
        "the wrong-way-round bound is violated (control fires)",
        wrong_violations > 0,
        f"{wrong_violations} of {times.size} samples",
    )
    out.check(
        "the correct bound is not violated on the same data",
        correct_violations == 0,
        f"{correct_violations} of {times.size} samples",
    )
    out("")
    out("  The two models here are offset in opposite directions by 1 km, so their")
    out("  errors are each about half the separation and the D-bound fails at")
    out("  almost every sample. This is the equality case of the triangle")
    out("  inequality, which is exactly where a factor of two matters.")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
