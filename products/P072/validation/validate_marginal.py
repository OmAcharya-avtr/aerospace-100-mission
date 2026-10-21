"""The marginal value of contact time, and the left-behind report.

Checks:

1. **monotonicity, conditional on the expiry structure.** Adding capacity
   cannot lower the optimal objective *provided the extension crosses no
   product's expiry*. A window's end is also its delivery epoch, so an extension
   that pushes the end past a product's expiry removes that product's column
   from the model and the objective may legitimately fall. The sweep is
   therefore checked in two parts: decreasing steps that cross no expiry, which
   must be zero, and decreasing steps that do cross one, which are measured and
   reported. The first run of this script found a 12.89 per cent drop between a
   440 s and a 460 s extension and the cause was the second kind.
2. **non-concavity, reported rather than asserted.** The objective of an integer
   program is a step function of capacity and is not concave. The count of
   convexity violations in the second difference is reported as a property of
   the problem; a product that claimed a smooth shadow price here would be
   claiming something false.
3. **the finite difference against an actual extension.** The marginal value
   reported for a window must equal the objective gain from actually extending
   that window by the same step and re-solving, which is how the number is
   defined. This is a self-consistency check on the implementation, and it is
   exact by construction, so the measured residual is a check that the
   construction is what is documented.
4. **sign.** The marginal value of contact time is non-negative wherever the
   step crosses no expiry, and is allowed to be negative where it does. Both
   counts are reported.
5. **extension bias.** The extension holds the window's final rate. An
   extension that instead followed the sine profile down delivers fewer bits,
   so the reported marginal value is an upper estimate. The size of the bias is
   measured here against a profile-following extension, because it is the
   largest modelling caveat on the headline number.
6. **the left-behind report.** Every product is either selected or has exactly
   one reason; the value left behind plus the value selected accounts for the
   whole inventory; and the ``capacity`` shortfall, when converted to seconds
   and actually added to that window, admits the product.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from downlinkpick.contacts import ContactPlan, ContactWindow  # noqa: E402
from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.marginal import (  # noqa: E402
    LEFT_BEHIND_REASONS,
    best_window_to_extend,
    capacity_sweep,
    left_behind_report,
    marginal_value_of_contact,
    value_left_behind,
)
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402

#: Relative tolerance on an objective comparison.
TOLERANCE_REL = 1.0e-9

SWEEP_STEPS = tuple(float(step) for step in range(0, 601, 20))


def _problem(instance) -> SelectionProblem:
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    return SelectionProblem(
        inventory=instance.inventory,
        plan=instance.plan,
        values=values,
        sizes_bits=sizes_from_ratios(instance.inventory, instance.true_ratios),
    )


def _profile_following_extension(window: ContactWindow, extra_s: float) -> ContactWindow:
    """Extend a window by continuing its final linear segment down to zero rate.

    The final segment of the sine profile has a negative slope; continuing it
    delivers fewer bits than holding the final rate, and the rate is clamped at
    zero. This is the comparison used in check 5 and is itself a model, not a
    link budget.
    """
    times = [point[0] for point in window.rate_breakpoints]
    rates = [point[1] for point in window.rate_breakpoints]
    slope = (rates[-1] - rates[-2]) / (times[-1] - times[-2]) if times[-1] > times[-2] else 0.0
    end_rate = max(rates[-1] + slope * extra_s, 0.0)
    return ContactWindow(
        name=window.name,
        start_s=window.start_s,
        end_s=window.end_s + extra_s,
        rate_breakpoints=(*window.rate_breakpoints, (window.end_s + extra_s, end_rate)),
        reliability=window.reliability,
        station=window.station,
    )


def main() -> int:
    out = Tee("validate_marginal_output.txt")
    out.rule("downlinkpick validation: the marginal value of contact time")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("1 and 2. capacity sweep: monotone where no expiry is crossed, not concave")
    out(f"steps added to window 1: {SWEEP_STEPS[0]:.0f} to {SWEEP_STEPS[-1]:.0f} s "
        f"in {len(SWEEP_STEPS)} points")
    out("monotonicity tolerance : 1.0e-9 relative on the objective")
    out("a step is classified by whether the extension newly crosses a product's expiry")
    out("")
    out(f"{'seed':>6s} {'dec clean':>10s} {'dec crossing':>13s} {'worst drop':>12s} "
        f"{'convexity viol':>15s} {'value 0 -> max':>22s}")
    decreasing_clean = 0
    decreasing_crossing = 0
    convexity_total = 0
    worst_drop_relative = 0.0
    worst_drop_seed = 0
    for seed in range(1, 41):
        instance = generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.4)
        problem = _problem(instance)
        points = capacity_sweep(problem, problem.plan.names[0], SWEEP_STEPS)
        values = np.array([point.total_value for point in points])
        crossings = np.array([point.n_expiry_crossings for point in points])
        differences = np.diff(values)
        new_crossing = np.diff(crossings) > 0
        scale = max(abs(values[0]), 1.0)
        falling = differences < -TOLERANCE_REL * scale
        clean = int(np.sum(falling & ~new_crossing))
        crossing = int(np.sum(falling & new_crossing))
        decreasing_clean += clean
        decreasing_crossing += crossing
        if clean:
            failures += 1
            out(
                f"  FAIL seed {seed}: {clean} decreasing step(s) that cross no expiry; "
                "adding capacity alone must never lower the optimum"
            )
        local_drop = 0.0
        if falling.any():
            index = int(np.argmin(differences))
            local_drop = float(-differences[index] / max(values[index], 1.0))
            if local_drop > worst_drop_relative:
                worst_drop_relative = local_drop
                worst_drop_seed = seed
        second = np.diff(differences)
        convexity = int(np.sum(second > TOLERANCE_REL * scale))
        convexity_total += convexity
        if seed <= 8:
            out(
                f"{seed:6d} {clean:10d} {crossing:13d} {local_drop:12.4e} {convexity:15d} "
                f"{values[0]:10.4f} -> {values[-1]:8.4f}"
            )
    out("")
    out("instances swept                            : 40")
    out(f"decreasing steps crossing no expiry        : {decreasing_clean} (must be 0)")
    out(f"decreasing steps that cross an expiry      : {decreasing_crossing}")
    out(f"worst relative drop                        : {worst_drop_relative:.6e} "
        f"at seed {worst_drop_seed}")
    out(f"convexity violations across all sweeps     : {convexity_total}")
    out("a non-zero convexity count is the expected result and is reported, not fixed:")
    out("the objective of an integer program is a step function of capacity, so the")
    out("marginal value of contact time is not a derivative and not monotone in capacity.")
    out("the decreasing steps that cross an expiry are the other expected result: a")
    out("window's end is its delivery epoch, so a longer pass can be too late for a")
    out("product that the shorter pass would have delivered in time.")

    out.rule("3 and 4. the finite difference against an actual extension, and its sign")
    out("identity: reported value_per_second * delta_s == objective(extended) - objective")
    out("tolerance: 1.0e-9 relative")
    out("")
    out(f"{'seed':>6s} {'window':>8s} {'+value':>10s} {'value/s':>12s} {'residual':>11s}")
    worst_residual = 0.0
    negative = 0
    priced = 0
    for seed in range(1, 31):
        instance = generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.4)
        problem = _problem(instance)
        base = exact_select(problem)
        marginals = marginal_value_of_contact(problem, base=base, delta_s=30.0)
        for marginal in marginals:
            priced += 1
            extended = exact_select(
                problem.with_plan(problem.plan.extended_window(marginal.window, 30.0))
            )
            expected = extended.total_value - base.total_value
            residual = abs(marginal.delta_value - expected) / max(abs(expected), 1.0)
            worst_residual = max(worst_residual, residual)
            if residual > TOLERANCE_REL:
                failures += 1
                out(f"  FAIL seed {seed} {marginal.window}: reported {marginal.delta_value!r} "
                    f"against measured {expected!r}")
            if marginal.value_per_second < -TOLERANCE_REL:
                negative += 1
                if not marginal.expiry_crossings:
                    failures += 1
                    out(
                        f"  FAIL seed {seed} {marginal.window}: negative marginal value "
                        f"{marginal.value_per_second!r} with no expiry crossing to explain it"
                    )
                else:
                    out(
                        f"  note seed {seed} {marginal.window}: marginal value "
                        f"{marginal.value_per_second:.6e} is negative because the extension "
                        f"crosses the expiry of {', '.join(marginal.expiry_crossings)}"
                    )
            if seed <= 3:
                out(
                    f"{seed:6d} {marginal.window:>8s} {marginal.delta_value:10.4f} "
                    f"{marginal.value_per_second:12.6f} {residual:11.3e}"
                )
    out("")
    out(f"window-prices computed       : {priced}")
    out(f"worst relative residual      : {worst_residual:.6e}")
    out(f"negative marginal values     : {negative}, every one of them explained by an")
    out("                               expiry crossing or the check above would have failed")

    out.rule("5. the bias from holding the final rate across the extension")
    out("the library's extension holds the final rate; this compares it against an")
    out("extension that continues the final linear segment down, clamped at zero rate")
    out("")
    out(f"{'step [s]':>9s} {'held [bit]':>15s} {'following [bit]':>16s} {'held/following':>15s}")
    for step in (1.0, 10.0, 30.0, 120.0, 300.0):
        held_total = 0.0
        following_total = 0.0
        for seed in range(1, 21):
            instance = generate_instance(seed, n_products=20, n_windows=3)
            for window in instance.plan:
                held_total += window.extended(step).capacity_bits() - window.capacity_bits()
                following_total += (
                    _profile_following_extension(window, step).capacity_bits()
                    - window.capacity_bits()
                )
        ratio = held_total / following_total if following_total > 0.0 else float("inf")
        out(f"{step:9.1f} {held_total:15.6e} {following_total:16.6e} {ratio:15.6f}")
    out("")
    out("the ratio above 1 is the over-estimate: the reported marginal value of contact")
    out("time is an upper estimate by this factor, and it grows with the step because the")
    out("profile is falling at the end of a pass")

    out.rule("6. the left-behind report")
    out("checks: one reason per unselected product; selected value plus left-behind")
    out("        value equals the whole inventory value; and the reported capacity")
    out("        shortfall in seconds, actually added, admits the product")
    out("")
    out(f"{'seed':>6s} {'left':>5s} " + " ".join(f"{r:>11s}" for r in LEFT_BEHIND_REASONS)
        + f" {'accounting res':>15s}")
    shortfall_checks = 0
    shortfall_failures = 0
    shortfall_resolves = 0
    for seed in range(1, 41):
        instance = generate_instance(
            seed, n_products=22, n_windows=3, capacity_fraction=0.4, dependency_fraction=0.4
        )
        problem = _problem(instance)
        selection = exact_select(problem)
        rows = left_behind_report(problem, selection)
        totals = value_left_behind(rows)
        selected_value = sum(float(problem.values[n]) for n in selection.selected_set)
        accounted = selected_value + sum(totals.values())
        available = problem.total_value_available()
        residual = abs(accounted - available) / max(available, 1.0)
        if residual > TOLERANCE_REL:
            failures += 1
            out(f"  FAIL seed {seed}: value accounting residual {residual:.3e}")
        if len(rows) + len(selection.assignments) != len(problem.inventory):
            failures += 1
            out(f"  FAIL seed {seed}: {len(rows)} left behind plus "
                f"{len(selection.assignments)} selected is not {len(problem.inventory)}")
        for row in rows:
            if row.reason not in LEFT_BEHIND_REASONS:
                failures += 1
                out(f"  FAIL seed {seed}: unknown reason {row.reason!r}")
        used = selection.bits_by_window()
        for row in rows:
            if row.reason != "capacity" or row.best_window is None:
                continue
            shortfall_checks += 1
            seconds = row.extra_seconds_needed
            extended_plan = problem.plan.extended_window(row.best_window, seconds)
            free = (
                extended_plan[row.best_window].capacity_bits()
                - used.get(row.best_window, 0.0)
            )
            if row.bits > free + 1.0:
                shortfall_failures += 1
                failures += 1
                out(
                    f"  FAIL seed {seed}: {row.product} needs {row.bits:.6e} bits but "
                    f"{seconds:.3f} s of extension leaves only {free:.6e} bits free"
                )
            if row.product in exact_select(
                problem.with_plan(
                    problem.plan.extended_window(row.best_window, seconds * 1.05 + 1.0)
                )
            ).selected_set:
                shortfall_resolves += 1
        if seed <= 8:
            out(
                f"{seed:6d} {len(rows):5d} "
                + " ".join(f"{totals[r]:11.4f}" for r in LEFT_BEHIND_REASONS)
                + f" {residual:15.3e}"
            )
    out("")
    out("instances                                  : 40")
    out(f"capacity shortfalls probed                 : {shortfall_checks}")
    out(f"shortfalls that do not free enough room    : {shortfall_failures} (must be 0)")
    out(f"shortfalls after which a re-solve selects  : {shortfall_resolves} of "
        f"{shortfall_checks}")
    out("the two counts measure different claims. The first is the claim the report")
    out("makes: adding the stated seconds to the stated window frees enough room for")
    out("this product alongside everything already committed to that window, and it must")
    out("hold. The second is a question the report does not answer: re-solving with the")
    out("extra room may spend it on a different product of higher value, so a product")
    out("that fits is not a product that is chosen. The gap between the two counts is")
    out("the honest limit of a per-product shortfall number.")

    out.rule("7. the best window to extend, and the case where none pays")
    instance = generate_instance(3, n_products=20, n_windows=3, capacity_fraction=0.4)
    problem = _problem(instance)
    marginals = marginal_value_of_contact(problem, delta_s=60.0)
    best = best_window_to_extend(marginals)
    out(f"tight instance  : best window {best.window if best else 'none'}, "
        f"{best.value_per_second:.6f} value/s" if best else "tight instance  : none pays")
    roomy_plan = ContactPlan(
        ContactWindow(
            name=window.name,
            start_s=window.start_s,
            end_s=window.end_s,
            rate_breakpoints=tuple((t, rate * 50.0) for t, rate in window.rate_breakpoints),
            reliability=window.reliability,
            station=window.station,
        )
        for window in problem.plan
    )
    roomy = problem.with_plan(roomy_plan)
    roomy_marginals = marginal_value_of_contact(roomy, delta_s=60.0)
    roomy_best = best_window_to_extend(roomy_marginals)
    out(f"ample capacity  : best window "
        f"{roomy_best.window if roomy_best else 'none pays'}")
    out(f"selected with ample capacity: {len(exact_select(roomy).assignments)} of "
        f"{len(problem.inventory)}")
    out("with every rate multiplied by 50 the binding constraint is expiry and not")
    out("capacity, and the correct answer to 'should I ask for another minute' is no.")
    if roomy_best is not None:
        failures += 1
        out(f"  FAIL extra capacity still pays on a plan with 50x the rate: "
            f"{roomy_best.window} at {roomy_best.value_per_second:.6e} value/s")

    out.rule("8. the pinned seed-36 sweep, re-measured")
    out("the two points either side of the drop, printed so that the numbers quoted in")
    out("README.md and validation/VALIDATION.md come from a committed output file")
    out("")
    instance = generate_instance(36, n_products=20, n_windows=3, capacity_fraction=0.4)
    drop_problem = _problem(instance)
    drop_points = capacity_sweep(drop_problem, "pass1", (440.0, 460.0))
    for point in drop_points:
        out(
            f"  +{point.extra_s:6.1f} s  value {point.total_value!r}  "
            f"selected {point.n_selected}  expiry crossings {point.n_expiry_crossings}"
        )
    change = drop_points[1].total_value - drop_points[0].total_value
    out(f"  change           : {change!r}")
    out(f"  relative drop    : {-change / drop_points[0].total_value!r}")
    if change >= 0.0:
        failures += 1
        out("  FAIL the pinned drop no longer reproduces")
    if drop_points[1].n_expiry_crossings <= drop_points[0].n_expiry_crossings:
        failures += 1
        out("  FAIL the drop is no longer explained by an expiry crossing")

    out.rule("9. greedy and exact price the same extension differently")
    out("the marginal value of contact time is a property of the selector as well as of")
    out("the plan, and an operator running greedy should be quoted greedy's price")
    out("")
    out(f"{'seed':>6s} {'window':>8s} {'greedy value/s':>15s} {'exact value/s':>14s}")
    for seed in (1, 2, 3):
        instance = generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.4)
        problem = _problem(instance)
        greedy_prices = marginal_value_of_contact(
            problem, selector=greedy_select, delta_s=60.0
        )
        exact_prices = marginal_value_of_contact(problem, selector=exact_select, delta_s=60.0)
        for g, e in zip(greedy_prices, exact_prices, strict=True):
            out(f"{seed:6d} {g.window:>8s} {g.value_per_second:15.6f} "
                f"{e.value_per_second:14.6f}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
