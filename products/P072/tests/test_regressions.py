"""Pinned counterexamples, each with its literal inputs in the test body.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis or a validation script found
during this build is reproduced here as an explicit named test, so that the
defect it exposed cannot come back silently.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from downlinkpick.contacts import (
    ROUNDING_TOLERANCE_REL,
    ContactPlan,
    ContactWindow,
    constant_rate_profile,
    feasibility_slack_bits,
    rate_profile_from_elevation,
)
from downlinkpick.exact import (
    ROUNDED_STATUS,
    SOLVER_ABS_GAP_VALUE_UNITS,
    enumerate_select,
    exact_select,
)
from downlinkpick.greedy import dantzig_bound, greedy_select
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.marginal import capacity_sweep, marginal_value_of_contact
from downlinkpick.selection import SelectionProblem, check_invariants, sizes_from_ratios


def test_regression_rate_step_is_integrated_per_segment() -> None:
    """``validation/validate_capacity.py``, first run: a 6.065e-02 relative error.

    The closed-form capacity integral read the rate at each segment's endpoints
    back through :meth:`ContactWindow.rate_bps`, which is right-continuous at a
    rate step. For the segment *ending* at a step, that returned the step's
    higher value and over-integrated the segment. The profile below is the one
    that failed, with two breakpoints sharing t = 111.5 s.
    """
    breakpoints = (
        (0.0, 1.0e7),
        (37.0, 2.3e8),
        (111.5, 1.1e8),
        (111.5, 2.6e8),
        (260.0, 3.05e8),
        (400.0, 7.0e7),
        (431.0, 1.0e7),
    )
    window = ContactWindow("probe", 0.0, 431.0, breakpoints)
    closed = window.capacity_bits()
    integrated, _error = quad(window.rate_bps, 0.0, 431.0, limit=500)
    # The value the buggy version produced was 9.2133750000e+10; the correct
    # integral is 8.6546250000e+10.
    assert closed == pytest.approx(8.654625e10, rel=1e-12)
    assert closed != pytest.approx(9.213375e10, rel=1e-6)
    assert integrated == pytest.approx(closed, rel=1e-7)


def test_regression_unscaled_capacity_rows_made_highs_report_a_wrong_optimum() -> None:
    """``validation/validate_exact.py``, first run: 12 of 240 wrong optima.

    With the capacity rows left in bits -- coefficients of order 1e9 against
    objective coefficients of order 1e1 -- HiGHS returned ``status == 0`` with an
    objective below the true optimum on 12 of 240 seeded small instances, and a
    hard solve error on one. The rows are now divided by the window capacity.
    Seed 69 of that family had the worst disagreement, 4.568e-01 relative.
    """
    from downlinkpick.dataset import generate_instance

    instance = generate_instance(69, n_products=7, n_windows=2, capacity_fraction=0.4)
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    problem = SelectionProblem(
        instance.inventory,
        instance.plan,
        values,
        sizes_from_ratios(instance.inventory, instance.true_ratios),
    )
    solved = exact_select(problem)
    enumerated = enumerate_select(problem)
    assert solved.status == "optimal"
    assert solved.total_value == pytest.approx(enumerated.total_value, rel=1e-9)
    # The pre-fix MILP returned 7.3013203604977335 where the optimum is
    # 13.441203829952284.
    assert solved.total_value == pytest.approx(13.441203829952284, rel=1e-9)


def test_regression_absolute_capacity_tolerance_split_the_two_solvers() -> None:
    """Hypothesis, ``test_milp_agrees_with_enumeration``: ``[(1.0, 9.0)]``, capacity 8.0.

    One product of 9 bits against a window of 8 bits. The exhaustive
    enumeration allowed an absolute one-bit overflow, so ``9 > 8 + 1`` was false
    and it called the selection feasible, returning an optimum of 1.0; the MILP,
    whose feasibility test is relative to the scaled row, returned 0. The
    feasibility slack is now relative, so both return 0.
    """
    inventory = Inventory(
        [
            DataProduct(
                "p0",
                raw_bytes=9.0 / 8.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=1.0,
                compressibility=1.0,
            )
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 8.0 / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert problem.capacity_bits()["w"] == pytest.approx(8.0, rel=1e-12)
    assert problem.sizes_bits["p0"] == pytest.approx(9.0, rel=1e-12)
    assert feasibility_slack_bits(8.0) == pytest.approx(8.0 * ROUNDING_TOLERANCE_REL)
    assert exact_select(problem).total_value == pytest.approx(0.0, abs=1e-12)
    assert enumerate_select(problem).total_value == pytest.approx(0.0, abs=1e-12)
    assert problem.unreachable() == ("p0",)


def test_regression_a_product_that_exactly_fills_the_window_is_selected() -> None:
    """The other side of the tolerance boundary, which must stay feasible.

    A product of exactly 8 bits in a window of exactly 8 bits must be selected
    by both routes. Tightening the feasibility slack from one absolute bit to a
    relative 1e-12 must not have made an exact fit infeasible.
    """
    inventory = Inventory(
        [
            DataProduct(
                "p0",
                raw_bytes=1.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=1.0,
                compressibility=1.0,
            )
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 8.0 / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert problem.sizes_bits["p0"] == pytest.approx(8.0, rel=1e-12)
    assert exact_select(problem).selected_set == {"p0"}
    assert enumerate_select(problem).selected_set == {"p0"}


def test_regression_extending_a_window_past_an_expiry_lowers_the_optimum() -> None:
    """``validation/validate_marginal.py``, first run: a 12.89 per cent drop.

    The capacity sweep of seed 36 fell from 40.146248887935315 to
    34.97126699929167 between a 440-second and a 460-second extension of pass1.
    This is not a solver failure. A window's end is also its delivery epoch, so
    the longer pass is too late for a product whose expiry lies inside the
    extension, and that product's columns leave the model. The marginal value of
    contact time is therefore allowed to be negative, and
    :attr:`downlinkpick.marginal.WindowMarginal.expiry_crossings` reports why.
    """
    from downlinkpick.dataset import generate_instance

    instance = generate_instance(36, n_products=20, n_windows=3, capacity_fraction=0.4)
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    problem = SelectionProblem(
        instance.inventory,
        instance.plan,
        values,
        sizes_from_ratios(instance.inventory, instance.true_ratios),
    )
    points = capacity_sweep(problem, "pass1", (440.0, 460.0))
    assert points[0].total_value == pytest.approx(40.146248887935315, rel=1e-9)
    assert points[1].total_value == pytest.approx(34.97126699929167, rel=1e-9)
    assert points[1].total_value < points[0].total_value
    # The drop is explained: the 460 s extension crosses an expiry the 440 s one
    # does not.
    assert points[1].n_expiry_crossings > points[0].n_expiry_crossings
    marginals = marginal_value_of_contact(
        problem.with_plan(problem.plan.extended_window("pass1", 440.0)),
        delta_s=20.0,
        windows=("pass1",),
    )
    assert marginals[0].expiry_crossings != ()


def test_regression_extending_an_unknown_window_must_raise() -> None:
    """``tests/test_contacts.py``: a typo silently returned an unchanged plan.

    :meth:`ContactPlan.extended_window` mapped over the windows and extended the
    one whose name matched, so an unknown name extended nothing and returned a
    plan identical to the input. A caller pricing ``"pass4"`` on a three-pass
    plan would have been told the extension was worth zero.
    """
    plan = ContactPlan(
        [ContactWindow("w1", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0e6))]
    )
    with pytest.raises(KeyError, match="no window named 'w4'"):
        plan.extended_window("w4", 10.0)


def test_regression_a_malformed_case_file_is_a_value_error(tmp_path) -> None:
    """``tests/test_cli.py``: a PyYAML ParserError reached the user as a traceback.

    ``load_case`` let ``yaml.YAMLError`` propagate, so pointing the CLI at a
    non-YAML file produced a traceback and a non-2 exit status instead of a
    message. The parser error is now translated to :class:`ValueError`.
    """
    from downlinkpick.cases import load_case

    target = tmp_path / "not_yaml.toml"
    target.write_text('[build-system]\nrequires = ["setuptools>=68"]\n')
    with pytest.raises(ValueError, match="not parseable as YAML"):
        load_case(target)


def test_regression_the_invariant_checker_allows_a_solvers_own_tolerance() -> None:
    """A selection one part in 1e9 over capacity must not be reported as a defect.

    HiGHS has a primal feasibility tolerance of its own, so a returned solution
    can sit slightly outside a constraint. The checker's capacity slack is
    ``max(1 bit, 1e-6 * capacity)`` for that reason, and a violation an order of
    magnitude larger must still be caught.
    """
    inventory = Inventory(
        [
            DataProduct(
                "p0",
                raw_bytes=1.25e8,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=1.0,
                compressibility=1.0,
            )
        ]
    )
    capacity = 1.0e9
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    oversized = problem.with_sizes({"p0": capacity * (1.0 + 1.0e-9)})
    selection = type(exact_select(problem))(
        exact_select(problem).assignments, exact_select(problem).total_value, "manual"
    )
    rebuilt = oversized.with_values(dict(problem.values), value_source="declared")
    from downlinkpick.selection import Assignment, Selection

    within = Selection(
        (Assignment("p0", "w", capacity * (1.0 + 1.0e-9), 1.0),), 1.0, "manual"
    )
    assert check_invariants(rebuilt, within).ok
    beyond = Selection(
        (Assignment("p0", "w", capacity * (1.0 + 1.0e-3), 1.0),), 1.0, "manual"
    )
    far = rebuilt.with_sizes({"p0": capacity * (1.0 + 1.0e-3)})
    assert not check_invariants(far, beyond).ok
    del selection


def test_regression_additivity_of_the_capacity_integral_at_a_kink() -> None:
    """``validation/validate_capacity.py``: additivity failed at a split of 73.270 s.

    The same root cause as the rate-step defect, exposed through a different
    identity: splitting the irregular profile at 73.270 s gave parts whose sum
    differed from the whole by 2.953e-02 relative.
    """
    breakpoints = (
        (0.0, 1.0e7),
        (37.0, 2.3e8),
        (111.5, 1.1e8),
        (111.5, 2.6e8),
        (260.0, 3.05e8),
        (400.0, 7.0e7),
        (431.0, 1.0e7),
    )
    window = ContactWindow("probe", 0.0, 431.0, breakpoints)
    whole = window.delivered_bits(0.0, 431.0)
    parts = window.delivered_bits(0.0, 73.270) + window.delivered_bits(73.270, 431.0)
    assert parts == pytest.approx(whole, rel=1e-12)


def test_regression_midpoint_reference_error_at_a_step_is_bounded() -> None:
    """The Riemann reference disagreed by 5.200e-07 on the stepped profile.

    That is a property of the reference, not of the library: one sub-interval
    straddles the discontinuity and the midpoint value there is wrong by up to
    the step height. The disagreement must stay inside the derived bound
    ``4 * sum(step heights) * h``, and the tolerance is derived rather than
    raised to fit.
    """
    breakpoints = (
        (0.0, 1.0e7),
        (37.0, 2.3e8),
        (111.5, 1.1e8),
        (111.5, 2.6e8),
        (260.0, 3.05e8),
        (400.0, 7.0e7),
        (431.0, 1.0e7),
    )
    window = ContactWindow("probe", 0.0, 431.0, breakpoints)
    closed = window.capacity_bits()
    points = 400_001
    edges = np.linspace(0.0, 431.0, points)
    centres = 0.5 * (edges[:-1] + edges[1:])
    riemann = float(
        np.sum(np.array([window.rate_bps(float(t)) for t in centres]) * np.diff(edges))
    )
    width = 431.0 / (points - 1)
    step_height = abs(2.6e8 - 1.1e8)
    bound = 4.0 * step_height * width
    assert abs(riemann - closed) <= bound
    assert abs(riemann - closed) / closed == pytest.approx(5.2e-7, rel=0.2)


def test_regression_elevation_profile_needs_keyword_arguments() -> None:
    """``tests/test_properties.py``: the first version passed them positionally.

    ``floor_fraction`` and ``points`` are keyword-only on
    :func:`rate_profile_from_elevation`, and four property tests raised
    ``TypeError`` on their first run because they were passed positionally. The
    signature is pinned here so a future reordering is caught by a test rather
    than by a stack trace in a property run.
    """
    with pytest.raises(TypeError):
        rate_profile_from_elevation(0.0, 100.0, 1.0e6, 0.2, 9)  # type: ignore[misc]
    points = rate_profile_from_elevation(0.0, 100.0, 1.0e6, floor_fraction=0.2, points=9)
    assert len(points) == 9


def test_regression_greedy_may_beat_the_milp_inside_the_solvers_mip_gap() -> None:
    """Hypothesis, ``test_greedy_never_exceeds_the_optimum``.

    Falsifying example: six products of values
    ``[1.0, 0.0, 0.0, 0.0, 0.0, 5.960464477539063e-08]`` and sizes
    ``[9385924, 8, 8, 8, 8, 8]`` bits against a window of 9385942 bits.
    Greedy and the exhaustive enumeration both take the 5.96e-08 product --
    there is room for it -- and HiGHS stops without it, because the improvement
    is below its absolute MIP gap of 1e-6 value units, which
    ``scipy.optimize.milp`` does not expose. The claim "greedy never exceeds the
    MILP" is therefore false as written and true to within the gap, which is
    what :data:`downlinkpick.exact.SOLVER_ABS_GAP_VALUE_UNITS` records.
    """
    sizes = [9385924.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    values = [1.0, 0.0, 0.0, 0.0, 0.0, 5.960464477539063e-08]
    capacity = 9385942.0
    inventory = Inventory(
        [
            DataProduct(
                f"p{index}",
                raw_bytes=size / 8.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=value,
                compressibility=1.0,
            )
            for index, (value, size) in enumerate(zip(values, sizes, strict=True))
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w",
                100.0,
                200.0,
                constant_rate_profile(100.0, 200.0, capacity / 100.0),
                reliability=1.0,
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    greedy_value = greedy_select(problem).total_value
    milp = exact_select(problem)
    enumerated = enumerate_select(problem).total_value

    # Greedy and the enumeration agree, and both exceed the MILP.
    assert greedy_value == pytest.approx(1.0000000596046448, rel=1e-12)
    assert enumerated == pytest.approx(1.0000000596046448, rel=1e-12)
    assert milp.status == "optimal"
    assert milp.total_value == pytest.approx(1.0, rel=1e-12)

    # The excess is real but inside the solver's absolute MIP gap.
    excess = greedy_value - milp.total_value
    assert excess > 0.0
    assert excess == pytest.approx(5.960464477539063e-08, rel=1e-9)
    assert excess <= SOLVER_ABS_GAP_VALUE_UNITS

    # Greedy's selection is genuinely feasible, so this is a solver tolerance
    # and not an invariant violation.
    assert check_invariants(problem, greedy_select(problem)).ok


def test_regression_rounding_a_near_integral_solution_overflowed_a_window() -> None:
    """Hypothesis, ``test_milp_agrees_with_enumeration``: ``[(1.0, 8.0), (1.0, 7999535.0)]``.

    Capacity 7999538 bits; the two products total 7999543 bits, so only one can
    be carried and the optimum is 1.0. HiGHS returned a vector integral only to
    within its own 1e-6 integrality tolerance, this package rounded it at 0.5
    and charged both products in full, and the reported objective was 2.0 with
    the window 5 bits over capacity. The repair in
    :func:`downlinkpick.exact._repair_overflow` now drops the least valuable
    product and the status says the solution was rounded.
    """
    sizes = [8.0, 7999535.0]
    values = [1.0, 1.0]
    capacity = 7999538.0
    inventory = Inventory(
        [
            DataProduct(
                f"p{index}",
                raw_bytes=size / 8.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=value,
                compressibility=1.0,
            )
            for index, (value, size) in enumerate(zip(values, sizes, strict=True))
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    milp = exact_select(problem)
    assert milp.total_value == pytest.approx(1.0, rel=1e-12)
    assert milp.total_value == pytest.approx(
        enumerate_select(problem).total_value, rel=1e-9
    )
    assert milp.status == ROUNDED_STATUS
    assert check_invariants(problem, milp).ok
    assert len(milp.assignments) == 1


def test_regression_a_rounded_solution_exceeded_dantzigs_fractional_bound() -> None:
    """Hypothesis, ``test_dantzig_bound_dominates_the_optimum``.

    Products ``[(1.0, 8.0), (5.0, 7999609.0)]`` bits against a window of
    7999615 bits. Dantzig's fractional bound is 5.999998749938903 -- the small
    product whole plus 0.99999975 of the large one -- and no feasible integral
    selection can exceed it. The MILP reported 6.0, which is the signature of an
    infeasible rounded solution rather than of a loose bound, and that is how
    this defect was found: a violated upper bound is a stronger signal than a
    violated capacity row, because it cannot be dismissed as a tolerance.
    """
    sizes = [8.0, 7999609.0]
    values = [1.0, 5.0]
    capacity = 7999615.0
    inventory = Inventory(
        [
            DataProduct(
                f"p{index}",
                raw_bytes=size / 8.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=value,
                compressibility=1.0,
            )
            for index, (value, size) in enumerate(zip(values, sizes, strict=True))
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    bound = dantzig_bound(problem)
    milp = exact_select(problem)
    assert bound == pytest.approx(5.999998749938903, rel=1e-12)
    assert milp.total_value == pytest.approx(5.0, rel=1e-12)
    assert milp.total_value <= bound
    assert milp.status == ROUNDED_STATUS
    assert check_invariants(problem, milp).ok


def test_regression_the_repair_never_orphans_a_dependent() -> None:
    """The repair drops a prerequisite together with everything that needs it.

    A window too small for both members of a dependency pair must not come back
    with the dependent selected and its prerequisite dropped. The repair removes
    the victim's whole selected dependent set, so the dependency invariant holds
    whatever the solver returned.
    """
    inventory = Inventory(
        [
            DataProduct(
                "cal", 1.0, 0.0, 1000.0, declared_value=0.1, compressibility=1.0
            ),
            DataProduct(
                "img",
                999999.0,
                0.0,
                1000.0,
                declared_value=9.0,
                compressibility=1.0,
                depends_on=("cal",),
            ),
        ]
    )
    capacity = 8.0 * 1000000.0 - 4.0
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity / 100.0)
            )
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    selection = exact_select(problem)
    report = check_invariants(problem, selection)
    assert report.ok, report.violations
    if "img" in selection.selected_set:
        assert "cal" in selection.selected_set
