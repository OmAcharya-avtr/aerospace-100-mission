"""The selection problem, the selection, and the invariants a selection must hold.

The problem is a multiple-knapsack assignment with side constraints. Given

* a set of products ``i`` with estimated downlink size ``s_i`` bits and base
  value ``v_i`` value units,
* a set of windows ``w`` with capacity ``C_w`` bits and reliability ``p_w``,

the realised value of assigning product ``i`` to window ``w`` is

    V(i, w) = v_i * p_w   if  end(w) <= expiry(i),  otherwise infeasible.    (2)

Selecting ``x_iw`` in ``{0, 1}`` maximises ``sum_iw V(i, w) x_iw`` subject to

    sum_w x_iw <= 1                      for every product i   (no double downlink)
    sum_i s_i x_iw <= C_w                for every window w    (capacity)
    x_iw <= sum_{w': end(w') <= end(w)} x_jw'
                                         for every feasible (i, w) and every
                                         direct prerequisite j of i (precedence)

The precedence row is the dependency constraint and it is stronger than set
membership: a prerequisite must be delivered in the same window or an earlier
one, never later. A calibration frame that lands on tomorrow's pass does not
help today's image, so the weaker "both are selected somewhere" reading would
produce selections that are infeasible on the ground. Because window ends are a
total order, imposing the row on *direct* prerequisites implies it for the whole
transitive closure, which is proved by induction on the closure depth and is
checked over the seeded instance family in ``validation/validate_invariants.py``.

Equation (2) is a modelling decision with consequences and both parts are
stated because neither is forced by the physics:

* **Reliability multiplies value.** The objective is therefore an *expected*
  value under the assumption that window outcomes are independent of what was
  packed into them, and that a failed window delivers nothing rather than part
  of its queue. Both are approximations. There is no re-planning after a failed
  pass anywhere in this package.
* **Expiry is a hard feasibility cut, not a decay.** A product delivered one
  second after its deadline is worth zero, not nearly its full value. This is
  right for a deadline and wrong for a decay, and a decay model would change
  every number this package reports.
* **Precedence is by window, not by time within a window.** Two products in the
  same window satisfy precedence in either transmission order. Ordering the
  queue inside a pass is not modelled.

The two reference implementations -- greedy in :mod:`downlinkpick.greedy` and
exact in :mod:`downlinkpick.exact` -- both consume a :class:`SelectionProblem`
and both produce a :class:`Selection`, so that every comparison between them is
over the same arithmetic and the same feasibility definition. The invariant
checker in this module is run against the output of both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contacts import ContactPlan, check_slack_bits, feasibility_slack_bits
from .inventory import MIN_COMPRESSION_RATIO, Inventory

#: Relative tolerance used when checking that the reported total value equals
#: the sum of its per-assignment parts. Value sums are O(1e2) over O(1e2) terms,
#: so this is several orders of magnitude above the accumulated rounding and far
#: below any difference that would matter to a selection.
VALUE_TOLERANCE_REL = 1.0e-9


@dataclass(frozen=True)
class Assignment:
    """One product assigned to one window.

    Attributes
    ----------
    product:
        Product name.
    window:
        Window name.
    bits:
        Estimated size on the link in bits, as used by the capacity constraint.
    value:
        Realised value in value units, equation (2).
    """

    product: str
    window: str
    bits: float
    value: float


@dataclass(frozen=True)
class Selection:
    """A set of assignments with its accounting.

    Attributes
    ----------
    assignments:
        Assignments in window order then product order.
    total_value:
        Sum of :attr:`Assignment.value`, in value units.
    method:
        Which selector produced this, for reporting.
    status:
        Solver status string. ``"optimal"`` is only ever set by a selector that
        proved optimality.
    bound:
        An upper bound on the optimal objective in value units where one is
        available (the LP relaxation, or the proved optimum), else ``None``.
    """

    assignments: tuple[Assignment, ...]
    total_value: float
    method: str
    status: str = "feasible"
    bound: float | None = None

    @property
    def selected(self) -> tuple[str, ...]:
        """Selected product names, in assignment order."""
        return tuple(a.product for a in self.assignments)

    @property
    def selected_set(self) -> frozenset[str]:
        """Selected product names as a set."""
        return frozenset(a.product for a in self.assignments)

    @property
    def window_of(self) -> dict[str, str]:
        """Window name per selected product."""
        return {a.product: a.window for a in self.assignments}

    def bits_by_window(self) -> dict[str, float]:
        """Committed bits per window name."""
        totals: dict[str, float] = {}
        for assignment in self.assignments:
            totals[assignment.window] = totals.get(assignment.window, 0.0) + assignment.bits
        return totals

    def value_by_window(self) -> dict[str, float]:
        """Realised value per window name, in value units."""
        totals: dict[str, float] = {}
        for assignment in self.assignments:
            totals[assignment.window] = totals.get(assignment.window, 0.0) + assignment.value
        return totals

    def total_bits(self) -> float:
        """Sum of committed bits."""
        return sum(a.bits for a in self.assignments)


@dataclass(frozen=True)
class SelectionProblem:
    """Inventory, contact plan, value estimates and size estimates in one object.

    ``values`` and ``sizes_bits`` are supplied explicitly rather than read off
    the products so that the same instance can be solved against a declared
    value, a learned value and a true value with nothing else changing. That is
    the whole mechanism by which :mod:`downlinkpick.benchmark` attributes a
    difference in realised value to the value source and not to the solver.
    """

    inventory: Inventory
    plan: ContactPlan
    values: Mapping[str, float]
    sizes_bits: Mapping[str, float]
    value_source: str = "declared"

    def __post_init__(self) -> None:
        missing_value = [n for n in self.inventory.names if n not in self.values]
        if missing_value:
            raise ValueError(
                "no value supplied for product(s): " + ", ".join(sorted(missing_value))
            )
        missing_size = [n for n in self.inventory.names if n not in self.sizes_bits]
        if missing_size:
            raise ValueError(
                "no size supplied for product(s): " + ", ".join(sorted(missing_size))
            )
        for name in self.inventory.names:
            if self.values[name] < 0.0:
                raise ValueError(f"{name}: value must be non-negative, got {self.values[name]!r}")
            if not (self.sizes_bits[name] > 0.0):
                raise ValueError(
                    f"{name}: size must be strictly positive bits, got {self.sizes_bits[name]!r}"
                )

    @classmethod
    def from_declared(
        cls, inventory: Inventory, plan: ContactPlan, *, value_source: str = "declared"
    ) -> SelectionProblem:
        """Build a problem from declared values and measured compressibility.

        Raises :class:`ValueError` naming every product whose declared value or
        compressibility is ``None``, because silently substituting a default for
        a missing declaration is the failure mode this package exists to expose.
        """
        undeclared = inventory.missing_declarations()
        if undeclared:
            raise ValueError(
                "declared_value is None for: "
                + ", ".join(undeclared)
                + "; estimate it with downlinkpick.valuemodel or supply values explicitly"
            )
        unmeasured = inventory.missing_compressibility()
        if unmeasured:
            raise ValueError(
                "compressibility is None for: "
                + ", ".join(unmeasured)
                + "; estimate it with downlinkpick.compress or supply sizes explicitly"
            )
        values = {p.name: float(p.declared_value or 0.0) for p in inventory}
        sizes = {p.name: p.downlink_bits() for p in inventory}
        return cls(inventory, plan, values, sizes, value_source=value_source)

    def with_values(
        self, values: Mapping[str, float], *, value_source: str
    ) -> SelectionProblem:
        """A copy using different value estimates."""
        return SelectionProblem(
            self.inventory, self.plan, dict(values), dict(self.sizes_bits), value_source
        )

    def with_sizes(self, sizes_bits: Mapping[str, float]) -> SelectionProblem:
        """A copy using different size estimates."""
        return SelectionProblem(
            self.inventory, self.plan, dict(self.values), dict(sizes_bits), self.value_source
        )

    def with_plan(self, plan: ContactPlan) -> SelectionProblem:
        """A copy over a different contact plan."""
        return SelectionProblem(
            self.inventory, plan, dict(self.values), dict(self.sizes_bits), self.value_source
        )

    def feasible_windows(self, name: str) -> tuple[str, ...]:
        """Windows that end at or before the product's expiry, in time order."""
        expiry = self.inventory[name].expiry_s
        return self.plan.windows_ending_by(expiry)

    def pair_value(self, name: str, window: str) -> float:
        """Realised value of assigning ``name`` to ``window``, equation (2).

        Returns ``0.0`` for an infeasible pair rather than raising, so that a
        caller scoring an externally supplied selection gets the honest value of
        a late delivery, which is nothing.
        """
        if window not in self.plan:
            raise KeyError(f"no window named {window!r} in contact plan")
        if self.plan[window].end_s > self.inventory[name].expiry_s:
            return 0.0
        return float(self.values[name]) * self.plan[window].reliability

    def feasible_pairs(self) -> tuple[tuple[str, str], ...]:
        """All ``(product, window)`` pairs permitted by expiry, in model order."""
        return tuple(
            (name, window)
            for name in self.inventory.names
            for window in self.feasible_windows(name)
        )

    def capacity_bits(self) -> dict[str, float]:
        """Per-window capacity in bits."""
        return self.plan.capacity_bits()

    def unreachable(self) -> tuple[str, ...]:
        """Products that no selector can ever select, with any capacity.

        A product is unreachable when it has no window ending before its expiry,
        when its size exceeds the capacity of every such window, or when any
        member of its dependency closure is itself unreachable.
        """
        capacity = self.capacity_bits()
        direct: set[str] = set()
        for name in self.inventory.names:
            windows = self.feasible_windows(name)
            if not windows:
                direct.add(name)
                continue
            size = self.sizes_bits[name]
            if all(
                size > capacity[w] + feasibility_slack_bits(capacity[w]) for w in windows
            ):
                direct.add(name)
        blocked = set(direct)
        for name in self.inventory.names:
            if self.inventory.dependency_closure(name) & direct:
                blocked.add(name)
        return tuple(n for n in self.inventory.names if n in blocked)

    def score(self, selection: Selection) -> float:
        """Realised value of ``selection`` under *this* problem's values.

        This is how a selection made with one value source is scored against
        another: build the problem with the true values and call this on the
        selection produced from the estimated ones.
        """
        return sum(
            self.pair_value(a.product, a.window) for a in selection.assignments
        )

    def total_value_available(self) -> float:
        """Sum of all product values ignoring every constraint, in value units."""
        return sum(float(self.values[n]) for n in self.inventory.names)


@dataclass(frozen=True)
class InvariantReport:
    """Result of :func:`check_invariants`.

    Attributes
    ----------
    violations:
        Human-readable messages, empty when the selection is valid.
    worst_capacity_overshoot_bits:
        Largest ``committed - capacity`` over windows, in bits. Negative when
        every window has room left.
    value_residual_rel:
        ``|reported total - recomputed total| / max(1, |recomputed total|)``.
    """

    violations: tuple[str, ...]
    worst_capacity_overshoot_bits: float
    value_residual_rel: float

    @property
    def ok(self) -> bool:
        """True when no violation was found."""
        return not self.violations


def check_invariants(
    problem: SelectionProblem,
    selection: Selection,
    *,
    capacity_tolerance_bits: float | None = None,
    value_tolerance_rel: float = VALUE_TOLERANCE_REL,
) -> InvariantReport:
    """Check the five invariants every selection must satisfy.

    ``capacity_tolerance_bits`` defaults to ``None``, which uses
    :func:`downlinkpick.contacts.check_slack_bits` per window: an absolute floor
    of one bit, widening to 1e-6 of the capacity, because a branch-and-bound
    solver has a primal feasibility tolerance of its own and reporting it as a
    defect in this package would be wrong.

    The invariants are: no product appears twice (no double downlink); every
    product and window named exists; no window is over capacity beyond the
    capacity tolerance; no assignment delivers a product after its
    expiry; every selected product's dependency closure is also selected and
    every prerequisite is delivered in the same window or earlier (precedence);
    and the reported total value agrees with the sum of its parts to within
    ``value_tolerance_rel``.

    The value check is a relative comparison with a two-sided tolerance, not an
    equality: the total is a sum of products of floats and reassociating that
    sum changes its last bits. What is guaranteed is agreement to within
    rounding, not bit-exact equality.
    """
    violations: list[str] = []
    seen: set[str] = set()
    for assignment in selection.assignments:
        if assignment.product in seen:
            violations.append(
                f"double downlink: {assignment.product!r} assigned more than once"
            )
        seen.add(assignment.product)
        if assignment.product not in problem.inventory:
            violations.append(f"unknown product in selection: {assignment.product!r}")
            continue
        if assignment.window not in problem.plan:
            violations.append(f"unknown window in selection: {assignment.window!r}")
            continue
        product = problem.inventory[assignment.product]
        window = problem.plan[assignment.window]
        if window.end_s > product.expiry_s:
            violations.append(
                f"expiry violated: {assignment.product!r} in window {assignment.window!r} "
                f"ends at {window.end_s:.3f} s, expiry {product.expiry_s:.3f} s"
            )

    capacity = problem.capacity_bits()
    committed = selection.bits_by_window()
    worst = -float("inf")
    for name, limit in capacity.items():
        used = committed.get(name, 0.0)
        worst = max(worst, used - limit)
        slack = (
            check_slack_bits(limit)
            if capacity_tolerance_bits is None
            else capacity_tolerance_bits
        )
        if used > limit + slack:
            violations.append(
                f"capacity violated: window {name!r} committed {used:.6e} bits against a "
                f"capacity of {limit:.6e} bits"
            )
    if worst == -float("inf"):
        worst = -min(capacity.values()) if capacity else 0.0

    selected = selection.selected_set
    window_of = selection.window_of
    for name in sorted(selected):
        if name not in problem.inventory:
            continue
        unmet = problem.inventory.dependency_closure(name) - selected
        if unmet:
            violations.append(
                f"dependency unmet: {name!r} selected without " + ", ".join(sorted(unmet))
            )
        own = window_of.get(name)
        if own is None or own not in problem.plan:
            continue
        own_end = problem.plan[own].end_s
        for prerequisite in sorted(problem.inventory[name].depends_on):
            other = window_of.get(prerequisite)
            if other is None or other not in problem.plan:
                continue
            if problem.plan[other].end_s > own_end:
                violations.append(
                    f"precedence violated: {name!r} delivered in window {own!r} ending "
                    f"{own_end:.3f} s before its prerequisite {prerequisite!r} in window "
                    f"{other!r} ending {problem.plan[other].end_s:.3f} s"
                )

    recomputed = sum(a.value for a in selection.assignments)
    residual = abs(selection.total_value - recomputed) / max(1.0, abs(recomputed))
    if residual > value_tolerance_rel:
        violations.append(
            f"value not conserved: reported {selection.total_value!r} against recomputed "
            f"{recomputed!r}, relative residual {residual:.3e}"
        )

    for assignment in selection.assignments:
        if assignment.product not in problem.inventory:
            continue
        if assignment.window not in problem.plan:
            continue
        expected = problem.pair_value(assignment.product, assignment.window)
        if abs(assignment.value - expected) > value_tolerance_rel * max(1.0, abs(expected)):
            violations.append(
                f"value mismatch: {assignment.product!r} in {assignment.window!r} reported "
                f"{assignment.value!r} against equation (2) value {expected!r}"
            )
        size = problem.sizes_bits[assignment.product]
        size_slack = (
            check_slack_bits(size) if capacity_tolerance_bits is None else capacity_tolerance_bits
        )
        if abs(assignment.bits - size) > size_slack:
            violations.append(
                f"size mismatch: {assignment.product!r} reported {assignment.bits!r} bits "
                f"against the problem's {size!r} bits"
            )

    return InvariantReport(tuple(violations), worst, residual)


def sizes_from_ratios(inventory: Inventory, ratios: Mapping[str, float]) -> dict[str, float]:
    """Downlink sizes in bits from per-product compression ratios.

    Raises :class:`ValueError` on a ratio outside
    ``[MIN_COMPRESSION_RATIO, 1.0]``, which is how a mis-scaled estimator (a
    percentage rather than a fraction, or a compression *factor* rather than a
    ratio) is caught at the boundary instead of silently shrinking the queue.
    """
    sizes: dict[str, float] = {}
    for product in inventory:
        if product.name not in ratios:
            raise ValueError(f"no compression ratio supplied for {product.name!r}")
        ratio = float(ratios[product.name])
        if not (MIN_COMPRESSION_RATIO <= ratio <= 1.0):
            raise ValueError(
                f"{product.name}: compression ratio must lie in "
                f"[{MIN_COMPRESSION_RATIO}, 1.0], got {ratio!r}"
            )
        sizes[product.name] = product.downlink_bits(ratio)
    return sizes
