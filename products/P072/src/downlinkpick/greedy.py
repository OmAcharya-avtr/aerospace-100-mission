"""Baseline 1: value-density ordering with window packing.

This is the operational default and it is implemented first, before the exact
solver, because it is the thing the exact solver has to beat and because an
operator reading a selection wants to know how far the clever answer is from the
obvious one.

The algorithm, stated exactly as implemented:

1. For each unselected product, form its *bundle*: the product together with
   every member of its transitive dependency closure that is not already
   selected. A product cannot be downlinked usefully without its prerequisites,
   so the unit of decision is the bundle, not the product.
2. Score each bundle by value density, ``sum(values) / sum(sizes)`` in value
   units per bit. Order descending; break ties on bundle size ascending, then on
   the triggering product's inventory position, so the order is total and
   reproducible.
3. Take bundles in that order. For each bundle, attempt two placement rules
   and keep whichever produces the greater added value:

   * **reliability-first** -- place each member, in dependency-topological
     order, into the feasible window with the highest reliability that has room,
     breaking ties on earliest start then on name;
   * **earliest-first** -- the same walk, but preferring the earliest feasible
     window with room.

   Feasible means the window ends at or before the member's expiry and no
   earlier than the window already chosen for any of its prerequisites, which is
   the precedence rule of :mod:`downlinkpick.selection`. Two rules are tried
   because one rule alone is easy to beat and that would make the baseline a
   straw man: reliability-first will park a prerequisite in a late,
   high-reliability window and strand a deadline-constrained dependent, while
   earliest-first will spend early capacity on products that any window would
   have carried. Both are things operators do; taking the better of the two is
   what makes this baseline hard to beat.
4. If both rules fail to place a bundle, roll it back. A partially placed bundle
   would spend capacity on a product nobody can use and would violate the
   dependency invariant.

Why value density: for a single knapsack with divisible items the optimal order
*is* descending value density, and Dantzig's bound (1957) uses exactly that
order. The integral problem is NP-hard, so the ordering is a heuristic, and the
classical worst case for greedy-by-density on a single 0/1 knapsack is a ratio
of 1/2 of the optimum, achieved by one high-density small item blocking one
item that fills the knapsack. ``validation/validate_greedy_gap.py`` measures the
realised gap distribution on a seeded instance family rather than asserting the
worst case; the distribution, not its mean, is the reported result.

Picking the highest-reliability feasible window is locally optimal under
equation (2) of :mod:`downlinkpick.selection`, where reliability multiplies
value, and is not globally optimal: it can fill a high-reliability window with
a product that any window would have carried, leaving a deadline-constrained
product with nowhere to go. That failure mode is real, is visible in the
measured gap distribution, and is the reason the exact solver exists.

References
----------
Dantzig, G. B. (1957), Discrete-variable extremum problems, Operations Research
5(2), 266-277. The linear-relaxation bound obtained by taking items in
descending value density, which is the ordering used here.

Martello, S. and Toth, P. (1990), Knapsack Problems: Algorithms and Computer
Implementations, Wiley. Chapters 2 and 6 for greedy bounds on the 0/1 knapsack
and for the multiple-knapsack problem respectively. Structure only; no numerical
value in this repository is taken from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contacts import feasibility_slack_bits
from .selection import Assignment, Selection, SelectionProblem


@dataclass(frozen=True)
class _Bundle:
    trigger: str
    members: tuple[str, ...]
    value: float
    bits: float
    order: int

    @property
    def density(self) -> float:
        return self.value / self.bits


#: The two placement rules tried for every bundle. ``"reliability"`` prefers the
#: window with the highest reliability, ``"earliest"`` the window that ends
#: soonest; both break ties on the remaining key and then on name, so each rule
#: is a total order and the selector is deterministic.
PLACEMENT_RULES: tuple[str, ...] = ("reliability", "earliest")


def _window_rank(problem: SelectionProblem, window: str, rule: str) -> tuple[float, float, str]:
    """Sort key for the placement rule; smaller sorts first."""
    contact = problem.plan[window]
    if rule == "reliability":
        return (-contact.reliability, contact.start_s, contact.name)
    if rule == "earliest":
        return (contact.start_s, -contact.reliability, contact.name)
    raise ValueError(f"unknown placement rule {rule!r}; expected one of {PLACEMENT_RULES}")


def _place_bundle(
    problem: SelectionProblem,
    bundle: _Bundle,
    placement: dict[str, str],
    remaining: dict[str, float],
    slack: dict[str, float],
    rule: str,
) -> tuple[dict[str, str], dict[str, float]] | None:
    """Try to place every member of ``bundle`` under ``rule``.

    Returns ``(placement delta, bits spent per window)`` or ``None`` if any
    member could not be placed, in which case nothing is spent.
    """
    trial: dict[str, str] = {}
    spent: dict[str, float] = {}
    for member in bundle.members:
        size = problem.sizes_bits[member]
        earliest_end = 0.0
        for prerequisite in problem.inventory[member].depends_on:
            window = trial.get(prerequisite) or placement.get(prerequisite)
            if window is not None:
                earliest_end = max(earliest_end, problem.plan[window].end_s)
        candidates = sorted(
            (
                w
                for w in problem.feasible_windows(member)
                if problem.plan[w].end_s >= earliest_end
            ),
            key=lambda w: _window_rank(problem, w, rule),
        )
        chosen: str | None = None
        for window in candidates:
            free = remaining[window] - spent.get(window, 0.0)
            if size <= free + slack[window]:
                chosen = window
                break
        if chosen is None:
            return None
        trial[member] = chosen
        spent[chosen] = spent.get(chosen, 0.0) + size
    return trial, spent


def greedy_select(problem: SelectionProblem) -> Selection:
    """Select by descending bundle value density with first-fit window packing.

    Parameters
    ----------
    problem:
        The selection problem. Values are in value units, sizes in bits.

    Returns
    -------
    Selection
        With ``method="greedy"`` and ``status="feasible"``. Greedy proves
        nothing about optimality, so ``status`` is never ``"optimal"`` and
        ``bound`` is left ``None``.

    Notes
    -----
    Deterministic: the result depends only on the problem, never on dictionary
    iteration order or on wall-clock time. Complexity is
    ``O(n^2 (log n + m log m))`` for ``n`` products and ``m`` windows, dominated
    by re-scoring and re-placing bundles after each placement.
    """
    inventory = problem.inventory
    topological = {name: index for index, name in enumerate(inventory.topological_order())}
    capacity = problem.capacity_bits()
    remaining = dict(capacity)
    slack = {name: feasibility_slack_bits(capacity[name]) for name in problem.plan.names}
    placement: dict[str, str] = {}

    while True:
        bundles: list[_Bundle] = []
        for order, name in enumerate(inventory.names):
            if name in placement:
                continue
            members = {name, *inventory.dependency_closure(name)} - set(placement)
            value = sum(float(problem.values[member]) for member in members)
            bits = sum(problem.sizes_bits[member] for member in members)
            if bits <= 0.0:
                continue
            ordered_members = tuple(sorted(members, key=lambda m: topological[m]))
            bundles.append(_Bundle(name, ordered_members, value, bits, order))
        if not bundles:
            break

        bundles.sort(key=lambda b: (-b.density, b.bits, b.order))
        progressed = False
        for bundle in bundles:
            if bundle.value <= 0.0:
                continue
            best: tuple[float, dict[str, str], dict[str, float]] | None = None
            for rule in PLACEMENT_RULES:
                attempt = _place_bundle(problem, bundle, placement, remaining, slack, rule)
                if attempt is None:
                    continue
                trial, spent = attempt
                added = sum(problem.pair_value(m, w) for m, w in trial.items())
                if best is None or added > best[0]:
                    best = (added, trial, spent)
            if best is None:
                continue
            _, trial, spent = best
            placement.update(trial)
            for window, bits in spent.items():
                remaining[window] -= bits
            progressed = True
            break
        if not progressed:
            break

    assignments = tuple(
        Assignment(
            product=name,
            window=placement[name],
            bits=problem.sizes_bits[name],
            value=problem.pair_value(name, placement[name]),
        )
        for name in sorted(
            placement,
            key=lambda n: (problem.plan[placement[n]].start_s, inventory.position(n)),
        )
    )
    return Selection(
        assignments=assignments,
        total_value=sum(a.value for a in assignments),
        method="greedy",
        status="feasible",
    )


def density_order(problem: SelectionProblem) -> tuple[str, ...]:
    """Product names in descending single-product value density.

    Exposed because it is the order an operator writes on a whiteboard, and
    because the first divergence between it and the exact solver's selection is
    the most informative line in a comparison report.
    """
    return tuple(
        sorted(
            problem.inventory.names,
            key=lambda n: (
                -float(problem.values[n]) / problem.sizes_bits[n],
                problem.sizes_bits[n],
                problem.inventory.position(n),
            ),
        )
    )


def dantzig_bound(problem: SelectionProblem) -> float:
    """Dantzig's fractional upper bound on the objective, in value units.

    Fills one aggregate knapsack of capacity ``sum_w C_w`` with items of value
    ``v_i * max_w p_w`` over the windows feasible for ``i``, in descending
    density, allowing the last item to be split. This ignores the per-window
    capacity split and the dependency rows, so it is a valid upper bound on the
    integral optimum, and a loose one: ``validation/validate_greedy_gap.py``
    reports how loose against the proved optimum.
    """
    total_capacity = problem.plan.total_capacity_bits()
    items: list[tuple[float, float]] = []
    for name in problem.inventory.names:
        windows = problem.feasible_windows(name)
        if not windows:
            continue
        best = max(problem.plan[w].reliability for w in windows)
        value = float(problem.values[name]) * best
        bits = problem.sizes_bits[name]
        items.append((value, bits))
    items.sort(key=lambda item: -item[0] / item[1])
    bound = 0.0
    room = total_capacity
    for value, bits in items:
        if bits <= room:
            bound += value
            room -= bits
        else:
            bound += value * (room / bits)
            break
    return bound
