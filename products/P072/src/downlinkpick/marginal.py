"""What the choice cost: the price of the next second of contact, and what was left.

A selection on its own is an instruction. A selection with a price attached is
an argument, and the two numbers an operator actually needs are:

* **the marginal value of the next unit of contact time**, which answers "is it
  worth asking the network for another minute, and on which pass"; and
* **the identity of what was left behind, with the reason**, which answers "what
  did I just decide not to have, and could I have had it".

Neither is a dual variable. The problem is an integer program, so the LP dual
exists only for the relaxation and generally misprices the integral problem: the
relaxation can hand back a fractional product that the integral problem cannot
take. What is computed here instead is a **finite difference on the integral
optimum**. The window is extended by ``delta_s`` seconds, the capacity integral
over the extension is added, the problem is re-solved by the same selector, and
the change in objective is divided by ``delta_s``. That is a measurable,
reproducible number with an explicit step size, and because the integral
objective is a step function of capacity it is *not* a derivative: the marginal
value at a step is zero up to the step and a jump at it, which is why
:func:`capacity_sweep` exists and why ``validation/validate_marginal.py``
reports the non-concavity it measures instead of asserting concavity.

The extension holds the window's final rate, as documented on
:meth:`downlinkpick.contacts.ContactWindow.extended`. A real extension follows
the pass down and delivers less, so the marginal value reported here is an
upper estimate of what an extension would actually buy. That bias is stated in
README.md under Limitations and is the single largest modelling caveat on the
marginal number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contacts import feasibility_slack_bits
from .exact import exact_select
from .selection import Selection, SelectionProblem

#: Default finite-difference step for the marginal value, in seconds. One second
#: of a 100 Mbit/s downlink is 1e8 bits, which is comparable to one compressed
#: product in the shipped cases, so the step is small enough to be local and
#: large enough to cross a step boundary when one is near.
DEFAULT_DELTA_S = 1.0

Selector = Callable[[SelectionProblem], Selection]


@dataclass(frozen=True)
class WindowMarginal:
    """Finite-difference price of extra contact time on one window.

    Attributes
    ----------
    window:
        Window name.
    delta_s:
        Step used, in seconds.
    delta_bits:
        Capacity added by the step, in bits. This is the integral of the held
        final rate over the extension, not ``rate * delta_s`` for the peak rate.
    base_value, extended_value:
        Objective before and after, in value units.
    value_per_second:
        ``(extended_value - base_value) / delta_s``, in value units per second.
    value_per_gbit:
        ``(extended_value - base_value) / (delta_bits / 1e9)``, in value units
        per gigabit, or ``None`` when ``delta_bits`` is zero.
    entering, leaving:
        Products that joined and left the selection across the step.
    expiry_crossings:
        Products whose expiry falls inside the extension, so that the extended
        window is too late for them and they lose it as a candidate. A non-empty
        tuple here is the explanation for a zero or negative price: the step
        bought capacity and spent feasibility.
    """

    window: str
    delta_s: float
    delta_bits: float
    base_value: float
    extended_value: float
    value_per_second: float
    value_per_gbit: float | None
    entering: tuple[str, ...]
    leaving: tuple[str, ...]
    expiry_crossings: tuple[str, ...] = ()

    @property
    def delta_value(self) -> float:
        """Objective gain from the step, in value units."""
        return self.extended_value - self.base_value


@dataclass(frozen=True)
class LeftBehind:
    """One product that was not selected, with the reason and the shortfall.

    Attributes
    ----------
    product:
        Product name.
    value:
        The product's value under the problem's value estimates, in value units.
    bits:
        Its estimated size on the link, in bits.
    reason:
        One of :data:`LEFT_BEHIND_REASONS`.
    best_window:
        The feasible window with the smallest shortfall, or ``None`` when no
        window is feasible.
    extra_bits_needed:
        Additional capacity in ``best_window`` that would have admitted the
        product, in bits. ``0.0`` when the reason is not capacity.
    extra_seconds_needed:
        ``extra_bits_needed`` converted to seconds at the window's final rate,
        which is the rate an extension would run at. ``inf`` when the final rate
        is zero.
    """

    product: str
    value: float
    bits: float
    reason: str
    best_window: str | None
    extra_bits_needed: float
    extra_seconds_needed: float


#: The four reasons a product can be left behind, in order of how early they
#: are decided. ``expired`` and ``oversized`` are properties of the instance and
#: no amount of re-planning changes them; ``dependency`` means a prerequisite is
#: itself unreachable; ``capacity`` is the only one an extra second of contact
#: can fix.
LEFT_BEHIND_REASONS: tuple[str, ...] = ("expired", "oversized", "dependency", "capacity")


def left_behind_report(
    problem: SelectionProblem, selection: Selection
) -> tuple[LeftBehind, ...]:
    """Every unselected product, with its reason and its shortfall.

    Ordered by value descending then by inventory position, so the first line is
    the most expensive thing the selection gave up.
    """
    selected = selection.selected_set
    capacity = problem.capacity_bits()
    used = selection.bits_by_window()
    unreachable = set(problem.unreachable())

    rows: list[LeftBehind] = []
    for name in problem.inventory.names:
        if name in selected:
            continue
        size = problem.sizes_bits[name]
        windows = problem.feasible_windows(name)
        if not windows:
            rows.append(
                LeftBehind(name, float(problem.values[name]), size, "expired", None, 0.0, 0.0)
            )
            continue
        if all(size > capacity[w] + feasibility_slack_bits(capacity[w]) for w in windows):
            rows.append(
                LeftBehind(name, float(problem.values[name]), size, "oversized", None, 0.0, 0.0)
            )
            continue
        if problem.inventory.dependency_closure(name) & unreachable:
            rows.append(
                LeftBehind(
                    name, float(problem.values[name]), size, "dependency", None, 0.0, 0.0
                )
            )
            continue
        shortfalls: list[tuple[float, str]] = []
        for window in windows:
            free = capacity[window] - used.get(window, 0.0)
            shortfalls.append((max(size - free, 0.0), window))
        shortfall, window = min(shortfalls, key=lambda item: (item[0], item[1]))
        rate = problem.plan[window].final_rate_bps
        seconds = shortfall / rate if rate > 0.0 else float("inf")
        rows.append(
            LeftBehind(
                product=name,
                value=float(problem.values[name]),
                bits=size,
                reason="capacity",
                best_window=window,
                extra_bits_needed=shortfall,
                extra_seconds_needed=seconds,
            )
        )
    rows.sort(key=lambda row: (-row.value, problem.inventory.position(row.product)))
    return tuple(rows)


def value_left_behind(rows: tuple[LeftBehind, ...]) -> dict[str, float]:
    """Total value left behind per reason, in value units.

    Every key in :data:`LEFT_BEHIND_REASONS` is present, zero where nothing was
    left for that reason, so a report table never has missing rows.
    """
    totals = dict.fromkeys(LEFT_BEHIND_REASONS, 0.0)
    for row in rows:
        totals[row.reason] += row.value
    return totals


def marginal_value_of_contact(
    problem: SelectionProblem,
    *,
    selector: Selector = exact_select,
    delta_s: float = DEFAULT_DELTA_S,
    windows: tuple[str, ...] | None = None,
    base: Selection | None = None,
) -> tuple[WindowMarginal, ...]:
    """Price ``delta_s`` more seconds on each window by re-solving.

    Parameters
    ----------
    problem:
        The selection problem.
    selector:
        The selector to re-solve with. Pass
        :func:`downlinkpick.greedy.greedy_select` to price the extension under
        the operational baseline rather than under the optimum; the two differ
        and the difference is itself informative.
    delta_s:
        Finite-difference step in seconds, strictly positive.
    windows:
        Window names to price. ``None`` prices all of them.
    base:
        A selection for the unextended problem, to avoid re-solving it. Must
        have been produced by ``selector`` on ``problem``.

    Returns
    -------
    tuple of WindowMarginal
        In contact-plan order.
    """
    if not (delta_s > 0.0):
        raise ValueError(f"delta_s must be strictly positive seconds, got {delta_s!r}")
    names = problem.plan.names if windows is None else tuple(windows)
    for name in names:
        if name not in problem.plan:
            raise KeyError(f"no window named {name!r} in contact plan")
    nominal = selector(problem) if base is None else base
    nominal_set = nominal.selected_set

    results: list[WindowMarginal] = []
    for name in names:
        window = problem.plan[name]
        crossings = tuple(
            product.name
            for product in problem.inventory
            if window.end_s <= product.expiry_s < window.end_s + delta_s
        )
        try:
            extended_plan = problem.plan.extended_window(name, delta_s)
        except ValueError:
            results.append(
                WindowMarginal(
                    window=name,
                    delta_s=delta_s,
                    delta_bits=0.0,
                    base_value=nominal.total_value,
                    extended_value=nominal.total_value,
                    value_per_second=0.0,
                    value_per_gbit=None,
                    entering=(),
                    leaving=(),
                    expiry_crossings=crossings,
                )
            )
            continue
        added = extended_plan[name].capacity_bits() - window.capacity_bits()
        extended_selection = selector(problem.with_plan(extended_plan))
        gain = extended_selection.total_value - nominal.total_value
        results.append(
            WindowMarginal(
                window=name,
                delta_s=delta_s,
                delta_bits=added,
                base_value=nominal.total_value,
                extended_value=extended_selection.total_value,
                value_per_second=gain / delta_s,
                value_per_gbit=(gain / (added / 1.0e9)) if added > 0.0 else None,
                entering=tuple(sorted(extended_selection.selected_set - nominal_set)),
                leaving=tuple(sorted(nominal_set - extended_selection.selected_set)),
                expiry_crossings=crossings,
            )
        )
    return tuple(results)


def best_window_to_extend(marginals: tuple[WindowMarginal, ...]) -> WindowMarginal | None:
    """The window whose next second buys the most value, or ``None``.

    ``None`` when no window's extension raises the objective, which is the
    common and important case: it means the binding constraint is not contact
    time, and asking the network for more of it would buy nothing -- or, when
    :attr:`WindowMarginal.expiry_crossings` is non-empty, would cost value.
    """
    paying = [m for m in marginals if m.delta_value > 0.0]
    if not paying:
        return None
    return max(paying, key=lambda m: (m.value_per_second, -m.delta_bits, m.window))


@dataclass(frozen=True)
class SweepPoint:
    """One point of a capacity sweep.

    Attributes
    ----------
    extra_s:
        Seconds added to the swept window.
    capacity_bits:
        Total plan capacity at this point, in bits.
    total_value:
        Objective at this point, in value units.
    n_selected:
        Number of products selected.
    n_expiry_crossings:
        Number of products whose expiry falls inside the added time, so that the
        swept window is too late for them. Non-zero means the sweep has removed
        candidates as well as added capacity, and the objective may legitimately
        fall at this point.
    """

    extra_s: float
    capacity_bits: float
    total_value: float
    n_selected: int
    n_expiry_crossings: int = 0


def capacity_sweep(
    problem: SelectionProblem,
    window: str,
    extra_seconds: tuple[float, ...],
    *,
    selector: Selector = exact_select,
) -> tuple[SweepPoint, ...]:
    """Objective against added contact time on one window.

    The curve is a step function, not a smooth concave one, and it is
    non-decreasing only where the added time crosses no product's expiry.
    Extending a window moves its delivery epoch later, so a point whose
    ``n_expiry_crossings`` is non-zero may sit *below* its predecessor: the
    extension bought capacity and removed candidates.
    ``validation/validate_marginal.py`` measures three things here -- decreasing
    steps that cross an expiry, decreasing steps that do not (which must be
    zero), and convexity violations, which are not zero and are reported as a
    property of the integer program rather than as a defect.
    """
    if window not in problem.plan:
        raise KeyError(f"no window named {window!r} in contact plan")
    points: list[SweepPoint] = []
    for extra in extra_seconds:
        if extra < 0.0:
            raise ValueError(f"extra_seconds must be non-negative, got {extra!r}")
        plan = problem.plan if extra == 0.0 else problem.plan.extended_window(window, extra)
        local = problem.with_plan(plan)
        selection = selector(local)
        end = problem.plan[window].end_s
        points.append(
            SweepPoint(
                extra_s=extra,
                capacity_bits=plan.total_capacity_bits(),
                total_value=selection.total_value,
                n_selected=len(selection.assignments),
                n_expiry_crossings=sum(
                    1 for product in problem.inventory if end <= product.expiry_s < end + extra
                ),
            )
        )
    return tuple(points)
