"""Baseline 2: the multiple-knapsack MILP, and the enumeration that checks it.

The formulation is equations (2) and the three constraint families of
:mod:`downlinkpick.selection`, written once as a :class:`MilpModel` and then
emitted twice: as a PuLP problem, and as the coefficient arrays that
:func:`scipy.optimize.milp` consumes. Both emissions read the *same* model
object, and ``validation/validate_exact.py`` checks them against each other term
by term, so a divergence between the two is a measured quantity rather than an
assumption.

Why both. PuLP is the modelling layer named in the specification and the LP file
it writes is the portable, human-readable statement of the problem, which is
what you hand to someone who wants to re-solve it with a solver you do not have.
But PuLP is a modelling layer and not a solver: it needs a backend. The measured
environment for this build has **PuLP 4.1.0 with no solver attached** --
``pulp.listSolvers(onlyAvailable=True)`` returns the empty list, 4.x no longer
bundles CBC, and ``pulp.PULP_CBC_CMD`` no longer exists -- so ``LpProblem.solve``
raises ``PulpError: No solver available``. The optimum reported by this package
is therefore computed by ``scipy.optimize.milp``, which is the HiGHS
branch-and-bound solver shipped inside SciPy and needs nothing installed. The
PuLP path is kept, exercised and tested, and is used when a backend is present.
This is recorded in the committed validation output and in README.md rather than
papered over, because which solver produced an optimum is part of the claim that
it is one.

Exhaustive enumeration is the third route and the one that makes "optimal"
checkable. For an instance small enough that the product of per-product choice
counts is bounded, every assignment is enumerated, the infeasible ones are
dropped, and the best feasible objective is the optimum by construction with no
solver in the loop at all. ``validation/validate_exact.py`` compares the MILP
optimum against it over a seeded family of small instances.

References
----------
Martello, S. and Toth, P. (1990), Knapsack Problems: Algorithms and Computer
Implementations, Wiley, chapter 6. The multiple-knapsack problem and the
assignment formulation used here. Structure only; no numerical value in this
repository is taken from it.

Huangfu, Q. and Hall, J. A. J. (2018), Parallelizing the dual revised simplex
method, Mathematical Programming Computation 10(1), 119-142. The HiGHS solver
that ``scipy.optimize.milp`` calls.

Mitchell, S., O'Sullivan, M. and Dunning, I. (2011), PuLP: a linear programming
toolkit for Python. The modelling layer used to emit the LP file.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contacts import feasibility_slack_bits
from .selection import Assignment, Selection, SelectionProblem

#: Backends :func:`exact_select` understands. ``"auto"`` prefers ``"pulp"`` when
#: PuLP reports an available solver and falls back to ``"scipy"``.
BACKENDS: tuple[str, ...] = ("auto", "scipy", "pulp", "enumerate")

#: Whether to silence the solver's own C-level chatter during a solve. HiGHS
#: occasionally writes a diagnostic line straight to file descriptor 1 from
#: inside its MIP solver, which a library has no business doing to its caller's
#: stdout and which would otherwise land in the middle of a committed validation
#: transcript. Nothing is lost: ``scipy.optimize.milp`` returns the status and
#: the message in its result object, and this package reports both. Set to
#: ``False`` to see the solver's output while debugging.
SUPPRESS_SOLVER_OUTPUT = True

#: Largest number of complete assignments :func:`enumerate_select` will visit
#: before refusing. 3^12 is 531441; the enumeration is used only as a reference
#: on small instances and a cap keeps an accidental call from hanging a suite.
ENUMERATION_LIMIT = 2_000_000

#: HiGHS's absolute MIP gap, in value units. A branch-and-bound solver stops
#: when the gap between its incumbent and its dual bound is within tolerance,
#: and HiGHS's default absolute tolerance is 1e-6. ``scipy.optimize.milp``
#: exposes ``mip_rel_gap`` but not the absolute gap, so this tolerance cannot be
#: tightened from here and is instead stated: a ``Selection`` whose status is
#: ``"optimal"`` is optimal to within this many value units, not exactly.
#:
#: This is not academic. A Hypothesis property test falsified the claim that
#: greedy can never beat the MILP, with an instance containing a product of
#: value 5.960464477539063e-08: greedy and the exhaustive enumeration both took
#: it, HiGHS stopped without it, and the difference -- 5.96e-08 -- is below the
#: gap. The counterexample is pinned in ``tests/test_regressions.py``. On a
#: value scale where products are worth of order one, 1e-6 value units is
#: nothing; on a scale where they are worth 1e-8, this package's optimality
#: claim is vacuous and the values should be rescaled.
SOLVER_ABS_GAP_VALUE_UNITS = 1.0e-6

#: Relative MIP gap requested from the solver, dimensionless. Zero, so that the
#: relative gap is never the limiting tolerance and
#: :data:`SOLVER_ABS_GAP_VALUE_UNITS` is the only one that applies.
SOLVER_REL_GAP = 0.0

#: Status set when the rounded solution had to be repaired. A branch-and-bound
#: solver returns a vector that is integral only to within its own integrality
#: tolerance -- HiGHS's default is 1e-6 -- and this package rounds at 0.5 and
#: then charges the *whole* size of every rounded-up product. Rounding 0.9999994
#: up to 1 therefore commits bits the solver's own arithmetic never committed,
#: and the window can end up over capacity. Two Hypothesis property tests
#: falsified the formulation this way: on a window of 7999538 bits with products
#: of 8 and 7999535 bits, the MILP reported both selected, 5 bits over capacity,
#: and on a window of 7999615 bits it reported an objective of 6.0 above
#: Dantzig's fractional bound of 5.999998749938903, which is impossible for a
#: feasible selection. Both counterexamples are pinned in
#: ``tests/test_regressions.py``.
#:
#: :func:`_repair_overflow` drops the least valuable products from any
#: over-committed window, together with everything that depends on them, until
#: every window fits. The result is feasible by construction, its objective is a
#: lower bound on the optimum rather than the optimum, and the status says so.
#: A selector that returns an infeasible selection is worse than a selector that
#: returns a suboptimal one, so feasibility wins and the loss is reported.
ROUNDED_STATUS = "optimal-rounded"


@contextlib.contextmanager
def _quiet_solver() -> Iterator[None]:
    """Redirect file descriptor 1 to the null device for the duration.

    Python-level ``sys.stdout`` is flushed first so that nothing already written
    is swallowed, and the original descriptor is restored unconditionally.
    """
    if not SUPPRESS_SOLVER_OUTPUT:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


@dataclass(frozen=True)
class MilpModel:
    """The multiple-knapsack MILP in explicit coefficient form.

    Attributes
    ----------
    pairs:
        Decision variables as ``(product, window)``, one per feasible pair, in
        model column order.
    values:
        Objective coefficient per column, in value units (equation 2).
    sizes_bits:
        Size coefficient per column, in bits.
    products:
        Product names in row order for the no-double-downlink rows.
    windows:
        Window names in row order for the capacity rows.
    capacities_bits:
        Capacity per window row, in bits.
    precedence_rows:
        One entry per precedence row, as
        ``(dependent column index, tuple of prerequisite column indices)``. The
        row reads ``x_dependent - sum(x_prerequisite) <= 0``.

    Notes
    -----
    The model states the problem in natural units -- bits and value units -- and
    both emitters divide each capacity row by that window's capacity through
    :meth:`capacity_scale` before handing it to a solver. This is not
    cosmetic. Capacities are of order 1e10 bits and objective coefficients of
    order 1e1 value units, and on the unscaled rows HiGHS returned a
    *wrongly proved optimum* -- ``status == 0``, objective below the true
    optimum -- on 12 of 240 small instances, and a hard solve error on one. The
    exhaustive enumeration in ``validation/validate_exact.py`` is what found it;
    nothing in the solver's own status reporting did. The scaled rows read
    ``sum_i (s_i / C_w) x_iw <= 1``, which is the same constraint with
    coefficients in a range a double can discriminate.
    """

    pairs: tuple[tuple[str, str], ...]
    values: tuple[float, ...]
    sizes_bits: tuple[float, ...]
    products: tuple[str, ...]
    windows: tuple[str, ...]
    capacities_bits: tuple[float, ...]
    precedence_rows: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def n_columns(self) -> int:
        """Number of binary decision variables."""
        return len(self.pairs)

    @property
    def n_rows(self) -> int:
        """Number of inequality rows."""
        return len(self.products) + len(self.windows) + len(self.precedence_rows)

    def column_of(self, product: str, window: str) -> int:
        """Column index of the ``(product, window)`` variable."""
        return self.pairs.index((product, window))

    def capacity_scale(self, window_index: int) -> float:
        """Divisor applied to one capacity row before solving, in bits.

        The window's capacity, or ``1.0`` for a window of zero capacity, which
        would otherwise divide by zero. A zero-capacity window admits nothing
        either way.
        """
        capacity = self.capacities_bits[window_index]
        return capacity if capacity > 0.0 else 1.0


def build_model(problem: SelectionProblem) -> MilpModel:
    """Assemble the MILP in coefficient form from a :class:`SelectionProblem`.

    Columns are generated in inventory order and, within a product, in window
    time order, so the model is a deterministic function of the problem and two
    builds of the same problem are identical object-for-object.
    """
    pairs: list[tuple[str, str]] = []
    values: list[float] = []
    sizes: list[float] = []
    for name in problem.inventory.names:
        for window in problem.feasible_windows(name):
            pairs.append((name, window))
            values.append(problem.pair_value(name, window))
            sizes.append(problem.sizes_bits[name])
    column = {pair: index for index, pair in enumerate(pairs)}

    windows = problem.plan.names
    capacity = problem.capacity_bits()

    precedence: list[tuple[int, tuple[int, ...]]] = []
    for name in problem.inventory.names:
        for prerequisite in problem.inventory[name].depends_on:
            for window in problem.feasible_windows(name):
                end = problem.plan[window].end_s
                allowed = tuple(
                    column[(prerequisite, other)]
                    for other in problem.feasible_windows(prerequisite)
                    if problem.plan[other].end_s <= end
                )
                precedence.append((column[(name, window)], allowed))

    return MilpModel(
        pairs=tuple(pairs),
        values=tuple(values),
        sizes_bits=tuple(sizes),
        products=tuple(problem.inventory.names),
        windows=tuple(windows),
        capacities_bits=tuple(capacity[w] for w in windows),
        precedence_rows=tuple(precedence),
    )


def model_arrays(
    model: MilpModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ``(objective, A_ub, b_ub)`` arrays for the model, capacity rows scaled.

    Each capacity row is divided by that window's capacity, so its coefficients
    are the fraction of the window a product occupies and its right-hand side is
    1. See the notes on :class:`MilpModel` for why: the unscaled rows made HiGHS
    report a wrong optimum on 5 per cent of a seeded small-instance family.

    The objective is returned for *maximisation*; callers minimising negate it.
    Dense is adequate because the instances this package solves exactly have
    hundreds of columns at most, and a dense matrix makes the term-by-term
    comparison against the PuLP emission in ``validation/validate_exact.py``
    straightforward to write and to read.
    """
    columns = model.n_columns
    rows: list[np.ndarray] = []
    bounds: list[float] = []

    for product in model.products:
        row = np.zeros(columns)
        for index, (name, _window) in enumerate(model.pairs):
            if name == product:
                row[index] = 1.0
        rows.append(row)
        bounds.append(1.0)

    for position, window in enumerate(model.windows):
        scale = model.capacity_scale(position)
        row = np.zeros(columns)
        for index, (_name, other) in enumerate(model.pairs):
            if other == window:
                row[index] = model.sizes_bits[index] / scale
        rows.append(row)
        bounds.append(model.capacities_bits[position] / scale)

    for dependent, allowed in model.precedence_rows:
        row = np.zeros(columns)
        row[dependent] += 1.0
        for index in allowed:
            row[index] -= 1.0
        rows.append(row)
        bounds.append(0.0)

    objective = np.asarray(model.values, dtype=float)
    matrix = np.vstack(rows) if rows else np.zeros((0, columns))
    return objective, matrix, np.asarray(bounds, dtype=float)


def build_pulp_problem(problem: SelectionProblem, *, name: str = "downlinkpick") -> Any:
    """The same model as a :class:`pulp.LpProblem`.

    PuLP 4.x creates variables through ``LpProblem.add_variable`` rather than
    through the ``LpVariable`` constructor, which 4.x reserves for wrapping its
    Rust core object. This function therefore does not work on PuLP 2.x or 3.x,
    which is why ``pyproject.toml`` bounds ``pulp`` on both sides.
    """
    import pulp

    model = build_model(problem)
    lp = pulp.LpProblem(name, pulp.LpMaximize)
    variables = [
        lp.add_variable(f"x_{index}", cat="Binary") for index in range(model.n_columns)
    ]
    lp += pulp.lpSum(
        coefficient * variable
        for coefficient, variable in zip(model.values, variables, strict=True)
    )
    for product in model.products:
        columns = [i for i, (n, _w) in enumerate(model.pairs) if n == product]
        if columns:
            lp += pulp.lpSum(variables[i] for i in columns) <= 1, f"once_{product}"
    for position, window in enumerate(model.windows):
        columns = [i for i, (_n, w) in enumerate(model.pairs) if w == window]
        if columns:
            scale = model.capacity_scale(position)
            lp += (
                pulp.lpSum(model.sizes_bits[i] / scale * variables[i] for i in columns)
                <= model.capacities_bits[position] / scale,
                f"capacity_{window}",
            )
    for row, (dependent, allowed) in enumerate(model.precedence_rows):
        lp += (
            variables[dependent] - pulp.lpSum(variables[i] for i in allowed) <= 0,
            f"precedence_{row}",
        )
    return lp


def write_lp(problem: SelectionProblem, path: str | Path, *, name: str = "downlinkpick") -> str:
    """Write the PuLP LP file for ``problem`` and return the path written."""
    lp = build_pulp_problem(problem, name=name)
    target = Path(path)
    lp.writeLP(str(target))
    return str(target)


def pulp_available() -> bool:
    """True when PuLP reports at least one usable solver backend."""
    try:
        import pulp
    except ImportError:  # pragma: no cover - pulp is a hard dependency
        return False
    try:
        return bool(pulp.listSolvers(onlyAvailable=True))
    except Exception:  # pragma: no cover - defensive, PuLP probes the filesystem
        return False


def _repair_overflow(
    problem: SelectionProblem, model: MilpModel, chosen: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Drop columns until every window fits, returning ``(kept, dropped)``.

    The victim is the lowest-value column in the most over-committed window;
    everything in the selection that transitively depends on the victim's
    product is dropped with it, so the repair can never leave a dependent
    without its prerequisite. Ties break on the larger size first and then on
    the column index, so the repair is deterministic.
    """
    selected = set(chosen)
    dropped: list[int] = []
    window_index = {name: index for index, name in enumerate(model.windows)}
    for _ in range(len(model.pairs) + 1):
        committed: dict[str, float] = {}
        for column in selected:
            committed[model.pairs[column][1]] = (
                committed.get(model.pairs[column][1], 0.0) + model.sizes_bits[column]
            )
        excess: list[tuple[float, str]] = []
        for name, used in committed.items():
            limit = model.capacities_bits[window_index[name]]
            over = used - limit
            if over > feasibility_slack_bits(limit):
                excess.append((over, name))
        if not excess:
            break
        _over, window = max(excess)
        candidates = [c for c in selected if model.pairs[c][1] == window]
        if not candidates:
            break
        victim = min(
            candidates,
            key=lambda c: (model.values[c], -model.sizes_bits[c], c),
        )
        victim_product = model.pairs[victim][0]
        casualties = {victim} | {
            c
            for c in selected
            if victim_product in problem.inventory.dependency_closure(model.pairs[c][0])
        }
        selected -= casualties
        dropped.extend(sorted(casualties))
    return sorted(selected), dropped


def _selection_from_columns(
    problem: SelectionProblem,
    model: MilpModel,
    chosen: Sequence[int],
    *,
    method: str,
    status: str,
    bound: float | None,
) -> Selection:
    kept, dropped = _repair_overflow(problem, model, chosen)
    if dropped:
        chosen = kept
        if status == "optimal":
            status = ROUNDED_STATUS
    assignments = tuple(
        Assignment(
            product=model.pairs[index][0],
            window=model.pairs[index][1],
            bits=model.sizes_bits[index],
            value=model.values[index],
        )
        for index in sorted(
            chosen,
            key=lambda i: (
                problem.plan[model.pairs[i][1]].start_s,
                problem.inventory.position(model.pairs[i][0]),
            ),
        )
    )
    return Selection(
        assignments=assignments,
        total_value=sum(a.value for a in assignments),
        method=method,
        status=status,
        bound=bound,
    )


def solve_scipy(problem: SelectionProblem, *, time_limit_s: float | None = None) -> Selection:
    """Solve the MILP with ``scipy.optimize.milp`` (HiGHS branch and bound)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    model = build_model(problem)
    if model.n_columns == 0:
        return Selection((), 0.0, "exact-scipy", "optimal", 0.0)
    objective, matrix, upper = model_arrays(model)
    options: dict[str, Any] = {"mip_rel_gap": SOLVER_REL_GAP}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    with _quiet_solver():
        result = milp(
            c=-objective,
            constraints=LinearConstraint(matrix, -np.inf, upper),
            integrality=np.ones(model.n_columns),
            bounds=Bounds(0.0, 1.0),
            options=options,
        )
    if result.x is None:
        raise RuntimeError(
            f"scipy.optimize.milp returned no solution (status {result.status}): {result.message}"
        )
    chosen = [index for index, value in enumerate(result.x) if value > 0.5]
    status = "optimal" if result.status == 0 else f"scipy-status-{result.status}"
    bound = float(-result.fun) if result.status == 0 else None
    return _selection_from_columns(
        problem, model, chosen, method="exact-scipy", status=status, bound=bound
    )


def solve_pulp(problem: SelectionProblem, *, solver: Any | None = None) -> Selection:
    """Solve the MILP through PuLP with whatever backend PuLP has.

    Raises :class:`RuntimeError` with an actionable message when PuLP has no
    solver, which is the state of the environment this package was measured in.
    """
    import pulp

    model = build_model(problem)
    lp = build_pulp_problem(problem)
    if model.n_columns == 0:
        return Selection((), 0.0, "exact-pulp", "optimal", 0.0)
    available = pulp.listSolvers(onlyAvailable=True)
    if solver is None and not available:
        raise RuntimeError(
            "PuLP reports no available solver, so the MILP cannot be solved through PuLP in "
            "this environment. PuLP 4.x does not bundle CBC and pulp.PULP_CBC_CMD no longer "
            "exists. Use backend='scipy', which solves the same model with the HiGHS solver "
            "shipped inside SciPy, or install a solver PuLP can find."
        )
    try:
        lp.solve(solver) if solver is not None else lp.solve()
    except pulp.PulpSolverError as error:
        raise RuntimeError(f"PuLP failed to solve the model: {error}") from error
    variables = {variable.name: variable for variable in lp.variables()}
    chosen = [
        index
        for index in range(model.n_columns)
        if (variables[f"x_{index}"].value() or 0.0) > 0.5
    ]
    status = "optimal" if pulp.LpStatus[lp.status] == "Optimal" else pulp.LpStatus[lp.status]
    bound = float(pulp.value(lp.objective)) if status == "optimal" else None
    return _selection_from_columns(
        problem, model, chosen, method="exact-pulp", status=status, bound=bound
    )


def exact_select(
    problem: SelectionProblem,
    *,
    backend: str = "auto",
    time_limit_s: float | None = None,
) -> Selection:
    """Solve the selection problem to proved optimality.

    Parameters
    ----------
    problem:
        The selection problem.
    backend:
        One of :data:`BACKENDS`. ``"auto"`` uses PuLP when PuLP has a solver and
        SciPy otherwise, which in the measured environment means SciPy.
    time_limit_s:
        Wall-clock limit passed to the SciPy backend, in seconds. ``None`` means
        no limit. A truncated solve returns ``status`` other than ``"optimal"``
        and the caller must not treat it as an optimum.

    Returns
    -------
    Selection
        With ``status == "optimal"`` only when the backend proved optimality.
        For the SciPy backend that means optimal to within
        :data:`SOLVER_ABS_GAP_VALUE_UNITS`; only the ``"enumerate"`` backend
        proves optimality with no tolerance at all.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "enumerate":
        return enumerate_select(problem)
    if backend == "pulp":
        return solve_pulp(problem)
    if backend == "scipy":
        return solve_scipy(problem, time_limit_s=time_limit_s)
    if pulp_available():
        return solve_pulp(problem)
    return solve_scipy(problem, time_limit_s=time_limit_s)


def enumeration_size(problem: SelectionProblem) -> int:
    """Number of complete assignments :func:`enumerate_select` would visit."""
    total = 1
    for name in problem.inventory.names:
        total *= len(problem.feasible_windows(name)) + 1
        if total > ENUMERATION_LIMIT:
            return total
    return total


def enumerate_select(problem: SelectionProblem, *, limit: int = ENUMERATION_LIMIT) -> Selection:
    """The optimum by exhaustive enumeration, with no solver in the loop.

    Every product independently takes one of its feasible windows or none; each
    complete assignment is tested against the capacity, precedence and
    dependency constraints and scored. Used as the reference the MILP optimum is
    checked against on small instances.

    Raises :class:`ValueError` when the assignment count exceeds ``limit``.
    """
    size = enumeration_size(problem)
    if size > limit:
        raise ValueError(
            f"enumeration would visit {size} assignments, above the limit of {limit}; "
            "enumeration is a small-instance reference only"
        )
    names = problem.inventory.names
    choices = [(None, *problem.feasible_windows(name)) for name in names]
    capacity = problem.capacity_bits()
    sizes = problem.sizes_bits
    ends = {w: problem.plan[w].end_s for w in problem.plan.names}

    best_value = 0.0
    best: tuple[tuple[str, str], ...] = ()
    for combination in itertools.product(*choices):
        placement = {
            name: window
            for name, window in zip(names, combination, strict=True)
            if window is not None
        }
        used: dict[str, float] = {}
        feasible = True
        for name, window in placement.items():
            used[window] = used.get(window, 0.0) + sizes[name]
        for window, bits in used.items():
            if bits > capacity[window] + feasibility_slack_bits(capacity[window]):
                feasible = False
                break
        if not feasible:
            continue
        for name, window in placement.items():
            for prerequisite in problem.inventory[name].depends_on:
                other = placement.get(prerequisite)
                if other is None or ends[other] > ends[window]:
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue
        value = sum(problem.pair_value(name, window) for name, window in placement.items())
        if value > best_value:
            best_value = value
            best = tuple(sorted(placement.items()))

    assignments = tuple(
        Assignment(
            product=name,
            window=window,
            bits=sizes[name],
            value=problem.pair_value(name, window),
        )
        for name, window in sorted(
            best,
            key=lambda item: (
                problem.plan[item[1]].start_s,
                problem.inventory.position(item[0]),
            ),
        )
    )
    return Selection(
        assignments=assignments,
        total_value=sum(a.value for a in assignments),
        method="enumerate",
        status="optimal",
        bound=sum(a.value for a in assignments),
    )


def lp_relaxation_bound(problem: SelectionProblem) -> float:
    """Upper bound from the linear relaxation of the MILP, in value units.

    Solved with ``scipy.optimize.linprog`` (HiGHS dual simplex) over the same
    coefficient arrays, with the integrality requirement dropped. Always an
    upper bound on the integral optimum, and tighter than
    :func:`downlinkpick.greedy.dantzig_bound` because it keeps the per-window
    capacity split and the precedence rows.
    """
    from scipy.optimize import linprog

    model = build_model(problem)
    if model.n_columns == 0:
        return 0.0
    objective, matrix, upper = model_arrays(model)
    with _quiet_solver():
        result = linprog(
            c=-objective,
            A_ub=matrix,
            b_ub=upper,
            bounds=[(0.0, 1.0)] * model.n_columns,
            method="highs",
        )
    if not result.success:
        raise RuntimeError(f"LP relaxation failed: {result.message}")
    return float(-result.fun)
