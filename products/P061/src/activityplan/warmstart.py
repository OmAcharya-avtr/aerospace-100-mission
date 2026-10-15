"""Warm-starting the exact solver from a plan, and measuring whether it helps.

What a warm start can and cannot be here
----------------------------------------
``scipy.optimize.milp`` exposes no MIP start: there is no argument for an
initial integer assignment, and the HiGHS option set it forwards
(``time_limit``, ``mip_rel_gap``, ``presolve``, ``node_limit``, ``disp``)
contains no objective cutoff either. The warm start available through that
interface is therefore a **primal-bound cut**: knowing a feasible plan of value
``V`` licenses the row ``sum_i p_i * sum_t x[i,t] >= V``, which is valid because
a solution attaining it exists, and which gives branch-and-bound the same
pruning information a primal bound from a MIP start would.

**It is not an integer start.** It supplies no assignment for the solver to
repair, so it cannot help a solver that is struggling to find any feasible
solution -- and in this model the empty plan is always feasible, so that was
never the difficulty. The distinction matters for reading the measurement and is
stated here, in the README and in ``MODEL_CARD.md``.

The bound must be verified before it is used
--------------------------------------------
A plan produced from predicted durations is a plan for a *different* problem.
Used as a bound on the target problem it can be wrong in the dangerous
direction: if no plan of that value is feasible for the target, the cut removes
the optimum and the solver returns infeasible or a worse answer. So
:func:`warm_start_from_plan` **simulates the candidate plan against the target
problem** and returns a bound only if the simulator says it is feasible there.
An unverified candidate yields ``bound=None`` and the solve runs cold. How often
that happens is itself a measurement and is reported in
``validation/validate_warmstart_output.txt``.

Reading the node counts
-----------------------
With HiGHS presolve on, every shipped instance of this size is solved at the
root or in presolve, so ``mip_node_count`` is 1 and measures presolve rather
than search. The comparison is therefore run with ``presolve`` both on and off,
and both are reported. A node count of 1 in both arms is not evidence that the
warm start helped; it is evidence that there was nothing to help.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exact import ExactResult, solve_exact
from .feasibility import evaluate
from .timeline import Plan, RasterisedProblem

__all__ = ["WarmStart", "WarmStartComparison", "compare_warm_start", "warm_start_from_plan"]


@dataclass(frozen=True)
class WarmStart:
    """A candidate primal bound and the verification that licensed it.

    Attributes
    ----------
    source
        Where the candidate plan came from, for the report.
    candidate_value
        Objective value of the candidate plan **measured on the target
        problem's priorities**, which are the same priorities in every mode
        because only durations and energies are predicted.
    verified
        Whether the candidate plan is feasible for the target problem under the
        resource simulator.
    bound
        ``candidate_value`` when verified, otherwise ``None``.
    reason
        Human-readable verification outcome, naming the violated constraint when
        verification failed.
    """

    source: str
    candidate_value: float
    verified: bool
    bound: float | None
    reason: str


@dataclass(frozen=True)
class WarmStartComparison:
    """Cold and warm solves of the same target problem.

    Attributes
    ----------
    instance
        Target problem name.
    presolve
        Whether HiGHS presolve was enabled in both arms.
    cold, warm
        The two solves. ``warm`` is ``None`` when the candidate did not verify,
        in which case there was no bound to apply.
    warm_start
        The candidate and its verification.
    """

    instance: str
    presolve: bool
    cold: ExactResult
    warm: ExactResult | None
    warm_start: WarmStart

    @property
    def optimum_preserved(self) -> bool:
        """Whether the warm solve returned the same objective as the cold solve.

        Compared with a tolerance of ``1e-9`` on the objective, which is a sum of
        declared priorities and so is exact in principle but is reconstructed
        from a relaxed binary vector in practice.
        """
        if self.warm is None:
            return True
        return abs(self.warm.value - self.cold.value) <= 1e-9

    @property
    def node_ratio(self) -> float | None:
        """``warm nodes / cold nodes``, or ``None`` when there was no warm solve."""
        if self.warm is None or self.cold.node_count == 0:
            return None
        return self.warm.node_count / self.cold.node_count

    @property
    def time_ratio(self) -> float | None:
        """``warm solve time / cold solve time``, or ``None`` when there was no warm solve.

        Wall clocks on a shared 2-core container move by tens of percent between
        runs. Read the node ratio first; this is reported because it is what a
        user feels, not because it is stable.
        """
        if self.warm is None or self.cold.solve_s <= 0.0:
            return None
        return self.warm.solve_s / self.cold.solve_s


def warm_start_from_plan(
    target: RasterisedProblem, plan: Plan, *, source: str
) -> WarmStart:
    """Verify ``plan`` against ``target`` and turn its value into a primal bound.

    Parameters
    ----------
    target
        The problem the MILP will actually solve.
    plan
        A candidate plan, typically from the greedy planner on a problem built
        with predicted durations, so possibly infeasible here.
    source
        Label recorded in the result.

    Returns
    -------
    WarmStart
        With ``bound`` set only if the simulator finds ``plan`` feasible for
        ``target``.
    """
    try:
        report = evaluate(target, plan)
    except ValueError as error:
        return WarmStart(
            source=source,
            candidate_value=0.0,
            verified=False,
            bound=None,
            reason=f"candidate plan could not be simulated on the target: {error}",
        )
    value = plan.value(target.problem)
    if report.feasible:
        return WarmStart(
            source=source,
            candidate_value=value,
            verified=True,
            bound=value,
            reason=(
                f"verified feasible on the target; binding constraint "
                f"{report.binding.name} at normalised slack {report.binding.normalised:+.6f}"
            ),
        )
    worst = report.violations[0]
    return WarmStart(
        source=source,
        candidate_value=value,
        verified=False,
        bound=None,
        reason=(
            f"rejected: {worst.name} has slack {worst.slack:.6g} {worst.unit} on the "
            "target problem, so the value is not a valid bound"
        ),
    )


def compare_warm_start(
    target: RasterisedProblem,
    warm_start: WarmStart,
    *,
    presolve: bool = True,
    time_limit_s: float = 60.0,
) -> WarmStartComparison:
    """Solve ``target`` cold, then with ``warm_start``'s bound if it verified.

    The cold solve runs first so that any warm-up cost in the solver library is
    paid by the cold arm, which is the conservative direction for a claim that
    the warm start is faster.
    """
    cold = solve_exact(target, time_limit_s=time_limit_s, presolve=presolve)
    warm: ExactResult | None = None
    if warm_start.bound is not None:
        warm = solve_exact(
            target,
            primal_bound=warm_start.bound,
            time_limit_s=time_limit_s,
            presolve=presolve,
        )
    return WarmStartComparison(
        instance=target.problem.name,
        presolve=presolve,
        cold=cold,
        warm=warm,
        warm_start=warm_start,
    )
