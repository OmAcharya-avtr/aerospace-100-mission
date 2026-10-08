"""Text rendering of results. Every function returns a string; nothing prints.

The CLI is the only thing in this package that writes to a stream, and it
writes what these functions return. That split is why there is no ``print`` in
the library.

Vocabulary rule, enforced by ``tests/test_vocabulary.py``
---------------------------------------------------------
**Falsification is one-sided. Finding no violation is not evidence of
correctness.** So nothing rendered here may read as a verdict of correctness:
the words "pass", "passed", "safe", "verified", "certified", "compliant" and
"clean" do not appear in any rendered line, and the absence of a violation is
always rendered as "NO VIOLATION FOUND" with the one-sidedness restated. The
test asserts it over the rendered output of every function in this module rather
than trusting the author to remember.
"""

from __future__ import annotations

import numpy as np

from .benchmark import BenchmarkReport, CellSummary
from .curves import clopper_pearson
from .instances import Instance
from .search import BASELINE, SearchResult
from .systems import LoopInput

#: The one-sidedness statement, used verbatim wherever a search found nothing.
ONE_SIDED_NOTE = (
    "Falsification is one-sided: finding no violation is not evidence of correctness."
)


def _fmt_optional(value: float | None, budget: int) -> str:
    return f"> {budget}" if value is None else f"{value:.1f}"


def render_search_result(result: SearchResult, instance: Instance) -> str:
    """Render one search run.

    A found violation is reported with the simulation index and the decision
    vector that produced it. A run that found nothing is reported as having
    found nothing, with :data:`ONE_SIDED_NOTE`.
    """
    lines = [
        f"instance    : {result.instance_id}  [{instance.tier}]",
        f"requirement : {instance.requirement}",
        f"strategy    : {result.strategy}",
        f"seed        : {result.seed}",
        f"budget      : {result.budget} simulations",
        f"simulations : {result.simulations} actually run",
        f"min robustness: {result.best_robustness:+.6f} (dimensionless; "
        "negative exactly when the requirement is violated)",
    ]
    if result.found:
        lines.append(f"VIOLATION FOUND at simulation {result.first_violation}")
        lines.append("counterexample (the decision vector that violates the requirement):")
        for name, value in zip(LoopInput.FIELDS, result.best_vector, strict=True):
            lines.append(f"  {name:<16s} {value:+.6f}")
    else:
        lines.append(f"NO VIOLATION FOUND within {result.budget} simulations.")
        lines.append(ONE_SIDED_NOTE)
        lines.append(
            "The search spent its budget and returned nothing. That is a statement about "
            "the search, not about the requirement."
        )
    return "\n".join(lines)


def render_instances(instances: tuple[Instance, ...]) -> str:
    """Render the instance suite, one block each."""
    return "\n\n".join(inst.describe() for inst in instances)


def render_cell_table(report: BenchmarkReport) -> str:
    """Per-instance, per-strategy table. The primary report.

    Columns: the fraction of runs that found a violation within the budget,
    median simulations to the first
    violation (``> budget`` when censored), and the mean curve probability.
    """
    header = (
        f"{'instance':<20s} {'strategy':<20s} {'found/runs':>10s} "
        f"{'rate':>6s} {'median sims':>12s} {'mean P':>8s} {'min rho':>10s}"
    )
    lines = [
        f"budget {report.budget} simulations, {report.repeats} seeds per cell, "
        f"base seed {report.base_seed}",
        header,
        "-" * len(header),
    ]
    for iid in report.instance_ids:
        for name in report.strategy_names:
            cell = report.cell(iid, name)
            found = sum(1 for v in cell.first_violations if v is not None)
            lines.append(
                f"{iid:<20s} {name:<20s} {found:>5d}/{cell.repeats:<4d} "
                f"{cell.success_rate:>6.3f} "
                f"{_fmt_optional(cell.median_simulations, cell.budget):>12s} "
                f"{cell.mean_curve_probability:>8.4f} {cell.best_robustness:>+10.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_aggregate_table(report: BenchmarkReport) -> str:
    """Aggregate view, with the baseline first and the per-instance losses named."""
    header = (
        f"{'strategy':<20s} {'mean P (aggregate)':>19s} {'vs baseline':>12s} "
        f"{'instances where the baseline wins':<40s}"
    )
    lines = [header, "-" * len(header)]
    base = report.aggregate_mean_probability(BASELINE)
    for name in report.strategy_names:
        value = report.aggregate_mean_probability(name)
        if name == BASELINE:
            losses = "(is the baseline)"
            delta = "--"
        else:
            lost = report.baseline_wins(name)
            losses = ", ".join(lost) if lost else "none"
            delta = f"{value - base:+.4f}"
        lines.append(f"{name:<20s} {value:>19.4f} {delta:>12s} {losses:<40s}")
    lines.append("")
    lines.append(
        "The aggregate is the unweighted mean of the per-instance curves. Read the "
        "per-instance table first: a strategy can win here and lose on the instance "
        "you care about."
    )
    lines.append(
        f"Hardest instance for the baseline: {report.hardest_instance()}."
    )
    lines.append(ONE_SIDED_NOTE)
    return "\n".join(lines)


def render_difficulty_table(
    rows: tuple[tuple[str, str, int, int, float], ...], alpha: float = 0.05
) -> str:
    """Render measured instance difficulty with exact binomial intervals.

    Parameters
    ----------
    rows:
        ``(instance_id, tier, violations, draws, design_target)`` per instance.
    alpha:
        Two-sided miss rate for the Clopper-Pearson interval.
    """
    header = (
        f"{'instance':<20s} {'tier':<10s} {'violations':>11s} {'draws':>8s} "
        f"{'p measured':>11s} {'95% CI':>24s} {'design target':>14s}"
    )
    lines = [header, "-" * len(header)]
    for iid, tier, k, n, target in rows:
        lo, hi = clopper_pearson(k, n, alpha)
        p = k / n
        lines.append(
            f"{iid:<20s} {tier:<10s} {k:>11d} {n:>8d} {p:>11.6f} "
            f"[{lo:>9.6f}, {hi:>9.6f}] {target:>14.4f}"
        )
    return "\n".join(lines)


def render_curve_points(
    curve: np.ndarray, lower: np.ndarray, upper: np.ndarray, marks: tuple[int, ...]
) -> str:
    """Render an efficiency curve and its band at selected simulation counts.

    Parameters
    ----------
    curve, lower, upper:
        Arrays of equal length ``budget``.
    marks:
        1-based simulation counts to tabulate.
    """
    if not (curve.shape == lower.shape == upper.shape):
        raise ValueError(
            f"curve, lower and upper must have the same shape, got "
            f"{curve.shape}, {lower.shape}, {upper.shape}"
        )
    header = f"{'n sims':>7s} {'P(found by n)':>14s} {'95% band':>24s}"
    lines = [header, "-" * len(header)]
    for n in marks:
        if not 1 <= n <= curve.size:
            raise ValueError(f"mark {n} is outside 1..{curve.size}")
        i = n - 1
        lines.append(
            f"{n:>7d} {curve[i]:>14.4f} [{lower[i]:>9.4f}, {upper[i]:>9.4f}]"
        )
    return "\n".join(lines)


def render_cell_detail(cell: CellSummary) -> str:
    """Render the raw first-violation indices of one cell, censored runs included."""
    entries = ", ".join("none" if v is None else str(v) for v in cell.first_violations)
    return (
        f"{cell.instance_id} / {cell.strategy_name}: "
        f"budget {cell.budget}, {cell.repeats} seeds\n"
        f"first-violation simulation index per seed: [{entries}]\n"
        f"'none' marks a run that found nothing. {ONE_SIDED_NOTE}"
    )
