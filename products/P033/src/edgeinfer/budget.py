"""The declared budget as a first-class object, with pass/fail and uncertainty.

A deployment decision for an aerospace edge computer is a budget question:
does the candidate model fit inside a declared envelope of latency, peak
memory, power and duty cycle, and how confident is that answer? This module
makes the envelope an object that can be stated once, checked against an
analytic estimate and against a measurement, and reported with the
uncertainty that the measurement actually supports.

Two deliberate choices
----------------------
**Worst case and median are separate limits.** A control loop misses its
deadline on the tail, not on the median, so :class:`Budget` carries
``latency_s`` (a worst-case ceiling, checked against a high quantile) and an
optional ``median_latency_s`` (checked against p50). A model that passes on
the median and fails on the tail is reported as a failure, and the report says
which limit failed.

**A verdict carries its uncertainty.** ``PASS``/``FAIL`` is decided against
the point estimate; a separate ``margin_sigma`` says how many combined
standard uncertainties separate the estimate from the limit, and the verdict
is annotated ``MARGINAL`` when that is below
:data:`MARGINAL_SIGMA_THRESHOLD`. A pass by less than the measurement noise is
not a pass anyone should act on.

Duty cycle and power
--------------------
``duty_cycle`` is the fraction of each control period the inference may
occupy, dimensionless in (0, 1]. With a declared ``period_s`` the implied
latency ceiling is ``duty_cycle * period_s``; when that is tighter than
``latency_s`` the budget is **infeasible by construction** and
:meth:`Budget.feasibility` says so before any model is measured. Energy per
inference follows the definition ``E = P * t`` [J] (SI; power is the time
derivative of energy, so average power over an interval times the interval is
the energy in it).

Power is **not measured** by this package in any environment it currently
runs in: there is no instrumented rail in a cloud container, and a Jetson
Orin Nano's own INA3221 rails would have to be read on the device. Power
limits are therefore checked only against a caller-declared figure, and
:class:`BudgetReport` marks the power row ``NOT MEASURED`` so the distinction
cannot be lost in a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "MARGINAL_SIGMA_THRESHOLD",
    "Budget",
    "BudgetReport",
    "CheckRow",
    "Feasibility",
    "Verdict",
    "build_report",
]

#: Below this many combined standard uncertainties of margin, a pass or a fail
#: is annotated ``MARGINAL``.
MARGINAL_SIGMA_THRESHOLD: float = 2.0


class Verdict(StrEnum):
    """Outcome of one budget check."""

    PASS = "PASS"
    FAIL = "FAIL"
    MARGINAL_PASS = "MARGINAL PASS"
    MARGINAL_FAIL = "MARGINAL FAIL"
    NOT_MEASURED = "NOT MEASURED"
    NOT_DECLARED = "NOT DECLARED"

    @property
    def is_pass(self) -> bool:
        """True for ``PASS`` and ``MARGINAL PASS`` only."""
        return self in (Verdict.PASS, Verdict.MARGINAL_PASS)


@dataclass(frozen=True)
class Feasibility:
    """Whether a budget can be satisfied by any model at all.

    Attributes
    ----------
    feasible
        False when the declared limits contradict each other.
    reasons
        One line per contradiction, empty when feasible.
    """

    feasible: bool
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.feasible


@dataclass(frozen=True)
class CheckRow:
    """One line of a budget report.

    Attributes
    ----------
    quantity
        What is checked, e.g. ``"worst-case latency (p99)"``.
    unit
        Unit of ``value`` and ``limit``.
    value
        Point estimate, or ``None`` when not measured.
    limit
        Declared limit, or ``None`` when not declared.
    uncertainty
        Combined standard uncertainty of ``value`` in the same unit, or
        ``None``.
    verdict
        See :class:`Verdict`.
    method
        How ``value`` was obtained, verbatim into the report.
    """

    quantity: str
    unit: str
    value: float | None
    limit: float | None
    uncertainty: float | None
    verdict: Verdict
    method: str

    @property
    def margin(self) -> float | None:
        """``limit - value`` in the row's unit; positive means inside budget."""
        if self.value is None or self.limit is None:
            return None
        return self.limit - self.value

    @property
    def margin_sigma(self) -> float | None:
        """Margin in combined standard uncertainties; ``None`` if unavailable."""
        margin = self.margin
        if margin is None or self.uncertainty is None or self.uncertainty <= 0:
            return None
        return margin / self.uncertainty

    @property
    def utilisation(self) -> float | None:
        """``value / limit`` [dimensionless]; > 1 means over budget."""
        if self.value is None or self.limit is None or self.limit == 0:
            return None
        return self.value / self.limit


@dataclass(frozen=True)
class Budget:
    """A declared inference envelope.

    Parameters
    ----------
    name
        Budget name, carried into the report.
    latency_s
        Worst-case latency ceiling [s], strictly positive. Checked against the
        measured high quantile named by ``worst_case_quantile``.
    peak_memory_bytes
        Peak resident-plus-activation memory ceiling [B], strictly positive.
    median_latency_s
        Optional median latency ceiling [s]. Must not exceed ``latency_s``.
    power_w
        Optional average power ceiling [W]. Never measured by this package;
        see the module docstring.
    duty_cycle
        Optional fraction of each period the inference may occupy
        [dimensionless, (0, 1]].
    period_s
        Optional control period [s], strictly positive. Needed to turn
        ``duty_cycle`` into a latency ceiling.
    worst_case_quantile
        Quantile of the latency distribution that ``latency_s`` applies to
        [dimensionless, (0.5, 1.0)]. Default 0.99.
    """

    name: str
    latency_s: float
    peak_memory_bytes: int
    median_latency_s: float | None = None
    power_w: float | None = None
    duty_cycle: float | None = None
    period_s: float | None = None
    worst_case_quantile: float = 0.99

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Budget.name must be non-empty")
        if not np.isfinite(self.latency_s) or self.latency_s <= 0:
            raise ValueError(f"latency_s must be finite and > 0 s, got {self.latency_s}")
        if self.peak_memory_bytes <= 0:
            raise ValueError(
                f"peak_memory_bytes must be > 0 B, got {self.peak_memory_bytes}"
            )
        if self.median_latency_s is not None and (
            not np.isfinite(self.median_latency_s) or self.median_latency_s <= 0
        ):
            raise ValueError(
                f"median_latency_s must be finite and > 0 s, got {self.median_latency_s}"
            )
        if self.power_w is not None and (not np.isfinite(self.power_w) or self.power_w <= 0):
            raise ValueError(f"power_w must be finite and > 0 W, got {self.power_w}")
        if self.duty_cycle is not None and not 0 < self.duty_cycle <= 1:
            raise ValueError(
                f"duty_cycle must lie in (0, 1] (dimensionless), got {self.duty_cycle}"
            )
        if self.period_s is not None and (not np.isfinite(self.period_s) or self.period_s <= 0):
            raise ValueError(f"period_s must be finite and > 0 s, got {self.period_s}")
        if not 0.5 < self.worst_case_quantile < 1.0:
            raise ValueError(
                f"worst_case_quantile must lie in (0.5, 1.0), got {self.worst_case_quantile}"
            )

    @property
    def duty_cycle_latency_s(self) -> float | None:
        """Latency ceiling implied by ``duty_cycle * period_s`` [s]."""
        if self.duty_cycle is None or self.period_s is None:
            return None
        return self.duty_cycle * self.period_s

    @property
    def effective_latency_s(self) -> float:
        """The binding worst-case latency ceiling [s].

        The smaller of ``latency_s`` and the duty-cycle-implied ceiling.
        """
        implied = self.duty_cycle_latency_s
        return self.latency_s if implied is None else min(self.latency_s, implied)

    def feasibility(self) -> Feasibility:
        """Check the declared limits against each other.

        Catches the budget that is infeasible by construction: a median
        ceiling above the worst-case ceiling, or a duty cycle that leaves less
        time than the declared worst-case latency.
        """
        reasons: list[str] = []
        if self.median_latency_s is not None and self.median_latency_s > self.latency_s:
            reasons.append(
                f"median ceiling {self.median_latency_s:.6g} s exceeds worst-case ceiling "
                f"{self.latency_s:.6g} s; a median above the tail limit cannot be met"
            )
        implied = self.duty_cycle_latency_s
        if implied is not None and implied < self.latency_s:
            reasons.append(
                f"duty cycle {self.duty_cycle:.4g} of period {self.period_s:.6g} s allows "
                f"{implied:.6g} s, which is less than the declared worst-case latency "
                f"{self.latency_s:.6g} s; the two limits contradict"
            )
        if implied is not None and self.median_latency_s is not None:
            if self.median_latency_s > implied:
                reasons.append(
                    f"median ceiling {self.median_latency_s:.6g} s exceeds the "
                    f"duty-cycle-implied ceiling {implied:.6g} s"
                )
        return Feasibility(feasible=not reasons, reasons=tuple(reasons))

    def energy_per_inference_j(self, latency_s: float) -> float | None:
        """``power_w * latency_s`` [J], or ``None`` if no power is declared.

        Uses the declared power ceiling, not a measurement. ``E = P t`` (SI).
        """
        if self.power_w is None:
            return None
        if latency_s < 0:
            raise ValueError("latency_s must be non-negative")
        return self.power_w * latency_s

    def summary_lines(self) -> list[str]:
        """Human-readable declaration, every limit with its unit."""
        lines = [
            f"budget                  : {self.name}",
            f"worst-case latency      : <= {self.latency_s * 1e3:.4f} ms "
            f"at p{self.worst_case_quantile * 100:g}",
        ]
        if self.median_latency_s is not None:
            lines.append(
                f"median latency          : <= {self.median_latency_s * 1e3:.4f} ms at p50"
            )
        lines.append(f"peak memory             : <= {self.peak_memory_bytes} B")
        if self.power_w is not None:
            lines.append(
                f"average power           : <= {self.power_w:.4g} W (declared, NOT MEASURED)"
            )
        if self.duty_cycle is not None:
            lines.append(f"duty cycle              : <= {self.duty_cycle:.4g} (dimensionless)")
        if self.period_s is not None:
            lines.append(f"control period          : {self.period_s * 1e3:.4f} ms")
        implied = self.duty_cycle_latency_s
        if implied is not None:
            lines.append(f"duty-cycle ceiling      : {implied * 1e3:.4f} ms")
        return lines


def _verdict(
    value: float | None, limit: float | None, uncertainty: float | None
) -> Verdict:
    if limit is None:
        return Verdict.NOT_DECLARED
    if value is None:
        return Verdict.NOT_MEASURED
    inside = value <= limit
    if uncertainty is not None and uncertainty > 0:
        sigma = abs(limit - value) / uncertainty
        if sigma < MARGINAL_SIGMA_THRESHOLD:
            return Verdict.MARGINAL_PASS if inside else Verdict.MARGINAL_FAIL
    return Verdict.PASS if inside else Verdict.FAIL


@dataclass(frozen=True)
class BudgetReport:
    """The result of checking one candidate against one budget.

    Attributes
    ----------
    budget_name
        Name of the budget checked.
    candidate
        Name of the model or pipeline checked.
    environment
        One line naming the machine and its sharing state, written into every
        report so a number is never read without its environment.
    rows
        One :class:`CheckRow` per checked quantity.
    feasibility
        The budget's own self-consistency check.
    notes
        Free lines appended to the report, e.g. what is still unmeasured.
    """

    budget_name: str
    candidate: str
    environment: str
    rows: tuple[CheckRow, ...]
    feasibility: Feasibility
    notes: tuple[str, ...] = ()

    @property
    def overall(self) -> Verdict:
        """Worst verdict over the decided rows.

        ``NOT_MEASURED``/``NOT_DECLARED`` rows do not decide the outcome, but
        they are listed in :attr:`undecided`. An infeasible budget fails
        outright.
        """
        if not self.feasibility.feasible:
            return Verdict.FAIL
        decided = [
            r.verdict
            for r in self.rows
            if r.verdict not in (Verdict.NOT_MEASURED, Verdict.NOT_DECLARED)
        ]
        if not decided:
            return Verdict.NOT_MEASURED
        for worst in (Verdict.FAIL, Verdict.MARGINAL_FAIL, Verdict.MARGINAL_PASS):
            if worst in decided:
                return worst
        return Verdict.PASS

    @property
    def undecided(self) -> tuple[str, ...]:
        """Quantities that could not be decided, with the reason."""
        return tuple(
            f"{r.quantity}: {r.verdict.value}"
            for r in self.rows
            if r.verdict in (Verdict.NOT_MEASURED, Verdict.NOT_DECLARED)
        )

    def to_table(self) -> str:
        """Fixed-width table of every row, units included."""
        header = (
            f"{'quantity':<28} {'value':>14} {'u(value)':>12} {'limit':>14} "
            f"{'util':>7} {'verdict':<14} method"
        )
        lines = [header, "-" * len(header)]
        for row in self.rows:
            value = "-" if row.value is None else f"{row.value:.6g}"
            unc = "-" if row.uncertainty is None else f"{row.uncertainty:.3g}"
            limit = "-" if row.limit is None else f"{row.limit:.6g}"
            util = "-" if row.utilisation is None else f"{row.utilisation * 100:.1f}%"
            lines.append(
                f"{row.quantity + ' [' + row.unit + ']':<28} {value:>14} {unc:>12} "
                f"{limit:>14} {util:>7} {row.verdict.value:<14} {row.method}"
            )
        return "\n".join(lines)

    def summary_lines(self) -> list[str]:
        """Full report text including environment, feasibility and notes."""
        lines = [
            f"candidate   : {self.candidate}",
            f"budget      : {self.budget_name}",
            f"environment : {self.environment}",
            "",
            *self.to_table().splitlines(),
            "",
            f"overall     : {self.overall.value}",
        ]
        if not self.feasibility.feasible:
            lines.append("budget is INFEASIBLE BY CONSTRUCTION:")
            lines.extend(f"  - {r}" for r in self.feasibility.reasons)
        if self.undecided:
            lines.append("undecided rows:")
            lines.extend(f"  - {u}" for u in self.undecided)
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return lines


def build_report(
    budget: Budget,
    candidate: str,
    environment: str,
    *,
    worst_case_latency_s: float | None = None,
    worst_case_uncertainty_s: float | None = None,
    median_latency_s: float | None = None,
    median_uncertainty_s: float | None = None,
    peak_memory_bytes: float | None = None,
    peak_memory_uncertainty_bytes: float | None = None,
    declared_power_w: float | None = None,
    latency_method: str = "unspecified",
    memory_method: str = "unspecified",
    notes: tuple[str, ...] = (),
) -> BudgetReport:
    """Assemble a :class:`BudgetReport` from whatever has been measured.

    Any quantity left ``None`` produces a ``NOT MEASURED`` row rather than
    being dropped, so the report always shows the full envelope and what of it
    is still unknown.

    Parameters
    ----------
    budget
        The declared envelope.
    candidate
        Model or pipeline name.
    environment
        One line describing the machine the numbers came from.
    worst_case_latency_s, median_latency_s, peak_memory_bytes
        Point estimates [s, s, B].
    worst_case_uncertainty_s, median_uncertainty_s, peak_memory_uncertainty_bytes
        Combined standard uncertainties in the same units.
    declared_power_w
        A caller-declared average power [W]. Reported with verdict
        ``NOT MEASURED`` whatever its value, because this package does not
        measure power.
    latency_method, memory_method
        Measurement-method strings written into the report verbatim.
    notes
        Extra lines for the report.
    """
    quantile_label = f"worst-case latency (p{budget.worst_case_quantile * 100:g})"
    rows = [
        CheckRow(
            quantity=quantile_label,
            unit="s",
            value=worst_case_latency_s,
            limit=budget.effective_latency_s,
            uncertainty=worst_case_uncertainty_s,
            verdict=_verdict(
                worst_case_latency_s, budget.effective_latency_s, worst_case_uncertainty_s
            ),
            method=latency_method,
        ),
        CheckRow(
            quantity="median latency (p50)",
            unit="s",
            value=median_latency_s,
            limit=budget.median_latency_s,
            uncertainty=median_uncertainty_s,
            verdict=_verdict(median_latency_s, budget.median_latency_s, median_uncertainty_s),
            method=latency_method,
        ),
        CheckRow(
            quantity="peak memory",
            unit="B",
            value=peak_memory_bytes,
            limit=float(budget.peak_memory_bytes),
            uncertainty=peak_memory_uncertainty_bytes,
            verdict=_verdict(
                peak_memory_bytes,
                float(budget.peak_memory_bytes),
                peak_memory_uncertainty_bytes,
            ),
            method=memory_method,
        ),
        CheckRow(
            quantity="average power",
            unit="W",
            value=declared_power_w,
            limit=budget.power_w,
            uncertainty=None,
            verdict=Verdict.NOT_MEASURED,
            method="not measured by edgeinfer; declared value only",
        ),
    ]
    return BudgetReport(
        budget_name=budget.name,
        candidate=candidate,
        environment=environment,
        rows=tuple(rows),
        feasibility=budget.feasibility(),
        notes=notes,
    )
