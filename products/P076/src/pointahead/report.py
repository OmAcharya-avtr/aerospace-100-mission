"""Text rendering of a point-ahead result and its budget.

Returns strings. Nothing in this module prints, because library code that
prints cannot be used from a test, a notebook or another program's log. The
command-line interface in :mod:`pointahead.__main__` is the only place in this
package that writes to a stream.

Angles are rendered in microradians with six decimal places, which resolves
picoradians. That is far finer than any terminal can point and it is
deliberate: the budget lines span fourteen orders of magnitude, and rounding
them to something a gimbal could achieve would print a column of zeros where
the attribution is.
"""

from __future__ import annotations

from .angle import BaselineScore, PointAheadDecomposition, PointAheadResult
from .budget import ErrorBudget, SensitivityCheck

_WIDTH = 78


def _rule(title: str = "") -> str:
    if not title:
        return "-" * _WIDTH
    return f"-- {title} ".ljust(_WIDTH, "-")


def format_point_ahead(result: PointAheadResult) -> str:
    """One point-ahead result: the total, the offset and the method."""
    lines = [
        _rule(f"point-ahead ({result.method})"),
        f"  frame                      : {result.frame}",
        f"  total                      : {result.total_rad * 1.0e6:15.6f} urad",
        f"  transverse offset (e1, e2) : {result.offset_rad[0] * 1.0e6:15.6f} "
        f"{result.offset_rad[1] * 1.0e6:15.6f} urad",
    ]
    return "\n".join(lines)


def format_decomposition(decomposition: PointAheadDecomposition) -> str:
    """The three first-order magnitude contributions and their closure."""
    lines = [
        _rule("magnitude attribution (frame-dependent split, invariant total)"),
        f"  frame                      : {decomposition.frame}",
    ]
    for name, value in decomposition.as_lines():
        lines.append(f"  {name:26s} : {value * 1.0e6:15.6f} urad")
    lines.append(f"  {'vector sum total':26s} : {decomposition.total_rad * 1.0e6:15.6f} urad")
    lines.append(
        f"  {'closure vs reference':26s} : {decomposition.closure_rad * 1.0e6:15.6e} urad"
    )
    return "\n".join(lines)


def format_score(score: BaselineScore) -> str:
    """The closed-form baseline scored against the iterated reference."""
    lines = [
        _rule("baseline vs reference"),
        f"  closed form                : {score.closed_form_rad * 1.0e6:15.6f} urad",
        f"  iterated reference         : {score.reference_rad * 1.0e6:15.6f} urad",
        f"  magnitude difference       : {score.magnitude_difference_rad * 1.0e6:15.6e} urad",
        f"  offset difference (closed) : {score.offset_difference_rad * 1.0e6:15.6e} urad",
        f"  transmit direction (open)  : {score.transmit_direction_rad * 1.0e6:15.6e} urad",
    ]
    return "\n".join(lines)


def format_budget(budget: ErrorBudget) -> str:
    """The attributed error budget, one line per term."""
    inputs = budget.inputs
    lines = [
        _rule(f"error budget: {budget.link_label}"),
        f"  frame                      : {budget.frame}",
        f"  range                      : {budget.range_m / 1.0e3:15.3f} km",
        f"  point-ahead angle          : {budget.point_ahead_rad * 1.0e6:15.6f} urad",
        f"  angle method               : {inputs.angle_method}",
        f"  aberration route           : {inputs.aberration_route}",
        f"  light-time iterations      : {inputs.light_time_iterations}",
        f"  sigma v per axis           : {inputs.sigma_velocity_per_axis_mps:15.6f} m/s",
        f"  ephemeris age              : {inputs.ephemeris_age_s:15.3f} s",
        "",
        f"  {'term':<32s} {'paid /urad':>16s} {'if paid /urad':>16s}  origin",
        f"  {'-' * 32} {'-' * 16} {'-' * 16}  --------",
    ]
    for term in budget.terms:
        lines.append(
            f"  {term.name:<32s} {term.value_rad * 1.0e6:16.6e} "
            f"{term.measured_rad * 1.0e6:16.6e}  {term.origin}"
        )
    lines += [
        f"  {'-' * 32} {'-' * 16} {'-' * 16}",
        f"  {'total (root sum of squares)':<32s} {budget.total_rss_rad * 1.0e6:16.6e}",
        f"  {'total (linear sum)':<32s} {budget.total_linear_rad * 1.0e6:16.6e}",
        f"  dominant term              : {budget.dominant_term} "
        f"({budget.dominant_fraction:.4f} of the root sum of squares)",
        "",
        _rule("beam context"),
        f"  beam                       : {inputs.beam.label}",
        f"  divergence half-angle      : {budget.beam.divergence_half_angle_rad * 1.0e6:15.6f}"
        " urad (1/e^2 intensity)",
        f"  far field beyond           : {budget.beam.far_field_from_m:15.3f} m "
        f"(link is {'in' if budget.beam.in_far_field else 'NOT in'} the far field)",
        f"  total as fraction of beam  : {budget.beam.error_fraction_of_beam:15.6e}",
        f"  pointing loss at total     : {budget.beam.loss_db:15.6e} dB",
        f"  affordable at {inputs.loss_budget_db:.2f} dB         "
        f": {budget.beam.affordable_error_rad * 1.0e6:15.6f} urad",
        f"  headroom                   : {budget.beam.headroom:15.6e} x",
    ]
    return "\n".join(lines)


def format_sensitivity_checks(checks: tuple[SensitivityCheck, ...]) -> str:
    """The analytic-versus-numeric verification of each analytic budget line."""
    lines = [
        _rule("analytic lines checked against numerical propagation"),
        f"  {'term':<32s} {'analytic':>13s} {'numeric':>13s} {'rel':>11s}  check",
        f"  {'-' * 32} {'-' * 13} {'-' * 13} {'-' * 11}  -----",
    ]
    for check in checks:
        verdict = "pass" if check.passed else "FAIL"
        lines.append(
            f"  {check.name:<32s} {check.analytic_rad:13.6e} {check.numeric_rad:13.6e} "
            f"{check.relative_difference:11.4e}  {verdict} ({check.comparison})"
        )
    return "\n".join(lines)


def format_light_time(budget_label: str, receive, transmit) -> str:
    """The two light-time solutions and their iteration histories."""
    lines = [_rule(f"light-time solutions: {budget_label}")]
    for solution in (receive, transmit):
        lines.append(f"  {solution.direction}")
        lines.append(f"    tau                      : {solution.tau_s:.15e} s")
        lines.append(f"    light-time range         : {solution.range_m / 1.0e3:.6f} km")
        lines.append(f"    iterations               : {solution.iterations}")
        lines.append(f"    residual                 : {solution.residual_s:.6e} s")
        lines.append(f"    converged                : {solution.converged}")
        for index, value in enumerate(solution.iterates):
            lines.append(f"    iterate[{index}]               : {value:.15e} s")
    return "\n".join(lines)
