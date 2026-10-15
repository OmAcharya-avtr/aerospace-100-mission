"""Text rendering. Every function returns a string; nothing here prints.

The command-line interface in :mod:`twincalib.__main__` is the only part of the
package that writes to a stream. These renderers exist so that the same text
appears in the CLI, in the examples and in the validation scripts without being
written three times, and so that a report can be asserted on in a test.

House rule for every renderer here: a parameter value is never printed without
the identifiability verdict that qualifies it. That is the whole point of the
package, and a formatter that can print a number on its own would undo it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .fitting import FitResult
from .identifiability import IdentifiabilityReport
from .profile import ProfileCurve
from .scenarios import ExperimentDesign
from .sensitivity import SensitivityMatrix

__all__ = [
    "render_correlation",
    "render_design",
    "render_fit",
    "render_identifiability",
    "render_profiles",
    "render_sensitivity",
]


def _fmt(value: float, width: int = 12) -> str:
    """A number in a fixed field, switching to exponent form when it has to."""
    if not np.isfinite(value):
        return f"{'inf' if value > 0 else '-inf':>{width}}"
    if value != 0.0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:>{width}.4e}"
    return f"{value:>{width}.5g}"


def render_design(d: ExperimentDesign) -> str:
    """One block describing an experiment design."""
    lines = [
        f"design            : {d.name}",
        f"measured channels : {', '.join(d.channel_labels)}",
        f"record            : {d.duration:g} s at dt = {d.dt:g} s, "
        f"stride {d.sample_stride} -> {d.n_samples} samples per channel",
        f"observations      : {d.n_observations} scalar values",
        f"thermometer noise : {d.sigma_k:g} K, independent Gaussian, known to the fitter",
    ]
    if d.sun_constant is None:
        lines.append(
            f"illumination      : orbital, period {d.orbit_period:g} s, "
            f"eclipse fraction {d.eclipse_fraction:g}"
        )
    else:
        lines.append(f"illumination      : CONSTANT at {d.sun_constant:g}")
    if d.dissipation_high == d.dissipation_low:
        lines.append(f"unit dissipation  : CONSTANT at {d.dissipation_high:g} W")
    else:
        lines.append(
            f"unit dissipation  : square wave {d.dissipation_low:g}/"
            f"{d.dissipation_high:g} W, period {d.duty_period:g} s, "
            f"duty {d.duty_fraction:g}"
        )
    if d.theta_override:
        items = ", ".join(f"{k} = {v:g}" for k, v in sorted(d.theta_override.items()))
        lines.append(f"parameter override: {items}")
    lines.append(f"note              : {d.description}")
    return "\n".join(lines)


def render_sensitivity(sens: SensitivityMatrix) -> str:
    """Per-parameter sensitivity magnitudes."""
    lines = [
        "per-parameter sensitivity (Brun et al. 2001 delta_msqr)",
        "  a value of 1 means a 100 per cent change in the parameter moves the",
        "  data by about one measurement standard deviation per sample",
        "",
        f"  {'parameter':<10}{'delta_msqr':>14}{'column norm':>14}",
    ]
    delta = sens.delta_msqr()
    norms = sens.column_norms()
    for name, dv, nv in zip(sens.parameter_names, delta, norms, strict=True):
        lines.append(f"  {name:<10}{_fmt(float(dv), 14)}{_fmt(float(nv), 14)}")
    return "\n".join(lines)


def render_identifiability(
    report: IdentifiabilityReport, min_weight: float = 0.05
) -> str:
    """The identifiability analysis, verdict first."""
    lines = [
        f"VERDICT: {report.verdict()}",
        "",
        f"rank              : {report.rank} of {report.n_parameters} "
        f"at s/s_max >= {report.rank_tol:g}",
        f"condition number  : {report.condition_number:.6g}",
        f"collinearity index: {report.collinearity:.6g}  "
        "(Brun et al. 2001; above 10-15 is their non-identifiable range)",
        "",
        "singular spectrum of the weighted log-sensitivity matrix",
        f"  {'i':>3}{'singular value':>18}{'s_i/s_max':>14}",
    ]
    s_max = float(report.singular_values[0])
    for i, sv in enumerate(report.singular_values):
        lines.append(f"  {i:>3}{_fmt(float(sv), 18)}{_fmt(float(sv) / s_max, 14)}")
    weak = report.weak_combinations
    lines.append("")
    if weak:
        lines.append(
            f"{len(weak)} unconstrained or near-unconstrained parameter "
            "combination(s), written out:"
        )
        for c in weak:
            tag = "UNCONSTRAINED" if c.ratio < report.rank_tol else "near-null"
            lines.append(f"  direction {c.index}  s/s_max = {c.ratio:.4e}  [{tag}]")
            lines.append(f"    {c.as_text(min_weight)}")
            lines.append(
                "    moving the parameters along this combination leaves the "
                "predictions"
            )
            lines.append(
                f"    unchanged to within {c.singular_value:.4e} noise units "
                "per unit log step"
            )
    else:
        lines.append(
            "no parameter combination has s/s_max below "
            f"{report.weak_ratio:g}; nothing is reported as unconstrained at "
            "this point"
        )
    lines.append("")
    lines.append("Cramer-Rao relative standard deviations (lower bound, log coordinates)")
    lines.append(f"  {'parameter':<10}{'rel. sd':>14}{'delta_msqr':>14}")
    for name, sd, dv in zip(
        report.parameter_names, report.relative_sd, report.delta_msqr, strict=True
    ):
        lines.append(f"  {name:<10}{_fmt(float(sd), 14)}{_fmt(float(dv), 14)}")
    if report.pseudo_inverse_used:
        lines.append("")
        lines.append(
            "  the Fisher matrix is rank deficient, so the above is a "
            "pseudo-inverse; the"
        )
        lines.append(
            "  entries marked inf are the parameters loading on an "
            "unconstrained direction"
        )
    return "\n".join(lines)


def render_correlation(
    names: Sequence[str], correlation: np.ndarray, flag_above: float = 0.95
) -> str:
    """A correlation matrix with the strongly correlated pairs called out."""
    corr = np.asarray(correlation, dtype=float)
    n = len(names)
    if corr.shape != (n, n):
        raise ValueError(f"correlation must be {(n, n)}, got {corr.shape}")
    width = max(8, max(len(s) for s in names) + 2)
    lines = ["parameter correlation", "  " + "".join(f"{'':<{width}}") + "".join(
        f"{s:>{width}}" for s in names
    )]
    for i, row_name in enumerate(names):
        cells = "".join(
            f"{corr[i, j]:>{width}.4f}" if np.isfinite(corr[i, j]) else f"{'nan':>{width}}"
            for j in range(n)
        )
        lines.append(f"  {row_name:<{width}}{cells}")
    flagged = [
        (names[i], names[j], float(corr[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= flag_above
    ]
    lines.append("")
    if flagged:
        lines.append(f"pairs with |correlation| >= {flag_above:g}:")
        for a, b, v in sorted(flagged, key=lambda t: -abs(t[2])):
            lines.append(f"  {a:<8} {b:<8} {v:+.6f}")
    else:
        lines.append(f"no pair reaches |correlation| = {flag_above:g}")
    return "\n".join(lines)


def render_fit(
    result: FitResult,
    units: Sequence[str] | None = None,
    theta_true: Sequence[float] | None = None,
) -> str:
    """The fit, its caveats first so that they cannot be skipped."""
    lines: list[str] = []
    caveats = result.caveats()
    if caveats:
        lines.append("CAVEATS, which travel with every number below:")
        for c in caveats:
            lines.append(f"  - {c}")
        lines.append("")
    unit_list = list(units) if units is not None else [""] * result.theta.size
    if len(unit_list) != result.theta.size:
        raise ValueError("units must have one entry per parameter")
    header = f"  {'parameter':<10}{'estimate':>14}{'rel. sd':>12}"
    if theta_true is not None:
        header += f"{'true':>14}{'rel. error':>12}"
    header += "   unit"
    lines.append("weighted least-squares estimate")
    lines.append(header)
    rel_err = (
        result.relative_error(theta_true) if theta_true is not None else None
    )
    for i, name in enumerate(result.parameter_names):
        row = (
            f"  {name:<10}{_fmt(float(result.theta[i]), 14)}"
            f"{_fmt(float(result.relative_sd[i]), 12)}"
        )
        if theta_true is not None and rel_err is not None:
            row += f"{_fmt(float(theta_true[i]), 14)}{_fmt(float(rel_err[i]), 12)}"
        row += f"   {unit_list[i]}"
        lines.append(row)
    lines.append("")
    lines.append(f"chi-squared        : {result.chi2:.6g}")
    lines.append(f"reduced chi-squared: {result.reduced_chi2:.6g}  (about 1 if the model is right)")
    lines.append(f"residual RMS       : {result.rmse:.6g} K")
    lines.append(f"residual evaluations: {result.n_function_evaluations}")
    lines.append(f"converged          : {result.success}  ({result.message})")
    lines.append("")
    lines.append(
        "the relative standard deviations above are a LOCAL QUADRATIC "
        "approximation; read"
    )
    lines.append("the profile intervals before reporting any of them")
    return "\n".join(lines)


def render_profiles(curves: Sequence[ProfileCurve]) -> str:
    """Profile intervals beside the quadratic intervals they correct."""
    if not curves:
        raise ValueError("at least one profile curve is required")
    level = curves[0].level
    lines = [
        f"profile-likelihood intervals at level {level:g} "
        f"(Wilks threshold delta chi2 = {curves[0].threshold:.4f})",
        "",
        f"  {'parameter':<10}{'estimate':>13}{'profile low':>14}{'profile high':>14}"
        f"{'quad low':>14}{'quad high':>14}{'width ratio':>13}",
    ]
    for c in curves:
        lo = "OPEN" if c.lower is None else _fmt(c.lower, 14).strip()
        hi = "OPEN" if c.upper is None else _fmt(c.upper, 14).strip()
        qlo = "n/a" if c.hessian_lower is None else _fmt(c.hessian_lower, 14).strip()
        qhi = "n/a" if c.hessian_upper is None else _fmt(c.hessian_upper, 14).strip()
        ratio = c.width_ratio
        rtxt = "inf" if np.isinf(ratio) else ("nan" if np.isnan(ratio) else f"{ratio:.4g}")
        lines.append(
            f"  {c.name:<10}{_fmt(c.estimate, 13)}{lo:>14}{hi:>14}"
            f"{qlo:>14}{qhi:>14}{rtxt:>13}"
        )
    open_sides = [c for c in curves if not c.bounded]
    lines.append("")
    if open_sides:
        lines.append(
            "OPEN means the profile did not cross the threshold inside the "
            "searched range:"
        )
        for c in open_sides:
            which = []
            if c.lower is None:
                which.append("below")
            if c.upper is None:
                which.append("above")
            lines.append(
                f"  {c.name}: open {' and '.join(which)}; searched "
                f"[{_fmt(c.search_lower).strip()}, "
                f"{_fmt(c.search_upper).strip()}] and the likelihood did not "
                f"rise by {c.threshold:.4f} on the open side"
            )
        lines.append(
            "  a parameter with an open side is NOT determined by this data set, "
            "whatever"
        )
        lines.append("  quadratic standard deviation the covariance matrix reports for it")
    else:
        lines.append("every profile closed on both sides inside the searched range")
    worst = max(
        (c for c in curves if np.isfinite(c.width_ratio)),
        key=lambda c: c.width_ratio,
        default=None,
    )
    if worst is not None:
        lines.append("")
        lines.append(
            f"largest finite profile-to-quadratic width ratio: "
            f"{worst.width_ratio:.4g} on {worst.name}"
        )
    return "\n".join(lines)
