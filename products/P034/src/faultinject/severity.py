"""Severity scoring from the target's response.

Severity is a *design choice*, not a physical quantity, and this module states
the choice explicitly rather than hiding it behind a number.  The score
compares a faulted trace with the fault-free trace of the **same seed**, so the
measurement and process noise realisation is identical in both and cancels from
the deviation.  Nominal tracking lag therefore does not inflate the score.

Score definition
----------------
Let ``d_p = max_k |p_fault(k) - p_nominal(k)|`` (metres) and ``d_v`` the same
for velocity (metres per second).  Let ``rho`` be the ratio of the root mean
square tracking error of the faulted run to that of the nominal run, and ``phi``
the fraction of steps on which ``|p_fault| > POS_LIMIT``.  Then

    s_dev  = min(1, d_p / DEV_REF)
    s_rmse = min(1, max(0, rho - 1) / (RMSE_SAT - 1))
    s_viol = phi
    severity = clip(W_DEV*s_dev + W_RMSE*s_rmse + W_VIOL*s_viol, 0, 1)

Note that ``s_viol`` is **absolute**, not relative to the nominal run: a step
whose position exceeds ``POS_LIMIT`` counts as a violation whether or not the
injected fault caused it.  For this target the fault-free trajectory stays
inside 1.01 m against a 2.0 m limit, so a nominal run scores exactly zero; a
different target with a nominal excursion past the limit would carry a constant
floor, and the floor would be visible in the components rather than hidden.

The score has two overrides, applied in this order:

* any non-finite number in the faulted trace  ->  severity = 1.0
* ``max_k |p_fault(k)| > DIVERGENCE_LIMIT``    ->  severity = max(severity, 0.9)

The weights and reference scales below are the only tunables, they are module
constants rather than arguments so that every number in the README refers to
the same scoring function, and ``SEVERE_THRESHOLD`` is the single threshold at
which a case counts as "severe" in the campaign benchmark.

What this is not: a hazard, criticality or DAL assessment.  It is a scalar
ordering over observed closed-loop responses of one analysable target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .harness import Trace

DEV_REF = 0.5
"""Position deviation from nominal that saturates the deviation term, metres."""

RMSE_SAT = 5.0
"""Tracking-RMSE ratio that saturates the RMSE term, dimensionless."""

POS_LIMIT = 2.0
"""Position beyond which a step counts as a constraint violation, metres."""

DIVERGENCE_LIMIT = 1.0e3
"""Position beyond which the run is called divergent, metres."""

W_DEV = 0.5
W_RMSE = 0.3
W_VIOL = 0.2
"""Component weights. They sum to 1 by construction; checked in the tests."""

SEVERE_THRESHOLD = 0.6
"""Severity at or above which a case is counted as severe."""

LABELS: tuple[tuple[float, str], ...] = (
    (0.1, "negligible"),
    (0.3, "minor"),
    (0.6, "moderate"),
    (float("inf"), "severe"),
)


def label_for(severity: float) -> str:
    """Categorical label for a severity in ``[0, 1]``."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    for upper, name in LABELS:
        if severity < upper:
            return name
    raise AssertionError("unreachable: LABELS must end with inf")  # pragma: no cover


def _rms(xs: list[float]) -> float:
    if not xs:
        return 0.0
    total = 0.0
    for x in xs:
        total += x * x
    return math.sqrt(total / len(xs))


@dataclass(frozen=True)
class SeverityReport:
    """Severity score plus every input that produced it."""

    severity: float
    label: str
    max_pos_deviation: float
    max_vel_deviation: float
    rmse_ratio: float
    violation_fraction: float
    nonfinite: bool
    divergent: bool
    components: dict[str, float] = field(default_factory=dict)

    @property
    def severe(self) -> bool:
        """True if ``severity >= SEVERE_THRESHOLD``."""
        return self.severity >= SEVERE_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form."""
        return {
            "severity": self.severity,
            "label": self.label,
            "max_pos_deviation": self.max_pos_deviation,
            "max_vel_deviation": self.max_vel_deviation,
            "rmse_ratio": self.rmse_ratio,
            "violation_fraction": self.violation_fraction,
            "nonfinite": self.nonfinite,
            "divergent": self.divergent,
            "components": dict(self.components),
        }


def score(faulted: Trace, nominal: Trace) -> SeverityReport:
    """Severity of ``faulted`` relative to the fault-free ``nominal`` trace.

    Both traces must have the same seed and length; otherwise the deviation is
    not a fault effect and a ``ValueError`` is raised rather than a misleading
    number returned.
    """
    if faulted.seed != nominal.seed:
        raise ValueError(
            f"severity needs matching seeds: faulted={faulted.seed} nominal={nominal.seed}"
        )
    if faulted.n_steps != nominal.n_steps:
        raise ValueError(
            f"severity needs matching lengths: {faulted.n_steps} vs {nominal.n_steps}"
        )

    nonfinite = not faulted.finite()
    d_p = 0.0
    d_v = 0.0
    violations = 0
    max_abs_p = 0.0
    for pf, pn, vf, vn in zip(
        faulted.p_true, nominal.p_true, faulted.v_true, nominal.v_true, strict=True
    ):
        if math.isfinite(pf) and math.isfinite(pn):
            d_p = max(d_p, abs(pf - pn))
        if math.isfinite(vf) and math.isfinite(vn):
            d_v = max(d_v, abs(vf - vn))
        if math.isfinite(pf):
            max_abs_p = max(max_abs_p, abs(pf))
            if abs(pf) > POS_LIMIT:
                violations += 1
        else:
            violations += 1
    violation_fraction = violations / faulted.n_steps

    err_f = [e for e in faulted.tracking_error if math.isfinite(e)]
    rms_n = _rms(nominal.tracking_error)
    rms_f = _rms(err_f)
    rmse_ratio = rms_f / rms_n if rms_n > 0.0 else 1.0

    divergent = nonfinite or max_abs_p > DIVERGENCE_LIMIT

    s_dev = min(1.0, d_p / DEV_REF)
    s_rmse = min(1.0, max(0.0, rmse_ratio - 1.0) / (RMSE_SAT - 1.0))
    s_viol = violation_fraction
    sev = W_DEV * s_dev + W_RMSE * s_rmse + W_VIOL * s_viol
    sev = min(1.0, max(0.0, sev))
    if nonfinite:
        sev = 1.0
    elif max_abs_p > DIVERGENCE_LIMIT:
        sev = max(sev, 0.9)

    return SeverityReport(
        severity=sev,
        label=label_for(sev),
        max_pos_deviation=d_p,
        max_vel_deviation=d_v,
        rmse_ratio=rmse_ratio,
        violation_fraction=violation_fraction,
        nonfinite=nonfinite,
        divergent=divergent,
        components={"s_dev": s_dev, "s_rmse": s_rmse, "s_viol": s_viol},
    )
