"""Feature extraction for the learned screening classifier.

Every feature here is computable from what a caller actually has: the element
set text, the object's own element history, and element sets from other
sources. The uncorrupted reference element set is **never** a feature and is
never reachable from this module, because it is not available in an operational
setting; it exists in this repository only to measure consequence.

Three feature configurations are declared and all three are reported:

``elements_and_history``
    The default and the one the specification names. Element values, derived
    orbit geometry, checksum and parse validity, epoch age, the history
    screen's robust z-scores, and the cross-source separation in kilometres.

``elements_only``
    The same minus every history and cross-source feature. It measures what
    the history contributes rather than asserting that it contributes.

``with_rule_score``
    ``elements_and_history`` plus equation (15), the rule baseline's own risk
    score. This is the configuration that answers the question the product
    specification actually asks: given the rules, does a learned model add
    anything on top of them.

An infinite robust z-score is a real outcome — it is what a zero history scale
with a non-zero residual produces — and a tree ensemble cannot be fitted on
infinities. It is encoded as the declared cap :data:`Z_CAP` together with a
separate indicator column, so the information that the value was infinite
survives rather than being silently flattened into a large finite number.

Units
-----
Kilometres, days, degrees, revolutions per day, inverse Earth radii, and
dimensionless z-scores. No feature is standardised here; the tree ensemble does
not need it and standardising would hide the units.
"""

from __future__ import annotations

import math

import numpy as np

from . import bounds as bounds_module
from . import screens as screens_module
from .dataset import ELEMENT_SPANS, Record
from .history import LEVEL_ELEMENTS, TREND_ELEMENTS
from .verdict import classify

Z_CAP = 1000.0
"""Declared cap substituted for an infinite robust z-score."""

TESTED_ELEMENTS: tuple[str, ...] = LEVEL_ELEMENTS + TREND_ELEMENTS

BASE_NAMES: tuple[str, ...] = (
    "checksum_line1_ok",
    "checksum_line2_ok",
    "parse_fault_count",
    "line1_length_deviation",
    "line2_length_deviation",
    "non_digit_columns_in_element_spans",
    "eccentricity",
    "inclination_deg",
    "mean_motion_rev_per_day",
    "mean_motion_dot_rev_per_day2",
    "bstar",
    "bstar_sign",
    "log10_abs_bstar",
    "semi_major_axis_km",
    "perigee_altitude_km",
    "apogee_altitude_km",
    "physical_bound_findings",
    "policy_bound_findings",
    "format_bound_findings",
    "epoch_age_days",
    "epoch_age_over_requirement",
)

HISTORY_NAMES: tuple[str, ...] = (
    ("history_length", "history_usable")
    + tuple(f"z_{name}" for name in TESTED_ELEMENTS)
    + tuple(f"z_{name}_infinite" for name in TESTED_ELEMENTS)
    + (
        "z_max",
        "continuity_usable",
        "continuity_offset_km",
        "continuity_median_km",
        "continuity_scale_km",
        "continuity_z",
        "continuity_excess_km",
        "continuity_resolution_km",
        "continuity_resolves_threshold",
        "source_count",
        "cross_source_max_separation_km",
        "cross_source_usable",
    )
)

RULE_NAMES: tuple[str, ...] = ("rule_silent_risk_score",)

CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "elements_and_history": BASE_NAMES + HISTORY_NAMES,
    "elements_only": BASE_NAMES,
    "with_rule_score": BASE_NAMES + HISTORY_NAMES + RULE_NAMES,
}


def feature_names(configuration: str = "elements_and_history") -> tuple[str, ...]:
    """Column names of ``configuration``, in order.

    Raises
    ------
    ValueError
        If ``configuration`` is not declared.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(
            f"unknown configuration {configuration!r}; expected one of "
            f"{sorted(CONFIGURATIONS)}"
        )
    return CONFIGURATIONS[configuration]


def _cap(value: float) -> tuple[float, float]:
    """Return a finite value and an indicator that the original was infinite."""
    if value == math.inf:
        return Z_CAP, 1.0
    if not math.isfinite(value):
        return 0.0, 0.0
    return min(value, Z_CAP), 0.0


def _non_digit_columns(element_set) -> int:
    count = 0
    for line_number, spans in ELEMENT_SPANS.items():
        line = element_set.line1 if line_number == 1 else element_set.line2
        for low, high in spans:
            for index in range(low - 1, min(high, len(line))):
                character = line[index]
                if not (character.isdigit() or character in " .+-"):
                    count += 1
    return count


def row(record: Record, evidence: screens_module.ScreenEvidence) -> dict[str, float]:
    """All features for one record, as a name-to-value mapping.

    The mapping holds every feature of every configuration; a configuration
    selects its columns by name, so two configurations can never disagree
    about what a column means.
    """
    candidate = record.candidate
    derived = evidence.derived or bounds_module.derive(candidate)
    values: dict[str, float] = {
        "checksum_line1_ok": float(
            candidate.checksum_line1_stated == candidate.checksum_line1_computed
        ),
        "checksum_line2_ok": float(
            candidate.checksum_line2_stated == candidate.checksum_line2_computed
        ),
        "parse_fault_count": float(len(candidate.faults)),
        "line1_length_deviation": float(len(candidate.line1.rstrip()) - 69),
        "line2_length_deviation": float(len(candidate.line2.rstrip()) - 69),
        "non_digit_columns_in_element_spans": float(_non_digit_columns(candidate)),
        "eccentricity": _finite(candidate.eccentricity),
        "inclination_deg": _finite(candidate.inclination_deg),
        "mean_motion_rev_per_day": _finite(candidate.mean_motion_rev_per_day),
        "mean_motion_dot_rev_per_day2": _finite(candidate.mean_motion_dot_rev_per_day2),
        "bstar": _finite(candidate.bstar),
        "bstar_sign": float(np.sign(candidate.bstar)) if math.isfinite(candidate.bstar) else 0.0,
        "log10_abs_bstar": (
            math.log10(abs(candidate.bstar) + 1.0e-12)
            if math.isfinite(candidate.bstar)
            else -12.0
        ),
        "semi_major_axis_km": _finite(derived.semi_major_axis_km),
        "perigee_altitude_km": _finite(derived.perigee_altitude_km),
        "apogee_altitude_km": _finite(derived.apogee_altitude_km),
        "physical_bound_findings": float(
            sum(1 for f in evidence.bound_findings if f.kind == "physical")
        ),
        "policy_bound_findings": float(
            sum(1 for f in evidence.bound_findings if f.kind == "policy")
        ),
        "format_bound_findings": float(
            sum(1 for f in evidence.bound_findings if f.kind == "format")
        ),
        "epoch_age_days": (
            _finite(evidence.epoch.age_days) if evidence.epoch is not None else 0.0
        ),
        "epoch_age_over_requirement": (
            _finite(evidence.epoch.age_days / evidence.epoch.requirement_days)
            if evidence.epoch is not None and evidence.epoch.requirement_days > 0.0
            else 0.0
        ),
    }

    history = evidence.history
    values["history_length"] = float(history.history_length if history else 0)
    values["history_usable"] = float(bool(history and history.usable))
    capped: list[float] = []
    for name in TESTED_ELEMENTS:
        raw = history.z_scores.get(name, float("nan")) if history and history.usable else 0.0
        value, infinite = _cap(raw)
        values[f"z_{name}"] = value
        values[f"z_{name}_infinite"] = infinite
        capped.append(value)
    values["z_max"] = max(capped) if capped else 0.0

    continuity = evidence.continuity
    usable_continuity = bool(continuity and continuity.usable)
    values["continuity_usable"] = float(usable_continuity)
    values["continuity_offset_km"] = (
        _finite(continuity.offset_km) if usable_continuity else 0.0
    )
    values["continuity_median_km"] = (
        _finite(continuity.median_km) if usable_continuity else 0.0
    )
    values["continuity_scale_km"] = (
        _finite(continuity.scale_km) if usable_continuity else 0.0
    )
    values["continuity_z"] = (
        min(_finite(continuity.z_score), Z_CAP) if usable_continuity else 0.0
    )
    values["continuity_excess_km"] = (
        _finite(continuity.excess_km) if usable_continuity else 0.0
    )
    values["continuity_resolution_km"] = (
        _finite(continuity.resolution_km) if usable_continuity else 0.0
    )
    values["continuity_resolves_threshold"] = (
        float(continuity.resolves_threshold) if usable_continuity else 0.0
    )

    values["source_count"] = float(1 + len(evidence.divergences))
    separation = 0.0
    usable = 0.0
    for divergence in evidence.divergences:
        if divergence.usable:
            usable = 1.0
            separation = max(separation, divergence.max_separation_km)
    values["cross_source_max_separation_km"] = separation
    values["cross_source_usable"] = usable

    _predicted, score, _problems, _intact = classify(evidence)
    values["rule_silent_risk_score"] = min(score, Z_CAP) if math.isfinite(score) else Z_CAP
    return values


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def build(
    records: list[Record],
    evidences: list[screens_module.ScreenEvidence],
    configuration: str = "elements_and_history",
) -> np.ndarray:
    """Design matrix of shape ``(len(records), len(feature_names(configuration)))``.

    Raises
    ------
    ValueError
        If the two lists differ in length, or ``configuration`` is undeclared.
    """
    names = feature_names(configuration)
    if len(records) != len(evidences):
        raise ValueError(
            f"{len(records)} records and {len(evidences)} evidence objects supplied"
        )
    matrix = np.zeros((len(records), len(names)), dtype=float)
    for index, (record, evidence) in enumerate(zip(records, evidences, strict=True)):
        values = row(record, evidence)
        matrix[index] = [values[name] for name in names]
    if not np.isfinite(matrix).all():
        raise ValueError(
            "the design matrix contains a non-finite value; every feature is "
            "capped or defaulted and this indicates a defect in tlecheck.features"
        )
    return matrix
