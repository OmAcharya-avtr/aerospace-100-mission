"""Typed model of an STPA analysis.

Vocabulary follows the STPA Handbook (Leveson and Thomas, 2018), which names
the four steps as: define the purpose of the analysis (losses, hazards,
system-level constraints), model the control structure, identify unsafe control
actions, and identify loss scenarios. The underlying accident model is STAMP
(Leveson, "A new accident model for engineering safer systems", Safety Science
42(4), 2004, pp. 237-270; Leveson, *Engineering a Safer World*, MIT Press,
2011).

Everything here is a declaration made by a person. Nothing in this module
infers, scores or judges. The only quantities this package computes are counts
over the declared sets and reachability over the declared links; units are
therefore counts and dimensionless ratios, never physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class UcaClass(StrEnum):
    """The four classes of unsafe control action.

    Named as in the STPA Handbook (Leveson and Thomas, 2018), step 3: a control
    action is unsafe when it is *not provided*, when *providing* it causes a
    hazard, when it is provided with *wrong timing or order*, or when it is
    *stopped too soon or applied too long*. The fourth class applies only to
    continuous control actions; the Handbook says so, and this package records
    a declared exclusion for it rather than pretending it does not exist.
    """

    NOT_PROVIDED = "not_provided"
    PROVIDED = "provided"
    WRONG_TIMING_OR_ORDER = "wrong_timing_or_order"
    STOPPED_TOO_SOON_OR_APPLIED_TOO_LONG = "stopped_too_soon_or_applied_too_long"

    @property
    def label(self) -> str:
        """Human-readable class name for reports and diagrams."""
        return _UCA_LABELS[self]


#: Canonical ordering of the four UCA classes. Report and grid order is this
#: order everywhere in the package, so that two runs are byte-comparable.
UCA_CLASSES: tuple[UcaClass, ...] = (
    UcaClass.NOT_PROVIDED,
    UcaClass.PROVIDED,
    UcaClass.WRONG_TIMING_OR_ORDER,
    UcaClass.STOPPED_TOO_SOON_OR_APPLIED_TOO_LONG,
)

_UCA_LABELS: dict[UcaClass, str] = {
    UcaClass.NOT_PROVIDED: "not provided",
    UcaClass.PROVIDED: "provided",
    UcaClass.WRONG_TIMING_OR_ORDER: "wrong timing or order",
    UcaClass.STOPPED_TOO_SOON_OR_APPLIED_TOO_LONG: "stopped too soon / applied too long",
}


class ComponentKind(StrEnum):
    """Role of a control-structure component."""

    CONTROLLER = "controller"
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    PROCESS = "controlled_process"


class EvidenceKind(StrEnum):
    """Declared kind of an evidence artifact."""

    TEST = "test"
    ANALYSIS = "analysis"
    REVIEW = "review"
    DOCUMENT = "document"


@dataclass(frozen=True)
class Loss:
    """A loss the analysis exists to prevent (STPA step 1)."""

    id: str
    description: str
    stakeholders: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class Hazard:
    """A system state that, in a worst-case environment, leads to a loss."""

    id: str
    description: str
    losses: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class SystemConstraint:
    """A system-level constraint that prevents or limits a hazard."""

    id: str
    description: str
    hazards: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class Component:
    """A controller, actuator, sensor or controlled process."""

    id: str
    name: str
    kind: ComponentKind
    process_model: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class ControlAction:
    """A command issued by a controller, applied through an actuator."""

    id: str
    name: str
    controller: str
    actuator: str
    process: str
    continuous: bool = True
    line: int = 0


@dataclass(frozen=True)
class FeedbackPath:
    """A measurement path from the controlled process back to a controller."""

    id: str
    name: str
    sensor: str
    controller: str
    process: str
    line: int = 0


@dataclass(frozen=True)
class Uca:
    """An unsafe control action: one control action in one of the four classes."""

    id: str
    control_action: str
    uca_class: UcaClass
    description: str
    context: str = ""
    hazards: tuple[str, ...] = ()
    line: int = 0

    @property
    def cell(self) -> tuple[str, UcaClass]:
        """The enumeration cell this UCA occupies."""
        return (self.control_action, self.uca_class)


@dataclass(frozen=True)
class Exclusion:
    """An explicit, justified statement that one enumeration cell has no UCA.

    The STPA Handbook requires every control action to be considered in all
    four classes. A cell with no UCA is therefore either an omission or a
    decision; this record is the only way to declare the second, and a
    justification shorter than
    :data:`hazardtrace.checks.MIN_JUSTIFICATION_CHARS` is treated as an
    omission wearing a decision's clothes.
    """

    control_action: str
    uca_class: UcaClass
    justification: str
    author: str = ""
    date: str = ""
    line: int = 0

    @property
    def cell(self) -> tuple[str, UcaClass]:
        """The enumeration cell this exclusion covers."""
        return (self.control_action, self.uca_class)


@dataclass(frozen=True)
class LossScenario:
    """A causal scenario explaining how a UCA could occur (STPA step 4)."""

    id: str
    description: str
    ucas: tuple[str, ...] = ()
    causal_factor_class: str = ""
    mitigations: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class Mitigation:
    """A declared design or process change intended to remove a scenario."""

    id: str
    description: str
    requirements: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class Evidence:
    """An artifact offered as evidence that a mitigation is in place."""

    id: str
    kind: EvidenceKind
    description: str
    tests: tuple[str, ...] = ()
    path: str = ""
    line: int = 0


@dataclass(frozen=True)
class ControlStructure:
    """The modelled control structure (STPA step 2)."""

    components: tuple[Component, ...] = ()
    control_actions: tuple[ControlAction, ...] = ()
    feedback: tuple[FeedbackPath, ...] = ()

    def by_kind(self, kind: ComponentKind) -> tuple[Component, ...]:
        """Components of one kind, in declaration order."""
        return tuple(c for c in self.components if c.kind is kind)

    @property
    def component_ids(self) -> frozenset[str]:
        """Identifiers of every declared component."""
        return frozenset(c.id for c in self.components)


@dataclass(frozen=True)
class Analysis:
    """A whole declared STPA analysis, with identifier lookup tables.

    All collections preserve declaration order so that every report produced
    from the same document is byte-identical between runs.
    """

    title: str
    system: str = ""
    revision: str = ""
    losses: tuple[Loss, ...] = ()
    hazards: tuple[Hazard, ...] = ()
    constraints: tuple[SystemConstraint, ...] = ()
    structure: ControlStructure = field(default_factory=ControlStructure)
    ucas: tuple[Uca, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    loss_scenarios: tuple[LossScenario, ...] = ()
    mitigations: tuple[Mitigation, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    source: str = ""

    # -- lookup -----------------------------------------------------------
    @property
    def loss_ids(self) -> tuple[str, ...]:
        """Declared loss identifiers, in order."""
        return tuple(x.id for x in self.losses)

    @property
    def hazard_ids(self) -> tuple[str, ...]:
        """Declared hazard identifiers, in order."""
        return tuple(x.id for x in self.hazards)

    @property
    def constraint_ids(self) -> tuple[str, ...]:
        """Declared system-constraint identifiers, in order."""
        return tuple(x.id for x in self.constraints)

    @property
    def uca_ids(self) -> tuple[str, ...]:
        """Declared UCA identifiers, in order."""
        return tuple(x.id for x in self.ucas)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        """Declared loss-scenario identifiers, in order."""
        return tuple(x.id for x in self.loss_scenarios)

    @property
    def mitigation_ids(self) -> tuple[str, ...]:
        """Declared mitigation identifiers, in order."""
        return tuple(x.id for x in self.mitigations)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Declared evidence identifiers, in order."""
        return tuple(x.id for x in self.evidence)

    def uca(self, uca_id: str) -> Uca | None:
        """The UCA with this identifier, or None."""
        for item in self.ucas:
            if item.id == uca_id:
                return item
        return None

    def control_action(self, action_id: str) -> ControlAction | None:
        """The control action with this identifier, or None."""
        for item in self.structure.control_actions:
            if item.id == action_id:
                return item
        return None

    def component(self, component_id: str) -> Component | None:
        """The control-structure component with this identifier, or None."""
        for item in self.structure.components:
            if item.id == component_id:
                return item
        return None

    def cited_requirements(self) -> tuple[str, ...]:
        """Every requirement identifier cited anywhere, sorted and deduplicated."""
        seen: set[str] = set()
        for constraint in self.constraints:
            seen.update(constraint.requirements)
        for mitigation in self.mitigations:
            seen.update(mitigation.requirements)
        return tuple(sorted(seen))

    def cited_tests(self) -> tuple[str, ...]:
        """Every test identifier cited by any evidence artifact, sorted."""
        seen: set[str] = set()
        for item in self.evidence:
            seen.update(item.tests)
        return tuple(sorted(seen))
