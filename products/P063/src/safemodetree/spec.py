"""Declarative FMEA-shaped input: faults, modes, responses, conditions, observability.

The declaration is the product's input language. Every field is declared by the
analyst; nothing here is inferred from data. The vocabulary follows the FMEA /
FMECA worksheet columns of:

- MIL-STD-1629A, "Procedures for Performing a Failure Mode, Effects and
  Criticality Analysis", US Department of Defense, 24 November 1980. Task 101
  defines the severity classification Category I catastrophic, II critical,
  III marginal, IV minor. The standard was cancelled in 1998 and is cited here
  for the worksheet columns and the severity vocabulary only, not as an active
  requirement.
- ECSS-Q-ST-30-02C, "Space product assurance - Failure modes, effects (and
  criticality) analysis (FMEA/FMECA)", ESA Requirements and Standards
  Division, 6 March 2009.
- SAE ARP4761, "Guidelines and Methods for Conducting the Safety Assessment
  Process on Civil Airborne Systems and Equipment", SAE International, 1996.

Severity categories are ORDINAL. This module orders by them and never performs
arithmetic on them, because the categories of MIL-STD-1629A Task 101 are not a
measured scale. No Risk Priority Number is computed anywhere in this package.

Units used throughout:

- detection latency: seconds (s)
- response cost: minutes of mission outage (min), declared by the analyst
- observation channels: whatever unit the channel declares (A, V, s, deg,
  mN*m, dimensionless fraction)

Assumptions and validity range of the declaration itself:

- Discrete modes; exactly one active fault at a time. Simultaneous independent
  faults are out of scope.
- Preconditions are CONJUNCTIONS of literals over declared conditions. This is
  a deliberately restricted fragment: emptiness of a conjunction under declared
  exclusivity and per-mode availability is decidable in time linear in the
  number of literals, so the vacuous-response check is exact rather than a
  heuristic. General Boolean preconditions would need a SAT solver and this
  package does not ship one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "SEVERITY_CATEGORIES",
    "Condition",
    "Fault",
    "Literal",
    "Mode",
    "ObservabilityModel",
    "ObservationChannel",
    "Precondition",
    "Response",
    "SafeModeSpec",
    "load_spec",
    "load_spec_text",
    "packaged_spec_names",
    "packaged_spec_path",
]

#: Ordinal severity categories of MIL-STD-1629A Task 101, most severe first.
SEVERITY_CATEGORIES: tuple[str, ...] = ("I", "II", "III", "IV")

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SPEC_DIR = Path(__file__).resolve().parent / "specs"


def _check_id(value: object, what: str) -> str:
    """Validate a declared identifier and return it unchanged.

    Raises:
        TypeError: value is not a string.
        ValueError: value is not of the form ``[A-Za-z][A-Za-z0-9_]*``.
    """
    if not isinstance(value, str):
        raise TypeError(f"{what} id must be a string, got {type(value).__name__}")
    if not _ID_RE.match(value):
        raise ValueError(
            f"{what} id {value!r} is not a valid identifier: it must start with a letter "
            "and contain only letters, digits and underscores"
        )
    return value


def _check_unique(ids: Sequence[str], what: str) -> None:
    """Raise ValueError naming the duplicates if ``ids`` is not unique."""
    seen: set[str] = set()
    dupes: list[str] = []
    for value in ids:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    if dupes:
        raise ValueError(f"duplicate {what} id(s): {', '.join(sorted(dupes))}")


def _as_float(value: object, what: str) -> float:
    """Coerce a declared scalar to float, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{what} must be a real number, got {type(value).__name__}")
    return float(value)


@dataclass(frozen=True)
class Literal:
    """One literal of a precondition: a declared condition, asserted or negated.

    Attributes:
        condition: declared condition id.
        positive: True for ``condition``, False for ``not condition``.
    """

    condition: str
    positive: bool = True

    def __post_init__(self) -> None:
        _check_id(self.condition, "condition")
        if not isinstance(self.positive, bool):
            raise TypeError("Literal.positive must be a bool")

    def __str__(self) -> str:
        return self.condition if self.positive else f"!{self.condition}"

    @classmethod
    def parse(cls, text: str) -> Literal:
        """Parse ``cond``, ``!cond`` or ``not cond`` into a Literal."""
        if not isinstance(text, str):
            raise TypeError(f"literal must be a string, got {type(text).__name__}")
        stripped = text.strip()
        if stripped.startswith("!"):
            return cls(stripped[1:].strip(), positive=False)
        if stripped.lower().startswith("not "):
            return cls(stripped[4:].strip(), positive=False)
        return cls(stripped, positive=True)


@dataclass(frozen=True)
class Precondition:
    """A conjunction of literals over declared conditions.

    An empty conjunction is the constant True. Satisfiability is decided
    exactly by :meth:`SafeModeSpec.precondition_conflict`.
    """

    literals: tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.literals, tuple):
            raise TypeError("Precondition.literals must be a tuple of Literal")
        for lit in self.literals:
            if not isinstance(lit, Literal):
                raise TypeError("Precondition.literals must contain Literal instances")

    @property
    def is_trivial(self) -> bool:
        """True when the precondition is the empty conjunction (always holds)."""
        return not self.literals

    def __str__(self) -> str:
        return "true" if self.is_trivial else " and ".join(str(lit) for lit in self.literals)

    @classmethod
    def parse(cls, items: Iterable[str] | str | None) -> Precondition:
        """Build a Precondition from a literal string or an iterable of them."""
        if items is None:
            return cls(())
        if isinstance(items, str):
            items = [items]
        return cls(tuple(Literal.parse(text) for text in items))


@dataclass(frozen=True)
class Condition:
    """A declared Boolean condition a response may require.

    Attributes:
        id: identifier.
        description: free text.
        available_in_modes: modes in which the condition CAN be true. Empty
            tuple means every mode. A positive literal over a condition that is
            not available in the source mode can never hold there.
        always_false: the analyst declares the condition unachievable in this
            configuration (for example a hardware string that is not populated).
    """

    id: str
    description: str = ""
    available_in_modes: tuple[str, ...] = ()
    always_false: bool = False

    def __post_init__(self) -> None:
        _check_id(self.id, "condition")
        if not isinstance(self.description, str):
            raise TypeError("Condition.description must be a string")
        if not isinstance(self.available_in_modes, tuple):
            raise TypeError("Condition.available_in_modes must be a tuple of mode ids")
        for mode_id in self.available_in_modes:
            _check_id(mode_id, "mode")
        if not isinstance(self.always_false, bool):
            raise TypeError("Condition.always_false must be a bool")

    def can_hold_in(self, mode_id: str) -> bool:
        """True if this condition can be true while in ``mode_id``."""
        if self.always_false:
            return False
        if not self.available_in_modes:
            return True
        return mode_id in self.available_in_modes


@dataclass(frozen=True)
class Mode:
    """A declared system mode.

    Attributes:
        id: identifier.
        description: free text.
        safe: the analyst declares this mode a safe mode. Safe-mode
            reachability is checked against the set of modes with ``safe=True``.
        absorbing: no response may leave this mode (a terminal mode such as a
            hardware-commanded survival state awaiting ground action).
    """

    id: str
    description: str = ""
    safe: bool = False
    absorbing: bool = False

    def __post_init__(self) -> None:
        _check_id(self.id, "mode")
        if not isinstance(self.description, str):
            raise TypeError("Mode.description must be a string")
        if not isinstance(self.safe, bool):
            raise TypeError("Mode.safe must be a bool")
        if not isinstance(self.absorbing, bool):
            raise TypeError("Mode.absorbing must be a bool")


@dataclass(frozen=True)
class Fault:
    """One FMEA row: a failure mode with its effect, detectability and latency.

    Attributes:
        id: identifier.
        effect: the local or end effect, in the FMEA sense (MIL-STD-1629A
            Task 101 worksheet column "failure effects").
        severity: ordinal severity category, one of ``SEVERITY_CATEGORIES``.
        detectable: the analyst declares the fault detectable by the onboard
            monitor set. An undetectable fault admits only fault-agnostic
            responses.
        detection_latency_s: declared worst-case time from fault occurrence to
            detection, in seconds. Compared against the detection assumption's
            latency budget.
        occurs_in_modes: modes in which this fault can occur. Empty tuple means
            every non-absorbing mode.
    """

    id: str
    effect: str = ""
    severity: str = "III"
    detectable: bool = True
    detection_latency_s: float = 0.0
    occurs_in_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_id(self.id, "fault")
        if not isinstance(self.effect, str):
            raise TypeError("Fault.effect must be a string")
        if self.severity not in SEVERITY_CATEGORIES:
            raise ValueError(
                f"Fault {self.id!r}: severity {self.severity!r} is not one of "
                f"{', '.join(SEVERITY_CATEGORIES)} (MIL-STD-1629A Task 101)"
            )
        if not isinstance(self.detectable, bool):
            raise TypeError("Fault.detectable must be a bool")
        latency = _as_float(self.detection_latency_s, f"Fault {self.id!r}.detection_latency_s")
        if latency < 0.0:
            raise ValueError(
                f"Fault {self.id!r}: detection_latency_s must be >= 0 s, got {latency} s"
            )
        object.__setattr__(self, "detection_latency_s", latency)
        if not isinstance(self.occurs_in_modes, tuple):
            raise TypeError("Fault.occurs_in_modes must be a tuple of mode ids")
        for mode_id in self.occurs_in_modes:
            _check_id(mode_id, "mode")

    @property
    def severity_rank(self) -> int:
        """Ordinal rank, 0 for Category I (most severe). Ordering only."""
        return SEVERITY_CATEGORIES.index(self.severity)


@dataclass(frozen=True)
class Response:
    """A declared response: a mode transition with a cost and a precondition.

    Attributes:
        id: identifier.
        fault: the fault this response addresses, or None for a fault-agnostic
            response that may fire without a fault identification.
        from_mode: source mode id, or None for "any non-absorbing mode".
        to_mode: target mode id.
        cost_min: declared cost in minutes of mission outage (min).
        precondition: conjunction of literals that must hold to fire.
        clears_fault: the response is declared to clear the fault (for example
            a power cycle of the failed unit), so the automaton leaves the
            faulted state. Default False: the response only changes mode.
    """

    id: str
    to_mode: str
    fault: str | None = None
    from_mode: str | None = None
    cost_min: float = 0.0
    precondition: Precondition = field(default_factory=Precondition)
    clears_fault: bool = False

    def __post_init__(self) -> None:
        _check_id(self.id, "response")
        _check_id(self.to_mode, "mode")
        if self.fault is not None:
            _check_id(self.fault, "fault")
        if self.from_mode is not None:
            _check_id(self.from_mode, "mode")
        cost = _as_float(self.cost_min, f"Response {self.id!r}.cost_min")
        if cost < 0.0:
            raise ValueError(f"Response {self.id!r}: cost_min must be >= 0 min, got {cost} min")
        object.__setattr__(self, "cost_min", cost)
        if not isinstance(self.precondition, Precondition):
            raise TypeError("Response.precondition must be a Precondition")
        if not isinstance(self.clears_fault, bool):
            raise TypeError("Response.clears_fault must be a bool")

    @property
    def is_fault_agnostic(self) -> bool:
        """True when the response does not name a fault."""
        return self.fault is None


@dataclass(frozen=True)
class ObservationChannel:
    """One continuous onboard measurement with its declared decision threshold.

    Attributes:
        id: identifier.
        unit: physical unit of the raw measurement, as a string.
        nominal: declared nominal value, in ``unit``.
        threshold: declared decision threshold, in ``unit``.
        direction: ``"above"`` if the symptom asserts when the measurement is
            strictly above the threshold, ``"below"`` if strictly below.
        noise_sd: declared 1-sigma measurement noise, in ``unit``. Used by the
            synthetic generator and by nothing in the deterministic core.
        faulted: declared value of the measurement when the symptom is
            asserted, in ``unit``.
    """

    id: str
    unit: str
    nominal: float
    threshold: float
    direction: str
    noise_sd: float
    faulted: float

    def __post_init__(self) -> None:
        _check_id(self.id, "channel")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError(f"channel {self.id!r}: unit must be a non-empty string")
        if self.direction not in ("above", "below"):
            raise ValueError(
                f"channel {self.id!r}: direction must be 'above' or 'below', "
                f"got {self.direction!r}"
            )
        for name in ("nominal", "threshold", "noise_sd", "faulted"):
            object.__setattr__(self, name, _as_float(getattr(self, name), f"{self.id}.{name}"))
        if self.noise_sd <= 0.0:
            raise ValueError(
                f"channel {self.id!r}: noise_sd must be > 0 {self.unit}, got {self.noise_sd}"
            )
        if self.direction == "above" and not self.nominal < self.threshold < self.faulted:
            raise ValueError(
                f"channel {self.id!r}: direction 'above' requires "
                f"nominal < threshold < faulted, got "
                f"{self.nominal} < {self.threshold} < {self.faulted}"
            )
        if self.direction == "below" and not self.faulted < self.threshold < self.nominal:
            raise ValueError(
                f"channel {self.id!r}: direction 'below' requires "
                f"faulted < threshold < nominal, got "
                f"{self.faulted} < {self.threshold} < {self.nominal}"
            )

    def asserted(self, value: float) -> bool:
        """True if ``value`` (in ``unit``) asserts the symptom at the threshold.

        The comparison is strict in the declared direction. A value exactly on
        the threshold does NOT assert. This is a strict threshold comparison on
        a floating-point quantity, so the returned boolean is not a stable
        function of an algebraically equivalent rescaling of the channel; see
        README Limitations.
        """
        return value > self.threshold if self.direction == "above" else value < self.threshold

    def margin(self, value: float) -> float:
        """Signed distance from the threshold in units of ``noise_sd``.

        Positive when the symptom is asserted. Dimensionless.
        """
        delta = value - self.threshold
        if self.direction == "below":
            delta = -delta
        return delta / self.noise_sd


@dataclass(frozen=True)
class ObservabilityModel:
    """Which faults are distinguishable from which observations.

    The model is a declared binary signature per fault over the declared
    observation channels. Two faults are *declared indistinguishable* when
    their signatures are equal. This is the observability relation of the
    diagnosability literature restricted to a single static observation vector:

    Sampath, Sengupta, Lafortune, Sinnamohideen and Teneketzis, "Diagnosability
    of Discrete-Event Systems", IEEE Transactions on Automatic Control 40(9),
    pp. 1555-1575, September 1995. That paper decides diagnosability over
    event traces of a discrete-event system; the static-signature relation used
    here is the special case in which the observation is one vector taken after
    the declared detection latency, and it is strictly weaker. See README
    Limitations.

    Attributes:
        channels: declared channels, in a fixed order. The order is the
            observation-vector order everywhere in this package.
        signatures: fault id -> tuple of 0/1 over ``channels``.
    """

    channels: tuple[ObservationChannel, ...]
    signatures: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("ObservabilityModel needs at least one channel")
        _check_unique([c.id for c in self.channels], "channel")
        width = len(self.channels)
        for fault_id, signature in self.signatures.items():
            _check_id(fault_id, "fault")
            if len(signature) != width:
                raise ValueError(
                    f"signature for fault {fault_id!r} has {len(signature)} entries, "
                    f"expected {width} (one per declared channel)"
                )
            for bit in signature:
                if bit not in (0, 1):
                    raise ValueError(
                        f"signature for fault {fault_id!r} must contain only 0 or 1, got {bit!r}"
                    )

    @property
    def channel_ids(self) -> tuple[str, ...]:
        """Channel ids in observation-vector order."""
        return tuple(c.id for c in self.channels)

    def signature(self, fault_id: str) -> tuple[int, ...]:
        """Declared signature of ``fault_id``. Raises KeyError if undeclared."""
        return tuple(self.signatures[fault_id])

    def indistinguishable_groups(self) -> tuple[tuple[str, ...], ...]:
        """Faults grouped by equal declared signature.

        Returns groups of size >= 2 only, each sorted by fault id, the groups
        themselves sorted by their first member. A group of size >= 2 is an
        ambiguity class in the sense this product reports.
        """
        buckets: dict[tuple[int, ...], list[str]] = {}
        for fault_id, signature in self.signatures.items():
            buckets.setdefault(tuple(signature), []).append(fault_id)
        groups = [tuple(sorted(v)) for v in buckets.values() if len(v) > 1]
        return tuple(sorted(groups))

    def nominal_vector(self) -> tuple[float, ...]:
        """Declared nominal measurement vector, one entry per channel."""
        return tuple(c.nominal for c in self.channels)

    def expected_vector(self, signature: Sequence[int]) -> tuple[float, ...]:
        """Declared noiseless measurement vector for a binary signature."""
        if len(signature) != len(self.channels):
            raise ValueError(
                f"signature has {len(signature)} entries, expected {len(self.channels)}"
            )
        return tuple(
            channel.faulted if bit else channel.nominal
            for channel, bit in zip(self.channels, signature, strict=True)
        )


@dataclass(frozen=True)
class SafeModeSpec:
    """A complete fault-response declaration.

    Attributes:
        name: free-text name of the declaration.
        description: free text.
        modes: declared modes.
        faults: declared faults (the FMEA rows).
        responses: declared responses.
        conditions: declared conditions referenced by response preconditions.
        observability: the declared observability model.
        initial_mode: the mode the system is in before any fault.
    """

    name: str
    modes: tuple[Mode, ...]
    faults: tuple[Fault, ...]
    responses: tuple[Response, ...]
    observability: ObservabilityModel
    initial_mode: str
    conditions: tuple[Condition, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SafeModeSpec.name must be a non-empty string")
        if not self.modes:
            raise ValueError("SafeModeSpec needs at least one mode")
        if not self.faults:
            raise ValueError("SafeModeSpec needs at least one fault")
        _check_unique([m.id for m in self.modes], "mode")
        _check_unique([f.id for f in self.faults], "fault")
        _check_unique([r.id for r in self.responses], "response")
        _check_unique([c.id for c in self.conditions], "condition")

        mode_ids = {m.id for m in self.modes}
        fault_ids = {f.id for f in self.faults}
        condition_ids = {c.id for c in self.conditions}

        if self.initial_mode not in mode_ids:
            raise ValueError(
                f"initial_mode {self.initial_mode!r} is not a declared mode "
                f"(declared: {', '.join(sorted(mode_ids))})"
            )
        if not any(m.safe for m in self.modes):
            raise ValueError(
                "SafeModeSpec declares no safe mode; safe-mode reachability is undefined. "
                "Set safe: true on at least one mode."
            )
        for fault in self.faults:
            for mode_id in fault.occurs_in_modes:
                if mode_id not in mode_ids:
                    raise ValueError(
                        f"fault {fault.id!r} declares occurs_in_modes {mode_id!r}, "
                        "which is not a declared mode"
                    )
        for response in self.responses:
            if response.to_mode not in mode_ids:
                raise ValueError(
                    f"response {response.id!r} targets mode {response.to_mode!r}, "
                    "which is not declared"
                )
            if response.from_mode is not None and response.from_mode not in mode_ids:
                raise ValueError(
                    f"response {response.id!r} declares from_mode {response.from_mode!r}, "
                    "which is not declared"
                )
            if response.fault is not None and response.fault not in fault_ids:
                raise ValueError(
                    f"response {response.id!r} addresses fault {response.fault!r}, "
                    "which is not declared"
                )
            for lit in response.precondition.literals:
                if lit.condition not in condition_ids:
                    raise ValueError(
                        f"response {response.id!r} references condition {lit.condition!r}, "
                        "which is not declared"
                    )
        for condition in self.conditions:
            for mode_id in condition.available_in_modes:
                if mode_id not in mode_ids:
                    raise ValueError(
                        f"condition {condition.id!r} lists mode {mode_id!r}, which is not declared"
                    )
        declared_signatures = set(self.observability.signatures)
        if declared_signatures != fault_ids:
            missing = sorted(fault_ids - declared_signatures)
            extra = sorted(declared_signatures - fault_ids)
            parts = []
            if missing:
                parts.append(f"faults with no signature: {', '.join(missing)}")
            if extra:
                parts.append(f"signatures for undeclared faults: {', '.join(extra)}")
            raise ValueError(
                "the observability model must declare exactly one signature per fault; "
                + "; ".join(parts)
            )
        zero = (0,) * len(self.observability.channels)
        for fault_id, signature in self.observability.signatures.items():
            if tuple(signature) == zero:
                raise ValueError(
                    f"fault {fault_id!r} has an all-zero signature, which is identical to "
                    "the nominal observation; declare at least one asserted channel or "
                    "declare the fault undetectable and remove it from the observability model"
                )

    # -- lookups ---------------------------------------------------------

    @property
    def mode_ids(self) -> tuple[str, ...]:
        """Declared mode ids in declaration order."""
        return tuple(m.id for m in self.modes)

    @property
    def fault_ids(self) -> tuple[str, ...]:
        """Declared fault ids in declaration order."""
        return tuple(f.id for f in self.faults)

    @property
    def safe_mode_ids(self) -> tuple[str, ...]:
        """Ids of modes declared safe, in declaration order."""
        return tuple(m.id for m in self.modes if m.safe)

    def mode(self, mode_id: str) -> Mode:
        """Look up a mode by id. Raises KeyError if undeclared."""
        for candidate in self.modes:
            if candidate.id == mode_id:
                return candidate
        raise KeyError(f"no declared mode {mode_id!r}")

    def fault(self, fault_id: str) -> Fault:
        """Look up a fault by id. Raises KeyError if undeclared."""
        for candidate in self.faults:
            if candidate.id == fault_id:
                return candidate
        raise KeyError(f"no declared fault {fault_id!r}")

    def response(self, response_id: str) -> Response:
        """Look up a response by id. Raises KeyError if undeclared."""
        for candidate in self.responses:
            if candidate.id == response_id:
                return candidate
        raise KeyError(f"no declared response {response_id!r}")

    def condition(self, condition_id: str) -> Condition:
        """Look up a condition by id. Raises KeyError if undeclared."""
        for candidate in self.conditions:
            if candidate.id == condition_id:
                return candidate
        raise KeyError(f"no declared condition {condition_id!r}")

    def fault_occurs_in(self, fault_id: str, mode_id: str) -> bool:
        """True if the declaration allows ``fault_id`` to occur in ``mode_id``."""
        fault = self.fault(fault_id)
        if self.mode(mode_id).absorbing:
            return False
        if not fault.occurs_in_modes:
            return True
        return mode_id in fault.occurs_in_modes

    # -- precondition emptiness (exact over the declared fragment) --------

    def precondition_conflict(self, precondition: Precondition, mode_id: str) -> str | None:
        """Explain why a conjunction can never hold in ``mode_id``, or None.

        Exact over the declared fragment (conjunctions of literals plus
        per-condition mode availability and an ``always_false`` flag). The
        three ways a conjunction is empty:

        1. it contains both ``c`` and ``!c``;
        2. a positive literal names a condition declared ``always_false``;
        3. a positive literal names a condition not available in ``mode_id``.

        Returns the first conflict found as a one-line string, or None when the
        conjunction is satisfiable. Runs in time linear in the literal count.
        """
        self.mode(mode_id)
        positives: set[str] = set()
        negatives: set[str] = set()
        for lit in precondition.literals:
            (positives if lit.positive else negatives).add(lit.condition)
        both = sorted(positives & negatives)
        if both:
            return f"requires both {both[0]} and !{both[0]}"
        for condition_id in sorted(positives):
            condition = self.condition(condition_id)
            if condition.always_false:
                return f"requires {condition_id}, declared always_false"
            if not condition.can_hold_in(mode_id):
                available = ", ".join(condition.available_in_modes) or "(any)"
                return (
                    f"requires {condition_id}, which is only available in {available}, "
                    f"not in {mode_id}"
                )
        return None

    def precondition_can_hold(self, precondition: Precondition, mode_id: str) -> bool:
        """True if the conjunction is satisfiable in ``mode_id``."""
        return self.precondition_conflict(precondition, mode_id) is None

    def summary(self) -> str:
        """One-line description of the declaration's size."""
        groups = self.observability.indistinguishable_groups()
        return (
            f"{self.name}: {len(self.modes)} modes ({len(self.safe_mode_ids)} safe), "
            f"{len(self.faults)} faults, {len(self.responses)} responses, "
            f"{len(self.observability.channels)} channels, "
            f"{len(groups)} declared ambiguity class(es)"
        )


# -- loading ------------------------------------------------------------------


def _require_mapping(node: object, what: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError(f"{what} must be a mapping, got {type(node).__name__}")
    return node


def _tuple_of_str(node: object, what: str) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, str):
        raise ValueError(f"{what} must be a list of ids, not a bare string {node!r}")
    if not isinstance(node, Sequence):
        raise ValueError(f"{what} must be a list of ids, got {type(node).__name__}")
    return tuple(str(item) for item in node)


def _unknown_keys(node: Mapping[str, Any], allowed: set[str], what: str) -> None:
    extra = sorted(set(node) - allowed)
    if extra:
        raise ValueError(
            f"{what}: unknown key(s) {', '.join(extra)}; allowed: {', '.join(sorted(allowed))}"
        )


def load_spec_text(text: str) -> SafeModeSpec:
    """Parse a declaration from YAML text.

    Unknown keys are rejected rather than ignored, so a typo in a declaration
    is an error and not a silently dropped field.
    """
    document = yaml.safe_load(text)
    root = _require_mapping(document, "the declaration")
    _unknown_keys(
        root,
        {
            "name",
            "description",
            "initial_mode",
            "modes",
            "faults",
            "responses",
            "conditions",
            "channels",
            "signatures",
        },
        "the declaration",
    )
    for key in ("name", "initial_mode", "modes", "faults", "channels", "signatures"):
        if key not in root:
            raise ValueError(f"the declaration is missing required key {key!r}")

    modes: list[Mode] = []
    for node in root["modes"]:
        item = _require_mapping(node, "a mode")
        _unknown_keys(item, {"id", "description", "safe", "absorbing"}, f"mode {item.get('id')!r}")
        modes.append(
            Mode(
                id=item["id"],
                description=str(item.get("description", "")),
                safe=bool(item.get("safe", False)),
                absorbing=bool(item.get("absorbing", False)),
            )
        )

    conditions: list[Condition] = []
    for node in root.get("conditions") or []:
        item = _require_mapping(node, "a condition")
        _unknown_keys(
            item,
            {"id", "description", "available_in_modes", "always_false"},
            f"condition {item.get('id')!r}",
        )
        conditions.append(
            Condition(
                id=item["id"],
                description=str(item.get("description", "")),
                available_in_modes=_tuple_of_str(
                    item.get("available_in_modes"), f"condition {item['id']}.available_in_modes"
                ),
                always_false=bool(item.get("always_false", False)),
            )
        )

    faults: list[Fault] = []
    for node in root["faults"]:
        item = _require_mapping(node, "a fault")
        _unknown_keys(
            item,
            {"id", "effect", "severity", "detectable", "detection_latency_s", "occurs_in_modes"},
            f"fault {item.get('id')!r}",
        )
        faults.append(
            Fault(
                id=item["id"],
                effect=str(item.get("effect", "")),
                severity=str(item.get("severity", "III")),
                detectable=bool(item.get("detectable", True)),
                detection_latency_s=item.get("detection_latency_s", 0.0),
                occurs_in_modes=_tuple_of_str(
                    item.get("occurs_in_modes"), f"fault {item['id']}.occurs_in_modes"
                ),
            )
        )

    responses: list[Response] = []
    for node in root.get("responses") or []:
        item = _require_mapping(node, "a response")
        _unknown_keys(
            item,
            {"id", "fault", "from_mode", "to_mode", "cost_min", "precondition", "clears_fault"},
            f"response {item.get('id')!r}",
        )
        responses.append(
            Response(
                id=item["id"],
                to_mode=item["to_mode"],
                fault=item.get("fault"),
                from_mode=item.get("from_mode"),
                cost_min=item.get("cost_min", 0.0),
                precondition=Precondition.parse(item.get("precondition")),
                clears_fault=bool(item.get("clears_fault", False)),
            )
        )

    channels: list[ObservationChannel] = []
    for node in root["channels"]:
        item = _require_mapping(node, "a channel")
        _unknown_keys(
            item,
            {"id", "unit", "nominal", "threshold", "direction", "noise_sd", "faulted"},
            f"channel {item.get('id')!r}",
        )
        channels.append(
            ObservationChannel(
                id=item["id"],
                unit=str(item["unit"]),
                nominal=item["nominal"],
                threshold=item["threshold"],
                direction=str(item["direction"]),
                noise_sd=item["noise_sd"],
                faulted=item["faulted"],
            )
        )

    channel_ids = [c.id for c in channels]
    signatures: dict[str, tuple[int, ...]] = {}
    for fault_id, node in _require_mapping(root["signatures"], "signatures").items():
        item = _require_mapping(node, f"signature for {fault_id!r}")
        extra = sorted(set(item) - set(channel_ids))
        if extra:
            raise ValueError(
                f"signature for fault {fault_id!r} names undeclared channel(s): "
                f"{', '.join(extra)}"
            )
        signatures[str(fault_id)] = tuple(int(bool(item.get(cid, 0))) for cid in channel_ids)

    return SafeModeSpec(
        name=str(root["name"]),
        description=str(root.get("description", "")),
        modes=tuple(modes),
        faults=tuple(faults),
        responses=tuple(responses),
        conditions=tuple(conditions),
        observability=ObservabilityModel(channels=tuple(channels), signatures=signatures),
        initial_mode=str(root["initial_mode"]),
    )


def load_spec(path: str | Path) -> SafeModeSpec:
    """Load a declaration from a YAML file path."""
    return load_spec_text(Path(path).read_text(encoding="utf-8"))


def packaged_spec_names() -> tuple[str, ...]:
    """Names of the declarations shipped inside the package, sorted."""
    return tuple(sorted(p.stem for p in _SPEC_DIR.glob("*.yaml")))


def packaged_spec_path(name: str) -> Path:
    """Path of a shipped declaration by stem. Raises KeyError if absent."""
    candidate = _SPEC_DIR / f"{name}.yaml"
    if not candidate.is_file():
        raise KeyError(
            f"no packaged declaration {name!r}; available: {', '.join(packaged_spec_names())}"
        )
    return candidate
