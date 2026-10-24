"""Declarations: command dictionary, mode machine, initial state, candidate sequence.

Everything in this module is a declaration supplied by the user. Nothing here is
derived from flight data and nothing here is a physical model. The package
checks a sequence against *these declarations* and against nothing else, which
is the whole of what it can honestly claim: a sequence that passes has been
shown consistent with the dictionary and the mode machine written down in the
input file, and a declaration that is wrong produces a check that is wrong.

Units are carried in the field names. Times are seconds after a sequence epoch
the caller defines; there is no time scale, no leap second and no UTC anywhere
in this package. Resource quantities are whatever unit the caller declares in
``ResourceLimits``, compared like-for-like against the same key in a command's
``resource_draw``; the shipped examples use watts and megabits per second
because those are the two a command sequence usually trips over.

Semantics of the three-valued predicate
---------------------------------------
A predicate is ``TRUE``, ``FALSE`` or ``UNKNOWN``. ``UNKNOWN`` is not a
convenience for missing data: it is the state of a condition that has not been
verified, and it is the state that makes the irreversible-command gate
meaningful. A deployment command whose precondition is ``FALSE`` is a defect
anybody's linter finds. A deployment command whose precondition is ``UNKNOWN``
is the one that reaches the vehicle, because every per-command check it passed
was evaluated against an assumption rather than against a verification. Treating
``UNKNOWN`` as satisfiable is the single modelling decision in this package most
likely to make it useless, so it is not satisfiable here.

References for the modelling, such as it is
-------------------------------------------
The mode machine is a plain deterministic finite automaton over declared mode
names with a declared transition relation; see Hopcroft, Motwani & Ullman 2006,
*Introduction to Automata Theory, Languages, and Computation*, 3rd ed., ch. 2,
for the definition used. The forward walk in :mod:`cmdguard.trace` is symbolic
execution in the sense of King 1976, "Symbolic execution and program testing",
*Communications of the ACM* 19(7), restricted to a single straight-line path,
which is why it is exact rather than approximate: a linear command sequence has
no branches, so there is one path and it is enumerated. No numerical value in
this repository is taken from either reference.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "Verity",
    "CommandSpec",
    "ResourceLimits",
    "CommandDictionary",
    "ModeMachine",
    "InitialState",
    "SequenceStep",
    "CommandSequence",
    "Case",
    "EffectPolicy",
    "parse_verity",
]


class Verity(enum.Enum):
    """Three-valued predicate state: verified true, verified false, unverified.

    ``UNKNOWN`` never satisfies a precondition. See the module docstring for why.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class EffectPolicy(enum.Enum):
    """What the forward executor does with the effects of a violating command.

    ``APPLY`` (the default) applies the declared effects even when the command
    raised a violation, so that an independent defect later in the sequence is
    still found instead of being buried under a cascade from the first one.
    ``BLOCK`` models an onboard command rejection: a command that violated
    anything changes no state at all.

    Neither is the correct model of a real flight computer, because that depends
    on which checks the flight software performs onboard and which it assumes
    the ground performed. Both are offered and the choice is recorded in the
    report, rather than one being chosen silently.
    """

    APPLY = "apply"
    BLOCK = "block"


def parse_verity(value: object) -> Verity:
    """Coerce a YAML scalar to a :class:`Verity`.

    Accepts the booleans ``True``/``False``, the strings ``"true"``,
    ``"false"``, ``"unknown"`` in any case, and ``None`` for ``UNKNOWN``.
    Anything else is a declaration error and raises :class:`ValueError`.
    """
    if isinstance(value, Verity):
        return value
    if value is None:
        return Verity.UNKNOWN
    if isinstance(value, bool):
        return Verity.TRUE if value else Verity.FALSE
    if isinstance(value, str):
        lowered = value.strip().lower()
        for member in Verity:
            if member.value == lowered:
                return member
    raise ValueError(
        f"predicate value {value!r} is not one of true, false, unknown "
        "(a predicate that has not been verified is 'unknown', not absent)"
    )


@dataclass(frozen=True)
class CommandSpec:
    """One declared command.

    Parameters
    ----------
    identifier:
        Command mnemonic. Unique within a dictionary.
    subsystem:
        Free-text owner, used only for reporting and for one ranker feature.
    required_modes:
        Modes in which the command is permitted. Empty means no mode gating.
    preconditions:
        Predicate name to the :class:`Verity` the command requires. A
        requirement of ``UNKNOWN`` is a declaration error: requiring that
        something be unverified is not a precondition.
    effects:
        Predicate name to the :class:`Verity` the command leaves it in. An
        effect of ``UNKNOWN`` is legal and useful: it is how a command that
        invalidates a verification is declared.
    mode_effect:
        Mode the command transitions to, or ``None`` for no transition.
    resource_draw:
        Resource key to quantity drawn, held for ``hold_s`` seconds from the
        command's execution time.
    hold_s:
        Duration in seconds for which the draw and any mutual exclusion are
        held. Zero means the draw exists only at the execution instant.
    min_interval_s:
        Minimum seconds between two executions of *this* command.
    min_gap_after_s:
        Minimum seconds after this command before *any* next command.
    irreversible:
        True for a command whose effect cannot be undone from the ground
        (separation, deployment, pyro, one-shot valve, memory scrub).
    exclusion_group:
        Name of a mutual-exclusion group, or ``None``.
    """

    identifier: str
    subsystem: str = "unassigned"
    required_modes: tuple[str, ...] = ()
    preconditions: Mapping[str, Verity] = field(default_factory=dict)
    effects: Mapping[str, Verity] = field(default_factory=dict)
    mode_effect: str | None = None
    resource_draw: Mapping[str, float] = field(default_factory=dict)
    hold_s: float = 0.0
    min_interval_s: float = 0.0
    min_gap_after_s: float = 0.0
    irreversible: bool = False
    exclusion_group: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier or not self.identifier.strip():
            raise ValueError("command identifier must be a non-empty string")
        for name, required in self.preconditions.items():
            if required is Verity.UNKNOWN:
                raise ValueError(
                    f"command {self.identifier}: precondition {name!r} requires "
                    "'unknown', which is not a condition that can be met; a "
                    "precondition requires true or false"
                )
        for key, amount in self.resource_draw.items():
            if not isinstance(amount, (int, float)) or isinstance(amount, bool):
                raise ValueError(
                    f"command {self.identifier}: resource draw {key!r} must be a number"
                )
            if amount < 0.0:
                raise ValueError(
                    f"command {self.identifier}: resource draw {key!r} is {amount}, "
                    "which is negative; a command that returns a resource is modelled "
                    "by the end of its hold, not by a negative draw"
                )
        for label, value in (
            ("hold_s", self.hold_s),
            ("min_interval_s", self.min_interval_s),
            ("min_gap_after_s", self.min_gap_after_s),
        ):
            if value < 0.0:
                raise ValueError(f"command {self.identifier}: {label} is {value}, must be >= 0")

    @property
    def draw_keys(self) -> tuple[str, ...]:
        """Resource keys this command draws on, sorted for determinism."""
        return tuple(sorted(self.resource_draw))


@dataclass(frozen=True)
class ResourceLimits:
    """Declared ceilings per resource key. A key with no limit is unchecked."""

    limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.limits.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"resource limit {key!r} must be a number")
            if value <= 0.0:
                raise ValueError(f"resource limit {key!r} is {value}, must be > 0")

    def limit_for(self, key: str) -> float | None:
        """Ceiling for ``key``, or ``None`` when the key is undeclared."""
        value = self.limits.get(key)
        return None if value is None else float(value)


@dataclass(frozen=True)
class CommandDictionary:
    """The declared command set, its resource ceilings and its exclusion groups."""

    commands: Mapping[str, CommandSpec]
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        if not self.commands:
            raise ValueError("command dictionary is empty")
        for identifier, spec in self.commands.items():
            if identifier != spec.identifier:
                raise ValueError(
                    f"dictionary key {identifier!r} does not match command identifier "
                    f"{spec.identifier!r}"
                )

    def __contains__(self, identifier: object) -> bool:
        return identifier in self.commands

    def get(self, identifier: str) -> CommandSpec | None:
        """The spec for ``identifier``, or ``None`` when undeclared."""
        return self.commands.get(identifier)

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Declared command mnemonics, sorted."""
        return tuple(sorted(self.commands))

    def exclusion_members(self, group: str) -> tuple[str, ...]:
        """Commands in mutual-exclusion group ``group``, sorted."""
        return tuple(
            sorted(c.identifier for c in self.commands.values() if c.exclusion_group == group)
        )


@dataclass(frozen=True)
class ModeMachine:
    """Deterministic finite automaton over declared mode names.

    ``transitions[m]`` is the set of modes reachable from ``m`` in one command.
    A mode is always reachable from itself (a command with no mode effect does
    not transition), and that self-loop is implicit rather than declared.
    """

    modes: tuple[str, ...]
    transitions: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("mode machine declares no modes")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(f"mode machine declares a duplicate mode: {self.modes}")
        known = set(self.modes)
        for source, targets in self.transitions.items():
            if source not in known:
                raise ValueError(f"transition declared from undeclared mode {source!r}")
            for target in targets:
                if target not in known:
                    raise ValueError(
                        f"transition {source!r} -> {target!r} names undeclared mode {target!r}"
                    )

    def allows(self, source: str, target: str) -> bool:
        """True when ``source -> target`` is declared, or is the self-loop."""
        if source == target:
            return True
        return target in tuple(self.transitions.get(source, ()))

    def reachable_from(self, source: str) -> tuple[str, ...]:
        """Modes reachable from ``source`` in one transition, sorted."""
        return tuple(sorted(set(self.transitions.get(source, ()))))


@dataclass(frozen=True)
class InitialState:
    """Declared state the sequence starts from.

    A predicate absent from ``predicates`` is ``UNKNOWN``, not ``FALSE``. This
    is deliberate and it is the opposite of what a convenience default would
    do: an undeclared condition has not been verified.
    """

    mode: str
    predicates: Mapping[str, Verity] = field(default_factory=dict)

    def verity(self, name: str) -> Verity:
        """Declared state of predicate ``name``; ``UNKNOWN`` when undeclared."""
        return self.predicates.get(name, Verity.UNKNOWN)


@dataclass(frozen=True)
class SequenceStep:
    """One command occurrence at one time.

    ``t_s`` is seconds after the sequence epoch. Steps are non-decreasing in
    ``t_s``; a sequence that goes backwards in time is a declaration error and
    is rejected at construction rather than reported as a violation, because it
    is not something a spacecraft can be asked to do.
    """

    t_s: float
    command: str
    note: str = ""


@dataclass(frozen=True)
class CommandSequence:
    """A candidate sequence of :class:`SequenceStep`, time-ordered."""

    steps: tuple[SequenceStep, ...]

    def __post_init__(self) -> None:
        previous: float | None = None
        for index, step in enumerate(self.steps):
            if previous is not None and step.t_s < previous:
                raise ValueError(
                    f"sequence step {index} ({step.command}) is at t={step.t_s} s, "
                    f"before step {index - 1} at t={previous} s; steps must be "
                    "non-decreasing in time"
                )
            previous = step.t_s

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def duration_s(self) -> float:
        """Span from the first to the last step, in seconds; 0 when empty."""
        if not self.steps:
            return 0.0
        return self.steps[-1].t_s - self.steps[0].t_s


@dataclass(frozen=True)
class Case:
    """A complete checkable input: dictionary, machine, initial state, sequence."""

    name: str
    dictionary: CommandDictionary
    machine: ModeMachine
    initial: InitialState
    sequence: CommandSequence
    description: str = ""

    def __post_init__(self) -> None:
        if self.initial.mode not in self.machine.modes:
            raise ValueError(
                f"case {self.name}: initial mode {self.initial.mode!r} is not a declared mode "
                f"(declared: {', '.join(self.machine.modes)})"
            )
        for spec in self.dictionary.commands.values():
            for mode in spec.required_modes:
                if mode not in self.machine.modes:
                    raise ValueError(
                        f"case {self.name}: command {spec.identifier} gates on undeclared "
                        f"mode {mode!r}"
                    )
            if spec.mode_effect is not None and spec.mode_effect not in self.machine.modes:
                raise ValueError(
                    f"case {self.name}: command {spec.identifier} transitions to undeclared "
                    f"mode {spec.mode_effect!r}"
                )

    @property
    def undeclared_commands(self) -> tuple[str, ...]:
        """Commands used by the sequence that the dictionary does not declare."""
        return tuple(
            sorted({s.command for s in self.sequence.steps if s.command not in self.dictionary})
        )


def steps_from_pairs(pairs: Sequence[tuple[float, str]]) -> CommandSequence:
    """Build a :class:`CommandSequence` from ``(t_s, command)`` pairs."""
    return CommandSequence(tuple(SequenceStep(t_s=float(t), command=c) for t, c in pairs))
