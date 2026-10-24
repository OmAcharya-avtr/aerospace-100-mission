"""Seeded synthetic generator: declarations, clean sequences, defective sequences.

No flight data is used anywhere in this package. Everything measured here is a
statement about this generator, and the generator's equations are its code.
That is a real limitation and it is recorded in ``DATASET_CARD.md`` and in the
README rather than left for a reader to discover.

The command dictionary is **fixed**, not random. A randomly wired dictionary
produces sequences whose legality is an accident of the wiring and a risk
ranker trained on it learns the wiring rather than anything about commanding.
What varies per case is the part that varies on a real mission: which
conditions are verified at the start, where the resource ceilings sit, the
per-command draws and holds within declared tolerances, and the sequence
itself.

The base dictionary is a sketch of a small Earth-observation spacecraft --
power, thermal, ADCS, propulsion, payload, comms, solid-state recorder -- with
twenty-four commands, five modes, ten predicates, two resource ceilings and one
mutual-exclusion group. It is a plausible shape, not any real vehicle's
dictionary, and no numerical value in it is taken from a published command
dictionary.

Clean by construction
---------------------
:func:`generate_clean_case` builds its sequence by rejection: a candidate
command is appended only if the forward executor reports no new violation at
that step. The resulting sequence is therefore violation-free against the
declared model by construction, and :func:`verify_clean` re-checks it with an
independent call. Those sequences are the negative control set: they must
report nothing, and ``validation/validate_negative_control_output.txt`` is the
measurement.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .defects import DefectRecord, inject, inject_random
from .model import (
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
)
from .trace import CheckReport, forward_execute

__all__ = [
    "MODES",
    "PREDICATES",
    "base_dictionary",
    "base_machine",
    "GeneratorConfig",
    "generate_clean_case",
    "generate_defective_case",
    "verify_clean",
    "build_corpus",
    "CorpusEntry",
]

#: Declared modes of the generated mode machine.
MODES: tuple[str, ...] = ("SAFE", "NOMINAL", "SCIENCE", "MANOEUVRE", "COMMS")

#: Declared predicates. Every one of them is written by at least one command.
PREDICATES: tuple[str, ...] = (
    "antenna_deployed",
    "attitude_verified",
    "bus_voltage_ok",
    "heater_on",
    "link_acquired",
    "payload_enabled",
    "propellant_valve_open",
    "recorder_armed",
    "star_tracker_locked",
    "thruster_armed",
)

_T = Verity.TRUE
_F = Verity.FALSE
_U = Verity.UNKNOWN

# (id, subsystem, required_modes, preconditions, effects, mode_effect,
#  draw, hold_s, min_interval_s, min_gap_after_s, irreversible, exclusion_group)
_BASE: tuple[tuple, ...] = (
    (
        "PWR_BUS_CHECK", "power", (), {}, {"bus_voltage_ok": _T}, None, {"power_w": 2.0}, 0.0,
        30.0, 2.0, False, None,
    ),
    (
        "PWR_BUS_SHED", "power", (), {}, {"bus_voltage_ok": _F}, None, {}, 0.0, 0.0, 2.0, False,
        None,
    ),
    (
        "THERM_HEATER_ON", "thermal", (), {"bus_voltage_ok": _T}, {"heater_on": _T}, None,
        {"power_w": 24.0}, 300.0, 60.0, 2.0, False, None,
    ),
    (
        "THERM_HEATER_OFF", "thermal", (), {}, {"heater_on": _F}, None, {}, 0.0, 0.0, 2.0,
        False, None,
    ),
    (
        "ADCS_ATT_VERIFY", "adcs", (), {"bus_voltage_ok": _T}, {"attitude_verified": _T}, None,
        {"power_w": 8.0}, 20.0, 45.0, 5.0, False, None,
    ),
    (
        "ADCS_SLEW", "adcs", ("NOMINAL", "SCIENCE", "MANOEUVRE"), {"bus_voltage_ok": _T},
        {"attitude_verified": _U, "star_tracker_locked": _F}, None, {"power_w": 34.0}, 120.0,
        180.0, 10.0, False, None,
    ),
    (
        "ADCS_ST_ACQUIRE", "adcs", (), {"attitude_verified": _T}, {"star_tracker_locked": _T},
        None, {"power_w": 12.0}, 60.0, 90.0, 5.0, False, None,
    ),
    (
        "ADCS_ST_RELEASE", "adcs", (), {}, {"star_tracker_locked": _F}, None, {}, 0.0, 0.0, 2.0,
        False, None,
    ),
    (
        "PAYLOAD_ENABLE", "payload", ("NOMINAL", "SCIENCE"),
        {"bus_voltage_ok": _T, "heater_on": _T}, {"payload_enabled": _T}, None,
        {"power_w": 18.0}, 90.0, 120.0, 5.0, False, None,
    ),
    (
        "PAYLOAD_SAFE", "payload", (), {}, {"payload_enabled": _F}, None, {}, 0.0, 0.0, 2.0,
        False, None,
    ),
    (
        "PAYLOAD_SCAN", "payload", ("SCIENCE",),
        {"payload_enabled": _T, "attitude_verified": _T}, {}, None,
        {"power_w": 46.0, "data_rate_mbps": 22.0}, 90.0, 300.0, 5.0, False, "high_power",
    ),
    (
        "PAYLOAD_CAL", "payload", ("SCIENCE",),
        {"payload_enabled": _T, "star_tracker_locked": _T}, {}, None,
        {"power_w": 30.0, "data_rate_mbps": 8.0}, 60.0, 600.0, 5.0, False, None,
    ),
    (
        "SSR_ARM", "recorder", (), {"bus_voltage_ok": _T}, {"recorder_armed": _T}, None,
        {"power_w": 6.0}, 0.0, 30.0, 2.0, False, None,
    ),
    (
        "SSR_DISARM", "recorder", (), {}, {"recorder_armed": _F}, None, {}, 0.0, 0.0, 2.0,
        False, None,
    ),
    (
        "SSR_PLAYBACK", "recorder", ("COMMS",), {"recorder_armed": _T}, {}, None,
        {"power_w": 14.0, "data_rate_mbps": 30.0}, 60.0, 240.0, 5.0, False, None,
    ),
    (
        "COMM_ACQUIRE", "comms", ("NOMINAL", "COMMS"), {"antenna_deployed": _T},
        {"link_acquired": _T}, None, {"power_w": 10.0}, 30.0, 60.0, 5.0, False, None,
    ),
    (
        "COMM_RELEASE", "comms", (), {}, {"link_acquired": _F}, None, {}, 0.0, 0.0, 2.0, False,
        None,
    ),
    (
        "DOWNLINK_START", "comms", ("COMMS",), {"link_acquired": _T, "recorder_armed": _T}, {},
        None, {"power_w": 38.0, "data_rate_mbps": 34.0}, 120.0, 300.0, 5.0, False, "high_power",
    ),
    (
        "ANT_DEPLOY", "structures", ("NOMINAL",),
        {"antenna_deployed": _F, "bus_voltage_ok": _T, "attitude_verified": _T},
        {"antenna_deployed": _T}, None, {"power_w": 20.0}, 10.0, 0.0, 30.0, True, None,
    ),
    (
        "PROP_ARM", "propulsion", ("NOMINAL", "MANOEUVRE"), {"bus_voltage_ok": _T},
        {"thruster_armed": _T}, None, {"power_w": 9.0}, 0.0, 60.0, 5.0, False, None,
    ),
    (
        "PROP_DISARM", "propulsion", (), {}, {"thruster_armed": _F}, None, {}, 0.0, 0.0, 2.0,
        False, None,
    ),
    (
        "PROP_VALVE_OPEN", "propulsion", ("MANOEUVRE",),
        {"propellant_valve_open": _F, "thruster_armed": _T, "attitude_verified": _T},
        {"propellant_valve_open": _T}, None, {"power_w": 16.0}, 20.0, 0.0, 10.0, True, None,
    ),
    (
        "PROP_BURN", "propulsion", ("MANOEUVRE",),
        {"propellant_valve_open": _T, "attitude_verified": _T, "thruster_armed": _T}, {}, None,
        {"power_w": 52.0, "data_rate_mbps": 2.0}, 45.0, 900.0, 20.0, True, "high_power",
    ),
    (
        "MEM_SCRUB", "obdh", ("SAFE", "NOMINAL"), {"bus_voltage_ok": _T}, {}, None,
        {"power_w": 11.0}, 180.0, 1800.0, 5.0, True, None,
    ),
    (
        "MODE_NOMINAL", "obdh", (), {"bus_voltage_ok": _T}, {}, "NOMINAL", {}, 0.0, 0.0, 5.0,
        False, None,
    ),
    (
        "MODE_SCIENCE", "obdh", ("NOMINAL",), {"attitude_verified": _T, "payload_enabled": _T},
        {}, "SCIENCE", {}, 0.0, 0.0, 5.0, False, None,
    ),
    (
        "MODE_COMMS", "obdh", ("NOMINAL",), {"link_acquired": _T}, {}, "COMMS", {}, 0.0, 0.0,
        5.0, False, None,
    ),
    (
        "MODE_MANOEUVRE", "obdh", ("NOMINAL",), {"attitude_verified": _T, "thruster_armed": _T},
        {}, "MANOEUVRE", {}, 0.0, 0.0, 5.0, False, None,
    ),
    (
        "MODE_SAFE", "obdh", (), {}, {"payload_enabled": _F, "link_acquired": _F}, "SAFE", {},
        0.0, 0.0, 5.0, False, None,
    ),
)

#: Declared resource ceilings of the base dictionary, before per-case jitter.
BASE_LIMITS: dict[str, float] = {"power_w": 120.0, "data_rate_mbps": 40.0}


def base_machine() -> ModeMachine:
    """The generated mode machine. Five modes, SAFE reachable from everywhere."""
    return ModeMachine(
        modes=MODES,
        transitions={
            "SAFE": ("NOMINAL",),
            "NOMINAL": ("SAFE", "SCIENCE", "MANOEUVRE", "COMMS"),
            "SCIENCE": ("NOMINAL", "SAFE"),
            "MANOEUVRE": ("NOMINAL", "SAFE"),
            "COMMS": ("NOMINAL", "SAFE"),
        },
    )


def base_dictionary(
    draw_scale: float = 1.0,
    hold_scale: float = 1.0,
    limits: dict[str, float] | None = None,
) -> CommandDictionary:
    """The fixed base dictionary, optionally with scaled draws and holds.

    Parameters
    ----------
    draw_scale:
        Multiplies every declared resource draw. Per-case jitter, not physics.
    hold_scale:
        Multiplies every declared hold duration.
    limits:
        Resource ceilings; :data:`BASE_LIMITS` when ``None``.

    Raises
    ------
    ValueError
        When either scale is not positive.
    """
    if draw_scale <= 0.0 or hold_scale <= 0.0:
        raise ValueError(
            f"draw_scale={draw_scale} and hold_scale={hold_scale} must both be > 0"
        )
    commands: dict[str, CommandSpec] = {}
    for (
        identifier,
        subsystem,
        modes,
        preconditions,
        effects,
        mode_effect,
        draw,
        hold_s,
        interval,
        gap,
        irreversible,
        exclusion,
    ) in _BASE:
        commands[identifier] = CommandSpec(
            identifier=identifier,
            subsystem=subsystem,
            required_modes=tuple(modes),
            preconditions=dict(preconditions),
            effects=dict(effects),
            mode_effect=mode_effect,
            resource_draw={k: round(v * draw_scale, 6) for k, v in sorted(draw.items())},
            hold_s=round(hold_s * hold_scale, 6),
            min_interval_s=interval,
            min_gap_after_s=gap,
            irreversible=irreversible,
            exclusion_group=exclusion,
        )
    return CommandDictionary(
        commands=commands,
        resource_limits=ResourceLimits(dict(sorted((limits or BASE_LIMITS).items()))),
    )


@dataclass(frozen=True)
class GeneratorConfig:
    """Shape of a generated case.

    Parameters
    ----------
    n_steps:
        Target sequence length. The builder stops early rather than emit a step
        it cannot place legally, so the realised length can be shorter and the
        realised length is what is reported.
    unverified_fraction:
        Expected fraction of predicates the initial state leaves ``UNKNOWN``.
        Zero makes the irreversible gate unreachable, which is why the default
        is not zero.
    dt_min_s, dt_max_s:
        Range the time advance between steps is drawn from, before any
        enlargement needed to satisfy a declared minimum spacing.
    draw_jitter, hold_jitter:
        Half-width of the uniform multiplicative jitter on declared draws and
        holds.
    limit_jitter:
        Half-width of the uniform multiplicative jitter on the resource
        ceilings.
    """

    n_steps: int = 36
    unverified_fraction: float = 0.25
    dt_min_s: float = 8.0
    dt_max_s: float = 70.0
    draw_jitter: float = 0.15
    hold_jitter: float = 0.20
    limit_jitter: float = 0.10

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps={self.n_steps} must be >= 1")
        if not 0.0 <= self.unverified_fraction <= 1.0:
            raise ValueError(
                f"unverified_fraction={self.unverified_fraction} must be in [0, 1]"
            )
        if self.dt_min_s <= 0.0 or self.dt_max_s < self.dt_min_s:
            raise ValueError(
                f"dt range [{self.dt_min_s}, {self.dt_max_s}] must be positive and ordered"
            )
        for label, value in (
            ("draw_jitter", self.draw_jitter),
            ("hold_jitter", self.hold_jitter),
            ("limit_jitter", self.limit_jitter),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{label}={value} must be in [0, 1)")


def _initial_state(rng: random.Random, config: GeneratorConfig) -> InitialState:
    """Declared initial state: SAFE or NOMINAL, with some conditions unverified.

    ``bus_voltage_ok`` is always verified true, because a generated case in
    which the bus is unverified has one violation at step zero and nothing
    interesting after it.
    """
    predicates: dict[str, Verity] = {}
    for name in PREDICATES:
        if name == "bus_voltage_ok":
            predicates[name] = _T
            continue
        if rng.random() < config.unverified_fraction:
            predicates[name] = _U
        elif rng.random() < 0.45:
            predicates[name] = _T
        else:
            predicates[name] = _F
    mode = "NOMINAL" if rng.random() < 0.8 else "SAFE"
    return InitialState(mode=mode, predicates=dict(sorted(predicates.items())))


def _jittered(rng: random.Random, config: GeneratorConfig) -> CommandDictionary:
    """Base dictionary with per-case jitter on draws, holds and ceilings."""

    def jitter(width: float) -> float:
        return 1.0 + rng.uniform(-width, width)

    limits = {
        key: round(value * jitter(config.limit_jitter), 6)
        for key, value in sorted(BASE_LIMITS.items())
    }
    return base_dictionary(
        draw_scale=jitter(config.draw_jitter),
        hold_scale=jitter(config.hold_jitter),
        limits=limits,
    )


#: Goal commands a generated sequence pursues, with selection weights. The
#: weights make the interesting commands common: a uniform random walk over the
#: dictionary places mostly teardown commands, which have no preconditions and
#: therefore always succeed, and produces sequences in which PAYLOAD_SCAN,
#: DOWNLINK_START and PROP_BURN never appear at all. That was measured on a
#: first version of this generator and it made four defect families
#: uninjectable, so the generator is goal-directed instead.
ACTIVITY_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("PAYLOAD_SCAN", 4),
    ("PAYLOAD_CAL", 3),
    ("DOWNLINK_START", 4),
    ("SSR_PLAYBACK", 3),
    ("PROP_BURN", 3),
    ("ANT_DEPLOY", 2),
    ("MEM_SCRUB", 2),
    ("ADCS_SLEW", 2),
    ("THERM_HEATER_ON", 1),
    ("PAYLOAD_SAFE", 1),
    ("SSR_DISARM", 1),
    ("COMM_RELEASE", 1),
    ("MODE_SAFE", 1),
    ("PWR_BUS_SHED", 1),
)

#: Maximum depth of the prerequisite resolution. Three is enough for the base
#: dictionary's longest chain (MODE_NOMINAL -> MODE_SCIENCE -> PAYLOAD_SCAN
#: with PAYLOAD_ENABLE and THERM_HEATER_ON underneath it) and a depth limit is
#: needed because a dictionary can declare a precondition cycle.
MAX_CHAIN_DEPTH = 4


def _providers(dictionary: CommandDictionary, name: str, required: Verity) -> tuple[str, ...]:
    """Commands whose declared effect sets ``name`` to ``required``, sorted."""
    return tuple(
        sorted(
            spec.identifier
            for spec in dictionary.commands.values()
            if spec.effects.get(name) is required
        )
    )


def _mode_providers(dictionary: CommandDictionary, mode: str) -> tuple[str, ...]:
    """Commands whose declared mode effect is ``mode``, sorted."""
    return tuple(
        sorted(
            spec.identifier for spec in dictionary.commands.values() if spec.mode_effect == mode
        )
    )


def _resolve_chain(
    dictionary: CommandDictionary,
    machine: ModeMachine,
    goal: str,
    predicates: dict[str, Verity],
    mode: str,
    rng: random.Random,
    depth: int = 0,
) -> list[str] | None:
    """Prerequisite chain ending in ``goal``, or ``None`` when unreachable.

    Walks the declared preconditions and mode gate of ``goal`` and emits, in
    order, the commands that satisfy them. ``predicates`` and ``mode`` are
    mutated to the state the chain leaves behind, so a caller can resolve
    several goals in sequence without re-simulating.

    This is a planner over the declarations, not over any vehicle. It can fail,
    and a failure returns ``None`` rather than a partial chain: half a
    prerequisite chain is a defect, and the generator's job is to emit clean
    sequences.
    """
    if depth > MAX_CHAIN_DEPTH:
        return None
    spec = dictionary.get(goal)
    if spec is None:
        return None

    chain: list[str] = []
    working = dict(predicates)
    working_mode = mode

    if spec.required_modes and working_mode not in spec.required_modes:
        placed = False
        for target in spec.required_modes:
            if not machine.allows(working_mode, target):
                hop = [m for m in machine.reachable_from(working_mode) if machine.allows(m, target)]
                if not hop:
                    continue
                for intermediate in hop:
                    for provider in _mode_providers(dictionary, intermediate):
                        sub = _resolve_chain(
                            dictionary, machine, provider, working, working_mode, rng, depth + 1
                        )
                        if sub is not None:
                            chain.extend(sub)
                            working_mode = intermediate
                            break
                    if working_mode == intermediate:
                        break
            for provider in _mode_providers(dictionary, target):
                if not machine.allows(working_mode, target):
                    continue
                sub = _resolve_chain(
                    dictionary, machine, provider, working, working_mode, rng, depth + 1
                )
                if sub is not None:
                    chain.extend(sub)
                    working_mode = target
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    for name in sorted(spec.preconditions):
        required = spec.preconditions[name]
        if working.get(name, Verity.UNKNOWN) is required:
            continue
        options = list(_providers(dictionary, name, required))
        rng.shuffle(options)
        satisfied = False
        for provider in options:
            if provider == goal:
                continue
            sub = _resolve_chain(
                dictionary, machine, provider, working, working_mode, rng, depth + 1
            )
            if sub is None:
                continue
            chain.extend(sub)
            satisfied = True
            break
        if not satisfied:
            return None

    for name in sorted(spec.effects):
        working[name] = spec.effects[name]
    if spec.mode_effect is not None and machine.allows(working_mode, spec.mode_effect):
        working_mode = spec.mode_effect

    chain.append(goal)
    predicates.clear()
    predicates.update(working)
    return chain


def _weighted_goal(rng: random.Random) -> str:
    """Draw a goal command from :data:`ACTIVITY_WEIGHTS`."""
    population = [name for name, weight in ACTIVITY_WEIGHTS for _ in range(weight)]
    return population[rng.randrange(len(population))]


def generate_clean_case(
    seed: int,
    config: GeneratorConfig | None = None,
    name: str | None = None,
) -> Case:
    """Build a violation-free case by goal-directed planning plus rejection.

    The generator repeatedly picks an activity from :data:`ACTIVITY_WEIGHTS`,
    resolves its prerequisite chain over the declared dictionary and mode
    machine, and appends the chain one command at a time. Each append is
    accepted only when forward execution of the extended sequence reports no
    violation at the new step, so the result is violation-free against the
    declared model by construction. A command that cannot be placed legally at
    the drawn time is retried at a later time up to
    :data:`_PLACEMENT_ATTEMPTS` times and then abandoned along with the rest of
    its chain.

    The cost is one forward walk per placement attempt, which is linear in the
    sequence built so far, so the generator is quadratic in the realised length.
    Measured generation time for the shipped corpus is recorded in
    ``validation/validate_negative_control_output.txt``.

    Returns
    -------
    Case
        A case whose sequence has no violation against its own declarations.
        The realised length is at most ``config.n_steps`` and is reported
        rather than assumed.
    """
    config = config or GeneratorConfig()
    rng = random.Random(seed)
    dictionary = _jittered(rng, config)
    machine = base_machine()
    initial = _initial_state(rng, config)

    case = Case(
        name=name or f"clean-{seed}",
        dictionary=dictionary,
        machine=machine,
        initial=initial,
        sequence=CommandSequence(()),
        description=(
            "Seeded clean sequence: goal-directed plan over the declared dictionary, "
            "accepted step by step against forward execution."
        ),
    )

    steps: list[SequenceStep] = []
    predicates = dict(initial.predicates)
    mode = initial.mode
    t_s = 0.0
    stalls = 0

    while len(steps) < config.n_steps and stalls < 8:
        goal = _weighted_goal(rng)
        chain = _resolve_chain(dictionary, machine, goal, dict(predicates), mode, rng)
        if chain is None:
            stalls += 1
            continue
        progressed = False
        for command in chain:
            if len(steps) >= config.n_steps:
                break
            placed = False
            for attempt in range(_PLACEMENT_ATTEMPTS):
                dt = _required_dt(dictionary, steps, command) + rng.uniform(
                    config.dt_min_s, config.dt_max_s
                ) * (1.0 + attempt)
                trial_t = round(t_s + dt, 3)
                trial = steps + [SequenceStep(t_s=trial_t, command=command)]
                report = forward_execute(replace(case, sequence=CommandSequence(tuple(trial))))
                if not report.snapshots[-1].violations:
                    steps = trial
                    t_s = trial_t
                    predicates = {
                        k: v for k, v in sorted(report.snapshots[-1].predicates.items())
                    }
                    mode = report.snapshots[-1].mode
                    placed = True
                    progressed = True
                    break
            if not placed:
                break
        stalls = 0 if progressed else stalls + 1

    return replace(case, sequence=CommandSequence(tuple(steps)))


#: Placement attempts per command before its chain is abandoned. Each attempt
#: pushes the command later, which is what clears an exclusion hold or a
#: resource hold that blocked the previous attempt.
_PLACEMENT_ATTEMPTS = 4


def _required_dt(
    dictionary: CommandDictionary, steps: list[SequenceStep], command: str
) -> float:
    """Minimum time advance that satisfies the declared spacing for ``command``.

    Hand-checkable: the gap owed to the previous command is its
    ``min_gap_after_s``; the interval owed to this command's own previous
    execution is ``min_interval_s`` measured from that execution, which is
    ``min_interval_s - (t_last_step - t_last_execution)`` from the last step.
    The maximum of the two is returned, floored at zero. Unit-checked in
    ``tests/test_dataset.py`` and in ``validation/validate_timing.py``.
    """
    if not steps:
        return 0.0
    owed = 0.0
    previous = dictionary.get(steps[-1].command)
    if previous is not None:
        owed = max(owed, float(previous.min_gap_after_s))
    spec = dictionary.get(command)
    if spec is not None and spec.min_interval_s > 0.0:
        for step in reversed(steps):
            if step.command == command:
                owed = max(owed, float(spec.min_interval_s) - (steps[-1].t_s - step.t_s))
                break
    return max(owed, 0.0)


def verify_clean(case: Case) -> CheckReport:
    """Forward-execute ``case`` and raise when it is not violation-free.

    Raises
    ------
    AssertionError
        When the case the generator declared clean is not clean. This is the
        generator's own self-check and a failure here is a generator defect,
        not a finding about the sequence.
    """
    report = forward_execute(case)
    if not report.clean:
        raise AssertionError(
            f"case {case.name} was generated as clean but reports {report.n_violations} "
            f"violation(s): {report.kind_counts()}"
        )
    return report


def generate_defective_case(
    seed: int,
    family: str | None = None,
    config: GeneratorConfig | None = None,
) -> tuple[Case, Case, DefectRecord]:
    """Generate a clean case and inject one defect into it.

    Returns
    -------
    tuple
        ``(clean_case, defective_case, record)``. When the injection was a
        no-op, ``defective_case is clean_case`` and ``record.applied`` is False.
    """
    clean = generate_clean_case(seed, config)
    if family is None:
        defective, record = inject_random(clean, seed)
    else:
        defective, record = inject(clean, family, seed)
    return clean, defective, record


@dataclass(frozen=True)
class CorpusEntry:
    """One corpus member: the case, its forward report and its defect record."""

    case: Case
    report: CheckReport
    record: DefectRecord | None
    seed: int

    @property
    def clean(self) -> bool:
        """True when the forward walk found nothing."""
        return self.report.clean


def build_corpus(
    seeds: Iterable[int],
    families: Iterable[str] | None = None,
    config: GeneratorConfig | None = None,
    include_clean: bool = True,
) -> tuple[CorpusEntry, ...]:
    """Build a corpus of clean and defective cases with their forward reports.

    Parameters
    ----------
    seeds:
        One clean case per seed, plus one defective case per seed per family.
    families:
        Defect families to inject; all of :data:`cmdguard.defects.FAMILIES`
        when ``None``.
    config:
        Generator shape.
    include_clean:
        Whether to include the clean cases. The negative control needs them;
        the ranker's training set does not benefit from sequences with no
        label to learn.
    """
    from .defects import FAMILIES

    families = tuple(families) if families is not None else FAMILIES
    entries: list[CorpusEntry] = []
    for seed in seeds:
        clean = generate_clean_case(seed, config)
        if include_clean:
            entries.append(CorpusEntry(clean, forward_execute(clean), None, seed))
        for family in families:
            defective, record = inject(clean, family, seed)
            if not record.applied:
                entries.append(CorpusEntry(clean, forward_execute(clean), record, seed))
                continue
            entries.append(
                CorpusEntry(defective, forward_execute(defective), record, seed)
            )
    return tuple(entries)
