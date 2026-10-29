"""Deterministic generation of dictionary pairs with known injected changes.

Everything measured in this repository is measured against pairs built here.
No flight database is used, read or approximated: the generator produces
plausible-looking mnemonics, bit layouts, calibrations and enumerations from a
seeded pseudo-random stream, and nothing in it is taken from any mission.

The point of the generator is that the ground truth is known exactly. Each
injector makes exactly one change of exactly one class to exactly one item,
and records it. Per-class precision and recall are then counts, not estimates.

Determinism: every function takes a ``seed`` and uses :class:`random.Random`
alone, so a given seed produces the same pair on any machine and across numpy
versions. No global random state is touched.

The reordering control
----------------------

:func:`reordering_control` re-serialises a dictionary with entry order, key
order and enumeration order shuffled and with a different key spelling for
most fields, producing a file that differs from the original on most of its
lines while being semantically identical. A semantic differ that reports
anything at all on that pair is reporting noise.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .changes import ChangeClass
from .corpus import Corpus, CorpusSource
from .model import Command, CommandArgument, Dictionary, Parameter
from .parse import command_to_mapping, dumps, parameter_to_mapping

#: Quantity templates: token, data type, unit, bit width, scale, offset, limits.
_QUANTITIES: tuple[tuple[str, str, str, int, float, float, tuple[float, float] | None], ...] = (
    ("T", "int", "K", 12, 0.0625, 0.0, (120.0, 360.0)),
    ("V", "uint", "V", 16, 0.001, 0.0, (0.0, 60.0)),
    ("I", "int", "A", 14, 0.0005, -4.0, (-10.0, 10.0)),
    ("P", "uint", "Pa", 16, 0.5, 0.0, (0.0, 30000.0)),
    ("W", "uint", "rad/s", 16, 1.0e-4, -3.2768, (-3.0, 3.0)),
    ("Q", "int", "", 16, 3.0517578125e-05, 0.0, (-1.0, 1.0)),
    ("N", "uint", "count", 32, 1.0, 0.0, None),
    ("E", "uint", "s", 32, 0.015625, 0.0, None),
    ("F", "float", "degC", 32, 1.0, 0.0, (-60.0, 120.0)),
    ("B", "bool", "", 1, 1.0, 0.0, None),
)

#: Subsystem prefixes and the packets they report in.
_SUBSYSTEMS: tuple[tuple[str, str], ...] = (
    ("THERM", "THERM_HK"),
    ("PWR", "PWR_HK"),
    ("AOCS", "AOCS_HK"),
    ("COMMS", "COMMS_HK"),
    ("PAYL", "PAYLOAD_HK"),
    ("OBDH", "OBDH_HK"),
)

#: Location tokens that make mnemonics look like mnemonics.
_LOCATIONS: tuple[str, ...] = (
    "BAT",
    "PANEL",
    "BUS",
    "RW",
    "STR",
    "GYRO",
    "MAG",
    "TANK",
    "HTR",
    "TX",
    "RX",
    "CPU",
    "MEM",
    "CAM",
    "LNA",
    "PCDU",
)

#: Status enumerations, used for the enumerated parameters.
_ENUMS: tuple[tuple[tuple[int, str], ...], ...] = (
    ((0, "OFF"), (1, "ON")),
    ((0, "NOMINAL"), (1, "DEGRADED"), (2, "FAILED")),
    ((0, "IDLE"), (1, "SAFE_MODE"), (2, "NOMINAL_MODE"), (3, "SCIENCE_MODE")),
    ((0, "CLOSED"), (1, "OPEN"), (2, "TRANSIT")),
    ((0, "UNLOCKED"), (1, "ACQUIRING"), (2, "LOCKED")),
)

_DESCRIPTIONS: tuple[str, ...] = (
    "{location} {quantity} on the {subsystem} chain, sampled each housekeeping cycle",
    "{subsystem} {location} {quantity}, calibrated on ground",
    "raw {quantity} reported by the {subsystem} {location} unit",
    "{location} {quantity} used by the {subsystem} limit check",
    "{subsystem} {location} {quantity} downlinked in the housekeeping packet",
)

_QUANTITY_WORDS: dict[str, str] = {
    "T": "temperature",
    "V": "voltage",
    "I": "current",
    "P": "pressure",
    "W": "angular rate",
    "Q": "quaternion component",
    "N": "counter",
    "E": "elapsed time",
    "F": "temperature",
    "B": "flag",
    "S": "status",
}

#: Command verbs and the arguments they take.
_COMMAND_VERBS: tuple[tuple[str, tuple[tuple[str, str, int, str], ...]], ...] = (
    ("SET_MODE", (("MODE", "uint", 8, ""), ("TIMEOUT", "uint", 16, "s"))),
    ("SET_HEATER", (("HEATER_ID", "uint", 8, ""), ("DUTY", "uint", 8, "percent"))),
    ("ARM_UNIT", (("UNIT_ID", "uint", 8, ""),)),
    ("SET_LIMIT", (("PARAM_ID", "uint", 16, ""), ("LOW", "int", 16, ""), ("HIGH", "int", 16, ""))),
    ("DUMP_MEMORY", (("START", "uint", 32, ""), ("LENGTH", "uint", 16, "byte"))),
    ("SET_RATE", (("CHANNEL", "uint", 8, ""), ("RATE", "uint", 16, "Hz")),),
    ("POWER_CYCLE", (("UNIT_ID", "uint", 8, ""), ("DELAY", "uint", 16, "s"))),
    ("CALIBRATE", (("TARGET", "uint", 8, ""),)),
)

#: Alias spellings used by :func:`reordering_control` to make the textual
#: difference substantial while the meaning stays identical.
REORDER_ALIASES: dict[str, str] = {
    "name": "mnemonic",
    "description": "desc",
    "data_type": "type",
    "bit_width": "bits",
    "unit": "units",
    "scale": "factor",
    "offset": "bias",
    "low_limit": "min",
    "high_limit": "max",
    "packet": "apid",
    "enum": "states",
    "arguments": "args",
    "position": "index",
}


@dataclass(frozen=True)
class InjectedChange:
    """One change the generator made on purpose, with the class it belongs to."""

    change_class: ChangeClass
    kind: str
    name: str
    field: str = ""
    new_name: str = ""

    @property
    def key(self) -> tuple[ChangeClass, str]:
        """``(class, name)``, the key detection is matched on."""
        return (self.change_class, self.name)


@dataclass(frozen=True)
class SynthPair:
    """A generated dictionary pair with its ground truth."""

    old: Dictionary
    new: Dictionary
    truth: tuple[InjectedChange, ...]

    @property
    def truth_keys(self) -> set[tuple[ChangeClass, str]]:
        """Ground-truth ``(class, name)`` keys."""
        return {change.key for change in self.truth}

    def truth_for(self, change_class: ChangeClass) -> tuple[InjectedChange, ...]:
        """Injected changes of one class."""
        return tuple(c for c in self.truth if c.change_class is change_class)


def _mnemonic(rng: random.Random, subsystem: str, token: str, index: int) -> str:
    location = rng.choice(_LOCATIONS)
    return f"{subsystem[:4]}_{token}_{location}_{index:02d}"


#: Bit offsets are byte-aligned and wrap within each packet. This is what makes
#: structural collisions possible and therefore makes a false-match rate
#: measurable: a generator that gave every parameter a globally unique bit
#: offset would make every structural signature unique, every rename trivially
#: separable, and the measured false-match rate zero at every threshold -- which
#: is what the first version of this generator did, and the resulting sweep was
#: vacuous. Real housekeeping packets reuse the same offsets across containers.
_OFFSET_WRAP_BITS = 512


def _fresh_parameter(
    rng: random.Random,
    index: int,
    *,
    offsets: dict[str, int] | None = None,
    path_prefix: str = "TM",
) -> Parameter:
    """One generated parameter, from the same distribution wherever it is needed.

    Used by :func:`baseline_dictionary` and by the ``added`` injector, so that
    an added parameter is structurally as plausible a rename partner as any
    existing one. An injector drawing from a different distribution would make
    false matches artificially rare.
    """
    subsystem, packet = _SUBSYSTEMS[index % len(_SUBSYSTEMS)]
    if index % 6 == 5:
        token = "S"
        data_type = "enum"
        unit = ""
        bit_width = 8
        scale = 1.0
        offset_value = 0.0
        limits = None
        enum = rng.choice(_ENUMS)
    else:
        token, data_type, unit, bit_width, scale, offset_value, limits = rng.choice(_QUANTITIES)
        enum = ()
    if offsets is None:
        bit_offset = (rng.randrange(0, _OFFSET_WRAP_BITS // 8)) * 8
    else:
        bit_offset = offsets.get(packet, 0) % _OFFSET_WRAP_BITS
        offsets[packet] = offsets.get(packet, 0) + 8 * max(1, (bit_width + 7) // 8)
    name = _mnemonic(rng, subsystem, token, index)
    description = rng.choice(_DESCRIPTIONS).format(
        location=name.split("_")[2].lower(),
        quantity=_QUANTITY_WORDS[token],
        subsystem=subsystem,
    )
    return Parameter(
        name=name,
        path=f"{path_prefix}/{subsystem}/{name}",
        description=description,
        data_type=data_type,
        bit_offset=bit_offset,
        bit_width=bit_width,
        byte_order="big",
        scale=scale,
        offset=offset_value,
        unit=unit,
        enum=enum,
        low_limit=None if limits is None else limits[0],
        high_limit=None if limits is None else limits[1],
        packet=packet,
    )


def baseline_dictionary(size: int, seed: int, *, version: str = "1.0") -> Dictionary:
    """A dictionary of ``size`` parameters plus ``max(3, size // 4)`` commands.

    ``size`` must be at least 4. Parameter names are unique by construction
    through the trailing index. Roughly one parameter in six carries an
    enumeration. Bit offsets are byte-aligned and wrap within each packet, so
    parameters in different packets share offsets, which is what makes
    structural signature collisions -- and therefore a measurable false-match
    rate -- possible.
    """
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError(f"size must be an int, got {type(size).__name__}")
    if size < 4:
        raise ValueError(f"size must be >= 4 to leave room for injections, got {size}")
    rng = random.Random(seed)

    offsets: dict[str, int] = {}
    parameters: list[Parameter] = [
        _fresh_parameter(rng, index, offsets=offsets) for index in range(size)
    ]

    # One command per four parameters, floored at three. The floor is what lets a
    # pair carry one injection of every command-side class at the smallest
    # dictionary size this generator supports; with one command per five
    # parameters a size-30 pair ran out of commands partway through the
    # all-classes injection and raised rather than silently skipping one.
    commands: list[Command] = []
    for index in range(max(3, size // 4)):
        verb, argument_specs = _COMMAND_VERBS[index % len(_COMMAND_VERBS)]
        subsystem, _ = _SUBSYSTEMS[index % len(_SUBSYSTEMS)]
        name = f"{subsystem[:4]}_{verb}_{index:02d}"
        commands.append(
            Command(
                name=name,
                path=f"TC/{subsystem}/{name}",
                description=f"{verb.replace('_', ' ').lower()} on the {subsystem} subsystem",
                arguments=tuple(
                    CommandArgument(
                        name=argument_name,
                        position=position,
                        data_type=data_type,
                        bit_width=bit_width,
                        unit=unit,
                    )
                    for position, (argument_name, data_type, bit_width, unit) in enumerate(
                        argument_specs
                    )
                ),
            )
        )
    return Dictionary(version=version, parameters=tuple(parameters), commands=tuple(commands))


#: Rename transforms, applied to produce a renamed mnemonic. Each is a
#: plausible database-maintenance rename, and they span a range of textual
#: similarity on purpose, so measured rename recall is not an artefact of one
#: easy transform.
_RENAME_TRANSFORMS: tuple[tuple[str, str], ...] = (
    ("expand_bat", "BAT->BATTERY"),
    ("expand_token", "T->TEMP"),
    ("suffix_raw", "append _RAW"),
    ("drop_index", "drop trailing index"),
    ("subsystem_first", "reorder subsystem and token"),
)


def _rename(name: str, transform: str, taken: set[str]) -> str:
    parts = name.split("_")
    if transform == "expand_bat":
        candidate = name.replace("BAT", "BATTERY").replace("PANEL", "SOLAR_PANEL")
    elif transform == "expand_token":
        expansions = {"T": "TEMP", "V": "VOLT", "I": "CURR", "P": "PRESS", "W": "RATE"}
        if len(parts) >= 2 and parts[1] in expansions:
            parts[1] = expansions[parts[1]]
        candidate = "_".join(parts)
    elif transform == "suffix_raw":
        candidate = f"{name}_RAW"
    elif transform == "drop_index":
        candidate = "_".join(parts[:-1]) if len(parts) > 2 else f"{name}_A"
    else:
        candidate = "_".join([*parts[1:], parts[0]])
    if candidate == name or candidate in taken:
        candidate = f"{candidate}_NEW"
    suffix = 2
    while candidate in taken:
        candidate = f"{candidate}{suffix}"
        suffix += 1
    return candidate


def _eligible(parameters: Sequence[Parameter], predicate: str) -> list[int]:
    out: list[int] = []
    for index, parameter in enumerate(parameters):
        if predicate == "enum" and parameter.enum:
            out.append(index)
        elif predicate == "multibyte" and parameter.bit_width > 8:
            out.append(index)
        elif predicate == "numeric" and parameter.data_type in ("int", "uint", "float"):
            out.append(index)
        elif predicate == "limits" and parameter.low_limit is not None:
            out.append(index)
        elif predicate == "any":
            out.append(index)
    return out


#: Injectable classes and the eligibility predicate each needs.
INJECTABLE: dict[ChangeClass, str] = {
    ChangeClass.RESCALED: "numeric",
    ChangeClass.UNIT_CHANGED: "numeric",
    ChangeClass.REENUMERATED: "enum",
    ChangeClass.BIT_SHIFTED: "any",
    ChangeClass.RETYPED: "numeric",
    ChangeClass.ENDIANNESS_CHANGED: "multibyte",
    ChangeClass.PACKET_MOVED: "any",
    ChangeClass.LIMITS_CHANGED: "limits",
    ChangeClass.PATH_MOVED: "any",
    ChangeClass.DESCRIPTION_CHANGED: "any",
    ChangeClass.REMOVED: "any",
    ChangeClass.RENAMED: "any",
    ChangeClass.ADDED: "any",
    ChangeClass.ARGUMENT_REORDERED: "command",
    ChangeClass.ARGUMENT_RETYPED: "command",
    ChangeClass.ARGUMENT_RESCALED: "command",
    ChangeClass.ARGUMENT_REMOVED: "command",
    ChangeClass.ARGUMENT_ADDED: "command",
    ChangeClass.COMMAND_REMOVED: "command",
    ChangeClass.COMMAND_RENAMED: "command",
    ChangeClass.COMMAND_ADDED: "command",
}

#: The silent tier, as injectable classes. Measured separately and first.
INJECTABLE_SILENT: tuple[ChangeClass, ...] = tuple(
    change_class for change_class in INJECTABLE if change_class.is_silent
)


def _inject_parameter(
    parameter: Parameter, change_class: ChangeClass, rng: random.Random, taken: set[str]
) -> Parameter:
    if change_class is ChangeClass.RESCALED:
        factor = rng.choice([2.0, 0.5, 10.0, 1.0 / 16.0, 1.8])
        return parameter.with_fields(scale=parameter.scale * factor)
    if change_class is ChangeClass.UNIT_CHANGED:
        swaps = {"K": "degC", "V": "mV", "A": "mA", "Pa": "bar", "s": "ms", "degC": "K"}
        return parameter.with_fields(unit=swaps.get(parameter.unit, parameter.unit + "_SI"))
    if change_class is ChangeClass.REENUMERATED:
        codes = dict(parameter.enum)
        labels = sorted(codes.items())
        first, second = labels[0], labels[-1]
        codes[first[0]] = second[1]
        codes[second[0]] = first[1]
        return parameter.with_fields(enum=tuple(sorted(codes.items())))
    if change_class is ChangeClass.BIT_SHIFTED:
        return parameter.with_fields(bit_offset=parameter.bit_offset + rng.choice([1, 2, 8]))
    if change_class is ChangeClass.RETYPED:
        swaps = {"uint": "int", "int": "uint", "float": "int"}
        return parameter.with_fields(data_type=swaps[parameter.data_type])
    if change_class is ChangeClass.ENDIANNESS_CHANGED:
        return parameter.with_fields(
            byte_order="little" if parameter.byte_order == "big" else "big"
        )
    if change_class is ChangeClass.PACKET_MOVED:
        others = [packet for _, packet in _SUBSYSTEMS if packet != parameter.packet]
        return parameter.with_fields(packet=rng.choice(others))
    if change_class is ChangeClass.LIMITS_CHANGED:
        low = parameter.low_limit
        high = parameter.high_limit
        span = (high - low) if (low is not None and high is not None) else 1.0
        return parameter.with_fields(
            low_limit=None if low is None else low + 0.05 * span,
            high_limit=None if high is None else high - 0.05 * span,
        )
    if change_class is ChangeClass.PATH_MOVED:
        return parameter.with_fields(path=parameter.path.replace("TM/", "TM/HK/", 1))
    if change_class is ChangeClass.DESCRIPTION_CHANGED:
        return parameter.with_fields(description=parameter.description + ", revised wording")
    if change_class is ChangeClass.RENAMED:
        transform = rng.choice([name for name, _ in _RENAME_TRANSFORMS])
        return parameter.with_fields(name=_rename(parameter.name, transform, taken))
    raise ValueError(f"{change_class} is not a parameter injection")


def _inject_command(
    command: Command, change_class: ChangeClass, rng: random.Random, taken: set[str]
) -> Command:
    arguments = list(command.arguments)
    if change_class is ChangeClass.ARGUMENT_REORDERED:
        if len(arguments) < 2:
            raise ValueError("argument_reordered needs a command with at least two arguments")
        first, second = arguments[0], arguments[1]
        arguments[0] = first.__class__(**{**first.__dict__, "position": second.position})
        arguments[1] = second.__class__(**{**second.__dict__, "position": first.position})
        return Command(
            name=command.name,
            path=command.path,
            description=command.description,
            arguments=tuple(arguments),
        )
    if change_class is ChangeClass.ARGUMENT_RETYPED:
        target = arguments[0]
        arguments[0] = CommandArgument(
            **{**target.__dict__, "bit_width": target.bit_width * 2}
        )
    elif change_class is ChangeClass.ARGUMENT_RESCALED:
        target = arguments[0]
        arguments[0] = CommandArgument(
            **{**target.__dict__, "scale": target.scale * rng.choice([2.0, 0.5])}
        )
    elif change_class is ChangeClass.ARGUMENT_REMOVED:
        if len(arguments) < 2:
            raise ValueError("argument_removed needs a command with at least two arguments")
        arguments.pop()
    elif change_class is ChangeClass.ARGUMENT_ADDED:
        arguments.append(
            CommandArgument(
                name="EXTRA_FLAG", position=len(arguments), data_type="uint", bit_width=8
            )
        )
    elif change_class is ChangeClass.COMMAND_RENAMED:
        return Command(
            name=_rename(command.name, "suffix_raw", taken),
            path=command.path,
            description=command.description,
            arguments=command.arguments,
        )
    else:
        raise ValueError(f"{change_class} is not a command injection")
    return Command(
        name=command.name,
        path=command.path,
        description=command.description,
        arguments=tuple(arguments),
    )


def inject_changes(
    old: Dictionary,
    classes: Sequence[ChangeClass],
    seed: int,
    *,
    per_class: int = 1,
    new_version: str = "1.1",
) -> SynthPair:
    """Apply ``per_class`` injections of each class in ``classes`` to ``old``.

    Each injection targets an item no other injection touches, so the classes
    do not interact and the ground truth is unambiguous. An injection whose
    eligibility predicate cannot be satisfied raises ``ValueError`` rather than
    being silently skipped: a skipped injection would make a recall figure
    wrong in the flattering direction.
    """
    if not isinstance(old, Dictionary):
        raise TypeError("inject_changes expects a Dictionary")
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")
    rng = random.Random(seed)
    parameters = list(old.parameters)
    commands = list(old.commands)
    truth: list[InjectedChange] = []
    used_parameters: set[int] = set()
    used_commands: set[int] = set()
    taken = set(old.item_names)

    for change_class in classes:
        if change_class not in INJECTABLE:
            raise ValueError(f"{change_class} cannot be injected by this generator")
        predicate = INJECTABLE[change_class]
        for _ in range(per_class):
            if change_class is ChangeClass.ADDED:
                index = rng.randrange(60, 600)
                extra = _fresh_parameter(rng, index)
                name = extra.name
                while name in taken:
                    index += 6
                    extra = _fresh_parameter(rng, index)
                    name = extra.name
                parameters.append(extra.with_fields(path=f"TM/{name.split('_')[0]}/{name}"))
                taken.add(name)
                truth.append(InjectedChange(change_class, "parameter", name, "name"))
                continue
            if change_class is ChangeClass.COMMAND_ADDED:
                index = len(commands) + rng.randrange(1000, 9000)
                extra = baseline_dictionary(6, seed + index, version=old.version).commands[0]
                name = f"{extra.name}_X{index}"
                commands.append(
                    Command(
                        name=name,
                        path=f"TC/NEW/{name}",
                        description=extra.description,
                        arguments=extra.arguments,
                    )
                )
                taken.add(name)
                truth.append(InjectedChange(change_class, "command", name, "name"))
                continue

            if predicate == "command":
                eligible = [
                    index
                    for index in range(len(commands))
                    if index not in used_commands
                    and (
                        len(commands[index].arguments) >= 2
                        or change_class
                        in (
                            ChangeClass.ARGUMENT_RETYPED,
                            ChangeClass.ARGUMENT_RESCALED,
                            ChangeClass.ARGUMENT_ADDED,
                            ChangeClass.COMMAND_REMOVED,
                            ChangeClass.COMMAND_RENAMED,
                        )
                    )
                ]
                if not eligible:
                    raise ValueError(
                        f"no command left to inject {change_class.value} into; raise the "
                        "dictionary size or lower per_class"
                    )
                index = rng.choice(eligible)
                used_commands.add(index)
                command = commands[index]
                if change_class is ChangeClass.COMMAND_REMOVED:
                    commands.pop(index)
                    # The removed command no longer exists, so it leaves the used
                    # set rather than being renumbered into its neighbour: keeping
                    # it marked an untouched command as used and exhausted the pool
                    # one injection early.
                    used_commands = {
                        i if i < index else i - 1 for i in used_commands - {index}
                    }
                    truth.append(InjectedChange(change_class, "command", command.name, "name"))
                    continue
                replacement = _inject_command(command, change_class, rng, taken)
                commands[index] = replacement
                taken.add(replacement.name)
                truth.append(
                    InjectedChange(
                        change_class,
                        "command",
                        command.name,
                        "name",
                        replacement.name if replacement.name != command.name else "",
                    )
                )
                continue

            eligible = [
                index
                for index in _eligible(parameters, predicate)
                if index not in used_parameters and index < len(old.parameters)
            ]
            if not eligible:
                raise ValueError(
                    f"no parameter left to inject {change_class.value} into; raise the "
                    "dictionary size or lower per_class"
                )
            index = rng.choice(eligible)
            used_parameters.add(index)
            parameter = parameters[index]
            if change_class is ChangeClass.REMOVED:
                parameters[index] = parameter  # placeholder, removed below
                truth.append(InjectedChange(change_class, "parameter", parameter.name, "name"))
                parameters[index] = None  # type: ignore[call-overload]
                continue
            replacement = _inject_parameter(parameter, change_class, rng, taken)
            parameters[index] = replacement
            taken.add(replacement.name)
            truth.append(
                InjectedChange(
                    change_class,
                    "parameter",
                    parameter.name,
                    "name",
                    replacement.name if replacement.name != parameter.name else "",
                )
            )

    new = Dictionary(
        version=new_version,
        parameters=tuple(p for p in parameters if p is not None),
        commands=tuple(commands),
    )
    return SynthPair(old=old, new=new, truth=tuple(truth))


def change_pair(
    size: int,
    seed: int,
    *,
    classes: Sequence[ChangeClass] | None = None,
    per_class: int = 1,
) -> SynthPair:
    """A baseline dictionary and a changed copy, with ground truth.

    With ``classes=None`` every injectable class is applied once. Seven of the
    classes consume a distinct command, and :func:`baseline_dictionary` builds
    ``max(3, size // 4)`` commands, so the all-classes case needs ``size >=
    28``. A smaller size raises ``ValueError`` naming the class that ran out
    rather than silently skipping it: a skipped injection would make a recall
    figure wrong in the flattering direction.
    """
    old = baseline_dictionary(size, seed)
    active = tuple(INJECTABLE) if classes is None else tuple(classes)
    return inject_changes(old, active, seed + 1, per_class=per_class)


def rename_pair(
    size: int, seed: int, *, renames: int, removals: int, additions: int
) -> SynthPair:
    """A pair with ``renames`` true renames plus unrelated removals and additions.

    The removals and additions are what make a false-match rate measurable: a
    matcher with no unpaired items cannot make a false match.
    """
    old = baseline_dictionary(size, seed)
    classes: list[ChangeClass] = []
    classes.extend([ChangeClass.RENAMED] * renames)
    classes.extend([ChangeClass.REMOVED] * removals)
    classes.extend([ChangeClass.ADDED] * additions)
    rng = random.Random(seed + 7)
    rng.shuffle(classes)
    return inject_changes(old, classes, seed + 2, per_class=1)


def _shuffled_mapping(
    mapping: dict[str, object], rng: random.Random, *, aliases: bool
) -> dict[str, object]:
    keys = list(mapping)
    rng.shuffle(keys)
    out: dict[str, object] = {}
    for key in keys:
        value = mapping[key]
        if isinstance(value, dict):
            inner_keys = list(value)
            rng.shuffle(inner_keys)
            value = {inner: value[inner] for inner in inner_keys}
        name = REORDER_ALIASES.get(key, key) if aliases else key
        out[name] = value
    return out


def reordering_control(
    dictionary: Dictionary, seed: int, *, aliases: bool = True, indent: int = 4
) -> tuple[str, str]:
    """Return ``(canonical_text, reordered_text)`` for the negative control.

    The reordered text differs from the canonical text in entry order, key
    order within every entry, enumeration order, command-argument declaration
    order and the spelling of most keys. It contains the same information. The
    semantic diff of the two must be empty, and
    ``validation/validate_negative_control.py`` measures how many text lines
    differ so that "substantial textual reordering" is a number.
    """
    rng = random.Random(seed)
    parameters = [parameter_to_mapping(p) for p in dictionary.parameters]
    commands = [command_to_mapping(c) for c in dictionary.commands]
    rng.shuffle(parameters)
    rng.shuffle(commands)
    shuffled_parameters = [_shuffled_mapping(p, rng, aliases=aliases) for p in parameters]
    shuffled_commands = []
    for command in commands:
        arguments = list(command.get("arguments", []))
        rng.shuffle(arguments)
        command = {
            **command,
            "arguments": [_shuffled_mapping(a, rng, aliases=aliases) for a in arguments],
        }
        shuffled_commands.append(_shuffled_mapping(command, rng, aliases=aliases))
    body = {
        "version": dictionary.version,
        REORDER_ALIASES.get("parameters", "parameters"): shuffled_parameters,
        "commands": shuffled_commands,
    }
    keys = list(body)
    rng.shuffle(keys)
    reordered = {key: body[key] for key in keys}
    return (dumps(dictionary), json.dumps(reordered, indent=indent) + "\n")


def synthetic_corpus(
    dictionary: Dictionary,
    seed: int,
    *,
    names: Sequence[str] | None = None,
    sources: int = 6,
    undecidable_per_script: int = 2,
    include_unreadable: bool = False,
) -> Corpus:
    """Build a corpus of procedures, scripts and configuration files.

    ``names`` fixes which items are referenced, so a caller measuring impact
    can guarantee that every changed item appears. With ``names=None`` a
    deterministic two-thirds sample of the dictionary is referenced, which
    leaves the rest uncovered on purpose: partial coverage is the normal case
    and the coverage verdict has to report it.
    """
    rng = random.Random(seed)
    parameters = list(dictionary.parameters)
    commands = list(dictionary.commands)
    if names is None:
        chosen = rng.sample(
            [p.name for p in parameters], k=max(1, (2 * len(parameters)) // 3)
        )
    else:
        chosen = list(names)
    parameter_by_name = dictionary.parameter_map
    command_by_name = dictionary.command_map
    chosen_parameters = [parameter_by_name[n] for n in chosen if n in parameter_by_name]
    chosen_commands = [command_by_name[n] for n in chosen if n in command_by_name] or commands[:1]

    blocks: list[CorpusSource] = []
    per_source = max(1, len(chosen_parameters) // max(1, sources - 2))

    procedure_lines: list[list[str]] = []
    script_lines: list[list[str]] = []
    config_lines: list[list[str]] = []
    for start in range(0, len(chosen_parameters), per_source):
        group = chosen_parameters[start : start + per_source]
        procedure: list[str] = ["* generated procedure, no mission heritage"]
        script: list[str] = ["# generated analysis script", "from client import tm"]
        config: list[str] = ["# generated limit configuration"]
        for parameter in group:
            if parameter.low_limit is not None:
                procedure.append(f"CHECK {parameter.name} > {parameter.low_limit}")
            else:
                procedure.append(f"MONITOR {parameter.path}")
            if parameter.enum:
                label = parameter.enum[rng.randrange(len(parameter.enum))][1]
                procedure.append(f'WAIT UNTIL {parameter.name} == "{label}"')
            script.append(f'series = tm["{parameter.name}"]')
            config.append(f"parameter: {parameter.name}")
        for index in range(undecidable_per_script):
            script.append(f'channel = tm["{group[0].name[:-2]}%02d" % index_{index}]')
        procedure_lines.append(procedure)
        script_lines.append(script)
        config_lines.append(config)

    for command in chosen_commands:
        arguments = ", ".join(f"{a.name}=1" for a in command.arguments)
        procedure_lines[0].append(f"SEND {command.name}({arguments})")
        script_lines[0].append(f'send_command("{command.name}", {arguments})')
        config_lines[0].append(f"command: {command.name}")

    for index, lines in enumerate(procedure_lines):
        blocks.append(
            CorpusSource(
                label=f"procedures/p{index:02d}.proc",
                kind="procedure",
                text="\n".join(lines),
            )
        )
    for index, lines in enumerate(script_lines):
        blocks.append(
            CorpusSource(label=f"analysis/a{index:02d}.py", kind="script", text="\n".join(lines))
        )
    for index, lines in enumerate(config_lines):
        blocks.append(
            CorpusSource(label=f"configs/c{index:02d}.cfg", kind="config", text="\n".join(lines))
        )

    unreadable = ("binary/dump.bin (suffix .bin not recognised)",) if include_unreadable else ()
    return Corpus(sources=tuple(blocks), unreadable=unreadable)
