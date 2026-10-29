"""Telemetry and command dictionary data model.

The record layout here is a deliberately small subset of what a real
telemetry/command (TM/TC) database carries. The fields chosen are the ones
whose change can alter the *meaning* of a parameter without altering its name,
which is the failure this package exists to find.

Field semantics and the standards they follow
---------------------------------------------

- ``bit_offset``, ``bit_width``, ``byte_order``: position of the raw field
  inside a telemetry source packet data field. Packet structure follows
  CCSDS 133.0-B-2, *Space Packet Protocol*; the convention that a parameter is
  located by an offset and a width inside the user data field, rather than by
  a C struct layout, follows CCSDS 660.0-B, *XML Telemetric and Command
  Exchange* (XTCE), which models location as ``LocationInContainerInBits``.
  This package implements none of the XTCE schema and reads none of its files.
- ``scale``, ``offset``: a degree-one polynomial calibration, engineering
  value = ``scale * raw + offset``. Degree-one polynomial calibration is the
  ``PolynomialCalibrator`` case of XTCE and the ``MCF``/numerical calibration
  case of the ECSS packet utilisation standard ECSS-E-ST-70-41C (2016),
  *Telemetry and telecommand packet utilization*, clause 5 service 3
  housekeeping reporting. Units are whatever ``unit`` declares; this package
  does not interpret unit strings beyond string identity.
- ``enum``: a raw-code to label map, the ``EnumeratedParameterType`` of XTCE
  and the status-code calibration of ECSS-E-ST-70-41C. Stored canonically
  sorted by code, because the order of an enumeration in a file is not
  semantic and the negative control in this package relies on that.
- ``low_limit``, ``high_limit``: engineering-unit soft limits, in the sense of
  the out-of-limit checking of ECSS-E-ST-70-41C service 12. Changing them does
  not change what a value means but does change which values raise an alarm,
  so they are classified separately from a calibration change.

Nothing in this module models a real spacecraft. No validity range, no
engineering unit and no bit layout here is taken from any flight database.

Validity range of the model itself: ``bit_width`` up to 64, integer codes
inside the declared width, finite ``scale`` and ``offset``. Values outside
those ranges raise ``ValueError`` at construction rather than being silently
accepted, because a dictionary that cannot be constructed is a better outcome
than a diff computed over a record the tool misunderstands.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

#: Data types this package understands. A type outside this set is rejected at
#: construction: a diff over a type whose width and signedness are unknown
#: cannot say whether a reinterpretation is silent.
DATA_TYPES: tuple[str, ...] = ("uint", "int", "float", "bool", "enum", "string")

#: Byte orders. Both spellings of each are accepted and normalised.
BYTE_ORDERS: tuple[str, ...] = ("big", "little")

_BYTE_ORDER_ALIASES = {
    "big": "big",
    "little": "little",
    "be": "big",
    "le": "little",
    "msb": "big",
    "lsb": "little",
    "network": "big",
}

#: Maximum supported ``bit_width``. 64 is the width of the widest integer the
#: raw-code range check can perform exactly in Python's int type without the
#: check itself becoming the limiting factor.
MAX_BIT_WIDTH = 64


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _normalise_enum(
    enum: Mapping[int, str] | Sequence[tuple[int, str]] | None,
    *,
    owner: str,
) -> tuple[tuple[int, str], ...]:
    """Canonicalise an enumeration to a tuple of ``(code, label)`` sorted by code.

    Sorting is what makes a reordered enumeration in a source file a non-change.
    """
    if enum is None:
        return ()
    items: list[tuple[int, str]]
    if isinstance(enum, Mapping):
        items = [(k, v) for k, v in enum.items()]
    elif isinstance(enum, (str, bytes)):
        raise TypeError(f"{owner}: enum must be a mapping or a sequence of pairs, not a string")
    elif isinstance(enum, Iterable):
        items = []
        for entry in enum:
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes))
                or len(entry) != 2
            ):
                raise TypeError(f"{owner}: each enum entry must be a (code, label) pair")
            items.append((entry[0], entry[1]))
    else:
        raise TypeError(f"{owner}: enum must be a mapping or a sequence of pairs")

    cleaned: list[tuple[int, str]] = []
    for code, label in items:
        code_int = _require_int(code, f"{owner}: enum code")
        label_str = _require_str(label, f"{owner}: enum label for code {code_int}")
        if not label_str:
            raise ValueError(f"{owner}: enum label for code {code_int} is empty")
        cleaned.append((code_int, label_str))

    codes = [code for code, _ in cleaned]
    if len(set(codes)) != len(codes):
        duplicates = sorted({c for c in codes if codes.count(c) > 1})
        raise ValueError(f"{owner}: duplicate enum codes {duplicates}")
    labels = [label for _, label in cleaned]
    if len(set(labels)) != len(labels):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"{owner}: duplicate enum labels {duplicates}")
    return tuple(sorted(cleaned, key=lambda pair: pair[0]))


@dataclass(frozen=True)
class Parameter:
    """One telemetry parameter definition.

    Units: ``bit_offset`` and ``bit_width`` in bits; ``scale`` in engineering
    units per raw count; ``offset`` in engineering units; ``low_limit`` and
    ``high_limit`` in engineering units.
    """

    name: str
    path: str = ""
    description: str = ""
    data_type: str = "uint"
    bit_offset: int = 0
    bit_width: int = 8
    byte_order: str = "big"
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    enum: tuple[tuple[int, str], ...] = ()
    low_limit: float | None = None
    high_limit: float | None = None
    packet: str = ""
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        name = _require_str(self.name, "Parameter.name").strip()
        if not name:
            raise ValueError("Parameter.name must be a non-empty, non-blank string")
        object.__setattr__(self, "name", name)
        for text_field in ("path", "description", "unit", "packet"):
            object.__setattr__(
                self, text_field, _require_str(getattr(self, text_field), f"Parameter.{text_field}")
            )

        data_type = _require_str(self.data_type, "Parameter.data_type").lower()
        if data_type not in DATA_TYPES:
            raise ValueError(
                f"Parameter {name!r}: data_type {self.data_type!r} is not one of "
                f"{', '.join(DATA_TYPES)}"
            )
        object.__setattr__(self, "data_type", data_type)

        order_key = _require_str(self.byte_order, "Parameter.byte_order").lower()
        if order_key not in _BYTE_ORDER_ALIASES:
            raise ValueError(
                f"Parameter {name!r}: byte_order {self.byte_order!r} is not one of "
                f"{', '.join(sorted(_BYTE_ORDER_ALIASES))}"
            )
        object.__setattr__(self, "byte_order", _BYTE_ORDER_ALIASES[order_key])

        bit_offset = _require_int(self.bit_offset, "Parameter.bit_offset")
        if bit_offset < 0:
            raise ValueError(f"Parameter {name!r}: bit_offset must be >= 0, got {bit_offset}")
        bit_width = _require_int(self.bit_width, "Parameter.bit_width")
        if bit_width <= 0:
            raise ValueError(f"Parameter {name!r}: bit_width must be > 0, got {bit_width}")
        if bit_width > MAX_BIT_WIDTH:
            raise ValueError(
                f"Parameter {name!r}: bit_width {bit_width} exceeds the supported maximum "
                f"{MAX_BIT_WIDTH}"
            )

        scale = _require_finite_float(self.scale, f"Parameter {name!r}: scale")
        if scale == 0.0:
            raise ValueError(
                f"Parameter {name!r}: scale must be non-zero; a zero scale maps every raw "
                "count onto one engineering value and cannot be inverted"
            )
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self, "offset", _require_finite_float(self.offset, f"Parameter {name!r}: offset")
        )

        object.__setattr__(self, "enum", _normalise_enum(self.enum, owner=f"Parameter {name!r}"))
        if self.enum:
            if data_type in ("float", "string"):
                raise ValueError(
                    f"Parameter {name!r}: data_type {data_type!r} cannot carry an enumeration; "
                    "use data_type 'enum', 'uint', 'int' or 'bool'"
                )
            limit = 1 << bit_width
            for code, label in self.enum:
                if code < 0 and data_type == "uint":
                    raise ValueError(
                        f"Parameter {name!r}: enum code {code} for label {label!r} is negative "
                        "but the data_type is unsigned"
                    )
                if abs(code) >= limit:
                    raise ValueError(
                        f"Parameter {name!r}: enum code {code} for label {label!r} does not fit "
                        f"in bit_width {bit_width}"
                    )

        for limit_name in ("low_limit", "high_limit"):
            value = getattr(self, limit_name)
            if value is not None:
                object.__setattr__(
                    self,
                    limit_name,
                    _require_finite_float(value, f"Parameter {name!r}: {limit_name}"),
                )
        if (
            self.low_limit is not None
            and self.high_limit is not None
            and self.low_limit > self.high_limit
        ):
            raise ValueError(
                f"Parameter {name!r}: low_limit {self.low_limit} is above high_limit "
                f"{self.high_limit}"
            )

        extras: list[tuple[str, str]] = []
        for key, value in self.extra:
            extras.append((_require_str(key, "Parameter.extra key"), str(value)))
        object.__setattr__(self, "extra", tuple(sorted(extras)))

    @property
    def enum_map(self) -> dict[int, str]:
        """The enumeration as a ``{code: label}`` dictionary."""
        return dict(self.enum)

    @property
    def label_map(self) -> dict[str, int]:
        """The enumeration as a ``{label: code}`` dictionary."""
        return {label: code for code, label in self.enum}

    @property
    def extra_map(self) -> dict[str, str]:
        """Unmodelled fields carried through from the source, as strings."""
        return dict(self.extra)

    def engineering(self, raw: float) -> float:
        """Engineering value for ``raw``: ``scale * raw + offset``.

        Units: returns engineering units as declared by :attr:`unit`. Valid for
        the degree-one polynomial calibration this model carries and for no
        other calibration form.
        """
        return self.scale * float(raw) + self.offset

    def structural_signature(self) -> tuple[Any, ...]:
        """The fields a rename candidate is scored on, excluding every name field.

        Deliberately excludes ``name``, ``path`` and ``description``: a rename
        is by definition a change of name, so a matcher that reads the name as
        structure cannot find one.
        """
        return (
            self.data_type,
            self.bit_offset,
            self.bit_width,
            self.byte_order,
            self.scale,
            self.offset,
            self.unit,
            self.enum,
            self.low_limit,
            self.high_limit,
            self.packet,
        )

    def with_fields(self, **changes: object) -> Parameter:
        """A copy with ``changes`` applied, re-validated."""
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CommandArgument:
    """One telecommand argument.

    ``position`` is the zero-based index of the argument in the command's
    encoded argument list. Two commands whose argument names match but whose
    positions differ encode different bits for the same written call, which is
    the command-side analogue of a bit-shifted telemetry parameter.
    """

    name: str
    position: int
    data_type: str = "uint"
    bit_width: int = 8
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    enum: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        name = _require_str(self.name, "CommandArgument.name").strip()
        if not name:
            raise ValueError("CommandArgument.name must be a non-empty, non-blank string")
        object.__setattr__(self, "name", name)
        position = _require_int(self.position, "CommandArgument.position")
        if position < 0:
            raise ValueError(f"CommandArgument {name!r}: position must be >= 0, got {position}")
        data_type = _require_str(self.data_type, "CommandArgument.data_type").lower()
        if data_type not in DATA_TYPES:
            raise ValueError(
                f"CommandArgument {name!r}: data_type {self.data_type!r} is not one of "
                f"{', '.join(DATA_TYPES)}"
            )
        object.__setattr__(self, "data_type", data_type)
        bit_width = _require_int(self.bit_width, "CommandArgument.bit_width")
        if bit_width <= 0:
            raise ValueError(f"CommandArgument {name!r}: bit_width must be > 0, got {bit_width}")
        if bit_width > MAX_BIT_WIDTH:
            raise ValueError(
                f"CommandArgument {name!r}: bit_width {bit_width} exceeds the supported maximum "
                f"{MAX_BIT_WIDTH}"
            )
        object.__setattr__(self, "unit", _require_str(self.unit, "CommandArgument.unit"))
        scale = _require_finite_float(self.scale, f"CommandArgument {name!r}: scale")
        if scale == 0.0:
            raise ValueError(f"CommandArgument {name!r}: scale must be non-zero")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self, "offset", _require_finite_float(self.offset, f"CommandArgument {name!r}: offset")
        )
        object.__setattr__(
            self, "enum", _normalise_enum(self.enum, owner=f"CommandArgument {name!r}")
        )

    def structural_signature(self) -> tuple[Any, ...]:
        """Argument fields excluding the name."""
        return (self.position, self.data_type, self.bit_width, self.unit, self.scale, self.offset)


@dataclass(frozen=True)
class Command:
    """One telecommand definition with its ordered argument list."""

    name: str
    path: str = ""
    description: str = ""
    arguments: tuple[CommandArgument, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        name = _require_str(self.name, "Command.name").strip()
        if not name:
            raise ValueError("Command.name must be a non-empty, non-blank string")
        object.__setattr__(self, "name", name)
        for text_field in ("path", "description"):
            object.__setattr__(
                self, text_field, _require_str(getattr(self, text_field), f"Command.{text_field}")
            )
        arguments = tuple(self.arguments)
        for argument in arguments:
            if not isinstance(argument, CommandArgument):
                raise TypeError(
                    f"Command {name!r}: arguments must be CommandArgument instances, got "
                    f"{type(argument).__name__}"
                )
        argument_names = [argument.name for argument in arguments]
        if len(set(argument_names)) != len(argument_names):
            duplicates = sorted({n for n in argument_names if argument_names.count(n) > 1})
            raise ValueError(f"Command {name!r}: duplicate argument names {duplicates}")
        positions = [argument.position for argument in arguments]
        if len(set(positions)) != len(positions):
            raise ValueError(f"Command {name!r}: duplicate argument positions {sorted(positions)}")
        # Canonical order is by position, so that reordering the declarations in
        # a source file is not a change while changing a position is.
        object.__setattr__(
            self, "arguments", tuple(sorted(arguments, key=lambda argument: argument.position))
        )
        extras = [(_require_str(k, "Command.extra key"), str(v)) for k, v in self.extra]
        object.__setattr__(self, "extra", tuple(sorted(extras)))

    @property
    def argument_map(self) -> dict[str, CommandArgument]:
        """Arguments keyed by name."""
        return {argument.name: argument for argument in self.arguments}

    @property
    def extra_map(self) -> dict[str, str]:
        """Unmodelled fields carried through from the source, as strings."""
        return dict(self.extra)

    def structural_signature(self) -> tuple[Any, ...]:
        """Command fields excluding the command name, including argument names."""
        return tuple(
            (argument.name, *argument.structural_signature()) for argument in self.arguments
        )


@dataclass(frozen=True)
class Dictionary:
    """A versioned telemetry and command dictionary.

    Parameter names are unique, command names are unique, and a parameter name
    may not collide with a command name: a reference extracted from a procedure
    carries no type tag, so a collision would make resolution ambiguous and the
    impact verdict unsound.
    """

    version: str
    parameters: tuple[Parameter, ...] = ()
    commands: tuple[Command, ...] = ()
    source: str = ""
    extra: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_str(self.version, "Dictionary.version"))
        if not self.version.strip():
            raise ValueError("Dictionary.version must be a non-empty, non-blank string")
        object.__setattr__(self, "source", _require_str(self.source, "Dictionary.source"))
        parameters = tuple(self.parameters)
        commands = tuple(self.commands)
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError(
                    "Dictionary.parameters must contain Parameter instances, got "
                    f"{type(parameter).__name__}"
                )
        for command in commands:
            if not isinstance(command, Command):
                raise TypeError(
                    f"Dictionary.commands must contain Command instances, got "
                    f"{type(command).__name__}"
                )
        parameter_names = [parameter.name for parameter in parameters]
        if len(set(parameter_names)) != len(parameter_names):
            duplicates = sorted({n for n in parameter_names if parameter_names.count(n) > 1})
            raise ValueError(f"Dictionary: duplicate parameter names {duplicates}")
        command_names = [command.name for command in commands]
        if len(set(command_names)) != len(command_names):
            duplicates = sorted({n for n in command_names if command_names.count(n) > 1})
            raise ValueError(f"Dictionary: duplicate command names {duplicates}")
        collisions = sorted(set(parameter_names) & set(command_names))
        if collisions:
            raise ValueError(
                f"Dictionary: names used for both a parameter and a command {collisions}; "
                "a reference extracted from a procedure carries no type tag, so this would "
                "make resolution ambiguous"
            )
        object.__setattr__(self, "parameters", tuple(sorted(parameters, key=lambda p: p.name)))
        object.__setattr__(self, "commands", tuple(sorted(commands, key=lambda c: c.name)))
        extras = [(_require_str(k, "Dictionary.extra key"), str(v)) for k, v in self.extra]
        object.__setattr__(self, "extra", tuple(sorted(extras)))

    def __len__(self) -> int:
        return len(self.parameters) + len(self.commands)

    @property
    def parameter_map(self) -> dict[str, Parameter]:
        """Parameters keyed by name."""
        return {parameter.name: parameter for parameter in self.parameters}

    @property
    def command_map(self) -> dict[str, Command]:
        """Commands keyed by name."""
        return {command.name: command for command in self.commands}

    @property
    def path_map(self) -> dict[str, str]:
        """Non-empty item paths mapped to item names, for by-path reference resolution."""
        out: dict[str, str] = {}
        for item in (*self.parameters, *self.commands):
            if item.path:
                out[item.path] = item.name
        return out

    @property
    def item_names(self) -> tuple[str, ...]:
        """Every parameter and command name, sorted."""
        return tuple(sorted([*self.parameter_map, *self.command_map]))

    @property
    def enum_labels(self) -> dict[str, tuple[str, int]]:
        """Every enumeration label mapped to ``(parameter name, code)``.

        A label used by more than one parameter is reported under the first
        parameter in name order, and the ambiguity is counted by
        :func:`tmtcdiff.impact.assess_impact` rather than resolved here.
        """
        out: dict[str, tuple[str, int]] = {}
        for parameter in self.parameters:
            for code, label in parameter.enum:
                out.setdefault(label, (parameter.name, code))
        return out

    def resolve(self, reference: str) -> str | None:
        """Resolve a reference by name or by path. Returns the item name or ``None``."""
        if reference in self.parameter_map or reference in self.command_map:
            return reference
        return self.path_map.get(reference)
