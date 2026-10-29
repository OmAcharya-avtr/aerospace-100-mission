"""Read and write dictionary definitions as JSON, tolerantly and reversibly.

Two decisions in here matter for what the rest of the package can claim.

**Unmodelled fields are carried, not rejected.** A real TM/TC database has
dozens of columns this model does not represent. Rejecting an unknown key
would force a user to strip their own data before diffing it, and silently
dropping it would let a semantic change hide in a field the tool never looked
at. Instead every unknown item key is carried through as an opaque string in
``Parameter.extra``, and the diff reports those fields as *incomparable* —
present, differing, and of unknown meaning to this tool. That report is part
of the coverage verdict, not a footnote.

**Key order and entry order are not semantic.** Both are discarded on read:
items are sorted by name, enumerations by code, command arguments by position.
This is what makes the negative control in :mod:`tmtcdiff.synth` meaningful —
a file whose text has been reordered substantially must produce an empty
semantic diff, and it can only do so if ordering is dropped at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .model import Command, CommandArgument, Dictionary, Parameter

#: Accepted aliases for parameter keys. The canonical name is the value.
PARAMETER_ALIASES: dict[str, str] = {
    "mnemonic": "name",
    "id": "name",
    "type": "data_type",
    "datatype": "data_type",
    "bits": "bit_width",
    "width": "bit_width",
    "offset_bits": "bit_offset",
    "bitoffset": "bit_offset",
    "endianness": "byte_order",
    "byteorder": "byte_order",
    "factor": "scale",
    "gain": "scale",
    "bias": "offset",
    "units": "unit",
    "enumeration": "enum",
    "states": "enum",
    "low": "low_limit",
    "high": "high_limit",
    "min": "low_limit",
    "max": "high_limit",
    "apid": "packet",
    "container": "packet",
    "desc": "description",
}

#: Accepted aliases for command keys.
COMMAND_ALIASES: dict[str, str] = {
    "mnemonic": "name",
    "id": "name",
    "desc": "description",
    "args": "arguments",
    "parameters": "arguments",
}

#: Accepted aliases for command-argument keys.
ARGUMENT_ALIASES: dict[str, str] = {
    "mnemonic": "name",
    "type": "data_type",
    "datatype": "data_type",
    "bits": "bit_width",
    "width": "bit_width",
    "index": "position",
    "order": "position",
    "units": "unit",
    "factor": "scale",
    "gain": "scale",
    "bias": "offset",
    "enumeration": "enum",
    "states": "enum",
}

_PARAMETER_FIELDS = frozenset(
    {
        "name",
        "path",
        "description",
        "data_type",
        "bit_offset",
        "bit_width",
        "byte_order",
        "scale",
        "offset",
        "unit",
        "enum",
        "low_limit",
        "high_limit",
        "packet",
    }
)

_COMMAND_FIELDS = frozenset({"name", "path", "description", "arguments"})

_ARGUMENT_FIELDS = frozenset(
    {"name", "position", "data_type", "bit_width", "unit", "scale", "offset", "enum"}
)


class DictionaryFormatError(ValueError):
    """Raised when a dictionary document cannot be read as a dictionary.

    A subclass of ``ValueError``, so callers that only care that the input was
    invalid need no import.
    """


def _canonical_keys(
    raw: Mapping[str, Any], aliases: Mapping[str, str], known: frozenset[str], label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Split ``raw`` into canonical modelled fields and opaque extras."""
    fields: dict[str, Any] = {}
    extras: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise DictionaryFormatError(f"{label}: key {key!r} is not a string")
        canonical = aliases.get(key.lower(), key.lower())
        if canonical in known:
            if canonical in fields:
                raise DictionaryFormatError(
                    f"{label}: key {key!r} and an alias both map to {canonical!r}"
                )
            fields[canonical] = value
        else:
            extras[key] = _opaque(value)
    return fields, extras


def _opaque(value: Any) -> str:
    """Render an unmodelled value as a stable string for comparison only."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _enum_from_json(value: Any, label: str) -> tuple[tuple[int, str], ...]:
    """Read an enumeration from a mapping with string keys or a sequence of pairs."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs: list[tuple[int, str]] = []
        for code, name in value.items():
            try:
                pairs.append((int(code), name))
            except (TypeError, ValueError) as error:
                raise DictionaryFormatError(
                    f"{label}: enum key {code!r} is not an integer code"
                ) from error
        return tuple(pairs)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = []
        for entry in value:
            if isinstance(entry, Mapping):
                if "code" not in entry or "label" not in entry:
                    raise DictionaryFormatError(
                        f"{label}: enum entry {entry!r} needs both 'code' and 'label'"
                    )
                pairs.append((int(entry["code"]), entry["label"]))
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                if len(entry) != 2:
                    raise DictionaryFormatError(
                        f"{label}: enum entry {entry!r} is not a (code, label) pair"
                    )
                pairs.append((int(entry[0]), entry[1]))
            else:
                raise DictionaryFormatError(f"{label}: enum entry {entry!r} is not readable")
        return tuple(pairs)
    raise DictionaryFormatError(f"{label}: enum must be a mapping or a sequence of pairs")


def parameter_from_mapping(raw: Mapping[str, Any], *, strict: bool = False) -> Parameter:
    """Build a :class:`~tmtcdiff.model.Parameter` from a mapping.

    With ``strict=True`` an unmodelled key raises instead of being carried as
    an extra. The default is tolerant: see the module docstring.
    """
    if not isinstance(raw, Mapping):
        raise DictionaryFormatError(f"parameter entry must be a mapping, got {type(raw).__name__}")
    label = f"parameter {raw.get('name', raw.get('mnemonic', '<unnamed>'))!r}"
    fields, extras = _canonical_keys(raw, PARAMETER_ALIASES, _PARAMETER_FIELDS, label)
    if "name" not in fields:
        raise DictionaryFormatError(f"{label}: no 'name' or 'mnemonic' key")
    if strict and extras:
        raise DictionaryFormatError(f"{label}: unmodelled keys {sorted(extras)}")
    if "enum" in fields:
        fields["enum"] = _enum_from_json(fields["enum"], label)
    try:
        return Parameter(extra=tuple(sorted(extras.items())), **fields)
    except (TypeError, ValueError) as error:
        raise DictionaryFormatError(f"{label}: {error}") from error


def argument_from_mapping(
    raw: Mapping[str, Any], *, position: int, strict: bool = False
) -> CommandArgument:
    """Build a :class:`~tmtcdiff.model.CommandArgument`, defaulting ``position`` to file order."""
    if not isinstance(raw, Mapping):
        raise DictionaryFormatError(f"argument entry must be a mapping, got {type(raw).__name__}")
    label = f"argument {raw.get('name', '<unnamed>')!r}"
    fields, extras = _canonical_keys(raw, ARGUMENT_ALIASES, _ARGUMENT_FIELDS, label)
    if "name" not in fields:
        raise DictionaryFormatError(f"{label}: no 'name' key")
    if strict and extras:
        raise DictionaryFormatError(f"{label}: unmodelled keys {sorted(extras)}")
    fields.setdefault("position", position)
    if "enum" in fields:
        fields["enum"] = _enum_from_json(fields["enum"], label)
    try:
        return CommandArgument(**fields)
    except (TypeError, ValueError) as error:
        raise DictionaryFormatError(f"{label}: {error}") from error


def command_from_mapping(raw: Mapping[str, Any], *, strict: bool = False) -> Command:
    """Build a :class:`~tmtcdiff.model.Command` from a mapping."""
    if not isinstance(raw, Mapping):
        raise DictionaryFormatError(f"command entry must be a mapping, got {type(raw).__name__}")
    label = f"command {raw.get('name', raw.get('mnemonic', '<unnamed>'))!r}"
    fields, extras = _canonical_keys(raw, COMMAND_ALIASES, _COMMAND_FIELDS, label)
    if "name" not in fields:
        raise DictionaryFormatError(f"{label}: no 'name' or 'mnemonic' key")
    if strict and extras:
        raise DictionaryFormatError(f"{label}: unmodelled keys {sorted(extras)}")
    raw_arguments = fields.pop("arguments", ())
    if raw_arguments is None:
        raw_arguments = ()
    if not isinstance(raw_arguments, Sequence) or isinstance(raw_arguments, (str, bytes)):
        raise DictionaryFormatError(f"{label}: 'arguments' must be a list")
    arguments = tuple(
        argument_from_mapping(entry, position=index, strict=strict)
        for index, entry in enumerate(raw_arguments)
    )
    try:
        return Command(arguments=arguments, extra=tuple(sorted(extras.items())), **fields)
    except (TypeError, ValueError) as error:
        raise DictionaryFormatError(f"{label}: {error}") from error


def dictionary_from_mapping(
    raw: Mapping[str, Any], *, strict: bool = False, source: str = ""
) -> Dictionary:
    """Build a :class:`~tmtcdiff.model.Dictionary` from a mapping.

    Required keys: ``version``. Optional: ``parameters``, ``commands``. Unknown
    top-level keys are carried as extras unless ``strict``.
    """
    if not isinstance(raw, Mapping):
        raise DictionaryFormatError(f"document must be a mapping, got {type(raw).__name__}")
    known = frozenset({"version", "parameters", "commands", "source"})
    fields: dict[str, Any] = {}
    extras: dict[str, str] = {}
    for key, value in raw.items():
        if key in known:
            fields[key] = value
        else:
            extras[str(key)] = _opaque(value)
    if "version" not in fields:
        raise DictionaryFormatError("document has no 'version' key")
    if strict and extras:
        raise DictionaryFormatError(f"document: unmodelled top-level keys {sorted(extras)}")
    raw_parameters = fields.get("parameters") or ()
    raw_commands = fields.get("commands") or ()
    for name, value in (("parameters", raw_parameters), ("commands", raw_commands)):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise DictionaryFormatError(f"document: {name!r} must be a list")
    parameters = tuple(
        parameter_from_mapping(entry, strict=strict) for entry in raw_parameters
    )
    commands = tuple(command_from_mapping(entry, strict=strict) for entry in raw_commands)
    try:
        return Dictionary(
            version=str(fields["version"]),
            parameters=parameters,
            commands=commands,
            source=source or str(fields.get("source", "")),
            extra=tuple(sorted(extras.items())),
        )
    except (TypeError, ValueError) as error:
        raise DictionaryFormatError(f"document: {error}") from error


def loads(text: str, *, strict: bool = False, source: str = "") -> Dictionary:
    """Read a dictionary from JSON text."""
    if not isinstance(text, str):
        raise TypeError(f"loads expects str, got {type(text).__name__}")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise DictionaryFormatError(f"not valid JSON: {error}") from error
    return dictionary_from_mapping(raw, strict=strict, source=source)


def load_dictionary(path: str | Path, *, strict: bool = False) -> Dictionary:
    """Read a dictionary from a JSON file.

    The error message quotes the path as given, so a caller that passes a
    relative path gets a relative path back. Nothing in this package passes an
    absolute path to this function.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise DictionaryFormatError(f"cannot read {target.as_posix()}: {error.strerror}") from error
    return loads(text, strict=strict, source=target.name)


def parameter_to_mapping(parameter: Parameter) -> dict[str, Any]:
    """Render a parameter as a JSON-ready mapping, extras included."""
    out: dict[str, Any] = {
        "name": parameter.name,
        "path": parameter.path,
        "description": parameter.description,
        "data_type": parameter.data_type,
        "bit_offset": parameter.bit_offset,
        "bit_width": parameter.bit_width,
        "byte_order": parameter.byte_order,
        "scale": parameter.scale,
        "offset": parameter.offset,
        "unit": parameter.unit,
        "packet": parameter.packet,
    }
    if parameter.enum:
        out["enum"] = {str(code): label for code, label in parameter.enum}
    if parameter.low_limit is not None:
        out["low_limit"] = parameter.low_limit
    if parameter.high_limit is not None:
        out["high_limit"] = parameter.high_limit
    out.update(parameter.extra_map)
    return out


def command_to_mapping(command: Command) -> dict[str, Any]:
    """Render a command as a JSON-ready mapping, extras included."""
    out: dict[str, Any] = {
        "name": command.name,
        "path": command.path,
        "description": command.description,
        "arguments": [
            {
                "name": argument.name,
                "position": argument.position,
                "data_type": argument.data_type,
                "bit_width": argument.bit_width,
                "unit": argument.unit,
                "scale": argument.scale,
                "offset": argument.offset,
                **(
                    {"enum": {str(code): label for code, label in argument.enum}}
                    if argument.enum
                    else {}
                ),
            }
            for argument in command.arguments
        ],
    }
    out.update(command.extra_map)
    return out


def dictionary_to_mapping(dictionary: Dictionary) -> dict[str, Any]:
    """Render a dictionary as a JSON-ready mapping."""
    out: dict[str, Any] = {
        "version": dictionary.version,
        "parameters": [parameter_to_mapping(p) for p in dictionary.parameters],
        "commands": [command_to_mapping(c) for c in dictionary.commands],
    }
    out.update(dict(dictionary.extra))
    return out


def dumps(dictionary: Dictionary, *, indent: int = 2) -> str:
    """Render a dictionary as JSON text in canonical order."""
    return json.dumps(dictionary_to_mapping(dictionary), indent=indent, sort_keys=False) + "\n"


def round_trip(dictionary: Dictionary) -> Dictionary:
    """Serialise and re-read ``dictionary``. Used as an identity check in the suite."""
    return loads(dumps(dictionary), source=dictionary.source)
