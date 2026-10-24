"""YAML in, :class:`~cmdguard.model.Case` out, with declaration errors named.

A loader that accepts a malformed declaration and checks the sequence against
it produces a clean report from a broken input, which is worse than a crash.
Every structural error raises :class:`ValueError` naming the key and what was
expected. The shipped cases in ``examples/cases/`` are the format's reference,
and ``examples/cases/missing_declarations.yaml`` is a deliberately broken one
used by the tests and by ``validation/validate_cli.py``.

Paths are resolved by the caller. Nothing in this module reads a path relative
to the process working directory: :func:`load_case` takes a path, and the
examples and validation scripts anchor theirs on ``__file__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

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
    parse_verity,
)

__all__ = ["load_case", "case_from_mapping", "case_to_mapping", "dump_case"]

_COMMAND_KEYS = frozenset(
    {
        "id",
        "subsystem",
        "required_modes",
        "preconditions",
        "effects",
        "mode_effect",
        "resource_draw",
        "hold_s",
        "min_interval_s",
        "min_gap_after_s",
        "irreversible",
        "exclusion_group",
    }
)

_TOP_KEYS = frozenset(
    {"name", "description", "dictionary", "mode_machine", "initial_state", "sequence"}
)


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: required key {key!r} is missing")
    return mapping[key]


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a mapping, got {type(value).__name__}")
    return value


def _as_float(value: Any, context: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a number, got {value!r}")
    return float(value)


def _as_str_tuple(value: Any, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{context}: expected a list of strings, got the string {value!r}")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context}: expected a list of strings, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{context}: list entry {item!r} is not a string")
        out.append(item)
    return tuple(out)


def _command_from_mapping(raw: Any, index: int) -> CommandSpec:
    context = f"dictionary.commands[{index}]"
    mapping = _as_mapping(raw, context)
    unknown = sorted(set(mapping) - _COMMAND_KEYS)
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {', '.join(unknown)}; a misspelt key is silently "
            f"ignored by a permissive loader and that is how a declaration goes unchecked"
        )
    identifier = _require(mapping, "id", context)
    if not isinstance(identifier, str):
        raise ValueError(f"{context}: 'id' must be a string, got {identifier!r}")
    preconditions = {
        str(k): parse_verity(v)
        for k, v in sorted(
            _as_mapping(mapping.get("preconditions"), f"{context}.preconditions").items()
        )
    }
    effects = {
        str(k): parse_verity(v)
        for k, v in sorted(_as_mapping(mapping.get("effects"), f"{context}.effects").items())
    }
    draw_raw = _as_mapping(mapping.get("resource_draw"), f"{context}.resource_draw")
    resource_draw = {
        str(k): _as_float(v, f"{context}.resource_draw[{k!r}]")
        for k, v in sorted(draw_raw.items())
    }
    exclusion = mapping.get("exclusion_group")
    if exclusion is not None and not isinstance(exclusion, str):
        raise ValueError(f"{context}: 'exclusion_group' must be a string or absent")
    mode_effect = mapping.get("mode_effect")
    if mode_effect is not None and not isinstance(mode_effect, str):
        raise ValueError(f"{context}: 'mode_effect' must be a string or absent")
    irreversible = mapping.get("irreversible", False)
    if not isinstance(irreversible, bool):
        raise ValueError(f"{context}: 'irreversible' must be true or false, got {irreversible!r}")
    subsystem = mapping.get("subsystem", "unassigned")
    if not isinstance(subsystem, str):
        raise ValueError(f"{context}: 'subsystem' must be a string")
    return CommandSpec(
        identifier=identifier,
        subsystem=subsystem,
        required_modes=_as_str_tuple(mapping.get("required_modes"), f"{context}.required_modes"),
        preconditions=preconditions,
        effects=effects,
        mode_effect=mode_effect,
        resource_draw=resource_draw,
        hold_s=_as_float(mapping.get("hold_s"), f"{context}.hold_s"),
        min_interval_s=_as_float(mapping.get("min_interval_s"), f"{context}.min_interval_s"),
        min_gap_after_s=_as_float(mapping.get("min_gap_after_s"), f"{context}.min_gap_after_s"),
        irreversible=irreversible,
        exclusion_group=exclusion,
    )


def case_from_mapping(raw: Any, name_default: str = "unnamed") -> Case:
    """Build a :class:`~cmdguard.model.Case` from a parsed YAML mapping."""
    mapping = _as_mapping(raw, "case")
    unknown = sorted(set(mapping) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"case: unknown top-level key(s) {', '.join(unknown)}")

    name = mapping.get("name", name_default)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("case: 'name' must be a non-empty string")

    dict_raw = _as_mapping(_require(mapping, "dictionary", "case"), "case.dictionary")
    commands_raw = dict_raw.get("commands")
    if not isinstance(commands_raw, (list, tuple)) or not commands_raw:
        raise ValueError("case.dictionary.commands: expected a non-empty list of commands")
    specs: dict[str, CommandSpec] = {}
    for index, entry in enumerate(commands_raw):
        spec = _command_from_mapping(entry, index)
        if spec.identifier in specs:
            raise ValueError(
                f"case.dictionary.commands[{index}]: command {spec.identifier!r} is declared twice"
            )
        specs[spec.identifier] = spec
    limits_raw = _as_mapping(dict_raw.get("resource_limits"), "case.dictionary.resource_limits")
    limits = ResourceLimits(
        {
            str(k): _as_float(v, f"case.dictionary.resource_limits[{k!r}]")
            for k, v in sorted(limits_raw.items())
        }
    )
    dictionary = CommandDictionary(commands=specs, resource_limits=limits)

    machine_raw = _as_mapping(_require(mapping, "mode_machine", "case"), "case.mode_machine")
    modes = _as_str_tuple(
        _require(machine_raw, "modes", "case.mode_machine"), "case.mode_machine.modes"
    )
    transitions_raw = _as_mapping(machine_raw.get("transitions"), "case.mode_machine.transitions")
    transitions = {
        str(k): _as_str_tuple(v, f"case.mode_machine.transitions[{k!r}]")
        for k, v in sorted(transitions_raw.items())
    }
    machine = ModeMachine(modes=modes, transitions=transitions)

    initial_raw = _as_mapping(_require(mapping, "initial_state", "case"), "case.initial_state")
    initial_mode = _require(initial_raw, "mode", "case.initial_state")
    if not isinstance(initial_mode, str):
        raise ValueError("case.initial_state.mode: expected a string")
    predicates_raw = _as_mapping(initial_raw.get("predicates"), "case.initial_state.predicates")
    initial = InitialState(
        mode=initial_mode,
        predicates={str(k): parse_verity(v) for k, v in sorted(predicates_raw.items())},
    )

    sequence_raw = _require(mapping, "sequence", "case")
    if not isinstance(sequence_raw, (list, tuple)):
        raise ValueError("case.sequence: expected a list of steps")
    steps: list[SequenceStep] = []
    for index, entry in enumerate(sequence_raw):
        context = f"case.sequence[{index}]"
        step_map = _as_mapping(entry, context)
        unknown_step = sorted(set(step_map) - {"t_s", "command", "note"})
        if unknown_step:
            raise ValueError(f"{context}: unknown key(s) {', '.join(unknown_step)}")
        command = _require(step_map, "command", context)
        if not isinstance(command, str):
            raise ValueError(f"{context}.command: expected a string")
        note = step_map.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"{context}.note: expected a string")
        steps.append(
            SequenceStep(
                t_s=_as_float(_require(step_map, "t_s", context), f"{context}.t_s"),
                command=command,
                note=note,
            )
        )

    description = mapping.get("description", "")
    if not isinstance(description, str):
        raise ValueError("case.description: expected a string")

    return Case(
        name=name,
        dictionary=dictionary,
        machine=machine,
        initial=initial,
        sequence=CommandSequence(tuple(steps)),
        description=description,
    )


def load_case(path: str | Path) -> Case:
    """Load one case from a YAML file.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    ValueError
        On any structural or semantic error in the declaration, naming the key.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"case file not found: {target}")
    raw = yaml.safe_load(target.read_text())
    if raw is None:
        raise ValueError(f"case file is empty: {target}")
    return case_from_mapping(raw, name_default=target.stem)


def _verity_out(value: Verity) -> str:
    return value.value


def case_to_mapping(case: Case) -> dict[str, Any]:
    """Render ``case`` back to a plain mapping, round-tripping through YAML.

    Key order is fixed and every mapping is sorted, so that
    ``validation/regenerate_cases.py`` produces byte-identical files on re-run.
    """
    commands: list[dict[str, Any]] = []
    for identifier in case.dictionary.identifiers:
        spec = case.dictionary.commands[identifier]
        entry: dict[str, Any] = {"id": spec.identifier, "subsystem": spec.subsystem}
        if spec.required_modes:
            entry["required_modes"] = list(spec.required_modes)
        if spec.preconditions:
            entry["preconditions"] = {
                k: _verity_out(spec.preconditions[k]) for k in sorted(spec.preconditions)
            }
        if spec.effects:
            entry["effects"] = {k: _verity_out(spec.effects[k]) for k in sorted(spec.effects)}
        if spec.mode_effect is not None:
            entry["mode_effect"] = spec.mode_effect
        if spec.resource_draw:
            entry["resource_draw"] = {
                k: float(spec.resource_draw[k]) for k in sorted(spec.resource_draw)
            }
        for key, value in (
            ("hold_s", spec.hold_s),
            ("min_interval_s", spec.min_interval_s),
            ("min_gap_after_s", spec.min_gap_after_s),
        ):
            if value:
                entry[key] = float(value)
        if spec.irreversible:
            entry["irreversible"] = True
        if spec.exclusion_group is not None:
            entry["exclusion_group"] = spec.exclusion_group
        commands.append(entry)

    out: dict[str, Any] = {"name": case.name}
    if case.description:
        out["description"] = case.description
    out["dictionary"] = {
        "resource_limits": {
            k: float(case.dictionary.resource_limits.limits[k])
            for k in sorted(case.dictionary.resource_limits.limits)
        },
        "commands": commands,
    }
    out["mode_machine"] = {
        "modes": list(case.machine.modes),
        "transitions": {
            k: list(case.machine.transitions[k]) for k in sorted(case.machine.transitions)
        },
    }
    out["initial_state"] = {
        "mode": case.initial.mode,
        "predicates": {
            k: _verity_out(case.initial.predicates[k]) for k in sorted(case.initial.predicates)
        },
    }
    out["sequence"] = [
        {"t_s": float(s.t_s), "command": s.command, **({"note": s.note} if s.note else {})}
        for s in case.sequence.steps
    ]
    return out


def dump_case(case: Case) -> str:
    """YAML text for ``case``, deterministic and round-trippable."""
    return yaml.safe_dump(case_to_mapping(case), sort_keys=False, default_flow_style=False)
