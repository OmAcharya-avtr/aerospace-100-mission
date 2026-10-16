"""Strict YAML parser for a watchdog specification.

Every rejection names the key path that caused it. Nothing is inferred and no
default is invented for a quantity the arithmetic depends on: a missing
``heartbeat_period_ms`` is an error, not a zero.

Document shape
--------------
.. code-block:: yaml

    spec_version: 1
    system: "flight computer supervision"
    notes: "optional free text"
    supervisor:
      poll_period_ms: 10.0
      poll_jitter_ms: 2.0             # optional, default 0
      declaration_overhead_ms: 1.0    # optional, default 0
      name: "fc-supervisor"           # optional
    ladders:
      - id: LADDER-STD
        description: "optional"
        rungs:
          - action: log               # log | notify | restart_task | safe_mode | reset
            latency_ms: 2.0
            recovery_cost: 0.0        # optional, default 0
            note: "optional"
    tasks:
      - id: TASK-GNC
        name: "optional"
        heartbeat_period_ms: 20.0
        heartbeat_jitter_ms: 3.0      # optional, default 0
        jitter_sigma_ms: 0.8          # optional, default 0
        missed_beats_tolerated: 2
        liveness_deadline_ms: 500.0
        escalation: LADDER-STD
        false_trip_budget_per_hour: 0.001   # optional, omit to disable WS005
        note: "optional"

``spec_version`` must be ``1``. Unknown keys are rejected rather than ignored,
because a typo in a key the arithmetic reads is otherwise a silent default.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .errors import SpecFormatError
from .model import Action, Ladder, Rung, Spec, Supervisor, Task

SPEC_VERSION = 1

_SUPERVISOR_KEYS = {"poll_period_ms", "poll_jitter_ms", "declaration_overhead_ms", "name"}
_RUNG_KEYS = {"action", "latency_ms", "recovery_cost", "note"}
_LADDER_KEYS = {"id", "rungs", "description"}
_TASK_KEYS = {
    "id",
    "name",
    "heartbeat_period_ms",
    "heartbeat_jitter_ms",
    "jitter_sigma_ms",
    "missed_beats_tolerated",
    "liveness_deadline_ms",
    "escalation",
    "false_trip_budget_per_hour",
    "note",
}
_TOP_KEYS = {"spec_version", "system", "notes", "supervisor", "ladders", "tasks"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecFormatError(f"expected a mapping, found {type(value).__name__}", path)
    for key in value:
        if not isinstance(key, str):
            raise SpecFormatError(f"mapping key {key!r} is not a string", path)
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SpecFormatError(
            f"unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}", path
        )


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise SpecFormatError(f"required key '{key}' is missing", path)
    return mapping[key]


def _number(value: Any, path: str, *, minimum: float | None, allow_equal: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecFormatError(
            f"expected a number, found {type(value).__name__}", path
        )
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        raise SpecFormatError(f"expected a finite number, found {value!r}", path)
    if minimum is not None:
        if allow_equal and out < minimum:
            raise SpecFormatError(f"must be >= {minimum:g}, found {out:g}", path)
        if not allow_equal and out <= minimum:
            raise SpecFormatError(f"must be > {minimum:g}, found {out:g}", path)
    return out


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecFormatError(f"expected an integer, found {type(value).__name__}", path)
    if value < minimum:
        raise SpecFormatError(f"must be >= {minimum}, found {value}", path)
    return value


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SpecFormatError(f"expected a string, found {type(value).__name__}", path)
    if not allow_empty and not value.strip():
        raise SpecFormatError("must not be empty", path)
    return value


def _sequence(value: Any, path: str, *, minimum: int) -> list[Any]:
    if not isinstance(value, list):
        raise SpecFormatError(f"expected a list, found {type(value).__name__}", path)
    if len(value) < minimum:
        raise SpecFormatError(f"must hold at least {minimum} item(s), found {len(value)}", path)
    return value


def _parse_supervisor(raw: Any, path: str) -> Supervisor:
    mapping = _mapping(raw, path)
    _reject_unknown(mapping, _SUPERVISOR_KEYS, path)
    poll = _number(
        _require(mapping, "poll_period_ms", path),
        f"{path}.poll_period_ms",
        minimum=0.0,
        allow_equal=False,
    )
    jitter = _number(
        mapping.get("poll_jitter_ms", 0.0),
        f"{path}.poll_jitter_ms",
        minimum=0.0,
        allow_equal=True,
    )
    overhead = _number(
        mapping.get("declaration_overhead_ms", 0.0),
        f"{path}.declaration_overhead_ms",
        minimum=0.0,
        allow_equal=True,
    )
    name = _text(mapping.get("name", "supervisor"), f"{path}.name")
    return Supervisor(
        poll_period_ms=poll,
        poll_jitter_ms=jitter,
        declaration_overhead_ms=overhead,
        name=name,
    )


def _parse_rung(raw: Any, path: str) -> Rung:
    mapping = _mapping(raw, path)
    _reject_unknown(mapping, _RUNG_KEYS, path)
    action_text = _text(_require(mapping, "action", path), f"{path}.action")
    try:
        action = Action(action_text)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in Action)
        raise SpecFormatError(
            f"unknown action '{action_text}'; allowed: {allowed}", f"{path}.action"
        ) from exc
    latency = _number(
        _require(mapping, "latency_ms", path),
        f"{path}.latency_ms",
        minimum=0.0,
        allow_equal=True,
    )
    cost = _number(
        mapping.get("recovery_cost", 0.0),
        f"{path}.recovery_cost",
        minimum=0.0,
        allow_equal=True,
    )
    note = _text(mapping.get("note", ""), f"{path}.note", allow_empty=True)
    return Rung(action=action, latency_ms=latency, recovery_cost=cost, note=note)


def _parse_ladder(raw: Any, path: str) -> Ladder:
    mapping = _mapping(raw, path)
    _reject_unknown(mapping, _LADDER_KEYS, path)
    ladder_id = _text(_require(mapping, "id", path), f"{path}.id")
    rungs_raw = _sequence(_require(mapping, "rungs", path), f"{path}.rungs", minimum=1)
    rungs = tuple(
        _parse_rung(item, f"{path}.rungs[{index}]") for index, item in enumerate(rungs_raw)
    )
    description = _text(mapping.get("description", ""), f"{path}.description", allow_empty=True)
    return Ladder(id=ladder_id, rungs=rungs, description=description)


def _parse_task(raw: Any, path: str) -> Task:
    mapping = _mapping(raw, path)
    _reject_unknown(mapping, _TASK_KEYS, path)
    task_id = _text(_require(mapping, "id", path), f"{path}.id")
    period = _number(
        _require(mapping, "heartbeat_period_ms", path),
        f"{path}.heartbeat_period_ms",
        minimum=0.0,
        allow_equal=False,
    )
    jitter = _number(
        mapping.get("heartbeat_jitter_ms", 0.0),
        f"{path}.heartbeat_jitter_ms",
        minimum=0.0,
        allow_equal=True,
    )
    sigma = _number(
        mapping.get("jitter_sigma_ms", 0.0),
        f"{path}.jitter_sigma_ms",
        minimum=0.0,
        allow_equal=True,
    )
    tolerance = _integer(
        _require(mapping, "missed_beats_tolerated", path),
        f"{path}.missed_beats_tolerated",
        minimum=0,
    )
    deadline = _number(
        _require(mapping, "liveness_deadline_ms", path),
        f"{path}.liveness_deadline_ms",
        minimum=0.0,
        allow_equal=False,
    )
    escalation = _text(_require(mapping, "escalation", path), f"{path}.escalation")
    budget_raw = mapping.get("false_trip_budget_per_hour")
    budget = (
        None
        if budget_raw is None
        else _number(
            budget_raw, f"{path}.false_trip_budget_per_hour", minimum=0.0, allow_equal=True
        )
    )
    return Task(
        id=task_id,
        heartbeat_period_ms=period,
        missed_beats_tolerated=tolerance,
        liveness_deadline_ms=deadline,
        escalation=escalation,
        heartbeat_jitter_ms=jitter,
        jitter_sigma_ms=sigma,
        false_trip_budget_per_hour=budget,
        name=_text(mapping.get("name", ""), f"{path}.name", allow_empty=True),
        note=_text(mapping.get("note", ""), f"{path}.note", allow_empty=True),
    )


def parse_spec(text: str, source: str = "<string>") -> Spec:
    """Parse a specification document.

    Parameters
    ----------
    text:
        The YAML document.
    source:
        Name recorded on the :class:`watchdogspec.model.Spec` and printed in
        reports. Normally the file path.

    Raises
    ------
    SpecFormatError
        On malformed YAML, a missing or misspelled key, a value of the wrong
        type, a value outside its validity range, or a duplicate identifier.
        The exception's ``path`` attribute names the offending key.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecFormatError(f"document is not valid YAML: {exc}") from exc
    if raw is None:
        raise SpecFormatError("document is empty")
    document = _mapping(raw, "<document>")
    _reject_unknown(document, _TOP_KEYS, "<document>")

    version = _integer(_require(document, "spec_version", "<document>"), "spec_version", minimum=1)
    if version != SPEC_VERSION:
        raise SpecFormatError(
            f"spec_version {version} is not supported by this release (expected "
            f"{SPEC_VERSION})",
            "spec_version",
        )
    system = _text(_require(document, "system", "<document>"), "system")
    notes = _text(document.get("notes", ""), "notes", allow_empty=True)
    supervisor = _parse_supervisor(_require(document, "supervisor", "<document>"), "supervisor")

    ladders_raw = _sequence(_require(document, "ladders", "<document>"), "ladders", minimum=1)
    ladders = tuple(
        _parse_ladder(item, f"ladders[{index}]") for index, item in enumerate(ladders_raw)
    )
    seen: set[str] = set()
    for index, ladder in enumerate(ladders):
        if ladder.id in seen:
            raise SpecFormatError(f"duplicate ladder id '{ladder.id}'", f"ladders[{index}].id")
        seen.add(ladder.id)

    tasks_raw = _sequence(_require(document, "tasks", "<document>"), "tasks", minimum=1)
    tasks = tuple(_parse_task(item, f"tasks[{index}]") for index, item in enumerate(tasks_raw))
    seen = set()
    for index, task in enumerate(tasks):
        if task.id in seen:
            raise SpecFormatError(f"duplicate task id '{task.id}'", f"tasks[{index}].id")
        seen.add(task.id)

    return Spec(
        system=system,
        supervisor=supervisor,
        ladders=ladders,
        tasks=tasks,
        notes=notes,
        source=source,
        spec_version=version,
    )


def load_spec(path: str | Path) -> Spec:
    """Parse a specification from a UTF-8 file.

    Raises
    ------
    SpecFormatError
        If the file does not exist, is a directory, or is not valid UTF-8, as
        well as for every parse failure of :func:`parse_spec`.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpecFormatError(f"no such specification file: {target}") from exc
    except IsADirectoryError as exc:
        raise SpecFormatError(f"specification path is a directory: {target}") from exc
    except UnicodeDecodeError as exc:
        raise SpecFormatError(f"specification file is not valid UTF-8: {target}") from exc
    return parse_spec(text, source=str(target))
