"""Read an STPA analysis from YAML, strictly.

Design decisions worth knowing before you write a document:

* **Unknown keys are rejected, not ignored.** A misspelt ``mitigations`` key
  silently drops every mitigation, and the checks would then report the
  document as having none. Rejecting the key is the only safe behaviour for a
  tool whose output is a gap list.
* **Dangling references are rejected at parse time.** A UCA naming a control
  action that does not exist is a malformed document, not a finding; the
  checks downstream are then free to assume every reference resolves.
* **An empty link list is accepted.** A hazard with no loss, a constraint with
  no requirement and a mitigation with no evidence are all well-formed
  documents with gaps in them, and reporting those gaps is the whole point of
  the package.
* Every rejection carries the dotted key path and, where the YAML parser
  supplied one, the one-based line number.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .errors import AnalysisFormatError
from .model import (
    Analysis,
    Component,
    ComponentKind,
    ControlAction,
    ControlStructure,
    Evidence,
    EvidenceKind,
    Exclusion,
    FeedbackPath,
    Hazard,
    Loss,
    LossScenario,
    Mitigation,
    SystemConstraint,
    Uca,
    UcaClass,
)

#: Key used internally to carry the source line of a mapping. It is stripped
#: before any mapping is handed to a model constructor.
LINE_KEY = "__line__"

_TOP_KEYS = frozenset(
    {
        "meta",
        "losses",
        "hazards",
        "constraints",
        "control_structure",
        "ucas",
        "exclusions",
        "loss_scenarios",
        "mitigations",
        "evidence",
    }
)
_META_KEYS = frozenset({"title", "system", "revision", "notes"})
_STRUCTURE_KEYS = frozenset(
    {
        "controllers",
        "actuators",
        "sensors",
        "controlled_processes",
        "control_actions",
        "feedback",
    }
)


class _LineLoader(yaml.SafeLoader):
    """SafeLoader that records the source line of every mapping."""


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode) -> dict[str, Any]:
    """Construct a mapping, recording its line and rejecting a duplicate key.

    PyYAML's default behaviour on a duplicate key is last-one-wins, silently.
    For a tool whose output is a gap list that is the worst possible failure
    mode: a document with two ``ucas:`` sections would lose the first one
    without a word, and the report would then be wrong in the direction of
    looking more complete than the document is. Found while writing
    ``validation/validate_parser_rejections.py``, which had written a second
    ``hazards:`` section by accident and watched the first one vanish.
    """
    seen: set[str] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            continue
        if key in seen:
            raise AnalysisFormatError(
                f"key {key!r} is declared twice in the same mapping; PyYAML would "
                f"silently keep only the last one",
                f"line {key_node.start_mark.line + 1}",
            )
        seen.add(key)
    mapping = yaml.SafeLoader.construct_mapping(loader, node, deep=True)
    mapping[LINE_KEY] = node.start_mark.line + 1
    return mapping


_LineLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


# -- primitive accessors --------------------------------------------------
def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisFormatError(
            f"expected a mapping, found {type(value).__name__}", path
        )
    return value


def _line_of(mapping: dict[str, Any]) -> int:
    line = mapping.get(LINE_KEY, 0)
    return int(line) if isinstance(line, int) else 0


def _reject_unknown(mapping: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(k for k in mapping if k != LINE_KEY and k not in allowed)
    if unknown:
        raise AnalysisFormatError(
            "unknown key(s) "
            + ", ".join(repr(k) for k in unknown)
            + "; allowed here: "
            + ", ".join(sorted(allowed)),
            path,
        )


def _get_str(mapping: dict[str, Any], key: str, path: str, *, required: bool) -> str:
    if key not in mapping:
        if required:
            raise AnalysisFormatError(f"required key {key!r} is missing", path)
        return ""
    value = mapping[key]
    if value is None:
        if required:
            raise AnalysisFormatError(f"required key {key!r} is empty", f"{path}.{key}")
        return ""
    if not isinstance(value, str):
        raise AnalysisFormatError(
            f"expected a string, found {type(value).__name__}", f"{path}.{key}"
        )
    text = value.strip()
    if required and not text:
        raise AnalysisFormatError("must not be blank", f"{path}.{key}")
    return text


def _get_bool(mapping: dict[str, Any], key: str, path: str, default: bool) -> bool:
    if key not in mapping or mapping[key] is None:
        return default
    value = mapping[key]
    if not isinstance(value, bool):
        raise AnalysisFormatError(
            f"expected true or false, found {type(value).__name__}", f"{path}.{key}"
        )
    return value


def _get_str_list(mapping: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    if key not in mapping or mapping[key] is None:
        return ()
    value = mapping[key]
    if not isinstance(value, list):
        raise AnalysisFormatError(
            f"expected a list, found {type(value).__name__}", f"{path}.{key}"
        )
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise AnalysisFormatError(
                "expected a non-empty string", f"{path}.{key}[{index}]"
            )
        text = item.strip()
        if text in out:
            raise AnalysisFormatError(f"duplicate entry {text!r}", f"{path}.{key}[{index}]")
        out.append(text)
    return tuple(out)


def _get_list_of_mappings(
    document: dict[str, Any], key: str, path: str
) -> list[dict[str, Any]]:
    if key not in document or document[key] is None:
        return []
    value = document[key]
    if not isinstance(value, list):
        raise AnalysisFormatError(
            f"expected a list, found {type(value).__name__}", f"{path}{key}"
        )
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        out.append(_require_mapping(item, f"{path}{key}[{index}]"))
    return out


def _get_uca_class(mapping: dict[str, Any], path: str) -> UcaClass:
    raw = _get_str(mapping, "uca_class", path, required=True)
    try:
        return UcaClass(raw)
    except ValueError as exc:
        allowed = ", ".join(c.value for c in UcaClass)
        raise AnalysisFormatError(
            f"{raw!r} is not an STPA UCA class; allowed: {allowed}",
            f"{path}.uca_class",
        ) from exc


def _unique_ids(records: list[tuple[str, str, int]]) -> None:
    """Raise if any identifier appears twice. ``records`` is (id, path, line)."""
    seen: dict[str, str] = {}
    for identifier, path, _line in records:
        if identifier in seen:
            raise AnalysisFormatError(
                f"identifier {identifier!r} is already declared at {seen[identifier]}",
                f"{path}.id",
            )
        seen[identifier] = path


def _check_refs(
    refs: tuple[str, ...], known: frozenset[str], path: str, key: str, what: str
) -> None:
    for index, ref in enumerate(refs):
        if ref not in known:
            raise AnalysisFormatError(
                f"{what} {ref!r} is not declared in this document",
                f"{path}.{key}[{index}]",
            )


def _check_ref(ref: str, known: frozenset[str], path: str, key: str, what: str) -> None:
    if ref not in known:
        raise AnalysisFormatError(
            f"{what} {ref!r} is not declared in this document", f"{path}.{key}"
        )


# -- public API -----------------------------------------------------------
def parse_analysis(text: str, source: str = "<string>") -> Analysis:
    """Parse an STPA analysis from a YAML string.

    Parameters
    ----------
    text : str
        YAML document text.
    source : str
        Label used in reports; usually the file path.

    Returns
    -------
    Analysis
        The parsed analysis, with every internal reference resolved.

    Raises
    ------
    AnalysisFormatError
        On any YAML syntax error, schema violation or dangling reference. The
        exception's ``key_path`` names the offending key.
    """
    try:
        raw = yaml.load(text, Loader=_LineLoader)
    except yaml.YAMLError as exc:
        raise AnalysisFormatError(f"YAML syntax error: {exc}") from exc
    if raw is None:
        raise AnalysisFormatError("document is empty")
    document = _require_mapping(raw, "<document>")
    _reject_unknown(document, _TOP_KEYS, "<document>")

    meta = _require_mapping(document.get("meta", {LINE_KEY: 0}), "meta")
    _reject_unknown(meta, _META_KEYS, "meta")
    title = _get_str(meta, "title", "meta", required=True)
    system = _get_str(meta, "system", "meta", required=False)
    revision = _get_str(meta, "revision", "meta", required=False)

    losses = _parse_losses(document)
    hazards = _parse_hazards(document, frozenset(x.id for x in losses))
    structure = _parse_structure(document)
    constraints = _parse_constraints(document, frozenset(x.id for x in hazards))
    action_ids = frozenset(a.id for a in structure.control_actions)
    ucas = _parse_ucas(document, action_ids, frozenset(x.id for x in hazards))
    exclusions = _parse_exclusions(document, action_ids)
    evidence = _parse_evidence(document)
    mitigations = _parse_mitigations(document, frozenset(x.id for x in evidence))
    scenarios = _parse_scenarios(
        document,
        frozenset(x.id for x in ucas),
        frozenset(x.id for x in mitigations),
    )

    return Analysis(
        title=title,
        system=system,
        revision=revision,
        losses=losses,
        hazards=hazards,
        constraints=constraints,
        structure=structure,
        ucas=ucas,
        exclusions=exclusions,
        loss_scenarios=scenarios,
        mitigations=mitigations,
        evidence=evidence,
        source=source,
    )


def load_analysis(path: str | os.PathLike[str]) -> Analysis:
    """Parse an STPA analysis from a YAML file.

    Raises
    ------
    AnalysisFormatError
        If the file is missing, is a directory, is not UTF-8, or fails
        :func:`parse_analysis`.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AnalysisFormatError(f"no such file: {file_path}") from exc
    except IsADirectoryError as exc:
        raise AnalysisFormatError(f"not a file: {file_path}") from exc
    except UnicodeDecodeError as exc:
        raise AnalysisFormatError(f"{file_path} is not valid UTF-8: {exc}") from exc
    return parse_analysis(text, source=str(file_path))


# -- section parsers ------------------------------------------------------
def _parse_losses(document: dict[str, Any]) -> tuple[Loss, ...]:
    allowed = frozenset({"id", "description", "stakeholders"})
    out: list[Loss] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "losses", "")):
        path = f"losses[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        out.append(
            Loss(
                id=identifier,
                description=_get_str(item, "description", path, required=True),
                stakeholders=_get_str_list(item, "stakeholders", path),
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


def _parse_hazards(document: dict[str, Any], loss_ids: frozenset[str]) -> tuple[Hazard, ...]:
    allowed = frozenset({"id", "description", "losses"})
    out: list[Hazard] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "hazards", "")):
        path = f"hazards[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        refs = _get_str_list(item, "losses", path)
        _check_refs(refs, loss_ids, path, "losses", "loss")
        out.append(
            Hazard(
                id=identifier,
                description=_get_str(item, "description", path, required=True),
                losses=refs,
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


def _parse_constraints(
    document: dict[str, Any], hazard_ids: frozenset[str]
) -> tuple[SystemConstraint, ...]:
    allowed = frozenset({"id", "description", "hazards", "requirements"})
    out: list[SystemConstraint] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "constraints", "")):
        path = f"constraints[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        refs = _get_str_list(item, "hazards", path)
        _check_refs(refs, hazard_ids, path, "hazards", "hazard")
        out.append(
            SystemConstraint(
                id=identifier,
                description=_get_str(item, "description", path, required=True),
                hazards=refs,
                requirements=_get_str_list(item, "requirements", path),
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


_COMPONENT_SECTIONS: tuple[tuple[str, ComponentKind], ...] = (
    ("controllers", ComponentKind.CONTROLLER),
    ("actuators", ComponentKind.ACTUATOR),
    ("sensors", ComponentKind.SENSOR),
    ("controlled_processes", ComponentKind.PROCESS),
)


def _parse_structure(document: dict[str, Any]) -> ControlStructure:
    raw = document.get("control_structure")
    if raw is None:
        return ControlStructure()
    section = _require_mapping(raw, "control_structure")
    _reject_unknown(section, _STRUCTURE_KEYS, "control_structure")

    components: list[Component] = []
    records: list[tuple[str, str, int]] = []
    comp_allowed = frozenset({"id", "name", "process_model"})
    for key, kind in _COMPONENT_SECTIONS:
        for index, item in enumerate(
            _get_list_of_mappings(section, key, "control_structure.")
        ):
            path = f"control_structure.{key}[{index}]"
            _reject_unknown(item, comp_allowed, path)
            identifier = _get_str(item, "id", path, required=True)
            model = _get_str_list(item, "process_model", path)
            if model and kind is not ComponentKind.CONTROLLER:
                raise AnalysisFormatError(
                    "process_model may only be declared on a controller",
                    f"{path}.process_model",
                )
            components.append(
                Component(
                    id=identifier,
                    name=_get_str(item, "name", path, required=True),
                    kind=kind,
                    process_model=model,
                    line=_line_of(item),
                )
            )
            records.append((identifier, path, _line_of(item)))

    known = frozenset(c.id for c in components)
    controllers = frozenset(
        c.id for c in components if c.kind is ComponentKind.CONTROLLER
    )
    actuators = frozenset(c.id for c in components if c.kind is ComponentKind.ACTUATOR)
    sensors = frozenset(c.id for c in components if c.kind is ComponentKind.SENSOR)
    processes = frozenset(c.id for c in components if c.kind is ComponentKind.PROCESS)
    del known

    actions: list[ControlAction] = []
    action_allowed = frozenset(
        {"id", "name", "controller", "actuator", "process", "continuous"}
    )
    for index, item in enumerate(
        _get_list_of_mappings(section, "control_actions", "control_structure.")
    ):
        path = f"control_structure.control_actions[{index}]"
        _reject_unknown(item, action_allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        controller = _get_str(item, "controller", path, required=True)
        actuator = _get_str(item, "actuator", path, required=True)
        process = _get_str(item, "process", path, required=True)
        _check_ref(controller, controllers, path, "controller", "controller")
        _check_ref(actuator, actuators, path, "actuator", "actuator")
        _check_ref(process, processes, path, "process", "controlled process")
        actions.append(
            ControlAction(
                id=identifier,
                name=_get_str(item, "name", path, required=True),
                controller=controller,
                actuator=actuator,
                process=process,
                continuous=_get_bool(item, "continuous", path, True),
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))

    feedback: list[FeedbackPath] = []
    fb_allowed = frozenset({"id", "name", "sensor", "controller", "process"})
    for index, item in enumerate(
        _get_list_of_mappings(section, "feedback", "control_structure.")
    ):
        path = f"control_structure.feedback[{index}]"
        _reject_unknown(item, fb_allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        sensor = _get_str(item, "sensor", path, required=True)
        controller = _get_str(item, "controller", path, required=True)
        process = _get_str(item, "process", path, required=True)
        _check_ref(sensor, sensors, path, "sensor", "sensor")
        _check_ref(controller, controllers, path, "controller", "controller")
        _check_ref(process, processes, path, "process", "controlled process")
        feedback.append(
            FeedbackPath(
                id=identifier,
                name=_get_str(item, "name", path, required=True),
                sensor=sensor,
                controller=controller,
                process=process,
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))

    _unique_ids(records)
    return ControlStructure(
        components=tuple(components),
        control_actions=tuple(actions),
        feedback=tuple(feedback),
    )


def _parse_ucas(
    document: dict[str, Any], action_ids: frozenset[str], hazard_ids: frozenset[str]
) -> tuple[Uca, ...]:
    allowed = frozenset(
        {"id", "control_action", "uca_class", "description", "context", "hazards"}
    )
    out: list[Uca] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "ucas", "")):
        path = f"ucas[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        action = _get_str(item, "control_action", path, required=True)
        _check_ref(action, action_ids, path, "control_action", "control action")
        refs = _get_str_list(item, "hazards", path)
        _check_refs(refs, hazard_ids, path, "hazards", "hazard")
        out.append(
            Uca(
                id=identifier,
                control_action=action,
                uca_class=_get_uca_class(item, path),
                description=_get_str(item, "description", path, required=True),
                context=_get_str(item, "context", path, required=False),
                hazards=refs,
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


def _parse_exclusions(
    document: dict[str, Any], action_ids: frozenset[str]
) -> tuple[Exclusion, ...]:
    allowed = frozenset({"control_action", "uca_class", "justification", "author", "date"})
    out: list[Exclusion] = []
    seen: dict[tuple[str, UcaClass], str] = {}
    for index, item in enumerate(_get_list_of_mappings(document, "exclusions", "")):
        path = f"exclusions[{index}]"
        _reject_unknown(item, allowed, path)
        action = _get_str(item, "control_action", path, required=True)
        _check_ref(action, action_ids, path, "control_action", "control action")
        uca_class = _get_uca_class(item, path)
        cell = (action, uca_class)
        if cell in seen:
            raise AnalysisFormatError(
                f"cell ({action}, {uca_class.value}) is already excluded at {seen[cell]}",
                path,
            )
        seen[cell] = path
        out.append(
            Exclusion(
                control_action=action,
                uca_class=uca_class,
                justification=_get_str(item, "justification", path, required=True),
                author=_get_str(item, "author", path, required=False),
                date=_get_str(item, "date", path, required=False),
                line=_line_of(item),
            )
        )
    return tuple(out)


def _parse_scenarios(
    document: dict[str, Any], uca_ids: frozenset[str], mitigation_ids: frozenset[str]
) -> tuple[LossScenario, ...]:
    allowed = frozenset(
        {"id", "description", "ucas", "causal_factor_class", "mitigations"}
    )
    out: list[LossScenario] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "loss_scenarios", "")):
        path = f"loss_scenarios[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        ucas = _get_str_list(item, "ucas", path)
        _check_refs(ucas, uca_ids, path, "ucas", "UCA")
        mitigations = _get_str_list(item, "mitigations", path)
        _check_refs(mitigations, mitigation_ids, path, "mitigations", "mitigation")
        out.append(
            LossScenario(
                id=identifier,
                description=_get_str(item, "description", path, required=True),
                ucas=ucas,
                causal_factor_class=_get_str(
                    item, "causal_factor_class", path, required=False
                ),
                mitigations=mitigations,
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


def _parse_mitigations(
    document: dict[str, Any], evidence_ids: frozenset[str]
) -> tuple[Mitigation, ...]:
    allowed = frozenset({"id", "description", "requirements", "evidence"})
    out: list[Mitigation] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "mitigations", "")):
        path = f"mitigations[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        refs = _get_str_list(item, "evidence", path)
        _check_refs(refs, evidence_ids, path, "evidence", "evidence artifact")
        out.append(
            Mitigation(
                id=identifier,
                description=_get_str(item, "description", path, required=True),
                requirements=_get_str_list(item, "requirements", path),
                evidence=refs,
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)


def _parse_evidence(document: dict[str, Any]) -> tuple[Evidence, ...]:
    allowed = frozenset({"id", "kind", "description", "tests", "path"})
    out: list[Evidence] = []
    records: list[tuple[str, str, int]] = []
    for index, item in enumerate(_get_list_of_mappings(document, "evidence", "")):
        path = f"evidence[{index}]"
        _reject_unknown(item, allowed, path)
        identifier = _get_str(item, "id", path, required=True)
        raw_kind = _get_str(item, "kind", path, required=True)
        try:
            kind = EvidenceKind(raw_kind)
        except ValueError as exc:
            allowed_kinds = ", ".join(k.value for k in EvidenceKind)
            raise AnalysisFormatError(
                f"{raw_kind!r} is not an evidence kind; allowed: {allowed_kinds}",
                f"{path}.kind",
            ) from exc
        tests = _get_str_list(item, "tests", path)
        if tests and kind is not EvidenceKind.TEST:
            raise AnalysisFormatError(
                "test identifiers may only be declared on evidence of kind 'test'",
                f"{path}.tests",
            )
        out.append(
            Evidence(
                id=identifier,
                kind=kind,
                description=_get_str(item, "description", path, required=True),
                tests=tests,
                path=_get_str(item, "path", path, required=False),
                line=_line_of(item),
            )
        )
        records.append((identifier, path, _line_of(item)))
    _unique_ids(records)
    return tuple(out)
