"""Parse and validate the YAML assurance-case format.

The format is a direct transcription of Core GSN (GSN Community Standard
Version 3, SCSC-141C, SCSC Assurance Case Working Group, May 2021): a list of
elements, each with one of six types, and a list of relationships, each of one
of two types.

Everything this module rejects is a *format* defect: the document is not a
well-formed GSN case. Defects in the argument the document encodes -- a claim
with nothing behind it, an artifact that has changed since it was cited -- are
not rejected here. They are computed by :mod:`assuregraph.checks` and reported
as findings, because a case with gaps is still a case.

Schema::

    name: <str>                       # required
    top_goals: [<node id>, ...]       # optional; inferred when absent
    nodes:                            # required, non-empty list
      - id: <str>                     # required, unique
        type: goal | strategy | solution | context | assumption | justification
        statement: <str>              # required
        undeveloped: <bool>           # optional, goal/strategy only
        discharged: <bool>            # optional, assumption only
        discharged_by: <node id>      # optional, assumption only
        evidence:                     # required on solution, forbidden elsewhere
          path: <str>                 # required
          sha256: <64 hex chars>      # optional
          recorded: <str>             # optional
    edges:                            # optional list
      - from: <node id>
        to: <node id>
        type: supported_by | in_context_of
"""

from __future__ import annotations

import difflib
import os
from typing import Any

import yaml

from .errors import CaseFormatError, CaseIOError
from .model import PERMITTED_EDGES, AssuranceCase, Edge, EdgeKind, Evidence, Node, NodeKind

__all__ = ["LOADER_NAME", "load_case", "parse_case"]

_TOP_LEVEL_KEYS = frozenset({"name", "top_goals", "nodes", "edges"})
_NODE_KEYS = frozenset(
    {"id", "type", "statement", "undeveloped", "discharged", "discharged_by", "evidence"}
)
_EDGE_KEYS = frozenset({"from", "to", "type"})
_EVIDENCE_KEYS = frozenset({"path", "sha256", "recorded"})
_HEX = frozenset("0123456789abcdef")
_DECORATED_KINDS = frozenset({NodeKind.GOAL, NodeKind.STRATEGY})

#: PyYAML's C loader when the wheel was built against libyaml, otherwise the
#: pure-Python one. Both are *safe* loaders: neither constructs arbitrary Python
#: objects from a document, which matters because a case file is often something
#: a reviewer received rather than wrote. The choice affects speed only, and by
#: roughly an order of magnitude on a large case; see
#: ``validation/validate_scale.py``, which records which loader it ran with.
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
LOADER_NAME = _LOADER.__name__


def _suggest(bad: str, candidates: object) -> str:
    """Return ' (did you mean X?)' when ``bad`` is close to one of ``candidates``."""
    pool = sorted(str(c) for c in candidates)  # type: ignore[union-attr]
    close = difflib.get_close_matches(bad, pool, n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _require_mapping(obj: object, location: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise CaseFormatError(
            f"expected a mapping, found {type(obj).__name__}", location=location
        )
    return obj


def _reject_unknown_keys(mapping: dict[str, Any], allowed: frozenset[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise CaseFormatError(
            f"unknown key {unknown[0]!r}{_suggest(unknown[0], allowed)}; "
            f"permitted keys are {sorted(allowed)}",
            location=location,
        )


def _require_str(mapping: dict[str, Any], key: str, location: str) -> str:
    if key not in mapping:
        raise CaseFormatError(f"missing required key {key!r}", location=location)
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise CaseFormatError(
            f"key {key!r} must be a non-empty string, found {type(value).__name__} {value!r}",
            location=location,
        )
    return value


def _optional_bool(mapping: dict[str, Any], key: str, location: str) -> bool:
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        raise CaseFormatError(
            f"key {key!r} must be true or false, found {type(value).__name__} {value!r}",
            location=location,
        )
    return value


def _parse_evidence(raw: object, location: str) -> Evidence:
    mapping = _require_mapping(raw, location)
    _reject_unknown_keys(mapping, _EVIDENCE_KEYS, location)
    path = _require_str(mapping, "path", location)
    digest = mapping.get("sha256")
    if digest is not None:
        if not isinstance(digest, str):
            raise CaseFormatError(
                f"key 'sha256' must be a string, found {type(digest).__name__}",
                location=location,
            )
        lowered = digest.strip().lower()
        if len(lowered) != 64 or not set(lowered) <= _HEX:
            raise CaseFormatError(
                "key 'sha256' must be exactly 64 lowercase hexadecimal characters "
                f"(a SHA-256 hex digest); found {len(digest.strip())} characters {digest!r}",
                location=location,
            )
        digest = lowered
    recorded = mapping.get("recorded")
    if recorded is not None and not isinstance(recorded, str):
        raise CaseFormatError(
            f"key 'recorded' must be a string, found {type(recorded).__name__}",
            location=location,
        )
    return Evidence(path=path, sha256=digest, recorded=recorded)


def _parse_node(raw: object, index: int) -> Node:
    location = f"nodes[{index}]"
    mapping = _require_mapping(raw, location)
    _reject_unknown_keys(mapping, _NODE_KEYS, location)
    node_id = _require_str(mapping, "id", location)
    location = f"nodes[{index}] (id {node_id!r})"
    type_name = _require_str(mapping, "type", location)
    try:
        kind = NodeKind(type_name)
    except ValueError:
        valid = sorted(k.value for k in NodeKind)
        raise CaseFormatError(
            f"unknown node type {type_name!r}{_suggest(type_name, valid)}; "
            f"the six Core GSN element types are {valid}",
            location=location,
        ) from None
    statement = _require_str(mapping, "statement", location)

    undeveloped = _optional_bool(mapping, "undeveloped", location)
    if undeveloped and kind not in _DECORATED_KINDS:
        raise CaseFormatError(
            "the GSN 'Undeveloped' decorator applies only to a Goal or a Strategy; "
            f"node is a {kind.value}",
            location=location,
        )

    discharged = _optional_bool(mapping, "discharged", location)
    discharged_by = mapping.get("discharged_by")
    if discharged_by is not None and not isinstance(discharged_by, str):
        raise CaseFormatError(
            f"key 'discharged_by' must be a node id string, found {type(discharged_by).__name__}",
            location=location,
        )
    if (discharged or discharged_by is not None) and kind is not NodeKind.ASSUMPTION:
        raise CaseFormatError(
            "keys 'discharged' and 'discharged_by' apply only to an Assumption; "
            f"node is a {kind.value}",
            location=location,
        )
    if discharged_by is not None and not discharged:
        raise CaseFormatError(
            f"'discharged_by' names {discharged_by!r} but 'discharged' is not true; "
            "set discharged: true or remove discharged_by",
            location=location,
        )

    evidence_raw = mapping.get("evidence")
    if kind is NodeKind.SOLUTION:
        if evidence_raw is None:
            raise CaseFormatError(
                "a Solution must cite an evidence artifact: add an 'evidence' mapping "
                "with at least a 'path' key",
                location=location,
            )
        evidence = _parse_evidence(evidence_raw, f"{location} evidence")
    else:
        if evidence_raw is not None:
            raise CaseFormatError(
                f"only a Solution may carry 'evidence'; node is a {kind.value}",
                location=location,
            )
        evidence = None

    return Node(
        node_id=node_id,
        kind=kind,
        statement=statement,
        undeveloped=undeveloped,
        discharged=discharged,
        discharged_by=discharged_by,
        evidence=evidence,
    )


def _parse_edge(raw: object, index: int, nodes: dict[str, Node]) -> Edge:
    location = f"edges[{index}]"
    mapping = _require_mapping(raw, location)
    _reject_unknown_keys(mapping, _EDGE_KEYS, location)
    source = _require_str(mapping, "from", location)
    target = _require_str(mapping, "to", location)
    type_name = _require_str(mapping, "type", location)
    location = f"edges[{index}] ({source!r} -> {target!r})"
    try:
        kind = EdgeKind(type_name)
    except ValueError:
        valid = sorted(k.value for k in EdgeKind)
        raise CaseFormatError(
            f"unknown relationship type {type_name!r}{_suggest(type_name, valid)}; "
            f"the two Core GSN relationship types are {valid}",
            location=location,
        ) from None
    for role, node_id in (("from", source), ("to", target)):
        if node_id not in nodes:
            raise CaseFormatError(
                f"{role} references unknown node id {node_id!r}"
                f"{_suggest(node_id, nodes.keys())}",
                location=location,
            )
    if source == target:
        raise CaseFormatError(
            f"a node may not be related to itself ({source!r} -> {target!r})",
            location=location,
        )
    source_kind = nodes[source].kind
    target_kind = nodes[target].kind
    permitted = PERMITTED_EDGES.get((source_kind, kind))
    if permitted is None:
        raise CaseFormatError(
            f"a {source_kind.value} may not be the source of a {kind.value} relationship; "
            "in Core GSN only a Goal or a Strategy may be",
            location=location,
        )
    if target_kind not in permitted:
        raise CaseFormatError(
            f"Core GSN does not permit {source_kind.value} --{kind.value}--> "
            f"{target_kind.value}; permitted targets are "
            f"{sorted(k.value for k in permitted)}",
            location=location,
        )
    return Edge(source=source, target=target, kind=kind)


def parse_case(
    document: object, *, base_dir: str = ".", name_hint: str = "<string>"
) -> AssuranceCase:
    """Validate an already-loaded YAML document and build an :class:`AssuranceCase`.

    Args:
        document: The object ``yaml.safe_load`` returned. Must be a mapping.
        base_dir: Directory that relative evidence paths resolve against.
        name_hint: Used only in error messages to say where the document came from.

    Returns:
        The validated case.

    Raises:
        CaseFormatError: on any schema violation, with the YAML key path and an
            instruction in the message.
    """
    mapping = _require_mapping(document, name_hint)
    _reject_unknown_keys(mapping, _TOP_LEVEL_KEYS, name_hint)
    case_name = _require_str(mapping, "name", name_hint)

    raw_nodes = mapping.get("nodes")
    if raw_nodes is None:
        raise CaseFormatError("missing required key 'nodes'", location=name_hint)
    if not isinstance(raw_nodes, list):
        raise CaseFormatError(
            f"key 'nodes' must be a list, found {type(raw_nodes).__name__}", location=name_hint
        )
    if not raw_nodes:
        raise CaseFormatError(
            "key 'nodes' is empty; an assurance case needs at least one Goal",
            location=name_hint,
        )

    nodes: dict[str, Node] = {}
    for index, raw in enumerate(raw_nodes):
        node = _parse_node(raw, index)
        if node.node_id in nodes:
            raise CaseFormatError(
                f"duplicate node id {node.node_id!r}; node ids must be unique within a case",
                location=f"nodes[{index}]",
            )
        nodes[node.node_id] = node

    for node in nodes.values():
        if node.discharged_by is not None and node.discharged_by not in nodes:
            raise CaseFormatError(
                f"'discharged_by' references unknown node id {node.discharged_by!r}"
                f"{_suggest(node.discharged_by, nodes.keys())}",
                location=f"node {node.node_id!r}",
            )

    raw_edges = mapping.get("edges", [])
    if not isinstance(raw_edges, list):
        raise CaseFormatError(
            f"key 'edges' must be a list, found {type(raw_edges).__name__}", location=name_hint
        )
    edges: list[Edge] = []
    seen: set[tuple[str, str, EdgeKind]] = set()
    for index, raw in enumerate(raw_edges):
        edge = _parse_edge(raw, index, nodes)
        key = (edge.source, edge.target, edge.kind)
        if key in seen:
            raise CaseFormatError(
                f"duplicate {edge.kind.value} relationship {edge.source!r} -> {edge.target!r}",
                location=f"edges[{index}]",
            )
        seen.add(key)
        edges.append(edge)

    raw_top = mapping.get("top_goals", [])
    if not isinstance(raw_top, list):
        raise CaseFormatError(
            f"key 'top_goals' must be a list, found {type(raw_top).__name__}", location=name_hint
        )
    top_goals: list[str] = []
    for index, value in enumerate(raw_top):
        location = f"top_goals[{index}]"
        if not isinstance(value, str):
            raise CaseFormatError(
                f"must be a node id string, found {type(value).__name__}", location=location
            )
        if value not in nodes:
            raise CaseFormatError(
                f"unknown node id {value!r}{_suggest(value, nodes.keys())}", location=location
            )
        if nodes[value].kind is not NodeKind.GOAL:
            raise CaseFormatError(
                f"{value!r} is a {nodes[value].kind.value}; a top goal must be a Goal",
                location=location,
            )
        if value in top_goals:
            raise CaseFormatError(f"duplicate top goal {value!r}", location=location)
        top_goals.append(value)

    return AssuranceCase(
        name=case_name,
        nodes=nodes,
        edges=tuple(edges),
        base_dir=base_dir,
        top_goals=tuple(top_goals),
    )


def load_case(path: str | os.PathLike[str]) -> AssuranceCase:
    """Read and validate a YAML assurance case from ``path``.

    Relative evidence paths in the returned case resolve against the directory
    containing ``path``.

    Raises:
        CaseIOError: the file could not be read, or is not parsable as YAML.
        CaseFormatError: the YAML parsed but does not match the schema.
    """
    text_path = os.fspath(path)
    try:
        with open(text_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise CaseIOError(f"cannot read case file {text_path!r}: {exc}") from exc
    try:
        document = yaml.load(text, Loader=_LOADER)
    except yaml.YAMLError as exc:
        raise CaseIOError(f"case file {text_path!r} is not valid YAML: {exc}") from exc
    base_dir = os.path.dirname(os.path.abspath(text_path)) or "."
    return parse_case(document, base_dir=base_dir, name_hint=text_path)
