"""Shared fixtures and helpers.

``EXAMPLES`` points at the shipped example cases so the integration tests run
against the same files the README shows output from, rather than a copy that
can drift.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from assuregraph import AssuranceCase, parse_case

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "examples", "cases")
COMPLETE_CASE = os.path.join(EXAMPLES, "complete_case.yaml")
INCOMPLETE_CASE = os.path.join(EXAMPLES, "incomplete_case.yaml")
CYCLIC_CASE = os.path.join(EXAMPLES, "cyclic_case.yaml")


def node(node_id: str, kind: str, statement: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build a node mapping for the YAML schema."""
    out: dict[str, Any] = {
        "id": node_id,
        "type": kind,
        "statement": statement or f"statement for {node_id}",
    }
    out.update(extra)
    return out


def edge(source: str, target: str, kind: str = "supported_by") -> dict[str, Any]:
    """Build an edge mapping for the YAML schema."""
    return {"from": source, "to": target, "type": kind}


def document(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    *,
    name: str = "test case",
    top_goals: list[str] | None = None,
) -> dict[str, Any]:
    """Build a whole case document."""
    out: dict[str, Any] = {"name": name, "nodes": nodes}
    if edges is not None:
        out["edges"] = edges
    if top_goals is not None:
        out["top_goals"] = top_goals
    return out


@pytest.fixture
def make_case() -> Callable[..., AssuranceCase]:
    """Return a factory that parses a document dict into an AssuranceCase."""

    def factory(doc: dict[str, Any], base_dir: str = ".") -> AssuranceCase:
        return parse_case(doc, base_dir=base_dir)

    return factory


@pytest.fixture
def write_case(tmp_path: Any) -> Callable[..., str]:
    """Return a factory that writes a document dict to a YAML file and returns its path."""

    def factory(doc: dict[str, Any], filename: str = "case.yaml") -> str:
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return str(path)

    return factory


@pytest.fixture
def artifact(tmp_path: Any) -> Callable[..., tuple[str, str]]:
    """Return a factory that writes an evidence artifact and returns (path, sha256)."""

    def factory(name: str, content: str = "evidence bytes\n") -> tuple[str, str]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path), hashlib.sha256(content.encode("utf-8")).hexdigest()

    return factory
