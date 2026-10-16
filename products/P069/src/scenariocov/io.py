"""Loading and saving the four JSON documents this package reads.

* a **space** document -- ``{"parameters": [...]}``, see :mod:`scenariocov.space`;
* a **campaign** document -- ``{"scenarios": [{param: value, ...}, ...]}``,
  optionally with a ``"metadata"`` object that is carried through untouched;
* a **classes** document -- ``{"classes": [...], "boundaries": [...]}``, see
  :mod:`scenariocov.classes`;
* a **pool** document -- the same shape as a campaign.

JSON is the only format read. There is no YAML support, so that the package
depends on nothing but numpy and scikit-learn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .classes import BoundarySpec, EquivalenceClass
from .space import ScenarioSpace

__all__ = [
    "load_campaign",
    "load_class_document",
    "load_space",
    "save_campaign",
    "save_space",
]


def _read_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {target}")
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_space(path: str | Path) -> ScenarioSpace:
    """Read a space document."""
    return ScenarioSpace.from_dict(_read_json(path))


def save_space(space: ScenarioSpace, path: str | Path) -> None:
    """Write a space document."""
    Path(path).write_text(json.dumps(space.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_campaign(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read a campaign (or pool) document and return its scenario records.

    Raises:
        ValueError: when the document has no non-empty ``scenarios`` list.
    """
    payload = _read_json(path)
    scenarios = payload.get("scenarios") if isinstance(payload, Mapping) else payload
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{path}: expected a non-empty 'scenarios' list")
    for i, record in enumerate(scenarios):
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: scenario {i} is not an object")
    return tuple(dict(r) for r in scenarios)


def save_campaign(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a campaign document, with optional metadata carried alongside."""
    payload: dict[str, Any] = {"scenarios": [dict(r) for r in records]}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_class_document(
    path: str | Path,
) -> tuple[tuple[EquivalenceClass, ...], tuple[BoundarySpec, ...]]:
    """Read a classes document and return ``(classes, boundaries)``."""
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object with 'classes' and 'boundaries'")
    classes = tuple(EquivalenceClass.from_dict(c) for c in payload.get("classes", ()))
    boundaries = tuple(BoundarySpec.from_dict(b) for b in payload.get("boundaries", ()))
    if not classes and not boundaries:
        raise ValueError(f"{path}: document declares neither classes nor boundaries")
    return classes, boundaries
