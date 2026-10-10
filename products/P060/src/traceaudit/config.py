"""Configuration for the requirement and test-report parsers.

Everything that depends on a project's local conventions lives here, so that
the audit itself contains no hard-coded notion of what a requirement id looks
like.  The defaults implement the `REQ-NNN` convention used by this
repository's own ``docs/REQUIREMENTS.md``.

Format of a configuration file (JSON, read with the standard library only)::

    {
      "id_pattern": "REQ-\\\\d{3}",
      "declaration_pattern": null,
      "skip_code_fences": true,
      "claim_property_names": ["requirement_id", "requirements", "requirement"],
      "claim_value_separators": ",; ",
      "claim_from_node_id": true,
      "node_id_claim_pattern": "REQ[-_ ]?(\\\\d{3})",
      "node_id_claim_template": "REQ-{0}",
      "ignored_codes": [],
      "heuristic_assertions": true
    }

Every key is optional; an omitted key takes the default below.  Unknown keys
are rejected with a :class:`ValueError` naming them, because a silently
ignored misspelled key is the way a configurable linter lies to its user.

No field in this module carries physical units.  The product audits text and
test identifiers; there is no physical quantity anywhere in it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_ID_PATTERN = r"REQ-\d{3}"
#: Applied per line of the requirements document.  A declaration is an id at
#: the start of a line, optionally behind a markdown heading marker, a list
#: bullet, or bold markers.  A bare mention of an id inside a sentence is a
#: reference, not a declaration, and must not create a requirement.
DECLARATION_TEMPLATE = (
    r"^[ \t]*(?:\#{{1,6}}[ \t]*)?(?:[-*+][ \t]*)?(?:\*\*)?"
    r"(?P<id>{id})(?:\*\*)?[ \t]*(?:[:.)—-][ \t]*)?(?P<title>.*?)[ \t]*$"
)
DEFAULT_NODE_ID_CLAIM_PATTERN = r"REQ[-_ ]?(\d{3})"
DEFAULT_NODE_ID_CLAIM_TEMPLATE = "REQ-{0}"
DEFAULT_PROPERTY_NAMES = ("requirement_id", "requirements", "requirement")


@dataclass(frozen=True)
class TraceConfig:
    """Parser and reporting options.

    Attributes
    ----------
    id_pattern:
        Regular expression for the *shape* of a requirement id, without
        anchors.  Must not contain a group named ``id``.
    declaration_pattern:
        Full per-line regular expression with named groups ``id`` and
        ``title``.  ``None`` builds it from :data:`DECLARATION_TEMPLATE` and
        ``id_pattern``.
    skip_code_fences:
        Ignore lines inside ``` fenced blocks when parsing requirements.
    claim_property_names:
        junit ``<property>`` names whose value lists claimed requirement ids.
    claim_value_separators:
        Characters that separate several ids inside one property value.
    claim_from_node_id:
        Also read claims out of the test node id itself.
    node_id_claim_pattern:
        Regular expression with one capturing group, applied to the node id.
    node_id_claim_template:
        ``str.format`` template turning the captured groups into an id.
    ignored_codes:
        Finding codes (``TA001`` ...) that are reported as informational and
        do not affect the exit status.
    heuristic_assertions:
        Run the assertion heuristic (:mod:`traceaudit.assertions`) at all.
    """

    id_pattern: str = DEFAULT_ID_PATTERN
    declaration_pattern: str | None = None
    skip_code_fences: bool = True
    claim_property_names: tuple[str, ...] = DEFAULT_PROPERTY_NAMES
    claim_value_separators: str = ",; "
    claim_from_node_id: bool = True
    node_id_claim_pattern: str = DEFAULT_NODE_ID_CLAIM_PATTERN
    node_id_claim_template: str = DEFAULT_NODE_ID_CLAIM_TEMPLATE
    ignored_codes: tuple[str, ...] = field(default_factory=tuple)
    heuristic_assertions: bool = True

    def __post_init__(self) -> None:
        if not self.id_pattern:
            raise ValueError("id_pattern must be a non-empty regular expression")
        for name, pattern in (
            ("id_pattern", self.id_pattern),
            ("node_id_claim_pattern", self.node_id_claim_pattern),
        ):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{name} is not a valid regular expression: {exc}") from exc
        if "(?P<id>" in self.id_pattern:
            raise ValueError("id_pattern must not define a group named 'id'")
        try:
            compiled = re.compile(self.declaration_regex)
        except re.error as exc:
            raise ValueError(
                f"declaration_pattern is not a valid regular expression: {exc}"
            ) from exc
        if "id" not in compiled.groupindex:
            raise ValueError("declaration_pattern must define a named group 'id'")
        if not self.claim_property_names:
            raise ValueError("claim_property_names must name at least one junit property")
        if not self.claim_value_separators:
            raise ValueError("claim_value_separators must contain at least one character")
        try:
            self.node_id_claim_template.format(*["000"] * 8)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"node_id_claim_template is not formattable with positional groups: {exc}"
            ) from exc

    @property
    def declaration_regex(self) -> str:
        """The per-line declaration pattern actually used."""
        if self.declaration_pattern is not None:
            return self.declaration_pattern
        return DECLARATION_TEMPLATE.format(id=self.id_pattern)

    @classmethod
    def default(cls) -> TraceConfig:
        """The ``REQ-NNN`` convention used by this repository."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceConfig:
        """Build a config from a mapping, rejecting unknown keys."""
        if not isinstance(data, dict):
            raise TypeError(f"configuration must be a JSON object, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown configuration key(s): {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(known))}"
            )
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("claim_property_names", "ignored_codes") and value is not None:
                if isinstance(value, str):
                    raise ValueError(f"{key} must be a list of strings, not a single string")
                kwargs[key] = tuple(str(v) for v in value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> TraceConfig:
        """Read a JSON configuration file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"configuration file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable mapping of every option."""
        return {
            "id_pattern": self.id_pattern,
            "declaration_pattern": self.declaration_pattern,
            "skip_code_fences": self.skip_code_fences,
            "claim_property_names": list(self.claim_property_names),
            "claim_value_separators": self.claim_value_separators,
            "claim_from_node_id": self.claim_from_node_id,
            "node_id_claim_pattern": self.node_id_claim_pattern,
            "node_id_claim_template": self.node_id_claim_template,
            "ignored_codes": list(self.ignored_codes),
            "heuristic_assertions": self.heuristic_assertions,
        }
