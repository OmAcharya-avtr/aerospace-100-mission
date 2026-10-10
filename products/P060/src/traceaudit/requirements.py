"""Parsing numbered requirements out of a markdown document.

The parser is deliberately line-oriented and dumb.  It recognises a
*declaration* — an id at the start of a line, optionally behind a heading
marker, a list bullet or bold markers — and nothing else.  An id appearing
inside a sentence is a reference and is not a declaration, because a document
that says "unlike REQ-004, this requirement ..." must not thereby declare
REQ-004 a second time.

There is no markdown parser here and none is wanted: the audit must behave
identically on a document a reviewer reads as plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import TraceConfig

_FENCE = re.compile(r"^[ \t]*(?:```|~~~)")


@dataclass(frozen=True)
class Requirement:
    """One declared requirement.

    Attributes
    ----------
    id:
        The requirement identifier exactly as written, e.g. ``REQ-003``.
    title:
        Whatever followed the id on the declaration line, stripped.  May be
        empty; an empty title is not a finding.
    source:
        Path of the document the declaration was read from, as given.
    line:
        One-based line number of the declaration.
    """

    id: str
    title: str
    source: str
    line: int

    @property
    def location(self) -> str:
        """``path:line`` for printing."""
        return f"{self.source}:{self.line}"


@dataclass(frozen=True)
class RequirementDocument:
    """Every declaration found in one or more documents, in file order."""

    declarations: tuple[Requirement, ...]
    sources: tuple[str, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        """Unique ids, in order of first declaration."""
        seen: dict[str, None] = {}
        for req in self.declarations:
            seen.setdefault(req.id, None)
        return tuple(seen)

    @property
    def duplicates(self) -> dict[str, tuple[Requirement, ...]]:
        """Ids declared more than once, mapped to every declaration."""
        grouped: dict[str, list[Requirement]] = {}
        for req in self.declarations:
            grouped.setdefault(req.id, []).append(req)
        return {k: tuple(v) for k, v in grouped.items() if len(v) > 1}

    def first(self, req_id: str) -> Requirement | None:
        """The first declaration of ``req_id``, or ``None``."""
        for req in self.declarations:
            if req.id == req_id:
                return req
        return None


def parse_requirements_text(
    text: str,
    *,
    source: str = "<string>",
    config: TraceConfig | None = None,
) -> RequirementDocument:
    """Parse requirement declarations from markdown text.

    Parameters
    ----------
    text:
        The whole document.
    source:
        Label used in :attr:`Requirement.source`.
    config:
        Parser options; :meth:`TraceConfig.default` when ``None``.

    Returns
    -------
    RequirementDocument
        Declarations in line order, duplicates retained.
    """
    cfg = config or TraceConfig.default()
    pattern = re.compile(cfg.declaration_regex)
    found: list[Requirement] = []
    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if cfg.skip_code_fences and _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = pattern.match(raw)
        if match is None:
            continue
        title = (match.groupdict().get("title") or "").strip()
        found.append(
            Requirement(id=match.group("id"), title=title, source=source, line=lineno)
        )
    return RequirementDocument(declarations=tuple(found), sources=(source,))


def parse_requirements_file(
    path: str | Path, *, config: TraceConfig | None = None, relative_to: Path | None = None
) -> RequirementDocument:
    """Parse one markdown requirements document from disk.

    Parameters
    ----------
    path:
        The document.
    config:
        Parser options.
    relative_to:
        If given, the recorded source is ``path`` expressed relative to this
        directory when possible, so that committed output does not contain an
        absolute path.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"requirements document not found: {p}")
    label = str(p)
    if relative_to is not None:
        try:
            label = str(p.resolve().relative_to(Path(relative_to).resolve()))
        except ValueError:
            label = p.name
    return parse_requirements_text(
        p.read_text(encoding="utf-8"), source=label, config=config
    )


def merge_documents(docs: list[RequirementDocument]) -> RequirementDocument:
    """Concatenate several parsed documents, retaining duplicates across them."""
    decls: list[Requirement] = []
    sources: list[str] = []
    for doc in docs:
        decls.extend(doc.declarations)
        sources.extend(doc.sources)
    return RequirementDocument(declarations=tuple(decls), sources=tuple(sources))
