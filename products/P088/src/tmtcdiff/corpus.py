"""Extract dictionary references from procedures, scripts and configuration files.

A corpus is whatever already depends on the dictionary: flight-control
procedures, offline analysis scripts, limit-checking configurations. The impact
report is only as good as the reference extraction, so this module is explicit
about the one thing it cannot do.

**A reference constructed at run time cannot be resolved statically.** A line
reading ``tm["T_PANEL_%d" % index]`` names eight parameters or none depending
on data this module never sees. Such a reference is extracted, counted, and
marked undecidable with the reason; it is never guessed at, and it is never
dropped. The count of undecidable references is a field of the coverage
verdict returned by :func:`tmtcdiff.impact.assess_impact`, which is the
mechanism by which this package reports what it cannot determine about the
corpus it was given.

Extraction is syntactic and line-based. It is not a Python parser: it finds
accessor calls and subscripts by their accessor name, and a corpus that
reaches the dictionary some other way will be under-extracted. That is also
reported, as the fraction of dictionary items the corpus never mentions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Accessor names whose first string argument is read as a dictionary
#: reference. Extend through the ``accessors`` argument rather than by editing.
DEFAULT_ACCESSORS: tuple[str, ...] = (
    "tm",
    "tc",
    "telemetry",
    "dictionary",
    "parameters",
    "get_parameter",
    "get_param",
    "get",
    "fetch",
    "read",
    "sample",
    "send_command",
    "send",
)

#: Accessors whose reference is a command rather than a parameter.
COMMAND_ACCESSORS: frozenset[str] = frozenset({"send_command", "send", "tc"})

#: Procedure verbs that take a parameter mnemonic or path as their argument.
PROCEDURE_VERBS: tuple[str, ...] = (
    "CHECK",
    "VERIFY",
    "MONITOR",
    "LOG",
    "REPORT",
    "LIMIT",
    "WAIT_FOR",
)

#: Source kinds this module understands.
SOURCE_KINDS: tuple[str, ...] = ("procedure", "script", "config")

#: File suffix to source kind, for :func:`load_corpus`.
SUFFIX_KINDS: dict[str, str] = {
    ".proc": "procedure",
    ".prc": "procedure",
    ".txt": "procedure",
    ".py": "script",
    ".json": "config",
    ".cfg": "config",
    ".ini": "config",
    ".conf": "config",
}

#: Reference kinds.
REFERENCE_KINDS: tuple[str, ...] = ("name", "path", "command", "argument", "enum_label")

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_LITERAL = re.compile(r"""^\s*(['"])(?P<value>[^'"]*)\1\s*$""")
_FSTRING = re.compile(r"""^\s*[fFrRbB]+['"]""")
_PROCEDURE = re.compile(
    r"^\s*(?P<verb>" + "|".join(PROCEDURE_VERBS) + r")\s+(?P<target>[A-Za-z_][A-Za-z0-9_/.]*)",
    re.IGNORECASE,
)
_SEND = re.compile(
    r"\b(?:SEND|EXECUTE)\s+(?P<command>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^)]*)\)"
)
_ARGUMENT = re.compile(r"(?P<name>" + _IDENTIFIER + r")\s*=")
_ENUM_LABEL = re.compile(r"""(?:==|!=|\bIS\b|\bEQ\b)\s*(['"])(?P<label>[A-Za-z_][\w ]*)\1""")
_CONFIG = re.compile(
    r"""^[\s"']*(?P<key>parameter|mnemonic|command|path)["']?\s*[:=]\s*"""
    r"""["']?(?P<value>[^\s,"'#\]}]+)""",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"(?:#|//).*$")


@dataclass(frozen=True)
class Reference:
    """One extracted reference to a dictionary item.

    ``target`` is the identifier as written. ``decidable`` is false when the
    reference is constructed at run time, in which case ``target`` holds the
    source expression and ``reason`` says why it could not be resolved.
    """

    source: str
    line: int
    kind: str
    text: str
    target: str
    decidable: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(
                f"Reference.kind must be one of {', '.join(REFERENCE_KINDS)}, got {self.kind!r}"
            )
        if self.line < 1:
            raise ValueError(f"Reference.line is 1-based, got {self.line}")
        if not self.decidable and not self.reason:
            raise ValueError("an undecidable reference must carry a reason")

    def one_line(self) -> str:
        """``source:line kind target`` on a single line."""
        mark = "" if self.decidable else "  UNDECIDABLE: " + self.reason
        return f"{self.source}:{self.line:<4d} {self.kind:<11s} {self.target}{mark}"


@dataclass(frozen=True)
class CorpusSource:
    """One corpus file: a label, a kind and its text. No path is stored."""

    label: str
    kind: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(
                f"CorpusSource.kind must be one of {', '.join(SOURCE_KINDS)}, got {self.kind!r}"
            )
        if not self.label:
            raise ValueError("CorpusSource.label must be non-empty")


@dataclass(frozen=True)
class Corpus:
    """A set of corpus sources, plus the labels of sources that could not be read."""

    sources: tuple[CorpusSource, ...] = ()
    unreadable: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.sources)

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels of the readable sources, in order."""
        return tuple(source.label for source in self.sources)


def _strip_comment(line: str, kind: str) -> str:
    if kind == "procedure":
        return line
    return _COMMENT.sub("", line)


def _balanced_argument(text: str, start: int, opener: str) -> tuple[str, int] | None:
    """Return the text inside the bracket opened at ``start``, and the index after it."""
    closer = {"[": "]", "(": ")"}[opener]
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
            if depth == 0:
                return (text[start + 1 : index], index + 1)
    _ = closer
    return None


def _classify_expression(expression: str) -> tuple[str, bool, str]:
    """Read an accessor argument. Returns ``(target, decidable, reason)``."""
    literal = _LITERAL.match(expression)
    if literal is not None:
        return (literal.group("value"), True, "")
    stripped = expression.strip()
    if not stripped:
        return (expression.strip(), False, "empty accessor argument")
    if _FSTRING.match(stripped):
        return (stripped, False, "f-string or prefixed literal built at run time")
    if any(token in stripped for token in ("+", "%", ".format", ".join", "*")):
        return (stripped, False, "name assembled by a run-time expression")
    if re.fullmatch(_IDENTIFIER + r"(\[[^\]]*\])?(\.\w+)*", stripped):
        return (stripped, False, "name held in a variable, unknown statically")
    return (stripped, False, "accessor argument is not a string literal")


def extract_from_source(
    source: CorpusSource, *, accessors: tuple[str, ...] = DEFAULT_ACCESSORS
) -> tuple[Reference, ...]:
    """Extract every reference from one corpus source."""
    references: list[Reference] = []
    accessor_pattern = re.compile(
        r"\b(?P<accessor>"
        + "|".join(sorted(accessors, key=len, reverse=True))
        + r")\s*(?P<open>[\[(])"
    )
    for number, raw_line in enumerate(source.text.splitlines(), start=1):
        line = _strip_comment(raw_line, source.kind)
        text = line.strip()
        if not text:
            continue

        for match in accessor_pattern.finditer(line):
            inside = _balanced_argument(line, match.start("open"), match.group("open"))
            if inside is None:
                continue
            expression = inside[0].split(",")[0]
            target, decidable, reason = _classify_expression(expression)
            if match.group("accessor") in COMMAND_ACCESSORS:
                kind = "command"
            elif "/" in target and decidable:
                kind = "path"
            else:
                kind = "name"
            references.append(
                Reference(
                    source=source.label,
                    line=number,
                    kind=kind,
                    text=text,
                    target=target,
                    decidable=decidable,
                    reason=reason,
                )
            )
            if kind == "command" and decidable:
                for argument in _ARGUMENT.finditer(inside[0]):
                    references.append(
                        Reference(
                            source=source.label,
                            line=number,
                            kind="argument",
                            text=text,
                            target=f"{target}.{argument.group('name')}",
                        )
                    )

        procedure = _PROCEDURE.match(line)
        if procedure is not None:
            target = procedure.group("target")
            references.append(
                Reference(
                    source=source.label,
                    line=number,
                    kind="path" if "/" in target else "name",
                    text=text,
                    target=target,
                )
            )

        for match in _SEND.finditer(line):
            command = match.group("command")
            references.append(
                Reference(
                    source=source.label, line=number, kind="command", text=text, target=command
                )
            )
            for argument in _ARGUMENT.finditer(match.group("args")):
                references.append(
                    Reference(
                        source=source.label,
                        line=number,
                        kind="argument",
                        text=text,
                        target=f"{command}.{argument.group('name')}",
                    )
                )

        for match in _ENUM_LABEL.finditer(line):
            references.append(
                Reference(
                    source=source.label,
                    line=number,
                    kind="enum_label",
                    text=text,
                    target=match.group("label"),
                )
            )

        config = _CONFIG.match(line)
        if config is not None:
            key = config.group("key").lower()
            value = config.group("value")
            kind = "command" if key == "command" else ("path" if key == "path" else "name")
            if "/" in value and kind == "name":
                kind = "path"
            references.append(
                Reference(source=source.label, line=number, kind=kind, text=text, target=value)
            )

    deduplicated: list[Reference] = []
    seen: set[tuple[str, int, str, str]] = set()
    for reference in references:
        key = (reference.source, reference.line, reference.kind, reference.target)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(reference)
    return tuple(deduplicated)


def extract_references(
    corpus: Corpus, *, accessors: tuple[str, ...] = DEFAULT_ACCESSORS
) -> tuple[Reference, ...]:
    """Extract every reference from every source in ``corpus``."""
    if not isinstance(corpus, Corpus):
        raise TypeError(f"extract_references expects a Corpus, got {type(corpus).__name__}")
    out: list[Reference] = []
    for source in corpus.sources:
        out.extend(extract_from_source(source, accessors=accessors))
    return tuple(out)


def load_corpus(directory: str | Path) -> Corpus:
    """Read every recognised file under ``directory`` into a :class:`Corpus`.

    The label of each source is its path relative to ``directory``, so nothing
    this function produces carries an absolute path. A file whose suffix is not
    in :data:`SUFFIX_KINDS`, or that cannot be decoded as UTF-8, is recorded in
    :attr:`Corpus.unreadable` and counted in the coverage verdict.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{root.as_posix()} is not a directory")
    sources: list[CorpusSource] = []
    unreadable: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        label = path.relative_to(root).as_posix()
        kind = SUFFIX_KINDS.get(path.suffix.lower())
        if kind is None:
            unreadable.append(f"{label} (suffix {path.suffix or 'none'} not recognised)")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            unreadable.append(f"{label} ({type(error).__name__})")
            continue
        sources.append(CorpusSource(label=label, kind=kind, text=text))
    return Corpus(sources=tuple(sources), unreadable=tuple(unreadable))
