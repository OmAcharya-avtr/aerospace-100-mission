"""Static detection of the self-polluting-check class.

A check whose own execution can create the condition it tests is not a check.
This module finds the static form of that defect: **a script that writes under a
path it also reads or globs.**

Real instances from this mission's history, which is the whole motivation:

- a validation probe writing its output into the directory it then scans for
  stray files;
- a ``ruff`` invocation writing its cache inside the product tree the audit then
  reports as dirty;
- a script whose first run creates the artifact whose absence it asserts.

How a path is modelled
----------------------
Paths in real scripts are expressions, not literals: ``ROOT / "validation" /
f"{stem}_output.txt"``. So each read, glob and write target is reduced to a
:class:`PathExpr`, a tuple of segments where a segment is either a literal
string, a **symbol** standing for a name the analyser cannot evaluate
(``$ROOT``), or :data:`WILDCARD` standing for a segment it cannot resolve at
all. Two symbols with the same name are the same segment, which is what makes
``ROOT / "validation"`` and ``ROOT / "validation" / name`` comparable.

Containment is prefix containment with wildcards matching anything, so the
analyser **over-flags rather than under-flags**: an unresolvable path becomes a
wildcard and a wildcard is under everything. That is a deliberate choice and it
has a measured cost -- the false-positive rate on the deterministic negative
controls of the injected corpus is measured and published rather than tuned to
zero. A detector for this class that is tuned until it is quiet is the defect it
is looking for, one level up.

What this cannot do
-------------------
It has no execution order and no reachability: a write guarded by a branch that
never runs still counts, and a read that happens before the write in source
order may happen after it at run time. It cannot see through a path handed in
from outside the module. It is therefore a **screen**, and
:mod:`seedaudit.dynamic` is the confirmation; the product reports both and never
only one.

Units: none. Every quantity here is a count or a source position.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

#: A path segment the analyser cannot resolve. It matches any single segment.
WILDCARD = "*?*"

#: Call names that read an existing path. ``open`` is handled separately
#: because its mode argument decides the direction.
READ_CALLS = frozenset(
    {
        "read_text",
        "read_bytes",
        "load",
        "loadtxt",
        "genfromtxt",
        "read_csv",
        "read_json",
        "is_file",
        "is_dir",
        "exists",
        "stat",
        "read_yaml",
        "safe_load",
    }
)

#: Call names that enumerate a directory. The receiver or first argument is the
#: scan ROOT, and anything written under it is a candidate self-pollution.
GLOB_CALLS = frozenset({"glob", "rglob", "iterdir", "walk", "listdir", "iglob", "scandir"})

#: Call names that create or modify a path.
WRITE_CALLS = frozenset(
    {
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "touch",
        "savefig",
        "to_csv",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "unlink",
        "rmtree",
        "rename",
        "symlink_to",
        "hardlink_to",
        "chmod",
    }
)

#: Calls whose SECOND positional argument is the destination.
WRITE_SECOND_ARG_CALLS = frozenset({"copy", "copy2", "copyfile", "copytree", "move"})

#: Calls whose path is the FIRST POSITIONAL ARGUMENT whatever the receiver is.
#: ``fig.savefig(path)`` and ``os.listdir(path)`` have a receiver that is not
#: the path, unlike ``Path(p).glob(...)`` where the receiver IS the path.
#: Getting this wrong made the screen report the receiver's variable name as a
#: path and lose the real finding, so the two groups are separated by name.
FIRST_ARG_PATH_CALLS = frozenset(
    {
        "savefig",
        "to_csv",
        "to_json",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "makedirs",
        "listdir",
        "walk",
        "scandir",
        "iglob",
        "load",
        "loadtxt",
        "genfromtxt",
        "read_csv",
        "read_json",
        "safe_load",
    }
)

#: Receiver names that make a ``glob(...)`` call the module-level function,
#: whose path is its first argument rather than its receiver.
GLOB_MODULE_RECEIVERS = frozenset({"glob", "globlib"})

#: Modes that make ``open`` a write.
_WRITE_MODES = ("w", "a", "x", "+")


@dataclass(frozen=True)
class PathUse:
    """One use of a path in a script.

    Attributes:
        kind: ``read``, ``glob`` or ``write``.
        expr: the path, as a tuple of segments.
        line: one-based source line.
        call: the callable name it was derived from, for the report.
    """

    kind: str
    expr: tuple[str, ...]
    line: int
    call: str

    @property
    def rendered(self) -> str:
        """The path expression as a readable string."""
        return "/".join(self.expr) if self.expr else "."


@dataclass(frozen=True)
class PollutionFinding:
    """A write that lands under a path the same script reads or enumerates.

    Attributes:
        write: the write use.
        observe: the read or glob use whose root contains it.
        subkind: ``write_under_glob`` when the observation enumerates a
            directory, ``write_to_read_path`` when it reads the very path
            written, and ``write_under_read`` otherwise.
    """

    write: PathUse
    observe: PathUse
    subkind: str

    def describe(self) -> str:
        """One line naming both source positions and both paths."""
        return (
            f"{self.subkind}: writes {self.write.rendered!r} at line {self.write.line}"
            f" via {self.write.call}, {self.observe.kind}s {self.observe.rendered!r}"
            f" at line {self.observe.line} via {self.observe.call}"
        )


def _segments(node: ast.expr) -> tuple[str, ...]:
    """Reduce a path expression to a tuple of segments.

    Handles string literals, ``pathlib`` division, ``Path(...)``,
    ``os.path.join``, f-strings with literal parts, and name references. An
    expression it cannot interpret becomes a single :data:`WILDCARD`.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parts = [p for p in node.value.replace("\\", "/").split("/") if p not in ("", ".")]
        return tuple(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _segments(node.left) + _segments(node.right)
    if isinstance(node, ast.Call):
        name, _ = _callable_name(node.func)
        if name in ("Path", "PurePath", "PosixPath", "resolve", "absolute", "expanduser"):
            if node.args:
                return _segments(node.args[0])
            if isinstance(node.func, ast.Attribute):
                return _segments(node.func.value)
            return (WILDCARD,)
        if name == "join":
            out: tuple[str, ...] = ()
            for arg in node.args:
                out += _segments(arg)
            return out
        if name in ("str", "fspath"):
            return _segments(node.args[0]) if node.args else (WILDCARD,)
        return (WILDCARD,)
    if isinstance(node, ast.Name):
        return (f"${node.id}",)
    if isinstance(node, ast.Attribute):
        if node.attr in ("parent", "parents"):
            return _segments(node.value) + (WILDCARD,)
        return (f"${node.attr}",)
    if isinstance(node, ast.JoinedStr):
        out = ()
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out += _segments(value)
            else:
                out += (WILDCARD,)
        return out
    return (WILDCARD,)


def _callable_name(node: ast.expr) -> tuple[str, str]:
    """Return ``(attribute_or_function_name, receiver_name)`` for a call target."""
    if isinstance(node, ast.Attribute):
        receiver = node.value
        if isinstance(receiver, ast.Name):
            return node.attr, receiver.id
        if isinstance(receiver, ast.Attribute):
            return node.attr, receiver.attr
        return node.attr, ""
    if isinstance(node, ast.Name):
        return node.id, ""
    return "", ""


def _open_is_write(node: ast.Call) -> bool:
    """Whether an ``open``/``Path.open`` call opens for writing."""
    mode: str | None = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        if isinstance(node.args[1].value, str):
            mode = node.args[1].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                mode = keyword.value.value
    if mode is None:
        return False
    return any(flag in mode for flag in _WRITE_MODES)


def _receiver_name(node: ast.Call) -> str:
    """The name of the object a method was called on, or the empty string."""
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return ""


def _path_argument(node: ast.Call, receiver: ast.expr | None, name: str) -> ast.expr | None:
    """Pick the expression that denotes the path for one call.

    For ``fig.savefig(path)`` and ``numpy.save(path, array)`` the path is the
    first positional argument and the receiver is not a path at all. For
    ``Path(p).write_text(text)`` the receiver IS the path. The two groups are
    separated by name in :data:`FIRST_ARG_PATH_CALLS`, because treating a
    receiver as a path reported variable names as directories and buried the
    real findings underneath.
    """
    first = node.args[0] if node.args else None
    if name in FIRST_ARG_PATH_CALLS:
        return first if first is not None else receiver
    return receiver if receiver is not None else first


def _receiver_of(node: ast.Call) -> ast.expr | None:
    """The object a method was called on, if the call is a method call."""
    if isinstance(node.func, ast.Attribute):
        return node.func.value
    return None


def path_uses(source: str) -> tuple[PathUse, ...]:
    """Every read, glob and write path use in a script, in source order.

    Args:
        source: the file's text.

    Returns:
        The uses, ordered by source line then kind.

    Raises:
        SyntaxError: if ``source`` does not parse.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, got {type(source).__name__}")
    tree = ast.parse(source)
    uses: list[PathUse] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, _ = _callable_name(node.func)
        receiver = _receiver_of(node)
        line = node.lineno
        if name == "open":
            target = node.args[0] if node.args else receiver
            if target is None:
                continue
            kind = "write" if _open_is_write(node) else "read"
            uses.append(PathUse(kind=kind, expr=_segments(target), line=line, call="open"))
        elif name in GLOB_CALLS:
            first = node.args[0] if node.args else None
            if name in FIRST_ARG_PATH_CALLS:
                target = first if first is not None else receiver
            elif name == "glob" and _receiver_name(node) in GLOB_MODULE_RECEIVERS:
                target = first
            else:
                target = receiver if receiver is not None else first
            if target is None:
                continue
            uses.append(PathUse(kind="glob", expr=_segments(target), line=line, call=name))
        elif name in WRITE_SECOND_ARG_CALLS:
            if len(node.args) >= 2:
                uses.append(
                    PathUse(kind="write", expr=_segments(node.args[1]), line=line, call=name)
                )
            if node.args:
                uses.append(
                    PathUse(kind="read", expr=_segments(node.args[0]), line=line, call=name)
                )
        elif name in WRITE_CALLS:
            target = _path_argument(node, receiver, name)
            if target is None:
                continue
            uses.append(PathUse(kind="write", expr=_segments(target), line=line, call=name))
        elif name in READ_CALLS:
            target = _path_argument(node, receiver, name)
            if target is None:
                continue
            uses.append(PathUse(kind="read", expr=_segments(target), line=line, call=name))
    return tuple(sorted(uses, key=lambda use: (use.line, use.kind, use.rendered)))


def segments_match(left: str, right: str) -> bool:
    """Whether two path segments can denote the same thing."""
    return left == WILDCARD or right == WILDCARD or left == right


def is_under(root: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """Whether ``path`` can lie inside or at ``root``.

    An empty ``root`` is the current directory and contains everything. A
    :data:`WILDCARD` segment matches any single segment on either side, so the
    answer is "can", not "does": this is a screen and it is conservative by
    design.
    """
    if len(path) < len(root):
        return False
    return all(segments_match(a, b) for a, b in zip(root, path, strict=False))


def is_informative(expr: tuple[str, ...]) -> bool:
    """Whether a path expression can carry a containment claim.

    An expression that is empty, or every segment of which is a
    :data:`WILDCARD`, is **not** informative: as a root it contains everything
    and as a path it lies under everything, so a containment test involving it
    is true by vacuity rather than by evidence.

    Such uses are therefore skipped, and :func:`path_use_coverage` reports how
    many were skipped. Reporting a vacuous containment as a finding would be
    the opposite of conservatism -- it would be the screen manufacturing the
    result it is looking for, which is the defect class one level up.
    """
    if not expr:
        return False
    return any(segment != WILDCARD for segment in expr)


def path_use_coverage(source: str) -> dict[str, int]:
    """How many path uses the screen can and cannot reason about.

    Returns:
        ``{"uses", "informative", "vacuous", "reads", "globs", "writes"}``, all
        counts. ``vacuous`` is the part of the file the screen is blind to and
        it is published rather than implied.
    """
    uses = path_uses(source)
    informative = [use for use in uses if is_informative(use.expr)]
    return {
        "uses": len(uses),
        "informative": len(informative),
        "vacuous": len(uses) - len(informative),
        "reads": sum(use.kind == "read" for use in informative),
        "globs": sum(use.kind == "glob" for use in informative),
        "writes": sum(use.kind == "write" for use in informative),
    }


def find_self_pollution(source: str) -> tuple[PollutionFinding, ...]:
    """Find every write that lands under a path the same script reads or globs.

    Args:
        source: the file's text.

    Returns:
        The findings, deduplicated on the pair of source lines and the subkind.

    Raises:
        SyntaxError: if ``source`` does not parse.
    """
    uses = [use for use in path_uses(source) if is_informative(use.expr)]
    writes = [use for use in uses if use.kind == "write"]
    observers = [use for use in uses if use.kind in ("read", "glob")]
    seen: set[tuple[int, int, str]] = set()
    findings: list[PollutionFinding] = []
    for write in writes:
        for observe in observers:
            if not is_under(observe.expr, write.expr):
                continue
            if observe.kind == "glob":
                subkind = "write_under_glob"
            elif observe.expr == write.expr:
                subkind = "write_to_read_path"
            else:
                subkind = "write_under_read"
            key = (write.line, observe.line, subkind)
            if key in seen:
                continue
            seen.add(key)
            findings.append(PollutionFinding(write=write, observe=observe, subkind=subkind))
    return tuple(findings)
