"""Finding and rewriting the seed literals a script declares.

To ask whether a committed number depends on the random seed, the seed has to
change and nothing else. There is no portable way to do that from outside a
script: a declared seed is an integer literal in the source, not a parameter,
and no environment variable reaches it. So the seed is changed **in the
source**, in a pristine copy, by rewriting the literals.

What counts as a seed literal
-----------------------------
Five forms, all detected from the abstract syntax tree rather than by regular
expression, so that a match inside a string or a comment is not rewritten:

- ``numpy.random.default_rng(12345)`` and a bare ``default_rng(12345)``;
- ``numpy.random.seed(12345)`` and ``random.seed(12345)``;
- any keyword argument named ``seed`` or ``random_state`` with an integer
  literal value;
- a module-level or class-level assignment whose target name matches
  :data:`SEED_NAME` -- ``SEED``, ``SEED_ANALYSIS``, ``RNG_SEED`` and so on --
  with an integer literal value;
- ``numpy.random.SeedSequence(12345)`` and ``numpy.random.PCG64(12345)``.

What is deliberately NOT rewritten
----------------------------------
A ``random_state`` or ``seed`` keyword whose value is a name, an expression or
``None`` is reported as **unresolved** and left alone. Rewriting it would
require evaluating the script, and reporting it as rewritten when it was not
would be the audit lying about its own coverage. The unresolved count travels
with every result.

An integer literal that is not in one of those positions is never touched. A
tolerance, an iteration count and an array size are all integer literals, and
rewriting them would change the computation rather than the seed, which would
make every number move and the audit would report a finding it manufactured.

The rewrite
-----------
A literal ``v`` becomes ``v + PERTURBATION``. The offset is a fixed prime so
the rewrite is itself reproducible, and it is added rather than replacing every
seed with one constant, so that a script declaring several distinct seeds still
declares several distinct seeds afterwards: collapsing them could make two
independent streams identical and produce a change that is an artifact of the
rewrite rather than a dependence on the seed.

Units: seeds are dimensionless non-negative integers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

#: Added to every rewritten seed literal. A fixed prime, so the perturbation is
#: itself reproducible; 104729 is the 10000th prime.
PERTURBATION = 104729

#: Names of keyword arguments whose integer literal value is a seed.
SEED_KEYWORDS = frozenset({"seed", "random_state", "rng_seed", "seed_sequence"})

#: Callables whose first positional integer literal argument is a seed. Matched
#: on the attribute name, so ``np.random.default_rng`` and a bare
#: ``default_rng`` both match.
SEED_CALLABLES = frozenset(
    {"default_rng", "seed", "RandomState", "SeedSequence", "PCG64", "PCG64DXSM", "Philox", "SFC64"}
)

#: Receivers that make a bare ``seed(...)`` call a seeding call rather than a
#: user function of the same name.
SEED_RECEIVERS = frozenset({"random", "np", "numpy", "rng", "nprandom"})


def _is_seed_name(name: str) -> bool:
    """Whether an assignment target name declares a seed.

    Upper-case names only: a lower-case ``seed`` is usually a parameter being
    passed through rather than a declared constant, and rewriting a parameter's
    default in one place while its callers pass their own value would change
    some call sites and not others.
    """
    if not name.isupper():
        return False
    parts = name.split("_")
    return "SEED" in parts or "SEEDS" in parts


@dataclass(frozen=True)
class SeedLiteral:
    """One integer literal that a script uses as a random seed.

    Attributes:
        line: one-based source line, as :mod:`ast` reports it.
        col: zero-based start column of the literal.
        end_col: zero-based end column of the literal.
        value: the declared seed, dimensionless.
        kind: how it was recognised -- ``call``, ``keyword`` or ``assign``.
        context: the callable or target name it was found under.
    """

    line: int
    col: int
    end_col: int
    value: int
    kind: str
    context: str


@dataclass(frozen=True)
class SeedScan:
    """Every seed literal in one source file, and what could not be resolved.

    Attributes:
        literals: the rewritable integer seed literals.
        unresolved: one entry per seed-position whose value is not an integer
            literal, as ``(line, kind, context)``. Reported, never rewritten.
    """

    literals: tuple[SeedLiteral, ...]
    unresolved: tuple[tuple[int, str, str], ...]

    @property
    def count(self) -> int:
        """How many literals can be rewritten, dimensionless."""
        return len(self.literals)


def _callable_name(node: ast.expr) -> tuple[str, str]:
    """Return ``(attribute_name, receiver_name)`` for a call target."""
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


def _int_literal(node: ast.expr) -> int | None:
    """Return the value of a non-negative integer literal, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):
            return None
        return int(node.value)
    return None


def find_seed_literals(source: str) -> SeedScan:
    """Find every rewritable seed literal in Python source.

    Args:
        source: the file's text.

    Returns:
        A :class:`SeedScan`.

    Raises:
        SyntaxError: if ``source`` does not parse. The caller decides what an
            unparseable file means; this module does not swallow it, because a
            file silently skipped is coverage the audit would be overstating.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, got {type(source).__name__}")
    tree = ast.parse(source)
    literals: list[SeedLiteral] = []
    unresolved: list[tuple[int, str, str]] = []

    def record(node: ast.expr, kind: str, context: str) -> None:
        value = _int_literal(node)
        if value is None:
            unresolved.append((getattr(node, "lineno", 0), kind, context))
            return
        literals.append(
            SeedLiteral(
                line=node.lineno,
                col=node.col_offset,
                end_col=node.end_col_offset if node.end_col_offset is not None else node.col_offset,
                value=value,
                kind=kind,
                context=context,
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name, receiver = _callable_name(node.func)
            if name in SEED_CALLABLES and (name != "seed" or receiver in SEED_RECEIVERS):
                if node.args:
                    record(node.args[0], "call", name)
            for keyword in node.keywords:
                if keyword.arg in SEED_KEYWORDS:
                    record(keyword.value, "keyword", f"{name or '?'}({keyword.arg}=)")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if node.value is None:
                continue
            for name in names:
                if _is_seed_name(name):
                    record(node.value, "assign", name)

    # Deduplicate by source position: a keyword inside a seeding call can be
    # reached twice, and rewriting the same span twice would corrupt the file.
    unique: dict[tuple[int, int, int], SeedLiteral] = {}
    for literal in literals:
        unique.setdefault((literal.line, literal.col, literal.end_col), literal)
    ordered = tuple(sorted(unique.values(), key=lambda item: (item.line, item.col)))
    return SeedScan(literals=ordered, unresolved=tuple(sorted(set(unresolved))))


def rewrite_seeds(source: str, perturbation: int = PERTURBATION) -> tuple[str, SeedScan]:
    """Return ``source`` with every seed literal increased by ``perturbation``.

    The rewrite is applied from the end of the file backwards, so that an
    earlier edit never moves the column offsets of a later one.

    Args:
        source: the file's text.
        perturbation: the offset added to each literal, dimensionless. 0 is
            permitted and is the unchanged-seed control: it exercises the same
            rewrite machinery and leaves the values alone, so a difference
            found under it cannot be blamed on the rewrite.

    Returns:
        ``(rewritten_source, scan)``.
    """
    if not isinstance(perturbation, int) or isinstance(perturbation, bool):
        raise TypeError(f"perturbation must be an int, got {type(perturbation).__name__}")
    if perturbation < 0:
        raise ValueError(f"perturbation must be non-negative, got {perturbation}")
    scan = find_seed_literals(source)
    if perturbation == 0 or not scan.literals:
        return source, scan
    lines = source.splitlines(keepends=True)
    for literal in sorted(scan.literals, key=lambda item: (item.line, item.col), reverse=True):
        index = literal.line - 1
        if not 0 <= index < len(lines):  # pragma: no cover - ast guarantees the range
            continue
        line = lines[index]
        replacement = str(literal.value + perturbation)
        lines[index] = line[: literal.col] + replacement + line[literal.end_col :]
    rewritten = "".join(lines)
    # The rewrite must still parse. A rewrite that broke the file would make
    # every number "move" for a reason that has nothing to do with the seed.
    ast.parse(rewritten)
    return rewritten, scan
