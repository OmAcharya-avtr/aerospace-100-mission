"""A heuristic for whether a test function asserts anything.

This module is the one part of the product that can be wrong about a correct
input, and it is labelled as a heuristic everywhere it is used.  It exists
because the central honest claim of this package is that traceability is
bookkeeping: a requirement traced to a test that asserts nothing is still
traced, and the audit should at least be able to *suspect* that case.

The rules, in full, applied to the body of one function definition as parsed
by :mod:`ast`:

=====  =========================================================  ===========
Rule   Condition                                                  Verdict
=====  =========================================================  ===========
R1     an ``assert`` statement whose test is not a constant        asserts
R2     an ``assert`` statement whose test is a truthy constant     ignored
       (``assert True``, ``assert 1``, ``assert "x"``)
R3     ``with pytest.raises(...)`` / ``pytest.warns(...)`` /       asserts
       ``pytest.deprecated_call(...)``
R4     a call whose final dotted name matches ``^assert``          asserts
       (``self.assertEqual``, ``np.testing.assert_allclose``,
       ``assert_that``)
R5     ``pytest.fail(...)`` or ``self.fail(...)``                  asserts
R6     none of the above                                           suspected
                                                                   empty
=====  =========================================================  ===========

The rules are applied to the function's own body only, including nested
``with``, ``for``, ``if`` and ``try`` blocks, and including nested function
definitions inside the test.  **Calls are not followed.**  A test whose
assertion lives in a module-level helper or in a fixture is therefore a false
positive of R6, and that is the dominant error mode.  The false-negative mode
is a non-constant assertion that is nonetheless vacuous — ``assert x == x``,
``assert len(r) >= 0`` — which no syntactic rule can catch.

Both error rates are measured on a hand-labelled fixture corpus; see
``validation/validate_heuristic.py`` and section 5 of
``validation/VALIDATION.md``.  **Do not treat the verdict as evidence.**
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

RAISES_NAMES = frozenset({"raises", "warns", "deprecated_call"})
FAIL_NAMES = frozenset({"fail"})


@dataclass(frozen=True)
class AssertionVerdict:
    """Result of the heuristic for one function.

    Attributes
    ----------
    function:
        Name of the function inspected.
    file:
        File the function was read from.
    line:
        One-based line of the ``def``.
    has_assertion:
        True when any of R1, R3, R4 or R5 fired.
    rules_fired:
        Rule labels that fired, in the order first seen.
    detail:
        Short human-readable reason.
    """

    function: str
    file: str
    line: int
    has_assertion: bool
    rules_fired: tuple[str, ...]
    detail: str


def _dotted_tail(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_truthy_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    return False


def inspect_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef, *, file: str
) -> AssertionVerdict:
    """Apply the rules of this module to one parsed function definition."""
    fired: list[str] = []
    details: list[str] = []

    def note(rule: str, text: str) -> None:
        if rule not in fired:
            fired.append(rule)
            details.append(text)

    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            if _is_truthy_constant(node.test):
                note("R2", "constant-true assert ignored")
            else:
                note("R1", "assert statement")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and _dotted_tail(call.func) in RAISES_NAMES:
                    note("R3", f"with ...{_dotted_tail(call.func)}(...)")
        elif isinstance(node, ast.Call):
            tail = _dotted_tail(node.func)
            if tail.startswith("assert"):
                note("R4", f"call to {tail}")
            elif tail in FAIL_NAMES:
                note("R5", f"call to {tail}")

    positive = [r for r in fired if r in ("R1", "R3", "R4", "R5")]
    if positive:
        return AssertionVerdict(
            function=func.name,
            file=file,
            line=func.lineno,
            has_assertion=True,
            rules_fired=tuple(fired),
            detail="; ".join(details),
        )
    note("R6", "no assert, no pytest.raises, no assert* call")
    return AssertionVerdict(
        function=func.name,
        file=file,
        line=func.lineno,
        has_assertion=False,
        rules_fired=tuple(fired),
        detail="; ".join(details),
    )


def inspect_source(text: str, *, file: str = "<string>") -> dict[str, AssertionVerdict]:
    """Run the heuristic over every top-level and method function in a source file.

    Returns
    -------
    dict
        Keyed by function name.  When two functions in one file share a name
        (a method and a free function, say) the first wins and the second is
        unreachable through this mapping; that collision is reported by
        :func:`index_test_files` as a note rather than silently resolved.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"{file} is not parseable Python: {exc}") from exc
    out: dict[str, AssertionVerdict] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, inspect_function(node, file=file))
    return out


def inspect_file(path: str | Path) -> dict[str, AssertionVerdict]:
    """Run the heuristic over one Python file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"test source file not found: {p}")
    return inspect_source(p.read_text(encoding="utf-8"), file=str(p))


def index_test_files(
    root: str | Path, *, pattern: str = "test_*.py"
) -> tuple[dict[tuple[str, str], AssertionVerdict], tuple[str, ...]]:
    """Index every matching test file under ``root``.

    Each verdict's :attr:`AssertionVerdict.file` is recorded **relative to**
    ``root``, so that nothing downstream prints the absolute layout of the
    machine the audit ran on.

    Returns
    -------
    index:
        Mapping ``(file_stem, function_name) -> AssertionVerdict`` plus
        ``("", function_name)`` fallbacks for unambiguous names.
    notes:
        Messages about files that could not be parsed or names that collide
        across files, so an ambiguity is reported rather than guessed.
    """
    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"test root is not a directory: {base}")
    index: dict[tuple[str, str], AssertionVerdict] = {}
    seen_names: dict[str, int] = {}
    notes: list[str] = []
    for path in sorted(base.rglob(pattern)):
        relative = path.relative_to(base).as_posix()
        try:
            verdicts = {
                name: AssertionVerdict(
                    function=v.function, file=relative, line=v.line,
                    has_assertion=v.has_assertion, rules_fired=v.rules_fired,
                    detail=v.detail,
                )
                for name, v in inspect_file(path).items()
            }
        except ValueError as exc:
            notes.append(str(exc).replace(str(path), relative))
            continue
        for name, verdict in verdicts.items():
            index[(path.stem, name)] = verdict
            seen_names[name] = seen_names.get(name, 0) + 1
    for name, count in seen_names.items():
        if count == 1:
            for (stem, fname), verdict in list(index.items()):
                if fname == name:
                    index[("", name)] = verdict
                    _ = stem
        elif name.startswith("test"):
            # Only a test name can ever be looked up by name alone, so a
            # collision among helpers or fixtures is not worth a note.
            notes.append(f"function name '{name}' occurs in {count} files; file stem required")
    return index, tuple(notes)
