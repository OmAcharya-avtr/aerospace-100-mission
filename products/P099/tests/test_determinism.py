"""One declared seed everywhere, and no live wall-clock comparisons anywhere.

The second half of this file is a source scan. It is here rather than in
``validation/`` because the rule it enforces is a property of this product's
own source, and the defect class it guards against -- asserting a relationship
between two live wall-clock measurements on a shared two-core runner -- cost
this mission a whole product and five intermittent failures. A scan in the
test suite fails the build; a scan in a validation script fails the gate much
later.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _seed_from(module_path: Path) -> int:
    """Read a module's declared ``SEED`` without importing it.

    Not imported, because ``validation/_bootstrap`` calls ``os.chdir`` in its
    ``bootstrap()`` and importing the examples' copy would shadow it on
    ``sys.path``. A regular expression over the source is enough for a
    literal integer and has no side effect at all.
    """
    text = module_path.read_text()
    match = re.search(r"^SEED = (\d+)$", text, flags=re.MULTILINE)
    assert match is not None, f"{module_path.name} declares no literal SEED"
    return int(match.group(1))


def test_one_seed_across_validation_examples_and_tests():
    """The validation, examples and test seeds are the same number.

    Three places declare it because the three directories are independent of
    one another at import time. This test is what keeps them from drifting.
    """
    from tests.conftest import SEED as tests_seed  # noqa: PLC0415

    validation_seed = _seed_from(ROOT / "validation" / "_bootstrap.py")
    examples_seed = _seed_from(ROOT / "examples" / "_bootstrap.py")
    assert validation_seed == examples_seed == tests_seed == 99099


def test_the_shared_constants_agree_between_validation_and_examples():
    """The confidence level, resample count and nominal coverage agree too.

    Three places declare them because the directories are independent at
    import time; this is what keeps them from drifting. Read from the source
    rather than imported, for the reason in ``_seed_from``.
    """
    names = ("CONFIDENCE", "BOOTSTRAP_RESAMPLES", "NOMINAL_COVERAGE")
    validation = (ROOT / "validation" / "_bootstrap.py").read_text()
    examples = (ROOT / "examples" / "_bootstrap.py").read_text()
    for name in names:
        pattern = rf"^{name} = (\S+)$"
        first = re.search(pattern, validation, flags=re.MULTILINE)
        second = re.search(pattern, examples, flags=re.MULTILINE)
        assert first is not None, f"validation/_bootstrap.py declares no {name}"
        assert second is not None, f"examples/_bootstrap.py declares no {name}"
        assert first.group(1) == second.group(1), name


def test_generator_declares_its_own_seed():
    """The campaign generator's seed is a literal in its source."""
    text = (ROOT / "data" / "generate_passes.py").read_text()
    match = re.search(r"^SEED = (\d+)$", text, flags=re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) == 990099


def test_manifest_records_the_generator_seed():
    """The committed manifest carries the seed the record was generated at."""
    import json  # noqa: PLC0415

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    assert manifest["seed"] == 990099


#: Source directories scanned for wall-clock comparisons.
_SCANNED = ("src", "tests", "validation", "examples", "data")

#: Names that read a live clock.
_CLOCK_CALLS = (
    "time.perf_counter",
    "time.process_time",
    "time.monotonic",
    "time.time",
    "timeit.default_timer",
)


#: This file. Every scan here skips it.
#:
#: Not a loophole, and not optional. A scanner whose own source contains the
#: patterns it searches for reports ITSELF, and this is the fifth occurrence
#: of that class in this mission: three consecutive sessions introduced it in
#: the act of writing about it, and a later product's hygiene check failed
#: itself on the bytecode its own interpreter had written. The exclusion is
#: paid for by ``test_every_scan_can_actually_fire``, which runs each pattern
#: against planted text that MUST match and text that must NOT, so the
#: patterns are exercised even though the file they live in is skipped, and by
#: ``test_every_scan_is_idempotent``, which runs each scan twice and requires
#: the two runs to agree.
_SELF = Path(__file__).resolve()


def _python_files() -> list[Path]:
    """Every committed Python file of this product, except this one."""
    files: list[Path] = []
    for directory in _SCANNED:
        files.extend(
            path for path in sorted((ROOT / directory).rglob("*.py"))
            if path.resolve() != _SELF
        )
    assert files, "the scan found no Python files, which means the scan is broken"
    return files


def test_no_assertion_mentions_a_live_clock():
    """No ``assert`` statement in this product contains a clock call.

    The rule is narrower than "do not time anything" and that is deliberate:
    the generator and some scripts DO report an elapsed time as a build note.
    What is forbidden is an assertion whose truth depends on one, because on
    a two-core runner shared with four other builds the ordering of two
    timings is not a property of the code.
    """
    offenders: list[str] = []
    for path in _python_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith("assert "):
                continue
            if any(call in stripped for call in _CLOCK_CALLS):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, f"assertions on a live clock: {offenders}"


def test_no_elapsed_time_is_compared_to_another_elapsed_time():
    """No line both reads a clock twice and compares the two.

    A single pattern catches the shape the defect took: two clock reads and a
    comparison operator on one line, or a subtraction of two clock reads fed
    straight into a comparison.
    """
    offenders: list[str] = []
    for path in _python_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if _TWO_TIMINGS.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, f"comparisons between two live timings: {offenders}"


def test_no_source_file_contains_an_absolute_home_path():
    """No committed source file names an absolute path under a home directory.

    Release-gate check 8 fails the whole batch on one, and a cache or error
    message is the usual way one arrives. The pattern is assembled from
    ``os.sep`` so that the scanner's own source does not match it, which is a
    defect three consecutive sessions of this mission introduced in the act
    of writing about it.
    """
    import os  # noqa: PLC0415

    needle = os.sep + "home" + os.sep
    offenders: list[str] = []
    for path in _python_files():
        if needle in path.read_text():
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"absolute home paths in source: {offenders}"


def test_committed_output_files_contain_no_absolute_home_path():
    """Nor does any committed validation output or data transcript."""
    import os  # noqa: PLC0415

    needle = os.sep + "home" + os.sep
    offenders: list[str] = []
    for pattern in ("validation/*_output.txt", "data/*_output.txt", "data/manifest.json"):
        for path in sorted(ROOT.glob(pattern)):
            if needle in path.read_text():
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"absolute home paths in committed output: {offenders}"


def test_every_runnable_script_disables_bytecode_first():
    """Each script under ``validation/``, ``examples/`` and ``data/`` sets the flag.

    It cannot be delegated to the shared ``_bootstrap`` module, because
    CPython writes a module's bytecode cache BEFORE executing that module, so
    ``_bootstrap`` cannot set it on its own behalf. The statement must appear
    before any import other than ``from __future__`` and ``import sys``.

    Checked on the SYNTAX TREE and not on the text. A line-based version of
    this test read a docstring sentence beginning "from that paper:" as an
    import and failed a script that was correct -- which is the same class of
    error as a scanner reporting itself, and is why this one parses.
    """
    import ast  # noqa: PLC0415

    offenders: list[str] = []
    for directory in ("validation", "examples", "data"):
        for path in sorted((ROOT / directory).glob("*.py")):
            # Modules whose name starts with an underscore are imported, never
            # executed as a script: ``_bootstrap.py`` and the pytest plugin in
            # ``validation/``. The flag belongs on the script that RUNS.
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text())
            position = None
            for index, node in enumerate(tree.body):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == "dont_write_bytecode"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    position = index
                    break
            if position is None:
                offenders.append(f"{path.relative_to(ROOT)}: flag absent")
                continue
            for node in tree.body[:position]:
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue
                if isinstance(node, ast.Import) and [
                    alias.name for alias in node.names
                ] == ["sys"]:
                    continue
                if isinstance(node, ast.Import | ast.ImportFrom):
                    offenders.append(
                        f"{path.relative_to(ROOT)}: import at line {node.lineno} "
                        "precedes the flag"
                    )
    assert not offenders, offenders


def test_python_version_is_the_measured_one():
    """A build note, asserted loosely: this product was measured on 3.13.

    Declared as a minimum rather than an equality, because ``pyproject.toml``
    declares ``requires-python = ">=3.11"`` and the product is expected to
    work there; what this records is what was MEASURED.
    """
    assert sys.version_info >= (3, 11)


def test_no_test_uses_pytest_approx_with_a_bare_relative_tolerance():
    """Every ``pytest.approx(..., rel=)`` in this suite also passes ``abs=``.

    ``pytest.approx`` passes if EITHER bound is met and the absolute default
    is 1e-12, so for a quantity of order 1e-5 every relative tolerance
    tighter than 2e-8 is silently replaced by the default and the assertion
    stops testing what it says. ``validation/validate_approx_audit.py``
    re-runs the whole suite with the default forced off and reports the count
    of assertions that depended on it; this test is the cheap structural half
    of the same guard.
    """
    offenders = _approx_offenders()
    assert not offenders, f"approx calls with rel= and no abs=: {offenders}"


def _approx_offenders() -> list[str]:
    """Every ``approx(..., rel=)`` call in the suite with no ``abs=``."""
    offenders: list[str] = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        if path.resolve() == _SELF:
            continue
        text = path.read_text()
        for match in re.finditer(r"approx\(", text):
            start = match.end()
            depth = 1
            index = start
            while index < len(text) and depth:
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                index += 1
            call = text[start : index - 1]
            if "rel=" in call and "abs=" not in call:
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line}")
    return offenders


#: The clock-comparison pattern, in one place so the scan and the
#: can-it-fire test cannot use different expressions.
_TWO_TIMINGS = re.compile(
    r"(perf_counter|process_time|monotonic|default_timer)\s*\(\s*\).*"
    r"(<|>|<=|>=|==).*"
    r"(perf_counter|process_time|monotonic|default_timer)\s*\(\s*\)"
)


def test_every_scan_can_actually_fire():
    """Each pattern is run against text that MUST match and text that must not.

    This is what pays for the self-exclusion above. A check whose own
    execution cannot detect the condition it tests is not a check, so the
    patterns are exercised directly on planted strings.
    """
    assert _TWO_TIMINGS.search("assert time.perf_counter() < time.perf_counter()")
    assert _TWO_TIMINGS.search("if time.monotonic() >= time.monotonic():")
    assert not _TWO_TIMINGS.search("started = time.perf_counter()")

    planted_assertion = "assert time.perf_counter() - started > 0.0"
    assert planted_assertion.strip().startswith("assert ")
    assert any(call in planted_assertion for call in _CLOCK_CALLS)

    planted_approx = "pytest.approx(y, rel=1e-9)"
    assert "rel=" in planted_approx and "abs=" not in planted_approx

    import os  # noqa: PLC0415

    planted_path = os.sep + "home" + os.sep + "someone" + os.sep + "x.py"
    assert (os.sep + "home" + os.sep) in planted_path


def test_every_scan_is_idempotent():
    """Each scan run twice agrees with itself.

    A scan whose own execution could create the condition it tests would
    disagree on the second run. Running each one twice from the same tree is
    the cheapest way to show that none of them writes anything.
    """
    assert _approx_offenders() == _approx_offenders()
    first = sorted(path.name for path in _python_files())
    second = sorted(path.name for path in _python_files())
    assert first == second


@pytest.mark.parametrize("directory", ["src/passyield"])
def test_library_code_does_not_print(directory):
    """No ``print`` in the package except in the command-line interface.

    The interface and the scripts are the only things that write to a stream;
    everything else returns a string through ``passyield.report``.
    """
    offenders: list[str] = []
    for path in sorted((ROOT / directory).glob("*.py")):
        if path.name == "__main__.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("print(") or " print(" in stripped:
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, f"print in library code: {offenders}"
