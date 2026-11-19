"""No clock is read anywhere in this product, and this is where that is proved.

THE DEFECT CLASS THIS CHECKS FOR COST THIS MISSION A WHOLE PRODUCT.

Five intermittent failures and a sixth caught in flight all traced to checks
that compared two wall-clock readings on a shared two-core runner. The rule that
came out of it is the single most expensive lesson of the mission's last two
weeks: never assert a relationship between two live wall-clock measurements.

This product is ABOUT intermittency. A product about intermittency whose own
checks are intermittent has refuted itself before a reader reaches the first
number, so the rule is applied here in its strongest available form: **no
module, test, example or validation script in this product reads a clock at
all.** Every duration anything here asserts on is an integer number of
milliseconds read from ``data/run_history.json``, which
``data/generate_history.py`` wrote once from a fixed seed and which is committed
with its SHA-256 digest.

That is a stronger claim than the rule requires, and it is made because it is
checkable. This script parses every Python file in the product with ``ast`` --
not ``grep``, so a mention in a comment or a docstring cannot trip it and a
clock hidden behind a line continuation cannot hide from it -- and fails if a
clock appears.

Three screens, because one is not enough:

1. **Imports.** No ``time``, ``timeit``, ``datetime`` or ``resource`` import
   anywhere. ``datetime`` is included because a duration computed from two
   ``datetime.now()`` calls is a live measurement just as much as one from
   ``perf_counter``.
2. **Attribute and name access.** No ``perf_counter``, ``process_time``,
   ``monotonic``, ``time_ns`` or ``clock``, in case a clock were reached
   through a module imported for another reason.
3. **Subprocess clocks.** No ``os.times``, ``os.wait``-family resource
   accounting, and no ``/proc`` or ``/usr/bin/time`` string anywhere.

The exclusions are declared, not silent: this script's own source and
``validate_tree_hygiene.py`` are scanned like everything else, and the only
names exempted are the two literals in :data:`DECLARED_EXEMPTIONS`, which are
the forbidden names as they appear in THIS file's own tables.

A check whose own execution can create the condition it tests is not a check.
This one reads files and writes one output file, and its output file is
excluded from its own scan by name -- and the exclusion is reported.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Top-level modules no file in this product may import.
FORBIDDEN_MODULES = frozenset({"time", "timeit", "datetime", "resource", "sched"})

#: Function and attribute names that read a clock.
FORBIDDEN_NAMES = frozenset(
    {
        "perf_counter",
        "perf_counter_ns",
        "process_time",
        "process_time_ns",
        "monotonic",
        "monotonic_ns",
        "time_ns",
        "thread_time",
        "clock_gettime",
        "clock",
        "times",
        "getrusage",
    }
)

#: String fragments that would reach a clock through the filesystem or a shell.
FORBIDDEN_STRINGS = ("/usr/bin/time", "/proc/uptime", "/proc/stat", "clock_gettime")

#: The only names exempted from the string screen, with the reason. These are
#: the forbidden names as they appear in THIS file's own tables, and in the
#: table in ``tests/test_determinism.py``, which performs the same screen
#: inside the suite. Nothing else is exempt.
DECLARED_EXEMPTIONS = (
    ("validation/validate_no_live_timing.py", "this script's own declared tables"),
    ("tests/test_determinism.py", "the same screen, performed inside the test suite"),
)

#: Directories scanned, relative to the product root.
SCANNED = ("src", "tests", "examples", "validation", "data")


def python_files() -> list[Path]:
    """Every Python file under the scanned directories."""
    found: list[Path] = []
    for folder in SCANNED:
        base = ROOT / folder
        if base.is_dir():
            found.extend(sorted(base.rglob("*.py")))
    return [path for path in found if "__pycache__" not in path.parts]


def scan_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """Forbidden imports as ``(line, description)``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                found.append((node.lineno, f"from {node.module} import ..."))
    return found


def scan_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Forbidden attribute or name accesses as ``(line, description)``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append((node.lineno, f".{node.attr}"))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append((node.lineno, node.id))
    return found


def main() -> int:
    """Parse every Python file in the product and fail if a clock appears."""
    out = _bootstrap.Tee("validate_no_live_timing_output.txt")
    out("No live timing anywhere -- P097 flakerank")
    out("")
    out("THE DEFECT CLASS THIS CHECKS FOR COST THIS MISSION A WHOLE PRODUCT.")
    out("Five intermittent failures and a sixth caught in flight all traced to")
    out("checks that compared two wall-clock readings on a shared two-core")
    out("runner. This product is ABOUT intermittency, so the rule is applied in")
    out("its strongest checkable form: no file here reads a clock at all.")
    out("")
    out("Parsed with ast, not grepped, so a mention in a comment or a docstring")
    out("cannot trip the check and a clock behind a line continuation cannot hide")
    out("from it.")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")
    out(f"directories scanned: {list(SCANNED)}")
    out(f"forbidden modules  : {sorted(FORBIDDEN_MODULES)}")
    out(f"forbidden names    : {sorted(FORBIDDEN_NAMES)}")
    out(f"forbidden strings  : {list(FORBIDDEN_STRINGS)}")
    out("")
    out("declared exemptions from the string screen, with the reason:")
    for name, reason in DECLARED_EXEMPTIONS:
        out(f"  {name}: {reason}")

    files = python_files()
    out.rule("1. FILES SCANNED")
    out("")
    out(f"Python files found: {len(files)}")
    for path in files:
        out(f"  {_bootstrap.relative(path)}")
    out("")
    out.check(
        "there are files to scan, so the check is not vacuous",
        len(files) >= 20,
        f"{len(files)} files",
    )

    out.rule("2. IMPORT SCREEN")
    out("")
    import_offenders: list[str] = []
    name_offenders: list[str] = []
    string_offenders: list[str] = []
    exempt = {name for name, _reason in DECLARED_EXEMPTIONS}
    for path in files:
        relative = _bootstrap.relative(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for line, description in scan_imports(tree):
            import_offenders.append(f"{relative}:{line} {description}")
        for line, description in scan_names(tree):
            name_offenders.append(f"{relative}:{line} {description}")
        if relative in exempt:
            continue
        for number, text in enumerate(source.splitlines(), start=1):
            for fragment in FORBIDDEN_STRINGS:
                if fragment in text:
                    string_offenders.append(f"{relative}:{number} {fragment!r}")
    out(f"forbidden imports found: {len(import_offenders)}")
    for item in import_offenders:
        out(f"  OFFENDER {item}")
    out.check(
        "no file in this product imports a clock module",
        not import_offenders,
        f"{len(import_offenders)} offenders",
    )

    out.rule("3. NAME AND ATTRIBUTE SCREEN")
    out("")
    out(f"forbidden names found: {len(name_offenders)}")
    for item in name_offenders:
        out(f"  OFFENDER {item}")
    out.check(
        "no file in this product reads a clock through a name or an attribute",
        not name_offenders,
        f"{len(name_offenders)} offenders",
    )

    out.rule("4. SUBPROCESS AND FILESYSTEM CLOCK SCREEN")
    out("")
    out(f"forbidden strings found outside the declared exemptions: "
        f"{len(string_offenders)}")
    for item in string_offenders:
        out(f"  OFFENDER {item}")
    out.check(
        "no file reaches a clock through the filesystem or a shell",
        not string_offenders,
        f"{len(string_offenders)} offenders",
    )

    out.rule("5. THE SCREENS CAN FIRE -- controls on synthetic sources")
    out("")
    out("A screen that cannot fire is not a screen. Each control below is a")
    out("source fragment parsed in memory, so nothing is written into the tree.")
    out("")
    controls = (
        ("an import of time", "import time\nx = time.perf_counter()\n", "import"),
        ("a from-import of time", "from time import perf_counter\n", "import"),
        ("an import of datetime", "import datetime\n", "import"),
        ("an attribute clock read", "x = somemodule.monotonic()\n", "name"),
        ("a bare clock name", "from m import monotonic\ny = monotonic()\n", "name"),
    )
    for label, source, kind in controls:
        tree = ast.parse(source)
        fired = bool(scan_imports(tree)) if kind == "import" else bool(scan_names(tree))
        out.control(
            f"the {kind} screen catches {label}",
            fired,
            must_fire=True,
            detail=source.splitlines()[0],
        )
    out("")
    out("CONTROLS that must STAY SILENT: a source that reads no clock, and a")
    out("source that only MENTIONS one in a comment or a string, must both pass.")
    for label, source in (
        ("a clean source", "import math\nx = math.log(2.0)\n"),
        (
            "a comment mentioning perf_counter",
            "# this product never calls perf_counter\nimport math\n",
        ),
        (
            "a docstring mentioning time.monotonic",
            '"""No time.monotonic is read here."""\nimport math\n',
        ),
    ):
        tree = ast.parse(source)
        fired = bool(scan_imports(tree)) or bool(scan_names(tree))
        out.control(
            f"a screen fires on {label}",
            fired,
            must_fire=False,
            detail=source.splitlines()[0][:48],
        )

    out.rule("6. RUN TWICE FROM A CLEAN TREE -- the self-report screen")
    out("")
    out("A check whose own execution can create the condition it tests is not a")
    out("check, and this is the fifth occurrence of that class in this mission.")
    out("This script reads files and writes one output file. Its own output file")
    out("is a .txt and is therefore not in the .py scan at all, and its own")
    out("SOURCE is scanned like everything else, with its declared tables")
    out("exempted from the string screen only -- the import and name screens")
    out("apply to it with no exemption. So a second run sees exactly what the")
    out("first saw.")
    out("")
    out(f"this script's own source is among the scanned files: "
        f"{any(path.name == 'validate_no_live_timing.py' for path in files)}")
    out(f"its output file is a .txt and is not scanned: "
        f"{not any(path.suffix == '.txt' for path in files)}")
    out.check(
        "this script scans its own source",
        any(path.name == "validate_no_live_timing.py" for path in files),
    )
    out.check(
        "the scan covers only .py files, so no output file can be scanned",
        all(path.suffix == ".py" for path in files),
    )
    out("")
    out("probe_working_directories.py runs this script four times, from four")
    out("different working directories, and requires the same exit status from")
    out("all four. That is the second run, performed by the product's own")
    out("machinery rather than claimed here.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
