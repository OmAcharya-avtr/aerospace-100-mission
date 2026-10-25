"""Every script in this repository, run from four different working directories.

Four products in one batch of this mission passed every check in their own build
and then failed the release gate, because the gate runs each validation and
example script with its working directory set to the script's own directory
rather than to the product root. The build agent, its self-report and the
coordinator's re-measurement had all run the scripts from the product root:
three independent verifications that shared an assumption rather than a method.

This script is the method. Every ``.py`` file in ``validation/`` and
``examples/`` is executed from the filesystem root, from the system temporary
directory, from the repository root and from its own directory, and the exit
status from each is recorded.

Two modes, because the exhaustive form costs too much wall clock for a single
gate script:

* **module body**, run for every script from all four directories. The file is
  executed with :func:`runpy.run_path` under a run name other than
  ``__main__``, so its module-level code -- the path bootstrap, the imports and
  any module-level data loading -- runs while ``main()`` does not. That is
  exactly the code the working-directory defect lives in.
* **full**, run for the scripts whose own runtime is short, end to end from all
  four directories.

Setting ``CONTACTSCHED_PROBE_FULL=1`` runs the full mode for every script from
all four directories. That form takes several minutes and was executed in the
build session; the committed output below is from the default mode and says so.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import Tee, Timer, bootstrap, environment_lines, relative

ROOT = bootstrap()

FAILURES: list[str] = []

SELF = Path(__file__).name

FAST = {
    "validate_geometry.py",
    "validate_visibility.py",
    "validate_convergence.py",
    "worked_example.py",
    "contact_windows.py",
    "schedule_comparison.py",
    "fairness_metrics.py",
    "contention_map.py",
}

MODULE_BODY = (
    "import os, runpy, sys; "
    "sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1]))); "
    "runpy.run_path(sys.argv[1], run_name='__contactsched_probe__')"
)
"""Run a script's module body without its ``main()``.

The directory insertion is required and is not boilerplate: ``python script.py``
puts the script's own directory on ``sys.path`` automatically, but
:func:`runpy.run_path` on a plain file does not, so ``from _bootstrap import ...``
fails without it. This probe's first run failed that way from all four
directories, which is a defect in the probe and not in the scripts.
"""


def scripts() -> list[Path]:
    """Every runnable script in ``validation/`` and ``examples/``, excluding this one.

    Support modules whose names begin with an underscore are excluded: they are
    imported by the scripts rather than run, and the release gate does not run
    them on their own.
    """
    found: list[Path] = []
    for folder in ("validation", "examples"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if path.name.startswith("_") or path.name == SELF:
                continue
            found.append(path)
    return found


def working_directories() -> list[tuple[str, Path]]:
    """The four directories every script is run from, labelled for the report."""
    return [
        ("filesystem root", Path(os.sep)),
        ("temporary directory", Path(tempfile.gettempdir())),
        ("repository root", ROOT),
        ("the script's own directory", Path("SCRIPT_DIRECTORY")),
    ]


def run(path: Path, cwd: Path, full: bool) -> tuple[int, str]:
    """Run ``path`` from ``cwd`` and return ``(exit status, trimmed stderr)``."""
    command = (
        [sys.executable, str(path)]
        if full
        else [sys.executable, "-c", MODULE_BODY, str(path)]
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MPLBACKEND": "Agg",
            "HOME": tempfile.gettempdir(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    tail = [line for line in result.stderr.splitlines() if line.strip()]
    message = tail[-1] if tail else ""
    return result.returncode, message.replace(str(ROOT), "<repository>")


def main() -> None:
    out = Tee("probe_working_directories_output.txt")
    out.rule("contactsched probe_working_directories")
    for line in environment_lines():
        out(line)

    probe_full = os.environ.get("CONTACTSCHED_PROBE_FULL") == "1"
    out.rule("Mode")
    out(f"CONTACTSCHED_PROBE_FULL set : {probe_full}")
    if probe_full:
        out("Every script is run END TO END from all four directories.")
    else:
        out("Every script has its MODULE BODY run from all four directories, and the")
        out("scripts listed as fast are additionally run END TO END from all four.")
        out("Set CONTACTSCHED_PROBE_FULL=1 for the exhaustive form, which takes several")
        out("minutes.")
    out("")
    out("Directories used:")
    for label, path in working_directories():
        # Rendered symbolically, never as a literal path, for the same reason the
        # scan patterns are assembled: this output file is committed.
        rendered = {
            "SCRIPT_DIRECTORY": "the script's own directory",
            str(ROOT): "<repository>",
            str(Path(os.sep)): "<filesystem root>",
            str(Path(tempfile.gettempdir())): "<system temporary directory>",
        }.get(str(path), "<redacted>")
        out(f"  {label:<28} {rendered}")

    found = scripts()
    out.rule(f"1. Module body, {len(found)} scripts, four directories each")
    out(f"{'script':<40} {'/':>5} {'tmp':>5} {'root':>5} {'own':>5} {'wall s':>8}")
    out("-" * 74)
    for path in found:
        statuses = []
        timer = Timer()
        for _label, directory in working_directories():
            cwd = path.parent if str(directory) == "SCRIPT_DIRECTORY" else directory
            with timer:
                status, message = run(path, cwd, full=False)
            statuses.append(status)
            if status != 0:
                FAILURES.append(
                    f"{relative(path)} module body exited {status} from the "
                    f"{_label} directory: {message[:90]}"
                )
        out(
            f"{relative(path):<40} {statuses[0]:>5d} {statuses[1]:>5d} "
            f"{statuses[2]:>5d} {statuses[3]:>5d} {timer.elapsed_s:>8.2f}"
        )

    full_set = found if probe_full else [p for p in found if p.name in FAST]
    out.rule(f"2. Full run, {len(full_set)} scripts, four directories each")
    if not full_set:
        out("No script selected for the full run.")
    else:
        out(f"{'script':<40} {'/':>5} {'tmp':>5} {'root':>5} {'own':>5} {'wall s':>8}")
        out("-" * 74)
        for path in full_set:
            statuses = []
            timer = Timer()
            for _label, directory in working_directories():
                cwd = path.parent if str(directory) == "SCRIPT_DIRECTORY" else directory
                with timer:
                    status, message = run(path, cwd, full=True)
                statuses.append(status)
                if status != 0:
                    FAILURES.append(
                        f"{relative(path)} exited {status} from the "
                        f"{_label} directory: {message[:90]}"
                    )
            out(
                f"{relative(path):<40} {statuses[0]:>5d} {statuses[1]:>5d} "
                f"{statuses[2]:>5d} {statuses[3]:>5d} {timer.elapsed_s:>8.2f}"
            )

    out.rule("3. No committed file contains an absolute path")
    out("Release-gate check 8 fails a whole batch on an absolute home-directory path in")
    out("any tracked file, and five products in one batch tripped it from committed")
    out("validation OUTPUT rather than from code. Every text file in the repository is")
    out("scanned here, including this script's own output.")
    out("")
    # The patterns are ASSEMBLED rather than written as literals. Writing them
    # out would put an absolute path prefix into this committed file, and the
    # scan would then report itself: two consecutive sessions of this mission
    # introduced exactly that defect in the act of writing about it, and the
    # first run of this probe reproduced it.
    separator = os.sep
    pattern_names = ("home", "Users", "root", "tmp")
    patterns = tuple(separator + name + separator for name in pattern_names)
    suffixes = {".py", ".txt", ".md", ".toml", ".cff", ".yml", ".yaml", ".gitignore"}
    # This script's own output file is removed before the scan and rewritten
    # after it. Without that, the scan reads the PREVIOUS run's output, which
    # named the offenders it found, and reports those names as fresh offenders:
    # the probe fails on its own report of a problem that was already fixed. The
    # file rewritten at the end names offenders by file and line only, never by
    # the literal pattern, so it cannot reintroduce one.
    out.path.unlink(missing_ok=True)
    offenders: list[tuple[str, str, int]] = []
    # Offenders are reported by the NAME of the pattern they matched, never by
    # the pattern itself, and the pattern list is printed the same way. Printing
    # the literals would put absolute path prefixes into this committed output
    # file, which the next run of this very scan would then report. The first
    # two runs of this probe did exactly that.
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".hypothesis"}
               for part in path.parts):
            continue
        if path.suffix not in suffixes and path.name != ".gitignore":
            continue
        scanned += 1
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in zip(pattern_names, patterns, strict=True):
                if pattern in line:
                    offenders.append((relative(path), name, number))
    out(f"text files scanned : {scanned}")
    out(
        "patterns searched  : the path separator, then one of "
        + ", ".join(pattern_names)
        + ", then the separator again"
    )
    out(f"offending lines    : {len(offenders)}")
    for name, pattern_name, number in offenders[:20]:
        out(f"  {name}:{number} matched the {pattern_name} pattern")
        FAILURES.append(f"{name}:{number} matched the absolute {pattern_name} path pattern")
    out("")
    out("This scan reads the working tree, which is what the gate reads. This")
    out("script's own output file is deleted immediately before the scan and written")
    out("again after it, so a previous run's report of an offender cannot be counted")
    out("as a fresh offender. A non-zero count above is a real offender.")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES[:30]:
            out(f"  - {failure}")
        if len(FAILURES) > 30:
            out(f"  ... and {len(FAILURES) - 30} more")
    else:
        out("Every script exited zero from every directory, and no tracked text file")
        out("contains an absolute path.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
