"""Every command line entry point, in a subprocess, and the tree afterwards.

Each subprocess is given a temporary ``HOME`` and ``MPLCONFIGDIR`` and
``PYTHONDONTWRITEBYTECODE=1``, because a validation script in an earlier product
of this mission set ``HOME`` to the product root and matplotlib wrote
``.config/matplotlib`` into the repository, which a release gate fails twice
over: once as a tracked build artifact and once as an absolute path inside one.
The tree is checked for those artifacts after every command has run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    OBJECTS,
    ROOT,
    Tee,
    bootstrap,
    declared_settings,
    environment_lines,
    relative,
    subprocess_environment,
)

bootstrap()


def _objects(full: int) -> str:
    """Corpus size for one command, shrunk when the probe shrinks the corpus.

    The committed output uses the full sizes below. ``OBJECTS`` is 420 unless
    ``probe_working_directories.py`` has reduced it, and that probe runs this
    script five times in a row while itself spawning thirteen subprocesses per
    run, so the sizes shrink with it.
    """
    return str(min(full, max(12, OBJECTS)))


COMMANDS: list[tuple[str, list[str], list[str]]] = [
    ("version", ["--version"], ["tlecheck 0.1.0"]),
    (
        "info",
        ["info"],
        ["ships no element catalogue", "Not flight-qualified", "17.043642 rev/day"],
    ),
    (
        "check_clean",
        ["check", "--objects", _objects(48), "--corruption", "none"],
        ["WHAT THIS SCREEN CANNOT DETERMINE", "predicted consequence class : benign"],
    ),
    (
        "check_sign_flip",
        ["check", "--objects", _objects(48), "--corruption", "sign_flip"],
        ["QUALITY VERDICT", "integrity intact"],
    ),
    (
        "check_data_digit",
        ["check", "--objects", _objects(48), "--corruption", "data_digit"],
        ["checksum_mismatch"],
    ),
    (
        "determinability",
        ["determinability", "--objects", _objects(12)],
        ["propagated_ephemeris_accuracy", "continuity_resolution_km"],
    ),
    (
        "consequence",
        ["consequence", "--objects", _objects(120)],
        ["median max error km", "checksum_digit"],
    ),
    (
        "screen",
        ["screen", "--objects", _objects(120)],
        ["rule baseline against measured consequence", "operating-characteristic area"],
    ),
    (
        "benchmark",
        ["benchmark", "--objects", _objects(180)],
        ["majority-class baseline", "learned elements_and_history"],
    ),
    (
        "divergence",
        ["divergence", "--objects", _objects(60)],
        ["which source is right: None", "not determinable"],
    ),
]

ARTIFACTS = (".config", ".pytest_cache", ".ruff_cache", ".hypothesis", "build", "dist")


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    environment = subprocess_environment()
    out("")
    out(
        "every subprocess is given a temporary HOME and MPLCONFIGDIR outside the "
        "repository and PYTHONDONTWRITEBYTECODE=1"
    )

    out.rule("1. EVERY SUBCOMMAND, IN A SUBPROCESS")
    for name, arguments, expected in COMMANDS:
        result = subprocess.run(
            [sys.executable, "-m", "tlecheck", *arguments],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        out("")
        out(f"$ python -m tlecheck {' '.join(arguments)}")
        out(f"exit status : {result.returncode}")
        out(f"stdout lines: {len(result.stdout.splitlines())}")
        if result.stderr.strip():
            out(f"stderr      : {result.stderr.strip()[:300]}")
        out.check(f"cli_{name}_exits_zero", result.returncode == 0, f"status {result.returncode}")
        for fragment in expected:
            out.check(
                f"cli_{name}_says_{fragment[:28].strip().replace(' ', '_')}",
                fragment in result.stdout,
                f"expected {fragment!r} in the output",
            )

    out.rule("2. ERROR PATHS RETURN A STATUS AND A MESSAGE, NOT A TRACEBACK")
    failures: list[tuple[str, list[str], str]] = [
        (
            "missing_family",
            ["check", "--objects", "6", "--corruption", "cross_source_divergence"],
            "no record with corruption",
        ),
        ("absent_file", ["check", "--file", "no_such_element_set.txt"], ""),
    ]
    short = ROOT / "validation" / "_cli_probe_short.txt"
    short.write_text("1 25544U\n")
    failures.append(("short_file", ["check", "--file", relative(short)], "two element set lines"))
    for name, arguments, fragment in failures:
        result = subprocess.run(
            [sys.executable, "-m", "tlecheck", *arguments],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        out("")
        out(f"$ python -m tlecheck {' '.join(arguments)}")
        out(f"exit status : {result.returncode}")
        out(f"stderr      : {result.stderr.strip()[:300]}")
        out.check(f"cli_{name}_exits_two", result.returncode == 2, f"status {result.returncode}")
        out.check(
            f"cli_{name}_has_no_traceback",
            "Traceback" not in result.stderr,
            "a bad input produces a message, not a stack trace",
        )
        if fragment:
            out.check(
                f"cli_{name}_explains_itself",
                fragment in result.stderr,
                f"expected {fragment!r} in stderr",
            )
    short.unlink()

    out.rule("3. THE TREE AFTER EVERY COMMAND HAS RUN")
    present = [name for name in ARTIFACTS if (ROOT / name).exists()]
    out(f"checked for : {', '.join(ARTIFACTS)}")
    out(f"present     : {', '.join(present) if present else 'none'}")
    out.check(
        "no_config_directory_in_the_repository",
        not (ROOT / ".config").exists(),
        "matplotlib wrote .config/matplotlib into an earlier product of this "
        "mission through exactly this path",
    )
    out.check(
        "no_probe_file_left_behind",
        not short.exists(),
        "the temporary element set file written by this script is removed",
    )
    bytecode = sorted(ROOT.rglob("__pycache__"))
    out(f"__pycache__ directories in the tree : {len(bytecode)}")
    for path in bytecode[:10]:
        out(f"  {relative(path)}")
    out(
        "Bytecode caches are gitignored and are produced by running the test "
        "suite, not by these subprocesses, which set PYTHONDONTWRITEBYTECODE."
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
