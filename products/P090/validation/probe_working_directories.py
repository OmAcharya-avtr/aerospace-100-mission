"""Prove that the working directory changes nothing about any script in this repository.

Four products in one batch of this mission passed every check in their own build
and then failed the release gate, which runs each validation script with its
working directory set to the script's own directory rather than to the product
root. Measured on one of them: exit 0 from the product root, exit 1 from
anywhere else. The build agent, the agent's own self-report and the
coordinator's re-measurement all ran the scripts the same way. Three
independent verifications agreeing does not help when they share an assumption,
so this probe does not share it.

**What is tested, stated precisely.** Every script in ``validation/`` and every
example in ``examples/`` is executed from the filesystem root, the system
temporary directory, the repository root, ``validation/`` and ``src/``, and the
exit status and the complete standard output must be **identical** across all
five. That is the contract the defect above violated, and it is a stronger
statement than requiring exit zero: a script whose numbers change with the
working directory would pass an exit-zero check and fail this one.

**Where the output goes.** The probe redirects every script's output file and
every example's figure into a temporary directory through
``TLECHECK_OUTPUT_DIR`` and ``TLECHECK_FIGURE_DIR``, so that ninety
small-corpus runs cannot overwrite the committed full-size outputs. The
redirect is also why the probe can assert that the committed files still exist
and still carry the full-size corpus line afterwards.

**What is not tested here.** Whether each script's own checks pass at the full
declared corpus size. That is what the committed ``validate_*_output.txt`` files
are, each produced by running the script directly at the default corpus of 420
objects per split. This probe sets ``TLECHECK_VALIDATION_OBJECTS`` to a small
value, which every script prints in its own header, because ninety full-size
runs is not an affordable way to test a path contract. Several checks in
``validate_learned.py`` and ``validate_rule_screens.py`` are measurements over a
population and do not hold on a corpus of twenty-four objects; those scripts
therefore exit non-zero under this probe, identically from all five directories,
and that identity is the thing being asserted.

Subprocesses are given a temporary ``HOME`` and ``MPLCONFIGDIR`` and
``PYTHONDONTWRITEBYTECODE=1`` so that matplotlib cannot write a cache into the
repository.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    ROOT,
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

bootstrap()

PROBE_OBJECTS = "24"
"""Corpus size for the probe. The committed outputs use the default of 420."""


def _digest(text: str) -> str:
    """Hash of a script's standard output, with machine-dependent lines removed.

    Two kinds of line are dropped. The environment header carries the number of
    CPU cores the run saw, which is a property of the machine. And every line a
    script tags ``[unstable]`` is a timing, which moves between runs on a shared
    machine; leaving those in would make this probe report a path defect every
    time and would make a timing able to fail a path check.
    """
    body = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("cpu cores", "[unstable]"))
    )
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def main() -> int:
    out = Tee("probe_working_directories_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)

    scripts = sorted(
        path
        for path in (ROOT / "validation").glob("*.py")
        if path.name not in {"_bootstrap.py", Path(__file__).name}
    )
    examples = sorted(
        path for path in (ROOT / "examples").glob("*.py") if path.name != "_common.py"
    )
    targets = scripts + examples

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tlecheck-probe-") as temporary:
        directories = [
            ("root", Path("/")),
            ("tmp", Path(temporary)),
            ("product", ROOT),
            ("validation", ROOT / "validation"),
            ("src", ROOT / "src"),
        ]
        out.rule("EVERY SCRIPT FROM EVERY WORKING DIRECTORY")
        out(
            f"{len(targets)} scripts times {len(directories)} working directories = "
            f"{len(targets) * len(directories)} runs, at "
            f"TLECHECK_VALIDATION_OBJECTS={PROBE_OBJECTS}"
        )
        out("")
        out(
            "The columns are the exit status and the first sixteen hexadecimal "
            "digits of a hash of the complete standard output. All five must "
            "match for the contract to hold."
        )
        out("")
        environment = subprocess_environment()
        environment["TLECHECK_VALIDATION_OBJECTS"] = PROBE_OBJECTS
        scratch_outputs = Path(temporary) / "outputs"
        scratch_figures = Path(temporary) / "figures"
        scratch_outputs.mkdir()
        scratch_figures.mkdir()
        environment["TLECHECK_OUTPUT_DIR"] = str(scratch_outputs)
        environment["TLECHECK_FIGURE_DIR"] = str(scratch_figures)
        out(
            "Output files and figures are redirected into a temporary directory "
            "for the duration of this probe, so the committed full-size outputs "
            "are not overwritten with small-corpus numbers."
        )
        out("")
        header = f"{'script':34s}" + "".join(
            f"{name:>22s}" for name, _ in directories
        )
        out(header)
        mismatched: list[str] = []
        for target in targets:
            row = f"{relative(target):34s}"
            signatures: set[tuple[int, str]] = set()
            for _name, directory in directories:
                result = subprocess.run(
                    [sys.executable, str(target)],
                    cwd=str(directory),
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1800,
                )
                signature = (result.returncode, _digest(result.stdout))
                signatures.add(signature)
                row += f"{signature[0]:>5d} {signature[1]:>16s}"
            out(row)
            if len(signatures) != 1:
                mismatched.append(
                    f"{relative(target)}: {len(signatures)} distinct "
                    f"(status, output) signatures across five working directories: "
                    f"{sorted(signatures)}"
                )
        out("")
        for failure in mismatched:
            out(f"MISMATCH: {failure}")
        out.check(
            "the_working_directory_changes_nothing",
            not mismatched,
            f"{len(targets)} scripts, {len(targets) * len(directories)} runs, "
            f"{len(mismatched)} scripts whose behaviour depended on the working "
            "directory",
        )

    out.rule("OUTPUT FILES LAND BESIDE THEIR SCRIPTS, NOT BESIDE THE CALLER")
    expected = [
        ROOT / "validation" / f"{path.stem}_output.txt"
        for path in scripts
        if path.name.startswith(("validate_", "worked_", "probe_"))
    ]
    missing = [relative(path) for path in expected if not path.exists()]
    out(f"expected output files : {len(expected)}")
    out(f"missing               : {len(missing)}")
    for path in missing:
        out(f"  {path}")
    out.check(
        "every_script_wrote_its_output_file_into_validation",
        not missing,
        f"{len(expected) - len(missing)} of {len(expected)} present",
    )
    full_size = [
        relative(path)
        for path in expected
        if path.exists() and "objects per split          : 420" not in path.read_text()
    ]
    out(f"committed outputs NOT at the full corpus of 420 : {len(full_size)}")
    for path in full_size:
        out(f"  {path}")
    out.check(
        "the_probe_did_not_overwrite_the_committed_outputs",
        not full_size,
        f"{len(expected) - len(full_size)} of {len(expected)} still carry the "
        "full-size corpus line",
    )
    figures = sorted((ROOT / "screenshots").glob("*.png"))
    out("")
    out(f"figures in screenshots/ : {len(figures)}")
    for path in figures:
        out(f"  {relative(path)}")
    out.check(
        "every_example_wrote_its_figure_into_screenshots",
        len(figures) == len(examples),
        f"{len(figures)} figures for {len(examples)} examples",
    )

    out.rule("THE TREE AFTERWARDS")
    for name in (".config", "build", "dist"):
        out.check(
            f"no_{name.strip('.')}_in_the_repository",
            not (ROOT / name).exists(),
            f"{name} absent",
        )
    out(
        "Bytecode caches and the pytest, ruff and hypothesis caches are gitignored "
        "and are produced by the test suite rather than by these subprocesses, "
        "which set PYTHONDONTWRITEBYTECODE."
    )
    out("")
    out(
        f"This probe took {time.perf_counter() - started:.0f} s of wall clock on "
        "two cores. That figure is a timing and is declared unstable; no "
        "conclusion rests on it."
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
