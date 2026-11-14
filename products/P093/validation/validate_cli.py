"""The command-line interface, exercised as a subprocess from a clean environment.

INPUT ORIGIN: SYNTHETIC, through the CLI's own ``--simulate`` path, plus the
CLI's refusal path against an empty directory. No measured data is involved.

Every subprocess is given a temporary ``HOME`` and ``MPLCONFIGDIR`` outside this
repository and ``PYTHONDONTWRITEBYTECODE=1``, because a validation script in an
earlier product of this mission set ``HOME`` to the product root and matplotlib
wrote ``.config/matplotlib`` into the repository, which the release gate fails
twice over.

What is checked:

1. ``--help`` exits 0 and names the program;
2. each subcommand runs under ``--simulate`` and exits 0;
3. every ``--simulate`` run labels its output SYNTHETIC, at the top, before any
   number;
4. ``--traces`` pointing at a directory with no campaign exits 3, says why on
   stderr, and does NOT fall back to the generator;
5. a directory without the completion marker also exits 3;
6. a corrupted header exits 2;
7. the console entry point declared in ``pyproject.toml`` resolves to the same
   function the module runs.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import tempfile  # noqa: E402
import tomllib  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

ENVIRONMENT = _bootstrap.subprocess_environment()

#: Temporary directory for the fixtures the refusal checks need on disk. It is
#: outside the repository, so the paths in a captured transcript would be
#: absolute. Release-gate check 8 fails the whole batch on an absolute path in
#: any tracked file, and an absolute path is not reproducible for a reader with
#: a cold clone, so :func:`sanitise` rewrites it out of everything this script
#: prints. That substitution is deliberate and is recorded here rather than left
#: for a reader to notice.
SCRATCH = Path(tempfile.mkdtemp(prefix="powerenvelope-cli-"))


def sanitise(text: str) -> str:
    """Replace absolute paths in captured output with relative placeholders."""
    return (
        text.replace(str(SCRATCH), "<scratch>")
        .replace(str(ROOT), "<product-root>")
        .replace(str(Path(tempfile.gettempdir())), "<tmp>")
    )


def _run(arguments: list[str], timeout: float = 900.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "powerenvelope", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=ENVIRONMENT,
        timeout=timeout,
        check=False,
    )


def main() -> int:
    out = _bootstrap.Tee("validate_cli_output.txt")
    out.rule("CLI -- every subcommand and every declared exit status")
    out("INPUT ORIGIN: SYNTHETIC via --simulate, plus the CLI's refusal path.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    out.rule("1  python -m powerenvelope --help")
    completed = _run(["--help"])
    out.check("--help exits 0", completed.returncode == 0, f"status {completed.returncode}")
    out.check(
        "--help names the program and the headline",
        "powerenvelope" in completed.stdout and "understates" in completed.stdout,
        f"{len(completed.stdout.splitlines())} line(s) of help",
    )
    for line in sanitise(completed.stdout).splitlines()[:6]:
        out(f"    {line}")

    out.rule("2  every subcommand under --simulate")
    for command, extra in (
        ("manifest", ["--simulate-cycles", "6"]),
        ("headline", ["--simulate-cycles", "6"]),
        ("report", ["--simulate-cycles", "10"]),
        ("learned", ["--simulate-cycles", "10"]),
    ):
        completed = _run(["--simulate", *extra, command])
        out.check(
            f"--simulate {command} exits 0",
            completed.returncode == 0,
            f"status {completed.returncode}, {len(completed.stdout.splitlines())} line(s)",
        )
        if completed.returncode != 0:
            for line in sanitise(completed.stderr).splitlines()[-10:]:
                out(f"    {line}")
            continue
        head = completed.stdout.splitlines()[:6]
        out.check(
            f"--simulate {command} labels its output SYNTHETIC before any number",
            any("SYNTHETIC CAMPAIGN" in line for line in head),
            "the banner is the first thing printed",
        )

    out.rule("3  the headline subcommand prints the decomposition in both variants")
    completed = _run(["--simulate", "--simulate-cycles", "6", "headline"])
    for line in sanitise(completed.stdout).splitlines():
        if "factor" in line or "variant" in line or "HEADLINE" in line:
            out(f"    {line}")
    out.check(
        "both Baseline-1 variants are reported",
        "measured-active" in completed.stdout and "nominal-rail" in completed.stdout,
        "measured-active and nominal-rail",
    )
    out.check(
        "the partition residual, the closure error and the occupancy diagnostic are "
        "reported, not just the three factors",
        "partition residual" in completed.stdout
        and "closure error" in completed.stdout
        and "DIAGNOSTIC, not a factor" in completed.stdout,
        "the decomposition is checked rather than claimed, and the phase occupancy is "
        "shown as the diagnostic it is rather than as a fourth factor",
    )

    out.rule("4  the report subcommand prints every baseline")
    completed = _run(["--simulate", "--simulate-cycles", "10", "report"])
    for fragment in (
        "measured background floor",
        "baseline 1, energy per inference",
        "baseline 2, the analytic duty-cycle model",
        "measured rail-accounting ratio against load",
        "baseline 3, ground truth",
        "between-run spread",
        "permissive-gate control",
        "the learned predictor's dataset",
    ):
        out.check(f"report includes: {fragment}", fragment in completed.stdout, "")
    out.check(
        "report says the rail ratio is not a converter efficiency",
        "not a converter efficiency" in completed.stdout,
        "",
    )
    out.check(
        "report says the floor is a background floor on a shared machine",
        "BACKGROUND FLOOR ON A SHARED MACHINE" in completed.stdout,
        "",
    )

    out.rule("5  the refusal path: absent traces exit 3 and do not fall back")
    # A RELATIVE path that does not exist, so the CLI's own error message quotes
    # a relative path rather than this machine's filesystem layout.
    completed = _run(["--traces", "data/traces_absent_probe", "report"])
    out.check(
        "an absent campaign directory exits 3",
        completed.returncode == _bootstrap_exit_absent(),
        f"status {completed.returncode}",
    )
    out.check(
        "the refusal names the marker it looked for",
        "CAMPAIGN_COMPLETE is absent" in completed.stderr,
        sanitise(completed.stderr).strip().splitlines()[0][:100]
        if completed.stderr.strip()
        else "",
    )
    out.control(
        "the refusal falls back to the synthetic generator",
        "SYNTHETIC" in completed.stdout,
        must_fire=False,
        detail="stdout carries no synthetic banner and no figures",
    )

    incomplete = SCRATCH / "incomplete"
    _bootstrap.synthetic_campaign(incomplete, cycles=3, complete=False)
    completed = _run(["--traces", str(incomplete), "manifest"])
    out.check(
        "a campaign without the completion marker exits 3",
        completed.returncode == _bootstrap_exit_absent(),
        f"status {completed.returncode}",
    )

    out.rule("6  a corrupted header exits 2")
    corrupt = SCRATCH / "corrupt"
    _bootstrap.synthetic_campaign(corrupt, cycles=3)
    target = corrupt / "power_w64_d050.csv"
    body = target.read_text().splitlines()[1:]
    target.write_text("a,b,c,d,e,f,g,h\n" + "\n".join(body) + "\n")
    completed = _run(["--traces", str(corrupt), "manifest"])
    out.check(
        "a corrupted header exits 2 with an input error",
        completed.returncode == 2 and "INPUT ERROR" in completed.stderr,
        f"status {completed.returncode}",
    )

    out.rule("7  the console entry point in pyproject.toml")
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    scripts = project.get("project", {}).get("scripts", {})
    out(f"  declared scripts: {scripts}")
    out.check(
        "a powerenvelope console script is declared",
        scripts.get("powerenvelope") == "powerenvelope.__main__:main",
        "it resolves to the same function python -m runs",
    )

    out.rule("8  the subprocess environment did not pollute this tree")
    for name in (".config", ".ruff_cache", "__pycache__"):
        created = (ROOT / name).exists()
        out(f"  {name} present in the tree: {created}")
    out(
        "  Presence here is reported and not failed: what a harness leaves in the tree "
        "before calling a script is not the script's doing. "
        "validate_tree_hygiene.py asserts what this product owns, which is that running "
        "a script ADDS nothing and that git tracks nothing of the kind."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


def _bootstrap_exit_absent() -> int:
    """The CLI's declared exit status for absent traces, read from the module."""
    from powerenvelope.__main__ import EXIT_ABSENT

    return EXIT_ABSENT


if __name__ == "__main__":
    raise SystemExit(main())
