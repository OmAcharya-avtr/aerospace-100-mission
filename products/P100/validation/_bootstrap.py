"""Path bootstrap, output tee and declared constants for every script here.

Six things in this module exist because of defects this mission has shipped,
and the sixth is this product's own subject.

**The working directory.** Every script calls :func:`bootstrap` as its first
statement after setting ``sys.dont_write_bytecode``. It puts ``<root>/src`` on
``sys.path`` and changes the process working directory to the product root,
derived from ``__file__`` and never from the caller. Three products in one batch
of this mission passed every check in their own build and then failed the
release gate, which runs each validation script with its working directory set
to the script's own directory. ``probe_working_directories.py`` runs every
script from four directories and requires the same exit status from all four.

**Absolute paths.** No committed file here may contain an absolute path: it is
not reproducible for a reader with a cold clone and it publishes the build
machine's filesystem layout. Everything printed goes through :func:`relative`.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME`` and
``MPLCONFIGDIR`` at a temporary directory outside the product and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib``
into the repository, which the release gate fails twice over.

**Bytecode.** Every runnable script here sets ``sys.dont_write_bytecode = True``
as its first statement, before any other import. It cannot be delegated to this
module, because CPython writes a module's bytecode cache BEFORE executing that
module, so this module cannot set it on its own behalf. The subsequent
module-level imports then carry ``# noqa: E402``.

**One declared seed.** :data:`SEED` is the only seed any script here uses. A
tool that audits reproducibility and is itself irreproducible is worthless.

**This script's own output is written into the directory this script's product
scans.** That is the self-polluting-check class, and it is the class this
product exists to detect. It cannot be avoided -- the mission's layout requires
``validation/<stem>_output.txt`` beside the script -- so it is handled the only
honest way: :class:`Tee` deletes its own target before writing, the repository
scan **excludes this product** and says so, and
``validate_self_audit.py`` points the product's own detectors at this tree
deliberately and publishes what they find, including findings against these
scripts.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The one declared seed for every stochastic step in this product's tests,
#: examples and validation scripts.
SEED = 1000044

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: Bootstrap resamples for every reported interval.
BOOTSTRAP_RESAMPLES = 2000

#: One-sided significance level of every reported bound.
ALPHA = 0.05

#: ``SEEDAUDIT_FAST`` lowers run counts and is used only by
#: ``probe_working_directories.py``, whose question is whether a script runs
#: from a given directory and not what the numbers are. It is never set when a
#: reported number is produced, and every script prints whether it was set.
FAST = os.environ.get("SEEDAUDIT_FAST") == "1"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the product root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the product root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def repository_root() -> Path | None:
    """The mission repository root, if this product sits inside one.

    Located by walking up from the product root looking for ``products.yaml``.
    Returns ``None`` when there is none, so a cold clone of this product alone
    reports the repository audit as unavailable rather than failing: the
    repository scan is a measurement of a corpus that a standalone clone does
    not have, and a missing measurement is not a code defect.
    """
    for parent in [ROOT, *ROOT.parents]:
        if (parent / "products.yaml").is_file() and (parent / "products").is_dir():
            return parent
    return None


def subprocess_environment(home: Path | None = None) -> dict[str, str]:
    """Environment for a spawned subprocess, with ``HOME`` outside the product."""
    base = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="seedaudit-home-"))
    env = dict(os.environ)
    env["HOME"] = str(base)
    env["MPLCONFIGDIR"] = str(base / "matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env.pop("PYTHONPATH", None)
    return env


class Tee:
    """Collect report lines, print them, and write them beside the script.

    Every script writes its raw output to ``validation/<stem>_output.txt`` so
    that every number in ``README.md`` and ``validation/VALIDATION.md`` traces
    to a file produced by a script that was actually run.

    The target is **deleted before it is written**, not appended to, because
    this product's own validation layout is an instance of the class it
    detects: the script writes into the directory its product scans. Deleting
    first means a run's output cannot be a merge of this run and the last one.
    """

    def __init__(self, path: str | Path, directory: str | Path | None = None) -> None:
        # ``directory`` exists so a TEST can exercise the reporting code
        # without overwriting a committed output file. Scripts never pass it.
        base = Path(directory) if directory is not None else ROOT / "validation"
        self.path = base / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.unmeasurable: list[str] = []
        self.checks = 0
        self.must_fire = 0
        self.must_fire_fired = 0
        self.must_not_fire = 0
        self.must_not_fire_silent = 0

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        """Print a section rule."""
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record a named check. A failure is reported, never suppressed."""
        self.checks += 1
        verdict = "PASS" if passed else "FAIL"
        self(f"[{verdict}] {name}{(' -- ' + detail) if detail else ''}")
        if not passed:
            self.failures.append(name)
        return passed

    def control(self, name: str, fired: bool, must_fire: bool, detail: str = "") -> bool:
        """Record a negative control and assert it behaved as declared.

        A control that was supposed to fire and did not is a failure, and so is
        one that was supposed to stay silent and did not. Both directions are
        counted, because a control that cannot fire reads as evidence and is
        the worse of the two failures.
        """
        if must_fire:
            self.must_fire += 1
            self.must_fire_fired += int(fired)
            return self.check(f"control fires: {name}", fired, detail)
        self.must_not_fire += 1
        self.must_not_fire_silent += int(not fired)
        return self.check(f"control stays silent: {name}", not fired, detail)

    def unavailable(self, name: str, reason: str) -> None:
        """Record a check that cannot run because its corpus is absent.

        It does not pass. It is counted separately and printed, so that a
        reader can never mistake an unavailable measurement for a clean one.
        """
        self.unmeasurable.append(name)
        self(f"[UNAVAILABLE] {name} -- {reason}")

    def finish(self) -> int:
        """Write the output file and return the process exit status."""
        self("")
        self("=" * 78)
        self(f"checks: {self.checks}")
        self(f"failures: {len(self.failures)}")
        if self.must_fire:
            self(f"controls that must fire: {self.must_fire_fired} of {self.must_fire} fired")
        if self.must_not_fire:
            self(
                f"controls that must stay silent: "
                f"{self.must_not_fire_silent} of {self.must_not_fire} silent"
            )
        if self.unmeasurable:
            self(f"unavailable: {len(self.unmeasurable)} -- {', '.join(self.unmeasurable)}")
        if self.failures:
            self("FAILED: " + ", ".join(self.failures))
        self("=" * 78)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return 1 if self.failures else 0
