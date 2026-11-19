"""Path bootstrap, output tee, pristine copies and declared constants.

Five things in this module exist because of defects this mission has actually
shipped, and each is recorded here so that a later reader does not undo one.

**The working directory.** Every script here calls :func:`bootstrap` as its
first statement after setting ``sys.dont_write_bytecode``. It puts
``<root>/src`` on ``sys.path`` and changes the process working directory to
the product root, derived from ``__file__`` and never from the caller. Three
products in one batch of this mission passed every check in their own build
and then failed the release gate, which runs each validation script with its
working directory set to the script's own directory. The build agent, its
self-report and the coordinator's re-measurement had all run the scripts from
the product root; three independent verifications agreeing does not help when
they share an assumption. ``probe_working_directories.py`` runs every script
from four directories and requires the same exit status from all four.

**Absolute paths.** No committed file here may contain an absolute path: it is
not reproducible for a reader with a cold clone and it publishes the build
machine's filesystem layout. Everything printed goes through
:func:`relative`.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME``
and ``MPLCONFIGDIR`` at a temporary directory outside the product and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib``
into the repository, which the release gate fails twice over.

**Bytecode.** Every runnable script here sets ``sys.dont_write_bytecode =
True`` as its first statement, before any other import. It cannot be delegated
to this module, because CPython writes a module's bytecode cache BEFORE
executing that module, so this module cannot set it on its own behalf. The
subsequent module-level imports then carry ``# noqa: E402``.

**One declared seed.** :data:`SEED` is the only seed any script here uses, and
``tests/test_determinism.py`` asserts that it agrees with the examples'.

WALL CLOCK
----------
Nothing in this product asserts a relationship between two live wall-clock
measurements. Where a script reports a duration it is a build note, printed
with that said, and no check compares two of them. That defect class cost this
mission a whole product, traced to checks comparing timings on a shared
two-core runner, and this build ran alongside four others on two cores.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The one declared seed for every stochastic step in this product's tests,
#: examples and validation scripts.
SEED = 99099

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: Bootstrap resamples for every reported interval.
BOOTSTRAP_RESAMPLES = 2000

#: Nominal coverage of every reported predictive interval.
NOMINAL_COVERAGE = 0.80

#: ``PASSYIELD_FAST`` lowers run counts and is used only by
#: ``probe_working_directories.py``, whose question is whether a script runs
#: from a given directory and not what the numbers are. It is never set when a
#: reported number is produced, and every script prints whether it was set.
FAST = os.environ.get("PASSYIELD_FAST") == "1"


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


def subprocess_environment(home: Path | None = None) -> dict[str, str]:
    """Environment for a spawned subprocess, with ``HOME`` outside the product."""
    base = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="passyield-home-"))
    environment = dict(os.environ)
    environment["HOME"] = str(base)
    environment["MPLCONFIGDIR"] = str(base / "matplotlib")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment.pop("PYTHONPATH", None)
    return environment


#: Directory names never copied into a pristine copy.
_EXCLUDED_DIRECTORIES = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".git",
    ".config",
)


@contextmanager
def pristine_copy(root: Path) -> Iterator[tuple[Path, Path]]:
    """A throwaway copy of the product, and a throwaway ``HOME`` beside it.

    ``probe_working_directories.py`` runs every script inside this copy rather
    than in the tree, because a probe that ran them in place would overwrite
    every committed ``*_output.txt`` with fast-path numbers. That is the
    self-polluting-check class: the probe's own execution would create the
    condition the rest of the validation set is judged on.
    """
    base = Path(tempfile.mkdtemp(prefix="passyield-probe-"))
    try:
        copy = base / "product"
        shutil.copytree(
            root, copy, ignore=shutil.ignore_patterns(*_EXCLUDED_DIRECTORIES, "*.pyc")
        )
        yield copy, base / "home"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def tree_snapshot(root: Path) -> set[str]:
    """Relative paths of every cache directory and bytecode file in a tree.

    Used by ``validate_tree_hygiene.py`` to compare the tree at a script's
    entry against the tree at its exit. The product owns what its own
    execution ADDS; it does not own what a harness left behind before calling
    it, and a check that asserted no such artifact EXISTS could never pass,
    because the release gate's own ``ruff`` run writes ``.ruff_cache`` into the
    tree one check earlier.
    """
    found: set[str] = set()
    for path in root.rglob("*"):
        name = path.name
        if name in _EXCLUDED_DIRECTORIES or name.endswith((".pyc", ".pyo")):
            try:
                found.add(str(path.relative_to(root)))
            except ValueError:  # pragma: no cover - rglob yields inside root
                continue
    return found


def gitignore_rules(root: Path | None = None) -> tuple[list[str], list[str]]:
    """``(ignore, negate)`` patterns from the product's own ``.gitignore``."""
    base = Path(root) if root is not None else ROOT
    ignore: list[str] = []
    negate: list[str] = []
    for line in (base / ".gitignore").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            negate.append(stripped[1:].lstrip("/"))
        else:
            ignore.append(stripped)
    return ignore, negate


def gitignore_match(path: str, ignore: list[str], negate: list[str]) -> tuple[bool, str]:
    """Whether ``path`` is excluded by these rules, and by which pattern.

    One implementation, used by ``validate_record.py`` and
    ``validate_tree_hygiene.py``, because two implementations of the same
    semantics in one product is two chances to be wrong. The rules are
    re-implemented rather than delegated to a git command because build agents
    in this batch are forbidden from running git.

    The semantics that matter, and the one that has bitten this mission:

    - a pattern with **no slash** matches a component at **any depth**. A bare
      ``junit.xml`` in another product's ``.gitignore`` silently excluded a
      required committed fixture and a cold clone failed with 17 failures and
      9 errors while every local check passed.
    - a pattern with a **leading slash** is anchored at the repository root,
      and also covers everything beneath it.
    - a pattern with a **trailing slash** names a directory, and matches that
      directory itself as well as everything under it. Matching only the
      parent components is wrong and makes the rule silently miss the very
      directory it names.
    - a later negation wins.
    """
    parts = path.split("/")
    ignored = False
    reason = ""
    for pattern in ignore:
        anchored = pattern.startswith("/")
        directory_only = pattern.endswith("/")
        body = pattern.strip("/")
        if anchored:
            if path == body or path.startswith(body + "/") or fnmatch.fnmatch(path, body):
                ignored, reason = True, pattern
            continue
        if directory_only:
            # The directory itself and everything under it, at any depth.
            if any(fnmatch.fnmatch(part, body) for part in parts):
                ignored, reason = True, pattern
            continue
        if any(fnmatch.fnmatch(part, body) for part in parts):
            ignored, reason = True, pattern
    for pattern in negate:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(parts[-1], pattern):
            return False, f"negated by !{pattern}"
    return ignored, reason


class Tee:
    """Collect report lines, print them, and write them beside the script.

    Every script writes its raw output to ``validation/<stem>_output.txt`` so
    that every number in ``README.md``, ``validation/VALIDATION.md`` and
    ``MODEL_CARD.md`` traces to a file produced by a script that was actually
    run.

    The target is **deleted before it is written**, not appended to, so a
    run's output cannot be a merge of this run and the last one.
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

        A control that was supposed to fire and did not is a failure, and so
        is one that was supposed to stay silent and did not. Both directions
        are counted, because a control that cannot fire reads as evidence and
        is the worse of the two failures.
        """
        if must_fire:
            self.must_fire += 1
            self.must_fire_fired += int(fired)
            return self.check(f"control fires: {name}", fired, detail)
        self.must_not_fire += 1
        self.must_not_fire_silent += int(not fired)
        return self.check(f"control stays silent: {name}", not fired, detail)

    def unavailable(self, name: str, reason: str) -> None:
        """Record a check that cannot run because its input is absent.

        It does not pass. It is counted separately and printed, so a reader
        can never mistake an unavailable measurement for a clean one.
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
