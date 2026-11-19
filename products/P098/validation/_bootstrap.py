"""Path bootstrap, output tee and declared constants for every script here.

**The working directory.** Every script calls :func:`bootstrap` as its first
statement after setting ``sys.dont_write_bytecode``. It puts ``<root>/src`` on
``sys.path`` and changes the process working directory to the product root,
derived from ``__file__`` and never from the caller. Three products in one
batch of this mission passed every check in their own build and then failed the
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

**No script here regenerates the corpus.** The committed corpora in ``data/``
are written once by ``data/generate_corpus.py``, which is deliberately NOT
under ``validation/``: the release gate runs ``sorted(glob('validation/*.py'))``,
so alphabetically adjacent scripts run back to back, and a script that
regenerated data another script pins by hash would be collided by the gate's own
ordering. ``validate_corpus_pin.py`` reads and hashes; nothing here writes to
``data/``.

**One declared seed.** :data:`SEED` is the only seed any script here uses.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The one declared seed for every stochastic step in this product's tests,
#: examples and validation scripts. The same value as the corpus generator's.
SEED = 980098

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: Bootstrap resamples for every reported interval.
BOOTSTRAP_RESAMPLES = 2000

#: Grouped folds for the held-out-artifact split.
GROUP_SPLITS = 5

#: ``EVIDENCESTALE_FAST`` lowers resample and fold counts and is used only by
#: ``probe_working_directories.py``, whose question is whether a script runs
#: from a given directory and not what the numbers are. It is never set when a
#: reported number is produced, and every script prints whether it was set.
FAST = os.environ.get("EVIDENCESTALE_FAST") == "1"

#: Resamples and folds actually used, after the fast path.
RESAMPLES = 100 if FAST else BOOTSTRAP_RESAMPLES
SPLITS = 2 if FAST else GROUP_SPLITS

#: Directory names never copied into a pristine copy of this product.
EXCLUDED_FROM_COPY = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".config",
    ".git",
    "build",
    "dist",
)


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
    base = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="evstale-home-"))
    environment = dict(os.environ)
    environment["HOME"] = str(base)
    environment["MPLCONFIGDIR"] = str(base / "matplotlib")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment.pop("PYTHONPATH", None)
    return environment


@contextmanager
def pristine_copy(root: Path) -> Iterator[tuple[Path, Path]]:
    """A throwaway copy of the product, plus a ``HOME`` outside it.

    The working-directory probe runs every script inside the copy, never in the
    tree, because a probe that ran them in place would overwrite every committed
    ``*_output.txt`` with fast-path numbers -- the probe's own execution would
    create the condition the rest of the validation set is judged on.
    """
    with tempfile.TemporaryDirectory(prefix="evstale-probe-") as temporary:
        base = Path(temporary)
        copy = base / root.name
        shutil.copytree(
            root, copy, ignore=shutil.ignore_patterns(*EXCLUDED_FROM_COPY)
        )
        home = base / "home"
        home.mkdir()
        yield copy, home


class Tee:
    """Collect report lines, print them, and write them beside the script.

    Every script writes its raw output to ``validation/<stem>_output.txt`` so
    that every number in ``README.md``, ``MODEL_CARD.md`` and
    ``validation/VALIDATION.md`` traces to a file produced by a script that was
    actually run. The target is DELETED before it is written, not appended to,
    so a run's output cannot be a merge of this run and the last one.
    """

    def __init__(self, path: str | Path, directory: str | Path | None = None) -> None:
        # ``directory`` exists so a test could exercise the reporting code
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
        """Record a check that cannot run because its input is absent."""
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
