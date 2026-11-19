"""Path bootstrap, output tee and declared constants for every script here.

Six things in this module exist because of defects this mission has actually
shipped, and the sixth is this product's own subject.

**The working directory.** Every script calls :func:`bootstrap` as its first
statement after setting ``sys.dont_write_bytecode``. It puts ``<root>/src`` on
``sys.path`` and changes the process working directory to the product root,
derived from ``__file__`` and never from the caller. Three products in one batch
of this mission passed every check in their own build and then failed the
release gate, which runs each validation script with its working directory set
to the script's own directory; the build agent, its self-report and the
coordinator all ran the scripts from the product root, and three independent
verifications agreeing does not help when they share an assumption.
``probe_working_directories.py`` runs every script from four directories and
requires the same exit status from all four.

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

**One declared seed.** :data:`SEED` is the only seed any script here uses, and
it is the same value as ``data/generate_history.py``'s. A tool that measures
intermittency and is itself irreproducible is worthless.

**NO CLOCK IS READ ANYWHERE.** This is the product's own subject and the
mission's most expensive lesson: a whole product was lost to checks that
compared two wall-clock readings on a shared two-core runner, and five
intermittent failures plus a sixth caught in flight all traced to that one
class. Every duration any script here asserts on is an integer number of
milliseconds read from ``data/run_history.json``, generated once by a committed
deterministic script and committed with its SHA-256 digest.
``validate_no_live_timing.py`` parses every source file in the product and
fails if a clock appears at all.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The one declared seed for every stochastic step in this product's tests,
#: examples and validation scripts. The same value as the generator's.
SEED = 97044

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: Bootstrap resamples for every reported interval.
BOOTSTRAP_RESAMPLES = 2000

#: Prefix length for every reported cost-weighted capture figure.
TOP_K = 10

#: The committed recorded run history, relative to the product root.
HISTORY = Path("data") / "run_history.json"

#: SHA-256 of the committed history. ``validate_history.py`` is the
#: authoritative check; the value is declared here so every script can state
#: which bytes its numbers came from.
HISTORY_SHA256 = "aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a"

#: The learned component's split geometry, declared once so the model card, the
#: validation script and the example cannot drift apart.
LOOKBACK = 120
HORIZON = 30
TRAIN_ORIGINS = (120, 150, 180, 210)
TIME_ORIGINS = (240, 270)
MINIMUM_ATTEMPTS = 8

#: ``FLAKERANK_FAST`` lowers resample and tree counts and is used only by
#: ``probe_working_directories.py``, whose question is whether a script RUNS
#: from a given directory and not what the numbers are. It is never set when a
#: reported number is produced, and every script prints whether it was set.
FAST = os.environ.get("FLAKERANK_FAST") == "1"


def resamples() -> int:
    """Bootstrap resamples: the declared count, or a fast-path count."""
    return 40 if FAST else BOOTSTRAP_RESAMPLES


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
    base = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="flakerank-home-"))
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
    that every number in ``README.md``, ``validation/VALIDATION.md`` and
    ``MODEL_CARD.md`` traces to a file produced by a script that was actually
    run.

    The target is **deleted before it is written**, not appended to, so that a
    run's output cannot be a merge of this run and the last one. A scanner that
    reports its own previous output is a defect three consecutive sessions of
    this mission introduced in the act of writing about it.
    """

    def __init__(self, path: str | Path, directory: str | Path | None = None) -> None:
        # ``directory`` exists so a TEST could exercise the reporting code
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
        """Record a check that cannot run because its input is absent.

        It does not pass. It is counted separately and printed, so a reader can
        never mistake an unavailable measurement for a clean one.
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
