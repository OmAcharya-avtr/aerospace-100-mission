"""Path bootstrap, output tee and timing, used by every validation script here.

Three things in this module exist because of defects this mission has shipped.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and changes the process working directory to the repository root,
derived from ``__file__`` and never from the caller. Three products in one
earlier batch passed every check in their own build and failed the release
gate, which runs each validation script with its working directory set to the
script's own directory rather than to the product root. One ``os.chdir``
fixes every script in the directory and is a no-op when the script is already
run from the product root. ``validation/probe_working_directories.py`` runs
every script here from five directories and requires exit 0 from all five.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Every path printed by these
scripts goes through :func:`relative`.

**Timings.** :class:`Timer` accumulates across context entries. Timings in
this repository appear only in tables explicitly declared unstable and never
in a prose conclusion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely.

    A path outside the repository is reduced to its final component, which is
    the only part reproducible for a reader with a cold clone.
    """
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


class Tee:
    """Collect report lines, print them, and write them beside the script at the end."""

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.checks = 0

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
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

    def save(self) -> str:
        self("")
        self(f"CHECKS RUN: {self.checks}")
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}): {', '.join(self.failures)}")
        else:
            self("FAILED CHECKS (0): none")
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result is only ever reported in a table."""

    def __init__(self) -> None:
        self.elapsed_s = 0.0
        self.calls = 0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.elapsed_s += time.perf_counter() - self._start
        self.calls += 1

    @property
    def mean_ms(self) -> float:
        """Mean duration per context entry, in milliseconds."""
        return 1000.0 * self.elapsed_s / self.calls if self.calls else 0.0


def _distribution_version(name: str) -> str:
    """Installed version of ``name``, or ``"not installed"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in the output of every script.

    ``deepdiff`` is listed deliberately. The specification for this product
    names it in the alternatives table, it is **not installed** in this
    environment, and nothing was installed to get it, so every output file
    states its absence in its own header rather than leaving a reader to infer
    that a comparison against it was run. No number in this repository comes
    from ``deepdiff``.
    """
    import numpy
    import scipy

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"deepdiff         : {_distribution_version('deepdiff')}  (alternatives table)",
        f"jsonschema       : {_distribution_version('jsonschema')}  (alternatives table)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess, with a temporary ``HOME``.

    A build must not pollute its own tree through an environment it sets: an
    earlier product in this mission set ``HOME`` to its own product root and
    matplotlib wrote a configuration directory into the repository, which the
    release gate fails on twice over. ``HOME`` and ``MPLCONFIGDIR`` point at a
    temporary directory outside the repository and bytecode writing is off.
    """
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="tmtcdiff-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "MPLBACKEND": "Agg",
        }
    )
    return environment
