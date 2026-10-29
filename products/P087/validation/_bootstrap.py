"""Path bootstrap, output tee and shared states, used by every validation script.

Three things here exist because of defects this mission has shipped, and all
three are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and then changes the process working directory to the repository
root, derived from ``__file__`` and never from the caller. Four products in one
batch of this mission passed every check in their own build and then failed the
release gate, which runs each validation script with its working directory set
to the script's own directory rather than to the product root. One ``os.chdir``
fixes every script in the directory and is a no-op when the script is already
run from the product root. Each script here is executed from the filesystem
root, the system temporary directory, the product root, ``validation/`` and
``src/`` by ``probe_working_directories.py`` before hand-off, and all five must
exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the repository root itself is
printed as its own directory name and nothing above it.

**Timings.** :class:`Timer` accumulates. Timings in this repository appear only
in tables explicitly declared unstable, never in a prose conclusion.
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
    """Collect report lines, print them, and write them to a file at the end.

    Every script writes its raw output beside itself as
    ``<script stem>_output.txt`` so that every number in ``README.md`` and
    ``validation/VALIDATION.md`` traces to a file produced by a script that was
    actually run.
    """

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
        self.path.write_text("\n".join(self.lines) + "\n")
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

    ``astropy``, ``ephem`` and ``pyerfa`` are listed deliberately. All three
    are named in this product's alternatives table, none of them is installed
    in the measured environment, and nothing was installed to get them, so
    every output file states their absence in its own header rather than
    leaving the reader to infer it. **No number in this repository is compared
    against any of them.**
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
        f"astropy          : {_distribution_version('astropy')}  (alternatives table only)",
        f"ephem            : {_distribution_version('ephem')}  (alternatives table only)",
        f"pyerfa           : {_distribution_version('pyerfa')}  (alternatives table only)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository.

    ``HOME`` and ``MPLCONFIGDIR`` are pointed at a temporary directory outside
    the repository and ``PYTHONDONTWRITEBYTECODE`` is set, because a validation
    script in an earlier product of this mission set ``HOME`` to the product
    root and matplotlib wrote ``.config/matplotlib`` into the repository, which
    the release gate fails twice over: once as a tracked build artifact and
    once as an absolute path inside one.
    """
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="refractpath-validation-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    return environment


def standard_state() -> object:
    """1013.25 hPa, 10 C, dry, 550 nm: the conditions the published anchors use."""
    from refractpath import AtmosphericState

    return AtmosphericState(
        pressure_hpa=1013.25, temperature_c=10.0, relative_humidity=0.0, wavelength_um=0.55
    )


def published_state() -> object:
    """1010 hPa, 10 C, dry, 550 nm: the conditions equations (15) and (16) assume."""
    from refractpath import AtmosphericState

    return AtmosphericState(
        pressure_hpa=1010.0, temperature_c=10.0, relative_humidity=0.0, wavelength_um=0.55
    )


def default_tolerances_arcsec() -> tuple[float, ...]:
    """The pointing tolerances every floor figure in this repository uses.

    One definition, imported by every script and example, so that no two
    reported floors rest on different tolerances. The five values span a
    sub-arcsecond astrometric requirement, a typical small-telescope pointing
    budget, and a wide tracking-antenna budget.
    """
    return (1.0, 5.0, 10.0, 30.0, 60.0)
