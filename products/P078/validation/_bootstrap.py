"""Path bootstrap and output tee shared by every validation script.

Three things in here exist because of defects this mission has shipped.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and then changes the process working directory to the repository
root, derived from ``__file__`` and never from the caller. Four products in
Batch 07 passed every check in their own build and then failed the release
gate, which runs each validation script with its working directory set to
``validation/`` rather than to the product root. One ``os.chdir`` fixes every
script in the directory and is a no-op when the script is already run from the
product root. Each script here is executed from several different working
directories before hand-off and all of them must exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the repository root itself is
printed as its own directory name and nothing above it.

**Timings.** :class:`Timer` accumulates, because a non-accumulating version of
it silently reported the duration of the last loop iteration only. Every timing
this repository publishes sits in a table explicitly declared unstable and
never in a prose conclusion: a Batch 07 sentence reading "a factor of 40"
became "a factor of 38" on re-run for exactly this reason.
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
    ``validation/<script stem>_output.txt`` so that every number in README.md,
    validation/VALIDATION.md and MODEL_CARD.md traces to a file produced by a
    script that was actually run. The layout is flat: not ``validation/outputs/``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        """A section banner."""
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        """Record one named check with its verdict. Returns ``passed``.

        A failing check is printed as FAIL and added to :attr:`failures`. It is
        never silently downgraded and no tolerance in this repository was
        widened to turn one green.
        """
        verdict = "PASS" if passed else "FAIL"
        self(f"  [{verdict}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            self.failures.append(label)
        return passed

    def summarise(self) -> int:
        """Print the failure summary. Returns the process exit code."""
        self("")
        self("-" * 78)
        if self.failures:
            self(f"FAILED CHECKS: {len(self.failures)}")
            for label in self.failures:
                self(f"  {label}")
        else:
            self("FAILED CHECKS: none")
        return 1 if self.failures else 0

    def save(self) -> str:
        """Write the transcript beside this script; return a relative path."""
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result is only reported in a table."""

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
    """Versions and core count, recorded at the head of every output file."""
    import numpy
    import sklearn
    import yaml

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"pyyaml           : {yaml.__version__}",
        f"scipy            : {_distribution_version('scipy')}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"joblib           : {_distribution_version('joblib')}",
        f"pandas           : {_distribution_version('pandas')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def header(tee: Tee, title: str) -> None:
    """Print the standard script header: title, then the environment."""
    tee("=" * 78)
    tee(title)
    tee("=" * 78)
    tee("")
    tee("environment this output was measured on:")
    for line in environment_lines():
        tee(f"  {line}")
