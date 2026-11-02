"""Path bootstrap, output tee and declared constants, used by every script here.

Three things in this module exist because of defects this mission has shipped,
and all three are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement after disabling bytecode writing. It
puts ``<repo>/src`` on ``sys.path`` and then changes the process working
directory to the repository root, derived from ``__file__`` and never from the
caller. Three products in one batch of this mission passed every check in
their own build and then failed the release gate, which runs each validation
script with its working directory set to the script's own directory rather
than to the product root. One ``os.chdir`` fixes every script in the directory
and is a no-op when the script is already run from the product root. Each
script here is executed from the filesystem root, the system temporary
directory, the product root and ``validation/`` -- which is the gate's working
directory -- by ``probe_working_directories.py`` before hand-off, and all four
must exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the repository root itself is
printed as its own directory name and nothing above it.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME``
and ``MPLCONFIGDIR`` at a temporary directory outside the repository and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib``
into the repository, which the release gate fails twice over: once as a tracked
build artifact and once as an absolute path inside one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The declared divergence tolerance. It is exactly 1 because the divergence
#: metric of equation (6) is scaled by the declared engineering tolerances, so
#: a divergence of 1 IS the tolerance. Nothing in this repository tunes it.
TOLERANCE = 1.0

#: Controller iterations per run. 200 at a 10 ms period is a 2 s run, which is
#: a realistic length for a manoeuvre card on a rig.
STEPS = 200

#: Runs the learned component is FITTED on. Disjoint from TEST_SEEDS, and the
#: split is by RUN rather than by miss: misses inside one run share its
#: trajectory and its timing trace and are not independent of each other, so a
#: split by miss would leak.
TRAIN_SEEDS: tuple[int, ...] = tuple(range(1000, 1072))

#: Runs every reported score is measured on. Never fitted on, never used to
#: choose a hyperparameter.
TEST_SEEDS: tuple[int, ...] = tuple(range(2000, 2045))

#: Runs the certificate's soundness against re-simulation is measured over.
#: Disjoint from both sets above, so the soundness measurement is not taken on
#: the data the learned comparison is taken on.
SOUNDNESS_SEEDS: tuple[int, ...] = tuple(range(3000, 3045))

#: Drift rates swept in the clock study, parts per million. Spans a decent
#: crystal, a bad one, and a misconfigured timer prescaler.
DRIFT_SWEEP_PPM: tuple[float, ...] = (-20000.0, -5000.0, -500.0, -50.0, 0.0,
                                      50.0, 500.0, 5000.0, 20000.0)

#: Timestamp jitter levels swept in the clock study, seconds. The last is
#: 100 microseconds, a tenth of a period, where recovery is expected to suffer.
JITTER_SWEEP_S: tuple[float, ...] = (2.0e-7, 1.0e-6, 1.0e-5, 1.0e-4)

#: Declared drift allowance for the run verdict, parts per million.
ALLOWANCE_PPM = 100.0

#: Declared recall target for the learned component. 1.0, because the whole
#: argument of this product is that dismissing a consequential miss is not
#: symmetric with re-simulating an inconsequential one.
TARGET_RECALL = 1.0


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
    ``<script stem>_output.txt`` so that every number in ``README.md``,
    ``MODEL_CARD.md`` and ``validation/VALIDATION.md`` traces to a file
    produced by a script that was actually run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.checks = 0
        self.controls_fired = 0
        self.controls_expected = 0

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

    def control(self, name: str, fired: bool, must_fire: bool, detail: str = "") -> bool:
        """Record a negative control and whether it FIRED as declared.

        A control that does not fire is a failure of the control, not a success
        of the product: it means the check it guards would not have caught the
        violation it was built to catch. Both directions are asserted, because
        a control that fires when it must not is equally wrong.
        """
        self.controls_expected += 1
        passed = fired == must_fire
        if fired:
            self.controls_fired += 1
        expectation = "must fire" if must_fire else "must NOT fire"
        status = "FIRED" if fired else "did not fire"
        self.check(f"control: {name} ({expectation})", passed, f"{status}; {detail}")
        return passed

    def save(self) -> str:
        self("")
        self(f"CHECKS RUN: {self.checks}")
        if self.controls_expected:
            self(
                f"CONTROLS: {self.controls_expected} declared, "
                f"{self.controls_fired} fired"
            )
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

    ``torch`` and ``control`` are listed deliberately. The first is named in
    this product's README as unavailable in the measured environment; the
    second is the mature control-systems package named in the alternatives
    table. Neither is installed, nothing was installed to get them, and no
    number in this repository is compared against either. Each output file
    states their absence in its own header rather than leaving the reader to
    infer it.
    """
    import numpy
    import scipy
    import sklearn

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"control          : {_distribution_version('control')}  (named in README; no "
        "number here is compared against it)",
        f"torch            : {_distribution_version('torch')}  (named in README as "
        "unavailable here)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="hilverdict-validation-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            # Hypothesis writes its example database into `.hypothesis` in the
            # CURRENT WORKING DIRECTORY, which for a subprocess started here is
            # the repository root. Pointing it at the scratch directory keeps
            # this product's own scripts from creating a cache directory inside
            # the tree they then check for cache directories.
            "HYPOTHESIS_STORAGE_DIRECTORY": str(scratch / "hypothesis"),
        }
    )
    return environment


def measure_integrator_floor(
    model: object, command_rad: object, substeps: int | None = None
) -> float:
    """The divergence a miss with a ZERO input step produces: the integrator floor.

    A miss that holds a command equal to the one it supersedes injects nothing,
    so the true divergence is exactly zero. The with-miss branch nevertheless
    integrates the affected period in two pieces while the reference integrates
    it in one, and two fixed-step Runge-Kutta integrations of the same interval
    with different substep layouts do not agree to the last bit. The difference
    is the floor below which no divergence measurement means anything.

    It is measured here, by the same code path the re-simulation uses, rather
    than declared as a constant, and both ``validate_plant.py`` and
    ``validate_certificate.py`` call this function so the number they quote is
    the same number.

    Returns the worst such divergence over a declared grid of miss positions
    and overrun fractions, in the declared metric's units.
    """
    import numpy

    from hilverdict.resimulate import simulate

    reference = simulate(model, command_rad, None, substeps=substeps)
    worst = 0.0
    for index in (1, 37, 83, 141, 190):
        for fraction in (0.1, 0.37, 0.5, 0.9):
            overrun = fraction * model.sampling.period_s
            stale = float(reference.computed_rad[index])
            perturbed = simulate(
                model,
                command_rad,
                {index: overrun},
                start_index=index,
                start_state=reference.states[index],
                prior_command_rad=stale,
                substeps=substeps,
            )
            clean = simulate(
                model,
                command_rad,
                None,
                start_index=index,
                start_state=reference.states[index],
                substeps=substeps,
            )
            worst = max(
                worst,
                float(
                    numpy.max(model.metric.norm(perturbed.states - clean.states))
                ),
            )
    return worst
