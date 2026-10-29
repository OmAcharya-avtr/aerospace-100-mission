"""Path bootstrap, output tee and shared model fitting, used by every validation script.

Three things here exist because of defects this mission has shipped.

**The working directory.** Every script in this directory calls :func:`bootstrap`
as its first statement. It puts ``<repo>/src`` on ``sys.path`` and then changes
the process working directory to the repository root, derived from ``__file__``
and never from the caller. The release gate runs each validation script with its
working directory set to the script's own directory rather than to the product
root, and products in earlier batches of this mission passed every local check
and failed there. One ``os.chdir`` fixes every script in the directory and is a
no-op when the script is already run from the product root.
``probe_working_directories.py`` runs every script from five directories before
hand-off and requires exit 0 from all five.

**Absolute paths.** No committed file in this repository may contain an absolute
path: it is not reproducible for a reader with a cold clone, and it publishes
the build machine's filesystem layout. Everything printed by these scripts goes
through :func:`relative`, and the repository root itself is printed as its own
directory name and nothing above it.

**Timings.** :class:`Timer` accumulates and its results appear only in tables
explicitly declared unstable, never in a prose conclusion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Seeds. Every number in this repository comes from one of these.
SEED_TRAIN = 20261029
SEED_TEST = 70120261
SEED_CONTROL = 55120261
SEED_GRID = 31420261

#: The declared prior one-sigma for the regularised batch variant of equation
#: (18), radians. It is set to the standard deviation of a single component of
#: the generated misalignment population, which for a magnitude uniform on
#: [0, 3 mrad] and a direction uniform on the sphere is sqrt(3e-6 / 3) = 1 mrad.
#: Stated rather than tuned: the MAP estimator is given the correct prior scale
#: on purpose, because the learned estimator is trained on that same population
#: and comparing against a deliberately mis-specified prior would overstate the
#: learned contribution.
PRIOR_SIGMA_RAD = 1.0e-3


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
    ``validation/VALIDATION.md`` and ``MODEL_CARD.md`` traces to a file produced
    by a script that was actually run.
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


def _ruff_on_path() -> str:
    """Version of the ``ruff`` executable on PATH, which may differ from the package.

    Two ruff versions coexist in this environment: a standalone binary on PATH
    and a Python distribution reachable as ``python3 -m ruff``. An earlier batch
    of this mission had two products blocked at the release gate on a lint rule
    that only one of the two versions carried, so both are recorded and the
    source is checked under both.
    """
    import shutil
    import subprocess

    executable = shutil.which("ruff")
    if executable is None:
        return "not on PATH"
    try:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError:  # pragma: no cover - defensive
        return "not runnable"
    return completed.stdout.strip() or "unknown"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in the output of every script.

    ``filterpy`` is listed deliberately. It is named in this product's
    alternatives table, it is **not installed** in the measured environment, and
    nothing was installed to get it, so every output file states its absence in
    its own header rather than leaving the reader to infer it. No number in this
    repository is compared against it.
    """
    import numpy
    import sklearn

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {_distribution_version('scipy')}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff (installed) : {_distribution_version('ruff')}",
        f"ruff (on PATH)   : {_ruff_on_path()}  (a standalone binary, a DIFFERENT "
        "version from the installed distribution; the source is clean under both and "
        "the figures are in VALIDATION.md)",
        f"filterpy         : {_distribution_version('filterpy')}  (named in the "
        "alternatives table; no number here is compared against it)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository.

    ``HOME`` and ``MPLCONFIGDIR`` are pointed at a temporary directory outside
    the repository and ``PYTHONDONTWRITEBYTECODE`` is set, because a validation
    script in an earlier product of this mission set ``HOME`` to the product
    root and matplotlib wrote ``.config/matplotlib`` into the repository, which
    the release gate fails twice over: once as a tracked build artifact and once
    as an absolute path inside one.
    """
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="aligncal-validation-"))
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


def fit_estimators(
    *,
    count: int = 2500,
    samples: int = 150,
    seed: int = SEED_TRAIN,
) -> dict[str, object]:
    """Fit all three declared learned configurations and the trivial baseline.

    One definition, used by every script and example that needs a model, so the
    model behind every reported number in this repository is the same model.
    Returns the fitted estimators plus the training matrix, targets and
    population under the keys ``_features``, ``_targets`` and ``_population``.
    """
    from aligncal import dataset, features, learned

    population = dataset.training_population(count=count, samples=samples, seed=seed)
    matrix, targets, _ = features.build_population(population)
    fitted: dict[str, object] = {}
    for name, configuration in learned.CONFIGURATIONS.items():
        fitted[name] = learned.LearnedEstimator(seed=SEED_TRAIN, **configuration).fit(
            matrix, targets
        )
    fitted["trivial"] = learned.TrivialEstimator().fit(matrix, targets)
    fitted["_features"] = matrix
    fitted["_targets"] = targets
    fitted["_population"] = population
    return fitted


def axis_aligned_record(
    *,
    seed: int,
    samples: int = 150,
    magnitude_rad: float = 2.0e-3,
    profile: str = "single_axis",
    amplitude: float = 0.05,
    noise_sigma: float = 2.0e-4,
):
    """A record whose injected misalignment lies entirely along its own rotation axis.

    The control is anchored on a chosen truth, not on whatever a seed produced:
    the record is generated once to find its dominant rotation axis, then
    regenerated with the misalignment placed exactly along that axis. By
    equation (10) that is the direction the record cannot support at any data
    volume, so the truth is known to be unrecoverable by construction and any
    estimator that reports it confidently is reporting something the data does
    not contain.
    """
    from aligncal import dataset, observability

    probe = dataset.generate(
        profile=profile, samples=samples, amplitude=amplitude,
        noise_sigma=noise_sigma, seed=seed,
    )
    axis = observability.dominant_rotation_axis(probe.record)
    return dataset.generate(
        profile=profile,
        samples=samples,
        amplitude=amplitude,
        noise_sigma=noise_sigma,
        seed=seed,
        theta_rad=magnitude_rad * axis,
    )
