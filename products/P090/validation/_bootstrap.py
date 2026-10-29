"""Path bootstrap, output tee and shared model fitting for every validation script.

Four things here exist because of defects this mission has shipped, and all four
are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls :func:`bootstrap`
as its first statement. It puts ``<repo>/src`` on ``sys.path`` and then changes
the process working directory to the repository root, derived from ``__file__``
and never from the caller. Four products in one batch of this mission passed
every check in their own build and then failed the release gate, which runs each
validation script with its working directory set to the script's own directory
rather than to the product root. One ``os.chdir`` fixes every script in the
directory and is a no-op when the script is already run from the product root.
Each script here is executed from the filesystem root, the system temporary
directory, the product root, ``validation/`` and ``src/`` by
``probe_working_directories.py`` before hand-off, and all five must exit zero.

**Absolute paths.** No committed file in this repository may contain an absolute
path: it is not reproducible for a reader with a cold clone, and it publishes the
build machine's filesystem layout. Everything printed by these scripts goes
through :func:`relative`, and the repository root itself is printed as its own
directory name and nothing above it.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME`` and
``MPLCONFIGDIR`` at a temporary directory outside the repository and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib`` into
the repository.

**Timings.** :class:`Timer` accumulates. Timings in this repository appear only
in tables explicitly declared unstable, never in a prose conclusion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OBJECTS = int(os.environ.get("TLECHECK_VALIDATION_OBJECTS", "420"))
"""Objects per split for every reported number. Declared once, used everywhere.

The environment variable exists for one purpose and it is named in the output of
every script: ``probe_working_directories.py`` runs every script from five
different working directories, and five full-size runs of seven scripts is not
an affordable way to test a one-line path contract. The probe sets it to 60. The
committed output files are all produced at the default of 420 and every script
prints the value it used, so a reader can see which.
"""

HISTORY_LENGTH = 12
"""Prior element sets per object. The sweep behind this choice is in VALIDATION.md."""

CADENCE_DAYS = 1.5
HORIZON_DAYS = 1.0
SAMPLES = 25
SILENT_THRESHOLD_KM = 1.0
EPOCH_REQUIREMENT_DAYS = 7.0
TARGET_FPR = 0.05


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
    ``validation/VALIDATION.md``, ``MODEL_CARD.md`` and ``DATASET_CARD.md``
    traces to a file produced by a script that was actually run.

    ``TLECHECK_OUTPUT_DIR`` redirects that file. It exists for one caller:
    ``probe_working_directories.py`` re-runs every script ninety times at a
    reduced corpus size, and without the redirect those runs would overwrite the
    committed full-size output files with small-corpus numbers.
    """

    def __init__(self, path: str | Path) -> None:
        directory = os.environ.get("TLECHECK_OUTPUT_DIR")
        self.path = (Path(directory) if directory else ROOT / "validation") / Path(path).name
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

    ``astropy`` and ``skyfield`` are listed deliberately. Both are named in this
    product's alternatives table, neither is installed in the measured
    environment, and nothing was installed to get them, so every output file
    states their absence in its own header rather than leaving the reader to
    infer it. No number in this repository is compared against either.
    """
    import numpy
    import sklearn

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"sgp4             : {_distribution_version('sgp4')}",
        f"scipy            : {_distribution_version('scipy')}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pandas           : {_distribution_version('pandas')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"astropy          : {_distribution_version('astropy')}  (named in the "
        "alternatives table; no number here is compared against it)",
        f"skyfield         : {_distribution_version('skyfield')}  (named in the "
        "alternatives table; no number here is compared against it)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="tlecheck-validation-"))
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


def declared_settings() -> list[str]:
    """The declared policy values every number in this repository rests on."""
    return [
        f"objects per split          : {OBJECTS}",
        f"element history per object : {HISTORY_LENGTH} sets at "
        f"{CADENCE_DAYS:g} day cadence",
        f"consequence horizon        : {HORIZON_DAYS:g} days, {SAMPLES} samples",
        f"silent threshold           : {SILENT_THRESHOLD_KM:g} km (declared policy)",
        f"epoch requirement          : {EPOCH_REQUIREMENT_DAYS:g} days (declared policy)",
        f"target false-positive rate : {TARGET_FPR:g} (declared policy)",
    ]


def prepare_splits(
    silent_threshold_km: float = SILENT_THRESHOLD_KM,
    history_length: int = HISTORY_LENGTH,
    objects: int = OBJECTS,
) -> dict[str, object]:
    """Screen, measure and featurise the three declared splits.

    One definition, imported by every script that needs them, so that the corpus
    behind every reported number is the same corpus.
    """
    from tlecheck import benchmark, dataset

    out: dict[str, object] = {}
    for name in ("train", "test", "calibration"):
        records = dataset.split(
            name, objects=objects, history_length=history_length, cadence_days=CADENCE_DAYS
        )
        out[name] = benchmark.prepare(
            records,
            epoch_requirement_days=EPOCH_REQUIREMENT_DAYS,
            horizon_days=HORIZON_DAYS,
            samples=SAMPLES,
            silent_threshold_km=silent_threshold_km,
        )
    return out


def fit_models(train: object) -> dict[str, object]:
    """Fit all three declared learned configurations plus the majority baseline."""
    from tlecheck.learned import CONFIGURATIONS, LearnedClassifier, MajorityClassifier

    models: dict[str, object] = {}
    for name, configuration in CONFIGURATIONS.items():
        feature_configuration = str(configuration["feature_configuration"])
        model = LearnedClassifier(**configuration)  # type: ignore[arg-type]
        model.fit(train.matrices[feature_configuration], train.measured_classes)  # type: ignore[attr-defined]
        models[name] = model
    models["majority"] = MajorityClassifier().fit(train.measured_classes)  # type: ignore[attr-defined]
    return models
