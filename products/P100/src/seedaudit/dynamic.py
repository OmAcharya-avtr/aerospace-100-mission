"""Running a script in a pristine copy, and the two-arrangement classifier.

Every execution this product performs happens on a **copy**, in a temporary
directory, and never in the tree being audited. That is not hygiene. The audit
runs against a live repository that another agent may be writing to, so an
in-place run would corrupt that tree and would also destroy the audit's own
reproducibility: the second run would see what the first run left behind, which
is precisely the confound the product exists to measure.

The classifier
--------------
The self-polluting-check class is detected by running the script twice in two
*different arrangements* and comparing the two answers against each other:

- **pristine pair** -- run 1 in a fresh copy, run 2 in a *second* fresh copy;
- **reused tree** -- run 1 and run 2 in the *same* copy, in sequence.

A verdict that changes between two pristine runs is intrinsic
non-determinism. A verdict that changes between run 1 and run 2 of the reused
tree while the pristine pair agrees is **self-pollution**: the first run changed
its own preconditions. The cross-tabulation is the classifier and both halves
are reported:

=============== ============= ==================================
pristine pair   reused tree   classification
=============== ============= ==================================
differ          differ        intrinsic non-determinism
differ          agree         intrinsic, masked in the reused tree
agree           differ        **self-polluting check**
agree           agree         no evidence, qualified by ``p_max``
=============== ============= ==================================

The fourth row is never reported as a pass. It is reported with the
escaped-rate upper bound from :mod:`seedaudit.power` for the number of runs
that were actually performed.

Independence of repeats
-----------------------
The closed-form power model assumes independent repeats, and repeats inside one
process are not independent: they share RNG state, import caches, a fixed
``PYTHONHASHSEED`` and a warm filesystem cache. So each repeat is a fresh
subprocess in a fresh copy, and ``PYTHONHASHSEED`` is left at ``random`` unless
a caller pins it. The residual correlation is not zero and is not estimated;
that is stated in :mod:`seedaudit.power` with the bound it weakens.

The subprocess environment
--------------------------
``HOME`` and ``MPLCONFIGDIR`` point into the temporary directory and
``PYTHONDONTWRITEBYTECODE`` is set, because a validation script in an earlier
product of this mission set ``HOME`` to the product root and matplotlib wrote
``.config/matplotlib`` into the repository, which the release gate fails twice
over.

Units: ``timeout`` and ``duration`` are seconds. Everything else is a count.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .power import escaped_rate_upper_bound

#: Never copied into a pristine copy: they are caches, not inputs, and copying
#: them would carry one run's state into the next, which is the confound.
COPY_IGNORE = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".git",
    ".config",
    ".venv",
    "venv",
    ".mypy_cache",
)

#: Default per-run wall-clock limit, seconds.
DEFAULT_TIMEOUT = 75.0

#: Classification labels. Exported so that no caller spells one differently.
INTRINSIC = "intrinsic_non_determinism"
INTRINSIC_MASKED = "intrinsic_masked_in_reused_tree"
SELF_POLLUTING = "self_polluting_check"
NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class RunResult:
    """One execution of one script.

    Attributes:
        exit_code: the process exit status, or ``None`` if it timed out.
        stdout: captured standard output, decoded with ``errors="replace"``.
        stderr: captured standard error.
        duration: wall-clock seconds the process took.
        artifact: the text of the script's named output artifact after the run,
            or ``None`` if there is none or it was not written.
        timed_out: whether the run hit the wall-clock limit.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    artifact: str | None
    timed_out: bool

    @property
    def verdict(self) -> tuple[int | None, str]:
        """The script's answer: exit status paired with its reported output.

        The artifact is preferred over stdout when the script writes one,
        because the artifact is what the repository commits as evidence. When
        there is none, stdout is the answer.
        """
        return self.exit_code, self.artifact if self.artifact is not None else self.stdout

    @property
    def ok(self) -> bool:
        """Whether the run completed with exit status 0."""
        return self.exit_code == 0


@dataclass
class ArrangementResult:
    """The outcome of the two-arrangement comparison for one script.

    Attributes:
        script: the script's path, relative to the audited root.
        pristine: the runs of the pristine pair.
        reused: the runs of the reused tree.
        classification: one of :data:`INTRINSIC`, :data:`INTRINSIC_MASKED`,
            :data:`SELF_POLLUTING`, :data:`NO_EVIDENCE`.
        escaped_bound: the escaped-rate upper bound for the number of pristine
            runs performed, dimensionless. Attached to every result and not
            only to the clean ones, so a reader never has to go and look it up.
        note: anything that qualifies the result, such as a timeout.
    """

    script: str
    pristine: list[RunResult] = field(default_factory=list)
    reused: list[RunResult] = field(default_factory=list)
    classification: str = NO_EVIDENCE
    escaped_bound: float = 1.0
    note: str = ""

    @property
    def pristine_agree(self) -> bool:
        """Whether every pristine run returned the same verdict."""
        if len(self.pristine) < 2:
            return True
        first = self.pristine[0].verdict
        return all(run.verdict == first for run in self.pristine[1:])

    @property
    def reused_agree(self) -> bool:
        """Whether every run in the reused tree returned the same verdict."""
        if len(self.reused) < 2:
            return True
        first = self.reused[0].verdict
        return all(run.verdict == first for run in self.reused[1:])


def classify(pristine_agree: bool, reused_agree: bool) -> str:
    """Map the two-arrangement outcome to a classification label.

    Args:
        pristine_agree: whether the pristine runs all agreed.
        reused_agree: whether the reused-tree runs all agreed.

    Returns:
        One of the four module-level label constants.
    """
    if not pristine_agree:
        return INTRINSIC if not reused_agree else INTRINSIC_MASKED
    return SELF_POLLUTING if not reused_agree else NO_EVIDENCE


def subprocess_environment(home: Path, hash_seed: str | None = None) -> dict[str, str]:
    """Build the environment for a run, with ``HOME`` outside the audited tree.

    Args:
        home: a directory inside the temporary workspace.
        hash_seed: value for ``PYTHONHASHSEED``. ``None`` leaves it at
            ``random``, which is what the independence assumption of the power
            model needs; a caller pins it only to make the product's own
            behaviour reproducible.

    Returns:
        The environment mapping.
    """
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["MPLCONFIGDIR"] = str(home / "matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "random" if hash_seed is None else str(hash_seed)
    env.pop("PYTHONPATH", None)
    return env


@contextmanager
def pristine_copy(source: Path) -> Iterator[tuple[Path, Path]]:
    """Copy ``source`` into a temporary directory and yield ``(copy, home)``.

    The copy and the fresh ``HOME`` are siblings inside one temporary
    directory, which is removed on exit. Nothing is written to ``source``.
    """
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f"source must be an existing directory, got {root.name!r}")
    with tempfile.TemporaryDirectory(prefix="seedaudit-") as temporary:
        base = Path(temporary)
        copy = base / "tree"
        home = base / "home"
        home.mkdir()
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns(*COPY_IGNORE), symlinks=True)
        yield copy, home


def _read_artifact(copy: Path, artifact: str | None) -> str | None:
    """Read the named artifact from inside a copy, or return ``None``."""
    if artifact is None:
        return None
    path = copy / artifact
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def execute(
    copy: Path,
    home: Path,
    script: str,
    artifact: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    hash_seed: str | None = None,
    cwd: str | None = None,
) -> RunResult:
    """Run one script inside an existing copy and collect its result.

    Args:
        copy: the copied tree.
        home: a fresh ``HOME`` outside the copy.
        script: the script's path relative to ``copy``.
        artifact: the output file to read afterwards, relative to ``copy``.
        timeout: wall-clock limit, seconds.
        hash_seed: ``PYTHONHASHSEED`` value, or ``None`` for ``random``.
        cwd: working directory relative to ``copy``. Defaults to the script's
            own directory, which is how this mission's release gate invokes a
            validation script and is therefore the arrangement worth testing.

    Returns:
        A :class:`RunResult`.
    """
    limit = float(timeout)
    if limit <= 0.0:
        raise ValueError(f"timeout must be positive, got {limit!r}")
    target = copy / script
    if not target.is_file():
        raise ValueError(f"script not found inside the copy: {script!r}")
    working = copy / cwd if cwd is not None else target.parent
    env = subprocess_environment(home, hash_seed)
    start = os.times().elapsed
    try:
        completed = subprocess.run(  # noqa: S603 - the audited script is the subject
            [sys.executable, str(target)],
            cwd=str(working),
            capture_output=True,
            env=env,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        duration = os.times().elapsed - start
        return RunResult(
            exit_code=None,
            stdout=(expired.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(expired.stderr or b"").decode("utf-8", errors="replace"),
            duration=duration,
            artifact=_read_artifact(copy, artifact),
            timed_out=True,
        )
    duration = os.times().elapsed - start
    return RunResult(
        exit_code=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
        duration=duration,
        artifact=_read_artifact(copy, artifact),
        timed_out=False,
    )


def run_once(
    source: Path,
    script: str,
    artifact: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    transform: Callable[[str], str] | None = None,
    hash_seed: str | None = None,
    cwd: str | None = None,
) -> RunResult:
    """Run ``script`` once in a fresh pristine copy of ``source``.

    Args:
        source: the tree to copy.
        script: the script's path relative to ``source``.
        artifact: the output file to read afterwards, relative to ``source``.
        timeout: wall-clock limit, seconds.
        transform: applied to the script's source text inside the copy before
            it runs. This is how a seed is changed: the rewrite happens in the
            copy and the audited tree is never touched.
        hash_seed: ``PYTHONHASHSEED`` value, or ``None`` for ``random``.
        cwd: working directory relative to the copy.

    Returns:
        A :class:`RunResult`.
    """
    with pristine_copy(source) as (copy, home):
        if transform is not None:
            target = copy / script
            target.write_text(transform(target.read_text(encoding="utf-8")), encoding="utf-8")
        return execute(copy, home, script, artifact, timeout, hash_seed, cwd)


def repeat_runs(
    source: Path,
    script: str,
    repeats: int,
    artifact: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    transform: Callable[[str], str] | None = None,
    cwd: str | None = None,
) -> list[RunResult]:
    """Run ``script`` ``repeats`` times, each in its own fresh pristine copy.

    Args:
        source: the tree to copy.
        script: the script's path relative to ``source``.
        repeats: how many independent runs, dimensionless, at least 1.
        artifact: the output file to read afterwards.
        timeout: wall-clock limit per run, seconds.
        transform: applied to the script's text inside each copy.
        cwd: working directory relative to each copy.

    Returns:
        One :class:`RunResult` per repeat, in order.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError(f"repeats must be an int, got {type(repeats).__name__}")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    return [
        run_once(source, script, artifact, timeout, transform, None, cwd) for _ in range(repeats)
    ]


def two_arrangements(
    source: Path,
    script: str,
    artifact: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    pristine_repeats: int = 2,
    reused_repeats: int = 2,
    cwd: str | None = None,
) -> ArrangementResult:
    """Run the pristine pair and the reused tree, and classify the pair of answers.

    Args:
        source: the tree to copy.
        script: the script's path relative to ``source``.
        artifact: the output file to read after each run.
        timeout: wall-clock limit per run, seconds.
        pristine_repeats: runs in the pristine arrangement, each in its own
            copy, dimensionless, at least 2.
        reused_repeats: sequential runs inside one copy, dimensionless, at
            least 2.
        cwd: working directory relative to each copy.

    Returns:
        An :class:`ArrangementResult` carrying both arrangements, the
        classification and the escaped-rate bound for the pristine repeats.
    """
    if pristine_repeats < 2 or reused_repeats < 2:
        raise ValueError(
            "both arrangements need at least 2 runs; one run has nothing to "
            f"disagree with (got pristine={pristine_repeats}, reused={reused_repeats})"
        )
    result = ArrangementResult(script=script)
    result.pristine = repeat_runs(
        source, script, pristine_repeats, artifact, timeout, None, cwd
    )
    with pristine_copy(source) as (copy, home):
        for _ in range(reused_repeats):
            result.reused.append(execute(copy, home, script, artifact, timeout, None, cwd))
    result.classification = classify(result.pristine_agree, result.reused_agree)
    result.escaped_bound = escaped_rate_upper_bound(pristine_repeats)
    timeouts = sum(run.timed_out for run in result.pristine + result.reused)
    if timeouts:
        result.note = f"{timeouts} of {pristine_repeats + reused_repeats} runs timed out"
    return result
