"""The committed board programs, located, hashed and imported — never copied.

SPEC §1 rests on one property: *the same file runs in both modes*. The six
stages timed in simulation mode in this container are the stages of
``data/pat_pipeline.py``, and the instrument that times them is
``data/staged_exec.py``, both byte-identical to the copies that executed on the
Jetson Orin Nano. This module is the only place in the package that reaches
into ``data/``, and it refuses to hand back a module whose md5 does not match
the recorded sum.

Re-implementing the pipeline inside ``src/`` would destroy the property: a
per-stage timing difference would then be a difference between two
implementations and not a property of the platform. So the pipeline is imported
from the measurement tree, and the md5 sums below are asserted rather than
trusted.

``staged_exec.py`` does ``from pat_pipeline import ...`` — a flat import, as it
must be to run unmodified on a board where both files sit in one directory. The
loader therefore puts ``data/`` on ``sys.path`` for the duration of the import
and removes it again, rather than leaving the measurement tree importable.

Units: every timing quantity in this package is **nanoseconds**, because
``time.perf_counter_ns`` returns nanoseconds and no conversion is applied
anywhere between the instrument and the report.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

#: Repository root of this product, derived from this file and never from the
#: caller's working directory.
ROOT = Path(__file__).resolve().parent.parent.parent

#: Where the committed board programs live, relative to :data:`ROOT`.
DATA_DIRECTORY = "data"

#: md5 of each committed board program, as verified by the coordinator against
#: the copy that executed on the board. These are literals on purpose: if a
#: byte of the measurement program changes, every number in this repository
#: stops being a measurement of that program, and the suite must fail rather
#: than quietly re-measure something else.
SOURCE_MD5: dict[str, str] = {
    "campaign.py": "8c137fc2c31c48c2e8f16d1ae1f206d3",
    "staged_exec.py": "0f654894ea7dc3c6ec5d3880e9150459",
    "pat_pipeline.py": "80ba9015b54fa8e95eae96b2c93eb9c2",
    "verify_outputs.py": "ac6b6aa2b486ea7d9a227d1598dc8293",
}

#: The two programs that are IMPORTED and executed by this package. The other
#: two in :data:`SOURCE_MD5` are the campaign driver and its output verifier;
#: they ran on the board and are hashed but never imported here.
TIMED_SOURCES: tuple[str, ...] = ("pat_pipeline.py", "staged_exec.py")


class SourceIdentityError(RuntimeError):
    """A committed board program is missing, or its md5 does not match."""


def data_directory() -> Path:
    """Absolute path to the committed measurement tree."""
    return ROOT / DATA_DIRECTORY


def source_path(name: str) -> Path:
    """Absolute path of one committed board program.

    Raises
    ------
    SourceIdentityError
        If ``name`` is not one of the four recorded programs.
    """
    if name not in SOURCE_MD5:
        raise SourceIdentityError(
            f"{name!r} is not one of the committed board programs. "
            f"Known: {', '.join(sorted(SOURCE_MD5))}."
        )
    return data_directory() / name


def md5_of_file(path: Path | str) -> str:
    """Hex md5 of a file read in binary, in 1 MiB chunks."""
    # md5 is used here as a file-identity check against sums recorded by the
    # coordinator, not as a security primitive. The sums are published in
    # validation/VALIDATION.md and the comparison is for byte-identity only.
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def measured_md5() -> dict[str, str]:
    """md5 of every committed board program, measured now from disk.

    Raises
    ------
    SourceIdentityError
        If any of the four programs is absent, naming the ones that are.
    """
    measured: dict[str, str] = {}
    missing: list[str] = []
    for name in sorted(SOURCE_MD5):
        path = source_path(name)
        if not path.is_file():
            missing.append(name)
            continue
        measured[name] = md5_of_file(path)
    if missing:
        raise SourceIdentityError(
            "Committed board programs absent from "
            f"{DATA_DIRECTORY}/: {', '.join(missing)}. The pipeline is not "
            "re-implemented inside src/ by design, so there is nothing to fall "
            "back to and no figure can be produced."
        )
    return measured


def source_identity_failures() -> dict[str, tuple[str, str]]:
    """Programs whose measured md5 differs from the recorded one.

    Returns a mapping ``name -> (recorded, measured)``, empty when every
    program matches.
    """
    measured = measured_md5()
    return {
        name: (SOURCE_MD5[name], measured[name])
        for name in sorted(SOURCE_MD5)
        if measured[name] != SOURCE_MD5[name]
    }


def verify_source_identity() -> dict[str, str]:
    """Assert every committed board program matches its recorded md5.

    Returns the measured sums on success.

    Raises
    ------
    SourceIdentityError
        Naming each program that differs, with both sums.
    """
    failures = source_identity_failures()
    if failures:
        detail = "; ".join(
            f"{name}: recorded {recorded}, measured {found}"
            for name, (recorded, found) in failures.items()
        )
        raise SourceIdentityError(
            "COMMITTED BOARD PROGRAM CHANGED. The simulation-mode timings in "
            "this package are only comparable with the hardware-mode traces "
            "while the timed source is byte-identical to the copy that ran on "
            f"the board. {detail}."
        )
    return measured_md5()


def load_board_module(name: str) -> ModuleType:
    """Import one committed board program after verifying its md5.

    ``name`` is the module name, without ``.py``. Only the two programs in
    :data:`TIMED_SOURCES` are importable: the campaign driver reads
    ``/sys/devices`` and the verifier is a command-line tool, and neither is
    part of the comparison.

    Raises
    ------
    SourceIdentityError
        If ``name`` is not importable here, or if any md5 does not match.
    """
    filename = f"{name}.py"
    if filename not in TIMED_SOURCES:
        raise SourceIdentityError(
            f"{filename!r} is not one of the importable timed sources "
            f"({', '.join(TIMED_SOURCES)}). The campaign driver and the output "
            "verifier ran on the board and are hashed, not imported."
        )
    verify_source_identity()
    directory = str(data_directory())
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        module = importlib.import_module(name)
    finally:
        if inserted and directory in sys.path:
            sys.path.remove(directory)
    return module


def pipeline_module() -> ModuleType:
    """The committed ``pat_pipeline`` module, md5-verified."""
    return load_board_module("pat_pipeline")


def executor_module() -> ModuleType:
    """The committed ``staged_exec`` instrument, md5-verified."""
    return load_board_module("staged_exec")


def stage_names() -> tuple[str, ...]:
    """The six stage names, in pipeline order, from the committed source."""
    return tuple(pipeline_module().STAGES)


def reject_reasons() -> tuple[str, ...]:
    """The four gate reject reasons, from the committed source."""
    return tuple(pipeline_module().REJECT_REASONS)


def declared_ops(width: int, candidates: float) -> dict[str, float]:
    """Declared per-stage operation count for one frame, from the board source.

    Parameters
    ----------
    width:
        Frame width ``W`` in pixels; frames are ``W x W``.
    candidates:
        Number of candidate regions ``K`` in the frame. Accepts a float because
        the campaign reports a per-pass mean ``k_mean`` and the declared model
        is linear in ``K`` for the two stages that depend on it.

    Returns
    -------
    dict
        Stage name to declared operation count, dimensionless.

    Notes
    -----
    This is the simulation's own cost model and it is the object SPEC §4
    baseline 1 refutes. It is a *correct* operation count used as a timing
    proxy, which is exactly what a schedulability analysis derives by reading
    the source.
    """
    if width <= 0:
        raise ValueError(f"width must be a positive pixel count, got {width!r}")
    if candidates < 0:
        raise ValueError(f"candidates must be non-negative, got {candidates!r}")
    return dict(pipeline_module().declared_ops(int(width), float(candidates)))
