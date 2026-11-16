"""Loader for the committed hardware campaign. No fallback, ever.

Every hardware-mode figure in this product comes from ``data/traces/``, written
by ``data/campaign.py`` on the Jetson Orin Nano. There is no synthetic
substitute and no estimator: a loader that quietly produced a plausible number
when the campaign was absent would make a simulated figure indistinguishable
from a measurement, which is the one failure mode a differential timing product
cannot survive.

What is on disk, and what each file is for
------------------------------------------

``meta.json``
    Campaign metadata: seed, randomised cell order and permutation,
    interpreter and numpy versions, governor, power mode, the calibration
    policy, the start and end calibration medians and their ratio, and
    ``raw_sample_every``.
``stages_pass.csv``
    3240 rows: per-cell, per-pass, per-stage summaries computed over **every**
    timed frame of that pass. This is the primary analysis table and the
    bootstrap unit of SPEC §4 baseline 3 is one of its rows.
``stages.csv``
    270000 rows of per-frame per-stage timings, **subsampled every third
    frame** (``meta.json: raw_sample_every = 3``). Any figure derived from it
    states the subsampling, because a subsampled series is not the series the
    per-pass summaries were computed over.
``runs.csv``
    Nine rows, one per cell: monotonic start and end, measured empty-bracket
    cost, clock resolution, and the five gate branch counts.
``env.csv``
    94 rows of per-core frequency, four thermal zones and load average, sampled
    before each cell, at six points within it, and after it, plus before and
    after each calibration block.
``calibration.csv``
    The fixed 160x160 float64 matmul drift reference. **Eleven** blocks of 2000
    repetitions: ``floor_start``, ``floor_00`` .. ``floor_08``, ``floor_end``.
    SPEC §8 says ten; the campaign program emits an explicit leading block as
    well as one before each of the nine cells and one at the end, so the
    specification's count is one low. The discrepancy is reported rather than
    reconciled by dropping a block.
``raw/<cell>.npy``
    Per-cell raw ``(frames, 6)`` int32 arrays, also subsampled every third
    frame.
``campaign.log``
    The campaign's own stdout, ending in ``CAMPAIGN_COMPLETE``.

Units: nanoseconds throughout, except ``t_mono``/``t0_mono``/``t1_mono`` which
are ``time.monotonic()`` seconds, frequencies which are kHz as the kernel
reports them, and temperatures which are milli-degrees Celsius.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import ROOT

#: Where the measured traces live, relative to the product root. Relative, so
#: that no committed output can carry this machine's filesystem layout.
TRACES_DIRECTORY = "data/traces"

#: Environment variable that changes WHICH directory is required. It exists so
#: the refusal path can be exercised against an empty directory. It never makes
#: a measurement optional and it never selects a generated substitute.
TRACES_ENVIRONMENT_VARIABLE = "HILTIMINGDIFF_TRACES"

#: Files a complete campaign must carry. Literals, so a truncated campaign is a
#: refusal and not a smaller number.
REQUIRED_FILES: tuple[str, ...] = (
    "meta.json",
    "stages_pass.csv",
    "stages.csv",
    "runs.csv",
    "env.csv",
    "calibration.csv",
    "campaign.log",
)

#: The nine cells, by name, in the order the campaign randomised them under
#: seed 20261115. Pinned as literals: a campaign with a different cell set is
#: not this campaign and must not be reported as though it were.
REQUIRED_CELLS: tuple[str, ...] = (
    "w48_dense",
    "w64_sparse_rep1",
    "w48_sparse",
    "w64_sparse",
    "w64_sparse_rep3",
    "w96_dense",
    "w96_sparse",
    "w64_dense",
    "w64_sparse_rep2",
)

#: The eleven calibration block labels, in the order they were timed.
CALIBRATION_BLOCKS: tuple[str, ...] = (
    ("floor_start",) + tuple(f"floor_{index:02d}" for index in range(9)) + ("floor_end",)
)

#: The line the campaign writes last of all. A campaign without it is partial.
COMPLETION_TOKEN = "CAMPAIGN_COMPLETE"

#: Expected row counts of the committed campaign, excluding header rows. Pinned
#: so a silently truncated file fails rather than shrinking a median.
EXPECTED_ROWS: dict[str, int] = {
    "stages_pass.csv": 3240,
    "stages.csv": 270000,
    "runs.csv": 9,
    "env.csv": 94,
    "calibration.csv": 22000,
}

#: The five gate branch counters in ``runs.csv``: accept plus the four reject
#: reasons of ``pat_pipeline.REJECT_REASONS``.
GATE_BRANCHES: tuple[str, ...] = (
    "accept",
    "no_candidate",
    "low_score",
    "area_out",
    "border",
)


class TraceDataAbsentError(RuntimeError):
    """The measured campaign is absent, incomplete or not the pinned one."""


def traces_directory(directory: Path | str | None = None) -> Path:
    """Directory the measured campaign is required in. Existence not implied."""
    if directory is not None:
        return Path(directory)
    import os

    override = os.environ.get(TRACES_ENVIRONMENT_VARIABLE)
    return Path(override) if override else ROOT / TRACES_DIRECTORY


def traces_label(directory: Path | str | None = None) -> str:
    """Relative label of the required directory, for error text and reports."""
    candidate = traces_directory(directory)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


@dataclass(frozen=True)
class Campaign:
    """One loaded hardware campaign. Read-only, loaded lazily per table."""

    directory: Path
    label: str
    meta: dict

    @cached_property
    def stages_pass(self) -> pd.DataFrame:
        """Per-cell, per-pass, per-stage summaries over EVERY timed frame."""
        frame = pd.read_csv(self.directory / "stages_pass.csv")
        frame["stage"] = frame["stage"].astype(str)
        frame["cell"] = frame["cell"].astype(str)
        return frame

    @cached_property
    def stages_subsampled(self) -> pd.DataFrame:
        """Per-frame timings, subsampled every ``raw_sample_every`` frames."""
        frame = pd.read_csv(self.directory / "stages.csv")
        frame["cell"] = frame["cell"].astype(str)
        return frame

    @cached_property
    def runs(self) -> pd.DataFrame:
        """One row per cell: bracket cost, clock resolution, gate census."""
        frame = pd.read_csv(self.directory / "runs.csv")
        frame["cell"] = frame["cell"].astype(str)
        return frame

    @cached_property
    def env(self) -> pd.DataFrame:
        """Per-core frequency, thermal zones and load average samples."""
        frame = pd.read_csv(self.directory / "env.csv")
        frame["run_id"] = frame["run_id"].astype(str)
        frame["point"] = frame["point"].astype(str)
        return frame

    @cached_property
    def calibration(self) -> pd.DataFrame:
        """The fixed matmul drift reference, eleven blocks of 2000 reps."""
        frame = pd.read_csv(self.directory / "calibration.csv")
        frame["which"] = frame["which"].astype(str)
        return frame

    @cached_property
    def log(self) -> str:
        """The campaign's own stdout."""
        return (self.directory / "campaign.log").read_text()

    @property
    def sample_every(self) -> int:
        """``raw_sample_every``: the subsampling stride of ``stages.csv``."""
        return int(self.meta["raw_sample_every"])

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Stage names as the campaign recorded them, in pipeline order."""
        return tuple(str(name) for name in self.meta["stages"])

    @property
    def frequency_columns(self) -> tuple[str, ...]:
        """The six per-core ``scaling_cur_freq`` columns of ``env.csv``."""
        return tuple(name for name in self.env.columns if name.startswith("freq_cpu"))

    def raw(self, cell: str) -> np.ndarray:
        """Subsampled raw ``(frames, 6)`` array for one cell, in nanoseconds.

        The array is subsampled every :attr:`sample_every` frames. Callers that
        report a figure from it must say so.
        """
        path = self.directory / "raw" / f"{cell}.npy"
        if not path.is_file():
            raise TraceDataAbsentError(
                f"{self.label}/raw/{cell}.npy is absent. Cells present: "
                + ", ".join(sorted(p.stem for p in (self.directory / "raw").glob("*.npy")))
            )
        return np.load(path)

    def cell_width(self, cell: str) -> int:
        """Frame width ``W`` of one cell, in pixels."""
        rows = self.runs.loc[self.runs["cell"] == cell, "w"]
        if rows.empty:
            raise TraceDataAbsentError(f"cell {cell!r} is not in {self.label}/runs.csv")
        return int(rows.iloc[0])

    def order_index(self, cell: str) -> int:
        """Position of one cell in the randomised execution order, 0-based."""
        rows = self.runs.loc[self.runs["cell"] == cell, "order_index"]
        if rows.empty:
            raise TraceDataAbsentError(f"cell {cell!r} is not in {self.label}/runs.csv")
        return int(rows.iloc[0])


def campaign_is_complete(directory: Path | str | None = None) -> bool:
    """Whether ``campaign.log`` ends with the completion token."""
    path = traces_directory(directory) / "campaign.log"
    if not path.is_file():
        return False
    return COMPLETION_TOKEN in path.read_text()


def load_campaign(
    directory: Path | str | None = None,
    *,
    require_cells: tuple[str, ...] = REQUIRED_CELLS,
    require_rows: bool = True,
) -> Campaign:
    """Load the measured campaign, or raise naming exactly what is absent.

    Parameters
    ----------
    directory:
        Where to look. Defaults to :data:`TRACES_DIRECTORY` under the product
        root, or to :data:`TRACES_ENVIRONMENT_VARIABLE` when that is set.
    require_cells:
        Cell names that must all be present. Defaults to the nine pinned cells.
    require_rows:
        Whether to assert the pinned row counts of :data:`EXPECTED_ROWS`.

    Raises
    ------
    TraceDataAbsentError
        With the relative directory and the missing files, cells or row counts
        named. There is no fallback: this loader never substitutes generated
        data for a measurement.
    """
    target = traces_directory(directory)
    label = traces_label(directory)
    if not target.is_dir():
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN ABSENT. Expected a directory at {label}/ holding "
            + ", ".join(REQUIRED_FILES)
            + " and raw/<cell>.npy for each of "
            + ", ".join(require_cells)
            + ". Nothing was substituted: every hardware-mode figure in this "
            "product comes from that campaign and there is no synthetic "
            "fallback."
        )
    missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN INCOMPLETE. {label}/ is missing: "
            + ", ".join(missing)
            + ". No figure is reported from a partial campaign."
        )
    if not campaign_is_complete(target):
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN DID NOT COMPLETE. {label}/campaign.log does not "
            f"contain {COMPLETION_TOKEN}, which the campaign writes last of all. "
            "A campaign that did not finish is not a campaign that measured."
        )
    meta = json.loads((target / "meta.json").read_text())
    loaded = Campaign(directory=target, label=label, meta=meta)

    present = set(loaded.runs["cell"])
    absent = [cell for cell in require_cells if cell not in present]
    if absent:
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN MISSING CELLS. {label}/runs.csv does not contain: "
            + ", ".join(absent)
            + f". Present: {', '.join(sorted(present))}."
        )
    for cell in require_cells:
        if not (target / "raw" / f"{cell}.npy").is_file():
            raise TraceDataAbsentError(
                f"MEASURED CAMPAIGN MISSING RAW ARRAY. {label}/raw/{cell}.npy is absent."
            )
    if require_rows:
        counts = {
            "stages_pass.csv": len(loaded.stages_pass),
            "stages.csv": len(loaded.stages_subsampled),
            "runs.csv": len(loaded.runs),
            "env.csv": len(loaded.env),
            "calibration.csv": len(loaded.calibration),
        }
        wrong = {
            name: (EXPECTED_ROWS[name], counts[name])
            for name in EXPECTED_ROWS
            if counts[name] != EXPECTED_ROWS[name]
        }
        if wrong:
            detail = "; ".join(
                f"{name}: expected {expected} rows, found {found}"
                for name, (expected, found) in sorted(wrong.items())
            )
            raise TraceDataAbsentError(
                f"MEASURED CAMPAIGN ROW COUNTS CHANGED in {label}/. {detail}. The "
                "counts are pinned as literals so a silently truncated file fails "
                "rather than shrinking a median."
            )
    return loaded


def hardware_pass_table(campaign: Campaign) -> pd.DataFrame:
    """Per-pass hardware medians with the declared operation count beside them.

    One row per ``(cell, pass_idx, stage)``. Columns added to
    ``stages_pass.csv``:

    ``declared_ops``
        ``declared_ops(W, k_mean)[stage]``, dimensionless, from the committed
        board source. ``k_mean`` is that pass's own mean candidate count, so the
        declared model is evaluated at the load it actually saw.
    ``ns_per_declared_op``
        ``median_ns / declared_ops``, nanoseconds per declared operation. This
        is the host-calibration-free form of the divergence: it carries no
        ``ns_per_op`` and is therefore reproducible from the committed traces
        alone.
    """
    from .sources import declared_ops as declared

    frame = campaign.stages_pass.copy()
    counts = np.empty(len(frame), dtype=np.float64)
    widths = frame["w"].to_numpy()
    kmeans = frame["k_mean"].to_numpy()
    stages = frame["stage"].to_numpy()
    cache: dict[tuple[int, float], dict[str, float]] = {}
    for position in range(len(frame)):
        key = (int(widths[position]), float(kmeans[position]))
        if key not in cache:
            cache[key] = declared(key[0], key[1])
        counts[position] = cache[key][str(stages[position])]
    frame["declared_ops"] = counts
    frame["ns_per_declared_op"] = frame["median_ns"].to_numpy() / counts
    return frame
