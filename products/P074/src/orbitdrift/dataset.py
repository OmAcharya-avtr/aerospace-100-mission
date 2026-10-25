"""Deterministic synthetic population and the divergence table built from it.

There is no two-line element archive in this environment, so the population is
synthetic and that is stated everywhere a number derived from it appears. What
"synthetic" means precisely:

- The orbits are drawn from declared distributions over perigee altitude,
  eccentricity, inclination and ``B*``, by a seeded
  :class:`numpy.random.Generator`. :data:`POPULATION_SPEC` is the whole
  specification and ``DATASET_CARD.md`` repeats it.
- Each drawn orbit is turned into a **real two-line element set**: the mean
  elements are fed to ``Satrec.sgp4init`` and then written out with
  ``sgp4.exporter.export_tle``, and the committed ``fixtures/population.tle``
  is re-read through ``Satrec.twoline2rv`` before use. The round trip through
  the fixed-width TLE format quantises every element -- eccentricity to seven
  decimal places, mean motion to eight, ``B*`` to five significant figures --
  and the quantised values are what everything downstream sees. A population
  held in floating point and never written to a TLE would be a population no
  operator could ever have.
- The divergence table is then produced by :func:`build_divergence_table`,
  which is pure computation: given the same TLE file it returns the same
  numbers, and ``validation/validate_reproducibility.py`` checks that it does
  across repeated runs and across batch groupings.

What is **not** claimed: that this population resembles the real catalogue in
its joint distribution. It is a spanning set over the features, drawn
independently, which is what a benchmark over feature space needs and is not
what a population-statistics study would need. ``DATASET_CARD.md`` says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4 import exporter
from sgp4.api import WGS72 as SGP4_WGS72
from sgp4.api import Satrec

from .constants import SECONDS_PER_DAY, WGS72
from .divergence import sgp4_states, window_rms
from .errors import InvalidTleError, PropagationError
from .features import (
    FEATURE_NAMES,
    epoch_jd,
    features_from_satrec,
    period_s_from_satrec,
)
from .forces import ForceModel, cd_area_over_mass_m2_kg
from .frames import to_ric
from .integrate import propagate

POPULATION_SPEC = {
    "n_objects": 192,
    "n_epoch_groups": 3,
    "perigee_altitude_km": "log-uniform over [320, 1600]",
    "eccentricity": "log-uniform over [5e-4, 5e-2], rejected if perigee < 300 km",
    "inclination_deg": "uniform over [2, 100]",
    "bstar": (
        "log-uniform magnitude over [1e-6, 5e-3]; the sign is negative for one "
        "object in sixteen, which is roughly the proportion of negative B* in the "
        "public catalogue and is included because a negative B* is a legal TLE "
        "that makes the drag term a thrust"
    ),
    "raan_argp_mean_anomaly_deg": "uniform over [0, 360)",
    "epochs": "2026-01-15, 2026-05-15, 2026-09-15 at 00:00 UT, one group each",
    "seed": 20740825,
}
"""The population specification, repeated verbatim in ``DATASET_CARD.md``."""

DEFAULT_SPAN_DAYS = 7.0
DEFAULT_STEPS_PER_REV = 360
DEFAULT_REPORT_EVERY_HOURS = 6.0
DEFAULT_SAMPLE_EVERY = 4

_EPOCH_DAYS_FROM_1949 = (
    # Days from 1949 December 31 00:00 UT, which is the epoch sgp4init counts
    # from. 2026-01-15, 2026-05-15 and 2026-09-15, computed from the Julian
    # dates 2461055.5, 2461175.5 and 2461298.5 minus 2433281.5.
    27774.0,
    27894.0,
    28017.0,
)

TABLE_COLUMNS: tuple[str, ...] = (
    "object_index",
    "epoch_group",
    *FEATURE_NAMES,
    "epoch_age_days",
    "drift_km",
    "drift_radial_km",
    "drift_in_track_km",
    "drift_cross_track_km",
)
"""Column order of the divergence table, as committed to the fixture."""


@dataclass(frozen=True)
class DivergenceTable:
    """The long-format divergence table plus the TLE text it came from.

    One row per (object, epoch age). ``values`` has the columns of
    :data:`TABLE_COLUMNS`.
    """

    values: np.ndarray
    columns: tuple[str, ...]
    tle_text: str
    detail: str

    def column(self, name: str) -> np.ndarray:
        """One column by name."""
        return self.values[:, self.columns.index(name)]

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_objects(self) -> int:
        return int(np.unique(self.column("object_index")).size)

    def design(self) -> np.ndarray:
        """The design matrix: the ten static features then epoch age."""
        names = (*FEATURE_NAMES, "epoch_age_days")
        return np.column_stack([self.column(name) for name in names])

    def save(self, path: str | Path) -> Path:
        """Write a compressed ``.npz`` holding the table, the columns and the TLEs."""
        target = Path(path)
        np.savez_compressed(
            target,
            values=self.values.astype(np.float64),
            columns=np.array(self.columns),
            tle_text=np.array(self.tle_text),
            detail=np.array(self.detail),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> DivergenceTable:
        """Read a table written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                values=archive["values"],
                columns=tuple(str(name) for name in archive["columns"]),
                tle_text=str(archive["tle_text"]),
                detail=str(archive["detail"]),
            )


def synthetic_population_tle(seed: int | None = None) -> str:
    """Generate the population and return it as two-line element set text.

    The returned text has a comment line per object giving its epoch group, so
    the committed fixture is readable, and comment lines are ignored by
    :func:`load_population`.
    """
    spec_seed = int(POPULATION_SPEC["seed"] if seed is None else seed)
    rng = np.random.default_rng(spec_seed)
    n = int(POPULATION_SPEC["n_objects"])
    groups = int(POPULATION_SPEC["n_epoch_groups"])
    lines = [
        "# orbitdrift synthetic population -- generated by "
        "orbitdrift.dataset.synthetic_population_tle",
        f"# seed {spec_seed}, {n} objects, {groups} epoch groups. "
        "See DATASET_CARD.md for the distributions.",
    ]
    for index in range(n):
        group = index % groups
        satrec = Satrec()
        for _ in range(64):
            perigee_km = float(np.exp(rng.uniform(np.log(320.0), np.log(1600.0))))
            ecc = float(np.exp(rng.uniform(np.log(5e-4), np.log(5e-2))))
            perigee_radius = WGS72.radius_km + perigee_km
            a = perigee_radius / (1.0 - ecc)
            if a * (1.0 - ecc) - WGS72.radius_km >= 300.0:
                break
        else:  # pragma: no cover - the rejection almost never runs out
            raise PropagationError("could not draw an orbit with perigee above 300 km")
        no_kozai = np.sqrt(WGS72.mu_km3_s2 / a**3) * 60.0
        magnitude = float(np.exp(rng.uniform(np.log(1e-6), np.log(5e-3))))
        bstar = -magnitude if rng.integers(0, 16) == 0 else magnitude
        satrec.sgp4init(
            SGP4_WGS72,
            "i",
            70000 + index,
            _EPOCH_DAYS_FROM_1949[group],
            bstar,
            0.0,
            0.0,
            ecc,
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(np.deg2rad(rng.uniform(2.0, 100.0))),
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(no_kozai),
            float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        line1, line2 = exporter.export_tle(satrec)
        lines.append(f"# object {index:3d} epoch group {group}")
        lines.append(line1.rstrip())
        lines.append(line2.rstrip())
    return "\n".join(lines) + "\n"


def load_population(text: str) -> list[Satrec]:
    """Parse two-line element set text, ignoring blank and ``#`` comment lines.

    The validation here is this function's own, and deliberately so.
    ``Satrec.twoline2rv`` in the accelerated C++ build of ``sgp4`` 2.27 is
    permissive: a truncated line 2, a line 2 belonging to a different
    catalogue number, a non-ASCII character in a numeric field and a line 1
    that does not begin with ``1`` are all accepted without an exception, and
    produce a silently wrong orbit. Measured, not assumed -- five such inputs
    were tried and none raised. So the checks below are not belt and braces
    around a library that already raises; they are the only checks there are.

    Raises
    ------
    InvalidTleError
        On an odd number of element lines, a line shorter than 69 characters,
        a line whose first character is not the expected digit, or a pair
        whose two catalogue numbers disagree -- naming the pair index in every
        case.
    """
    element_lines = [line for line in text.splitlines() if line[:2] in ("1 ", "2 ")]
    if not element_lines or len(element_lines) % 2 != 0:
        raise InvalidTleError(
            f"found {len(element_lines)} element lines, which is not a positive whole "
            "number of two-line sets"
        )
    satellites: list[Satrec] = []
    for index in range(0, len(element_lines), 2):
        pair = index // 2
        line1, line2 = element_lines[index], element_lines[index + 1]
        for number, line in ((1, line1), (2, line2)):
            if len(line) < 69:
                raise InvalidTleError(
                    f"element set {pair}: line {number} is {len(line)} characters, "
                    "and a two-line element set line is 69"
                )
            if not line.isascii():
                raise InvalidTleError(
                    f"element set {pair}: line {number} holds a non-ASCII character"
                )
        try:
            catalogue1 = int(line1[2:7])
            catalogue2 = int(line2[2:7])
        except ValueError as error:
            raise InvalidTleError(
                f"element set {pair}: columns 3 to 7 are not a catalogue number "
                f"on both lines ({line1[2:7]!r}, {line2[2:7]!r})"
            ) from error
        if catalogue1 != catalogue2:
            raise InvalidTleError(
                f"element set {pair}: line 1 is catalogue {catalogue1} and line 2 is "
                f"catalogue {catalogue2}; the two lines are not for the same object"
            )
        try:
            satellites.append(Satrec.twoline2rv(line1, line2))
        except Exception as error:  # noqa: BLE001 - re-raised with context
            raise InvalidTleError(
                f"element set {pair} could not be initialised: {error}"
            ) from error
    return satellites


def build_divergence_table(
    tle_text: str,
    *,
    span_days: float = DEFAULT_SPAN_DAYS,
    steps_per_rev: int = DEFAULT_STEPS_PER_REV,
    report_every_hours: float = DEFAULT_REPORT_EVERY_HOURS,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    force: ForceModel | None = None,
    reference_period_s: float | None = None,
    progress=None,
) -> DivergenceTable:
    """Propagate the whole population and reduce it to the divergence table.

    Objects are grouped into batches that share an epoch, because the drag and
    third-body terms are functions of absolute time while the integration grid
    is shared. One batch is one vectorised
    :func:`orbitdrift.integrate.propagate` call, which is what makes this
    affordable: the per-step Python cost is paid once for the batch rather than
    once per object.

    The integration step is ``reference_period_s / steps_per_rev``, with
    ``reference_period_s`` defaulting to the **shortest period in the whole
    file**. It is a single number for every batch, and it is recorded in
    :attr:`DivergenceTable.detail`, so that recomputing any subset of the
    population with the same ``reference_period_s`` reproduces the subset's
    rows bit for bit. An earlier version took the step from the shortest
    period in each *batch*, which made the result depend on which other
    objects happened to share an epoch: recomputing three objects on their own
    then differed from the committed table in the fourth significant figure.
    That is the truncation floor showing through a reproducibility claim, and
    the fix is to make the step an input rather than an emergent property of
    the batching.

    ``progress``, when given, is called with ``(batch_index, n_batches, label)``.
    """
    force = force or ForceModel()
    satellites = load_population(tle_text)
    epochs = np.array([epoch_jd(satrec) for satrec in satellites])
    periods = np.array([period_s_from_satrec(satrec) for satrec in satellites])

    reference_period = (
        float(np.min(periods)) if reference_period_s is None else float(reference_period_s)
    )
    step_s = reference_period / float(steps_per_rev)
    n_steps = int(round(span_days * SECONDS_PER_DAY / step_s))
    n_steps -= n_steps % sample_every

    rows: list[np.ndarray] = []
    unique_epochs = np.unique(epochs)
    for batch_index, epoch in enumerate(unique_epochs):
        members = np.flatnonzero(epochs == epoch)
        initial_r = np.empty((members.size, 3))
        initial_v = np.empty((members.size, 3))
        coefficients = np.empty(members.size)
        for slot, object_index in enumerate(members):
            satrec = satellites[object_index]
            code, r, v = satrec.sgp4(float(satrec.jdsatepoch), float(satrec.jdsatepochF))
            if code != 0:
                raise PropagationError(
                    f"object {object_index} (catalogue {satrec.satnum}) returned SGP4 "
                    f"code {code} at its own epoch and cannot seed an integration"
                )
            initial_r[slot] = r
            initial_v[slot] = v
            coefficients[slot] = cd_area_over_mass_m2_kg(float(satrec.bstar))
        if progress is not None:
            progress(
                batch_index,
                int(unique_epochs.size),
                f"{members.size} objects, step {step_s:.4f} s, {n_steps} steps",
            )
        trajectory = propagate(
            initial_r,
            initial_v,
            float(epoch),
            coefficients,
            n_steps=n_steps,
            step_s=step_s,
            force=force,
            sample_every=sample_every,
        )
        for slot, object_index in enumerate(members):
            satrec = satellites[object_index]
            reference_r, reference_v = sgp4_states(satrec, trajectory.times_s)
            delta = trajectory.positions_km[slot] - reference_r
            magnitude = np.linalg.norm(delta, axis=-1)
            ric = np.abs(to_ric(delta, reference_r, reference_v))
            period_s = periods[object_index]
            half = 0.5 * period_s
            ages_s = np.arange(
                half,
                trajectory.times_s[-1] - half + 1.0,
                report_every_hours * 3600.0,
            )
            if ages_s.size == 0:
                continue
            drift = window_rms(trajectory.times_s, magnitude, period_s, ages_s)
            drift_ric = window_rms(trajectory.times_s, ric, period_s, ages_s)
            static = features_from_satrec(satrec).as_array()
            block = np.empty((ages_s.size, len(TABLE_COLUMNS)))
            block[:, 0] = object_index
            block[:, 1] = int(np.flatnonzero(unique_epochs == epoch)[0])
            block[:, 2 : 2 + len(FEATURE_NAMES)] = static
            block[:, 2 + len(FEATURE_NAMES)] = ages_s / SECONDS_PER_DAY
            block[:, -4] = drift
            block[:, -3:] = drift_ric
            rows.append(block)

    if not rows:
        raise PropagationError("no object produced any reportable age")
    values = np.vstack(rows)
    detail = (
        f"{len(satellites)} objects, {unique_epochs.size} epoch groups, "
        f"span {span_days} days, step {step_s:.10f} s from a reference period of "
        f"{reference_period:.6f} s at {steps_per_rev} steps per revolution, "
        f"{n_steps} steps, reported every {report_every_hours} h, drift is the RMS "
        f"over a one-period window; force model {force.label()}"
    )
    return DivergenceTable(values, TABLE_COLUMNS, tle_text, detail)


def split_by_object(
    table: DivergenceTable, *, test_fraction: float = 0.3, seed: int = 20740825
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of a train/test split made **by object**, never by row.

    Splitting by row would put the same object's early ages in training and
    its late ages in test, which is a leak: the features are identical and
    only the age column differs, so a model could interpolate its own
    training rows. Every number in this repository that compares predictors
    uses this split.
    """
    object_index = table.column("object_index")
    unique = np.unique(object_index)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(test_fraction * unique.size)))
    test_objects = set(shuffled[:n_test].tolist())
    is_test = np.array([value in test_objects for value in object_index])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
