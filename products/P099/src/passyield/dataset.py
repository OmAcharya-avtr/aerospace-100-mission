"""The committed per-pass record: station catalogue, schema, loader and splits.

THE STATIONS ARE SYNTHETIC AND SAID SO
--------------------------------------
:data:`STATIONS` is a catalogue of **eight synthetic ground stations**. The
geodetic coordinates are declared placements chosen to span latitude and to
give eight visibly different pass statistics against the committed element
set; the figures of merit, zenith attenuations and elevation masks are
declared design values. **None of them is a measurement of, or a claim about,
any real ground station or network.** The names are deliberately neutral
(``GS-01`` and so on) so that no row of this catalogue can be read as a
statement about a named facility. ``DATASET_CARD.md`` repeats this.

The element set, by contrast, is real and published: see
:mod:`passyield.geometry`.

THE SCHEMA, AND WHY THE EVENT INTERVALS ARE STORED IN FULL
----------------------------------------------------------
Each row carries the pass geometry, the pre-pass state, the **event timeline**
and the decomposition the generator computed. The timeline is stored as its
full interval list in a compact ``start:duration;start:duration`` form rather
than only as totals, so that :func:`events_of` reconstructs the exact
:class:`passyield.decompose.PassEvents` and
``validation/validate_decomposition.py`` can recompute every decomposition
from the events and compare against the committed columns. Storing only the
totals would make the committed decomposition unfalsifiable from the
committed record, which is the state a validation suite exists to avoid.

THE TWO SPLITS
--------------
The specification requires held-out **stations** and held-out **time**, scored
separately.

- :func:`station_split` holds out whole stations. The grouping key is the
  station name, and no station contributes rows to both sides. This is the
  harder split: the model must transfer to a figure of merit, a mask and a
  latitude it never trained on.
- :func:`time_split` holds out the **last** fraction of the campaign in
  chronological order, never a random subset, because a random subset lets the
  model interpolate between passes that straddle the held-out one in time and
  a schedule cannot do that.

The gap between the two is itself a reported quantity; see
``validation/validate_learned.py``.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .decompose import PassEvents
from .geometry import GroundStation, PassWindow
from .predictor import FEATURE_NAMES

#: Product root, anchored on this file and never on the caller's working
#: directory. Three products of this mission failed a release gate because a
#: path was resolved against the caller's directory instead.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

#: The committed per-pass record.
DEFAULT_RECORD_PATH = PACKAGE_ROOT / "data" / "passes.csv"

#: The generator's manifest: seeds, hashes, counts.
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "data" / "manifest.json"

#: Eight synthetic ground stations. See the module docstring: declared
#: placements and declared design figures, not measurements of real sites.
STATIONS: tuple[GroundStation, ...] = (
    GroundStation("GS-01", 57.00, -2.20, 120.0, 10.0, 22.0, 0.50),
    GroundStation("GS-02", -46.53, 168.38, 5.0, 10.0, 20.5, 0.55),
    GroundStation("GS-03", -53.00, -70.85, 40.0, 12.0, 19.5, 0.60),
    GroundStation("GS-04", -25.89, 27.69, 1400.0, 10.0, 23.0, 0.35),
    GroundStation("GS-05", 56.00, -147.50, 200.0, 10.0, 21.5, 0.45),
    GroundStation("GS-06", 1.35, 103.82, 20.0, 15.0, 20.0, 0.70),
    GroundStation("GS-07", -33.15, -70.67, 730.0, 10.0, 22.5, 0.40),
    GroundStation("GS-08", 40.00, 20.00, 100.0, 10.0, 24.0, 0.30),
)

#: Stations held out by :func:`station_split` by default. Two of eight,
#: chosen as the best and the worst figure of merit in the catalogue so that
#: the split tests extrapolation in the station's own physics rather than
#: interpolation between two similar sites. Declared here, before any
#: measurement, and never changed to improve a score.
DEFAULT_HELDOUT_STATIONS: tuple[str, ...] = ("GS-03", "GS-08")

#: Fraction of the campaign held out by :func:`time_split` by default.
DEFAULT_TIME_HOLDOUT_FRACTION = 0.25


def station_by_name(name: str) -> GroundStation:
    """The catalogue entry with this name."""
    for station in STATIONS:
        if station.name == name:
            return station
    raise ValueError(f"no station named {name!r} in the catalogue; known: "
                     f"{[item.name for item in STATIONS]}")


@dataclass(frozen=True)
class PassRow:
    """One pass of the committed record. Units are in each field's name."""

    pass_id: int
    station: str
    rise_jd: float
    rise_fr: float
    campaign_hour: float
    local_hour: float
    duration_s: float
    max_elevation_deg: float
    min_slant_range_km: float
    rise_azimuth_deg: float
    slew_angle_deg: float
    backlog_mbit: float
    hours_since_last_contact: float
    prior_pass_outage_count: int
    acquisition_s: float
    outage_count: int
    resync_intervals: str
    idle_intervals: str
    nominal_mbit: float
    capacity_mbit: float
    realised_mbit: float
    shortfall_mbit: float
    geometry_mbit: float
    acquisition_mbit: float
    resync_mbit: float
    idle_mbit: float
    residual_mbit: float

    @property
    def realised_fraction(self) -> float:
        """``realised / capacity``, dimensionless."""
        return self.realised_mbit / self.capacity_mbit

    @property
    def resync_total_s(self) -> float:
        """Total declared outage time, seconds."""
        return sum(duration for _start, duration in parse_intervals(self.resync_intervals))

    @property
    def idle_total_s(self) -> float:
        """Total declared idle time, seconds."""
        return sum(duration for _start, duration in parse_intervals(self.idle_intervals))


#: CSV column order. Derived from :class:`PassRow` so the two cannot drift.
COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(PassRow))

_INTEGER_COLUMNS = frozenset({"pass_id", "prior_pass_outage_count", "outage_count"})
_TEXT_COLUMNS = frozenset({"station", "resync_intervals", "idle_intervals"})
#: Columns written at full double round-trip precision rather than at nine
#: significant digits. The rise instant is the one quantity where nine digits
#: is not enough: it resolved the mask crossing to about 9 ms, which moves the
#: endpoint elevation by 2e-4 degrees and trips the mask-clamp guard in
#: :mod:`passyield.decompose` when the window is reconstructed from the record.
_FULL_PRECISION_COLUMNS = frozenset({"rise_jd", "rise_fr"})


#: Decimal places the interval fields are written to, seconds.
#:
#: Nine, not six. The active set's endpoints come from these fields, so a
#: rounding here times the rate ceiling of 250 Mbit/s is a volume error: at
#: six places that is 2.5e-4 Mbit per endpoint, which on a pass whose realised
#: volume is only a few hundred Mbit is a relative error near 1e-7 and large
#: enough to need a tolerance that would hide a real defect. At nine places it
#: is 1.25e-7 Mbit per endpoint. The cost is three characters per number.
INTERVAL_DECIMAL_PLACES = 9


def format_intervals(intervals: tuple[tuple[float, float], ...]) -> str:
    """Render ``((start, duration), ...)`` as ``start:duration;...`` seconds."""
    places = INTERVAL_DECIMAL_PLACES
    return ";".join(f"{start:.{places}f}:{duration:.{places}f}" for start, duration in intervals)


def parse_intervals(text: str) -> tuple[tuple[float, float], ...]:
    """Inverse of :func:`format_intervals`. Raises on a malformed field."""
    if not text:
        return ()
    parsed: list[tuple[float, float]] = []
    for piece in text.split(";"):
        parts = piece.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"malformed interval field {piece!r}: expected 'start:duration' in seconds"
            )
        parsed.append((float(parts[0]), float(parts[1])))
    return tuple(parsed)


def window_of(row: PassRow) -> PassWindow:
    """Reconstruct the exact :class:`passyield.geometry.PassWindow` of a row.

    The rise instant round-trips to full double precision because
    ``rise_jd`` and ``rise_fr`` are written at seventeen significant digits;
    see :data:`_FULL_PRECISION_COLUMNS` for why that is not optional.
    """
    return PassWindow(
        station=row.station,
        rise_jd=row.rise_jd,
        rise_fr=row.rise_fr,
        duration_s=row.duration_s,
        max_elevation_deg=row.max_elevation_deg,
        min_slant_range_km=row.min_slant_range_km,
    )


def events_of(row: PassRow) -> PassEvents:
    """The exact event timeline of a row, for re-decomposition."""
    return PassEvents(
        acquisition_s=row.acquisition_s,
        resync_intervals_s=parse_intervals(row.resync_intervals),
        idle_intervals_s=parse_intervals(row.idle_intervals),
    )


def load_passes(path: str | Path | None = None) -> list[PassRow]:
    """Load and validate the committed per-pass record.

    Raises
    ------
    FileNotFoundError
        If the record is missing, with the path rendered relative to the
        product root rather than absolutely.
    ValueError
        If the header does not match :data:`COLUMNS` exactly, if a field will
        not parse, or if a row violates a declared invariant -- a non-positive
        duration or capacity, a negative acquisition time, a station not in
        the catalogue.
    """
    target = Path(path) if path is not None else DEFAULT_RECORD_PATH
    if not target.is_file():
        try:
            shown = target.resolve().relative_to(PACKAGE_ROOT)
        except ValueError:
            shown = Path(target.name)
        raise FileNotFoundError(
            f"the per-pass record {shown} is missing. Regenerate it with "
            "'python3 data/generate_passes.py' from the product root."
        )
    known = {station.name for station in STATIONS}
    rows: list[PassRow] = []
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("the per-pass record is empty: it has no header")
        if tuple(header) != COLUMNS:
            raise ValueError(
                f"the per-pass record header does not match the schema. Expected {COLUMNS}, "
                f"got {tuple(header)}"
            )
        for number, raw in enumerate(reader, start=2):
            if len(raw) != len(COLUMNS):
                raise ValueError(
                    f"line {number} of the per-pass record has {len(raw)} fields, "
                    f"expected {len(COLUMNS)}"
                )
            values: dict[str, object] = {}
            for name, text in zip(COLUMNS, raw, strict=True):
                if name in _TEXT_COLUMNS:
                    values[name] = text
                elif name in _INTEGER_COLUMNS:
                    values[name] = int(text)
                else:
                    values[name] = float(text)
            row = PassRow(**values)  # type: ignore[arg-type]
            _validate(row, number, known)
            rows.append(row)
    if not rows:
        raise ValueError("the per-pass record has a header but no rows")
    return rows


def _validate(row: PassRow, line: int, known: set[str]) -> None:
    """Row invariants. Each failure names the line and the quantity."""
    if row.station not in known:
        raise ValueError(f"line {line}: station {row.station!r} is not in the catalogue")
    if row.duration_s <= 0.0:
        raise ValueError(f"line {line}: duration_s must be positive, got {row.duration_s!r}")
    if row.capacity_mbit <= 0.0:
        raise ValueError(f"line {line}: capacity_mbit must be positive, got {row.capacity_mbit!r}")
    if row.acquisition_s < 0.0:
        raise ValueError(
            f"line {line}: acquisition_s must be non-negative, got {row.acquisition_s!r}"
        )
    if row.realised_mbit < 0.0:
        raise ValueError(
            f"line {line}: realised_mbit must be non-negative -- a negative realised yield is the "
            f"defect this product exists to prevent -- got {row.realised_mbit!r}"
        )
    if row.realised_mbit > row.capacity_mbit * (1.0 + 1.0e-9):
        raise ValueError(
            f"line {line}: realised_mbit {row.realised_mbit!r} exceeds the window capacity "
            f"{row.capacity_mbit!r}, which is impossible"
        )
    parse_intervals(row.resync_intervals)
    parse_intervals(row.idle_intervals)


def write_passes(rows: list[PassRow], path: str | Path) -> None:
    """Write a record in the canonical schema and column order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_render(row, name) for name in COLUMNS])


def _render(row: PassRow, name: str) -> str:
    """One CSV field, at the precision that column needs."""
    value = getattr(row, name)
    if name in _TEXT_COLUMNS or name in _INTEGER_COLUMNS:
        return str(value)
    if name in _FULL_PRECISION_COLUMNS:
        return f"{value:.17g}"
    return f"{value:.9g}"


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, hex. Used to pin the committed record."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_matrix(rows: list[PassRow]) -> np.ndarray:
    """Pre-pass features, ``(n, len(FEATURE_NAMES))``, in :data:`FEATURE_NAMES` order.

    Every column is knowable before the contact starts. The station enters
    through its own physical figures and never through its identity; see the
    :mod:`passyield.predictor` docstring.
    """
    if not rows:
        raise ValueError("feature_matrix needs at least one row")
    columns = []
    for row in rows:
        station = station_by_name(row.station)
        hour_angle = 2.0 * np.pi * row.local_hour / 24.0
        azimuth = np.radians(row.rise_azimuth_deg)
        columns.append(
            [
                row.duration_s,
                row.max_elevation_deg,
                row.min_slant_range_km,
                row.capacity_mbit,
                row.capacity_mbit / row.duration_s,
                station.gain_over_temperature_db,
                station.zenith_attenuation_db,
                station.mask_deg,
                abs(station.latitude_deg),
                row.slew_angle_deg,
                row.backlog_mbit,
                row.hours_since_last_contact,
                float(row.prior_pass_outage_count),
                float(np.sin(hour_angle)),
                float(np.cos(hour_angle)),
                float(np.sin(azimuth)),
                float(np.cos(azimuth)),
            ]
        )
    matrix = np.asarray(columns, dtype=float)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature_matrix produced {matrix.shape[1]} columns but FEATURE_NAMES has "
            f"{len(FEATURE_NAMES)}; the two must not drift"
        )
    return matrix


def target_vector(rows: list[PassRow]) -> np.ndarray:
    """Realised fraction of every row, dimensionless."""
    return np.asarray([row.realised_fraction for row in rows], dtype=float)


@dataclass(frozen=True)
class Split:
    """A train/test partition with its own name and the reason for it."""

    name: str
    description: str
    train: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(
                f"split {self.name!r} leaks {len(overlap)} indices between train and test"
            )
        if not self.train or not self.test:
            raise ValueError(f"split {self.name!r} has an empty side")


def station_split(
    rows: list[PassRow], holdout: tuple[str, ...] = DEFAULT_HELDOUT_STATIONS
) -> Split:
    """Hold out whole stations. No station appears on both sides."""
    known = {station.name for station in STATIONS}
    unknown = sorted(set(holdout) - known)
    if unknown:
        raise ValueError(f"holdout names {unknown} are not in the station catalogue")
    held = set(holdout)
    train = tuple(index for index, row in enumerate(rows) if row.station not in held)
    test = tuple(index for index, row in enumerate(rows) if row.station in held)
    return Split(
        name="held-out stations",
        description=(
            "whole stations held out: " + ", ".join(sorted(held)) + ". The model must transfer to "
            "a figure of merit, an elevation mask and a latitude it never trained on."
        ),
        train=train,
        test=test,
    )


def time_split(
    rows: list[PassRow], holdout_fraction: float = DEFAULT_TIME_HOLDOUT_FRACTION
) -> Split:
    """Hold out the last ``holdout_fraction`` of the campaign, chronologically."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            f"holdout_fraction must lie strictly inside (0, 1), got {holdout_fraction!r}"
        )
    order = sorted(range(len(rows)), key=lambda index: rows[index].campaign_hour)
    cut = int(round(len(order) * (1.0 - holdout_fraction)))
    cut = min(max(cut, 1), len(order) - 1)
    boundary = rows[order[cut]].campaign_hour
    train = tuple(index for index in order[:cut])
    test = tuple(index for index in order[cut:])
    return Split(
        name="held-out time",
        description=(
            f"the last {holdout_fraction:.0%} of the campaign held out in chronological order, "
            f"from campaign hour {boundary:.3f} onward. Not a random subset: a schedule cannot "
            "interpolate between passes that straddle the one it is forecasting."
        ),
        train=train,
        test=test,
    )
