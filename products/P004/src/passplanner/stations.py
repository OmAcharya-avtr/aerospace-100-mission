"""Ground-station model and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Station:
    """An optical ground station.

    Parameters
    ----------
    name : unique identifier.
    lat_deg, lon_deg : WGS-84 geodetic latitude / east longitude [deg].
    alt_km : altitude above the WGS-84 ellipsoid [km].
    min_elevation_deg : elevation mask [deg], in [0, 90).  Optical stations
        typically use 20-30 deg to limit atmospheric path length.
    data_rate_gbps : link data rate while the satellite is above the mask
        [Gbit/s].  Modelled as constant over a pass (simplification; real
        optical links vary with range and turbulence).
    monthly_clear_prob : optional 12 climatological clear-sky probabilities
        (Jan..Dec), each in [0, 1].
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 20.0
    data_rate_gbps: float = 1.0
    monthly_clear_prob: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("station name must be a non-empty string")
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"station latitude must be in [-90, 90] deg, got {self.lat_deg}")
        if not -180.0 <= self.lon_deg <= 180.0:
            raise ValueError(f"station longitude must be in [-180, 180] deg, got {self.lon_deg}")
        if not 0.0 <= self.min_elevation_deg < 90.0:
            raise ValueError(
                f"min_elevation_deg must be in [0, 90) deg, got {self.min_elevation_deg}")
        if self.data_rate_gbps <= 0.0:
            raise ValueError(f"data_rate_gbps must be > 0, got {self.data_rate_gbps}")
        if self.monthly_clear_prob is not None:
            probs = tuple(float(p) for p in self.monthly_clear_prob)
            if len(probs) != 12:
                raise ValueError(
                    f"monthly_clear_prob needs 12 values (Jan..Dec), got {len(probs)}")
            if any(not 0.0 <= p <= 1.0 for p in probs):
                raise ValueError("monthly_clear_prob values must be in [0, 1]")
            object.__setattr__(self, "monthly_clear_prob", probs)


def load_stations(path: str | Path) -> list[Station]:
    """Load a list of :class:`Station` from a YAML file.

    Expected layout::

        stations:
          - name: Example-1
            lat_deg: 47.0
            lon_deg: 8.5
            alt_km: 1.5
            min_elevation_deg: 20.0
            data_rate_gbps: 2.5
            monthly_clear_prob: [0.4, 0.4, ...]   # 12 values, optional
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or "stations" not in doc:
        raise ValueError(f"{path}: expected a top-level 'stations' list")
    stations = []
    for entry in doc["stations"]:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each station entry must be a mapping")
        mcp = entry.get("monthly_clear_prob")
        stations.append(Station(
            name=str(entry["name"]),
            lat_deg=float(entry["lat_deg"]),
            lon_deg=float(entry["lon_deg"]),
            alt_km=float(entry.get("alt_km", 0.0)),
            min_elevation_deg=float(entry.get("min_elevation_deg", 20.0)),
            data_rate_gbps=float(entry.get("data_rate_gbps", 1.0)),
            monthly_clear_prob=tuple(mcp) if mcp is not None else None,
        ))
    names = [s.name for s in stations]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: station names must be unique")
    return stations
