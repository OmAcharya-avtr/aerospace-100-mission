"""Cloud-availability models and expected-data weighting.

The availability of an optical downlink is dominated by cloud cover.  Two
simple models are provided:

* :class:`ClimatologyAvailability` -- per-station monthly clear-sky priors
  (probability that the line of sight is usable in a given calendar month).
* :class:`ForecastAvailability` -- wraps a base model and overrides the
  probability inside user-supplied time intervals (e.g. from a weather
  forecast).

Expected delivered data for a pass is modelled as

    E[data, Gbit] = data_rate_gbps * duration_s * p_clear(station, t_culminate)

i.e. an all-or-nothing cloud outcome sampled once per pass at culmination.
This ignores partial-pass cloud transits and rate variation with elevation;
it is the standard first-order site-availability treatment used in optical
ground-network studies (see e.g. Fuchs & Moll 2015, Journal of Optical Communications and
Networking, "Ground station network optimization for space-to-ground
optical communication links" -- methodology reference only, no numbers
reused).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from .frames import to_utc
from .passes import Pass
from .stations import Station


@runtime_checkable
class Availability(Protocol):
    """Anything that maps (station name, UTC time) -> clear-sky probability."""

    def p_clear(self, station: str, when: datetime) -> float:
        """Probability in [0, 1] that the optical path is usable."""
        ...


class ClimatologyAvailability:
    """Monthly climatological clear-sky priors per station.

    ``priors[name]`` is a 12-element sequence (Jan..Dec) of probabilities.
    """

    def __init__(self, priors: dict[str, tuple[float, ...]]):
        self._priors: dict[str, tuple[float, ...]] = {}
        for name, months in priors.items():
            months = tuple(float(p) for p in months)
            if len(months) != 12:
                raise ValueError(f"station '{name}': need 12 monthly values, got {len(months)}")
            if any(not 0.0 <= p <= 1.0 for p in months):
                raise ValueError(f"station '{name}': probabilities must be in [0, 1]")
            self._priors[name] = months

    @classmethod
    def from_stations(cls, stations: list[Station]) -> "ClimatologyAvailability":
        """Build from stations that carry ``monthly_clear_prob``."""
        priors = {}
        for st in stations:
            if st.monthly_clear_prob is None:
                raise ValueError(f"station '{st.name}' has no monthly_clear_prob")
            priors[st.name] = st.monthly_clear_prob
        return cls(priors)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClimatologyAvailability":
        """Load ``{priors: {name: [12 floats]}}`` from YAML."""
        with Path(path).open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or "priors" not in doc:
            raise ValueError(f"{path}: expected a top-level 'priors' mapping")
        return cls({str(k): tuple(v) for k, v in doc["priors"].items()})

    def p_clear(self, station: str, when: datetime) -> float:
        if station not in self._priors:
            raise KeyError(f"no climatological priors for station '{station}'")
        return self._priors[station][to_utc(when).month - 1]


@dataclass(frozen=True)
class ForecastInterval:
    """A forecast override: p_clear applies for station in [start, end)."""

    station: str
    start: datetime
    end: datetime
    p_clear: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", to_utc(self.start))
        object.__setattr__(self, "end", to_utc(self.end))
        if self.end <= self.start:
            raise ValueError("forecast interval end must be after start")
        if not 0.0 <= self.p_clear <= 1.0:
            raise ValueError(f"p_clear must be in [0, 1], got {self.p_clear}")


class ForecastAvailability:
    """Base availability with user-supplied forecast overrides.

    The last matching interval wins (later entries override earlier ones);
    outside all intervals the base model applies.
    """

    def __init__(self, base: Availability, intervals: list[ForecastInterval]):
        self._base = base
        self._intervals = list(intervals)

    def p_clear(self, station: str, when: datetime) -> float:
        when = to_utc(when)
        p = None
        for iv in self._intervals:
            if iv.station == station and iv.start <= when < iv.end:
                p = iv.p_clear
        return self._base.p_clear(station, when) if p is None else p


def expected_data(pass_: Pass, availability: Availability | None) -> float:
    """Expected delivered data volume for a pass [Gbit].

    E = data_rate_gbps * duration_s * p_clear evaluated at culmination
    (see module docstring for assumptions).  ``availability=None`` means
    p_clear = 1 (unweighted).
    """
    if pass_.duration_s <= 0:
        raise ValueError("pass has non-positive duration")
    p = 1.0 if availability is None else float(
        availability.p_clear(pass_.station.name, pass_.t_culminate))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"availability model returned p={p}, outside [0, 1]")
    return pass_.station.data_rate_gbps * pass_.duration_s * p
