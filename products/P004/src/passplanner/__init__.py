"""passplanner -- optical ground-station contact planner.

TLE propagation (SGP4) -> topocentric passes -> cloud-availability weighting
-> greedy / ILP downlink scheduling.  Research-grade; not flight-qualified.
"""

from .availability import (
    Availability,
    ClimatologyAvailability,
    ForecastAvailability,
    ForecastInterval,
    expected_data,
)
from .mlmodel import ClimatologyBaselineModel, PassSuccessModel
from .passes import TLE, Pass, find_passes, find_passes_from_position_fn
from .scheduler import ScheduleResult, passes_conflict, schedule_greedy, schedule_ilp
from .stations import Station, load_stations
from .synthdata import FEATURE_NAMES, WeatherDataset, generate_dataset

__version__ = "0.1.0"

__all__ = [
    "TLE",
    "Pass",
    "Station",
    "Availability",
    "ClimatologyAvailability",
    "ForecastAvailability",
    "ForecastInterval",
    "ScheduleResult",
    "ClimatologyBaselineModel",
    "PassSuccessModel",
    "WeatherDataset",
    "FEATURE_NAMES",
    "find_passes",
    "find_passes_from_position_fn",
    "expected_data",
    "schedule_greedy",
    "schedule_ilp",
    "passes_conflict",
    "load_stations",
    "generate_dataset",
    "__version__",
]
