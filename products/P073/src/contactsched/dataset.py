"""Seeded synthetic scheduling instances, and the labelled dataset built from them.

Every instance in this repository is synthetic and is produced by this module
from a single integer seed. No real catalogue elements, no real station
coordinates and no real mission request is used anywhere; see
:mod:`contactsched.tle` for why. The consequence, stated plainly here and in
README.md and DATASET_CARD.md, is that **the learned conflict predictor's
measured skill is skill on this generator**. It is evidence that the features
carry signal about contention outcomes in a model of a ground-station network.
It is not evidence about any real network's request mix, and the generator's
distributional choices -- priority skew, duration range, contact-count range --
are visible in the code below precisely so a reader can judge how far they are
from their own traffic.

Labelling
---------
A request is labelled **unsatisfiable** when the exact method, with no fairness
constraints, delivers fewer contacts than the request's declared
``min_contacts``. That is the thing a planner would want warning of: the exact
optimum is the best any scheduler can do over the candidate grid, so a request
starved there is starved by the geometry and the competition and not by a weak
heuristic. Labelling against the greedy baseline instead would teach the model
the baseline's quirks.

Train and test instances come from **disjoint seed ranges**, and the split is by
instance, never by request. Requests inside one instance share its network, its
horizon and its contention, so splitting by request would leak.

Compute budget
--------------
Window computation plus one exact solve per instance. On the two-core machine
these numbers were measured on, 360 instances take well under three minutes;
``validation/validate_dataset_output.txt`` records the measured figure and the
core count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import Assignment
from .ephemeris import Scenario, Spacecraft
from .exact import solve_exact
from .features import FEATURE_NAMES, request_features
from .problem import ContactRequest, SchedulingProblem, build_problem
from .station import GroundStation, HorizonMask
from .tle import OrbitElements
from .visibility import all_windows

STATION_POOL: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("SVALBARD", 78.23, 15.39, 0.45, 5.0, 300.0),
    ("PUNTA-ARENAS", -53.16, -70.91, 0.03, 5.0, 420.0),
    ("KOUROU", 5.25, -52.80, 0.01, 7.0, 360.0),
    ("TROLL", -72.01, 2.53, 1.30, 5.0, 300.0),
    ("ALICE-SPRINGS", -23.70, 133.88, 0.55, 6.0, 300.0),
    ("FAIRBANKS", 64.86, -147.85, 0.14, 5.0, 300.0),
)
"""Declared station pool: name, latitude, longitude, altitude, mask floor, turnaround.

Coordinates are near real high-latitude and equatorial sites because that is
where the geometry of polar-orbit contention lives, but these are declared
inputs with round numbers and are not any operator's facility. Nothing about a
real station's mask, keyhole, slew rate or service agreement is modelled.
"""

INCLINATION_POOL: tuple[float, ...] = (97.4, 97.8, 98.4, 51.6, 63.4, 74.0)
"""Inclinations the generator draws from, degrees. Four near-polar, two not."""

MISSION_POOL: tuple[str, ...] = ("POLARIS", "TERRASCAN", "EQUINOX", "MERIDIAN", "HELIOS")
"""Mission labels. Several spacecraft may share one, which is what makes
per-mission fairness differ from per-request fairness."""


@dataclass
class Instance:
    """One synthetic scheduling instance and everything needed to re-derive it.

    Attributes
    ----------
    seed : int
        The only input. Re-running :func:`generate_instance` with this seed
        reproduces the instance exactly.
    scenario : Scenario
        Epoch and horizon.
    stations : tuple of GroundStation
        The drawn subset of :data:`STATION_POOL`.
    fleet : tuple of Spacecraft
        The drawn spacecraft.
    problem : SchedulingProblem
        Windows, requests and candidates.
    load : float
        The contention knob actually drawn, dimensionless; higher means more
        declared demand per unit of supply.
    """

    seed: int
    scenario: Scenario
    stations: tuple[GroundStation, ...]
    fleet: tuple[Spacecraft, ...]
    problem: SchedulingProblem
    load: float


def generate_instance(
    seed: int,
    n_stations: int | None = None,
    n_spacecraft: int | None = None,
    n_requests: int | None = None,
    horizon_s: float = 43200.0,
    scan_step_s: float = 45.0,
    slot_step_s: float = 240.0,
) -> Instance:
    """Draw one synthetic instance from ``seed``.

    Parameters
    ----------
    seed : int
        Any non-negative integer.
    n_stations, n_spacecraft, n_requests : int or None
        Fixed sizes, or ``None`` to draw them (3 to 4 antennas, 4 to 6
        spacecraft, 7 to 11 requests).
    horizon_s : float
        Scheduling horizon, seconds.
    scan_step_s : float
        Visibility scan step, seconds. 45 s is fine enough for the drawn orbits;
        ``validation/validate_convergence.py`` is the measurement that says so.
    slot_step_s : float
        Candidate grid step, seconds.

    Returns
    -------
    Instance
    """
    rng = np.random.default_rng(int(seed))
    n_stations = int(n_stations if n_stations is not None else rng.integers(3, 5))
    n_spacecraft = int(n_spacecraft if n_spacecraft is not None else rng.integers(4, 7))
    n_requests = int(n_requests if n_requests is not None else rng.integers(7, 12))

    chosen = rng.choice(len(STATION_POOL), size=n_stations, replace=False)
    stations = tuple(
        GroundStation(
            name=STATION_POOL[i][0],
            lat_deg=STATION_POOL[i][1],
            lon_deg=STATION_POOL[i][2],
            alt_km=STATION_POOL[i][3],
            mask=HorizonMask(floor_deg=STATION_POOL[i][4]),
            turnaround_s=STATION_POOL[i][5],
        )
        for i in sorted(int(i) for i in chosen)
    )

    missions = list(MISSION_POOL[: max(2, min(len(MISSION_POOL), n_spacecraft - 1))])
    fleet: list[Spacecraft] = []
    for index in range(n_spacecraft):
        fleet.append(
            Spacecraft(
                OrbitElements(
                    satnum=90000 + index + 1,
                    epoch_year=26,
                    epoch_day=120.0,
                    inclination_deg=float(rng.choice(INCLINATION_POOL)),
                    raan_deg=float(rng.uniform(0.0, 360.0)),
                    eccentricity=float(rng.uniform(0.0002, 0.004)),
                    argp_deg=float(rng.uniform(0.0, 360.0)),
                    mean_anomaly_deg=float(rng.uniform(0.0, 360.0)),
                    mean_motion_rev_day=float(rng.uniform(13.9, 15.5)),
                    bstar=float(rng.uniform(1e-5, 3e-4)),
                    name=f"SC-{index + 1:02d}",
                ),
                mission=missions[index % len(missions)],
            )
        )
    fleet_tuple = tuple(fleet)

    scenario = Scenario.from_calendar(2026, 4, 30, 0, 0, 0.0, duration_s=float(horizon_s))
    windows = tuple(all_windows(scenario, stations, fleet_tuple, scan_step_s, 0.5))

    load = float(rng.uniform(0.7, 2.3))
    station_names = tuple(s.name for s in stations)
    requests: list[ContactRequest] = []
    for index in range(n_requests):
        spacecraft = fleet_tuple[int(rng.integers(0, len(fleet_tuple)))]
        priority = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]))
        duration = float(rng.choice([180.0, 300.0, 420.0, 600.0]))
        min_contacts = int(np.clip(round(rng.uniform(1.0, 2.0 + 2.0 * load)), 1, 6))
        max_contacts = int(min_contacts + rng.integers(0, 2))
        if rng.random() < 0.45:
            count = int(rng.integers(1, len(station_names) + 1))
            allowed = tuple(
                station_names[i]
                for i in sorted(
                    int(i) for i in rng.choice(len(station_names), size=count, replace=False)
                )
            )
        else:
            allowed = None
        requests.append(
            ContactRequest(
                request_id=f"R{index + 1:02d}",
                mission=spacecraft.mission,
                spacecraft=spacecraft.name,
                priority=priority,
                duration_s=duration,
                min_contacts=min_contacts,
                max_contacts=max_contacts,
                stations=allowed,
            )
        )

    problem = build_problem(stations, tuple(requests), windows, slot_step_s=slot_step_s)
    return Instance(
        seed=int(seed),
        scenario=scenario,
        stations=stations,
        fleet=fleet_tuple,
        problem=problem,
        load=load,
    )


def label_instance(
    instance: Instance, time_limit_s: float = 30.0
) -> tuple[Assignment, dict[str, int]]:
    """Exact solve and the per-request unsatisfiable label.

    Returns
    -------
    tuple
        ``(assignment, {request_id: 0 or 1})``. The label is 1 when the exact
        optimum leaves the request below its declared ``min_contacts``.
    """
    assignment = solve_exact(instance.problem, "weighted_contacts", None, "highs", time_limit_s)
    counts = assignment.contacts_by_request(instance.problem)
    labels = {
        r.request_id: int(counts[r.request_id] < int(r.min_contacts))
        for r in instance.problem.requests
    }
    return assignment, labels


def instance_rows(
    instance: Instance, bin_s: float = 1800.0, time_limit_s: float = 30.0
) -> list[dict[str, float | int | str]]:
    """One row per request: identifiers, every feature, and the label."""
    features = request_features(instance.problem, instance.scenario.duration_s, bin_s)
    _assignment, labels = label_instance(instance, time_limit_s)
    rows: list[dict[str, float | int | str]] = []
    for request in instance.problem.requests:
        row: dict[str, float | int | str] = {
            "seed": int(instance.seed),
            "request_id": request.request_id,
            "mission": request.mission,
            "spacecraft": request.spacecraft,
            "load": float(instance.load),
        }
        for name in FEATURE_NAMES:
            row[name] = float(features[request.request_id][name])
        row["unsatisfiable"] = int(labels[request.request_id])
        rows.append(row)
    return rows


def build_dataset(
    seeds: range | list[int],
    bin_s: float = 1800.0,
    horizon_s: float = 43200.0,
    time_limit_s: float = 30.0,
):
    """Labelled dataset over ``seeds`` as a :class:`pandas.DataFrame`.

    One row per request. Columns are ``seed``, ``request_id``, ``mission``,
    ``spacecraft``, ``load``, then :data:`contactsched.features.FEATURE_NAMES`,
    then ``unsatisfiable``.

    ``pandas`` is imported inside the function so that importing
    :mod:`contactsched` does not pay for it.
    """
    import pandas as pd

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        instance = generate_instance(int(seed), horizon_s=horizon_s)
        rows.extend(instance_rows(instance, bin_s=bin_s, time_limit_s=time_limit_s))
    return pd.DataFrame(rows)


def split_by_instance(frame, train_seeds: range | list[int], test_seeds: range | list[int]):
    """Split a dataset by seed, never by request.

    Raises
    ------
    ValueError
        If the two seed sets intersect. The check is here rather than in a
        comment because an overlapping split is the one mistake in this file
        that would silently inflate every reported score.
    """
    train_set = set(int(s) for s in train_seeds)
    test_set = set(int(s) for s in test_seeds)
    shared = train_set & test_set
    if shared:
        raise ValueError(f"train and test seed ranges overlap on {sorted(shared)[:10]}")
    return (
        frame[frame["seed"].isin(sorted(train_set))].reset_index(drop=True),
        frame[frame["seed"].isin(sorted(test_set))].reset_index(drop=True),
    )


def problem_of(instance: Instance) -> SchedulingProblem:
    """Convenience accessor used by the examples."""
    return instance.problem
