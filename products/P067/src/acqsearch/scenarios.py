"""Synthetic acquisition scenarios: declared ranges, fixed seed, JSON on disk.

A scenario is a complete set of declared terminal and link parameters. Nothing
here is measured from a real terminal; the ranges below are chosen to bracket
values that are plausible for a small free-space optical crosslink terminal,
and DATASET_CARD.md states that plainly. Every number a scenario carries is an
input to the physics in :mod:`acqsearch.detection` and
:mod:`acqsearch.geometry`, not an output, so the dataset cannot smuggle in a
result: the only thing it determines is which corners of the parameter space
the policies are compared on.

Generation is deterministic: ``generate_scenarios(n, seed)`` draws from
``numpy.random.default_rng(seed)`` and returns the same list every time.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .detection import PoissonDwellDetector
from .geometry import PointingUncertainty, TerminalGeometry, build_cell_grid
from .patterns import cell_visit_order, standard_paths
from .policy import AcquisitionEnvironment

#: Declared sampling ranges. Units are given by the suffix of each key.
PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "sigma_rad": (1.00e-4, 2.20e-4),
    "beam_divergence_rad": (8.0e-5, 1.4e-4),
    "fov_ratio": (1.5, 3.0),
    "slew_rate_rad_s": (5.0e-3, 2.0e-2),
    "settle_time_s": (1.0e-3, 4.0e-3),
    "signal_rate_hz": (1.2e4, 3.5e4),
    "background_rate_hz": (1.0e3, 4.0e3),
    "dwell_s": (8.0e-4, 2.2e-3),
    "verification_time_s": (0.2, 1.0),
    "rytov_variance": (0.05, 0.60),
}

#: Count threshold is integer-valued and is drawn from this inclusive range.
THRESHOLD_RANGE: tuple[int, int] = (6, 10)

#: Deadline, in units of the nominal time for one full pass of the grid.
DEADLINE_PERIODS: float = 3.0


@dataclass(frozen=True)
class Scenario:
    """One declared acquisition scenario. All angles radians, all times seconds."""

    scenario_id: int
    sigma_rad: float
    beam_divergence_rad: float
    detector_fov_rad: float
    slew_rate_rad_s: float
    settle_time_s: float
    signal_rate_hz: float
    background_rate_hz: float
    threshold_counts: int
    dwell_s: float
    verification_time_s: float
    rytov_variance: float

    def uncertainty(self) -> PointingUncertainty:
        """The pointing-uncertainty prior of this scenario."""
        return PointingUncertainty(self.sigma_rad)

    def geometry(self) -> TerminalGeometry:
        """The terminal geometry of this scenario."""
        return TerminalGeometry(
            beam_divergence_rad=self.beam_divergence_rad,
            detector_fov_rad=self.detector_fov_rad,
            slew_rate_rad_s=self.slew_rate_rad_s,
            settle_time_s=self.settle_time_s,
        )

    def detector(self) -> PoissonDwellDetector:
        """The dwell detector of this scenario."""
        return PoissonDwellDetector(
            signal_rate_hz=self.signal_rate_hz,
            background_rate_hz=self.background_rate_hz,
            threshold_counts=self.threshold_counts,
            rytov_variance=self.rytov_variance,
        )


def generate_scenarios(n: int, seed: int) -> list[Scenario]:
    """Draw ``n`` scenarios uniformly from :data:`PARAMETER_RANGES`.

    Independent uniform marginals, no correlation structure; the parameters of
    a real terminal are not independent, and DATASET_CARD.md records that as
    the dataset's principal limitation.
    """
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    rng = np.random.default_rng(int(seed))
    scenarios: list[Scenario] = []
    for index in range(int(n)):
        draw = {key: float(rng.uniform(*bounds)) for key, bounds in PARAMETER_RANGES.items()}
        threshold = int(rng.integers(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1] + 1))
        scenarios.append(
            Scenario(
                scenario_id=index,
                sigma_rad=draw["sigma_rad"],
                beam_divergence_rad=draw["beam_divergence_rad"],
                detector_fov_rad=draw["beam_divergence_rad"] * draw["fov_ratio"],
                slew_rate_rad_s=draw["slew_rate_rad_s"],
                settle_time_s=draw["settle_time_s"],
                signal_rate_hz=draw["signal_rate_hz"],
                background_rate_hz=draw["background_rate_hz"],
                threshold_counts=threshold,
                dwell_s=draw["dwell_s"],
                verification_time_s=draw["verification_time_s"],
                rytov_variance=draw["rytov_variance"],
            )
        )
    return scenarios


def save_scenarios(scenarios: list[Scenario], path: str | Path) -> None:
    """Write scenarios to JSON, one object per scenario, sorted by id."""
    payload = {
        "format": "acqsearch-scenarios-1",
        "parameter_ranges": {k: list(v) for k, v in PARAMETER_RANGES.items()},
        "threshold_range": list(THRESHOLD_RANGE),
        "scenarios": [asdict(s) for s in sorted(scenarios, key=lambda s: s.scenario_id)],
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Read scenarios written by :func:`save_scenarios`."""
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "acqsearch-scenarios-1":
        raise ValueError(f"unexpected scenario file format: {payload.get('format')!r}")
    return [Scenario(**entry) for entry in payload["scenarios"]]


def split_scenarios(
    scenarios: list[Scenario], train_fraction: float = 0.65
) -> tuple[list[Scenario], list[Scenario]]:
    """Split by ``scenario_id`` into train and held-out test lists.

    The split is by index and is therefore deterministic and reproducible from
    the file alone; no scenario appears in both halves.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction!r}")
    ordered = sorted(scenarios, key=lambda s: s.scenario_id)
    cut = int(math.floor(len(ordered) * train_fraction))
    if cut < 1 or cut >= len(ordered):
        raise ValueError("train_fraction leaves one side of the split empty")
    return ordered[:cut], ordered[cut:]


def build_environment(
    scenario: Scenario,
    extent_sigma: float = 2.5,
    deadline_periods: float = DEADLINE_PERIODS,
) -> tuple[AcquisitionEnvironment, dict[str, np.ndarray]]:
    """Build the environment and the four fixed-pattern cell orders for a scenario.

    Returns ``(environment, orders)`` where ``orders`` maps pattern name to a
    cell visit order. The deadline is ``deadline_periods`` times the nominal
    time for one full pass, so scenarios with different timing are censored at
    a comparable point in their own scan rather than at a common wall clock.
    """
    uncertainty = scenario.uncertainty()
    geometry = scenario.geometry()
    detector = scenario.detector()
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=extent_sigma)
    environment = AcquisitionEnvironment.build(
        grid=grid,
        geometry=geometry,
        detector=detector,
        uncertainty=uncertainty,
        dwell_s=scenario.dwell_s,
        verification_time_s=scenario.verification_time_s,
        deadline_s=1.0,
    )
    deadline = float(deadline_periods) * environment.nominal_period_s
    environment = AcquisitionEnvironment(
        grid=environment.grid,
        geometry=environment.geometry,
        cell_detection=environment.cell_detection,
        dwell_s=environment.dwell_s,
        false_alarm_probability=environment.false_alarm_probability,
        verification_time_s=environment.verification_time_s,
        deadline_s=deadline,
        nominal_period_s=environment.nominal_period_s,
        neighbours=environment.neighbours,
    )
    paths = standard_paths(grid, seed=20261016 + scenario.scenario_id)
    orders = {name: cell_visit_order(path, grid) for name, path in paths.items()}
    return environment, orders
