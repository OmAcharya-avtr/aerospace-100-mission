"""Shipped problems: five hand-declared instances and a seeded random family.

Two kinds, for two purposes.

**Hand-declared instances** (:func:`named_instance`) each exist to make one
constraint bind, so that the feasibility report can be read against a known
answer. They use piecewise power profiles, which the predictor path does not.
They are small, fixed and documented, and ``tiny-enumerable`` is sized so that
exhaustive enumeration over every assignment is tractable, which is what
``validation/validate_exact.py`` needs.

**The seeded family** (:func:`instance_family`) is a reproducible population of
oversubscribed instances used to measure the greedy-versus-exact optimality gap
and, through :class:`InstanceSpec`, to run the four planning modes
(oracle / analytic / learned point / learned robust) on identical instances.

Sizing of the random family, stated because the numbers it produces depend on it
-------------------------------------------------------------------------------
An instance is interesting only if it is oversubscribed: enough demand that not
every activity fits, not so much that almost nothing does. It must also be
**feasible before any activity is placed** -- the first draft of this generator
sized the battery against activity demand alone and produced instances whose
empty plan already violated the depth-of-discharge floor, so every mode scored
zero. The sizing is therefore derived in a fixed order:

1. Sample the requests, giving a total analytic activity energy ``Ea`` [W*h] and
   a total data volume ``Dd`` [Mbit].
2. Sample the eclipse fraction, phase, bus load and contact windows, giving the
   per-step fixed load and its total ``F`` [W*h].
3. Choose the array peak power so that the generated energy is
   ``F + beta * Ea`` for ``beta`` drawn from ``surplus_fraction_range``. The
   solar integral is linear in peak power, so this is a division, not a search.
   ``beta`` is the share of activity demand the array can cover directly.
4. Compute the **worst empty-plan drawdown** ``W`` [W*h]: the largest running
   sum of ``fixed - solar`` over any contiguous run of steps, by Kadane's
   maximum-subarray algorithm. This is exactly the usable energy the empty plan
   needs.
5. Set the usable energy to ``max(feasibility_margin * W, (tau - beta) * Ea)``
   for ``tau`` drawn from ``energy_tightness_range``, then divide by a sampled
   depth-of-discharge limit to get the nameplate capacity. The first term keeps
   the empty plan feasible; the second makes ``tau * Ea`` the energy available
   to activities over the horizon.
6. Recorder capacity ``= recorder_tightness * Dd`` and total contact capacity
   ``= downlink_tightness * Dd``.

This is a declared experimental design, not a model of any spacecraft. Its
consequence is that the measured optimality gap is a statement about *this*
population of instances and transfers to no other; that caveat is in the
README's Limitations.

No value in this module was identified from any spacecraft or mission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import (
    ActivityRequest,
    analytic_duration_s,
    analytic_energy_wh,
    sample_requests,
    to_activity,
)
from .model import Activity, Contact, PowerSegment, Problem, ResourceModel, SolarProfile

__all__ = [
    "INSTANCE_ORDER",
    "RECOMMENDED_STEP_S",
    "InstanceSpec",
    "instance_family",
    "named_instance",
    "sample_instance_spec",
]

INSTANCE_ORDER: tuple[str, ...] = (
    "tiny-enumerable",
    "energy-bound",
    "recorder-bound",
    "window-bound",
    "leo-imaging",
)
"""The shipped hand-declared instances, in the order the CLI lists them."""

_ORBIT_PERIOD_S = 5400.0
"""Declared repeat period [s] of the solar profile in every shipped instance.

About 90 minutes, the right order for a low Earth orbit, and a declared input.
Nothing in this package computes an orbital period.
"""


def _tiny_enumerable() -> Problem:
    """Four activities on a 12-step grid: small enough to enumerate exhaustively.

    Binding constraint by construction: the battery. Total activity energy
    demand exceeds what the array supplies over the horizon, so at most two of
    the four fit. 1200 s horizon with a 1200 s orbit period and a 0.35 eclipse
    fraction, which puts the whole second half of the horizon in darkness.
    """
    solar = SolarProfile(peak_power_w=90.0, orbit_period_s=1200.0, eclipse_fraction=0.35)
    resources = ResourceModel(
        battery_capacity_wh=20.0,
        max_depth_of_discharge=0.35,
        solar=solar,
        recorder_capacity_mbit=300.0,
        bus_load_w=30.0,
        contacts=(Contact(600.0, 800.0, 1.0, 8.0, "pass-1"),),
    )
    activities = (
        Activity(
            name="image-a",
            power_profile=(PowerSegment(60.0, 95.0), PowerSegment(140.0, 45.0)),
            data_mbit=120.0,
            priority=3.0,
        ),
        Activity(
            name="radar-b",
            power_profile=(PowerSegment(100.0, 90.0),),
            data_mbit=80.0,
            priority=2.0,
        ),
        Activity(
            name="spectro-c",
            power_profile=(PowerSegment(40.0, 70.0), PowerSegment(160.0, 38.0)),
            data_mbit=150.0,
            priority=1.5,
        ),
        Activity(
            name="cal-d",
            power_profile=(PowerSegment(100.0, 70.0),),
            data_mbit=60.0,
            priority=1.0,
        ),
    )
    return Problem(
        activities=activities,
        resources=resources,
        horizon_s=1200.0,
        exclusions=(("image-a", "radar-b"),),
        name="tiny-enumerable",
    )


def _energy_bound() -> Problem:
    """Twelve activities, a generous recorder and wide windows: the battery binds.

    The recorder is more than twice the total data and the contacts can carry
    all of it, so neither data constraint can bind. The energy available to
    activities over the horizon is bounded above by
    ``generated - fixed + usable`` = 144.3 - 133.0 + 36.0 = 47.3 W*h against a
    total activity demand of 141.1 W*h, so at most about a third of the
    requested work can be done and the choice of *which* third is the whole
    problem. The empty plan clears the depth-of-discharge floor by 8.8 W*h, so
    the declaration is feasible before any activity is placed -- a condition an
    earlier draft of this instance failed, which is why
    :func:`activityplan.exact.solve_exact` now names the empty-plan binding
    constraint when a model is infeasible.
    """
    solar = SolarProfile(peak_power_w=118.0, orbit_period_s=_ORBIT_PERIOD_S, eclipse_fraction=0.36)
    resources = ResourceModel(
        battery_capacity_wh=90.0,
        max_depth_of_discharge=0.40,
        solar=solar,
        recorder_capacity_mbit=6000.0,
        bus_load_w=42.0,
        contacts=(
            Contact(1800.0, 2250.0, 12.0, 28.0, "pass-1"),
            Contact(7200.0, 7650.0, 12.0, 28.0, "pass-2"),
        ),
    )
    activities = tuple(
        Activity(
            name=f"obs-{index:02d}",
            power_profile=(
                PowerSegment(60.0, 185.0 + 6.0 * (index % 3)),
                PowerSegment(240.0, 100.0 + 4.0 * (index % 4)),
            ),
            data_mbit=180.0 + 40.0 * (index % 3),
            priority=float(1 + index % 4),
        )
        for index in range(12)
    )
    return Problem(
        activities=activities,
        resources=resources,
        horizon_s=2.0 * _ORBIT_PERIOD_S,
        name="energy-bound",
    )


def _recorder_bound() -> Problem:
    """Eight data-heavy activities with one short contact: the recorder binds.

    Activity power is low and the array is large, so the battery has margin
    throughout; the recorder holds a fifth of the total data and the single
    contact can clear only part of it.
    """
    solar = SolarProfile(peak_power_w=190.0, orbit_period_s=_ORBIT_PERIOD_S, eclipse_fraction=0.30)
    resources = ResourceModel(
        battery_capacity_wh=180.0,
        max_depth_of_discharge=0.40,
        solar=solar,
        recorder_capacity_mbit=1400.0,
        bus_load_w=38.0,
        contacts=(Contact(6300.0, 6750.0, 1.6, 24.0, "pass-1"),),
    )
    activities = tuple(
        Activity(
            name=f"swath-{index:02d}",
            power_profile=(PowerSegment(300.0, 46.0 + 3.0 * (index % 3)),),
            data_mbit=420.0 + 60.0 * (index % 4),
            priority=float(1 + index % 3),
        )
        for index in range(8)
    )
    return Problem(
        activities=activities,
        resources=resources,
        horizon_s=2.0 * _ORBIT_PERIOD_S,
        name="recorder-bound",
    )


def _window_bound() -> Problem:
    """Six activities in three mutually exclusive pairs, in 750 s windows.

    Resources are ample: the battery keeps 87 % of its usable energy and the
    recorder 94 % of its capacity. What stops the plan is structure. Each pair
    is declared mutually exclusive and each pair shares a 750 s window, which
    cannot hold two 420 s activities, so exactly one of each pair can be placed
    and the best achievable objective is 8 of the 12 requested priority units.
    The binding constraint is a time window, with a normalised slack an order of
    magnitude smaller than the battery's.
    """
    solar = SolarProfile(peak_power_w=200.0, orbit_period_s=_ORBIT_PERIOD_S, eclipse_fraction=0.25)
    resources = ResourceModel(
        battery_capacity_wh=260.0,
        max_depth_of_discharge=0.45,
        solar=solar,
        recorder_capacity_mbit=9000.0,
        bus_load_w=35.0,
        contacts=(
            Contact(2700.0, 3150.0, 14.0, 26.0, "pass-1"),
            Contact(8100.0, 8550.0, 14.0, 26.0, "pass-2"),
        ),
    )
    windows = (
        (1800.0, 2550.0),
        (1800.0, 2550.0),
        (2100.0, 2850.0),
        (2100.0, 2850.0),
        (6600.0, 7350.0),
        (6600.0, 7350.0),
    )
    activities = tuple(
        Activity(
            name=f"target-{index:02d}",
            power_profile=(PowerSegment(120.0, 85.0), PowerSegment(300.0, 50.0)),
            data_mbit=240.0,
            priority=float(1 + index % 3),
            earliest_start_s=start,
            latest_finish_s=finish,
        )
        for index, (start, finish) in enumerate(windows)
    )
    return Problem(
        activities=activities,
        resources=resources,
        horizon_s=2.0 * _ORBIT_PERIOD_S,
        exclusions=(
            ("target-00", "target-01"),
            ("target-02", "target-03"),
            ("target-04", "target-05"),
        ),
        name="window-bound",
    )


def _leo_imaging() -> Problem:
    """The worked example: ten activities over three orbits, everything declared.

    Precedence (a calibration before two of the images), exclusions (radar
    against the fine-pointing images and against the spectrometer) and three
    contacts whose combined capacity is 2160 Mbit against 3160 Mbit of requested
    data. The binding constraint is therefore the recorder, and the exact
    optimum places 27 of the 30 requested priority units while the greedy
    baseline places 18. This is the instance the README's timeline figure and
    feasibility report come from, and it is the largest gap between the two
    baselines anywhere in this repository.

    ``require_empty_recorder`` is deliberately **not** set here; see
    :mod:`activityplan.feasibility` for why an always-tight constraint makes a
    poor binding-constraint report.
    """
    solar = SolarProfile(
        peak_power_w=165.0, orbit_period_s=_ORBIT_PERIOD_S, eclipse_fraction=0.34, phase_s=900.0
    )
    resources = ResourceModel(
        battery_capacity_wh=150.0,
        max_depth_of_discharge=0.30,
        solar=solar,
        recorder_capacity_mbit=700.0,
        bus_load_w=44.0,
        initial_state_of_charge=0.92,
        contacts=(
            Contact(4200.0, 4650.0, 1.6, 30.0, "kiruna-1"),
            Contact(9600.0, 10_050.0, 1.6, 30.0, "kiruna-2"),
            Contact(14_700.0, 15_150.0, 1.6, 30.0, "kiruna-3"),
        ),
    )
    activities = (
        Activity(
            name="cal-boresight",
            power_profile=(PowerSegment(120.0, 101.5), PowerSegment(180.0, 58.0)),
            data_mbit=90.0,
            priority=2.0,
            latest_finish_s=3600.0,
        ),
        Activity(
            name="image-01",
            power_profile=(PowerSegment(60.0, 174.0), PowerSegment(240.0, 84.1)),
            data_mbit=430.0,
            priority=5.0,
            earliest_start_s=1800.0,
            latest_finish_s=11_400.0,
        ),
        Activity(
            name="image-02",
            power_profile=(PowerSegment(60.0, 174.0), PowerSegment(240.0, 84.1)),
            data_mbit=430.0,
            priority=5.0,
            earliest_start_s=1800.0,
            latest_finish_s=11_400.0,
        ),
        Activity(
            name="image-03",
            power_profile=(PowerSegment(60.0, 174.0), PowerSegment(180.0, 84.1)),
            data_mbit=330.0,
            priority=3.0,
            earliest_start_s=5400.0,
            latest_finish_s=13_200.0,
        ),
        Activity(
            name="radar-01",
            power_profile=(PowerSegment(90.0, 217.5), PowerSegment(150.0, 137.75)),
            data_mbit=620.0,
            priority=4.0,
            latest_finish_s=7200.0,
        ),
        Activity(
            name="radar-02",
            power_profile=(PowerSegment(90.0, 217.5), PowerSegment(150.0, 137.75)),
            data_mbit=620.0,
            priority=4.0,
            earliest_start_s=7200.0,
            latest_finish_s=14_400.0,
        ),
        Activity(
            name="spectro-01",
            power_profile=(PowerSegment(90.0, 137.75), PowerSegment(330.0, 63.8)),
            data_mbit=280.0,
            priority=3.0,
            latest_finish_s=8400.0,
        ),
        Activity(
            name="spectro-02",
            power_profile=(PowerSegment(90.0, 137.75), PowerSegment(330.0, 63.8)),
            data_mbit=280.0,
            priority=2.0,
            earliest_start_s=9000.0,
            latest_finish_s=15_600.0,
        ),
        Activity(
            name="housekeep-01",
            power_profile=(PowerSegment(180.0, 52.2),),
            data_mbit=40.0,
            priority=1.0,
            latest_finish_s=6000.0,
        ),
        Activity(
            name="housekeep-02",
            power_profile=(PowerSegment(180.0, 52.2),),
            data_mbit=40.0,
            priority=1.0,
            earliest_start_s=10_800.0,
            latest_finish_s=15_600.0,
        ),
    )
    return Problem(
        activities=activities,
        resources=resources,
        horizon_s=3.0 * _ORBIT_PERIOD_S,
        precedences=(("cal-boresight", "image-01"), ("cal-boresight", "image-02")),
        exclusions=(
            ("radar-01", "image-01"),
            ("radar-01", "image-02"),
            ("radar-02", "image-03"),
            ("spectro-01", "radar-01"),
        ),
        name="leo-imaging",
    )


_BUILDERS = {
    "tiny-enumerable": _tiny_enumerable,
    "energy-bound": _energy_bound,
    "recorder-bound": _recorder_bound,
    "window-bound": _window_bound,
    "leo-imaging": _leo_imaging,
}

RECOMMENDED_STEP_S: dict[str, float] = {
    "tiny-enumerable": 100.0,
    "energy-bound": 150.0,
    "recorder-bound": 150.0,
    "window-bound": 150.0,
    "leo-imaging": 150.0,
}
"""Step size [s] each shipped instance is documented and validated at.

The step is a modelling choice, not a property of the instance; a different step
gives a different discretised optimum. These are the values every committed
number in this repository was produced with.
"""


def named_instance(name: str) -> Problem:
    """One of the shipped hand-declared instances.

    Raises ``KeyError`` with the list of valid names if ``name`` is unknown.
    """
    if name not in _BUILDERS:
        raise KeyError(f"unknown instance {name!r}; available: {', '.join(INSTANCE_ORDER)}")
    return _BUILDERS[name]()


@dataclass(frozen=True)
class InstanceSpec:
    """A random instance described by requests rather than by durations.

    The point of the indirection: the same instance can be turned into a
    :class:`activityplan.model.Problem` using the realised ("oracle") durations,
    the analytic ones, the learned point predictions, or the learned upper
    bounds, and the four plans are then directly comparable because every other
    declared quantity is identical.

    Attributes
    ----------
    requests
        The sampled activity requests.
    resources
        The declared resource envelope, sized against the sampled demand.
    horizon_s, step_s
        Horizon [s] and the grid step [s] the instance is documented at.
    exclusions, precedences
        Structural constraints, drawn by the sampler.
    seed
        The seed that produced this instance.
    """

    name: str
    requests: tuple[ActivityRequest, ...]
    resources: ResourceModel
    horizon_s: float
    step_s: float
    exclusions: tuple[tuple[str, str], ...]
    precedences: tuple[tuple[str, str], ...]
    seed: int

    def problem_from(
        self, durations_s: np.ndarray, energies_wh: np.ndarray, *, name_suffix: str = ""
    ) -> Problem:
        """Build the problem with the supplied per-activity duration and energy.

        Parameters
        ----------
        durations_s
            One duration [s] per request, in request order, strictly positive.
        energies_wh
            One energy [W*h] per request, in request order, non-negative.
        name_suffix
            Appended to the instance name, so the four planning modes produce
            distinguishable problem names in a report.

        Raises
        ------
        ValueError
            If a duration no longer fits its activity's declared window or the
            horizon. That is a real outcome of a realised duration overrun and
            the caller is expected to treat it as an infeasible realisation
            rather than an error; :mod:`activityplan.robust` does.
        """
        durations_s = np.asarray(durations_s, dtype=float)
        energies_wh = np.asarray(energies_wh, dtype=float)
        if durations_s.shape != (len(self.requests),) or energies_wh.shape != (
            len(self.requests),
        ):
            raise ValueError(
                f"expected {len(self.requests)} durations and energies, got "
                f"{durations_s.shape} and {energies_wh.shape}"
            )
        activities = tuple(
            to_activity(request, float(duration), float(energy))
            for request, duration, energy in zip(
                self.requests, durations_s, energies_wh, strict=True
            )
        )
        return Problem(
            activities=activities,
            resources=self.resources,
            horizon_s=self.horizon_s,
            precedences=self.precedences,
            exclusions=self.exclusions,
            name=self.name + name_suffix,
        )

    def analytic_problem(self) -> Problem:
        """The problem as the declared analytic model describes it."""
        durations = np.array([analytic_duration_s(r) for r in self.requests], dtype=float)
        energies = np.array([analytic_energy_wh(r) for r in self.requests], dtype=float)
        return self.problem_from(durations, energies, name_suffix="/analytic")


def _fixed_and_solar_steps(
    solar: SolarProfile,
    contacts: tuple[Contact, ...],
    bus_load_w: float,
    horizon_s: float,
    step_s: float,
) -> tuple[list[float], list[float]]:
    """Per-step fixed load and solar energy [W*h] without building a ResourceModel.

    Needed because the sizing in :func:`sample_instance_spec` has to know the
    worst empty-plan drawdown before it can choose a battery capacity, and
    :class:`activityplan.model.ResourceModel` refuses an inconsistent one.
    Mirrors the arithmetic in
    :meth:`activityplan.timeline.RasterisedProblem.build` exactly; the two are
    checked against each other in ``tests/test_instances.py``.
    """
    n_steps = int(round(horizon_s / step_s))
    fixed = [bus_load_w * step_s / 3600.0 for _ in range(n_steps)]
    for contact in contacts:
        for k in range(n_steps):
            overlap = min((k + 1) * step_s, contact.end_s) - max(k * step_s, contact.start_s)
            if overlap > 0.0:
                fixed[k] += contact.power_w * overlap / 3600.0
    generated = [solar.energy_wh(k * step_s, (k + 1) * step_s) for k in range(n_steps)]
    return fixed, generated


def _worst_drawdown_wh(fixed: list[float], generated: list[float]) -> float:
    """Largest running deficit [W*h] over any contiguous run of steps.

    Kadane's maximum-subarray algorithm (Bentley, J., "Programming pearls:
    algorithm design techniques", Communications of the ACM 27(9), 1984, which
    is where the algorithm was first published) applied to
    ``fixed[k] - generated[k]``. Equals the usable battery energy the empty plan
    requires, because the clamp at full capacity means the battery enters the
    worst deficit run at or below full and must cover the whole run.
    """
    best = 0.0
    running = 0.0
    for load, supply in zip(fixed, generated, strict=True):
        running = max(0.0, running + load - supply)
        best = max(best, running)
    return best


def sample_instance_spec(
    seed: int,
    *,
    n_activities: int = 14,
    n_orbits: int = 2,
    step_s: float = 150.0,
    surplus_fraction_range: tuple[float, float] = (0.15, 0.45),
    energy_tightness_range: tuple[float, float] = (0.25, 0.55),
    recorder_tightness_range: tuple[float, float] = (0.12, 0.30),
    downlink_tightness_range: tuple[float, float] = (0.25, 0.55),
    feasibility_margin: float = 1.45,
    n_exclusions: int = 2,
    n_precedences: int = 1,
) -> InstanceSpec:
    """Draw one oversubscribed instance from the declared family.

    Parameters
    ----------
    seed
        Seeds every draw. The same seed always gives the same instance.
    n_activities
        Number of requests, at least 2.
    n_orbits
        Horizon length in units of the declared 5400 s orbit period.
    step_s
        Grid step [s] the instance is documented at. Must divide the horizon.
    surplus_fraction_range
        Range for ``beta``, the share of total activity energy the array can
        supply directly beyond the fixed load. See the module docstring, step 3.
    energy_tightness_range
        Range for ``tau``, the share of total activity energy available over the
        horizon. Must exceed ``surplus_fraction_range`` for the second term of
        step 5 to bite; when it does not, the drawdown floor sets the capacity
        and the instance is not energy-limited, which is a legitimate member of
        the family.
    recorder_tightness_range, downlink_tightness_range
        Ranges for the recorder capacity and the total contact capacity, both as
        a share of the total data volume.
    feasibility_margin
        Multiple of the worst empty-plan drawdown the usable energy must cover.
        Must be strictly greater than 1, otherwise the empty plan sits exactly
        on the depth-of-discharge floor and rounding decides feasibility.
    n_exclusions, n_precedences
        How many structural constraints to draw. Exclusion pairs are drawn
        without replacement; precedence pairs are drawn between distinct
        activities in index order, which cannot create a cycle.

    Returns
    -------
    InstanceSpec
    """
    if n_activities < 2:
        raise ValueError(f"n_activities must be at least 2, got {n_activities!r}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits must be at least 1, got {n_orbits!r}")
    if feasibility_margin <= 1.0:
        raise ValueError(
            f"feasibility_margin must be strictly greater than 1, got {feasibility_margin!r}"
        )
    rng = np.random.default_rng(int(seed))
    horizon_s = float(n_orbits) * _ORBIT_PERIOD_S
    requests = sample_requests(
        n_activities,
        rng,
        name_prefix="act",
        data_mbit_range=(40.0, 420.0),
        slew_angle_range=(0.0, 70.0),
    )

    total_energy_wh = math.fsum(analytic_energy_wh(request) for request in requests)
    total_data_mbit = math.fsum(request.data_mbit for request in requests)

    bus_load_w = float(rng.uniform(30.0, 50.0))
    eclipse_fraction = float(rng.uniform(0.28, 0.38))
    phase_s = float(rng.uniform(0.0, _ORBIT_PERIOD_S))

    recorder_tightness = float(rng.uniform(*recorder_tightness_range))
    downlink_tightness = float(rng.uniform(*downlink_tightness_range))
    recorder_capacity_mbit = max(1.0, recorder_tightness * total_data_mbit)
    contact_capacity_mbit = downlink_tightness * total_data_mbit
    n_contacts = int(n_orbits)
    contact_duration_s = 450.0
    rate = contact_capacity_mbit / (n_contacts * contact_duration_s)
    contacts: list[Contact] = []
    for index in range(n_contacts):
        offset = float(rng.uniform(0.45, 0.80)) * _ORBIT_PERIOD_S
        start = index * _ORBIT_PERIOD_S + offset
        contacts.append(
            Contact(
                start_s=start,
                end_s=start + contact_duration_s,
                rate_mbit_per_s=rate,
                power_w=float(rng.uniform(22.0, 32.0)),
                name=f"pass-{index + 1}",
            )
        )

    unit_solar = SolarProfile(
        peak_power_w=1.0,
        orbit_period_s=_ORBIT_PERIOD_S,
        eclipse_fraction=eclipse_fraction,
        phase_s=phase_s,
    )
    unit_generated_wh = unit_solar.energy_wh(0.0, horizon_s)
    fixed_total_wh = bus_load_w * horizon_s / 3600.0 + math.fsum(
        contact.power_w * contact.duration_s / 3600.0 for contact in contacts
    )
    surplus_fraction = float(rng.uniform(*surplus_fraction_range))
    peak_power_w = (fixed_total_wh + surplus_fraction * total_energy_wh) / unit_generated_wh
    peak_power_w = float(min(400.0, max(60.0, peak_power_w)))
    solar = SolarProfile(
        peak_power_w=peak_power_w,
        orbit_period_s=_ORBIT_PERIOD_S,
        eclipse_fraction=eclipse_fraction,
        phase_s=phase_s,
    )

    fixed_steps, solar_steps = _fixed_and_solar_steps(
        solar, tuple(contacts), bus_load_w, horizon_s, float(step_s)
    )
    drawdown_wh = _worst_drawdown_wh(fixed_steps, solar_steps)

    energy_tightness = float(rng.uniform(*energy_tightness_range))
    usable_wh = max(
        feasibility_margin * drawdown_wh,
        (energy_tightness - surplus_fraction) * total_energy_wh,
    )
    max_dod = float(rng.uniform(0.25, 0.40))
    capacity_wh = usable_wh / max_dod

    resources = ResourceModel(
        battery_capacity_wh=capacity_wh,
        max_depth_of_discharge=max_dod,
        solar=solar,
        recorder_capacity_mbit=recorder_capacity_mbit,
        initial_state_of_charge=1.0,
        bus_load_w=bus_load_w,
        contacts=tuple(contacts),
    )

    names = [request.name for request in requests]
    available = [
        (names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    exclusions: list[tuple[str, str]] = []
    if available and n_exclusions > 0:
        chosen = rng.choice(len(available), size=min(n_exclusions, len(available)), replace=False)
        exclusions = [available[int(index)] for index in np.atleast_1d(chosen)]
    precedences: list[tuple[str, str]] = []
    if n_precedences > 0 and len(names) >= 2:
        chosen = rng.choice(len(available), size=min(n_precedences, len(available)), replace=False)
        precedences = [available[int(index)] for index in np.atleast_1d(chosen)]

    return InstanceSpec(
        name=f"family-{int(seed)}",
        requests=requests,
        resources=resources,
        horizon_s=horizon_s,
        step_s=float(step_s),
        exclusions=tuple(exclusions),
        precedences=tuple(precedences),
        seed=int(seed),
    )


def instance_family(
    count: int, base_seed: int, **kwargs: object
) -> tuple[InstanceSpec, ...]:
    """``count`` instances with seeds ``base_seed, base_seed + 1, ...``.

    Keyword arguments are forwarded to :func:`sample_instance_spec`, so the whole
    family shares one declared sizing design.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count!r}")
    return tuple(
        sample_instance_spec(base_seed + offset, **kwargs)  # type: ignore[arg-type]
        for offset in range(count)
    )
