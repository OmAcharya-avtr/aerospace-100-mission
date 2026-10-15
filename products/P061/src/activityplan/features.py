"""Activity requests, the declared analytic duration/energy model, and the generator.

Why there are two models in one file
------------------------------------
The analytic model (:func:`analytic_duration_s`, :func:`analytic_mean_power_w`)
is the engineering baseline the learned predictor has to beat. The generator
(:func:`realise`) is the synthetic truth that produces the labels. Both are here
so that a reader can see, side by side and without searching, **exactly which
terms the analytic model omits** -- because the size of the learned model's win
is entirely determined by that choice and by nothing about machine learning.

This is the single most important honesty statement about the AI component in
this package, and it is repeated in ``MODEL_CARD.md`` and ``README.md``:

    The learned predictor beats the analytic model on this data because the
    analytic model is missing two terms that the generator contains. The size of
    the win measures the misspecification this file declares. It is not evidence
    that a learned model would beat a duration model fitted to a real
    spacecraft, and no number in this repository should be read that way.

What the comparison *does* measure honestly is the calibration of the
prediction interval, which is a property of the conformal procedure and of the
exchangeability of the data, not of the generator's realism, and the
downstream effect of prediction error on a plan, which is a property of the
planner.

Declared constants
------------------
Every constant below is a declared value of this synthetic benchmark. **No value
in this module was identified from any spacecraft, instrument or mission.** The
*shapes* are the standard ones: a slew time linear in angle at a constant rate,
a fixed settle allowance, a readout time linear in data volume, a power that is
a base load plus a term proportional to data rate. For those shapes as they
appear in operations-concept timeline arithmetic see Wertz, J. R. and
Larson, W. J. (eds), *Space Mission Analysis and Design*, 3rd edition, Microcosm
Press / Kluwer, 1999, chapters 2 and 14.

Validity range
--------------
``slew_angle_deg`` in [0, 90], ``data_mbit`` in [10, 800],
``instrument_temperature_c`` in [-30, 35], ``gain`` in [0.6, 1.8]. The generator
samples inside those ranges; nothing in this module is valid outside them, and
the learned predictor is never evaluated outside them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import Activity, PowerSegment

__all__ = [
    "ACTIVITY_TYPES",
    "ActivityRequest",
    "FEATURE_NAMES",
    "RealisedActivity",
    "TYPE_CONSTANTS",
    "analytic_duration_s",
    "analytic_energy_wh",
    "analytic_mean_power_w",
    "feature_matrix",
    "feature_vector",
    "realise",
    "sample_requests",
    "to_activity",
]

ACTIVITY_TYPES: tuple[str, ...] = ("image", "spectro", "radar", "calibration")
"""The four declared activity types. Ordering fixes the one-hot encoding."""

TYPE_CONSTANTS: dict[str, dict[str, float]] = {
    #                  setup  slew_rate  settle  readout   base_p  data_p  cold   gain   heater
    #                    [s]   [deg/s]     [s]  [Mbit/s]      [W]  [W per   [s/   [s per  [W per
    #                                                           Mbit/s]  10C]  Mbit]   10C]
    "image": {
        "setup_s": 20.0,
        "slew_rate_deg_per_s": 0.80,
        "settle_s": 15.0,
        "readout_rate_mbit_per_s": 2.40,
        "base_power_w": 28.0,
        "power_per_data_rate_w": 6.0,
        "cold_settle_s_per_10c": 9.0,
        "gain_penalty_s_per_mbit": 0.030,
        "heater_w_per_10c": 3.2,
    },
    "spectro": {
        "setup_s": 35.0,
        "slew_rate_deg_per_s": 0.55,
        "settle_s": 25.0,
        "readout_rate_mbit_per_s": 1.10,
        "base_power_w": 34.0,
        "power_per_data_rate_w": 9.0,
        "cold_settle_s_per_10c": 14.0,
        "gain_penalty_s_per_mbit": 0.055,
        "heater_w_per_10c": 5.0,
    },
    "radar": {
        "setup_s": 45.0,
        "slew_rate_deg_per_s": 1.10,
        "settle_s": 10.0,
        "readout_rate_mbit_per_s": 4.50,
        "base_power_w": 62.0,
        "power_per_data_rate_w": 4.0,
        "cold_settle_s_per_10c": 4.0,
        "gain_penalty_s_per_mbit": 0.012,
        "heater_w_per_10c": 2.0,
    },
    "calibration": {
        "setup_s": 60.0,
        "slew_rate_deg_per_s": 0.95,
        "settle_s": 40.0,
        "readout_rate_mbit_per_s": 0.80,
        "base_power_w": 22.0,
        "power_per_data_rate_w": 3.0,
        "cold_settle_s_per_10c": 18.0,
        "gain_penalty_s_per_mbit": 0.070,
        "heater_w_per_10c": 6.5,
    },
}
"""Declared per-type constants. Units are given in the comment block above."""

REFERENCE_TEMPERATURE_C = 10.0
"""Temperature [C] below which the generator's cold-settling and heater terms act."""

MINIMUM_DURATION_S = 15.0
"""Floor applied to any realised duration [s], so a noise draw cannot go negative."""

MINIMUM_POWER_W = 5.0
"""Floor applied to any realised mean power [W]."""

FEATURE_NAMES: tuple[str, ...] = (
    "type_image",
    "type_spectro",
    "type_radar",
    "type_calibration",
    "slew_angle_deg",
    "data_mbit",
    "instrument_temperature_c",
    "gain",
    "analytic_duration_s",
    "analytic_mean_power_w",
)
"""Column names of the learned model's design matrix, in order.

The last two entries are the analytic model's own predictions. Supplying them
means the learned model is fitting the *correction* to the analytic model rather
than relearning it from scratch. That advantages the learned model by
construction and is stated here, in ``MODEL_CARD.md`` and in ``README.md``, so
the benchmark is not read as a like-for-like contest between a physics model and
a learned one.
"""


@dataclass(frozen=True)
class ActivityRequest:
    """A requested activity described by features rather than by a duration.

    This is the input the duration/energy predictors consume. Converting a
    request into an :class:`activityplan.model.Activity` requires a duration and
    an energy, which is exactly what the analytic model or the learned model
    supplies; see :func:`to_activity`.

    Parameters
    ----------
    name
        Unique identifier.
    activity_type
        One of :data:`ACTIVITY_TYPES`.
    slew_angle_deg
        Angle the platform must turn before the activity can start [deg],
        non-negative, declared valid in [0, 90].
    data_mbit
        Data the activity will write to the recorder [Mbit], non-negative.
    instrument_temperature_c
        Instrument temperature at the start of the activity [C].
    gain
        Dimensionless detector gain setting, strictly positive.
    priority
        Strictly positive dimensionless weight used by the objective.
    earliest_start_s, latest_finish_s
        Declared time window [s] from the start of the horizon.
    """

    name: str
    activity_type: str
    slew_angle_deg: float
    data_mbit: float
    instrument_temperature_c: float
    gain: float
    priority: float = 1.0
    earliest_start_s: float = 0.0
    latest_finish_s: float = math.inf

    def __post_init__(self) -> None:
        if self.activity_type not in TYPE_CONSTANTS:
            raise ValueError(
                f"activity_type must be one of {ACTIVITY_TYPES}, got {self.activity_type!r}"
            )
        if self.slew_angle_deg < 0.0:
            raise ValueError(f"slew_angle_deg must be non-negative, got {self.slew_angle_deg!r}")
        if self.data_mbit < 0.0:
            raise ValueError(f"data_mbit must be non-negative, got {self.data_mbit!r}")
        if self.gain <= 0.0:
            raise ValueError(f"gain must be strictly positive, got {self.gain!r}")
        if self.priority <= 0.0:
            raise ValueError(f"priority must be strictly positive, got {self.priority!r}")

    @property
    def constants(self) -> dict[str, float]:
        """The declared constants for this request's type."""
        return TYPE_CONSTANTS[self.activity_type]


@dataclass(frozen=True)
class RealisedActivity:
    """One draw from the generator: what the activity actually took.

    Attributes
    ----------
    request
        The request it came from.
    duration_s
        Realised duration [s].
    mean_power_w
        Realised energy-equivalent mean power [W].
    """

    request: ActivityRequest
    duration_s: float
    mean_power_w: float

    @property
    def energy_wh(self) -> float:
        """Realised energy [W*h] = ``mean_power_w * duration_s / 3600``."""
        return self.mean_power_w * self.duration_s / 3600.0


def analytic_duration_s(request: ActivityRequest) -> float:
    """Declared analytic duration [s]. The baseline the learned model must beat.

    Model::

        duration = setup_s
                 + slew_angle_deg / slew_rate_deg_per_s
                 + settle_s
                 + data_mbit / readout_rate_mbit_per_s

    Source of the shape: standard timeline arithmetic for a slew-settle-collect
    sequence (Wertz & Larson 1999, chapters 2 and 14). Every constant is
    declared in :data:`TYPE_CONSTANTS` and none is taken from a vehicle.

    Assumptions: constant slew rate independent of angle; a settle allowance
    independent of temperature; readout time linear in volume and independent of
    gain. The second and third are the terms the generator adds and this model
    omits.

    Validity range: see the module docstring.
    """
    constants = request.constants
    return (
        constants["setup_s"]
        + request.slew_angle_deg / constants["slew_rate_deg_per_s"]
        + constants["settle_s"]
        + request.data_mbit / constants["readout_rate_mbit_per_s"]
    )


def analytic_mean_power_w(request: ActivityRequest) -> float:
    """Declared analytic mean power [W].

    Model::

        power = base_power_w + power_per_data_rate_w * (data_mbit / duration)

    with ``duration`` the analytic duration. Shape: a fixed base load plus a
    term proportional to the instantaneous data rate. Assumptions: no thermal
    term, no gain dependence. The generator adds a heater term proportional to
    how far the instrument is below :data:`REFERENCE_TEMPERATURE_C`, which this
    model omits.
    """
    constants = request.constants
    duration = analytic_duration_s(request)
    rate = request.data_mbit / duration
    return constants["base_power_w"] + constants["power_per_data_rate_w"] * rate


def analytic_energy_wh(request: ActivityRequest) -> float:
    """Declared analytic energy [W*h] = ``power * duration / 3600``."""
    return analytic_mean_power_w(request) * analytic_duration_s(request) / 3600.0


def realise(
    request: ActivityRequest,
    rng: np.random.Generator,
    *,
    duration_noise_fraction: float = 0.08,
    power_noise_fraction: float = 0.03,
) -> RealisedActivity:
    """Draw what the activity actually took, from the declared generator.

    Generator, stated in full. ``d_a`` and ``p_a`` are the analytic duration and
    power; ``dT = max(0, REFERENCE_TEMPERATURE_C - temperature)`` [C]::

        d = d_a
          + cold_settle_s_per_10c * dT / 10                     (omitted by d_a)
          + gain_penalty_s_per_mbit * (gain - 1)^2 * data_mbit   (omitted by d_a)
          + Normal(0, duration_noise_fraction * d_a)

        p = p_a
          + heater_w_per_10c * dT / 10                           (omitted by p_a)
          + Normal(0, power_noise_fraction * p_a)

    with ``d`` floored at :data:`MINIMUM_DURATION_S` and ``p`` at
    :data:`MINIMUM_POWER_W`.

    The duration noise is **heteroscedastic**: its standard deviation scales with
    the analytic duration, so a long activity has a wider error in absolute terms
    than a short one. That is deliberate. It is the condition under which a
    constant-width conformal interval attains its marginal coverage guarantee
    while *failing* conditional coverage -- undercovering long activities and
    overcovering short ones -- which is measured per duration tercile in
    ``validation/validate_predictor.py`` and is the reason the normalised
    conformal variant exists in :mod:`activityplan.predictor`.

    Parameters
    ----------
    request
        The activity request.
    rng
        Seeded NumPy generator. The same request and the same generator state
        always give the same draw.
    duration_noise_fraction, power_noise_fraction
        Dimensionless noise scales, non-negative.

    Returns
    -------
    RealisedActivity
    """
    if duration_noise_fraction < 0.0 or power_noise_fraction < 0.0:
        raise ValueError("noise fractions must be non-negative")
    constants = request.constants
    base_duration = analytic_duration_s(request)
    base_power = analytic_mean_power_w(request)
    cold = max(0.0, REFERENCE_TEMPERATURE_C - request.instrument_temperature_c)

    duration = (
        base_duration
        + constants["cold_settle_s_per_10c"] * cold / 10.0
        + constants["gain_penalty_s_per_mbit"] * (request.gain - 1.0) ** 2 * request.data_mbit
        + float(rng.normal(0.0, duration_noise_fraction * base_duration))
    )
    power = (
        base_power
        + constants["heater_w_per_10c"] * cold / 10.0
        + float(rng.normal(0.0, power_noise_fraction * base_power))
    )
    return RealisedActivity(
        request=request,
        duration_s=max(MINIMUM_DURATION_S, duration),
        mean_power_w=max(MINIMUM_POWER_W, power),
    )


def feature_vector(request: ActivityRequest) -> np.ndarray:
    """Design row for ``request``, shape ``(10,)``, columns :data:`FEATURE_NAMES`."""
    one_hot = [1.0 if request.activity_type == name else 0.0 for name in ACTIVITY_TYPES]
    return np.array(
        [
            *one_hot,
            request.slew_angle_deg,
            request.data_mbit,
            request.instrument_temperature_c,
            request.gain,
            analytic_duration_s(request),
            analytic_mean_power_w(request),
        ],
        dtype=float,
    )


def feature_matrix(requests: list[ActivityRequest] | tuple[ActivityRequest, ...]) -> np.ndarray:
    """Design matrix for a sequence of requests, shape ``(n, 10)``."""
    if not requests:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.vstack([feature_vector(request) for request in requests])


def sample_requests(
    n: int,
    rng: np.random.Generator,
    *,
    priority_choices: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0),
    name_prefix: str = "act",
    data_mbit_range: tuple[float, float] = (10.0, 800.0),
    slew_angle_range: tuple[float, float] = (0.0, 90.0),
) -> tuple[ActivityRequest, ...]:
    """Draw ``n`` activity requests uniformly over the declared validity ranges.

    Distributions, all independent and all uniform over the ranges declared in
    the module docstring: type uniform over the four; ``slew_angle_deg`` in
    [0, 90]; ``data_mbit`` in [10, 800]; ``instrument_temperature_c`` in
    [-30, 35]; ``gain`` in [0.6, 1.8]; priority uniform over
    ``priority_choices``. ``data_mbit_range`` and ``slew_angle_range`` narrow the
    two ranges that dominate duration, which the instance builders in
    :mod:`activityplan.instances` use so that a sampled activity fits inside a
    few-orbit horizon; both must lie inside the declared validity ranges and are
    checked. Windows are left open (``earliest_start_s = 0``,
    ``latest_finish_s = inf``); instance builders in
    :mod:`activityplan.instances` narrow them.

    Independence is a strong and unrealistic assumption -- on a real mission the
    hot side of the orbit correlates with instrument temperature and with
    pointing -- and it is recorded in ``DATASET_CARD.md`` as the dataset's main
    limitation.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n!r}")
    if not 10.0 <= data_mbit_range[0] <= data_mbit_range[1] <= 800.0:
        raise ValueError(
            f"data_mbit_range {data_mbit_range!r} must lie inside the declared "
            "validity range [10, 800] Mbit and be ordered"
        )
    if not 0.0 <= slew_angle_range[0] <= slew_angle_range[1] <= 90.0:
        raise ValueError(
            f"slew_angle_range {slew_angle_range!r} must lie inside the declared "
            "validity range [0, 90] deg and be ordered"
        )
    requests: list[ActivityRequest] = []
    for index in range(n):
        requests.append(
            ActivityRequest(
                name=f"{name_prefix}{index:03d}",
                activity_type=ACTIVITY_TYPES[int(rng.integers(0, len(ACTIVITY_TYPES)))],
                slew_angle_deg=float(rng.uniform(*slew_angle_range)),
                data_mbit=float(rng.uniform(*data_mbit_range)),
                instrument_temperature_c=float(rng.uniform(-30.0, 35.0)),
                gain=float(rng.uniform(0.6, 1.8)),
                priority=float(priority_choices[int(rng.integers(0, len(priority_choices)))]),
            )
        )
    return tuple(requests)


def to_activity(
    request: ActivityRequest, duration_s: float, energy_wh: float
) -> Activity:
    """Build an :class:`activityplan.model.Activity` from a request and estimates.

    The activity has a **single constant-power segment** of length ``duration_s``
    [s] carrying ``energy_wh`` [W*h], so its mean power is
    ``energy_wh * 3600 / duration_s`` [W]. The library supports piecewise
    profiles and the hand-declared instances in
    :mod:`activityplan.instances` use them; the predictor path does not estimate
    a profile shape, only a duration and a total energy, and this function does
    not invent one.

    Raises ``ValueError`` for a non-positive duration or a negative energy.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be strictly positive, got {duration_s!r}")
    if energy_wh < 0.0:
        raise ValueError(f"energy_wh must be non-negative, got {energy_wh!r}")
    power_w = energy_wh * 3600.0 / duration_s
    return Activity(
        name=request.name,
        power_profile=(PowerSegment(duration_s, power_w),),
        data_mbit=request.data_mbit,
        priority=request.priority,
        earliest_start_s=request.earliest_start_s,
        latest_finish_s=request.latest_finish_s,
    )
