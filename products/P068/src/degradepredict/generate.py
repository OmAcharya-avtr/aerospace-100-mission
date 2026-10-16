"""Seeded synthetic degradation trajectories with known crossing times.

Why synthetic
-------------
The quantity this package forecasts -- the time at which a health metric crosses
a declared limit -- is only measurable after it happens. To measure whether a
*predicted interval* around that time is calibrated, the true crossing time has
to be known for every unit in the test set, including the units that never
crossed inside the observation record. No flight data set supplies that. A
declared generator does, and is therefore the only way to measure censored
coverage at all. The price is that every number this repository reports is a
statement about this generator, not about any spacecraft.

The metric
----------
Battery usable capacity expressed as a percentage of beginning-of-life (BOL)
capacity, dimensionless percent, decreasing with service age in days. The limit
is 80 percent of BOL. That figure is a *declared convention* borrowed from
IEEE Std 1188-2005 (IEEE Recommended Practice for Maintenance, Testing, and
Replacement of Valve-Regulated Lead-Acid Batteries for Stationary Applications),
which defines end of service life as the point at which measured capacity falls
below 80 percent of rated capacity. It is used here because it is a published,
checkable number; it is not derived from any spacecraft battery qualification.

Degradation shapes
------------------
Each trajectory is drawn from one of three mean shapes, so that no single fitted
model in this package is correctly specified for the whole population. ``y`` is
percent of BOL, ``t`` is service age in days, ``y0`` is BOL calibration in
percent.

``linear``
    ``y(t) = y0 - b * t``, ``b`` in percent per day. A first-order expansion of
    any smooth fade about the observation window. Valid only over a window short
    compared with the curvature scale; it has no mechanism behind it.
``exponential``
    ``y(t) = y0 * exp(-k * t)``, ``k`` in 1/day. The shape produced by a loss
    rate proportional to the remaining capacity. Used here as a declared
    alternative shape, not as a mechanism claim.
``powerlaw``
    ``y(t) = y0 * (1 - B * t**z)``, ``B`` in day**-z, ``z`` dimensionless. This
    is the sublinear capacity-fade form reported for lithium-ion cells whose
    capacity loss is limited by solid-electrolyte-interphase growth. Wang et al.
    (2011), "Cycle-life model for graphite-LiFePO4 cells", Journal of Power
    Sources 196(8):3942-3948, fit capacity loss as a power law in charge
    throughput with exponent z = 0.55. Broussely et al. (2005), "Main aging
    mechanisms in Li ion batteries", Journal of Power Sources 146(1-2):90-96,
    describe the diffusion-limited square-root-of-time calendar-fade regime that
    gives z = 0.5. The generator draws ``z`` per trajectory from a declared
    interval bracketing both values, so that a fitted model holding ``z`` fixed
    is slightly misspecified, which is the realistic case.
    Validity: these forms describe one ageing regime. They do not describe
    lithium plating, sudden-death failures, or any temperature excursion, and
    nothing here is extrapolated beyond the fitted window with a mechanism claim.

Censoring model
---------------
Each unit is observed from service age 0 to a forecast origin ``t0`` (the end of
the telemetry window the forecast is made from), then followed for a further
duration ``D`` drawn *independently of its degradation state* -- the unit leaves
the monitored fleet when it is decommissioned, when the housekeeping channel is
lost, or when the archive ends. If the crossing happens inside that follow-up
the crossing time is recorded (``event=True``, a complete trajectory); otherwise
all that is known is that the remaining life exceeds ``D`` (``event=False``,
right-censored). Independence of ``D`` and the degradation state is true here by
construction. It is an assumption, and in a real fleet it is often false:
units are frequently removed *because* they are degrading.

Selection
---------
A unit is only a forecasting case if it has not already crossed the limit at the
forecast origin, so trajectories with ``t0 >= T`` are rejected. This is the same
selection a real fleet imposes and it makes the retained sample length-biased
with respect to the crossing-time distribution: short-lived units are
under-represented. It is left in deliberately and stated rather than removed.

Observation noise
-----------------
Independent zero-mean Gaussian noise of declared standard deviation, in percent
of BOL, added to the mean shape at each sample. Real housekeeping telemetry has
autocorrelated residuals, temperature dependence, quantisation and outliers.
This generator has none of those, which makes every residual-based uncertainty
estimate in this package optimistic relative to flight data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .limits import LimitSpec

Family = Literal["linear", "exponential", "powerlaw"]

FAMILIES: tuple[Family, ...] = ("linear", "exponential", "powerlaw")

#: Conventional end-of-life limit used throughout the repository, percent of BOL.
DEFAULT_LIMIT = LimitSpec(
    value=80.0,
    direction="below",
    unit="percent of BOL capacity",
    name="battery_usable_capacity",
    atol=1e-9,
)

__all__ = [
    "DEFAULT_LIMIT",
    "FAMILIES",
    "Family",
    "GeneratorConfig",
    "Trajectory",
    "generate_population",
    "mean_curve",
]


@dataclass(frozen=True)
class GeneratorConfig:
    """Declared parameters of the synthetic fleet. Every field is an input, not a fit.

    Attributes
    ----------
    sample_interval_days:
        Housekeeping cadence, days between samples.
    y0_range:
        Beginning-of-life capacity calibration scatter, percent of BOL.
    log_crossing_mean, log_crossing_sd:
        Natural-log mean and standard deviation of the true crossing time in
        days. The crossing time is drawn first and the shape parameter is solved
        for, so the crossing-time distribution is exactly the declared one.
    t0_range:
        Forecast-origin service age, days. The lower end also sets the minimum
        window length: with the default cadence it guarantees at least
        ``t0_range[0] / sample_interval_days`` samples.
    followup_range:
        Follow-up duration after the forecast origin, days, drawn uniformly and
        independently of the degradation state.
    noise_sd_range:
        Observation-noise standard deviation, percent of BOL.
    powerlaw_z_range:
        Power-law fade exponent, dimensionless. Brackets the z = 0.5 of
        Broussely et al. (2005) and the z = 0.55 of Wang et al. (2011).
    family_weights:
        Probability of each shape in :data:`FAMILIES`, in that order.
    """

    sample_interval_days: float = 7.0
    y0_range: tuple[float, float] = (99.0, 101.0)
    log_crossing_mean: float = 7.0
    log_crossing_sd: float = 0.55
    t0_range: tuple[float, float] = (119.0, 700.0)
    followup_range: tuple[float, float] = (120.0, 1200.0)
    noise_sd_range: tuple[float, float] = (0.15, 0.60)
    powerlaw_z_range: tuple[float, float] = (0.45, 0.65)
    family_weights: tuple[float, float, float] = (0.30, 0.30, 0.40)

    def __post_init__(self) -> None:
        if self.sample_interval_days <= 0:
            raise ValueError(
                f"sample_interval_days must be positive, got {self.sample_interval_days!r}"
            )
        for name in ("y0_range", "t0_range", "followup_range", "noise_sd_range",
                     "powerlaw_z_range"):
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
                raise ValueError(f"{name} must be a finite (lo, hi) with hi >= lo, got {(lo, hi)}")
        if self.noise_sd_range[0] < 0:
            raise ValueError("noise_sd_range must be non-negative")
        if self.t0_range[0] < 2 * self.sample_interval_days:
            raise ValueError("t0_range[0] must admit at least two samples")
        w = np.asarray(self.family_weights, dtype=float)
        if w.shape != (3,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            raise ValueError(f"family_weights must be three non-negative weights summing to 1,"
                             f" got {self.family_weights!r}")
        if self.powerlaw_z_range[0] <= 0:
            raise ValueError("powerlaw_z_range must be strictly positive")


@dataclass(frozen=True)
class Trajectory:
    """One unit: its declared truth, its observed window, and its censoring record.

    Attributes
    ----------
    traj_id:
        Index within the generated population.
    family:
        Which mean shape generated it.
    params:
        The shape parameters actually used, with units in the key names.
    noise_sd:
        Observation-noise standard deviation, percent of BOL.
    crossing_time_days:
        The true crossing time of the *mean* curve, days of service age. Known
        exactly because the shape parameter was solved from it. The noisy samples
        are what a forecaster sees; this is the label.
    origin_days:
        Forecast origin ``t0``, days. Observations run over ``[0, t0]``.
    times_days, values_percent:
        The observed window: sample times and noisy metric values.
    followup_days:
        Follow-up duration ``D`` after the origin, days, drawn independently.
    remaining_true_days:
        ``crossing_time_days - origin_days``. Always positive by selection.
        Known to the validation layer for every unit, censored or not, which is
        the whole reason the generator exists.
    remaining_observed_days:
        ``min(remaining_true_days, followup_days)`` -- what an analyst would have.
    event:
        ``True`` if the crossing was observed within the follow-up (complete),
        ``False`` if right-censored.
    """

    traj_id: int
    family: Family
    params: dict[str, float]
    noise_sd: float
    crossing_time_days: float
    origin_days: float
    times_days: np.ndarray
    values_percent: np.ndarray
    followup_days: float
    remaining_true_days: float
    remaining_observed_days: float
    event: bool
    extras: dict[str, float] = field(default_factory=dict)


def mean_curve(
    times_days: np.ndarray, family: Family, params: dict[str, float]
) -> np.ndarray:
    """Noise-free mean metric curve, percent of BOL, for a declared shape.

    Parameters
    ----------
    times_days:
        Service ages, days, non-negative.
    family:
        One of :data:`FAMILIES`.
    params:
        ``{"y0_percent", "b_percent_per_day"}`` for ``linear``;
        ``{"y0_percent", "k_per_day"}`` for ``exponential``;
        ``{"y0_percent", "B_per_day_z", "z"}`` for ``powerlaw``.

    Validity: ``powerlaw`` is only meaningful while ``B * t**z < 1``; beyond that
    the modelled capacity is negative and the shape has left its regime.
    """
    t = np.asarray(times_days, dtype=float)
    if np.any(t < 0):
        raise ValueError("times_days must be non-negative")
    y0 = float(params["y0_percent"])
    if family == "linear":
        return y0 - float(params["b_percent_per_day"]) * t
    if family == "exponential":
        return y0 * np.exp(-float(params["k_per_day"]) * t)
    if family == "powerlaw":
        return y0 * (1.0 - float(params["B_per_day_z"]) * np.power(t, float(params["z"])))
    raise ValueError(f"unknown family {family!r}")


def _solve_shape(
    family: Family, y0: float, limit_value: float, crossing_days: float, z: float
) -> dict[str, float]:
    """Solve the shape parameter that puts the mean crossing exactly at ``crossing_days``."""
    if family == "linear":
        return {"y0_percent": y0, "b_percent_per_day": (y0 - limit_value) / crossing_days}
    if family == "exponential":
        return {"y0_percent": y0, "k_per_day": float(np.log(y0 / limit_value) / crossing_days)}
    if family == "powerlaw":
        b = (1.0 - limit_value / y0) / (crossing_days**z)
        return {"y0_percent": y0, "B_per_day_z": float(b), "z": float(z)}
    raise ValueError(f"unknown family {family!r}")


def generate_population(
    n: int,
    seed: int,
    config: GeneratorConfig | None = None,
    limit: LimitSpec = DEFAULT_LIMIT,
) -> list[Trajectory]:
    """Generate ``n`` trajectories deterministically from ``seed``.

    Determinism: each candidate unit draws from its own
    :class:`numpy.random.SeedSequence` child, so the k-th accepted unit is
    identical for any ``n >= k`` and the rejection step for one unit cannot
    perturb another. Re-running with the same ``seed``, ``config`` and ``limit``
    reproduces the population bit-for-bit on the same NumPy version.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer, or if the configuration rejects so
        many candidates that ``n`` units cannot be produced from the candidate
        budget of ``10 * n``.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    cfg = config or GeneratorConfig()
    if limit.direction != "below":
        raise ValueError("the shipped generator produces downward-degrading metrics only")

    budget = 10 * int(n)
    children = np.random.SeedSequence(seed).spawn(budget)
    out: list[Trajectory] = []
    weights = np.asarray(cfg.family_weights, dtype=float)

    for child in children:
        if len(out) >= n:
            break
        rng = np.random.default_rng(child)
        family = FAMILIES[int(rng.choice(3, p=weights))]
        y0 = float(rng.uniform(*cfg.y0_range))
        if y0 <= limit.value:
            continue
        crossing = float(np.exp(rng.normal(cfg.log_crossing_mean, cfg.log_crossing_sd)))
        z = float(rng.uniform(*cfg.powerlaw_z_range))
        params = _solve_shape(family, y0, limit.value, crossing, z)
        t0_draw = float(rng.uniform(*cfg.t0_range))
        noise_sd = float(rng.uniform(*cfg.noise_sd_range))
        followup = float(rng.uniform(*cfg.followup_range))

        # The forecast origin is the last sample on the declared cadence at or
        # before the drawn service age, so every quantity below is defined
        # against the same origin.
        n_samples = int(np.floor(t0_draw / cfg.sample_interval_days)) + 1
        times = np.arange(n_samples, dtype=float) * cfg.sample_interval_days
        origin = float(times[-1])
        if origin >= crossing:
            continue  # already across the limit: not a forecasting case

        values = mean_curve(times, family, params) + rng.normal(0.0, noise_sd, size=n_samples)
        remaining_true = crossing - origin
        out.append(
            Trajectory(
                traj_id=len(out),
                family=family,
                params=params,
                noise_sd=noise_sd,
                crossing_time_days=crossing,
                origin_days=origin,
                times_days=times,
                values_percent=values,
                followup_days=followup,
                remaining_true_days=float(remaining_true),
                remaining_observed_days=float(min(remaining_true, followup)),
                event=bool(remaining_true <= followup),
                extras={"t0_draw_days": t0_draw},
            )
        )

    if len(out) < n:
        raise ValueError(
            f"only {len(out)} of {n} requested trajectories survived selection from "
            f"{budget} candidates; widen t0_range or log_crossing_mean"
        )
    return out
