"""The three baselines, in the order the specification requires them implemented.

1. **Energy per inference** -- the quantity being refuted. Two variants, both
   quoted in practice: ``measured-active`` and ``nominal-rail``. Implemented
   carefully, because an under-strength version of the thing you are refuting
   inflates your own result; an earlier product of this mission inflated a
   learned-model win by about 50 per cent that way.
2. **The analytic duty-cycle model** -- a closed form whose every component is
   measured on this board rather than assumed.
3. **Ground truth** -- the integrated whole-run energy divided by the counted
   accepted decisions, with a between-run interval.

Everything here is in mJ, mW and seconds. Nothing here reads a datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .energy import (
    GAP_TOLERANCE_S,
    Interstitial,
    WindowEnergy,
    active_windows,
    clip_series,
    energy_over_windows,
    interstitial_samples,
    sleep_energy,
    trapezoid_energy_mj,
    whole_run_energy,
)
from .trace import (
    FLOOR_CELL,
    REFERENCE_REPEATS,
    Campaign,
    Cell,
    TraceDataAbsentError,
)

#: Settling interval after each wake over which the ramp excess is measured,
#: seconds. Three sampler periods. The choice is bounded from below by the
#: sampler (a shorter window is one or two samples and carries no shape) and
#: from above by the lowest duty cycle in the grid (0.10 gives a 0.2 s active
#: phase). It is comparable with the INA3221's own 512-conversion averaging
#: interval, which is why the resulting figure is a BANDWIDTH-LIMITED estimate
#: of the wake transient and is reported as one.
RAMP_WINDOW_S = 0.15

#: An active window must be at least this long for the ramp and the steady
#: level to be separable within it. Three ramp windows: one for the ramp, two
#: for a steady estimate that is not itself one sample.
RAMP_SEPARABLE_MIN_S = 3.0 * RAMP_WINDOW_S

#: Confidence level of the between-run interval.
CONFIDENCE = 0.95


# ------------------------------------------------------------------ baseline 1


@dataclass(frozen=True)
class EnergyPerInference:
    """Baseline 1: the quantity a datasheet or an edge-AI benchmark quotes.

    Attributes
    ----------
    variant
        ``"measured-active"`` or ``"nominal-rail"``.
    mean_active_power_mw
        Mean power over the run's active windows, mW. This is the
        TIME-WEIGHTED mean ``active energy / active duration``, which is the
        mean an energy integral implies; the arithmetic sample mean is reported
        beside it in :attr:`sample_mean_active_power_mw` so a reader can see
        whether the sampler's period was uniform enough for them to agree.
    mean_latency_s
        Mean inference latency ``mean(t_end - t_start)``, seconds.
    energy_mj
        ``mean_active_power_mw * mean_latency_s``, mJ per inference. No idle, no
        ramp, no rejection: that is the point of it.
    n_inferences
        Inferences counted in the run.
    active_duration_s
        Summed measured duration of the active windows, seconds.
    interstitial
        The within-active-phase edge-case count from
        :func:`powerenvelope.energy.interstitial_samples`: sampler timestamps
        that fell inside an active phase but between two inferences by more
        than the merge tolerance, with the widest such gap. Reported rather
        than absorbed.
    n_sleep_samples
        Sampler timestamps inside the span of the active windows but in none of
        them, which for a duty-cycled run are the sleep-window samples.
    """

    variant: str
    mean_active_power_mw: float
    sample_mean_active_power_mw: float
    mean_latency_s: float
    energy_mj: float
    n_inferences: int
    active_duration_s: float
    interstitial: Interstitial
    n_sleep_samples: int


#: The two Baseline-1 variants and the power column each one integrates.
VARIANTS: dict[str, str] = {
    "measured-active": "vdd_in_mw",
    "nominal-rail": "nominal_mw",
}


def energy_per_inference(cell: Cell, variant: str = "measured-active") -> EnergyPerInference:
    """Baseline 1 for one cell.

    ``measured-active`` uses the measured bus voltage. ``nominal-rail``
    substitutes the nominal 5.0 V rail for it and keeps the measured current,
    which is the specific simplification a power budget makes when it quotes a
    rail rather than measuring one.

    Raises
    ------
    ValueError
        On an unknown variant.
    TraceDataAbsentError
        For the background floor cell, which has no inferences.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    decisions = cell.require_decisions()
    starts, ends = active_windows(decisions)
    window = energy_over_windows(cell.power, starts, ends, column=VARIANTS[variant])
    if not np.isfinite(window.mean_power_mw):
        raise ValueError(
            f"cell {cell.label!r} has no measurable active duration, so a mean active "
            "power is undefined. Nothing is reported for it."
        )
    latency = float(np.mean(decisions.latency_s))
    return EnergyPerInference(
        variant=variant,
        mean_active_power_mw=window.mean_power_mw,
        sample_mean_active_power_mw=window.sample_mean_power_mw,
        mean_latency_s=latency,
        energy_mj=window.mean_power_mw * latency,
        n_inferences=decisions.inferences,
        active_duration_s=window.duration_s,
        interstitial=interstitial_samples(cell),
        n_sleep_samples=window.n_between_windows,
    )


# ------------------------------------------------------------------ baseline 2


@dataclass(frozen=True)
class FloorPower:
    """The measured background draw with the pipeline not running.

    **This is not an idle floor.** At campaign time the board had five
    logged-in users, a desktop browser running for nearly two days, swap in use
    and a 1-minute load average above 1. It is a background floor on a shared
    machine and it is reported as one. The foreign CPU occupancy measured during
    the same run is carried in :attr:`foreign_busy_mean` so that the
    contamination is a number rather than a caveat.
    """

    cell: str
    mean_mw: float
    sd_mw: float
    min_mw: float
    max_mw: float
    n_samples: int
    duration_s: float
    quantisation_mw: float
    foreign_busy_mean: float
    tj_mean_c: float


def floor_power(campaign: Campaign, cell_label: str = FLOOR_CELL) -> FloorPower:
    """Measured background floor power, from the no-pipeline cell.

    Raises
    ------
    TraceDataAbsentError
        If the floor cell is not in the campaign.
    """
    cell = campaign.require(cell_label)
    if cell.has_pipeline:
        raise TraceDataAbsentError(
            f"cell {cell_label!r} ran the pipeline, so it is not a background floor. "
            "The floor is the one run with the sampler and monitor running and the "
            "pipeline not running."
        )
    whole = whole_run_energy(cell.power)
    power = cell.power.vdd_in_mw
    return FloorPower(
        cell=cell_label,
        mean_mw=whole.mean_power_mw,
        sd_mw=float(power.std(ddof=1)) if power.size > 1 else 0.0,
        min_mw=float(power.min()),
        max_mw=float(power.max()),
        n_samples=int(power.size),
        duration_s=whole.duration_s,
        quantisation_mw=float(np.mean(cell.power.quantisation_floor_mw)),
        foreign_busy_mean=float(np.mean(cell.monitor.cpu_busy_fraction)),
        tj_mean_c=float(np.mean(cell.monitor.tj_c)),
    )


@dataclass(frozen=True)
class RampEnergy:
    """The measured excess energy in the settling interval after each wake.

    Attributes
    ----------
    separable
        Whether the cell's active windows are long enough
        (:data:`RAMP_SEPARABLE_MIN_S`) for a ramp and a steady level to be
        separated within them. For the 0.10 duty cell the active phase is 0.2 s
        and they are not: the ramp is then reported as zero and the steady level
        as the whole-window mean, which leaves the cycle energy unchanged and
        only the DECOMPOSITION of it unavailable. That is stated rather than
        estimated.
    energy_mj
        Mean excess energy per wake, mJ: the integral of ``P(t)`` over
        ``[wake, wake + RAMP_WINDOW_S]`` minus ``steady_mw * RAMP_WINDOW_S``,
        averaged over the wakes that fit.
    steady_mw
        Time-weighted mean power over the active windows with the first
        :data:`RAMP_WINDOW_S` of each removed, mW.
    n_wakes
        Wakes that contributed.
    window_s
        The declared settling interval, seconds.
    bandwidth_limited
        Always True, and carried so that no report can omit it. The INA3221
        averages 512 conversions in hardware, so the SHAPE of the transient is
        low-passed by the instrument and this figure is an estimate of the
        excess energy, not of the transient's rise time.
    """

    separable: bool
    energy_mj: float
    steady_mw: float
    n_wakes: int
    window_s: float
    bandwidth_limited: bool = True


def ramp_energy(cell: Cell, window_s: float = RAMP_WINDOW_S) -> RampEnergy:
    """Measured wake-transient excess energy for one cell.

    Raises
    ------
    ValueError
        On a non-positive window.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    decisions = cell.require_decisions()
    starts, ends = active_windows(decisions)
    widths = ends - starts
    separable = bool(np.all(widths >= RAMP_SEPARABLE_MIN_S)) and starts.size > 0
    if not separable:
        window = energy_over_windows(cell.power, starts, ends)
        return RampEnergy(
            separable=False,
            energy_mj=0.0,
            steady_mw=window.mean_power_mw,
            n_wakes=0,
            window_s=window_s,
        )
    tail = energy_over_windows(cell.power, starts + window_s, ends)
    steady = tail.mean_power_mw
    excesses: list[float] = []
    for lo in starts:
        t, p = clip_series(cell.power.t, cell.power.vdd_in_mw, float(lo), float(lo) + window_s)
        if t.size < 2:
            continue
        measured = trapezoid_energy_mj(t, p)
        excesses.append(measured - steady * float(t[-1] - t[0]))
    if not excesses:
        return RampEnergy(
            separable=False, energy_mj=0.0, steady_mw=steady, n_wakes=0, window_s=window_s
        )
    return RampEnergy(
        separable=True,
        energy_mj=float(np.mean(excesses)),
        steady_mw=steady,
        n_wakes=len(excesses),
        window_s=window_s,
    )


@dataclass(frozen=True)
class DutyCycleModel:
    """Baseline 2: ``E_cycle = P_floor*t_sleep + E_ramp + P_active*t_active``.

    Every component is measured. ``P_floor`` comes from the no-pipeline floor
    cell, ``P_active`` is the measured steady active draw of this cell,
    ``E_ramp`` is the measured excess in the settling interval after each wake,
    and ``t_active`` and ``t_sleep`` come from the cell's declared duty and the
    campaign's declared cycle length.

    Energy per useful decision follows as
    ``E_cycle / (n_inf_per_cycle * acceptance_rate)``, with both the inference
    count per cycle and the acceptance rate COUNTED from the decisions CSV.

    **The ramp is charged per WAKE, not per cycle.** At every duty below 1.00
    there is one wake per cycle and the two are the same. At duty 1.00 the
    active phases of consecutive cycles are contiguous, so the run has ONE wake
    and not 38, and charging the per-wake figure to every cycle would
    over-count it 38-fold. :attr:`e_ramp_per_cycle_mj` is therefore
    ``E_ramp_per_wake * n_wakes / cycles``. This was wrong in the first
    implementation and the measured data found it, through the identity below.

    **The two-term identity, which is a check and not a restatement.** Writing
    ``P_mean_active`` for the time-weighted mean power over the whole active
    windows, ramp samples included,

    .. code-block:: text

        P_active_steady * t_active + E_ramp_per_cycle == P_mean_active * t_active

    because the settling interval lies inside the active phase and the two
    sides partition the same integral. :attr:`two_term_e_cycle_mj` evaluates
    the cycle energy the second way and :attr:`identity_residual` reports the
    relative disagreement, which is a quadrature residual at the clip
    boundaries and nothing else. The two-term form is what
    ``validate_measured_learned.py`` substitutes a predicted ``P_active`` into,
    because it consumes exactly the quantity the window dataset targets.
    """

    cell: str
    duty: float
    cycle_s: float
    t_active_s: float
    t_sleep_s: float
    p_floor_mw: float
    p_active_mw: float
    e_ramp_mj: float
    ramp_separable: bool
    e_ramp_per_cycle_mj: float
    n_wakes: int
    e_cycle_mj: float
    e_sleep_mj: float
    e_active_mj: float
    two_term_e_cycle_mj: float
    identity_residual: float
    n_inf_per_cycle: float
    acceptance_rate: float
    energy_per_useful_decision_mj: float


def duty_cycle_model(
    campaign: Campaign, cell_label: str, floor: FloorPower | None = None
) -> DutyCycleModel:
    """Baseline 2 for one cell of ``campaign``.

    Raises
    ------
    TraceDataAbsentError
        If the cell or the floor cell is absent.
    ValueError
        If the cell accepted no decision at all, which makes energy per useful
        decision undefined; the refusal names the cell and its reject-reason
        histogram rather than returning an infinity.
    """
    cell = campaign.require(cell_label)
    decisions = cell.require_decisions()
    measured_floor = floor if floor is not None else floor_power(campaign)
    ramp = ramp_energy(cell)
    cycle_s = campaign.cycle_s
    t_active = cell.duty * cycle_s
    t_sleep = cycle_s - t_active
    e_sleep = measured_floor.mean_mw * t_sleep
    e_active = ramp.steady_mw * t_active
    e_ramp_per_cycle = ramp.energy_mj * ramp.n_wakes / cell.cycles
    e_cycle = e_sleep + e_ramp_per_cycle + e_active
    mean_active = mean_active_power_mw(cell)
    two_term = e_sleep + mean_active * t_active
    residual = (e_cycle / two_term - 1.0) if two_term else float("nan")
    n_inf_per_cycle = decisions.inferences / cell.cycles
    rate = decisions.acceptance_rate
    if rate <= 0.0:
        raise ValueError(
            f"cell {cell_label!r} accepted no decision ({decisions.reason_counts()}), so "
            "energy per useful decision is undefined. No figure is reported for it."
        )
    return DutyCycleModel(
        cell=cell_label,
        duty=cell.duty,
        cycle_s=cycle_s,
        t_active_s=t_active,
        t_sleep_s=t_sleep,
        p_floor_mw=measured_floor.mean_mw,
        p_active_mw=ramp.steady_mw,
        e_ramp_mj=ramp.energy_mj,
        ramp_separable=ramp.separable,
        e_ramp_per_cycle_mj=e_ramp_per_cycle,
        n_wakes=ramp.n_wakes,
        e_cycle_mj=e_cycle,
        e_sleep_mj=e_sleep,
        e_active_mj=e_active,
        two_term_e_cycle_mj=two_term,
        identity_residual=residual,
        n_inf_per_cycle=n_inf_per_cycle,
        acceptance_rate=rate,
        energy_per_useful_decision_mj=e_cycle / (n_inf_per_cycle * rate),
    )


@dataclass(frozen=True)
class RailAccounting:
    """The measured rail-accounting ratio against load, fitted over the grid.

    The ratio is ``(E_cpu_gpu_cv + E_soc) / E_vdd_in`` over a whole run, and the
    load is that run's mean VDD_IN power. One point per cell.

    **This is a rail-accounting ratio and not a converter efficiency.** The
    INA3221's three channels measure the input rail and two downstream rails
    that the board happens to expose. Their ratio is what is observable here: it
    omits every rail the part does not instrument, it includes the regulators'
    own losses on the paths it does see, and it is not a datasheet curve. A
    reader who wants an efficiency curve needs the regulator's own
    characterisation and this is not it.

    Attributes
    ----------
    slope_per_mw
        Fitted change in the ratio per mW of load.
    intercept
        Fitted ratio extrapolated to zero load. Extrapolated, not measured: the
        grid's lowest load is :attr:`load_min_mw`.
    r_squared
        Coefficient of determination of the linear fit.
    residual_max
        Largest absolute residual of the fit, dimensionless.
    """

    cells: tuple[str, ...]
    load_mw: tuple[float, ...]
    ratio: tuple[float, ...]
    slope_per_mw: float
    intercept: float
    r_squared: float
    residual_max: float
    load_min_mw: float
    load_max_mw: float


def rail_accounting_curve(campaign: Campaign) -> RailAccounting:
    """Fit the measured rail-accounting ratio against load across the campaign.

    Raises
    ------
    ValueError
        With fewer than three cells, where a two-parameter fit has no residual
        and an R-squared is meaningless.
    """
    labels: list[str] = []
    loads: list[float] = []
    ratios: list[float] = []
    for label, cell in sorted(campaign.cells.items()):
        whole = whole_run_energy(cell.power)
        downstream = energy_over_windows(
            cell.power,
            np.asarray([cell.power.t[0]]),
            np.asarray([cell.power.t[-1]]),
            column="cpu_gpu_cv_mw",
        ).energy_mj + energy_over_windows(
            cell.power,
            np.asarray([cell.power.t[0]]),
            np.asarray([cell.power.t[-1]]),
            column="soc_mw",
        ).energy_mj
        if whole.energy_mj <= 0.0:
            continue
        labels.append(label)
        loads.append(whole.mean_power_mw)
        ratios.append(downstream / whole.energy_mj)
    if len(labels) < 3:
        raise ValueError(
            f"a rail-accounting fit over {len(labels)} cell(s) has no residual to "
            "report; at least three are needed."
        )
    x = np.asarray(loads, dtype=np.float64)
    y = np.asarray(ratios, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else float("nan")
    return RailAccounting(
        cells=tuple(labels),
        load_mw=tuple(float(value) for value in x),
        ratio=tuple(float(value) for value in y),
        slope_per_mw=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        residual_max=float(np.max(np.abs(residual))),
        load_min_mw=float(x.min()),
        load_max_mw=float(x.max()),
    )


# ------------------------------------------------------------------ baseline 3


@dataclass(frozen=True)
class GroundTruth:
    """Baseline 3: integrated whole-run energy over counted accepted decisions.

    This is the measurement the headline is built on. Nothing is modelled: the
    numerator is the trapezoidal integral of the measured rail power across the
    whole run with its sleep windows in it, and the denominator is the number of
    rows of ``decisions_<cell>.csv`` whose ``accepted`` column is 1.
    """

    cell: str
    energy_mj: float
    duration_s: float
    mean_power_mw: float
    inferences: int
    accepted: int
    acceptance_rate: float
    energy_per_useful_decision_mj: float
    energy_per_inference_whole_run_mj: float
    reason_counts: dict[str, int]
    quantisation_mw: float


def ground_truth(cell: Cell) -> GroundTruth:
    """Baseline 3 for one cell.

    Raises
    ------
    ValueError
        If the cell accepted no decision, naming its reject-reason histogram.
    """
    decisions = cell.require_decisions()
    whole = whole_run_energy(cell.power)
    accepted = decisions.accepted_count
    if accepted == 0:
        raise ValueError(
            f"cell {cell.label!r} accepted no decision ({decisions.reason_counts()}), so "
            "energy per useful decision is undefined. No figure is reported for it."
        )
    return GroundTruth(
        cell=cell.label,
        energy_mj=whole.energy_mj,
        duration_s=whole.duration_s,
        mean_power_mw=whole.mean_power_mw,
        inferences=decisions.inferences,
        accepted=accepted,
        acceptance_rate=decisions.acceptance_rate,
        energy_per_useful_decision_mj=whole.energy_mj / accepted,
        energy_per_inference_whole_run_mj=whole.energy_mj / decisions.inferences,
        reason_counts=decisions.reason_counts(),
        quantisation_mw=float(np.mean(cell.power.quantisation_floor_mw)),
    )


@dataclass(frozen=True)
class BetweenRunSpread:
    """The uncertainty of the headline, taken from the repeats of one cell.

    **Why between-run and not a within-run bootstrap.** Two reasons, and the
    first is the decisive one.

    The INA3221 averages 512 conversions in hardware before a reading is
    visible in sysfs, so consecutive 20 Hz samples are not independent draws:
    they overlap in the instrument. A bootstrap over samples within one run
    would resample correlated values as if they were independent and would
    report an interval narrower than the data supports -- and it would be
    reporting the sampling noise of a quantity that is not the one at issue.

    The quantity at issue is run-to-run. An engineer who runs this pipeline
    again tomorrow gets a different die temperature, a different foreign load
    and a different clock trajectory, and the spread of the four repeats of the
    reference cell is the only measurement of that variation this campaign
    contains. With four runs it is a coarse measurement, and the interval
    reported below is a Student-t interval on four points with three degrees of
    freedom, which is wide for that reason rather than by choice.

    Attributes
    ----------
    half_width_mj
        Half-width of the two-sided interval at :data:`CONFIDENCE`.
    """

    cells: tuple[str, ...]
    values_mj: tuple[float, ...]
    mean_mj: float
    sd_mj: float
    half_width_mj: float
    low_mj: float
    high_mj: float
    range_mj: float
    n: int
    confidence: float
    t_critical: float


def between_run_spread(
    campaign: Campaign,
    cell_labels: tuple[str, ...] = REFERENCE_REPEATS,
    confidence: float = CONFIDENCE,
) -> BetweenRunSpread:
    """Between-run interval on the energy per useful decision of one cell.

    Raises
    ------
    ValueError
        With fewer than two repeats, where there is no spread to report.
    TraceDataAbsentError
        If any named repeat is absent, naming the ones that are.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = [
        ground_truth(campaign.require(label)).energy_per_useful_decision_mj
        for label in cell_labels
    ]
    if len(values) < 2:
        raise ValueError(
            f"a between-run spread needs at least two repeats, got {len(values)}. A "
            "single run has no measured run-to-run variation and none is invented."
        )
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    t_critical = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half_width = t_critical * sd / np.sqrt(n)
    return BetweenRunSpread(
        cells=tuple(cell_labels),
        values_mj=tuple(float(value) for value in array),
        mean_mj=mean,
        sd_mj=sd,
        half_width_mj=float(half_width),
        low_mj=mean - float(half_width),
        high_mj=mean + float(half_width),
        range_mj=float(array.max() - array.min()),
        n=int(n),
        confidence=confidence,
        t_critical=t_critical,
    )


def mean_active_power_mw(cell: Cell) -> float:
    """Time-weighted mean rail power over the cell's active windows, mW.

    This is the quantity the window dataset of :mod:`powerenvelope.windows`
    targets and the quantity the two-term cycle-energy form consumes, so a
    predictor of it can be substituted straight into Baseline 2.
    """
    decisions = cell.require_decisions()
    starts, ends = active_windows(decisions)
    return energy_over_windows(cell.power, starts, ends).mean_power_mw


def two_term_cycle_energy_mj(
    duty: float, cycle_s: float, floor_mw: float, mean_active_mw: float
) -> float:
    """``P_floor*t_sleep + P_mean_active*t_active``, mJ.

    The two-term form of Baseline 2's cycle energy. Algebraically the same
    number as the three-term form -- see :class:`DutyCycleModel` -- and the one
    a predicted active power is substituted into, because the quantity it
    consumes is exactly the quantity the window dataset targets.
    """
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"duty must be in [0, 1], got {duty}")
    if cycle_s <= 0.0:
        raise ValueError(f"cycle_s must be positive, got {cycle_s}")
    t_active = duty * cycle_s
    return floor_mw * (cycle_s - t_active) + mean_active_mw * t_active


def energy_per_useful_decision_from_cycle_mj(cell: Cell, e_cycle_mj: float) -> float:
    """``E_cycle / (n_inf_per_cycle * a)`` with both counts taken from the CSV."""
    decisions = cell.require_decisions()
    rate = decisions.acceptance_rate
    if rate <= 0.0:
        raise ValueError(
            f"cell {cell.label!r} accepted no decision, so energy per useful decision "
            "is undefined"
        )
    return e_cycle_mj / ((decisions.inferences / cell.cycles) * rate)


@dataclass(frozen=True)
class FloorRegression:
    """The measured floor's dependence on the foreign CPU occupancy of the board.

    In the floor cell the pipeline is not running, so every busy cycle counted
    by ``/proc/stat`` during it is foreign work. Regressing that run's
    per-second mean rail power on that second's ``cpu_busy_fraction`` therefore
    measures, rather than assumes, how much of the "floor" is other people's
    load.

    The fit is one second per point over the floor run. It is a weak fit and the
    correlation is reported so that the weakness is visible rather than implied:
    an :attr:`intercept` quoted from a fit with a low :attr:`pearson_r` is an
    extrapolation to an occupancy the campaign never observed, and it is used in
    this product only as one row of a published sensitivity table, never as the
    headline floor.

    Attributes
    ----------
    slope_mw_per_unit_busy
        Fitted mW per unit of occupancy fraction.
    intercept_mw
        Fitted power at zero foreign occupancy. **Extrapolated**: the lowest
        occupancy actually observed is :attr:`busy_min`.
    foreign_mw, foreign_fraction
        ``slope * mean occupancy``, in mW and as a fraction of the mean floor:
        the implied foreign contribution at the occupancy actually measured.
    """

    n_seconds: int
    busy_mean: float
    busy_min: float
    busy_max: float
    power_mean_mw: float
    power_sd_mw: float
    slope_mw_per_unit_busy: float
    intercept_mw: float
    pearson_r: float
    r_squared: float
    foreign_mw: float
    foreign_fraction: float


def floor_occupancy_regression(
    campaign: Campaign, cell_label: str = FLOOR_CELL
) -> FloorRegression:
    """Regress the floor run's per-second mean rail power on its own occupancy.

    Each monitor sample at ``t`` reports the occupancy of ``(t - 1, t]``, so the
    rail samples in that same interval are averaged to give the power for that
    second. A second with fewer than five rail samples is dropped.

    Raises
    ------
    ValueError
        With fewer than three usable seconds, or with no variation in occupancy,
        where a slope is not defined.
    TraceDataAbsentError
        If the floor cell is absent or ran the pipeline.
    """
    cell = campaign.require(cell_label)
    if cell.has_pipeline:
        raise TraceDataAbsentError(
            f"cell {cell_label!r} ran the pipeline, so its occupancy is not foreign and "
            "this regression would not measure what it claims to."
        )
    power, monitor = cell.power, cell.monitor
    seconds: list[float] = []
    busy: list[float] = []
    for t, fraction in zip(monitor.t, monitor.cpu_busy_fraction, strict=True):
        inside = (power.t > t - monitor.sample_interval_s) & (power.t <= t)
        if int(inside.sum()) >= 5:
            seconds.append(float(power.vdd_in_mw[inside].mean()))
            busy.append(float(fraction))
    x = np.asarray(busy, dtype=np.float64)
    y = np.asarray(seconds, dtype=np.float64)
    if x.size < 3:
        raise ValueError(
            f"the floor run gives {x.size} usable second(s); a regression needs at "
            "least three."
        )
    # Not an exact comparison with zero: a constant column still carries a
    # floating-point standard deviation of order 1e-17, and `== 0.0` on a
    # computed statistic is the one thing floating point cannot promise. The
    # threshold is relative to the occupancy's own scale.
    if float(x.std()) <= 1e-12 * max(1.0, float(abs(x.mean()))):
        raise ValueError(
            "the floor run's occupancy did not vary measurably "
            f"(sd {float(x.std()):.3e} on a mean of {float(x.mean()):.6f}), so its "
            "slope against occupancy is not defined. Nothing is reported."
        )
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    foreign = float(slope) * float(x.mean())
    return FloorRegression(
        n_seconds=int(x.size),
        busy_mean=float(x.mean()),
        busy_min=float(x.min()),
        busy_max=float(x.max()),
        power_mean_mw=float(y.mean()),
        power_sd_mw=float(y.std(ddof=1)),
        slope_mw_per_unit_busy=float(slope),
        intercept_mw=float(intercept),
        pearson_r=correlation,
        r_squared=correlation**2,
        foreign_mw=foreign,
        foreign_fraction=foreign / float(y.mean()),
    )


def floor_variants(
    campaign: Campaign, cell_label: str = FLOOR_CELL, include_regression: bool = True
) -> dict[str, float]:
    """The floor values the published sensitivity table uses, mW.

    ``measured mean`` is the headline value and the only one any reported
    figure uses. ``5th percentile`` and ``regression intercept`` are the two
    lower alternatives the sensitivity is quoted against: the first is a
    distribution-free lower quantile of the floor run's own samples, the second
    is :class:`FloorRegression`'s extrapolation to zero foreign occupancy.

    Parameters
    ----------
    include_regression
        Whether to include the regression intercept. It requires the floor
        run's occupancy to have varied; where it did not,
        :func:`floor_occupancy_regression` raises rather than returning a
        meaningless slope, and a caller that cannot supply that row passes
        False rather than catching the refusal.
    """
    cell = campaign.require(cell_label)
    variants = {
        "measured mean": floor_power(campaign, cell_label).mean_mw,
        "5th percentile": float(np.percentile(cell.power.vdd_in_mw, 5)),
    }
    if include_regression:
        variants["regression intercept"] = floor_occupancy_regression(
            campaign, cell_label
        ).intercept_mw
    return variants


@dataclass(frozen=True)
class RunOrderRow:
    """One row of the campaign's run-order table, for the drift limitation."""

    cell: str
    order: int
    elapsed_s: float
    busy_mean: float
    tj_mean_c: float
    load1_mean: float
    clk_mean_khz: float
    mean_power_mw: float


def run_order_rows(campaign: Campaign) -> list[RunOrderRow]:
    """The campaign's cells in the order they actually ran, with their environment.

    The run order was **not randomised**: the plan ran the floor first, then the
    three frame dimensions in turn, then the repeats, then the permissive
    control. Frame dimension is therefore confounded with elapsed time and with
    whatever the board's foreign load did over the campaign. This table exists
    so that a reader can see the drift rather than be told about it.
    """
    origin = min(cell.t0_mono for cell in campaign.cells.values())
    plan = list(campaign.manifest.get("plan", []))
    rows: list[RunOrderRow] = []
    for cell in sorted(campaign.cells.values(), key=lambda item: item.t0_mono):
        monitor = cell.monitor
        rows.append(
            RunOrderRow(
                cell=cell.label,
                order=plan.index(cell.label) if cell.label in plan else -1,
                elapsed_s=cell.t0_mono - origin,
                busy_mean=float(monitor.cpu_busy_fraction.mean()),
                tj_mean_c=float(monitor.tj_c.mean()),
                load1_mean=float(monitor.load1.mean()),
                clk_mean_khz=float(np.nanmean(monitor.clk_mean_khz)),
                mean_power_mw=whole_run_energy(cell.power).mean_power_mw,
            )
        )
    return rows


def sleep_power_mw(cell: Cell) -> tuple[float, float]:
    """``(mean, minimum)`` rail power over the cell's own sleep windows, mW.

    Measured in the cell itself, so it carries that cell's own foreign load
    rather than the floor run's. Returns ``(nan, nan)`` for a cell with no sleep
    window, which is every duty-1.00 cell.
    """
    decisions = cell.require_decisions()
    starts, ends = active_windows(decisions)
    inactive = np.ones(cell.power.t.size, dtype=bool)
    for low, high in zip(starts, ends, strict=True):
        inactive &= ~((cell.power.t >= low) & (cell.power.t <= high))
    if not inactive.any():
        return float("nan"), float("nan")
    window = sleep_energy(cell)
    return window.mean_power_mw, float(cell.power.vdd_in_mw[inactive].min())


def active_window_summary(cell: Cell) -> WindowEnergy:
    """The cell's active-window energy accounting, for a report to print."""
    decisions = cell.require_decisions()
    starts, ends = active_windows(decisions, GAP_TOLERANCE_S)
    return energy_over_windows(cell.power, starts, ends)
