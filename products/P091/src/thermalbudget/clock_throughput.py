"""The two measured relations baseline 2 needs, and what to do when neither exists.

Baseline 2 predicts a sustained throughput from a predicted steady-state die
temperature, and it needs two links to get there:

    die temperature  --(1)-->  CPU clock  --(2)-->  throughput

Link (1) is the clock-versus-temperature relation the board actually exhibits.
Link (2) is the clock-versus-throughput relation of the payload. Both are
MEASURED here, from the campaign's own traces. Neither is assumed.

**The relation does not exist on the measured board, and that is a result, not
an error.** Across the three duty-1.0 soaks the die rose 10.94 degC, from 46.34
to 57.28, and the mean CPU clock stayed at 1728000 kHz throughout: the fitted
within-duty slope at duty 1.00 is EXACTLY ZERO over n=1644 samples
(``validation/validate_clock_confound_output.txt``). Pooled over the whole
campaign the clock channel varies by 0.0000 per cent of its maximum over
45.88--57.28 degC, across one distinct level
(``validation/validate_measured_campaign_output.txt``). So link (1) is not
identifiable: there is no clock variation to regress against temperature, and
any number derived from fitting one would be an artefact of noise in a constant
channel.

Corrected 2026-11-12, when the campaign's traces arrived. This paragraph
previously read "The one real cold-burst trace available while this module was
written shows all six cores pinned at cpuinfo_max_freq -- 1728000 kHz -- for
the entire 20 s run", and went on to say it was "entirely possible" that the
board does not throttle. That was written from the FIVE rows of
``burst_1.00_a`` quoted in ``docs/SPEC.md``, not from the whole run; the claim
about all twenty rows had not been checked when it was made. It has now been
checked and it is true -- all 120 per-core samples of each burst run sit at
1728000 kHz -- but the old sentence asserted more than its evidence, which is
the reason it is replaced rather than merely confirmed.

**A caution that survives the measurement.** "Not identifiable" is a statement
about this board, this heatsink and fan, this payload, this power mode and
soaks of at most 720 s. It is not a statement about Jetson Orin Nano modules in
general, and nothing in this module should be read as one.

:class:`ClockTemperatureRelation` therefore has an ``identifiable`` flag and a
reason string, and :meth:`ClockTemperatureRelation.clock_at` returns the
observed clock level with ``degenerate=True`` when the relation is degenerate.
Baseline 2 then predicts the unthrottled clock, and so predicts no
overstatement, and reduces to baseline 1 -- which the report says in those
words. Nothing in this module, and no default anywhere in it, presupposes that
throttling happens.

**Link (1) is fitted as a single straight line, and that is an approximation
with a named failure mode.** A governor that holds the maximum clock up to a
trip temperature and then steps down produces a PIECEWISE relation with a knee,
not a line. A single least-squares line through the pooled samples of such a
board returns a slope between zero and the true post-knee slope -- measured on
this repository's own synthetic ``"throttled"`` fixture, whose true post-knee
slope is 42000 kHz/degC, the pooled single-line fit returns about 25000
kHz/degC, because the pooled samples include the flat region below the knee.
The fitted slope is therefore reported as what it is, the average gradient over
the measured temperature range, and the predicted clock is clamped at the
highest clock actually observed. Fitting a knee was not done, and the measured
campaign vindicates that: on the real board the relation has no gradient at all
at the duty cycle where throttling could act.

**POOLING DUTY CYCLES INTO THIS FUNCTION CONFOUNDS IT, and the measured campaign
shows how badly.** :func:`clock_temperature_relation` fits whatever traces it is
handed. Hand it runs at several duty cycles and the fit crosses a common cause:
duty sets the clock, through the ``schedutil`` governor responding to load, and
it sets the die temperature, through dissipated power. Measured on this
campaign (``validation/validate_clock_confound_output.txt``), the within-duty
slopes in kHz/degC are

    duty 1.00  EXACTLY ZERO (n=1644, die 46.34--57.28 degC, clock constant)
    duty 0.75  -41.6
    duty 0.50  +5293.3
    duty 0.25  +28201.2

while the slope POOLED across all four is +46110.5 -- larger than every
within-duty slope, because it is generated entirely between the groups. A
pooled figure is not a measurement of the clock's response to temperature, and
this function cannot tell the difference. Decompose by duty before reading any
slope from it; ``validation/validate_clock_confound.py`` is the worked form, and
``examples/clock_relation.py`` draws it.

**Link (2) is a declared proportionality, and it is checked.** At a fixed duty
cycle, a CPU-bound payload's throughput is taken proportional to clock
frequency: the payload is an EKF update plus a 96x96 float64 GEMM, which at that
size is resident in cache and is dominated by floating-point issue rather than
by memory latency, so the proportional model is the right first approximation.
It is nevertheless an approximation -- a memory-bound payload would show an
exponent below one, because DRAM latency does not scale with core clock. So
:func:`proportionality_exponent` measures the exponent from the data wherever
the clock varies, by regressing ``log throughput`` on ``log clock``, and reports
it; where the clock does not vary, it reports that the exponent is not
identifiable rather than returning 1.0 as though it had been measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .equilibrium import fit_linear
from .provenance import Origin, meet

#: A clock channel whose full observed range is below this fraction of its own
#: maximum is treated as constant. 0.005 is 0.5 per cent, which at 1728000 kHz is
#: 8640 kHz. MEASURED on the campaign's traces: the board exposes 14 distinct
#: clock levels and the smallest gap between adjacent levels is 76800 kHz, which
#: is 8.9 times this floor, so a real governor step cannot hide under it. Before
#: the traces arrived this comment asserted that the steps were "far larger"
#: without having measured one; the figure above replaces that assertion and is
#: reported by ``validation/validate_measured_campaign.py``.
IDENTIFIABILITY_FRACTION: float = 0.005

#: Fewest distinct clock values required before a slope is fitted at all.
MIN_CLOCK_LEVELS: int = 3

#: Fewest pooled samples required.
MIN_SAMPLES: int = 60


@dataclass(frozen=True)
class ClockTemperatureRelation:
    """Measured relation between die temperature and mean CPU clock.

    Attributes
    ----------
    identifiable:
        Whether a slope was fitted at all. False means the clock channel is
        constant to within :data:`IDENTIFIABILITY_FRACTION` of its maximum, or
        too few distinct levels were observed, or the fitted slope's interval
        contains zero.
    reason:
        One sentence saying why ``identifiable`` is what it is.
    slope_khz_per_c, slope_ci_khz_per_c, intercept_khz:
        The fit, or ``nan`` when not identifiable.
    n_samples, n_levels:
        Pooled samples and distinct clock values observed.
    temperature_range_c:
        Lowest and highest die temperature in the pooled samples. Nothing
        outside this range was measured and
        :meth:`clock_at` flags extrapolation beyond it.
    clock_range_khz:
        Lowest and highest mean clock observed.
    modal_clock_khz:
        The most common clock value, which is what a degenerate relation
        predicts.
    origin:
        Meet of the origins of the traces pooled.
    """

    identifiable: bool
    reason: str
    slope_khz_per_c: float
    slope_ci_khz_per_c: tuple[float, float]
    intercept_khz: float
    n_samples: int
    n_levels: int
    temperature_range_c: tuple[float, float]
    clock_range_khz: tuple[float, float]
    modal_clock_khz: float
    origin: Origin

    @property
    def relative_clock_range(self) -> float:
        """Observed clock range as a fraction of the maximum observed clock."""
        low, high = self.clock_range_khz
        return (high - low) / high if high else math.nan

    @property
    def throttling_observed(self) -> bool:
        """Whether any clock variation large enough to be a throttle was seen."""
        return self.relative_clock_range > IDENTIFIABILITY_FRACTION

    def clock_at(self, temperature_c: float) -> tuple[float, bool, bool]:
        """Predicted mean CPU clock at a die temperature.

        Returns
        -------
        (clock_khz, degenerate, extrapolating)
            ``degenerate`` is True when the relation was not identifiable and the
            modal observed clock is returned unchanged. ``extrapolating`` is True
            when ``temperature_c`` lies outside the measured temperature range,
            in which case the number is a model extrapolation and the report
            layer labels it so.
        """
        low, high = self.temperature_range_c
        extrapolating = not (low <= temperature_c <= high)
        if not self.identifiable:
            return self.modal_clock_khz, True, extrapolating
        predicted = self.intercept_khz + self.slope_khz_per_c * temperature_c
        ceiling = self.clock_range_khz[1]
        return min(predicted, ceiling), False, extrapolating


def clock_temperature_relation(traces: Sequence) -> ClockTemperatureRelation:
    """Pool die temperature against mean CPU clock over ``traces`` and fit.

    Parameters
    ----------
    traces:
        Traces to pool. Duty-0 runs carry no load and are accepted, since a
        constant clock at idle is still evidence about the clock channel.

    Raises
    ------
    ValueError
        If no trace is given, or fewer than :data:`MIN_SAMPLES` finite pairs
        survive. An empty input set cannot produce a relation.
    """
    if not traces:
        raise ValueError(
            "clock_temperature_relation got no traces; an empty input set cannot "
            "produce a relation"
        )
    temperatures: list[np.ndarray] = []
    clocks: list[np.ndarray] = []
    for trace in traces:
        temperature = trace.die_temperature_c
        clock = trace.mean_cpu_khz
        keep = np.isfinite(temperature) & np.isfinite(clock)
        temperatures.append(temperature[keep])
        clocks.append(clock[keep])
    temperature = np.concatenate(temperatures) if temperatures else np.empty(0)
    clock = np.concatenate(clocks) if clocks else np.empty(0)
    origin = meet([trace.origin for trace in traces])
    if temperature.size < MIN_SAMPLES:
        raise ValueError(
            f"only {temperature.size} finite (temperature, clock) pairs across "
            f"{len(traces)} trace(s), fewer than the {MIN_SAMPLES} required"
        )
    levels = np.unique(np.round(clock, 3))
    clock_range = (float(clock.min()), float(clock.max()))
    temperature_range = (float(temperature.min()), float(temperature.max()))
    values, counts = np.unique(np.round(clock, 3), return_counts=True)
    modal = float(values[int(np.argmax(counts))])
    relative = (clock_range[1] - clock_range[0]) / clock_range[1] if clock_range[1] else math.nan

    def degenerate(reason: str) -> ClockTemperatureRelation:
        return ClockTemperatureRelation(
            identifiable=False,
            reason=reason,
            slope_khz_per_c=math.nan,
            slope_ci_khz_per_c=(math.nan, math.nan),
            intercept_khz=math.nan,
            n_samples=int(temperature.size),
            n_levels=int(levels.size),
            temperature_range_c=temperature_range,
            clock_range_khz=clock_range,
            modal_clock_khz=modal,
            origin=origin,
        )

    if relative <= IDENTIFIABILITY_FRACTION:
        return degenerate(
            f"the mean CPU clock varies by {100.0 * relative:.4f} per cent of its maximum "
            f"over {temperature_range[0]:.2f}--{temperature_range[1]:.2f} degC, at or below "
            f"the declared identifiability floor of {100.0 * IDENTIFIABILITY_FRACTION:g} "
            "per cent: no clock-versus-temperature relation is identifiable from this "
            "campaign, which is the signature of a board that does not throttle "
            "measurably under this payload"
        )
    if levels.size < MIN_CLOCK_LEVELS:
        return degenerate(
            f"only {levels.size} distinct clock level(s) observed, fewer than the "
            f"{MIN_CLOCK_LEVELS} required to fit a slope"
        )
    fit = fit_linear(temperature, clock)
    half_width = 1.959963984540054 * fit.slope_stderr_ar1
    interval = (fit.slope - half_width, fit.slope + half_width)
    if interval[0] <= 0.0 <= interval[1]:
        return degenerate(
            f"the fitted slope {fit.slope:+.1f} kHz/degC has a 95 per cent interval "
            f"[{interval[0]:+.1f}, {interval[1]:+.1f}] containing zero; no relation is "
            "identifiable"
        )
    return ClockTemperatureRelation(
        identifiable=True,
        reason=(
            f"the mean CPU clock varies by {100.0 * relative:.3f} per cent over "
            f"{temperature_range[0]:.2f}--{temperature_range[1]:.2f} degC across "
            f"{levels.size} distinct levels; fitted slope {fit.slope:+.1f} kHz/degC "
            f"(AR(1)-corrected 95 per cent interval [{interval[0]:+.1f}, "
            f"{interval[1]:+.1f}])"
        ),
        slope_khz_per_c=fit.slope,
        slope_ci_khz_per_c=interval,
        intercept_khz=fit.intercept,
        n_samples=int(temperature.size),
        n_levels=int(levels.size),
        temperature_range_c=temperature_range,
        clock_range_khz=clock_range,
        modal_clock_khz=modal,
        origin=origin,
    )


@dataclass(frozen=True)
class ProportionalityExponent:
    """Measured exponent in ``throughput ~ clock ** exponent`` at fixed duty.

    ``identifiable`` is False when the clock does not vary enough to fit an
    exponent, in which case ``exponent`` is ``nan`` -- not 1.0. Returning the
    assumed value from a measurement function would make an assumption
    indistinguishable from a measurement, which is the whole thing this product
    is about.
    """

    identifiable: bool
    reason: str
    exponent: float
    exponent_ci: tuple[float, float]
    n_samples: int
    origin: Origin


def proportionality_exponent(traces: Sequence, *, duty: float = 1.0) -> ProportionalityExponent:
    """Fit ``log throughput`` against ``log clock`` over traces at one duty.

    Raises
    ------
    ValueError
        If no trace matches ``duty``.
    """
    selected = [
        trace for trace in traces if math.isfinite(trace.duty) and math.isclose(trace.duty, duty)
    ]
    if not selected:
        raise ValueError(f"no trace at commanded duty {duty}; cannot fit an exponent")
    origin = meet([trace.origin for trace in selected])
    logs_clock: list[np.ndarray] = []
    logs_throughput: list[np.ndarray] = []
    for trace in selected:
        clock = trace.mean_cpu_khz
        throughput = trace.throughput_iters_per_s
        keep = np.isfinite(clock) & np.isfinite(throughput) & (clock > 0) & (throughput > 0)
        logs_clock.append(np.log(clock[keep]))
        logs_throughput.append(np.log(throughput[keep]))
    x = np.concatenate(logs_clock) if logs_clock else np.empty(0)
    y = np.concatenate(logs_throughput) if logs_throughput else np.empty(0)
    if x.size < MIN_SAMPLES or float(np.ptp(x)) == 0.0:
        return ProportionalityExponent(
            identifiable=False,
            reason=(
                f"{x.size} usable sample(s) with a log-clock range of {float(np.ptp(x)):.3g}; "
                "the clock does not vary enough at this duty to identify an exponent, so "
                "the proportional model is ASSUMED and is not measured here"
            ),
            exponent=math.nan,
            exponent_ci=(math.nan, math.nan),
            n_samples=int(x.size),
            origin=origin,
        )
    fit = fit_linear(x, y)
    half_width = 1.959963984540054 * fit.slope_stderr_ar1
    return ProportionalityExponent(
        identifiable=True,
        reason=(
            f"fitted exponent {fit.slope:+.4f} over {x.size} samples "
            f"(95 per cent interval [{fit.slope - half_width:+.4f}, "
            f"{fit.slope + half_width:+.4f}]); 1.0 is the clock-proportional model"
        ),
        exponent=fit.slope,
        exponent_ci=(fit.slope - half_width, fit.slope + half_width),
        n_samples=int(x.size),
        origin=origin,
    )
