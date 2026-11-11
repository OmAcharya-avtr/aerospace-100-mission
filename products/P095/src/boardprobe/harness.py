"""The benchmark harness: timing and resource figures with propagated provenance.

What the harness does
---------------------
It runs a caller-supplied zero-argument callable ``repeats`` times, bracketing
each repeat with a reading from a clock channel, and samples the die
temperature, rail and counter channels around the run. Every duration is a
:class:`~boardprobe.statistics.Statistic` derived from two clock readings, so
it carries the clock's provenance; every resource figure carries its own
channel's provenance; and the result's certificate carries the **meet** of all
of them, which is why a benchmark with a live clock and a simulated
thermometer is ``SYNTHETIC`` and not ``MEASURED``. That is the intended
behaviour: a benchmark report is one artifact and it is no better than its
worst input.

Every label the harness produces goes through ``boardprobe.certify.certify``.
There is no other route, and
``validation/validate_adversarial_label.py`` tries to find one.

Why the overhead is characterised and reported
----------------------------------------------
An instrument that perturbs what it measures by an uncharacterised amount is
not an instrument. :func:`characterise_overhead` measures three things on the
host it runs on:

- the overhead that lands INSIDE a timed bracket, which is the part that
  biases a reported duration: a null payload run with sampling on, minus the
  same payload with sampling off. One clock read and one
  :class:`~boardprobe.readings.Reading` construction fall inside the bracket;
  the second read of the pair falls outside it;
- the cost of one clock read, measured a second and independent way, by
  bracketing a block of ``2 * repeats`` reads and dividing. The two numbers
  are predictions of each other from different arrangements, and
  ``validation/validate_independent_bound.py`` checks that they agree;
- the cost of constructing a sealed :class:`~boardprobe.readings.Reading`
  alone, which is the part of a read that this package added rather than
  inherited from the kernel;
- the perturbation of a declared reference payload, as the ratio of its
  per-repeat cost with sampling to its per-repeat cost without.

All three are properties of the host, not of the board, and they come back
inside a certificate whose headline says which host. Nothing in this
repository was measured on a Jetson Orin Nano.

Timing resolution, stated rather than assumed
---------------------------------------------
A clock channel's value is float seconds (the channel's declared scale turns
``CLOCK_MONOTONIC_RAW`` nanoseconds into seconds), so a duration is the
difference of two floats of magnitude equal to the host's uptime. The
representable resolution of such a difference is one ulp at that magnitude:
``2.2e-16 * uptime`` seconds, which is 2.2e-10 s after eleven days of uptime.
:func:`timing_resolution_s` reports it for the running host and
``validation/validate_harness.py`` prints it beside every duration, so a
reader can see the figure is far above the floor rather than being told so.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .certify import Certificate, Label, certify, label_for
from .platform_attest import PlatformAttestation, attest_platform
from .readings import Attribution, Reading, attribute, combine, mint
from .sources import Source, read_available
from .statistics import Statistic, _aggregates, derive, summarise

#: Channels the harness samples for resource figures when they are available.
#: Clock channels are excluded: the clock is the timing instrument, not a
#: resource being observed.
DEFAULT_ENVIRONMENT_CHANNELS: tuple[str, ...] = (
    "die_temp_cpu",
    "die_temp_gpu",
    "rail_vdd_in_voltage",
    "rail_vdd_in_current",
    "perf_instructions",
)


class HarnessError(RuntimeError):
    """The harness cannot produce a result, and says what is missing."""


def timing_resolution_s(uptime_s: float | None = None) -> float:
    """One ulp of a float-seconds clock value at the host's current magnitude.

    This is the floor on any duration this harness can resolve. Returns the
    spacing of float64 at ``uptime_s``; with ``uptime_s`` omitted, the current
    ``CLOCK_MONOTONIC_RAW`` value is used.
    """
    magnitude = (
        time.clock_gettime(time.CLOCK_MONOTONIC_RAW) if uptime_s is None else float(uptime_s)
    )
    return float(math.ulp(magnitude))


def reference_payload(iterations: int = 2000) -> Callable[[], float]:
    """A declared, deterministic payload for characterising the harness.

    A fixed-count floating-point accumulation. It is not a benchmark of
    anything interesting; it exists so that the perturbation ratio is measured
    against a payload whose cost is stable and whose work cannot be optimised
    away, and so that the number is reproducible from the committed source.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    def payload() -> float:
        total = 0.0
        for index in range(iterations):
            total += (index % 7) * 0.5
        return total

    return payload


def _duration_statistic(
    name: str, before: Reading, after: Reading, index: int
) -> Statistic:
    """One repeat's duration, derived from the bracketing clock readings."""
    if before.channel != after.channel:
        raise HarnessError(
            f"repeat {index}: the bracketing clock readings are on different channels "
            f"({before.channel!r} and {after.channel!r})"
        )
    if before.unit != "s":
        raise HarnessError(
            f"the clock channel {before.channel!r} reports {before.unit!r}; the harness "
            "needs a clock channel whose declared unit is seconds"
        )
    elapsed = after.value - before.value
    if elapsed < 0.0:
        raise HarnessError(
            f"repeat {index}: the clock went backwards by {-elapsed:.9g} s on channel "
            f"{before.channel!r}. A monotonic clock that decreases means the channel is "
            "not monotonic; see docs/RECOVERY.md, 'a run whose provenance cannot be "
            "established'."
        )
    return derive(
        name, elapsed, "s", [attribute([before]), attribute([after])]
    )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and resource figures from one benchmark, with their certificates."""

    name: str
    repeats: int
    warmup: int
    durations_s: tuple[float, ...]
    timing: tuple[Statistic, ...]
    resources: tuple[Statistic, ...]
    refusals: tuple[tuple[str, str], ...]
    timing_certificate: Certificate
    overall_certificate: Certificate
    timing_resolution_s: float

    @property
    def label(self) -> Label:
        """The overall label: the weakest of timing and resources."""
        return self.overall_certificate.label

    @property
    def headline(self) -> str:
        """The overall certificate's headline. Never a bare ``MEASURED``."""
        return self.overall_certificate.headline

    @property
    def mean_duration_s(self) -> float:
        """Mean per-repeat duration in seconds."""
        return float(np.mean(self.durations_s))


def benchmark(
    payload: Callable[[], object],
    name: str,
    clock: Source,
    *,
    repeats: int = 100,
    warmup: int = 5,
    environment: Source | None = None,
    environment_channels: Sequence[str] | None = None,
    platform: PlatformAttestation | None = None,
    clock_channel: str = "monotonic_clock",
) -> BenchmarkResult:
    """Run ``payload`` under sampling and return certified timing and resource figures.

    Parameters
    ----------
    payload:
        Zero-argument callable. Its return value is discarded; it is kept alive
        in a local so that a payload whose only effect is its return value is
        not optimised away.
    name:
        What is being benchmarked. Appears in every certificate subject.
    clock:
        Backend serving ``clock_channel``. Its provenance is the provenance of
        every duration.
    repeats:
        Timed repeats, at least 1.
    warmup:
        Untimed repeats first, at least 0. Reported, never discarded silently.
    environment:
        Backend for the resource channels, or ``None`` for timing only. A
        timing-only benchmark reports no resource figures and its overall
        certificate equals its timing certificate.
    environment_channels:
        Which channels to sample; defaults to
        :data:`DEFAULT_ENVIRONMENT_CHANNELS`. Channels the backend cannot
        serve are reported in ``refusals`` and contribute nothing, rather than
        being silently dropped.
    platform:
        Attestation for the certificates; defaults to attesting now.

    Raises
    ------
    HarnessError
        On a non-monotonic clock, a clock channel that is not in seconds, or an
        environment backend that can serve none of the requested channels.
    """
    if not callable(payload):
        raise TypeError(f"payload must be callable, got {type(payload).__name__}")
    if not isinstance(clock, Source):
        raise TypeError(
            f"clock must be a boardprobe.sources.Source, got {type(clock).__name__}"
        )
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be at least 0, got {warmup}")
    attestation = attest_platform() if platform is None else platform

    for _ in range(warmup):
        _ = payload()

    durations: list[Statistic] = []
    clock_readings: list[Reading] = []
    for index in range(repeats):
        before = clock.read(clock_channel)
        _ = payload()
        after = clock.read(clock_channel)
        clock_readings.extend((before, after))
        durations.append(_duration_statistic(f"{name} duration", before, after, index))

    duration_values = [statistic.value for statistic in durations]
    timing_attribution = attribute(clock_readings)
    timing = _timing_statistics(duration_values, timing_attribution)
    timing_certificate = certify(
        f"{name}: timing from channel {clock_channel!r}",
        timing_attribution,
        label_for(timing_attribution.provenance),
        attestation,
    )

    resources: tuple[Statistic, ...] = ()
    refusals: tuple[tuple[str, str], ...] = ()
    overall_attribution: Attribution = timing_attribution
    if environment is not None:
        names = tuple(
            DEFAULT_ENVIRONMENT_CHANNELS
            if environment_channels is None
            else environment_channels
        )
        readings, refusals = read_available(environment, names)
        if not readings:
            raise HarnessError(
                f"the environment backend {environment.source_id!r} could serve none of "
                f"the {len(names)} requested channels, so no resource figure can be "
                "produced. The board is not reachable from this host; see "
                "docs/RECOVERY.md, 'board unreachable mid-run'. First refusals: "
                + "; ".join(
                    f"{channel}: {reason[:110]}" for channel, reason in refusals[:3]
                )
            )
        resources = _resource_statistics(readings)
        overall_attribution = combine([timing_attribution, attribute(readings)])

    overall_certificate = certify(
        f"{name}: benchmark result",
        overall_attribution,
        label_for(overall_attribution.provenance),
        attestation,
    )
    return BenchmarkResult(
        name=name,
        repeats=repeats,
        warmup=warmup,
        durations_s=tuple(duration_values),
        timing=timing,
        resources=resources,
        refusals=refusals,
        timing_certificate=timing_certificate,
        overall_certificate=overall_certificate,
        timing_resolution_s=timing_resolution_s(),
    )


def _timing_statistics(
    durations: Sequence[float], attribution: Attribution
) -> tuple[Statistic, ...]:
    """Mean, sd, min, median, p95 and max of the per-repeat durations.

    Shares the overflow-safe aggregation of ``boardprobe.statistics``: a
    duration cannot plausibly overflow a float64 sum, but the two layers use
    one implementation so that the guard cannot be present in one report and
    absent in the other.
    """
    values = np.asarray(durations, dtype=float)
    figures = _aggregates(values, values.size)
    statistics = [
        Statistic("repeats", float(values.size), "repeats", attribution),
        Statistic("mean", figures["mean"], "s", attribution),
        Statistic("min", figures["min"], "s", attribution),
        Statistic("median", figures["median"], "s", attribution),
        Statistic("p95", figures["p95"], "s", attribution),
        Statistic("max", figures["max"], "s", attribution),
    ]
    if "sd" in figures:
        statistics.insert(2, Statistic("sd", figures["sd"], "s", attribution))
    return tuple(statistics)


def _resource_statistics(readings: Sequence[Reading]) -> tuple[Statistic, ...]:
    """One mean per sampled channel, each carrying only that channel's attribution."""
    by_channel: dict[str, list[Reading]] = {}
    for reading in readings:
        by_channel.setdefault(reading.channel, []).append(reading)
    statistics: list[Statistic] = []
    for channel, group in by_channel.items():
        for statistic in summarise(group):
            if statistic.name == "mean":
                statistics.append(
                    Statistic(f"{channel} mean", statistic.value, statistic.unit,
                              statistic.attribution)
                )
    return tuple(statistics)


@dataclass(frozen=True, slots=True)
class OverheadReport:
    """What the harness costs, and how much it perturbs what it measures."""

    repeats: int
    null_with_sampling_s: float
    null_without_sampling_s: float
    in_bracket_overhead_s: float
    clock_pair_cost_s: float
    clock_read_cost_s: float
    reading_cost_s: float
    reference_without_sampling_s: float
    reference_with_sampling_s: float
    perturbation_ratio: float
    timing_resolution_s: float
    certificate: Certificate

    @property
    def headline(self) -> str:
        """The certificate's headline: which host these numbers are from."""
        return self.certificate.headline


def characterise_overhead(
    clock: Source,
    *,
    repeats: int = 400,
    payload: Callable[[], object] | None = None,
    platform: PlatformAttestation | None = None,
    clock_channel: str = "monotonic_clock",
) -> OverheadReport:
    """Measure the harness's own per-repeat cost and its perturbation of a payload.

    The three measurements, in order:

    1. A null payload with sampling ON: one clock-reading pair, one
       :class:`~boardprobe.readings.Reading` construction each, and one
       derived duration per repeat.
    2. The same null payload with sampling OFF: the loop and the call, timed
       once around the whole block. The difference is the harness's per-repeat
       cost.
    3. A declared reference payload, timed both ways. The ratio is the
       perturbation.

    All figures are properties of the host this runs on and come back inside a
    certificate that names it.
    """
    if repeats < 2:
        raise ValueError(f"repeats must be at least 2 to characterise overhead, got {repeats}")
    attestation = attest_platform() if platform is None else platform
    work = reference_payload() if payload is None else payload

    def null() -> None:
        return None

    with_sampling_null = _mean_with_sampling(clock, null, repeats, clock_channel)
    without_sampling_null = _mean_without_sampling(clock, null, repeats, clock_channel)
    clock_pair = _mean_clock_pair(clock, repeats, clock_channel)
    reading_cost = _mean_reading_construction(clock, repeats, clock_channel)
    with_sampling_reference = _mean_with_sampling(clock, work, repeats, clock_channel)
    without_sampling_reference = _mean_without_sampling(clock, work, repeats, clock_channel)

    in_bracket_overhead = with_sampling_null[0] - without_sampling_null[0]
    ratio = (
        with_sampling_reference[0] / without_sampling_reference[0]
        if without_sampling_reference[0] > 0.0
        else float("inf")
    )
    attribution = combine(
        [
            with_sampling_null[1],
            without_sampling_null[1],
            clock_pair[1],
            reading_cost[1],
            with_sampling_reference[1],
            without_sampling_reference[1],
        ]
    )
    certificate = certify(
        "harness overhead characterisation",
        attribution,
        label_for(attribution.provenance),
        attestation,
    )
    return OverheadReport(
        repeats=repeats,
        null_with_sampling_s=with_sampling_null[0],
        null_without_sampling_s=without_sampling_null[0],
        in_bracket_overhead_s=in_bracket_overhead,
        clock_pair_cost_s=clock_pair[0],
        clock_read_cost_s=clock_pair[0] / 2.0,
        reading_cost_s=reading_cost[0],
        reference_without_sampling_s=without_sampling_reference[0],
        reference_with_sampling_s=with_sampling_reference[0],
        perturbation_ratio=ratio,
        timing_resolution_s=timing_resolution_s(),
        certificate=certificate,
    )


def _mean_with_sampling(
    clock: Source, payload: Callable[[], object], repeats: int, channel: str
) -> tuple[float, Attribution]:
    """Mean per-repeat cost with the full sampling path, and its attribution."""
    total = 0.0
    readings: list[Reading] = []
    for index in range(repeats):
        before = clock.read(channel)
        _ = payload()
        after = clock.read(channel)
        readings.extend((before, after))
        total += _duration_statistic("overhead probe", before, after, index).value
    return total / repeats, attribute(readings)


def _mean_without_sampling(
    clock: Source, payload: Callable[[], object], repeats: int, channel: str
) -> tuple[float, Attribution]:
    """Mean per-repeat cost with one clock pair around the whole block."""
    before = clock.read(channel)
    for _ in range(repeats):
        _ = payload()
    after = clock.read(channel)
    elapsed = _duration_statistic("block", before, after, 0).value
    return elapsed / repeats, attribute([before, after])


def _mean_reading_construction(
    clock: Source, repeats: int, channel: str
) -> tuple[float, Attribution]:
    """Mean cost of constructing one sealed Reading, with no kernel call.

    A block of ``repeats`` constructions of an identical reading, bracketed
    once. This is the part of a read that this package is responsible for --
    the dataclass, the field validation and the keyed BLAKE2b seal -- as
    opposed to the ``clock_gettime`` call underneath it.
    """
    before = clock.read(channel)
    for index in range(repeats):
        _ = mint(
            channel="overhead_probe",
            value=float(index),
            unit="s",
            timestamp_ns=index,
            provenance=before.provenance,
            source_id=clock.source_id,
            detail="overhead probe",
        )
    after = clock.read(channel)
    elapsed = _duration_statistic("reading construction block", before, after, 0).value
    return elapsed / repeats, attribute([before, after])


def _mean_clock_pair(
    clock: Source, repeats: int, channel: str
) -> tuple[float, Attribution]:
    """Mean cost of one clock-reading PAIR alone, measured a different way.

    ``repeats`` iterations of two reads each, bracketed once around the whole
    block. The per-read figure derived from this is an independent prediction
    of the in-bracket overhead measured by differencing two mean timings.
    """
    before = clock.read(channel)
    readings: list[Reading] = [before]
    for _ in range(repeats):
        readings.append(clock.read(channel))
        readings.append(clock.read(channel))
    after = clock.read(channel)
    readings.append(after)
    elapsed = _duration_statistic("clock pair block", before, after, 0).value
    return elapsed / repeats, attribute(readings)
