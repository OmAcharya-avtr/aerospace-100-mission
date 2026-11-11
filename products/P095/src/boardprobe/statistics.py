"""Aggregates over readings, each carrying the attribution of its inputs.

Every function here returns :class:`Statistic` objects, and a
:class:`Statistic` cannot exist without an
:class:`~boardprobe.readings.Attribution`. That is the propagation step: the
mean of a set of readings is no better than the worst reading in the set, and
the type makes it impossible to carry the mean forward without carrying that
fact with it.

Nothing here reaches past an attribution to the readings underneath it, so a
second-order derivation -- a statistic of statistics -- propagates correctly
through :func:`derive`, which meets the attributions of its parts.

Overflow, found by Hypothesis rather than reasoned about
-------------------------------------------------------
``numpy.mean`` sums before it divides, so a sample whose SUM exceeds the
float64 range returns ``inf`` for a mean that is perfectly representable.
Hypothesis falsified the property "the mean lies between the min and the max"
at ``[8.988465674311579e+307, 8.98846567431158e+307]`` during this build: the
mean of those two numbers is 8.98846567431158e+307 and ``numpy.mean`` returned
``inf``. ``numpy.std`` is worse, because it squares. Both are pinned with
literal inputs in ``tests/test_regressions.py``.

Every aggregate here is therefore computed in ``numpy.longdouble`` and cast
back, which on this build is an 80-bit extended float with an exponent range
to 1.19e4932 and on an aarch64 target is 128-bit IEEE quad with a wider one
still. The standard deviation is additionally computed on CENTRED values, so
the squares are of residuals and not of the values. Where a result is still
not representable in float64 -- the standard deviation of a sample spanning
nearly the whole float64 range genuinely is not -- the aggregate is REFUSED by
name rather than returned as ``inf``. On a platform where
``numpy.longdouble`` is the same width as ``numpy.double`` the wide-arithmetic
step is a no-op and the explicit refusal is what catches the case; that is a
documented degradation, not a silent one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .readings import Attribution, Reading, attribute, combine, verify


@dataclass(frozen=True, slots=True)
class Statistic:
    """One aggregate figure, its unit, and what it rests on."""

    name: str
    value: float
    unit: str
    attribution: Attribution

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a statistic must be named")
        if type(self.attribution) is not Attribution:
            raise TypeError(
                "attribution must be a boardprobe.readings.Attribution, got "
                f"{type(self.attribution).__module__}.{type(self.attribution).__name__}"
            )

    @property
    def provenance(self):  # noqa: ANN201 - returns Provenance; see attribution
        """Provenance of this statistic's inputs."""
        return self.attribution.provenance


def _finite(name: str, value: float, size: int) -> float:
    """Return ``value`` as a float64, or refuse by name if it is not representable.

    The finiteness check is made AFTER the cast, deliberately. A longdouble of
    2.4e+308 is finite in longdouble and infinite in float64, so checking
    before the cast would pass the value through and report ``inf``, which is
    the behaviour this function exists to prevent.
    """
    narrowed = float(value)
    if not np.isfinite(narrowed):
        raise ValueError(
            f"the {name} of these {size} readings is not representable as a float64 "
            f"(computed {value!r} in wide arithmetic, which overflows on the way "
            "down). The sample's dynamic range "
            "exceeds what a float64 summary can carry; rescale the channel or "
            "summarise a narrower window. It is refused rather than reported as "
            "infinity, because an infinite figure in a report reads as a number."
        )
    return narrowed


def _aggregates(values: np.ndarray, size: int) -> dict[str, float]:
    """Mean, sd, min, median, p95 and max, computed without intermediate overflow.

    See the module docstring: ``numpy.mean`` sums before dividing and
    ``numpy.std`` squares, so both overflow on samples whose aggregates are
    representable. Everything here is computed in ``numpy.longdouble``, the
    standard deviation on centred values, and each result is checked before it
    is cast down.
    """
    wide = np.asarray(values, dtype=np.longdouble)
    mean = wide.mean()
    result = {
        "mean": _finite("mean", mean, size),
        "min": _finite("minimum", wide.min(), size),
        "median": _finite("median", np.median(wide), size),
        "p95": _finite("95th percentile", np.quantile(wide, 0.95), size),
        "max": _finite("maximum", wide.max(), size),
    }
    if size >= 2:
        result["sd"] = _finite("standard deviation", (wide - mean).std(ddof=1), size)
    return result


def _series(readings: Sequence[Reading]) -> tuple[np.ndarray, str, Attribution]:
    """Verified values, their common unit, and the attribution of the set."""
    verified = [verify(reading) for reading in readings]
    if not verified:
        raise ValueError(
            "no readings were supplied; a statistic over an empty set has no provenance "
            "and is refused rather than returned with the lattice identity"
        )
    units = {reading.unit for reading in verified}
    if len(units) != 1:
        raise ValueError(
            "readings carry more than one unit ("
            + ", ".join(sorted(units))
            + "); aggregating across units would produce a number with no unit at all"
        )
    channels = {reading.channel for reading in verified}
    if len(channels) != 1:
        raise ValueError(
            "readings come from more than one channel ("
            + ", ".join(sorted(channels))
            + "); use summarise() per channel, or derive() to combine across channels"
        )
    return (
        np.asarray([reading.value for reading in verified], dtype=float),
        next(iter(units)),
        attribute(verified),
    )


def summarise(readings: Sequence[Reading]) -> tuple[Statistic, ...]:
    """Count, mean, sd, min, median, p95 and max of one channel's readings.

    The sample standard deviation uses ``ddof=1`` and is omitted entirely for a
    single reading, rather than returned as ``nan``: a ``nan`` propagates into
    a report and reads as a number that failed, when the honest statement is
    that the quantity is undefined at ``n = 1``.

    The 95th percentile uses ``numpy.quantile``'s default linear
    interpolation, which is a library default and is named here because a
    change in it would move the reported figure.
    """
    values, unit, attribution = _series(readings)
    figures = _aggregates(values, values.size)
    statistics = [
        Statistic("count", float(values.size), "readings", attribution),
        Statistic("mean", figures["mean"], unit, attribution),
        Statistic("min", figures["min"], unit, attribution),
        Statistic("median", figures["median"], unit, attribution),
        Statistic("p95", figures["p95"], unit, attribution),
        Statistic("max", figures["max"], unit, attribution),
    ]
    if "sd" in figures:
        statistics.insert(2, Statistic("sd", figures["sd"], unit, attribution))
    return tuple(statistics)


def derive(
    name: str, value: float, unit: str, parts: Iterable[Statistic | Attribution]
) -> Statistic:
    """A figure derived from other figures, carrying the meet of their attributions.

    ``parts`` may mix :class:`Statistic` and
    :class:`~boardprobe.readings.Attribution`. The result's provenance is the
    meet over all of them, so one simulated part makes the derived figure
    simulated however many live parts there are.
    """
    attributions: list[Attribution] = []
    for index, part in enumerate(parts):
        if type(part) is Statistic:
            attributions.append(part.attribution)
        elif type(part) is Attribution:
            attributions.append(part)
        else:
            raise TypeError(
                f"parts[{index}] must be a Statistic or an Attribution, got "
                f"{type(part).__module__}.{type(part).__name__}"
            )
    if not attributions:
        raise ValueError(
            f"derived figure {name!r} was given no parts, so its provenance cannot be "
            "established; see boardprobe.provenance.meet_all"
        )
    return Statistic(name, float(value), unit, combine(attributions))


def rail_power(voltage: Reading, current: Reading, name: str) -> Statistic:
    """Instantaneous rail power in watts, from a voltage and a current reading.

    The INA3221 driver on this board exposes bus voltage and current and not
    power (``Documentation/hwmon/ina3221.rst``), so power is a derivation and
    is treated as one: the result carries the meet of the two readings'
    provenances, which means a live voltage multiplied by a simulated current
    is ``SIMULATED``.

    Units: ``voltage`` in volts, ``current`` in amperes, result in watts. The
    units are checked, because a millivolt mistaken for a volt is a factor of
    a thousand in a power figure.
    """
    for reading, expected, role in (
        (voltage, "V", "voltage"),
        (current, "A", "current"),
    ):
        verify(reading)
        if reading.unit != expected:
            raise ValueError(
                f"rail power {name!r}: the {role} reading on channel "
                f"{reading.channel!r} is in {reading.unit!r} and must be in "
                f"{expected!r}; the channel's declared scale is what converts the raw "
                "interface unit, and a mismatch here is a factor-of-1000 error in watts"
            )
    return derive(
        name,
        voltage.value * current.value,
        "W",
        [attribute([voltage]), attribute([current])],
    )
