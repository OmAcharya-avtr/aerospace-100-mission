"""Cross-model disagreement, and the elevation at which it crosses a declared tolerance.

This is the output the product exists to produce. Every model in
:mod:`refractpath.models` and the ray trace of :mod:`refractpath.raytrace`
computes the same physical quantity from the same inputs, and below some
elevation they stop agreeing. The elevation at which they stop agreeing to
within a pointing tolerance the caller declares is **measured here by
bisection**, it is not asserted, and the measurement reports its own bracket.

Equation (22), the disagreement:

    D(h) = max over i, j of | R_i(h) - R_j(h) |

over the members of the ensemble that are **defined** at ``h``. A member that
returns ``nan`` is excluded from the maximum and named in
:attr:`SpreadPoint.undefined_members`, because a model that refuses to answer
has not disagreed.

Equation (22) is not monotone in ``h``, and this is not a numerical artefact.
When the apparent elevation falls below the analytic floor of equation (13) the
plane-parallel model leaves the ensemble, and because it is by then the most
discrepant member, ``D`` **drops**. :func:`validity_floor` therefore does not
assume a single crossing: it scans a declared grid, counts every crossing it
finds, reports whether the curve was monotone on that grid, and takes the
**highest** elevation at which the tolerance is exceeded as the floor. Taking
the lowest crossing, which a naive bisection from the horizon upward would
find, understates the floor by a factor that is measured in
``validation/validate_disagreement_floor.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import AtmosphericState, Observer
from .models import CLOSED_FORM_MODELS, ClosedFormModel
from .raytrace import raytrace_refraction


#: The disagreement ensemble: the four closed forms of baseline 1 plus the ray
#: trace of baseline 2. The ray trace is a member rather than only a reference
#: because it is itself a standard method, and because an ensemble of closed
#: forms alone agrees with itself for the wrong reason -- three of the four
#: were fitted to the same kind of rigorous calculation.
@dataclass(frozen=True)
class EnsembleMember:
    """One member of the disagreement ensemble.

    Attributes:
        name: Key used in every output.
        function: Callable ``(elevation_deg, state, observer,
            refractivity_model) -> arcsec``.
        kind: ``"closed_form"`` or ``"raytrace"``.
        validity_floor_deg: Declared or published floor in apparent elevation.
        citation: The published source.
    """

    name: str
    function: Callable[..., float]
    kind: str
    validity_floor_deg: float
    citation: str


def default_ensemble(
    closed_forms: Iterable[ClosedFormModel] = CLOSED_FORM_MODELS,
) -> tuple[EnsembleMember, ...]:
    """The default ensemble: every closed form plus the ray trace.

    Args:
        closed_forms: Closed-form models to include.

    Returns:
        The ensemble, in a fixed order.
    """
    members = [
        EnsembleMember(
            name=model.name,
            function=model.function,
            kind="closed_form",
            validity_floor_deg=model.validity_floor_deg,
            citation=model.citation,
        )
        for model in closed_forms
    ]
    members.append(
        EnsembleMember(
            name="raytrace",
            function=raytrace_refraction,
            kind="raytrace",
            validity_floor_deg=0.0,
            citation="Bouguer's formula; Auer and Standish (2000) eq. (4)",
        )
    )
    return tuple(members)


@dataclass(frozen=True)
class SpreadPoint:
    """The ensemble at one elevation.

    Attributes:
        elevation_deg: Apparent elevation, degrees.
        values_arcsec: Each defined member's refraction, arcseconds.
        undefined_members: Members that returned ``nan`` here.
        spread_arcsec: Equation (22).
        minimum_name: Member holding the smallest value.
        maximum_name: Member holding the largest value.
        mean_arcsec: Mean over the defined members, arcseconds.
    """

    elevation_deg: float
    values_arcsec: dict[str, float]
    undefined_members: tuple[str, ...]
    spread_arcsec: float
    minimum_name: str
    maximum_name: str
    mean_arcsec: float


def ensemble_at(
    elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    members: Sequence[EnsembleMember] | None = None,
    refractivity_model: str = "ciddor_1996",
) -> SpreadPoint:
    """Evaluate every ensemble member at one elevation and form equation (22).

    Args:
        elevation_deg: Apparent elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Observer. Defaults to sea level on the equator.
        members: Ensemble. Defaults to :func:`default_ensemble`.
        refractivity_model: Formulation passed to the members that take one.

    Returns:
        The spread at that elevation.

    Raises:
        ValueError: If no ensemble member is defined at ``elevation_deg``.
    """
    ensemble = default_ensemble() if members is None else tuple(members)
    if not ensemble:
        raise ValueError("members must not be empty")
    values: dict[str, float] = {}
    undefined: list[str] = []
    for member in ensemble:
        result = member.function(elevation_deg, state, observer, refractivity_model)
        if result is None or not math.isfinite(float(result)):
            undefined.append(member.name)
        else:
            values[member.name] = float(result)
    if not values:
        raise ValueError(
            f"no ensemble member is defined at {elevation_deg!r} degrees apparent "
            f"elevation; undefined: {', '.join(undefined)}"
        )
    minimum_name = min(values, key=lambda key: values[key])
    maximum_name = max(values, key=lambda key: values[key])
    return SpreadPoint(
        elevation_deg=float(elevation_deg),
        values_arcsec=values,
        undefined_members=tuple(undefined),
        spread_arcsec=values[maximum_name] - values[minimum_name],
        minimum_name=minimum_name,
        maximum_name=maximum_name,
        mean_arcsec=sum(values.values()) / len(values),
    )


def spread_curve(
    elevations_deg: Sequence[float] | np.ndarray,
    state: AtmosphericState,
    observer: Observer | None = None,
    members: Sequence[EnsembleMember] | None = None,
    refractivity_model: str = "ciddor_1996",
) -> tuple[SpreadPoint, ...]:
    """Equation (22) over a grid of elevations.

    Args:
        elevations_deg: Apparent elevations, degrees.
        state: Atmospheric state at the observer.
        observer: Observer.
        members: Ensemble.
        refractivity_model: Formulation passed to the members that take one.

    Returns:
        One :class:`SpreadPoint` per elevation, in the order given.
    """
    grid = np.asarray(elevations_deg, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"elevations_deg must be a non-empty 1-D array, got {grid.shape}")
    return tuple(
        ensemble_at(float(value), state, observer, members, refractivity_model)
        for value in grid
    )


#: Default elevation grid for the floor scan, degrees. Logarithmically dense
#: near the horizon, because that is where the disagreement is and because a
#: uniform grid wastes its resolution where every model agrees.
def default_floor_grid(
    low_deg: float = 0.0, high_deg: float = 90.0, points: int = 181
) -> np.ndarray:
    """Elevation grid for the floor scan, dense near the horizon.

    Args:
        low_deg: Lowest elevation, degrees.
        high_deg: Highest elevation, degrees.
        points: Number of grid points.

    Returns:
        Ascending grid of ``points`` elevations with
        ``h = low + (high - low) * (i / (points - 1)) ** 2``.

    Raises:
        ValueError: If the interval is empty or ``points`` is below 3.
    """
    if points < 3:
        raise ValueError(f"points must be at least 3, got {points!r}")
    if not high_deg > low_deg:
        raise ValueError(f"high_deg must exceed low_deg, got {low_deg!r}, {high_deg!r}")
    fraction = np.arange(points, dtype=float) / float(points - 1)
    return low_deg + (high_deg - low_deg) * fraction**2


@dataclass(frozen=True)
class ValidityFloor:
    """The measured elevation at which equation (22) crosses a declared tolerance.

    Attributes:
        tolerance_arcsec: The pointing tolerance the caller declared.
        floor_deg: Highest apparent elevation at which the disagreement is at
            or above the tolerance, refined by bisection. ``nan`` if the
            tolerance is never reached on the grid.
        bracket_deg: The bracket the bisection closed on. ``(nan, nan)`` if
            there was no crossing.
        bracket_width_deg: Width of that bracket, degrees.
        spread_below_arcsec: Equation (22) at the lower edge of the bracket.
        spread_above_arcsec: Equation (22) at the upper edge of the bracket.
        crossings: Number of grid intervals in which the sign of
            ``D(h) - tolerance`` changes. More than one means the disagreement
            curve is not monotone and a single bisection would have been wrong.
        monotone_on_grid: Whether ``D`` is non-increasing in elevation over the
            whole grid.
        exceeded_at_highest_grid_point: Whether the tolerance is exceeded at
            the top of the grid, in which case the floor is not inside it.
        never_exceeded: Whether the tolerance is met everywhere on the grid.
        grid_deg: The grid that was scanned.
        spread_arcsec: Equation (22) on that grid.
        undefined_counts: How many members were undefined at each grid point.
        iterations: Bisection iterations used.
    """

    tolerance_arcsec: float
    floor_deg: float
    bracket_deg: tuple[float, float]
    bracket_width_deg: float
    spread_below_arcsec: float
    spread_above_arcsec: float
    crossings: int
    monotone_on_grid: bool
    exceeded_at_highest_grid_point: bool
    never_exceeded: bool
    grid_deg: np.ndarray = field(repr=False)
    spread_arcsec: np.ndarray = field(repr=False)
    undefined_counts: np.ndarray = field(repr=False)
    iterations: int = 0


def validity_floor(
    state: AtmosphericState,
    tolerance_arcsec: float,
    observer: Observer | None = None,
    members: Sequence[EnsembleMember] | None = None,
    refractivity_model: str = "ciddor_1996",
    grid_deg: Sequence[float] | np.ndarray | None = None,
    bisection_tolerance_deg: float = 1e-4,
    max_iterations: int = 60,
) -> ValidityFloor:
    """Measure the elevation at which equation (22) crosses ``tolerance_arcsec``.

    The scan takes the **highest** crossing, not the first one found from the
    horizon. See the module docstring for why.

    Args:
        state: Atmospheric state at the observer.
        tolerance_arcsec: Declared pointing tolerance, arcseconds. Positive.
        observer: Observer.
        members: Ensemble. Defaults to :func:`default_ensemble`.
        refractivity_model: Formulation passed to the members that take one.
        grid_deg: Elevation grid to scan. Defaults to
            :func:`default_floor_grid`.
        bisection_tolerance_deg: Bracket width at which to stop bisecting.
        max_iterations: Hard iteration cap.

    Returns:
        The measured floor and the evidence for it.

    Raises:
        ValueError: On a non-positive or non-finite tolerance.
    """
    if not math.isfinite(tolerance_arcsec) or tolerance_arcsec <= 0.0:
        raise ValueError(
            f"tolerance_arcsec must be positive and finite, got {tolerance_arcsec!r}"
        )
    if bisection_tolerance_deg <= 0.0:
        raise ValueError(
            f"bisection_tolerance_deg must be positive, got {bisection_tolerance_deg!r}"
        )
    grid = default_floor_grid() if grid_deg is None else np.asarray(grid_deg, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError(f"grid_deg must be a 1-D array of at least 3 points, got {grid.shape}")

    points = spread_curve(grid, state, observer, members, refractivity_model)
    spread = np.array([point.spread_arcsec for point in points], dtype=float)
    undefined = np.array([len(point.undefined_members) for point in points], dtype=int)

    exceeded = spread >= tolerance_arcsec
    crossings = int(np.count_nonzero(exceeded[:-1] != exceeded[1:]))
    monotone = bool(np.all(np.diff(spread) <= 1e-12))

    if not exceeded.any():
        return ValidityFloor(
            tolerance_arcsec=float(tolerance_arcsec),
            floor_deg=math.nan,
            bracket_deg=(math.nan, math.nan),
            bracket_width_deg=math.nan,
            spread_below_arcsec=math.nan,
            spread_above_arcsec=math.nan,
            crossings=crossings,
            monotone_on_grid=monotone,
            exceeded_at_highest_grid_point=False,
            never_exceeded=True,
            grid_deg=grid,
            spread_arcsec=spread,
            undefined_counts=undefined,
        )

    highest = int(np.flatnonzero(exceeded)[-1])
    if highest == grid.size - 1:
        return ValidityFloor(
            tolerance_arcsec=float(tolerance_arcsec),
            floor_deg=float(grid[-1]),
            bracket_deg=(float(grid[-1]), float(grid[-1])),
            bracket_width_deg=0.0,
            spread_below_arcsec=float(spread[-1]),
            spread_above_arcsec=float(spread[-1]),
            crossings=crossings,
            monotone_on_grid=monotone,
            exceeded_at_highest_grid_point=True,
            never_exceeded=False,
            grid_deg=grid,
            spread_arcsec=spread,
            undefined_counts=undefined,
        )

    low, high = float(grid[highest]), float(grid[highest + 1])
    spread_low, spread_high = float(spread[highest]), float(spread[highest + 1])
    iterations = 0
    while high - low > bisection_tolerance_deg and iterations < max_iterations:
        middle = 0.5 * (low + high)
        value = ensemble_at(middle, state, observer, members, refractivity_model).spread_arcsec
        if value >= tolerance_arcsec:
            low, spread_low = middle, value
        else:
            high, spread_high = middle, value
        iterations += 1

    return ValidityFloor(
        tolerance_arcsec=float(tolerance_arcsec),
        floor_deg=low,
        bracket_deg=(low, high),
        bracket_width_deg=high - low,
        spread_below_arcsec=spread_low,
        spread_above_arcsec=spread_high,
        crossings=crossings,
        monotone_on_grid=monotone,
        exceeded_at_highest_grid_point=False,
        never_exceeded=False,
        grid_deg=grid,
        spread_arcsec=spread,
        undefined_counts=undefined,
        iterations=iterations,
    )


def pairwise_disagreement(point: SpreadPoint) -> dict[tuple[str, str], float]:
    """Every pairwise absolute difference at one elevation, arcseconds.

    Args:
        point: A spread point.

    Returns:
        Mapping from an ordered name pair to the absolute difference. Only
        defined members appear.
    """
    names = sorted(point.values_arcsec)
    out: dict[tuple[str, str], float] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            out[(first, second)] = abs(
                point.values_arcsec[first] - point.values_arcsec[second]
            )
    return out
