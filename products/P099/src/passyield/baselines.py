"""The three baselines, in the order the specification fixes them.

BASELINE 1 -- THE BOOKED FIGURE, IMPLEMENTED AT FULL STRENGTH
-------------------------------------------------------------
Nominal window duration times nominal rate. :func:`booked_shortfall_mbit`
returns 0.0 Mbit, because a schedule that books :math:`T R_{nom}` is asserting
that the realised yield will be :math:`T R_{nom}` and therefore that the
shortfall will be zero.

This is the thing the product exists to refute, and it is built at full
strength deliberately rather than as a straw man. "Full strength" means two
specific choices:

- the duration is the **true geometric mask-to-mask window**, computed from
  the committed SGP4 element set, not a padded or rounded block;
- the nominal rate is the rate the link budget gives at a declared reference
  elevation with the station's own figure of merit, which is exactly what a
  scheduler with the station catalogue and an ephemeris can compute before the
  pass. See :func:`passyield.decompose.nominal_rate_mbit_s`.

Weakening either would inflate every later comparison. A previous product of
this mission measured an under-strength baseline inflating an apparent learned
win by about half, so the strength of this one is a correctness requirement.

BASELINE 2 -- THE ANALYTIC YIELD MODEL, BY NUMERICAL CONVOLUTION
----------------------------------------------------------------
The mathematical core of the product.

The realised yield of a pass is the capacity of the window less three losses,
each the integral of the rate profile over a set of intervals. Write
:math:`\\Lambda(t)=\\int_0^t r(s)ds` for the cumulative volume, so that the
loss from an interval :math:`[u,u+d]` is :math:`\\Lambda(u+d)-\\Lambda(u)`
exactly. Then

- the **acquisition loss** is :math:`\\Lambda(T_{acq})`, a deterministic
  strictly increasing function of the acquisition time, so its law is the
  pushforward of the acquisition-time law under :math:`\\Lambda`;
- each **re-synchronisation loss** is
  :math:`\\Lambda(U+D)-\\Lambda(U)` with :math:`D` the outage duration and
  :math:`U` its start, so its law is the pushforward of the joint law of
  :math:`(U,D)`; the number of outages is :math:`N`, so the total
  re-synchronisation loss is a **compound Poisson sum**, whose law is
  :math:`\\sum_n P(N=n) f_1^{*n}` -- an :math:`n`-fold convolution;
- the **idle loss** is treated the same way as a single interval of random
  total duration and random placement.

The shortfall is the deterministic geometric term :math:`Y_{nom}-Y_{cap}` plus
the sum of the three, so its law is the convolution of the three laws, shifted.

**Numerical, not closed form.** There is no closed form: :math:`\\Lambda` is an
SGP4-driven integral with no elementary inverse, so neither pushforward has an
elementary density and the compound sum has no elementary transform.
:func:`analytic_shortfall_distribution` therefore works on a **uniform grid of
data volume in Mbit** and convolves with :func:`numpy.convolve`.

**Discretisation and its error.**

- Volume grid: ``bins`` uniform bins spanning ``[0, Y_cap]``, so the step is
  :math:`\\Delta = Y_{cap}/\\text{bins}`. A quantile read from the result
  carries at least :math:`\\Delta/2` of discretisation error by construction;
  :meth:`passyield.distribution.ShortfallDistribution.quantile_discretisation_mbit`
  states it. The pushforward is **mass-preserving** -- probability is assigned
  to the bin containing the image of each source cell, not interpolated -- so
  the error in the cumulative distribution function is first order in
  :math:`\\Delta` and :math:`O(1/n_{cells})` in the source grids.
  ``validation/validate_convolution.py`` halves the step twice and reports the
  **measured** convergence order rather than relying on this paragraph.
- Source grids: ``time_cells`` cells for the acquisition law, and
  ``duration_cells`` by ``placement_cells`` for the outage and idle laws.
- :math:`\\Lambda` itself is a cumulative composite Simpson integral of the
  rate profile on ``profile_nodes`` nodes (``scipy.integrate.cumulative_simpson``).

**The structural approximation, stated plainly.** The convolution treats the
three losses as independent and additive. They are not: the intervals must be
disjoint inside one window, so a draw in which they sum beyond the window is
physically impossible. The approximation is in the direction of over-stating
the shortfall, and it is **measurable**: any probability the convolution puts
above the window capacity is probability the independence assumption invented.
That mass is reported as ``truncated_mass`` on the returned distribution, and
``validation/validate_convolution.py`` publishes it over the campaign. This is
also why baseline 2 is not ground truth, which is baseline 3's job.

BASELINE 3 -- GROUND TRUTH, ACCOUNTED, WITH A PASS-LEVEL BOOTSTRAP
------------------------------------------------------------------
Directly accounted from the per-pass record by :mod:`passyield.decompose`, no
model of anything. Confidence intervals by the non-parametric percentile
bootstrap (Efron, B. (1979), "Bootstrap Methods: Another Look at the
Jackknife", Annals of Statistics 7(1), 1-26; Efron & Tibshirani (1993), "An
Introduction to the Bootstrap", chapter 13 for the percentile interval).

**The resampling unit is the PASS.** That is fixed by the specification, it is
recorded in :attr:`AccountedYield.resampling_unit` on the returned object, and
``tests/test_baselines.py`` asserts the attribute rather than trusting this
docstring. It matters: resampling individual seconds or individual outages
would treat within-pass structure as independent replication and shrink every
interval, which is the standard way a bootstrap interval becomes a fiction.
The pass is the unit at which a scheduler makes a decision, so it is also the
unit the interval should describe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_simpson
from sgp4.api import Satrec

from .decompose import Decomposition, DecompositionOptions, rate_at_offsets
from .distribution import (
    PASS_RESAMPLING_UNIT,
    ShortfallDistribution,
    distribution_from_samples,
)
from .geometry import GroundStation, PassWindow

#: Default bins of the volume grid. 1024 keeps a quantile's discretisation
#: error at roughly 0.05 per cent of the capacity and keeps every convolution
#: here well inside a millisecond.
DEFAULT_BINS = 1024

#: Poisson tail probability at which the compound sum is truncated. The
#: truncated probability is added to the distribution's ``truncated_mass``, so
#: it is reported and not silently dropped.
POISSON_TAIL_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class EventModel:
    """Measured laws of acquisition, outage and idle time for one station class.

    Every parameter here is fitted from the committed per-pass record by the
    method of moments in :func:`fit_event_model`; none is chosen by hand.

    The families are declared, not discovered, and the reason for each is
    stated so that a reader can disagree with it:

    - **acquisition time: Gamma.** Acquisition is a sum of positive stage
      durations -- antenna slew, carrier lock, symbol and frame
      synchronisation -- so a Gamma (the Erlang family when the stages are
      exponential and alike) is the natural two-parameter family on the
      positive half-line for a sum of stages. Units seconds.
    - **outage count: Poisson with an intensity per second.** Loss of lock
      treated as a homogeneous point process over the window. The homogeneity
      is wrong in a knowable direction -- low elevation is harder -- and that
      is one of the things the learned component is given the chance to find.
    - **outage duration: exponential.** A memoryless re-acquisition: the time
      already spent trying tells you nothing about the time left. Units
      seconds.
    - **idle total: Gamma.** Same argument as acquisition, applied to the sum
      of protocol stalls over the pass. Units seconds.

    Attributes
    ----------
    acquisition_shape, acquisition_scale:
        Gamma parameters of the acquisition time, shape dimensionless, scale
        seconds.
    outage_intensity_per_s:
        Poisson intensity of loss-of-lock events, 1/s.
    outage_mean_s:
        Mean outage duration, seconds.
    idle_shape, idle_scale:
        Gamma parameters of the total idle time, shape dimensionless, scale
        seconds.
    sample_passes:
        Number of passes the moments were taken from. Reported so a reader can
        judge the fit.
    """

    acquisition_shape: float
    acquisition_scale: float
    outage_intensity_per_s: float
    outage_mean_s: float
    idle_shape: float
    idle_scale: float
    sample_passes: int

    def __post_init__(self) -> None:
        for name in (
            "acquisition_shape",
            "acquisition_scale",
            "outage_mean_s",
            "idle_shape",
            "idle_scale",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not np.isfinite(self.outage_intensity_per_s) or self.outage_intensity_per_s < 0.0:
            raise ValueError(
                "outage_intensity_per_s must be finite and non-negative, got "
                f"{self.outage_intensity_per_s!r}"
            )
        if self.sample_passes < 1:
            raise ValueError(f"sample_passes must be at least 1, got {self.sample_passes!r}")


def _gamma_moments(values: np.ndarray, floor: float) -> tuple[float, float]:
    """Method-of-moments Gamma shape and scale from a sample.

    ``floor`` guards a degenerate sample: a sample with zero variance, or a
    mean at zero, has no Gamma in its closure, and the fitted scale is floored
    rather than producing a non-finite parameter. The floor is a declared
    number in seconds and is reported by the caller.
    """
    sample = np.asarray(values, dtype=float)
    mean = float(np.mean(sample))
    variance = float(np.var(sample, ddof=1)) if sample.size > 1 else 0.0
    mean = max(mean, floor)
    variance = max(variance, floor**2)
    scale = variance / mean
    shape = mean / scale
    return float(shape), float(max(scale, floor))


def fit_event_model(
    acquisition_s: Sequence[float],
    outage_counts: Sequence[int],
    window_durations_s: Sequence[float],
    outage_durations_s: Sequence[float],
    idle_totals_s: Sequence[float],
    floor_s: float = 1.0e-3,
) -> EventModel:
    """Fit :class:`EventModel` by the method of moments. Units seconds.

    The Poisson intensity is the total number of outages divided by the total
    observed window time, which is the maximum-likelihood estimator for a
    homogeneous Poisson process observed over several windows of unequal
    length.
    """
    acquisition = np.asarray(acquisition_s, dtype=float)
    counts = np.asarray(outage_counts, dtype=float)
    windows = np.asarray(window_durations_s, dtype=float)
    durations = np.asarray(outage_durations_s, dtype=float)
    idle = np.asarray(idle_totals_s, dtype=float)
    if acquisition.size == 0:
        raise ValueError("acquisition_s must be non-empty: there is nothing to fit")
    if windows.size != counts.size:
        raise ValueError(
            f"outage_counts and window_durations_s must align, got {counts.size} and {windows.size}"
        )
    if np.any(windows <= 0.0):
        raise ValueError("window_durations_s must be positive everywhere")
    acquisition_shape, acquisition_scale = _gamma_moments(acquisition, floor_s)
    idle_shape, idle_scale = _gamma_moments(idle, floor_s)
    intensity = float(counts.sum() / windows.sum())
    outage_mean = float(np.mean(durations)) if durations.size else floor_s
    return EventModel(
        acquisition_shape=acquisition_shape,
        acquisition_scale=acquisition_scale,
        outage_intensity_per_s=intensity,
        outage_mean_s=max(outage_mean, floor_s),
        idle_shape=idle_shape,
        idle_scale=idle_scale,
        sample_passes=int(acquisition.size),
    )


@dataclass(frozen=True)
class ConvolutionOptions:
    """Discretisation of the numerical convolution. All of it is reported."""

    bins: int = DEFAULT_BINS
    profile_nodes: int = 513
    time_cells: int = 257
    duration_cells: int = 49
    placement_cells: int = 33

    def __post_init__(self) -> None:
        if self.bins < 16:
            raise ValueError(f"bins must be at least 16, got {self.bins!r}")
        if self.profile_nodes < 5 or self.profile_nodes % 2 == 0:
            raise ValueError(
                f"profile_nodes must be an odd integer of at least 5, got {self.profile_nodes!r}"
            )
        for name in ("time_cells", "duration_cells", "placement_cells"):
            value = getattr(self, name)
            if value < 4:
                raise ValueError(f"{name} must be at least 4, got {value!r}")


def booked_shortfall_mbit() -> float:
    """Baseline 1's prediction of the shortfall, Mbit.

    Zero. A schedule that books ``T * R_nom`` predicts that ``T * R_nom`` will
    arrive. Returned as a named function rather than a literal so that every
    comparison table in this product has a call to point at, and so that the
    claim "baseline 1 predicts no shortfall" is in the code and not only in a
    document.
    """
    return 0.0


def cumulative_volume(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    options: DecompositionOptions | None = None,
    profile_nodes: int = 513,
) -> tuple[np.ndarray, np.ndarray]:
    """``(offsets_s, Lambda_mbit)``: cumulative delivered volume over the window.

    Cumulative composite Simpson integration of the rate profile
    (``scipy.integrate.cumulative_simpson``), with ``Lambda(0) = 0``. Units
    seconds in, Mbit out. The result is non-decreasing because the rate is
    non-negative, and ``tests/test_baselines.py`` asserts that.
    """
    settings = options if options is not None else DecompositionOptions()
    if profile_nodes < 5 or profile_nodes % 2 == 0:
        raise ValueError(f"profile_nodes must be an odd integer of at least 5, got {profile_nodes}")
    offsets = np.linspace(0.0, window.duration_s, profile_nodes)
    rates, _clamp = rate_at_offsets(satrec, station, window, offsets, settings.budget)
    cumulative = cumulative_simpson(rates, x=offsets, initial=0.0)
    return offsets, np.asarray(cumulative, dtype=float)


def backlog_dry_time_s(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    backlog_mbit: float,
    options: DecompositionOptions | None = None,
    profile_nodes: int = 513,
) -> float:
    r"""Seconds at the end of the window the backlog cannot fill, pre-pass.

    A scheduler knows the onboard backlog and the capacity profile before the
    contact, so it can compute the instant :math:`\Lambda^{-1}(\text{backlog})`
    at which the queue would run dry and the window time after it. This is the
    ingredient of the strengthened baseline 2 variant described in
    :func:`analytic_shortfall_distribution`.

    Returns 0.0 when the backlog exceeds the whole window capacity.
    """
    if backlog_mbit < 0.0:
        raise ValueError(f"backlog_mbit must be non-negative, got {backlog_mbit!r}")
    offsets, cumulative = cumulative_volume(satrec, station, window, options, profile_nodes)
    if backlog_mbit >= cumulative[-1]:
        return 0.0
    fill = float(np.interp(backlog_mbit, cumulative, offsets))
    return max(0.0, float(window.duration_s) - fill)


def _loss_of_interval(
    offsets_s: np.ndarray, cumulative_mbit: np.ndarray, start_s: np.ndarray, stop_s: np.ndarray
) -> np.ndarray:
    """``Lambda(stop) - Lambda(start)`` by linear interpolation, Mbit."""
    low = np.interp(np.clip(start_s, offsets_s[0], offsets_s[-1]), offsets_s, cumulative_mbit)
    high = np.interp(np.clip(stop_s, offsets_s[0], offsets_s[-1]), offsets_s, cumulative_mbit)
    return np.maximum(high - low, 0.0)


def _bin_mass(values_mbit: np.ndarray, weights: np.ndarray, bins: int, step: float) -> np.ndarray:
    """Mass-preserving assignment of weighted values to a ``[0, bins*step]`` grid.

    Values at or beyond the top of the grid land in the last bin. For the
    laws here the top of the grid is the window capacity, which every loss is
    bounded by, so that case is a rounding event and not a truncation; the
    truncation that matters happens later, in the convolution.
    """
    index = np.clip(np.floor(np.asarray(values_mbit) / step).astype(int), 0, bins - 1)
    mass = np.zeros(bins)
    np.add.at(mass, index, np.asarray(weights, dtype=float))
    return mass


def _convolve_truncated(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    """Convolve and fold everything past the grid length into a lost mass."""
    full = np.convolve(left, right)
    keep = left.size
    return full[:keep], float(full[keep:].sum())


def analytic_shortfall_distribution(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    model: EventModel,
    nominal_bits: float,
    capacity_bits: float,
    options: ConvolutionOptions | None = None,
    decomposition_options: DecompositionOptions | None = None,
    deterministic_idle_s: float | None = None,
) -> ShortfallDistribution:
    """Baseline 2: the analytic shortfall law of one pass, by convolution.

    See the module docstring for the mathematics, the discretisation, the
    measured error and the structural independence approximation.

    Parameters
    ----------
    nominal_bits, capacity_bits:
        The booked and capacity volumes of this window, Mbit, as
        :mod:`passyield.decompose` computes them. Passed in rather than
        recomputed so that baselines 1, 2 and 3 are provably looking at the
        same window.
    deterministic_idle_s:
        When given, the fitted idle law is **replaced** by a deterministic
        idle interval of this length at the end of the window, and the law of
        the remaining two losses is shifted by its exact volume. This is the
        **strengthened variant** of baseline 2: the backlog-limited dry time
        from :func:`backlog_dry_time_s` is knowable before the pass, so an
        analytic model is entitled to use it, and the specified unconditional
        form is leaving information on the table. Both forms are measured and
        both are published; see ``validation/validate_learned.py``. Shifting
        the grid rather than convolving a delta is the same operation and
        avoids a binning rounding.

    Returns
    -------
    ShortfallDistribution
        ``method="numerical-convolution"``, grid shifted by the geometric
        term, ``truncated_mass`` carrying both the Poisson truncation and the
        probability the independence approximation placed above capacity.
    """
    settings = options if options is not None else ConvolutionOptions()
    if capacity_bits <= 0.0:
        raise ValueError(f"capacity_bits must be positive, got {capacity_bits!r}")
    offsets, cumulative = cumulative_volume(
        satrec, station, window, decomposition_options, settings.profile_nodes
    )
    duration = float(window.duration_s)
    bins = settings.bins
    step = capacity_bits / bins

    # --- acquisition: pushforward of a Gamma under the cumulative volume ----
    edges = np.linspace(0.0, duration, settings.time_cells + 1)
    cdf = stats.gamma.cdf(edges, a=model.acquisition_shape, scale=model.acquisition_scale)
    cell_mass = np.diff(cdf)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    acquisition_loss = np.interp(midpoints, offsets, cumulative)
    # Acquisition longer than the pass loses the whole window: documented
    # edge case, and the mass is kept rather than dropped.
    beyond = float(1.0 - cdf[-1])
    acquisition_mass = _bin_mass(acquisition_loss, cell_mass, bins, step)
    acquisition_mass[-1] += beyond

    # --- one outage: pushforward of (duration, placement) -------------------
    outage_mass = _interval_loss_mass(
        offsets, cumulative, duration, bins, step,
        mean_s=model.outage_mean_s, shape=1.0,
        duration_cells=settings.duration_cells, placement_cells=settings.placement_cells,
    )

    # --- compound Poisson over the outage count -----------------------------
    # resync_mass = sum_n P(N=n) * outage_mass^{*n}, with the n=0 term a unit
    # mass at zero loss. The sum is truncated where the Poisson survival
    # function falls below POISSON_TAIL_TOLERANCE, and that truncated
    # probability is carried into the reported truncated mass below.
    intensity = model.outage_intensity_per_s * duration
    count_max = int(stats.poisson.isf(POISSON_TAIL_TOLERANCE, mu=intensity)) + 1
    resync_mass = np.zeros(bins)
    resync_mass[0] = float(stats.poisson.pmf(0, mu=intensity))
    term = np.zeros(bins)
    term[0] = 1.0
    for count in range(1, count_max + 1):
        term, _folded = _convolve_truncated(term, outage_mass)
        resync_mass += float(stats.poisson.pmf(count, mu=intensity)) * term

    # --- idle: either the fitted law, or a deterministic dry interval -------
    if deterministic_idle_s is None:
        idle_mass = _interval_loss_mass(
            offsets, cumulative, duration, bins, step,
            mean_s=model.idle_shape * model.idle_scale, shape=model.idle_shape,
            duration_cells=settings.duration_cells, placement_cells=settings.placement_cells,
        )
        idle_shift = 0.0
    else:
        if deterministic_idle_s < 0.0:
            raise ValueError(
                f"deterministic_idle_s must be non-negative, got {deterministic_idle_s!r}"
            )
        dry = min(float(deterministic_idle_s), duration)
        idle_mass = np.zeros(bins)
        idle_mass[0] = 1.0
        idle_shift = float(
            _loss_of_interval(
                offsets, cumulative, np.asarray([duration - dry]), np.asarray([duration])
            )[0]
        )

    total, _lost_one = _convolve_truncated(acquisition_mass, resync_mass)
    total, _lost_two = _convolve_truncated(total, idle_mass)
    # Everything that did not survive on the grid is reported in one number:
    # the Poisson truncation, and the probability the independence
    # approximation placed above the window capacity. Taking it as
    # ``1 - retained`` rather than summing the pieces is deliberate -- it
    # cannot under-count, because it is anchored on the normalisation of a
    # probability measure rather than on this function's own bookkeeping.
    retained = float(total.sum())
    truncated = float(np.clip(1.0 - retained, 0.0, 1.0 - 1.0e-12))
    geometry = nominal_bits - capacity_bits
    grid = geometry + idle_shift + (np.arange(bins) + 0.5) * step
    return ShortfallDistribution(
        method="numerical-convolution",
        grid_mbit=grid,
        mass=total,
        truncated_mass=truncated,
    )


def _interval_loss_mass(
    offsets: np.ndarray,
    cumulative: np.ndarray,
    duration_s: float,
    bins: int,
    step: float,
    mean_s: float,
    shape: float,
    duration_cells: int,
    placement_cells: int,
) -> np.ndarray:
    """Law of ``Lambda(U+D) - Lambda(U)`` on the volume grid.

    ``D`` is Gamma with the given shape and the given mean (so ``shape=1``
    gives the exponential outage law); ``U`` is uniform on
    ``[0, T-D]``, which is the maximum-entropy placement given that the
    interval must fit inside the window. Deterministic: both laws are
    discretised on uniform cells and the joint mass is accumulated, with no
    random number generator anywhere.

    The probability that ``D`` exceeds the whole window loses the whole
    window, so it is accumulated into the top bin rather than discarded. That
    is the documented edge case "an outage longer than the pass", handled
    here as a loss of the full capacity and never as a negative yield.
    """
    scale = mean_s / shape
    edges = np.linspace(0.0, duration_s, duration_cells + 1)
    cdf = stats.gamma.cdf(edges, a=shape, scale=scale)
    cell_mass = np.diff(cdf)
    lengths = 0.5 * (edges[:-1] + edges[1:])
    fractions = (np.arange(placement_cells) + 0.5) / placement_cells
    starts = fractions[None, :] * np.maximum(duration_s - lengths, 0.0)[:, None]
    stops = starts + lengths[:, None]
    losses = _loss_of_interval(offsets, cumulative, starts, stops)
    weights = (cell_mass[:, None] / placement_cells) * np.ones_like(losses)
    mass = _bin_mass(losses.ravel(), weights.ravel(), bins, step)
    mass[-1] += float(1.0 - cdf[-1])
    return mass


@dataclass(frozen=True)
class AccountedYield:
    """Baseline 3: ground truth from the record, with a pass-level bootstrap.

    Attributes
    ----------
    resampling_unit:
        ``"pass"``. The specification fixes it and
        ``tests/test_baselines.py`` asserts this attribute.
    passes:
        Number of passes accounted.
    nominal_mbit, capacity_mbit, realised_mbit, shortfall_mbit:
        Campaign totals, Mbit.
    mean_shortfall_mbit:
        Per-pass mean shortfall, Mbit.
    mean_shortfall_ci_mbit:
        Percentile bootstrap interval of the per-pass mean, Mbit.
    realised_fraction, realised_fraction_ci:
        Campaign realised volume over campaign capacity, dimensionless, and
        its percentile bootstrap interval.
    shortfall_share:
        Each decomposition term's share of the total shortfall,
        dimensionless, keyed by term name. Sums to 1 up to the residual.
    residual_share:
        The decomposition residual's share of the total shortfall. The
        reported closing error of the identity in
        :mod:`passyield.decompose`.
    resamples, confidence:
        Bootstrap settings.
    empirical_distribution:
        The per-pass shortfalls as a :class:`ShortfallDistribution` with
        ``method="empirical"``. This is the distribution baseline 2 and the
        learned model are scored against.
    """

    resampling_unit: str
    passes: int
    nominal_mbit: float
    capacity_mbit: float
    realised_mbit: float
    shortfall_mbit: float
    mean_shortfall_mbit: float
    mean_shortfall_ci_mbit: tuple[float, float]
    realised_fraction: float
    realised_fraction_ci: tuple[float, float]
    shortfall_share: dict[str, float]
    residual_share: float
    resamples: int
    confidence: float
    empirical_distribution: ShortfallDistribution


def accounted_yield(
    decompositions: Sequence[Decomposition],
    seed: int,
    resamples: int = 2000,
    confidence: float = 0.95,
    bins: int = DEFAULT_BINS,
) -> AccountedYield:
    """Baseline 3 over a per-pass record. The resampling unit is the pass.

    Parameters
    ----------
    decompositions:
        One per pass, from :func:`passyield.decompose.decompose_pass`.
    seed:
        Seed for the bootstrap's ``numpy.random.default_rng``. Declared by the
        caller so the interval is reproducible.
    resamples:
        Bootstrap resamples.
    confidence:
        Interval coverage, e.g. 0.95.
    """
    if len(decompositions) < 2:
        raise ValueError(
            f"a pass-level bootstrap needs at least 2 passes, got {len(decompositions)}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")
    if resamples < 100:
        raise ValueError(f"resamples must be at least 100, got {resamples!r}")

    shortfalls = np.asarray([item.shortfall_bits for item in decompositions], dtype=float)
    realised = np.asarray([item.realised_bits for item in decompositions], dtype=float)
    capacity = np.asarray([item.capacity_bits for item in decompositions], dtype=float)
    nominal = np.asarray([item.nominal_bits for item in decompositions], dtype=float)
    terms = {
        "geometry": np.asarray([item.geometry_bits for item in decompositions], dtype=float),
        "acquisition": np.asarray([item.acquisition_bits for item in decompositions], dtype=float),
        "resync": np.asarray([item.resync_bits for item in decompositions], dtype=float),
        "idle": np.asarray([item.idle_bits for item in decompositions], dtype=float),
    }
    residual = np.asarray([item.residual_bits for item in decompositions], dtype=float)

    rng = np.random.default_rng(seed)
    count = shortfalls.size
    # The resampling unit is the PASS: one index draw per resample selects
    # whole passes with replacement, never seconds and never outages.
    indices = rng.integers(0, count, size=(resamples, count))
    mean_replicates = shortfalls[indices].mean(axis=1)
    fraction_replicates = realised[indices].sum(axis=1) / capacity[indices].sum(axis=1)
    tail = 0.5 * (1.0 - confidence)
    mean_ci = (
        float(np.quantile(mean_replicates, tail)),
        float(np.quantile(mean_replicates, 1.0 - tail)),
    )
    fraction_ci = (
        float(np.quantile(fraction_replicates, tail)),
        float(np.quantile(fraction_replicates, 1.0 - tail)),
    )

    total_shortfall = float(shortfalls.sum())
    shares = {name: float(values.sum() / total_shortfall) for name, values in terms.items()}
    low = float(shortfalls.min())
    high = float(shortfalls.max())
    pad = 0.05 * max(high - low, 1.0)
    grid = np.linspace(low - pad, high + pad, bins)
    empirical = distribution_from_samples(
        shortfalls, grid, method="empirical", resampling_unit=PASS_RESAMPLING_UNIT
    )

    return AccountedYield(
        resampling_unit=PASS_RESAMPLING_UNIT,
        passes=count,
        nominal_mbit=float(nominal.sum()),
        capacity_mbit=float(capacity.sum()),
        realised_mbit=float(realised.sum()),
        shortfall_mbit=total_shortfall,
        mean_shortfall_mbit=float(shortfalls.mean()),
        mean_shortfall_ci_mbit=mean_ci,
        realised_fraction=float(realised.sum() / capacity.sum()),
        realised_fraction_ci=fraction_ci,
        shortfall_share=shares,
        residual_share=float(residual.sum() / total_shortfall),
        resamples=resamples,
        confidence=confidence,
        empirical_distribution=empirical,
    )


def monte_carlo_shortfall_distribution(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    model: EventModel,
    nominal_bits: float,
    capacity_bits: float,
    seed: int,
    draws: int = 200000,
    bins: int = DEFAULT_BINS,
    profile_nodes: int = 513,
    decomposition_options: DecompositionOptions | None = None,
) -> ShortfallDistribution:
    """A direct Monte Carlo of the SAME laws baseline 2 convolves.

    This is the second method, not a second model: it draws the acquisition
    time, the outage count, each outage's duration and placement, and the idle
    total and placement from exactly the laws in ``model``, evaluates each
    loss through the same cumulative volume, and sums. It shares the structural
    independence approximation with the convolution on purpose -- its job is to
    check the convolution's **discretisation and bookkeeping**, which is why
    the product also ships the bracket in :mod:`passyield.bounds`, whose
    mathematics neither of these two uses.

    Compute: ``draws`` samples is a few vectorised passes over arrays and runs
    in well under a second for the default; the budget is documented in
    ``README.md``.
    """
    if draws < 1000:
        raise ValueError(f"draws must be at least 1000 for a usable quantile, got {draws!r}")
    offsets, cumulative = cumulative_volume(
        satrec, station, window, decomposition_options, profile_nodes
    )
    duration = float(window.duration_s)
    rng = np.random.default_rng(seed)

    acquisition_time = rng.gamma(model.acquisition_shape, model.acquisition_scale, size=draws)
    acquisition_loss = np.interp(
        np.clip(acquisition_time, 0.0, duration), offsets, cumulative
    )

    intensity = model.outage_intensity_per_s * duration
    counts = rng.poisson(intensity, size=draws)
    resync_loss = np.zeros(draws)
    maximum = int(counts.max()) if counts.size else 0
    for index in range(maximum):
        active = counts > index
        if not np.any(active):
            break
        lengths = np.minimum(rng.exponential(model.outage_mean_s, size=int(active.sum())), duration)
        starts = rng.uniform(0.0, 1.0, size=lengths.size) * np.maximum(duration - lengths, 0.0)
        resync_loss[active] += _loss_of_interval(
            offsets, cumulative, starts, starts + lengths
        )

    idle_total = np.minimum(rng.gamma(model.idle_shape, model.idle_scale, size=draws), duration)
    idle_start = rng.uniform(0.0, 1.0, size=draws) * np.maximum(duration - idle_total, 0.0)
    idle_loss = _loss_of_interval(offsets, cumulative, idle_start, idle_start + idle_total)

    geometry = nominal_bits - capacity_bits
    samples = geometry + acquisition_loss + resync_loss + idle_loss
    step = capacity_bits / bins
    grid = geometry + (np.arange(bins) + 0.5) * step
    return distribution_from_samples(samples, grid, method="monte-carlo")
