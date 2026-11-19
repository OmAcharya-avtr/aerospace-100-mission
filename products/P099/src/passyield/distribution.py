"""The product's headline output object: a distribution, not a point shortfall.

WHY THE OUTPUT IS A DISTRIBUTION
--------------------------------
A contact scheduler books a window and a rate, which is a point. An operations
report states what was delivered, which is another point. The thing a scheduler
would actually need in order to stop over-booking is neither: it is the
distribution of the shortfall, so that the schedule can be written against a
quantile of it. That is the whole positioning of this product, so the API
returns a distribution object with its quantiles and not a number with an error
bar bolted on.

REPRESENTATION
--------------
A probability mass function on a uniform grid of data volume in Mbit. Uniform
because the analytic baseline's convolution requires a uniform grid to be a
convolution at all, and the same representation then serves the Monte Carlo
cross-check and the bootstrap, so the three are comparable without conversion.

Three numbers travel with every distribution and all three are reported:

- ``grid_step_mbit`` -- the discretisation. A quantile read off a binned
  distribution carries an error of up to half a bin by construction, which is
  stated by :meth:`ShortfallDistribution.quantile_discretisation_mbit` rather
  than left for the reader to work out.
- ``truncated_mass`` -- probability that fell off the ends of the grid. A
  convolution on a finite grid loses mass at the top; pretending otherwise is
  how a quantile silently becomes a lower bound. Any method that builds one of
  these sets it, and :meth:`ShortfallDistribution.quantile` raises if the
  requested quantile lies inside the truncated tail.
- ``resampling_unit`` -- what was resampled, for a distribution that came from
  a bootstrap, and ``None`` for one that did not. The specification for this
  product fixes the bootstrap resampling unit as the **pass**, so the unit is
  carried on the object and asserted, not left in a docstring.

QUANTILE CONVENTION
-------------------
Linear interpolation of the cumulative mass across the bin containing the
requested level, which is the convention that makes the quantile of a
fine-grid binned distribution converge to the continuous quantile as the step
shrinks. ``validation/validate_convolution.py`` measures that convergence by
halving the step twice and reporting the observed order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Resampling unit this product's specification fixes for baseline 3.
PASS_RESAMPLING_UNIT = "pass"


@dataclass(frozen=True)
class ShortfallDistribution:
    """A probability mass function of shortfall on a uniform Mbit grid.

    Attributes
    ----------
    method:
        How it was built. One of ``"numerical-convolution"``,
        ``"monte-carlo"``, ``"bootstrap"``. Carried so that a report cannot
        present a Monte Carlo cross-check as the analytic result.
    grid_mbit:
        Bin centres, Mbit, strictly increasing and uniformly spaced.
    mass:
        Probability mass per bin. Non-negative. Sums with
        ``truncated_mass`` to 1 up to floating-point rounding.
    truncated_mass:
        Probability that fell outside the grid, dimensionless.
    resampling_unit:
        For a bootstrap distribution, the unit resampled. ``None`` otherwise.
    """

    method: str
    grid_mbit: np.ndarray
    mass: np.ndarray
    truncated_mass: float = 0.0
    resampling_unit: str | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid_mbit, dtype=float)
        mass = np.asarray(self.mass, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"grid_mbit must be a 1-D array of at least 2 bins, got {grid.shape}")
        if mass.shape != grid.shape:
            raise ValueError(
                f"mass must have the same shape as grid_mbit, got {mass.shape} and {grid.shape}"
            )
        if np.any(mass < 0.0):
            raise ValueError("mass must be non-negative in every bin")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("grid_mbit must be strictly increasing")
        steps = np.diff(grid)
        if not np.allclose(steps, steps[0], rtol=1.0e-9, atol=0.0):
            raise ValueError(
                "grid_mbit must be uniformly spaced: the convolution in "
                "passyield.baselines is only a convolution on a uniform grid"
            )
        if not 0.0 <= self.truncated_mass < 1.0:
            raise ValueError(f"truncated_mass must lie in [0, 1), got {self.truncated_mass!r}")
        total = float(mass.sum()) + self.truncated_mass
        if not np.isclose(total, 1.0, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError(
                f"mass plus truncated_mass must total 1, got {total!r}; a distribution that "
                "does not normalise cannot have quantiles"
            )
        object.__setattr__(self, "grid_mbit", grid)
        object.__setattr__(self, "mass", mass)

    @property
    def grid_step_mbit(self) -> float:
        """Uniform bin width, Mbit."""
        return float(self.grid_mbit[1] - self.grid_mbit[0])

    def quantile_discretisation_mbit(self) -> float:
        """Half a bin, Mbit: the discretisation error any quantile carries."""
        return 0.5 * self.grid_step_mbit

    def mean_mbit(self) -> float:
        """Expected shortfall over the retained mass, Mbit.

        Conditional on the grid: the truncated mass is excluded, so when
        ``truncated_mass`` is not negligible this is an under-estimate and the
        caller is told the figure through that attribute.
        """
        retained = float(self.mass.sum())
        if retained <= 0.0:
            raise ValueError("the distribution retains no mass on its grid")
        return float(np.dot(self.mass, self.grid_mbit) / retained)

    def standard_deviation_mbit(self) -> float:
        """Standard deviation over the retained mass, Mbit."""
        retained = float(self.mass.sum())
        centre = self.mean_mbit()
        variance = float(np.dot(self.mass, (self.grid_mbit - centre) ** 2) / retained)
        return float(np.sqrt(max(variance, 0.0)))

    def cdf(self, value_mbit: float) -> float:
        """``P(S <= value_mbit)`` over the retained mass, dimensionless."""
        return float(self.mass[self.grid_mbit <= value_mbit].sum())

    def quantile(self, level: float) -> float:
        """The ``level`` quantile of the shortfall, Mbit.

        Raises
        ------
        ValueError
            If ``level`` is outside ``(0, 1)``, or if it lies inside the
            truncated upper tail, where the grid holds no information and any
            returned number would be the top bin masquerading as a quantile.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie strictly inside (0, 1), got {level!r}")
        retained = float(self.mass.sum())
        if level > retained:
            raise ValueError(
                f"level {level!r} lies inside the truncated upper tail: the grid retains only "
                f"{retained!r} of the mass, so this quantile is not represented. Widen the grid."
            )
        cumulative = np.cumsum(self.mass)
        index = int(np.searchsorted(cumulative, level, side="left"))
        index = min(index, self.grid_mbit.size - 1)
        below = cumulative[index - 1] if index > 0 else 0.0
        within = self.mass[index]
        if within <= 0.0:
            return float(self.grid_mbit[index])
        fraction = (level - below) / within
        step = self.grid_step_mbit
        return float(self.grid_mbit[index] - 0.5 * step + fraction * step)

    def quantiles(self, levels: tuple[float, ...]) -> dict[float, float]:
        """Several quantiles at once, Mbit, keyed by level."""
        return {level: self.quantile(level) for level in levels}

    def interval(self, coverage: float) -> tuple[float, float]:
        """Equal-tailed interval of the given coverage, Mbit."""
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must lie strictly inside (0, 1), got {coverage!r}")
        tail = 0.5 * (1.0 - coverage)
        return self.quantile(tail), self.quantile(1.0 - tail)

    def samples_for_scoring(self, count: int = 256) -> np.ndarray:
        """An equally weighted ensemble drawn from the quantile function, Mbit.

        Used only so that this distribution can be scored by the same
        ensemble CRPS as the learned model's predictive distribution, on equal
        terms. Deterministic: the levels are the mid-points of ``count`` equal
        probability intervals, so no random number generator is involved and
        two calls return identical arrays.
        """
        if count < 2:
            raise ValueError(f"count must be at least 2, got {count!r}")
        levels = (np.arange(count) + 0.5) / count
        retained = float(self.mass.sum())
        usable = levels[levels < retained]
        if usable.size < 2:
            raise ValueError(
                "the retained mass is too small to draw a scoring ensemble from this grid"
            )
        return np.asarray([self.quantile(float(level)) for level in usable], dtype=float)


def distribution_from_samples(
    samples_mbit: np.ndarray,
    grid_mbit: np.ndarray,
    method: str,
    resampling_unit: str | None = None,
) -> ShortfallDistribution:
    """Bin samples onto a given grid, preserving mass and reporting the loss.

    Samples outside the grid are counted into ``truncated_mass`` rather than
    being clipped into the end bins, because clipping turns an unrepresented
    tail into a spike at the boundary and the spike then reads as a quantile.
    """
    grid = np.asarray(grid_mbit, dtype=float)
    values = np.asarray(samples_mbit, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"samples_mbit must be a non-empty 1-D array, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples_mbit must be finite")
    step = float(grid[1] - grid[0])
    edges = np.concatenate([grid - 0.5 * step, [grid[-1] + 0.5 * step]])
    counts, _ = np.histogram(values, bins=edges)
    inside = int(counts.sum())
    total = values.size
    mass = counts.astype(float) / total
    return ShortfallDistribution(
        method=method,
        grid_mbit=grid,
        mass=mass,
        truncated_mass=float(total - inside) / total,
        resampling_unit=resampling_unit,
    )


def uniform_grid(low_mbit: float, high_mbit: float, bins: int) -> np.ndarray:
    """Bin centres of a uniform grid spanning ``[low_mbit, high_mbit]``."""
    if not high_mbit > low_mbit:
        raise ValueError(f"high_mbit must exceed low_mbit, got {low_mbit!r} and {high_mbit!r}")
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins!r}")
    step = (high_mbit - low_mbit) / bins
    return low_mbit + (np.arange(bins) + 0.5) * step
