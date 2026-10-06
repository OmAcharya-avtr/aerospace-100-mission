"""The optimisation problem, written down.

Decision variables
------------------
For each MODCOD ``m`` in a table of ``M`` entries:

* ``x_m`` in [0, 1] -- the fraction of link time assigned to MODCOD ``m``;
* ``y_m`` in {0, 1} -- whether MODCOD ``m`` appears in the operating table.

Data
----
* ``R_m`` -- net information rate of MODCOD ``m``, bits/symbol.
* ``A_m = P[margin + fade >= threshold_m]`` -- the probability that a frame
  sent on MODCOD ``m`` is delivered, from the fade model.
* ``A_min`` -- the availability target, dimensionless, in (0, 1).
* ``K`` -- the maximum number of table entries the modem may switch between.

Objective
---------
Maximise expected goodput, ``sum_m R_m A_m x_m``, bits/symbol. A frame sent
below threshold contributes nothing: this is the *outage* approximation to a
coded link, and it is an approximation -- see Assumptions below.

Constraints
-----------
1. ``sum_m x_m = 1`` -- all link time is assigned.
2. ``x_m <= y_m`` -- time only goes to selected entries.
3. ``sum_m y_m <= K`` -- table cardinality.
3b. ``x_m >= d y_m`` -- minimum dwell fraction, optional (``d = 0`` by
    default). An entry that appears in the table must be used for at least
    ``d`` of the time. Nobody puts a MODCOD in an operating table for 0.3 %
    of the link time, and this is the constraint that says so. It is also
    the constraint that makes the integer variables load-bearing: with
    ``d = 0`` the optimum has a provable closed form (see below), and with
    ``d > 0`` it does not.
4. The availability constraint, in one of **two** forms, which do not mean the
   same thing and do not give the same answer:

   * ``"long_run"``: ``sum_m A_m x_m >= A_min``. The *time-averaged* delivery
     probability meets the target. A low-rate, high-availability entry may be
     mixed with a high-rate, low-availability one; during the second entry's
     share of the time, availability is below target.
   * ``"per_interval"``: ``y_m = 0`` for every ``m`` with ``A_m < A_min``.
     *Every* interval meets the target, because every entry in the table does.

   At ``K = 1`` the two forms coincide, because a single entry's long-run
   availability is its own ``A_m``. They diverge only when mixing is allowed,
   and then ``"long_run"`` is weakly better in goodput and weakly worse in
   worst-interval availability. Conflating them is the usual error, so the
   mode has no default that hides the choice: ``"per_interval"`` is the
   default because it is the conservative reading, and the divergence is
   measured in ``validation/validate_mode_divergence.py``.

Assumptions, all of which bound what the answer means
-----------------------------------------------------
* **Outage goodput.** A frame is delivered with probability 1 above threshold
  and 0 below. Real codes have a waterfall a fraction of a dB wide; inside it
  this model is wrong in the optimistic direction just above threshold and in
  the pessimistic direction just below. The threshold is whatever operating
  point you define it as (commonly the quasi-error-free point).
* **Marginal fade statistics only.** ``A_m`` is a long-run probability from a
  marginal distribution. Fades are correlated in time, so the *number* of
  outage events and their *duration* are not determined by ``A_m`` alone, and
  this package says nothing about them.
* **Free switching.** Changing MODCOD is assumed to cost no time and to need
  no channel-state feedback. A real adaptive link pays acquisition time and
  acts on a channel estimate that is one round trip stale.
* **Ergodic time-sharing.** ``x_m`` is a long-run time fraction. The model
  does not schedule; it says what mix is optimal, not when to switch.

Structural results, which are the reason this package is honest about needing
an optimiser at all
-------------------------------------------------------------------------------
* In ``"per_interval"`` mode the optimum is always a **single** MODCOD, namely
  ``argmax{R_m A_m : A_m >= A_min}``. Mixing in any second entry moves time to
  a strictly lower ``R_m A_m``, which a linear objective never rewards. No
  optimiser is needed; ``numpy.argmax`` is.
* In ``"long_run"`` mode with ``d = 0`` the optimum needs at most **two**
  MODCODs whatever ``K >= 2`` is, because the linear programme has two active
  constraints (``sum x = 1`` and the availability row) and therefore a basic
  optimal solution with at most two non-zeros. Enumerating ``O(M**2)`` pairs
  in closed form solves it exactly, which is what
  :func:`coderateopt.exhaustive.solve_closed_form_k2` does with no solver call.
* With ``d > 0`` neither result holds: the indicator implication
  ``x_m > 0 => x_m >= d`` is not expressible as a linear programme, supports
  larger than two can be optimal, and a mixed-integer solver is doing real
  work. ``validation/validate_support_size.py`` measures how often that
  happens.

The practical consequence is stated in the README rather than buried: for the
base formulation you do not need this package's solver, and the README says
which single line of NumPy to write instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .fade import AvailabilityModel
from .modcod import ModcodSet

__all__ = ["AvailabilityMode", "InfeasibleProblem", "RateProblem", "Solution"]

AvailabilityMode = Literal["per_interval", "long_run"]


class InfeasibleProblem(ValueError):
    """No assignment satisfies the availability constraint.

    Carries the diagnostic numbers needed to act: the best availability any
    entry or mix achieves, which entry achieves it, and the shortfall.
    """

    def __init__(
        self,
        message: str,
        *,
        best_availability: float,
        best_modcod: str,
        target: float,
        mode: str,
    ) -> None:
        super().__init__(message)
        self.best_availability = best_availability
        self.best_modcod = best_modcod
        self.target = target
        self.mode = mode


@dataclass(frozen=True)
class RateProblem:
    """A fully specified rate-selection instance.

    Parameters
    ----------
    modcods
        The MODCOD table, in :class:`~coderateopt.modcod.ModcodSet` canonical
        order.
    fade
        Marginal fade model supplying ``A_m``.
    margin_db
        Nominal (clear-sky) link margin on the same dB scale as the MODCOD
        thresholds.
    availability_target
        ``A_min``, in (0, 1).
    max_entries
        ``K``, the table cardinality limit, >= 1. ``K = 1`` is fixed-rate
        selection.
    mode
        ``"per_interval"`` (default) or ``"long_run"``; see the module
        docstring, which states why these differ.
    min_dwell_fraction
        ``d``, the smallest time fraction an entry may receive if it is used
        at all, in [0, 1). Default 0.
    """

    modcods: ModcodSet
    fade: AvailabilityModel
    margin_db: float
    availability_target: float
    max_entries: int = 1
    mode: AvailabilityMode = "per_interval"
    min_dwell_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.modcods, ModcodSet):
            raise TypeError("modcods must be a ModcodSet")
        if not isinstance(self.fade, AvailabilityModel):
            raise TypeError("fade must be an AvailabilityModel")
        if not np.isfinite(self.margin_db):
            raise ValueError(f"margin_db must be finite, got {self.margin_db!r}")
        if not 0.0 < self.availability_target < 1.0:
            raise ValueError(
                f"availability_target must be in (0, 1), got {self.availability_target!r}"
            )
        if not isinstance(self.max_entries, int) or isinstance(self.max_entries, bool):
            raise TypeError(f"max_entries must be an int, got {type(self.max_entries).__name__}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.mode not in ("per_interval", "long_run"):
            raise ValueError(
                f"mode must be 'per_interval' or 'long_run', got {self.mode!r}"
            )
        d = self.min_dwell_fraction
        if not np.isfinite(d) or not 0.0 <= d < 1.0:
            raise ValueError(f"min_dwell_fraction must be in [0, 1), got {d!r}")

    @property
    def n_modcods(self) -> int:
        return len(self.modcods)

    @property
    def effective_k(self) -> int:
        """The cardinality limit that actually binds.

        ``min(K, M)``, further capped by ``floor(1 / d)`` when a minimum dwell
        fraction is set, because more than ``1 / d`` entries cannot share unit
        time while each holding at least ``d`` of it.
        """
        k = min(self.max_entries, self.n_modcods)
        if self.min_dwell_fraction > 0.0:
            k = min(k, int(1.0 // self.min_dwell_fraction))
        return max(k, 1)

    def availabilities(self) -> np.ndarray:
        """``A_m`` in canonical MODCOD order, dimensionless.

        Memoised on first use. Enumeration asks for this once per subset and a
        gamma-gamma evaluation costs a quadrature per MODCOD, so recomputing
        it dominated the solve time before the cache was added. The returned
        array is read-only, because handing out a mutable cache is how a
        memoised value silently becomes wrong.
        """
        cached = getattr(self, "_availabilities_cache", None)
        if cached is None:
            cached = np.asarray(
                self.fade.availability(self.margin_db, self.modcods.thresholds_db), dtype=float
            ).reshape(-1)
            cached.setflags(write=False)
            object.__setattr__(self, "_availabilities_cache", cached)
        return cached

    def goodputs(self) -> np.ndarray:
        """``R_m A_m`` in canonical order, bits/symbol -- single-entry goodput."""
        return self.modcods.rates * self.availabilities()

    def allowed_mask(self) -> np.ndarray:
        """Entries the mode permits in the table, boolean, canonical order."""
        if self.mode == "per_interval":
            return self.availabilities() >= self.availability_target
        return np.ones(self.n_modcods, dtype=bool)

    def with_fade(self, fade: AvailabilityModel) -> RateProblem:
        """Copy with a different fade model -- used by the sensitivity sweep."""
        return RateProblem(
            modcods=self.modcods,
            fade=fade,
            margin_db=self.margin_db,
            availability_target=self.availability_target,
            max_entries=self.max_entries,
            mode=self.mode,
            min_dwell_fraction=self.min_dwell_fraction,
        )

    def with_margin(self, margin_db: float) -> RateProblem:
        """Copy with a different nominal margin, dB."""
        return RateProblem(
            modcods=self.modcods,
            fade=self.fade,
            margin_db=margin_db,
            availability_target=self.availability_target,
            max_entries=self.max_entries,
            mode=self.mode,
            min_dwell_fraction=self.min_dwell_fraction,
        )

    def with_target(self, availability_target: float) -> RateProblem:
        """Copy with a different availability target."""
        return RateProblem(
            modcods=self.modcods,
            fade=self.fade,
            margin_db=self.margin_db,
            availability_target=availability_target,
            max_entries=self.max_entries,
            mode=self.mode,
            min_dwell_fraction=self.min_dwell_fraction,
        )


@dataclass(frozen=True)
class Solution:
    """An optimal assignment and everything needed to judge it.

    Attributes
    ----------
    time_fractions
        ``x_m`` in canonical MODCOD order, dimensionless, summing to 1.
    support
        Indices with ``x_m > 0``, ascending.
    expected_goodput
        ``sum_m R_m A_m x_m``, bits/symbol.
    achieved_availability
        ``sum_m A_m x_m``, the long-run delivery probability of the mix.
    worst_interval_availability
        ``min_{m in support} A_m`` -- the availability during the worst part of
        the mix. Equals ``achieved_availability`` only for a single entry.
    method
        ``"milp"``, ``"exhaustive"`` or ``"closed_form_k2"``.
    canonical
        True when the support was canonicalised by enumeration, so that ties
        resolve reproducibly; False when the instance was too large for that
        and the solver's own choice among tied optima stands.
    tied_supports
        Other supports attaining the same goodput within ``tolerance``, as
        sorted index tuples. Empty when the optimum is unique. **Populated
        only when ``canonical`` is True**: tie detection requires enumeration,
        so ``is_unique()`` carries no information when ``canonical`` is
        False.
    tolerance
        Absolute goodput tolerance used for optimality and tie comparisons.
    """

    time_fractions: np.ndarray
    support: tuple[int, ...]
    expected_goodput: float
    achieved_availability: float
    worst_interval_availability: float
    method: str
    canonical: bool
    tied_supports: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    tolerance: float = 1e-9

    def support_names(self, modcods: ModcodSet) -> tuple[str, ...]:
        return tuple(modcods.names[i] for i in self.support)

    def is_unique(self) -> bool:
        """True when no other support attains the optimum within tolerance."""
        return len(self.tied_supports) == 0
