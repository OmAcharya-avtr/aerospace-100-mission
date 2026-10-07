"""The exact switching condition.

Statement
---------
Let ``S = {x : C x <= d}`` be the robust invariant set of the baseline closed
loop computed by :mod:`simplexguard.invariant`, let ``W`` be the declared
disturbance set and let ``u`` be the input the performance controller proposes
at the current state ``x``. The performance input is admitted if and only if

    for all w in W:   A x + B u + w  in  S,                                 (1)

which, because the maximum of a linear function over a polytope is its support
function, is **exactly** equivalent to the finite system of inequalities

    c_j^T (A x + B u)  +  h_W(c_j)  <=  d_j      for every facet j of S,     (2)

with ``h_W`` the support function of equation (1) of
:mod:`simplexguard.polytope`. Equivalently, ``A x + B u`` must lie in the eroded
set ``S (-) W``. Condition (2) is a matrix product and a comparison; it involves
no sampling of ``W``, no gridding, no margin tuning and no heuristic. For a box
``W`` the offsets ``h_W(c_j)`` are ``|c_j|^T r``, computed once when the guard is
constructed.

If (2) fails for any facet, control is handed to the baseline, whose input is
admissible and keeps the state in ``S`` for every ``w`` in ``W`` by the
invariance property. The resulting recursive argument is the Simplex
architecture's (Seto et al., *Proc. ACC*, 1998; Sha, *IEEE Software* 18(4),
2001; Schierman et al., "Runtime assurance for autonomous aerospace systems",
*Journal of Guidance, Control, and Dynamics*, 2020):

    x_0 in S  and condition (2) enforced at every step  =>  x_k in S for all k,

**provided every realised disturbance lies in the declared W.** That proviso is
the entire safety argument and it is an assumption, not a result.
:mod:`simplexguard.boundviolation` measures what the accounting looks like when
it is false.

Two exact additional guards
---------------------------
* The proposed input must itself be admissible, ``u in U``. A saturating
  performance controller satisfies this by construction; a learned one supplied
  by the caller may not, so it is checked.
* The current state must lie in ``S``. If it does not, the certificate is
  already lost: handing control to the baseline is then the best available
  action but is **not** covered by any guarantee, and the decision record says
  so with :attr:`GuardDecision.certificate_lost`.

Hysteresis
----------
:class:`SimplexGuard` optionally enforces a minimum baseline dwell. Extending a
baseline interval is always safe, because ``S`` is invariant under the baseline
for every admissible disturbance, so holding the baseline longer than the
condition demands cannot leave ``S``. Extending a *performance* interval is not
safe and is therefore not offered: there is no minimum-performance-dwell
parameter in this package, by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .controllers import BaselineController
from .plant import Plant
from .polytope import Box, Polytope

__all__ = ["GuardDecision", "Mode", "SimplexGuard"]


class Mode(Enum):
    """Which controller holds authority at a step."""

    PERFORMANCE = 0
    BASELINE = 1

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class GuardDecision:
    """Everything the guard computed at one step, for the accounting layer.

    Attributes
    ----------
    mode :
        Which controller's input was applied.
    applied_input :
        The input actually applied, in input units.
    proposed_input :
        What the performance controller proposed, in input units.
    baseline_input :
        What the baseline would have applied, in input units.
    margin :
        ``min_j (d_j - c_j^T (A x + B u_perf) - h_W(c_j))`` of inequality (2),
        in the units of the facet offsets. Non-negative exactly when the
        performance input is admitted on the invariance test. Reported whether
        or not it was admitted, so the distance to the switch is measurable.
    binding_facet :
        Index of the facet attaining ``margin``.
    input_admissible :
        Whether ``u_perf in U``.
    certificate_lost :
        Whether the current state is outside ``S``. When true, no guarantee
        applies to this step regardless of the mode.
    state_margin :
        ``min_j (d_j - c_j^T x)``: how far inside ``S`` the current state is.
    forced_by_dwell :
        True when the baseline was held because of the minimum-dwell setting
        rather than because inequality (2) failed.
    """

    mode: Mode
    applied_input: np.ndarray
    proposed_input: np.ndarray
    baseline_input: np.ndarray
    margin: float
    binding_facet: int
    input_admissible: bool
    certificate_lost: bool
    state_margin: float
    forced_by_dwell: bool = False

    @property
    def condition_holds(self) -> bool:
        """True when inequality (2) holds and ``u_perf in U``."""
        return bool(self.margin >= 0.0 and self.input_admissible)


class SimplexGuard:
    """The runtime guard: inequality (2), evaluated exactly, plus bookkeeping.

    Parameters
    ----------
    plant :
        Supplies ``A``, ``B``, the declared ``W`` and the declared ``U``.
    baseline :
        The fallback controller that ``invariant_set`` certifies.
    invariant_set :
        ``S``, as returned by
        :func:`simplexguard.invariant.robust_invariant_set`. It is the caller's
        responsibility that this really is invariant for ``baseline``;
        :func:`simplexguard.invariant.verify_robust_invariance` checks it
        independently and the constructor calls that check when
        ``verify=True``.
    min_baseline_dwell :
        Minimum number of consecutive steps to hold the baseline once it takes
        over, ``>= 1``. 1 means no hysteresis. See the module docstring for why
        only the baseline side is offered.
    verify :
        Run :func:`verify_robust_invariance` at construction and raise if ``S``
        fails any of the three properties. Costs one linear programme per facet.
    """

    def __init__(
        self,
        plant: Plant,
        baseline: BaselineController,
        invariant_set: Polytope,
        min_baseline_dwell: int = 1,
        verify: bool = False,
    ) -> None:
        if invariant_set.dim != plant.n_states:
            raise ValueError(
                f"invariant set has dim {invariant_set.dim}, plant has {plant.n_states} states"
            )
        if int(min_baseline_dwell) < 1:
            raise ValueError(f"min_baseline_dwell must be at least 1, got {min_baseline_dwell}")
        if not isinstance(plant.input_constraints, Box):
            raise TypeError("SimplexGuard requires box input constraints")
        self.plant = plant
        self.baseline = baseline
        self.invariant_set = invariant_set
        self.min_baseline_dwell = int(min_baseline_dwell)
        # Precompute the eroded set S (-) W of inequality (2). One matrix
        # product for a box W; one LP per facet otherwise, done once.
        self._eroded = invariant_set.erode(plant.disturbance)
        self._disturbance_offsets = invariant_set.b - self._eroded.b
        if verify:
            from .invariant import verify_robust_invariance

            report = verify_robust_invariance(plant, baseline, invariant_set)
            if not (
                report["subset_of_X"]
                and report["baseline_input_admissible"]
                and report["robustly_invariant"]
            ):
                raise ValueError(f"the supplied set is not a valid certificate: {report}")
        self._baseline_steps_remaining = 0

    # ----------------------------------------------------------- the test
    @property
    def eroded_set(self) -> Polytope:
        """``S (-) W``: the set the one-step-ahead nominal state must lie in."""
        return self._eroded

    @property
    def disturbance_offsets(self) -> np.ndarray:
        """``h_W(c_j)`` for every facet ``j`` of ``S``, in facet-offset units."""
        return self._disturbance_offsets.copy()

    def condition_margin(self, x: np.ndarray, u: np.ndarray) -> tuple[float, int]:
        """Margin and binding facet of inequality (2) at ``(x, u)``.

        Returns ``(margin, index)`` where ``margin >= 0`` iff (2) holds. No
        sampling of ``W`` is involved.
        """
        nxt = self.plant.step(x, u)
        residual = self._eroded.A @ nxt - self._eroded.b
        idx = int(np.argmax(residual))
        return float(-residual[idx]), idx

    def allows(self, x: np.ndarray, u: np.ndarray) -> bool:
        """Inequality (2) and ``u in U``, as a single boolean."""
        if not self.plant.input_constraints.contains(u, tol=1e-12):
            return False
        return self.condition_margin(x, u)[0] >= 0.0

    def brute_force_allows(self, x: np.ndarray, u: np.ndarray) -> bool:
        """The same test by enumerating the vertices of a box ``W``.

        Only valid for a box disturbance set, where ``2**n`` vertices suffice
        because the maximum of a linear function over a polytope is attained at
        a vertex. Exists so that ``validation/validate_guard_exactness.py`` can
        confirm the support-function form against an independent computation
        rather than asserting it.
        """
        w_set = self.plant.disturbance
        if not isinstance(w_set, Box):
            raise TypeError("brute_force_allows requires a box disturbance set")
        if not self.plant.input_constraints.contains(u, tol=1e-12):
            return False
        nxt = self.plant.step(x, u)
        for vertex in w_set.vertices():
            if not self.invariant_set.contains(nxt + vertex, tol=0.0):
                return False
        return True

    # ------------------------------------------------------ one decision
    def decide(
        self,
        x: np.ndarray,
        proposed_input: np.ndarray,
        advance_dwell: bool = True,
    ) -> GuardDecision:
        """Apply the switching condition at one step.

        Parameters
        ----------
        x :
            Current state, in state units.
        proposed_input :
            What the performance controller wants to apply, in input units.
        advance_dwell :
            Update the internal minimum-dwell counter. Set ``False`` to probe a
            decision without changing the guard's state.
        """
        xv = np.asarray(x, dtype=float).ravel()
        if xv.shape[0] != self.plant.n_states:
            raise ValueError(f"x has length {xv.shape[0]}, expected {self.plant.n_states}")
        u_perf = np.atleast_1d(np.asarray(proposed_input, dtype=float)).ravel()
        if u_perf.shape[0] != self.plant.n_inputs:
            raise ValueError(
                f"proposed_input has length {u_perf.shape[0]}, expected {self.plant.n_inputs}"
            )
        if not np.all(np.isfinite(u_perf)):
            raise ValueError("proposed_input contains non-finite entries")
        u_base = self.baseline(xv)
        state_margin = self.invariant_set.slack(xv)
        certificate_lost = state_margin < 0.0
        input_ok = bool(self.plant.input_constraints.contains(u_perf, tol=1e-12))
        margin, facet = self.condition_margin(xv, u_perf)
        dwell_blocks = self._baseline_steps_remaining > 0
        condition_ok = (margin >= 0.0) and input_ok and not certificate_lost
        use_performance = condition_ok and not dwell_blocks
        if use_performance:
            mode = Mode.PERFORMANCE
            applied = u_perf
            forced = False
        else:
            mode = Mode.BASELINE
            applied = u_base
            forced = bool(dwell_blocks and condition_ok)
        if advance_dwell:
            if mode is Mode.BASELINE and not dwell_blocks:
                self._baseline_steps_remaining = self.min_baseline_dwell - 1
            elif dwell_blocks:
                self._baseline_steps_remaining -= 1
        return GuardDecision(
            mode=mode,
            applied_input=applied,
            proposed_input=u_perf,
            baseline_input=u_base,
            margin=margin,
            binding_facet=facet,
            input_admissible=input_ok,
            certificate_lost=certificate_lost,
            state_margin=state_margin,
            forced_by_dwell=forced,
        )

    def reset(self) -> None:
        """Clear the minimum-dwell counter, for the start of an episode."""
        self._baseline_steps_remaining = 0

    def describe(self) -> str:
        """Report lines for the CLI."""
        offs = self._disturbance_offsets
        return "\n".join(
            [
                f"facets of S               {self.invariant_set.n_halfspaces}",
                f"h_W over facets: min      {offs.min():.9e}",
                f"h_W over facets: max      {offs.max():.9e}",
                f"minimum baseline dwell    {self.min_baseline_dwell} steps",
                f"Chebyshev radius of S     {self.invariant_set.chebyshev_radius():.9f}",
                f"Chebyshev radius of S-W   {self._eroded.chebyshev_radius():.9f}",
            ]
        )
