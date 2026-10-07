"""The robust invariant set of the baseline controller.

The certificate the whole package rests on is a set ``S`` with three properties:

    (i)   S is a subset of the declared state constraints X;
    (ii)  the baseline input is admissible everywhere in S, that is
          -K_b x in U for all x in S;
    (iii) S is robustly positively invariant under the baseline closed loop,
          A_b x + w in S for all x in S and all w in W, with A_b = A - B K_b.

Together these give the only safety statement this package makes: if ``x_0`` is
in ``S`` and the baseline controller is applied at every step, then ``x_k`` is in
``S`` and therefore in ``X`` for every ``k``, for every disturbance sequence
inside the **declared** bound. Nothing here proves anything about a disturbance
outside that bound; :mod:`simplexguard.boundviolation` measures what happens
then.

Algorithm
---------
``S`` is computed as the maximal robust invariant set inside
``Omega_0 = X and {x : -K_b x in U}`` by the standard one-step-set recursion
(Gilbert and Tan, "Linear systems with state and control constraints: the theory
and application of maximal output admissible sets", *IEEE Transactions on
Automatic Control* 36(9), 1991, pp. 1008-1020; Kolmanovsky and Gilbert, *Math.
Problems in Engineering* 4(4), 1998; Blanchini and Miani, *Set-Theoretic Methods
in Control*, Birkhauser, 2008; Borrelli, Bemporad and Morari, *Predictive
Control for Linear and Hybrid Systems*, CUP, 2017, chapter 10):

    Omega_{i+1} = Omega_i and Pre(Omega_i),                                 (1)
    Pre(Omega)  = {x : A_b x + w in Omega for all w in W}
                = A_b^{-1} (Omega (-) W),                                   (2)

where ``(-)`` is the Pontryagin difference of :meth:`Polytope.erode` and the
preimage is :meth:`Polytope.preimage`. Both are exact in halfspace form, so the
only approximations in the recursion are floating-point arithmetic and the
tolerances of the redundancy and convergence linear programmes.

The recursion is monotone, ``Omega_{i+1}`` is a subset of ``Omega_i``, and it
terminates when ``Omega_i`` is a subset of ``Pre(Omega_i)``, tested facet by
facet with a support-function linear programme. For an asymptotically stable
``A_b`` and a bounded ``W`` containing the origin, termination in finitely many
steps is the standard result; this implementation does **not** assume it and
reports a non-converged recursion as such rather than returning the last
iterate as if it were invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controllers import BaselineController, closed_loop_matrix
from .plant import Plant
from .polytope import Box, Polytope

__all__ = [
    "InvariantSetResult",
    "EmptyInvariantSet",
    "RecursionDidNotConverge",
    "robust_invariant_set",
    "verify_robust_invariance",
]


class EmptyInvariantSet(RuntimeError):
    """Raised when the recursion (1) empties the set.

    The baseline controller then has no certified envelope at all under the
    declared disturbance bound, and no runtime guard built on it means anything.
    The message names the iteration at which the set emptied.
    """


class RecursionDidNotConverge(RuntimeError):
    """Raised when (1) hits ``max_iterations`` without the inclusion test passing.

    The last iterate is *not* returned as a certificate, because it is not one.
    """


@dataclass(frozen=True)
class InvariantSetResult:
    """The computed set and everything needed to audit how it was computed."""

    polytope: Polytope
    iterations: int
    converged: bool
    halfspaces_per_iteration: tuple[int, ...] = field(default=())
    closed_loop: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    initial_set: Polytope | None = None

    @property
    def n_halfspaces(self) -> int:
        """Facet count of the reduced set."""
        return self.polytope.n_halfspaces

    def describe(self) -> str:
        """Report lines for the CLI and the validation scripts."""
        spectral = np.max(np.abs(np.linalg.eigvals(self.closed_loop)))
        lines = [
            f"converged                 {self.converged}",
            f"iterations                {self.iterations}",
            f"halfspaces (reduced)      {self.n_halfspaces}",
            f"halfspaces per iteration  {list(self.halfspaces_per_iteration)}",
            f"baseline spectral radius  {spectral:.9f}",
            f"Chebyshev radius of S     {self.polytope.chebyshev_radius():.9f}",
        ]
        if self.polytope.dim == 2:
            lines.append(f"area of S                 {self.polytope.area_2d():.9f}")
            if self.initial_set is not None:
                lines.append(
                    f"area of Omega_0           {self.initial_set.area_2d():.9f}"
                )
        return "\n".join(lines)


def _initial_set(plant: Plant, baseline: BaselineController) -> Polytope:
    """``Omega_0 = X and {x : -K_b x in U}``, the largest set properties (i) and
    (ii) can hold on."""
    u_set = plant.input_constraints
    if not isinstance(u_set, Box):
        raise TypeError("robust_invariant_set requires box input constraints")
    # -K_b x in U  <=>  lower <= -K_b x <= upper  <=>  [-K_b; K_b] x <= [upper; -lower]
    rows = np.vstack([-baseline.gain, baseline.gain])
    offs = np.concatenate([u_set.upper, -u_set.lower])
    return plant.state_constraints.intersect(Polytope(rows, offs))


def robust_invariant_set(
    plant: Plant,
    baseline: BaselineController,
    max_iterations: int = 200,
    tol: float = 1e-10,
    reduce_every_iteration: bool = True,
) -> InvariantSetResult:
    """Compute the maximal robust invariant set of the baseline closed loop.

    Parameters
    ----------
    plant :
        Supplies ``A``, ``B``, ``W``, ``X`` and ``U``.
    baseline :
        The controller the set certifies. Its gain must match the plant.
    max_iterations :
        Cap on (1). Exceeding it raises :class:`RecursionDidNotConverge`.
    tol :
        Tolerance for the facet inclusion test and for redundancy removal.
    reduce_every_iteration :
        Remove redundant facets at every iteration. This keeps the linear
        programmes small and is strictly faster for the reference plant; set it
        to ``False`` to see the unreduced facet growth, which
        ``validation/validate_invariant_set.py`` reports.

    Returns
    -------
    InvariantSetResult
        With ``converged=True``; the function raises rather than returning a
        non-certificate.

    Raises
    ------
    EmptyInvariantSet
        If the recursion empties the set. This is the honest outcome for a
        baseline that is too weak, or a declared disturbance that is too large,
        for the declared constraints.
    RecursionDidNotConverge
        If ``max_iterations`` is reached.
    """
    if int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    if not (np.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"tol must be finite and non-negative, got {tol}")
    if baseline.gain.shape != (plant.n_inputs, plant.n_states):
        raise ValueError(
            f"baseline gain has shape {baseline.gain.shape}, "
            f"expected {(plant.n_inputs, plant.n_states)}"
        )
    a_b = closed_loop_matrix(plant, baseline.gain)
    omega = _initial_set(plant, baseline)
    omega_0 = omega.minimal(tol) if reduce_every_iteration else omega
    omega = omega_0
    counts: list[int] = [omega.n_halfspaces]
    for iteration in range(1, int(max_iterations) + 1):
        pre = omega.erode(plant.disturbance).preimage(a_b)
        # Convergence test: Omega_i is a subset of Pre(Omega_i), facet by facet.
        converged = True
        for row, offset in zip(pre.A, pre.b, strict=True):
            s = omega.support(row)
            if s == -np.inf or s > offset + tol:
                converged = False
                break
        if converged:
            return InvariantSetResult(
                polytope=omega,
                iterations=iteration,
                converged=True,
                halfspaces_per_iteration=tuple(counts),
                closed_loop=a_b,
                initial_set=omega_0,
            )
        candidate = omega.intersect(pre)
        if candidate.is_empty(tol=1e-12):
            raise EmptyInvariantSet(
                f"the robust invariant set emptied at iteration {iteration}: the baseline "
                f"controller cannot hold the declared constraints X under the declared "
                f"disturbance bound W. Either reduce W, relax X, or give the baseline more "
                f"authority (a smaller LQR input weight)."
            )
        omega = candidate.minimal(tol) if reduce_every_iteration else candidate
        counts.append(omega.n_halfspaces)
    raise RecursionDidNotConverge(
        f"the one-step-set recursion did not converge in {max_iterations} iterations; "
        f"the last iterate has {omega.n_halfspaces} halfspaces and is NOT returned, "
        f"because it is not known to be invariant"
    )


def verify_robust_invariance(
    plant: Plant,
    baseline: BaselineController,
    invariant: Polytope,
    tol: float = 1e-9,
) -> dict[str, float | bool | int]:
    """Check properties (i), (ii) and (iii) independently of how ``S`` was built.

    Property (iii) is checked **exactly**, not by sampling: for every facet
    ``(c, e)`` of ``S``, the worst case of ``c^T (A_b x + w)`` over ``x in S``
    and ``w in W`` is ``h_S(A_b^T c) + h_W(c)``, and invariance holds if and only
    if that is at most ``e`` for every facet. Both terms are support functions,
    so the test is a linear programme per facet and a closed form on a box ``W``.

    Returns a dict of the three verdicts and the worst margin of each, where a
    margin is ``e - worst_case`` and must be non-negative.
    """
    a_b = closed_loop_matrix(plant, baseline.gain)
    inside_x = plant.state_constraints.contains_polytope(invariant, tol=tol)
    x_margin = min(
        float(offset - invariant.support(row))
        for row, offset in zip(plant.state_constraints.A, plant.state_constraints.b, strict=True)
    )
    u_set = plant.input_constraints
    u_rows = np.vstack([-baseline.gain, baseline.gain])
    u_offs = np.concatenate([u_set.upper, -u_set.lower])
    u_margin = min(
        float(offset - invariant.support(row))
        for row, offset in zip(u_rows, u_offs, strict=True)
    )
    w_support = plant.disturbance.support_many(invariant.A)
    inv_margin = np.inf
    for row, offset, w_term in zip(invariant.A, invariant.b, w_support, strict=True):
        worst = invariant.support(a_b.T @ row) + float(w_term)
        inv_margin = min(inv_margin, float(offset - worst))
    return {
        "subset_of_X": bool(inside_x),
        "X_margin": float(x_margin),
        "baseline_input_admissible": bool(u_margin >= -tol),
        "U_margin": float(u_margin),
        "robustly_invariant": bool(inv_margin >= -tol),
        "invariance_margin": float(inv_margin),
        "n_facets": int(invariant.n_halfspaces),
    }
