"""Exact multi-step prediction of the switching condition: the analytic baseline.

The one-step switching condition of :mod:`simplexguard.guard` tells you whether
the guard fires *now*. Anticipating it is a different question, and the learned
predictor in :mod:`simplexguard.predictor` is benchmarked against the exact
answer to it, computed here.

Setting
-------
Hold the reference ``r`` constant over the horizon and let the performance
controller run **unsaturated**, so the loop is linear:

    u_j = -K_p (x_j - r),     x_{j+1} = A_p x_j + v + w_j,                  (1)

with ``A_p = A - B K_p`` and ``v = B K_p r``. Unrolling (1),

    x_j = A_p^j x_0 + T_j v + sum_{i<j} A_p^{j-1-i} w_i,   T_j = sum_{i<j} A_p^i. (2)

The guard fires at step ``j`` when some facet ``c_m`` of ``S`` fails inequality
(2) of :mod:`simplexguard.guard` at ``(x_j, u_j)``. Since
``A x_j + B u_j = A_p x_j + v``, that is

    c_m^T (A_p x_j + v) + h_W(c_m) > d_m.                                   (3)

Substituting (2) into (3) and maximising each disturbance term independently
over ``W`` -- which is exact, because each term is linear in its own ``w_i`` and
the ``w_i`` are unconstrained relative to one another -- gives the condition
"there exists an admissible disturbance sequence making the guard fire at step
``j``":

    c_m^T A_p^{j+1} x_0 + c_m^T (I + T_j) v + sum_{i=1}^{j} h_W((A_p^i)^T c_m)
        > d_m - h_W(c_m).                                                   (4)

Dropping the ``sum h_W`` term gives the **nominal** variant: "the guard fires at
step ``j`` if no disturbance occurs". Both are exact statements about the linear
model (1); neither involves sampling.

Inequality (4) is precomputed into one matrix triple per horizon step, so a
prediction over a horizon of ``L`` costs ``L + 1`` small matrix products.

What this is *not*
------------------
Three stated approximations, each of which is measured rather than assumed
harmless by ``validation/validate_predictor.py``:

* **Saturation is ignored.** The real performance controller saturates at the
  declared input box, and (1) does not. The validation script reports the
  measured fraction of steps at which the performance controller saturates, so
  the size of this gap is a number and not an adjective.
* **The guard's own intervention inside the horizon is ignored.** Once the guard
  fires, the realised trajectory follows the baseline, not (1). The predictions
  here are therefore about the hypothetical unguarded continuation.
* **The worst-case variant answers "may fire", not "will fire".** On a realised
  episode it is a sound over-approximation of the realised firing only up to the
  two points above; it is expected to over-predict, and the measured precision
  says by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import PerformanceController
from .guard import SimplexGuard

__all__ = ["ExactLeadPredictor"]


@dataclass(frozen=True)
class _HorizonTerms:
    state_rows: np.ndarray  # C A_p^{j+1}
    reference_rows: np.ndarray  # C (I + T_j) B K_p
    offsets: np.ndarray  # d - h_W(C) - sum_i h_W((A_p^i)^T C)


class ExactLeadPredictor:
    """Exact ``L``-step prediction of the switching condition by inequality (4).

    Parameters
    ----------
    guard :
        Supplies ``S``, the declared ``W`` and the plant.
    performance :
        Supplies ``K_p``. Only its linear part is used; see the module docstring.
    horizon :
        ``L >= 0``. Horizon 0 reproduces the one-step guard condition on the
        unsaturated input exactly.
    mode :
        ``"worst_case"`` for inequality (4), ``"nominal"`` for the same without
        the disturbance sum.
    """

    def __init__(
        self,
        guard: SimplexGuard,
        performance: PerformanceController,
        horizon: int,
        mode: str = "worst_case",
    ) -> None:
        if int(horizon) < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        if mode not in ("worst_case", "nominal"):
            raise ValueError(f"unknown mode {mode!r}; expected worst_case or nominal")
        plant = guard.plant
        s = guard.invariant_set
        c, d = s.A, s.b
        a_p = plant.A - plant.B @ performance.gain
        bk = plant.B @ performance.gain
        base_offsets = d - guard.disturbance_offsets
        n = plant.n_states
        terms: list[_HorizonTerms] = []
        power = a_p.copy()  # A_p^{j+1} with j = 0
        t_sum = np.zeros((n, n))  # T_j = sum_{i<j} A_p^i, so T_0 = 0
        w_sum = np.zeros(c.shape[0])
        for j in range(int(horizon) + 1):
            if j >= 1:
                # add h_W((A_p^j)^T c_m) for every facet m
                w_sum = w_sum + plant.disturbance.support_many(c @ np.linalg.matrix_power(a_p, j))
            offsets = base_offsets - (w_sum if mode == "worst_case" else 0.0)
            terms.append(
                _HorizonTerms(
                    state_rows=c @ power,
                    reference_rows=c @ (np.eye(n) + t_sum) @ bk,
                    offsets=offsets,
                )
            )
            t_sum = t_sum + np.linalg.matrix_power(a_p, j)
            power = power @ a_p
        self.horizon = int(horizon)
        self.mode = mode
        self._terms = tuple(terms)
        self._n_states = n

    def fires_at(self, x: np.ndarray, reference: np.ndarray, step: int) -> bool:
        """Inequality (4) at a single horizon step ``j = step``."""
        if not 0 <= int(step) <= self.horizon:
            raise ValueError(f"step must be in [0, {self.horizon}], got {step}")
        t = self._terms[int(step)]
        xv = np.asarray(x, dtype=float).ravel()
        rv = np.asarray(reference, dtype=float).ravel()
        if xv.shape[0] != self._n_states or rv.shape[0] != self._n_states:
            raise ValueError(f"x and reference must have length {self._n_states}")
        return bool(np.any(t.state_rows @ xv + t.reference_rows @ rv > t.offsets))

    def first_step(self, x: np.ndarray, reference: np.ndarray) -> int:
        """Earliest horizon step at which (4) holds, or ``-1`` if none does."""
        for j in range(self.horizon + 1):
            if self.fires_at(x, reference, j):
                return j
        return -1

    def predict(self, x: np.ndarray, reference: np.ndarray) -> bool:
        """True if (4) holds at any horizon step in ``[0, L]``."""
        return self.first_step(x, reference) >= 0

    def predict_many(self, states: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`predict` over rows of ``states`` and ``references``."""
        xs = np.atleast_2d(np.asarray(states, dtype=float))
        rs = np.atleast_2d(np.asarray(references, dtype=float))
        if xs.shape[0] != rs.shape[0]:
            raise ValueError(f"{xs.shape[0]} states but {rs.shape[0]} references")
        if xs.shape[1] != self._n_states or rs.shape[1] != self._n_states:
            raise ValueError(f"states and references must have width {self._n_states}")
        out = np.zeros(xs.shape[0], dtype=bool)
        for t in self._terms:
            resid = xs @ t.state_rows.T + rs @ t.reference_rows.T - t.offsets
            out |= np.any(resid > 0.0, axis=1)
        return out

    def first_step_many(self, states: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`first_step`; ``-1`` where (4) never holds."""
        xs = np.atleast_2d(np.asarray(states, dtype=float))
        rs = np.atleast_2d(np.asarray(references, dtype=float))
        out = np.full(xs.shape[0], -1, dtype=int)
        for j, t in enumerate(self._terms):
            resid = xs @ t.state_rows.T + rs @ t.reference_rows.T - t.offsets
            hit = np.any(resid > 0.0, axis=1) & (out < 0)
            out[hit] = j
        return out

    def score_many(self, states: np.ndarray, references: np.ndarray) -> np.ndarray:
        """A monotone score in ``[0, 1]`` for ROC comparison, not a probability.

        Defined as ``1 - first_step / (L + 1)`` when (4) holds somewhere in the
        horizon and ``0`` when it does not, so an earlier predicted firing scores
        higher. It is deliberately **not** called a probability: this predictor
        is a worst-case set computation and has no calibrated output. That
        asymmetry is the one thing the learned predictor has that it does not.
        """
        first = self.first_step_many(states, references)
        score = np.where(first >= 0, 1.0 - first / (self.horizon + 1.0), 0.0)
        return score.astype(float)
