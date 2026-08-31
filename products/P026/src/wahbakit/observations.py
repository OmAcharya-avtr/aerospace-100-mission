"""Vector observation set, weights, and the observability (degeneracy) test.

Measurement model
-----------------
Shuster's model for a unit vector sensor (Shuster 1978; Shuster & Oh 1981) is

    b_i = A r_i + n_i,    E[n_i] = 0,    E[n_i n_i^T] = sigma_i^2 (I - (A r_i)(A r_i)^T)
                                                                        (Eq. O1)

with the noise confined to the plane orthogonal to the true direction, because a
unit vector carries only two degrees of freedom.  ``sigma_i`` is the per-axis
transverse angular error in **radians** and is valid for ``sigma_i << 1 rad``;
star trackers are typically 1e-5 to 1e-4 rad, sun sensors 1e-3 to 1e-2 rad,
magnetometers 1e-2 to 1e-1 rad.

Wahba's problem
---------------
Find the proper orthogonal ``A`` minimising (Wahba 1965, *SIAM Review* 7(3), 409)

    L(A) = (1/2) sum_i w_i |b_i - A r_i|^2                              (Eq. O2)

The maximum-likelihood weights under Eq. O1 are ``w_i proportional to
1 / sigma_i^2`` (Shuster & Oh 1981).  Weights are stored normalised to sum to
one, which is the normalisation Shuster assumes when he writes the optimal
Wahba gain as ``lambda_max <= 1``.

Observability
-------------
The Fisher information of the attitude given Eq. O1 is

    F = sum_i sigma_i^-2 (I - b_i b_i^T)     [rad^-2]                   (Eq. O3)

and the attitude is determined only where ``F`` is non-singular.  With every
``b_i`` parallel, ``F`` has rank 2 and the rotation about the common direction
is unobservable: the loss Eq. O2 is flat along that axis and every solver
returns an arbitrary point on the flat.

Degeneracy is a property of the *directions*, so the gate is evaluated on the
unweighted, sigma-free form of that matrix:

    F_hat = (1/N) sum_i (I - b_i b_i^T),   trace(F_hat) = 2             (Eq. O4)

whose smallest eigenvalue lies in ``[0, 2/3]``.  For two observations separated
by an angle ``eta``, ``lambda_min(F_hat) = (1 - |cos eta|) / 2``, so the default
gate ``lambda_min >= 1e-6`` corresponds to ``eta >= 0.115 deg``.  The same test
is applied to the reference vectors, because a reference catalogue can be
degenerate even when the measurements are not.

The weights are deliberately kept out of Eq. O4.  A set of orthogonal
observations with one sensor a thousand times better than the other is
excellent geometry that the weights barely use; that is a precision question,
answered by the covariance of :mod:`wahbakit.covariance`, not an observability
question.  ``Observability.weighted_lambda_min`` reports the weighted value for
information, and :func:`wahbakit.covariance.optimal_covariance` refuses to
invert an Eq. O3 that is numerically singular.

References
----------
* G. Wahba, *SIAM Review* **7**(3), 409 (1965).
* M. D. Shuster, "Approximate algorithms for fast optimal attitude computation",
  AIAA Guidance and Control Conference, AIAA-78-1249 (1978).
* M. D. Shuster and S. D. Oh, *Journal of Guidance and Control* **4**(1), 70-77
  (1981).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import unit_vectors

__all__ = [
    "DEFAULT_DEGENERACY_TOL",
    "DegenerateObservationsError",
    "Observability",
    "VectorObservations",
]

#: Default gate on ``lambda_min`` of Eq. O4.  1e-6 is a 0.115 deg separation for
#: two equally weighted observations.
DEFAULT_DEGENERACY_TOL = 1e-6


class DegenerateObservationsError(ValueError):
    """Raised when the observation geometry does not determine three axes.

    Subclasses :class:`ValueError`, so ``except ValueError`` still catches it.
    The message always names the offending frame, the measured
    ``lambda_min`` of Eq. O4, the tolerance, and the equivalent separation
    angle in degrees.
    """


@dataclass(frozen=True)
class Observability:
    """Result of the degeneracy test (Eq. O4).  Dimensionless.

    Attributes
    ----------
    lambda_min_body, lambda_min_reference : float
        Smallest eigenvalue of Eq. O4 over the body and reference vectors
        respectively, unweighted.  Range ``[0, 2/3]``.
    lambda_min : float
        ``min`` of the two; the quantity that is gated.
    limiting_frame : str
        ``"body"`` or ``"reference"``, whichever produced ``lambda_min``.
    min_separation_deg : float
        Smallest angle between any two *body* observation directions, in
        degrees.  Reported for interpretation only; it is not the gate, because
        three coplanar directions can be well separated pairwise and still be
        poor geometry.
    equivalent_separation_deg : float
        Separation angle of two observations that would give this
        ``lambda_min``: ``arccos(1 - 2 lambda_min)`` in degrees.
    weighted_lambda_min : float
        Same as ``lambda_min`` but with the Wahba weights instead of ``1/N``.
        Informational: a small value here with a healthy ``lambda_min`` means
        one sensor dominates the solution, which is a precision statement, not
        an observability one.
    """

    lambda_min_body: float
    lambda_min_reference: float
    lambda_min: float
    limiting_frame: str
    min_separation_deg: float
    equivalent_separation_deg: float
    weighted_lambda_min: float

    def is_degenerate(self, tol: float = DEFAULT_DEGENERACY_TOL) -> bool:
        """True if ``lambda_min < tol``."""
        return self.lambda_min < tol


def _lambda_min(vectors: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    fisher = np.eye(3) * float(np.sum(weights)) - (vectors * weights[:, None]).T @ vectors
    return float(np.linalg.eigvalsh(0.5 * (fisher + fisher.T))[0])


class VectorObservations:
    """A set of paired body/reference unit vector observations.

    Parameters
    ----------
    body : array_like, shape (N, 3)
        Measured directions in the **body** frame.  Normalised on input.
    reference : array_like, shape (N, 3)
        Known directions in the **reference** frame, same order.  Normalised.
    sigmas : array_like, shape (N,), optional
        Per-observation transverse angular standard deviation in **radians**
        (Eq. O1).  Required for any covariance output.  Must be strictly
        positive.
    weights : array_like, shape (N,), optional
        Wahba weights (Eq. O2).  If omitted they default to ``1 / sigmas**2``
        when ``sigmas`` is given, and to equal weights otherwise.  Stored
        normalised to sum to one.  Must be strictly positive.

    Raises
    ------
    ValueError
        On mismatched shapes, fewer than two observations, non-finite entries,
        zero-length vectors, or non-positive sigmas or weights.

    Notes
    -----
    ``N >= 2`` is enforced: a single observation leaves one rotational degree of
    freedom free, which is a different problem (see :mod:`wahbakit.triad` for
    why the second observation is only used for its component orthogonal to the
    first).
    """

    __slots__ = ("_body", "_reference", "_sigmas", "_weights")

    def __init__(
        self,
        body: ArrayLike,
        reference: ArrayLike,
        *,
        sigmas: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        b = unit_vectors(body, name="body")
        r = unit_vectors(reference, name="reference")
        if b.shape != r.shape:
            raise ValueError(
                f"body and reference must have the same shape, got {b.shape} and {r.shape}"
            )
        if b.shape[0] < 2:
            raise ValueError(
                f"at least 2 observations are required, got {b.shape[0]}; "
                "a single pair leaves the rotation about it undetermined"
            )
        n = b.shape[0]

        s: NDArray[np.float64] | None = None
        if sigmas is not None:
            s = np.asarray(sigmas, dtype=float).reshape(-1)
            if s.size != n:
                raise ValueError(f"sigmas must have {n} entries, got {s.size}")
            if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
                raise ValueError("sigmas must be finite and strictly positive [rad]")

        if weights is None:
            w = np.ones(n) if s is None else 1.0 / (s**2)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != n:
                raise ValueError(f"weights must have {n} entries, got {w.size}")
            if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
                raise ValueError("weights must be finite and strictly positive")
        w = w / float(np.sum(w))

        self._body = b
        self._reference = r
        self._sigmas = s
        self._weights = w

    # -- accessors ---------------------------------------------------------
    @property
    def body(self) -> NDArray[np.float64]:
        """(N, 3) unit vectors in the body frame."""
        return self._body

    @property
    def reference(self) -> NDArray[np.float64]:
        """(N, 3) unit vectors in the reference frame."""
        return self._reference

    @property
    def weights(self) -> NDArray[np.float64]:
        """(N,) Wahba weights, normalised to sum to one (dimensionless)."""
        return self._weights

    @property
    def sigmas(self) -> NDArray[np.float64] | None:
        """(N,) transverse angular standard deviations [rad], or None."""
        return self._sigmas

    @property
    def n(self) -> int:
        """Number of observations."""
        return int(self._body.shape[0])

    @property
    def has_sigmas(self) -> bool:
        """True if per-observation sigmas were supplied."""
        return self._sigmas is not None

    def require_sigmas(self, what: str) -> NDArray[np.float64]:
        """Return the sigmas or raise a ValueError naming ``what`` needs them."""
        if self._sigmas is None:
            raise ValueError(
                f"{what} requires per-observation sigmas [rad]; construct with "
                "VectorObservations(body, reference, sigmas=...)"
            )
        return self._sigmas

    def subset(self, indices: ArrayLike) -> VectorObservations:
        """New :class:`VectorObservations` holding only ``indices``.

        Weights are re-normalised over the subset; sigmas are carried through
        unchanged, so a subset's weights are still ``1/sigma^2`` in proportion.
        """
        idx = np.asarray(indices, dtype=int).reshape(-1)
        return VectorObservations(
            self._body[idx],
            self._reference[idx],
            sigmas=None if self._sigmas is None else self._sigmas[idx],
            weights=self._weights[idx],
        )

    # -- derived quantities ------------------------------------------------
    def attitude_profile_matrix(self) -> NDArray[np.float64]:
        """Wahba attitude profile matrix ``B = sum_i w_i b_i r_i^T`` (3, 3).

        Dimensionless.  Maximising the gain ``trace(A B^T)`` is equivalent to
        minimising Eq. O2, since ``L(A) = sum_i w_i - trace(A B^T)`` for unit
        vectors (Wahba 1965; Markley & Crassidis 2014, Ch. 5).
        """
        return (self._body * self._weights[:, None]).T @ self._reference

    def observability(self) -> Observability:
        """Evaluate Eq. O4 in both frames.  See :class:`Observability`."""
        uniform = np.full(self.n, 1.0 / self.n)
        lam_b = _lambda_min(self._body, uniform)
        lam_r = _lambda_min(self._reference, uniform)
        lam = min(lam_b, lam_r)
        frame = "body" if lam_b <= lam_r else "reference"

        gram = np.clip(self._body @ self._body.T, -1.0, 1.0)
        np.fill_diagonal(gram, 1.0)
        off = np.abs(gram[np.triu_indices(self.n, k=1)])
        min_sep = float(np.degrees(np.arccos(np.max(off)))) if off.size else 180.0

        equivalent = float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * lam, -1.0, 1.0))))
        weighted = min(
            _lambda_min(self._body, self._weights),
            _lambda_min(self._reference, self._weights),
        )
        return Observability(
            lambda_min_body=lam_b,
            lambda_min_reference=lam_r,
            lambda_min=lam,
            limiting_frame=frame,
            min_separation_deg=min_sep,
            equivalent_separation_deg=equivalent,
            weighted_lambda_min=weighted,
        )

    def require_observable(self, tol: float = DEFAULT_DEGENERACY_TOL) -> Observability:
        """Raise :class:`DegenerateObservationsError` if Eq. O4 falls below ``tol``.

        Parameters
        ----------
        tol : float
            Gate on ``lambda_min`` of Eq. O4.  Default
            :data:`DEFAULT_DEGENERACY_TOL` = 1e-6, i.e. 0.115 deg for two
            equally weighted observations.
        """
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")
        obs = self.observability()
        if obs.is_degenerate(tol):
            raise DegenerateObservationsError(
                f"observations are degenerate in the {obs.limiting_frame} frame: "
                f"lambda_min = {obs.lambda_min:.3e} < tol = {tol:.3e} "
                f"(Eq. O4), equivalent to a separation of "
                f"{obs.equivalent_separation_deg:.4f} deg between two equally "
                f"weighted observations; smallest body-vector separation is "
                f"{obs.min_separation_deg:.4f} deg. The rotation about the common "
                "direction is not determined by this data. Add an independent "
                "observation, or lower tol only if you accept an arbitrary "
                "answer about that axis."
            )
        return obs

    def residual_angles(self, dcm: ArrayLike) -> NDArray[np.float64]:
        """Per-observation angle between ``b_i`` and ``A r_i``, in **rad**."""
        a = np.asarray(dcm, dtype=float)
        if a.shape != (3, 3):
            raise ValueError(f"dcm must have shape (3, 3), got {a.shape}")
        predicted = self._reference @ a.T
        cross = np.linalg.norm(np.cross(self._body, predicted), axis=1)
        dot = np.sum(self._body * predicted, axis=1)
        return np.arctan2(cross, dot)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        sig = "None" if self._sigmas is None else np.array2string(self._sigmas, precision=4)
        return (
            f"VectorObservations(n={self.n}, "
            f"weights={np.array2string(self._weights, precision=4)}, sigmas={sig})"
        )
