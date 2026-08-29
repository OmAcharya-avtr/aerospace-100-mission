"""Discrete linear Kalman filter, Joseph-form update, and the steady-state Riccati solution.

Filter recursions (Kalman, R. E. (1960), "A New Approach to Linear Filtering
and Prediction Problems", *Transactions of the ASME — Journal of Basic
Engineering* 82(D), 35-45; notation of Bar-Shalom, Li & Kirubarajan (2001),
*Estimation with Applications to Tracking and Navigation*, §5.2):

    predict   x⁻_k = F x⁺_{k-1} + B u_{k-1}
              P⁻_k = F P⁺_{k-1} Fᵀ + Q
    innovate  ν_k  = z_k − H x⁻_k
              S_k  = H P⁻_k Hᵀ + R
              K_k  = P⁻_k Hᵀ S_k⁻¹
    update    x⁺_k = x⁻_k + K_k ν_k
              P⁺_k = (I − K_k H) P⁻_k (I − K_k H)ᵀ + K_k R K_kᵀ   (Joseph form)

The Joseph form is used unconditionally.  It is algebraically equal to
``(I − K H) P⁻`` **only at the optimal gain**; it stays symmetric and positive
semi-definite for any gain and is markedly better conditioned in finite
precision (Bucy & Joseph 1968; Bierman 1977, *Factorization Methods for
Discrete Sequential Estimation*, §3.2).  Related prior art: product P017
(EstimKit) in this portfolio implements the same family for teaching; this
implementation is independent and adds per-step NEES/NIS bookkeeping, which
is NavBench's reason to exist.

Units are whatever the caller's state carries; the code is unit-agnostic and
every docstring states the shape contract instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "symmetrize",
    "joseph_update",
    "FilterResult",
    "KalmanFilter",
    "steady_state_riccati",
    "CovarianceCollapseError",
    "covariance_health",
]


class CovarianceCollapseError(RuntimeError):
    """Raised when a covariance matrix has lost positive definiteness.

    This is the observable symptom of numerical covariance collapse: the
    Cholesky factorisation of ``P`` (or of the innovation covariance ``S``)
    fails, or the minimum eigenvalue drops below the stated floor.
    """


def _as_matrix(m: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    arr = np.atleast_2d(np.asarray(m, dtype=float))
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def symmetrize(p: ArrayLike) -> NDArray[np.float64]:
    """Return a bit-exactly symmetric copy of ``p``.

    The upper triangle of ``(P + Pᵀ)/2`` is written into both triangles, so
    ``result == result.T`` holds exactly in floating point rather than to
    round-off.
    """
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"p must be square, got shape {a.shape}")
    out = 0.5 * (a + a.T)
    iu = np.triu_indices_from(out, k=1)
    out[(iu[1], iu[0])] = out[iu]
    return out


def joseph_update(
    p_prior: ArrayLike, gain: ArrayLike, h: ArrayLike, r: ArrayLike
) -> NDArray[np.float64]:
    """Joseph-form covariance update ``(I−KH) P (I−KH)ᵀ + K R Kᵀ``.

    Valid for **any** gain, not only the optimal Kalman gain.  Returns a
    bit-exactly symmetric matrix.
    """
    p = np.asarray(p_prior, dtype=float)
    n = p.shape[0]
    k = np.atleast_2d(np.asarray(gain, dtype=float))
    hm = np.atleast_2d(np.asarray(h, dtype=float))
    rm = np.atleast_2d(np.asarray(r, dtype=float))
    if p.ndim != 2 or p.shape != (n, n):
        raise ValueError(f"p_prior must be square, got {p.shape}")
    m = hm.shape[0]
    if hm.shape != (m, n):
        raise ValueError(f"h must have shape (m, {n}), got {hm.shape}")
    if k.shape != (n, m):
        raise ValueError(f"gain must have shape ({n}, {m}), got {k.shape}")
    if rm.shape != (m, m):
        raise ValueError(f"r must have shape ({m}, {m}), got {rm.shape}")
    a = np.eye(n) - k @ hm
    return symmetrize(a @ p @ a.T + k @ rm @ k.T)


def covariance_health(p: ArrayLike) -> dict[str, float]:
    """Diagnostics for a covariance matrix.

    Returns
    -------
    dict with keys
        ``asymmetry`` — ``max|P − Pᵀ|``;
        ``min_eigenvalue``, ``max_eigenvalue`` — of the symmetrised matrix;
        ``condition`` — ratio of extreme eigenvalues (``inf`` if singular);
        ``trace``.
    """
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"p must be square, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("p must be finite")
    eig = np.linalg.eigvalsh(0.5 * (a + a.T))
    lo, hi = float(eig.min()), float(eig.max())
    cond = float("inf") if lo <= 0.0 else hi / lo
    return {
        "asymmetry": float(np.max(np.abs(a - a.T))),
        "min_eigenvalue": lo,
        "max_eigenvalue": hi,
        "condition": cond,
        "trace": float(np.trace(a)),
    }


@dataclass(frozen=True)
class FilterResult:
    """Histories returned by a batch filter run.

    Attributes
    ----------
    x_prior, x_post : ndarray, shape (N, n)
        Predicted and updated state estimates for each of the ``N`` steps.
    p_prior, p_post : ndarray, shape (N, n, n)
        Corresponding covariances.
    innovation : ndarray, shape (N, m)
        ``ν_k = z_k − h(x⁻_k)``.  Rows of NaN mark skipped measurements.
    innovation_cov : ndarray, shape (N, m, m)
        ``S_k``.
    gain : ndarray, shape (N, n, m)
        ``K_k``.
    nis : ndarray, shape (N,)
        ``ν_kᵀ S_k⁻¹ ν_k``; NaN where no measurement was processed.
    updated : ndarray of bool, shape (N,)
        True where a measurement was processed at that step.
    """

    x_prior: NDArray[np.float64]
    x_post: NDArray[np.float64]
    p_prior: NDArray[np.float64]
    p_post: NDArray[np.float64]
    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64]
    gain: NDArray[np.float64]
    nis: NDArray[np.float64]
    updated: NDArray[np.bool_]

    @property
    def n_steps(self) -> int:
        """Number of filter steps recorded."""
        return int(self.x_post.shape[0])

    def mean_nis(self) -> float:
        """Mean NIS over the steps that actually processed a measurement."""
        vals = self.nis[self.updated]
        return float(np.mean(vals)) if vals.size else float("nan")


class KalmanFilter:
    """Discrete-time linear Kalman filter with Joseph-form covariance update.

    Parameters
    ----------
    f : array_like, shape (n, n)
        State transition matrix.
    h : array_like, shape (m, n)
        Measurement matrix.
    q : array_like, shape (n, n)
        Process noise covariance (discrete).
    r : array_like, shape (m, m)
        Measurement noise covariance.
    x0 : array_like, shape (n,)
        Initial state estimate.
    p0 : array_like, shape (n, n)
        Initial covariance; must be symmetric positive semi-definite.
    b : array_like, shape (n, p), optional
        Control input matrix.

    Raises
    ------
    ValueError
        On shape mismatch, non-finite entries, asymmetric or indefinite
        ``q``/``r``/``p0``, or singular ``r``.
    """

    def __init__(
        self,
        f: ArrayLike,
        h: ArrayLike,
        q: ArrayLike,
        r: ArrayLike,
        x0: ArrayLike,
        p0: ArrayLike,
        b: ArrayLike | None = None,
    ) -> None:
        x = np.asarray(x0, dtype=float).ravel()
        n = x.size
        if n == 0:
            raise ValueError("x0 must have at least one element")
        if not np.all(np.isfinite(x)):
            raise ValueError("x0 must be finite")
        self.f = _as_matrix(f, (n, n), "f")
        hm = np.atleast_2d(np.asarray(h, dtype=float))
        if hm.ndim != 2 or hm.shape[1] != n:
            raise ValueError(f"h must have shape (m, {n}), got {hm.shape}")
        if not np.all(np.isfinite(hm)):
            raise ValueError("h must be finite")
        self.h = hm
        m = hm.shape[0]
        self.q = self._check_psd(_as_matrix(q, (n, n), "q"), "q")
        self.r = self._check_psd(_as_matrix(r, (m, m), "r"), "r", strict=True)
        self.p = self._check_psd(_as_matrix(p0, (n, n), "p0"), "p0")
        self.x = x.copy()
        self.b = None if b is None else np.atleast_2d(np.asarray(b, dtype=float))
        if self.b is not None and self.b.shape[0] != n:
            raise ValueError(f"b must have {n} rows, got shape {self.b.shape}")
        self.n, self.m = n, m

    @staticmethod
    def _check_psd(a: NDArray[np.float64], name: str, strict: bool = False) -> NDArray[np.float64]:
        scale = max(1.0, float(np.max(np.abs(a))))
        if float(np.max(np.abs(a - a.T))) > 1e-9 * scale:
            raise ValueError(f"{name} must be symmetric")
        eig = float(np.linalg.eigvalsh(0.5 * (a + a.T)).min())
        if strict and eig <= 0.0:
            raise ValueError(f"{name} must be positive definite; min eigenvalue {eig:.3e}")
        if not strict and eig < -1e-9 * scale:
            raise ValueError(f"{name} must be positive semi-definite; min eigenvalue {eig:.3e}")
        return symmetrize(a)

    def predict(
        self, u: ArrayLike | None = None, f: ArrayLike | None = None, q: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Time update.  Optional per-step ``f``/``q`` override for time-varying systems."""
        fm = self.f if f is None else _as_matrix(f, (self.n, self.n), "f")
        qm = self.q if q is None else self._check_psd(_as_matrix(q, (self.n, self.n), "q"), "q")
        self.x = fm @ self.x
        if u is not None:
            if self.b is None:
                raise ValueError("control input u supplied but the filter has no b matrix")
            uu = np.asarray(u, dtype=float).ravel()
            if uu.size != self.b.shape[1]:
                raise ValueError(f"u must have {self.b.shape[1]} elements, got {uu.size}")
            self.x = self.x + self.b @ uu
        self.p = symmetrize(fm @ self.p @ fm.T + qm)
        return self.x.copy(), self.p.copy()

    def update(
        self, z: ArrayLike, h: ArrayLike | None = None, r: ArrayLike | None = None
    ) -> dict[str, object]:
        """Measurement update.

        Returns
        -------
        dict
            ``x``, ``p``, ``innovation``, ``innovation_cov``, ``gain``, ``nis``.

        Raises
        ------
        CovarianceCollapseError
            If ``S`` is not positive definite (its Cholesky factorisation
            fails), which is how a collapsed or grossly mis-specified
            covariance manifests.
        """
        hm = self.h if h is None else np.atleast_2d(np.asarray(h, dtype=float))
        if hm.shape[1] != self.n:
            raise ValueError(f"h must have {self.n} columns, got {hm.shape}")
        m = hm.shape[0]
        rm = self.r if r is None else _as_matrix(r, (m, m), "r")
        zz = np.asarray(z, dtype=float).ravel()
        if zz.size != m:
            raise ValueError(f"z must have {m} elements to match h, got {zz.size}")
        if not np.all(np.isfinite(zz)):
            raise ValueError("z must be finite")

        nu = zz - hm @ self.x
        s = symmetrize(hm @ self.p @ hm.T + rm)
        try:
            chol = np.linalg.cholesky(s)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by tests
            raise CovarianceCollapseError(
                f"innovation covariance S is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(s).min()):.3e}); "
                "check R > 0 and that P has not collapsed"
            ) from exc
        s_inv_nu = np.linalg.solve(s, nu)
        gain = np.linalg.solve(s, (self.p @ hm.T).T).T
        self.x = self.x + gain @ nu
        self.p = joseph_update(self.p, gain, hm, rm)
        del chol
        return {
            "x": self.x.copy(),
            "p": self.p.copy(),
            "innovation": nu,
            "innovation_cov": s,
            "gain": gain,
            "nis": float(nu @ s_inv_nu),
        }

    def run(
        self,
        measurements: ArrayLike,
        controls: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> FilterResult:
        """Run predict/update over a batch of measurements.

        Parameters
        ----------
        measurements : array_like, shape (N, m)
            One row per step.  Rows containing NaN are treated as missing and
            only the prediction is applied (sensor-dropout handling).
        controls : array_like, shape (N, p), optional
        mask : array_like of bool, shape (N,), optional
            Explicit availability flags, ANDed with the NaN check.
        """
        z = np.atleast_2d(np.asarray(measurements, dtype=float))
        if z.ndim != 2 or z.shape[1] != self.m:
            raise ValueError(f"measurements must have shape (N, {self.m}), got {z.shape}")
        n_steps = z.shape[0]
        avail = np.all(np.isfinite(z), axis=1)
        if mask is not None:
            mk = np.asarray(mask, dtype=bool).ravel()
            if mk.size != n_steps:
                raise ValueError(f"mask must have {n_steps} elements, got {mk.size}")
            avail = avail & mk
        u_all = None if controls is None else np.atleast_2d(np.asarray(controls, dtype=float))

        xp = np.zeros((n_steps, self.n))
        xu = np.zeros((n_steps, self.n))
        pp = np.zeros((n_steps, self.n, self.n))
        pu = np.zeros((n_steps, self.n, self.n))
        nu = np.full((n_steps, self.m), np.nan)
        ss = np.zeros((n_steps, self.m, self.m))
        kk = np.zeros((n_steps, self.n, self.m))
        nis = np.full(n_steps, np.nan)

        for k in range(n_steps):
            u = None if u_all is None else u_all[k]
            x_pred, p_pred = self.predict(u=u)
            xp[k], pp[k] = x_pred, p_pred
            if avail[k]:
                out = self.update(z[k])
                nu[k] = out["innovation"]  # type: ignore[assignment]
                ss[k] = out["innovation_cov"]  # type: ignore[assignment]
                kk[k] = out["gain"]  # type: ignore[assignment]
                nis[k] = out["nis"]  # type: ignore[assignment]
            else:
                ss[k] = self.h @ self.p @ self.h.T + self.r
            xu[k], pu[k] = self.x.copy(), self.p.copy()
        return FilterResult(xp, xu, pp, pu, nu, ss, kk, nis, avail)


def steady_state_riccati(
    f: ArrayLike,
    h: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    tol: float = 1e-14,
    max_iter: int = 100000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Fixed-point solution of the filtering discrete algebraic Riccati equation.

    Iterates ``P⁻ ← F [P⁻ − P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹HP⁻] Fᵀ + Q`` from ``P⁻ = Q``
    until the increment falls below ``tol`` in max-norm.

    Returns
    -------
    (P⁻_∞, P⁺_∞, K_∞, iterations)

    Raises
    ------
    ValueError
        On shape/definiteness problems, or if the iteration has not converged
        within ``max_iter`` (the pair may be undetectable/unstabilisable).

    Notes
    -----
    **The increment test is a stopping rule, not an error bound.**  The
    iteration converges linearly with a rate that approaches 1 as the
    signal-to-noise ratio ``tr(Q)/tr(R)`` becomes small, so the remaining
    error can exceed the last increment by a large factor.  Measured on the
    scalar random walk with ``q = 1e-3``, ``r = 1e3``: ``tol = 1e-14`` stops
    after 13351 iterations with a **relative** error of 5.1e-12 against the
    closed form, while ``tol = 1e-16`` reaches the floating-point floor of
    1.1e-13 after 15447 iterations.  For well-conditioned problems the two
    agree to round-off (see ``validation/v1_riccati_steady_state.py``).
    """
    fm = np.atleast_2d(np.asarray(f, dtype=float))
    n = fm.shape[0]
    if fm.shape != (n, n):
        raise ValueError(f"f must be square, got {fm.shape}")
    hm = np.atleast_2d(np.asarray(h, dtype=float))
    if hm.shape[1] != n:
        raise ValueError(f"h must have {n} columns, got {hm.shape}")
    m = hm.shape[0]
    qm = _as_matrix(q, (n, n), "q")
    rm = _as_matrix(r, (m, m), "r")
    if float(np.linalg.eigvalsh(symmetrize(rm)).min()) <= 0.0:
        raise ValueError("r must be positive definite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and > 0, got {tol!r}")

    p = symmetrize(qm.copy())
    for it in range(1, int(max_iter) + 1):
        s = symmetrize(hm @ p @ hm.T + rm)
        k = np.linalg.solve(s, (p @ hm.T).T).T
        p_post = joseph_update(p, k, hm, rm)
        p_next = symmetrize(fm @ p_post @ fm.T + qm)
        delta = float(np.max(np.abs(p_next - p)))
        p = p_next
        if delta < tol:
            s = symmetrize(hm @ p @ hm.T + rm)
            k = np.linalg.solve(s, (p @ hm.T).T).T
            return p, joseph_update(p, k, hm, rm), k, it
    raise ValueError(
        f"steady-state Riccati iteration did not converge in {max_iter} iterations "
        f"(last increment {delta:.3e}); the (F, H) pair may be undetectable"
    )
