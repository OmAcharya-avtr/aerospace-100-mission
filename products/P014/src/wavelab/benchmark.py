"""Benchmark harness: classical baselines versus the learned reconstructor.

Error metric
------------
Every reconstructor in this product is scored by the **residual wavefront RMS
over the estimated modes**, in radians:

    rms = sqrt( mean_samples( sum_j (a_hat_j - a_j)^2 ) )

Because the Zernike basis of :mod:`wavelab.zernike` is orthonormal under the
area-normalised weight on the unit disc (Noll 1976), the inner sum *is* the
spatial variance over the pupil of the reconstruction error restricted to those
modes. The metric therefore has a direct physical meaning and is comparable
between the modal least-squares baseline and the learned model, which estimate
exactly the same quantity from exactly the same input.

Fairness rules applied here
---------------------------
* The classical baseline is tuned on a **validation** split disjoint from both
  the training split used by the learned model and the test split used for the
  reported numbers.
* The baseline is allowed a regularisation strength chosen *per operating
  point* (per photon flux and dropout rate). That is the most favourable
  practical setting for it, and it is what a well-engineered classical system
  would do, since it knows its own flux and its own subaperture validity map.
* The baseline may also use a **coloured prior** (per-mode Zernike standard
  deviations measured on the training split), which makes it the diagonal
  minimum-variance estimator rather than a plain white-prior ridge. Without
  that concession the comparison would flatter the learned model, which sees
  the true prior through its training data.
* Both estimators see the identical noisy, masked slope vectors and the
  identical photon count.

Implementation note
-------------------
With per-sample random dropout, every sample generally has its own
reconstruction matrix. Building them one at a time is far too slow for a sweep,
so :class:`MaskedModalSolver` takes the SVD of every distinct masked
interaction matrix **once**, in a single batched `numpy.linalg.svd` call, and
then applies any spectral filter (TSVD or Tikhonov, at any strength) for the
cost of a multiply. Zeroing the rows of unavailable subapertures is exactly
equivalent to deleting them, because the corresponding entries of ``u`` are
zero too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dataset import SlopeDataset, make_measurements
from .geometry import SubapertureGeometry
from .reconstruct import ModalReconstructor

__all__ = [
    "residual_rms",
    "OperatingPoint",
    "make_operating_point",
    "MaskedModalSolver",
    "tune_modal_regularisation",
    "evaluate_modal_baseline",
]


def residual_rms(pred: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """Residual wavefront RMS [rad] between predicted and true coefficient sets."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    if p.shape != t.shape:
        raise ValueError(f"pred and truth must have the same shape, got {p.shape} and {t.shape}")
    if p.ndim != 2:
        raise ValueError(f"pred and truth must be 2-D, got shape {p.shape}")
    return float(np.sqrt(np.mean(np.sum((p - t) ** 2, axis=1))))


@dataclass(frozen=True)
class OperatingPoint:
    """One (photon flux, dropout rate) test condition with its realised measurements.

    Attributes
    ----------
    n_photons : float
        Photons per subaperture [-].
    dropout_rate : float
        Probability a subaperture is lost [-].
    u_meas : ndarray, shape (n, 2 * n_sub)
        Noisy, masked scaled slopes [rad].
    available : ndarray of bool, shape (n, n_sub)
        Subaperture availability.
    coeffs : ndarray, shape (n, n_modes)
        True Zernike coefficients [rad].
    """

    n_photons: float
    dropout_rate: float
    u_meas: NDArray[np.float64]
    available: NDArray[np.bool_]
    coeffs: NDArray[np.float64]


def make_operating_point(
    data: SlopeDataset, n_photons: float, dropout_rate: float, seed: int
) -> OperatingPoint:
    """Realise one operating point from a clean dataset with a fixed seed."""
    rng = np.random.default_rng(int(seed))
    u_meas, avail, _ = make_measurements(data.u, float(n_photons), float(dropout_rate), rng)
    return OperatingPoint(
        n_photons=float(n_photons),
        dropout_rate=float(dropout_rate),
        u_meas=u_meas,
        available=avail,
        coeffs=data.coeffs,
    )


class MaskedModalSolver:
    """Batched modal least-squares solver over many distinct dropout masks.

    Parameters
    ----------
    geom : SubapertureGeometry
        Lenslet layout.
    noll_indices : tuple[int, ...]
        Modes to estimate.
    available : ndarray of bool, shape (n, n_sub)
        Per-sample subaperture availability.
    prior_std : ndarray, optional
        Per-mode prior standard deviation [rad]; see
        :class:`wavelab.reconstruct.ModalReconstructor`.

    Notes
    -----
    The SVD is taken once per **distinct** mask. Changing the regularisation
    afterwards costs one filter multiply, so a sweep over regularisation
    strengths is nearly free.
    """

    def __init__(
        self,
        geom: SubapertureGeometry,
        noll_indices: tuple[int, ...],
        available: NDArray[np.bool_],
        prior_std: NDArray[np.float64] | None = None,
    ) -> None:
        avail = np.asarray(available, dtype=bool)
        if avail.ndim != 2 or avail.shape[1] != geom.n_valid_sub:
            raise ValueError(
                f"available must have shape (n, {geom.n_valid_sub}), got {avail.shape}"
            )
        rec = ModalReconstructor(geom, noll_indices, prior_std=prior_std)
        self._w = rec.prior_weights
        masks, self._inverse = np.unique(avail, axis=0, return_inverse=True)
        rows = np.concatenate([masks, masks], axis=1).astype(float)
        design = rec.interaction[None, :, :] * self._w[None, None, :] * rows[:, :, None]
        self._u, self._s, self._vt = np.linalg.svd(design, full_matrices=False)
        self._smax = self._s[:, :1]

    @property
    def n_masks(self) -> int:
        """Number of distinct dropout masks that had to be factorised [-]."""
        return int(self._s.shape[0])

    def solve(
        self, u_meas: NDArray[np.float64], method: str = "tikhonov", reg: float = 1e-3
    ) -> NDArray[np.float64]:
        """Estimate Zernike coefficients [rad] for every sample."""
        u = np.asarray(u_meas, dtype=float)
        if u.ndim != 2 or u.shape[0] != self._inverse.size:
            raise ValueError(
                f"u_meas must have shape ({self._inverse.size}, n_slopes), got {u.shape}"
            )
        r = float(reg)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"reg must be finite and > 0, got {reg!r}")
        m = str(method).strip().lower()
        lam = r * self._smax
        if m == "tsvd":
            filt = np.where(self._s > lam, 1.0 / np.maximum(self._s, np.finfo(float).tiny), 0.0)
        elif m == "tikhonov":
            filt = self._s / (self._s**2 + lam**2)
        else:
            raise ValueError(f"method must be 'tsvd' or 'tikhonov', got {method!r}")
        idx = self._inverse
        proj = np.einsum("nij,ni->nj", self._u[idx], u)
        return np.einsum("nji,nj->ni", self._vt[idx], filt[idx] * proj) * self._w


def evaluate_modal_baseline(
    geom: SubapertureGeometry,
    point: OperatingPoint,
    noll_indices: tuple[int, ...],
    method: str,
    reg: float,
    prior_std: NDArray[np.float64] | None = None,
) -> float:
    """Residual wavefront RMS [rad] of the modal least-squares baseline at one point.

    A separate reconstruction is performed for every distinct dropout pattern in
    the batch, which is what a real system with a live subaperture validity map
    does.
    """
    solver = MaskedModalSolver(geom, noll_indices, point.available, prior_std)
    return residual_rms(solver.solve(point.u_meas, method, reg), point.coeffs)


def tune_modal_regularisation(
    geom: SubapertureGeometry,
    point: OperatingPoint,
    noll_indices: tuple[int, ...],
    method: str,
    grid: NDArray[np.float64],
    prior_std: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """Pick the regularisation strength minimising the error on this (validation) point.

    Returns
    -------
    best_reg : float
        Regularisation strength [-].
    best_rms : float
        Residual wavefront RMS at that strength [rad].
    """
    g = np.asarray(grid, dtype=float).ravel()
    if g.size == 0 or np.any(g <= 0.0) or np.any(~np.isfinite(g)):
        raise ValueError("grid must be a non-empty array of finite positive values")
    solver = MaskedModalSolver(geom, noll_indices, point.available, prior_std)
    scores = [residual_rms(solver.solve(point.u_meas, method, r), point.coeffs) for r in g]
    k = int(np.argmin(scores))
    return float(g[k]), float(scores[k])
