"""Regularised least-squares slope-to-phase reconstruction.

Problem
-------
Given the measured phase differences across subapertures ``u`` [rad]
(``u = pitch * gradient``) and a geometry operator pair ``A p = B u`` from
:mod:`wavelab.geometry`, the least-squares estimate of the phase ``p`` [rad] is

    p_hat = argmin_p || A p - B u ||_2^2

which is a rank-deficient problem: ``A`` always annihilates piston, and the
Fried operator additionally has a numerically tiny "waffle" singular value.
Both need regularisation, and the two standard choices are implemented.

Why these two regularisers
--------------------------
**Truncated SVD (TSVD)** is the default for the *well-posed* configuration --
full illumination, all subapertures present. There the singular spectrum has a
clean gap: piston (and, for Fried, waffle) sit many orders of magnitude below
everything else. Truncating at a relative threshold removes exactly the
unobservable directions and leaves every observable mode **unbiased**, which is
what the "noise-free reconstruction is exact" validation requires. Reference:
P. C. Hansen, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM 1998,
ch. 3 (TSVD as a spectral filter with filter factors 0 or 1).

**Tikhonov** is the default for the *ill-posed* configuration -- low flux, or
missing subapertures. With dropout the spectrum has no gap: removing rows
degrades a continuum of directions, so any hard cut is arbitrary. Tikhonov
applies the smooth spectral filter ``f_i = sigma_i^2 / (sigma_i^2 + lambda^2)``,
which is the minimum-mean-square-error filter for white measurement noise and a
white prior on ``p``, i.e. it deliberately buys a bias to kill variance.
References: A. N. Tikhonov and V. Y. Arsenin, *Solutions of Ill-Posed
Problems*, Winston/Wiley 1977; Hansen 1998, ch. 5.

``lambda`` is specified **relative to the largest singular value** of the
operator actually being inverted, so a single number transfers across
geometries, array sizes and dropout patterns.

Noise propagation
-----------------
For a linear reconstructor ``R`` (``p_hat = R u``) and white measurement noise
``cov(u) = sigma_u^2 I``, the reconstructed phase error covariance is
``sigma_u^2 R R^T``. Removing the unobservable piston with the projector
``P = I - (1/M) 1 1^T`` and averaging over the ``M`` phase points gives the
mean-square reconstructed phase error

    <|p_err|^2> = sigma_u^2 * a_NP,      a_NP = ||P R||_F^2 / M

``a_NP`` is the dimensionless **noise-propagation coefficient** of the
reconstructor -- the quantity Fried (1977) and Southwell (1980, Sec. IV)
tabulate for their geometries and which is known to grow logarithmically with
the number of subapertures across the aperture (Fried 1977; R. H. Hudgin,
"Wave-front reconstruction for compensated imaging", *JOSA* **67** (3),
375-378, 1977). It is computed here **analytically from the reconstructor
matrix**, with no Monte Carlo, and `validation/VALIDATION.md` checks a Monte
Carlo against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import GeometryMatrices, SubapertureGeometry, build_geometry_matrices
from .zernike import zernike_gradient_basis

__all__ = [
    "regularised_pinv",
    "ZonalReconstructor",
    "ModalReconstructor",
    "noise_propagation_coefficient",
    "piston_remove",
]


def _check_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {method!r}")
    m = method.strip().lower()
    if m not in ("tsvd", "tikhonov"):
        raise ValueError(f"method must be 'tsvd' or 'tikhonov', got {method!r}")
    return m


def regularised_pinv(
    a: NDArray[np.float64], method: str = "tsvd", reg: float = 1e-6
) -> NDArray[np.float64]:
    """Regularised pseudo-inverse of ``a``.

    Parameters
    ----------
    a : ndarray, shape (n_eq, n_unknown)
        Operator to invert. Dimensionless.
    method : {'tsvd', 'tikhonov'}
        Spectral filter. TSVD keeps singular values above ``reg * sigma_max``;
        Tikhonov applies ``sigma / (sigma**2 + (reg * sigma_max)**2)``.
    reg : float
        Regularisation strength **relative to the largest singular value**
        [-], > 0.

    Returns
    -------
    ndarray, shape (n_unknown, n_eq)

    Raises
    ------
    ValueError
        If ``a`` is not 2-D, is empty, contains non-finite entries, or ``reg``
        is not a finite positive number.
    """
    m = _check_method(method)
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"a must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("a must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("a contains non-finite entries")
    r = float(reg)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"reg must be finite and > 0, got {reg!r}")

    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    smax = float(s[0]) if s.size else 0.0
    if smax <= 0.0:
        return np.zeros((arr.shape[1], arr.shape[0]))
    lam = r * smax
    if m == "tsvd":
        inv_s = np.where(s > lam, 1.0 / np.maximum(s, np.finfo(float).tiny), 0.0)
    else:
        inv_s = s / (s**2 + lam**2)
    return (vt.T * inv_s) @ u.T


def piston_remove(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Subtract the mean over the last axis (piston is unobservable from slopes)."""
    arr = np.asarray(p, dtype=float)
    return arr - arr.mean(axis=-1, keepdims=True)


def noise_propagation_coefficient(r: NDArray[np.float64]) -> float:
    """Dimensionless noise-propagation coefficient ``a_NP`` of a reconstructor.

    ``a_NP = ||P R||_F^2 / M`` with ``P`` the piston-removing projector and
    ``M`` the number of phase points. Then the mean-square reconstructed phase
    error caused by white measurement noise of variance ``sigma_u^2`` [rad^2]
    is ``sigma_u^2 * a_NP`` [rad^2].

    See the module docstring for the derivation and for the Fried (1977) /
    Hudgin (1977) / Southwell (1980) context.
    """
    mat = np.asarray(r, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"r must be 2-D, got shape {mat.shape}")
    if mat.size == 0:
        raise ValueError("r must be non-empty")
    centred = mat - mat.mean(axis=0, keepdims=True)
    return float(np.sum(centred**2) / mat.shape[0])


@dataclass
class ZonalReconstructor:
    """Regularised least-squares zonal reconstructor for one geometry.

    Parameters
    ----------
    geom : SubapertureGeometry
        Lenslet layout.
    geometry : {'southwell', 'fried'}
        Reconstruction geometry (see :mod:`wavelab.geometry`).
    method : {'tsvd', 'tikhonov'}
        Regulariser.
    reg : float
        Relative regularisation strength [-].

    Notes
    -----
    Inputs are the scaled slopes ``u = pitch * gradient`` [rad]; outputs are
    piston-removed phase estimates [rad] at the geometry's own phase points,
    whose normalised coordinates are ``geom.phase_points(geometry)``.

    Reconstruction matrices are cached per dropout pattern, so repeated calls
    with the same mask cost one matrix-vector product.
    """

    geom: SubapertureGeometry
    geometry: str = "southwell"
    method: str = "tsvd"
    reg: float = 1e-6

    def __post_init__(self) -> None:
        self.matrices: GeometryMatrices = build_geometry_matrices(self.geom, self.geometry)
        self.method = _check_method(self.method)
        r = float(self.reg)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"reg must be finite and > 0, got {self.reg!r}")
        self.reg = r
        self._cache: dict[bytes, NDArray[np.float64]] = {}

    @property
    def n_phase(self) -> int:
        """Number of reconstructed phase points [-]."""
        return self.matrices.n_phase

    def phase_points(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised ``(x, y)`` of the reconstructed phase points."""
        return self.geom.phase_points(self.geometry)

    def matrix(self, sub_available: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """Reconstruction matrix ``R`` with ``p_hat = R u``.

        Parameters
        ----------
        sub_available : ndarray of bool, shape (n_valid_sub,), optional
            Which subapertures delivered a usable measurement. ``None`` means
            all of them.

        Returns
        -------
        ndarray, shape (n_phase, n_slopes)
            Columns corresponding to dropped subapertures are exactly zero.
        """
        n_sub = self.geom.n_valid_sub
        avail = (
            np.ones(n_sub, dtype=bool)
            if sub_available is None
            else np.asarray(sub_available, dtype=bool).ravel()
        )
        if avail.shape != (n_sub,):
            raise ValueError(f"sub_available must have shape ({n_sub},), got {avail.shape}")
        key = np.packbits(avail).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rows = self.matrices.active_rows(avail)
        a = self.matrices.a[rows]
        b = self.matrices.b[rows]
        if a.shape[0] == 0:
            r_mat = np.zeros((self.matrices.n_phase, self.matrices.n_slopes))
        else:
            r_mat = regularised_pinv(a, self.method, self.reg) @ b
            r_mat = r_mat - r_mat.mean(axis=0, keepdims=True)
        self._cache[key] = r_mat
        return r_mat

    def reconstruct(
        self, u: NDArray[np.float64], sub_available: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        """Reconstruct piston-removed phase [rad] from scaled slopes ``u`` [rad].

        ``u`` may be 1-D ``(n_slopes,)`` or 2-D ``(n_samples, n_slopes)``.
        """
        arr = np.asarray(u, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError(f"u must be 1-D or 2-D, got shape {arr.shape}")
        if arr.shape[-1] != self.matrices.n_slopes:
            raise ValueError(
                f"u must have {self.matrices.n_slopes} slope entries, got {arr.shape[-1]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite entries")
        r_mat = self.matrix(sub_available)
        return arr @ r_mat.T

    def noise_propagation(self, sub_available: NDArray[np.bool_] | None = None) -> float:
        """Analytic ``a_NP`` for this reconstructor and dropout pattern [-]."""
        return noise_propagation_coefficient(self.matrix(sub_available))

    def singular_values(
        self, sub_available: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        """Singular values of the phase-side operator ``A`` after row selection."""
        n_sub = self.geom.n_valid_sub
        avail = (
            np.ones(n_sub, dtype=bool)
            if sub_available is None
            else np.asarray(sub_available, dtype=bool).ravel()
        )
        rows = self.matrices.active_rows(avail)
        return np.linalg.svd(self.matrices.a[rows], compute_uv=False)


@dataclass
class ModalReconstructor:
    """Regularised least-squares reconstruction of Zernike coefficients from slopes.

    The interaction ("poke") matrix ``D`` has one column per Zernike mode and
    holds the scaled slopes ``u = pitch * gradient`` that a unit coefficient of
    that mode produces, evaluated **as point samples of the analytic gradient
    at the subaperture centres**. Estimation is
    ``a_hat = regularised_pinv(D) u``.

    This is the standard modal reconstructor (Hardy 1998, *Adaptive Optics for
    Astronomical Telescopes*, OUP, ch. 5; Southwell 1980, Sec. VI discusses
    modal versus zonal estimation). It is the like-for-like baseline for the
    learned reconstructor of :mod:`wavelab.ml`, which maps the same slope
    vector to the same coefficient vector.

    Parameters
    ----------
    geom : SubapertureGeometry
        Lenslet layout.
    noll_indices : sequence of int
        Noll indices to estimate. Piston (``j = 1``) is unobservable from
        slopes and is rejected.
    method, reg
        Regularisation, as in :func:`regularised_pinv`.
    prior_std : ndarray, optional
        Per-mode prior standard deviation [rad]. When given, Tikhonov
        regularisation is applied to the *whitened* coefficients ``a / sigma_a``
        instead of to ``a`` itself, i.e. the penalty becomes
        ``lambda^2 sum_j (a_j / sigma_a_j)^2``. That is the diagonal
        minimum-variance (Bayesian) reconstructor for a diagonal prior, and it
        matters here because the Kolmogorov Zernike spectrum is steeply
        coloured -- a white penalty over-shrinks tilt and under-shrinks the high
        orders. The prior must be estimated on training or validation data
        only. See Hardy 1998 ch. 5 on minimum-variance reconstruction.

    Units
    -----
    Coefficients are in radians of wavefront phase, in the Noll-orthonormal
    basis, so ``sqrt(sum_j a_j^2)`` is the phase RMS over the pupil [rad].
    """

    geom: SubapertureGeometry
    noll_indices: tuple[int, ...] = tuple(range(2, 22))
    method: str = "tsvd"
    reg: float = 1e-6
    prior_std: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        js = [int(j) for j in self.noll_indices]
        if len(js) == 0:
            raise ValueError("noll_indices must contain at least one mode")
        if any(j < 2 for j in js):
            raise ValueError(
                "piston (Noll j = 1) is unobservable from slopes; noll_indices must all be >= 2"
            )
        if len(set(js)) != len(js):
            raise ValueError("noll_indices must not contain duplicates")
        self.noll_indices = tuple(js)
        self.method = _check_method(self.method)
        r = float(self.reg)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"reg must be finite and > 0, got {self.reg!r}")
        self.reg = r

        cx, cy = self.geom.subaperture_centres()
        gx, gy = zernike_gradient_basis(list(self.noll_indices), cx, cy)
        k = self.geom.scaled_slope_factor
        self.interaction: NDArray[np.float64] = np.vstack([gx, gy]) * k
        if self.prior_std is None:
            self._w = np.ones(self.n_modes)
        else:
            w = np.asarray(self.prior_std, dtype=float).ravel()
            if w.shape != (self.n_modes,):
                raise ValueError(
                    f"prior_std must have {self.n_modes} entries, got {w.shape}"
                )
            if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
                raise ValueError("prior_std entries must be finite and > 0")
            self._w = w
        self._interaction_w = self.interaction * self._w
        self._cache: dict[bytes, NDArray[np.float64]] = {}

    @property
    def n_modes(self) -> int:
        """Number of estimated Zernike modes [-]."""
        return len(self.noll_indices)

    @property
    def prior_weights(self) -> NDArray[np.float64]:
        """Per-mode prior standard deviations used to whiten the penalty [rad].

        All ones when no ``prior_std`` was supplied.
        """
        return self._w

    def matrix(self, sub_available: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """Reconstruction matrix ``R`` with ``a_hat = R u``, shape ``(n_modes, n_slopes)``."""
        n_sub = self.geom.n_valid_sub
        avail = (
            np.ones(n_sub, dtype=bool)
            if sub_available is None
            else np.asarray(sub_available, dtype=bool).ravel()
        )
        if avail.shape != (n_sub,):
            raise ValueError(f"sub_available must have shape ({n_sub},), got {avail.shape}")
        key = np.packbits(avail).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rows = np.concatenate([avail, avail])
        d = self._interaction_w[rows]
        r_mat = np.zeros((self.n_modes, 2 * n_sub))
        if d.shape[0] > 0:
            r_mat[:, rows] = self._w[:, None] * regularised_pinv(d, self.method, self.reg)
        self._cache[key] = r_mat
        return r_mat

    def reconstruct(
        self, u: NDArray[np.float64], sub_available: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        """Estimate Zernike coefficients [rad] from scaled slopes ``u`` [rad]."""
        arr = np.asarray(u, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError(f"u must be 1-D or 2-D, got shape {arr.shape}")
        if arr.shape[-1] != self.interaction.shape[0]:
            raise ValueError(
                f"u must have {self.interaction.shape[0]} slope entries, got {arr.shape[-1]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite entries")
        return arr @ self.matrix(sub_available).T

    def forward(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scaled slopes ``u`` [rad] produced by Zernike coefficients [rad]."""
        arr = np.asarray(coeffs, dtype=float)
        if arr.shape[-1] != self.n_modes:
            raise ValueError(f"coeffs must have {self.n_modes} entries, got {arr.shape[-1]}")
        return arr @ self.interaction.T

    def noise_propagation(self, sub_available: NDArray[np.bool_] | None = None) -> float:
        """Mean-square coefficient error per unit ``sigma_u^2`` [-].

        Equal to ``||R||_F^2 / n_modes``. Because the Zernike basis used here is
        orthonormal, ``||R||_F^2`` alone (without the ``/ n_modes``) is the
        total reconstructed phase variance per unit slope-noise variance.
        """
        r_mat = self.matrix(sub_available)
        return float(np.sum(r_mat**2) / r_mat.shape[0])
