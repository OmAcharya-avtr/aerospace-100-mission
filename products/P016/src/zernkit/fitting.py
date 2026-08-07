"""Least-squares fitting of a sampled wavefront to Zernike coefficients.

Model
-----
Given samples ``w_p`` of a wavefront at pupil points ``(x_p, y_p)`` normalised
so the pupil edge is at ``x^2 + y^2 = 1``, solve for the coefficient vector
``a`` of::

    w(x, y) ~= sum_k a_k Z_k(x, y)

by ordinary least squares, i.e. ``a = argmin ||A a - w||_2`` with design matrix
``A[p, k] = Z_k(x_p, y_p)``. Solved with :func:`numpy.linalg.lstsq`
(SVD-based, minimum-norm solution when ``A`` is rank deficient).

Why not projection integrals? Orthonormality (Noll 1976, Eq. 3) holds under a
continuous ``1/pi`` area weight. On a finite, non-uniform, or clipped sample
set the modes are only *approximately* orthogonal, so the projection integral
and the least-squares solution differ. Least squares is the honest estimator
for sampled data and is what this module implements; the returned
:attr:`FitResult.condition_number` tells you how badly the sampling has
degraded the basis (1.0 would be perfectly orthonormal columns after scaling).

**Policy for points outside the unit disc (explicit).** Zernike polynomials are
orthogonal only on the unit disc and grow rapidly outside it, so a sample at
``rho > 1`` is not merely unweighted -- it can dominate the fit. The
``outside`` argument selects one of:

``"raise"`` (default)
    Any sample with ``rho > 1 + tol`` raises :class:`ValueError` naming the
    offending count and the worst radius. Fails loudly; recommended.
``"drop"``
    Samples with ``rho > 1 + tol`` are silently excluded from the fit and
    counted in :attr:`FitResult.n_dropped`. Use when your wavefront array is a
    square grid and the corners are simply outside the pupil.
``"extrapolate"``
    All samples are kept and the polynomials are evaluated outside the disc.
    Provided for completeness only; the result is not an orthogonal
    decomposition and small errors near the rim are amplified.

``tol`` (default ``1e-9``) absorbs floating-point round-off in ``rho`` for
points intended to lie exactly on the rim. Non-finite samples (``nan``/``inf``
in ``w``) are always rejected with :class:`ValueError`, because ``lstsq``
would otherwise return an all-``nan`` solution silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indexing import nm_to_noll, nm_to_osa, noll_to_nm, osa_to_nm, validate_nm
from .polynomials import zernike_cartesian

__all__ = ["FitResult", "mode_list", "zernike_design_matrix", "fit_wavefront"]

_OUTSIDE_POLICIES = ("raise", "drop", "extrapolate")


@dataclass(frozen=True)
class FitResult:
    """Outcome of a Zernike least-squares wavefront fit.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Fitted coefficients, one per mode, in the same unit as the input
        wavefront (waves, radians, metres -- the library never converts).
    indices : list of (int, int)
        The ``(n, m)`` pairs in coefficient order.
    noll_indices : list of int
        Noll index (1-based) of each coefficient.
    osa_indices : list of int
        OSA/ANSI index (0-based) of each coefficient.
    residual : numpy.ndarray
        ``w_used - A a`` at the samples actually used, input units.
    residual_rms : float
        RMS of ``residual``, input units.
    input_rms : float
        RMS of the used input samples about zero, input units.
    condition_number : float
        2-norm condition number of the design matrix; large values (say
        ``> 1e3``) mean the sampling has made the modes nearly degenerate and
        individual coefficients are unreliable.
    n_used, n_dropped : int
        Sample counts kept and discarded by the ``outside`` policy.
    normalized : bool
        Whether the Noll/ANSI orthonormal scaling was used.
    outside : str
        The outside-disc policy that was applied.
    """

    coefficients: NDArray[np.float64]
    indices: list[tuple[int, int]]
    noll_indices: list[int]
    osa_indices: list[int]
    residual: NDArray[np.float64] = field(repr=False)
    residual_rms: float
    input_rms: float
    condition_number: float
    n_used: int
    n_dropped: int
    normalized: bool
    outside: str

    def coefficient(self, n: int, m: int) -> float:
        """Return the fitted coefficient for mode ``(n, m)``.

        Raises
        ------
        KeyError
            If that mode was not part of the fit.
        """
        validate_nm(n, m)
        try:
            return float(self.coefficients[self.indices.index((n, m))])
        except ValueError as exc:  # pragma: no cover - message path
            raise KeyError(f"mode (n={n}, m={m}) was not included in this fit") from exc

    @property
    def variance_explained(self) -> float:
        """Fraction of the sampled wavefront variance captured by the fit.

        ``1 - (residual_rms / input_rms)^2``; returns ``1.0`` when the input is
        identically zero.
        """
        if self.input_rms == 0.0:
            return 1.0
        return 1.0 - (self.residual_rms / self.input_rms) ** 2


def mode_list(n_modes: int, indexing: str = "noll") -> list[tuple[int, int]]:
    """First ``n_modes`` ``(n, m)`` pairs in the requested single-index ordering.

    Parameters
    ----------
    n_modes : int
        How many modes, counted from the first index of the convention
        (Noll ``j = 1``, OSA/ANSI ``j = 0``) -- both start at piston.
    indexing : {"noll", "osa"}, optional
        Ordering convention. ``"ansi"`` is accepted as a synonym of ``"osa"``.

    Returns
    -------
    list of (int, int)
        Modes in index order.
    """
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
        raise TypeError(f"n_modes must be an integer, got {n_modes!r}")
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    key = indexing.lower()
    if key == "noll":
        return [noll_to_nm(j) for j in range(1, int(n_modes) + 1)]
    if key in ("osa", "ansi"):
        return [osa_to_nm(j) for j in range(int(n_modes))]
    raise ValueError(f"indexing must be 'noll' or 'osa'/'ansi', got {indexing!r}")


def zernike_design_matrix(
    indices: list[tuple[int, int]],
    x: ArrayLike,
    y: ArrayLike,
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Design matrix ``A[p, k] = Z_k(x_p, y_p)`` for the given modes.

    Parameters
    ----------
    indices : list of (int, int)
        Modes as ``(n, m)`` pairs, in coefficient order.
    x, y : array_like
        Pupil coordinates normalised to pupil radius; flattened internally.
    normalized : bool, optional
        Noll/ANSI orthonormal scaling (default True).

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_points, n_modes)``, dimensionless.
    """
    if len(indices) == 0:
        raise ValueError("indices must contain at least one (n, m) pair")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same size, got {x_arr.size} and {y_arr.size}")
    mat = np.empty((x_arr.size, len(indices)), dtype=np.float64)
    for col, (n, m) in enumerate(indices):
        mat[:, col] = zernike_cartesian(n, m, x_arr, y_arr, normalized=normalized)
    return mat


def fit_wavefront(
    x: ArrayLike,
    y: ArrayLike,
    wavefront: ArrayLike,
    n_modes: int | None = None,
    *,
    indices: list[tuple[int, int]] | None = None,
    indexing: str = "noll",
    normalized: bool = True,
    outside: str = "raise",
    tol: float = 1e-9,
    rcond: float | None = None,
) -> FitResult:
    """Fit sampled wavefront values to Zernike coefficients by least squares.

    Parameters
    ----------
    x, y : array_like
        Pupil coordinates normalised so the pupil edge is at
        ``x^2 + y^2 = 1`` (dimensionless). Flattened internally; must be the
        same size as ``wavefront``.
    wavefront : array_like
        Sampled wavefront values. Units are arbitrary and are carried straight
        through to the coefficients (waves in, waves out).
    n_modes : int, optional
        Number of modes counted from the first index of ``indexing``. Ignored
        if ``indices`` is given; one of the two must be supplied.
    indices : list of (int, int), optional
        Explicit mode list, overriding ``n_modes``/``indexing``.
    indexing : {"noll", "osa"}, optional
        Ordering used to expand ``n_modes`` (default ``"noll"``).
    normalized : bool, optional
        Noll/ANSI orthonormal scaling (default True). With ``False`` the
        coefficients are in the unnormalised Born & Wolf convention and are
        **not** comparable to normalised ones.
    outside : {"raise", "drop", "extrapolate"}, optional
        Policy for samples with ``rho > 1 + tol``; see the module docstring.
    tol : float, optional
        Radial tolerance for "on the rim" (default ``1e-9``).
    rcond : float, optional
        Passed to :func:`numpy.linalg.lstsq` for singular-value truncation.
        Default ``None`` uses the NumPy machine-precision cutoff.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        On mismatched sizes, non-finite wavefront values, an unknown policy,
        out-of-disc samples under ``outside="raise"``, fewer usable samples
        than modes, or all samples dropped.
    """
    if outside not in _OUTSIDE_POLICIES:
        raise ValueError(f"outside must be one of {_OUTSIDE_POLICIES}, got {outside!r}")

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    w_arr = np.asarray(wavefront, dtype=np.float64).ravel()
    if not (x_arr.size == y_arr.size == w_arr.size):
        raise ValueError(
            "x, y and wavefront must have the same number of samples, got "
            f"{x_arr.size}, {y_arr.size}, {w_arr.size}"
        )
    if x_arr.size == 0:
        raise ValueError("no samples supplied")
    if not np.all(np.isfinite(w_arr)):
        raise ValueError(
            "wavefront contains non-finite values (nan/inf); mask them out before fitting "
            "or the least-squares solution is undefined"
        )
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("x and y must be finite")

    if indices is None:
        if n_modes is None:
            raise ValueError("supply either n_modes or an explicit indices list")
        indices = mode_list(n_modes, indexing=indexing)
    else:
        if len(indices) == 0:
            raise ValueError("indices must contain at least one (n, m) pair")
        for n, m in indices:
            validate_nm(n, m)

    rho = np.hypot(x_arr, y_arr)
    outside_mask = rho > 1.0 + tol
    n_outside = int(np.count_nonzero(outside_mask))
    if outside == "raise" and n_outside:
        raise ValueError(
            f"{n_outside} of {rho.size} samples lie outside the unit disc "
            f"(max rho = {rho.max():.6g}); Zernike polynomials are orthogonal only on "
            "rho <= 1. Rescale your coordinates, or pass outside='drop' to exclude them, "
            "or outside='extrapolate' to keep them (not an orthogonal decomposition)."
        )
    if outside == "drop" and n_outside:
        keep = ~outside_mask
        x_arr, y_arr, w_arr = x_arr[keep], y_arr[keep], w_arr[keep]
        n_dropped = n_outside
    else:
        n_dropped = 0

    if x_arr.size == 0:
        raise ValueError("all samples lie outside the unit disc; nothing to fit")
    if x_arr.size < len(indices):
        raise ValueError(
            f"under-determined fit: {x_arr.size} usable samples for {len(indices)} modes"
        )

    design = zernike_design_matrix(indices, x_arr, y_arr, normalized=normalized)
    coeffs, _, _, svals = np.linalg.lstsq(design, w_arr, rcond=rcond)
    cond = float(svals[0] / svals[-1]) if svals.size and svals[-1] > 0 else float("inf")

    residual = w_arr - design @ coeffs
    return FitResult(
        coefficients=coeffs,
        indices=list(indices),
        noll_indices=[nm_to_noll(n, m) for n, m in indices],
        osa_indices=[nm_to_osa(n, m) for n, m in indices],
        residual=residual,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        input_rms=float(np.sqrt(np.mean(w_arr**2))),
        condition_number=cond,
        n_used=int(x_arr.size),
        n_dropped=int(n_dropped),
        normalized=bool(normalized),
        outside=outside,
    )
