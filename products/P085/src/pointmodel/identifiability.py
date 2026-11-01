"""Rank and identifiability: which terms the observed geometry can separate.

The headline capability of this package is here, and it is a refusal.

The problem
-----------
A pointing model is a linear fit with a known design matrix, and the whole
difficulty is conditioning. Nine standard terms are mutually independent as
functions on the sphere, but a pointing run does not sample the sphere: it
samples whatever the schedule, the horizon mask, the keyhole and the available
sources allowed. Over a restricted geometry, pairs and triples of terms become
numerically indistinguishable, and least squares still returns one specific
split with a formal standard error attached to each term. The split is not
wrong because the arithmetic is wrong; it is arbitrary, and nothing in a
least-squares report says so.

Three degeneracies this module finds on this product's own validation
geometries, each with a physical reading:

- **Narrow elevation band.** With elevation confined to a few degrees,
  ``cos(E)``, ``1`` and ``sin(E)`` are nearly the same constant function, so
  the azimuth index ``IA``, the collimation ``CA`` and the non-perpendicularity
  ``NPAE`` collapse into one cross-elevation constant. Measured on a 43-47
  degree band: a near-null direction with weights dominated by ``CA``, ``IA``
  and ``NPAE``. Simultaneously ``1``, ``cos(E)``, ``sin(E)`` and
  ``rho_hat(E)`` collapse in elevation, so ``IE``, ``TF``, ``TS`` and ``RF``
  are one constant between them.
- **Single azimuth.** With azimuth fixed, ``cos(A)`` and ``sin(A)`` are
  constants, so the two azimuth-axis tilt components ``AN`` and ``AW`` enter
  the elevation residual as one number and enter the cross-elevation residual
  as one multiple of ``sin(E)``, which ``NPAE`` also supplies. The rank
  deficiency is exact, not approximate, and it is two.
- **Flexure against refraction, even on good geometry.** ``cos(E)`` and
  ``rho_hat(E)`` are different functions but similar in shape over a practical
  elevation band, so ``TF``, ``TS``, ``IE`` and ``RF`` share a weakly
  determined direction even on a 15-80 degree all-azimuth run. This one is not
  a pathology of a bad schedule: it is a property of the term set, and it is
  why refraction coefficients fitted from pointing data disagree with
  refraction coefficients computed from the atmosphere.

The analysis
------------
Let ``A_w`` be the whitened design matrix, ``S = diag(||column||)`` its column
norms and ``A_n = A_w S^{-1}`` the column-normalised matrix with unit-norm
columns. Column normalisation is what makes the analysis a statement about
geometry rather than about units; it is reported, not hidden. Take the thin SVD

    A_n = U diag(s_1 ... s_9) V^T,   s_1 >= ... >= s_9 >= 0.                (22)

The parameter vector in the normalised space is ``phi = S theta``, so
``A_w theta = A_n phi``.

- A right singular vector ``v_k`` with small ``s_k`` is a direction in
  ``phi``-space that the data barely constrain. The corresponding direction in
  **physical coefficient space** is

      d_k  proportional to  S^{-1} v_k,                                     (23)

  normalised here to unit Euclidean norm. Adding ``c d_k`` to the coefficient
  vector changes every predicted residual by at most ``c s_k`` in the whitened
  metric, so for ``s_k`` at machine precision it changes nothing at all. This
  is the quantity reported as a *near-null direction*, and it is named in terms
  of the physical coefficients.
- The combination the data **do** determine along that direction is

      g_k(theta) = v_k . (S theta),                                         (24)

  with an exact standard error of ``sigma_scale / s_k`` in the normalised
  metric, because the Gram matrix in ``phi``-space is ``V diag(s^2) V^T``. The
  nine ``g_k`` are a complete orthogonal reparameterisation of the model: the
  well-determined ones are what the run measured, and the badly determined ones
  are what it did not. Their units are arcseconds times a column norm, which is
  awkward and is reported with the dimensionless precision ``|g_k| / se_k``
  beside it, that being the number a reader needs.

Thresholds
----------
Two ratios on ``s_k / s_1``, both declared and both arguments:

- ``null_threshold`` (default 1e-3). At or below this, the direction is
  reported as **not identifiable** and the package refuses to split the terms
  involved. 1e-3 corresponds to a condition number of 1000 contributed by that
  direction alone, at which a 1-arcsec residual noise becomes a
  1000-arcsec uncertainty in the corresponding combination.
- ``marginal_threshold`` (default 1e-2). Between the two, the direction is
  reported as **weakly determined**: the split is returned but is flagged, and
  the flexure-against-refraction direction above falls here on good geometry.

Both defaults are policy, not theorems, and the thresholds are printed in
every report alongside the verdict so that a reader can apply their own. The
measured singular-value spectra for the five declared geometries are in
``validation/validate_identifiability_output.txt``; the gap between the good
and the restricted geometries is four to twelve orders of magnitude, so the
verdict is insensitive to the threshold anywhere inside that gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terms import N_TERMS, TERM_NAMES
from .wls import WlsFit

__all__ = [
    "Combination",
    "Direction",
    "IdentifiabilityReport",
    "analyse",
    "analyse_fit",
]

#: Default declared thresholds on ``s_k / s_1``.
DEFAULT_NULL_THRESHOLD = 1e-3
DEFAULT_MARGINAL_THRESHOLD = 1e-2


@dataclass(frozen=True)
class Direction:
    """A near-null direction in physical coefficient space, equation (23).

    Attributes
    ----------
    index
        Position in the descending singular-value ordering, 0-based.
    singular_value
        ``s_k`` of equation (22), dimensionless.
    ratio
        ``s_k / s_1``, dimensionless. This is what the thresholds act on.
    weights
        Length-9 unit-norm direction in coefficient space. Adding any multiple
        of this to the coefficients leaves the predicted residuals unchanged to
        within ``ratio`` of the whitened residual scale.
    verdict
        ``"not identifiable"`` or ``"weakly determined"``.
    """

    index: int
    singular_value: float
    ratio: float
    weights: np.ndarray
    verdict: str

    def dominant(self, count: int = 4) -> list[tuple[str, float]]:
        """The ``count`` largest components, as ``(term name, weight)`` pairs."""
        order = np.argsort(-np.abs(self.weights))[:count]
        return [(TERM_NAMES[int(i)], float(self.weights[int(i)])) for i in order]

    def statement(self, count: int = 4) -> str:
        """One line naming the combination the geometry cannot determine."""
        parts = " ".join(f"{weight:+.3f}*{name}" for name, weight in self.dominant(count))
        return f"{parts}  (s/s1 = {self.ratio:.3e}, {self.verdict})"


@dataclass(frozen=True)
class Combination:
    """An identifiable linear combination, equation (24).

    Attributes
    ----------
    index
        Position in the descending singular-value ordering, 0-based.
    singular_value, ratio
        As in :class:`Direction`.
    weights
        Length-9 weights on the physical coefficients, ``S v_k``, in the units
        that make ``value = weights . theta``.
    unit_weights
        ``weights`` rescaled to unit Euclidean norm, for reading.
    value
        ``g_k(theta_hat)`` of equation (24), arcseconds times a column norm.
    standard_error
        ``sigma_scale / s_k``, same units as ``value``.
    precision
        ``|value| / standard_error``, dimensionless. The number that says
        whether this combination was measured.
    determined
        True when ``ratio`` is above the declared marginal threshold.
    """

    index: int
    singular_value: float
    ratio: float
    weights: np.ndarray
    unit_weights: np.ndarray
    value: float
    standard_error: float
    precision: float
    determined: bool

    def statement(self, count: int = 4) -> str:
        """One line naming the combination and its measured precision."""
        order = np.argsort(-np.abs(self.unit_weights))[:count]
        parts = " ".join(
            f"{float(self.unit_weights[int(i)]):+.3f}*{TERM_NAMES[int(i)]}" for i in order
        )
        return f"{parts}  precision |g|/se = {self.precision:.3g}"


@dataclass(frozen=True)
class IdentifiabilityReport:
    """The rank verdict for one observation geometry.

    Attributes
    ----------
    singular_values
        The nine ``s_k`` of equation (22), descending, dimensionless.
    ratios
        ``s_k / s_1``.
    condition_number
        ``s_1 / s_9``, or ``inf`` if ``s_9`` is zero.
    numerical_rank
        Count of ``s_k`` above ``null_threshold * s_1``.
    separable
        True when ``numerical_rank == 9``, i.e. when the geometry supports a
        per-term split of all nine coefficients at the declared threshold.
    null_directions
        Directions at or below ``null_threshold``: what cannot be determined.
    marginal_directions
        Directions between the two thresholds: weakly determined.
    combinations
        All nine of equation (24), descending in ``s_k``. The first
        ``numerical_rank`` are what the run measured.
    vif
        Length-9 variance inflation factors, dimensionless, from the fit.
    column_scales
        The diagonal of ``S``, needed to read ``weights``.
    null_threshold, marginal_threshold
        The declared ratios actually applied.
    verdict
        ``"separable"``, ``"weakly determined"`` or ``"not separable"``.
    """

    singular_values: np.ndarray
    ratios: np.ndarray
    condition_number: float
    numerical_rank: int
    separable: bool
    null_directions: tuple[Direction, ...]
    marginal_directions: tuple[Direction, ...]
    combinations: tuple[Combination, ...]
    vif: np.ndarray
    column_scales: np.ndarray
    null_threshold: float
    marginal_threshold: float
    verdict: str

    @property
    def unidentifiable_terms(self) -> tuple[str, ...]:
        """Terms carrying a weight of 0.2 or more in any null direction.

        0.2 of a unit-norm nine-vector is a declared reading threshold, not a
        theorem: it is the level at which a term is a material participant in a
        degeneracy rather than a rounding contribution. The full weight vectors
        are in ``null_directions`` and a reader can apply any other cut.
        """
        names: list[str] = []
        for direction in self.null_directions:
            for position in np.flatnonzero(np.abs(direction.weights) >= 0.2):
                name = TERM_NAMES[int(position)]
                if name not in names:
                    names.append(name)
        return tuple(names)


def analyse(
    whitened_design: np.ndarray,
    coefficients: np.ndarray,
    sigma_scale: float = 1.0,
    vif: np.ndarray | None = None,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD,
) -> IdentifiabilityReport:
    """Rank analysis of a whitened design matrix.

    Parameters
    ----------
    whitened_design
        ``(2n, 9)`` design matrix already divided row-wise by ``sigma``,
        dimensionless per arcsecond.
    coefficients
        Length-9 estimate in arcseconds, used only to evaluate equation (24).
    sigma_scale
        Residual scale in the whitened metric, dimensionless. Pass
        ``sqrt(reduced chi-square)`` to propagate the fit's own scaling; 1.0
        means the declared sigma is trusted.
    vif
        Length-9 variance inflation factors if already computed by the fit;
        recomputed from the SVD when omitted.
    null_threshold, marginal_threshold
        Declared ratios on ``s_k / s_1``. Must satisfy
        ``0 < null_threshold <= marginal_threshold < 1``.

    Returns
    -------
    IdentifiabilityReport
    """
    matrix = np.asarray(whitened_design, dtype=float)
    theta = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != N_TERMS:
        raise ValueError(
            f"whitened_design must be (rows, {N_TERMS}), got shape {matrix.shape}"
        )
    if theta.shape != (N_TERMS,):
        raise ValueError(f"coefficients must have shape ({N_TERMS},), got {theta.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("whitened_design contains a non-finite value")
    if not 0.0 < null_threshold <= marginal_threshold < 1.0:
        raise ValueError(
            "thresholds must satisfy 0 < null_threshold <= marginal_threshold < 1; got "
            f"null_threshold={null_threshold}, marginal_threshold={marginal_threshold}"
        )
    if sigma_scale <= 0.0:
        raise ValueError(f"sigma_scale must be positive, got {sigma_scale}")

    column_scales = np.linalg.norm(matrix, axis=0)
    if np.any(column_scales <= 0.0):
        dead = [TERM_NAMES[i] for i in np.flatnonzero(column_scales <= 0.0)]
        raise ValueError(f"design column(s) {dead} are identically zero over these observations")
    normalised = matrix / column_scales
    _u, singular, vt = np.linalg.svd(normalised, full_matrices=False)
    ratios = singular / singular[0]

    if vif is None:
        inverse_sq = np.where(
            singular > 0.0, 1.0 / np.where(singular > 0.0, singular, 1.0) ** 2, np.inf
        )
        vif_values = np.einsum("jk,k->j", vt.T**2, inverse_sq)
    else:
        vif_values = np.asarray(vif, dtype=float)

    scaled_theta = theta * column_scales

    directions_null: list[Direction] = []
    directions_marginal: list[Direction] = []
    combinations: list[Combination] = []
    for position in range(singular.size):
        right = vt[position]
        physical = right / column_scales
        norm = float(np.linalg.norm(physical))
        unit_physical = physical / norm if norm > 0.0 else physical
        ratio = float(ratios[position])
        if ratio <= null_threshold:
            verdict = "not identifiable"
            directions_null.append(
                Direction(position, float(singular[position]), ratio, unit_physical, verdict)
            )
        elif ratio <= marginal_threshold:
            verdict = "weakly determined"
            directions_marginal.append(
                Direction(position, float(singular[position]), ratio, unit_physical, verdict)
            )

        weights = right * column_scales
        value = float(right @ scaled_theta)
        error = (
            float(sigma_scale / singular[position])
            if singular[position] > 0.0
            else float("inf")
        )
        weight_norm = float(np.linalg.norm(weights))
        combinations.append(
            Combination(
                index=position,
                singular_value=float(singular[position]),
                ratio=ratio,
                weights=weights,
                unit_weights=weights / weight_norm if weight_norm > 0.0 else weights,
                value=value,
                standard_error=error,
                precision=float(abs(value) / error) if np.isfinite(error) and error > 0 else 0.0,
                determined=ratio > marginal_threshold,
            )
        )

    rank = int(np.count_nonzero(ratios > null_threshold))
    if rank < N_TERMS:
        verdict = "not separable"
    elif directions_marginal:
        verdict = "weakly determined"
    else:
        verdict = "separable"
    return IdentifiabilityReport(
        singular_values=singular,
        ratios=ratios,
        condition_number=float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
        numerical_rank=rank,
        separable=rank == N_TERMS,
        null_directions=tuple(directions_null),
        marginal_directions=tuple(directions_marginal),
        combinations=tuple(combinations),
        vif=vif_values,
        column_scales=column_scales,
        null_threshold=float(null_threshold),
        marginal_threshold=float(marginal_threshold),
        verdict=verdict,
    )


def analyse_fit(
    design: np.ndarray,
    result: WlsFit,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD,
) -> IdentifiabilityReport:
    """Rank analysis for a completed :class:`pointmodel.wls.WlsFit`.

    ``design`` is the unwhitened ``(2n, 9)`` matrix; the fit's own weights are
    applied here so that the analysis is of the system the fit actually solved.
    The residual scale is taken as ``sqrt(reduced chi-square)`` when the fit
    used ``error_model="scaled"`` and as 1.0 when it trusted the declared
    sigma, so that the standard errors on the combinations are consistent with
    the intervals the fit reported.
    """
    matrix = np.asarray(design, dtype=float)
    whitened = matrix * np.sqrt(result.weights)[:, None]
    scale = (
        float(np.sqrt(result.reduced_chi_square))
        if result.error_model == "scaled" and np.isfinite(result.reduced_chi_square)
        else 1.0
    )
    return analyse(
        whitened,
        result.coefficients,
        sigma_scale=scale,
        vif=result.vif,
        null_threshold=null_threshold,
        marginal_threshold=marginal_threshold,
    )
