"""Motion features for the learned estimator. Twenty-four per record and secondary.

The feature set deliberately **contains the observability information**. The
excitation shares of equation (9), the dominant rotation axis, the normalised
Gramian structure and the information scale are all inputs. That matters for
the honesty of this product's headline comparison: if the learned estimator
produces a confident estimate for a degree of freedom the record cannot
support, it is not because it was starved of the evidence that the degree of
freedom is unsupported. It had every number the analytic method had.

What is NOT a feature, and why
------------------------------
The pseudo-inverse solution ``G^+ b`` is not a feature. Supplying it would make
the learned estimator a wrapper around the analytic baseline and the comparison
meaningless; it would also hide the shrinkage that is the only mechanism by
which a learned estimator can beat an unbiased solution on a short record. The
raw sufficient statistics ``b`` and ``G`` are supplied instead, in a scaled
form, so the model has the same information and must find its own map.

Units
-----
``b_scaled`` is in radians. The six normalised Gramian components, the three
excitation shares, the entropy and the three dominant-axis components are
dimensionless. ``log10_information`` is the base-10 logarithm of ``W / s^2`` in
s^-2 rad^-2. ``log10_samples`` is dimensionless, ``duration_s`` in seconds, the
three rate statistics in rad/s and the two residual statistics dimensionless
(they are divided by the pair sigma).
"""

from __future__ import annotations

import numpy as np

from .batch import normal_equation_rhs
from .observability import ObservabilityReport, excitation_spectrum, gramian
from .record import AlignmentRecord, residuals

__all__ = ["FEATURE_NAMES", "build", "build_population"]

FEATURE_NAMES: tuple[str, ...] = (
    "b_scaled_x",
    "b_scaled_y",
    "b_scaled_z",
    "gnorm_xx",
    "gnorm_yy",
    "gnorm_zz",
    "gnorm_xy",
    "gnorm_xz",
    "gnorm_yz",
    "log10_information",
    "log10_samples",
    "duration_s",
    "rate_mean",
    "rate_max",
    "rate_std",
    "excitation_p0",
    "excitation_p1",
    "excitation_p2",
    "excitation_entropy",
    "dominant_axis_x",
    "dominant_axis_y",
    "dominant_axis_z",
    "residual_rms_scaled",
    "residual_along_axis_scaled",
)


def build(
    record: AlignmentRecord,
    name: str,
    observability: ObservabilityReport | None = None,
) -> np.ndarray:
    """Feature vector for one secondary of one record, shape ``(24,)``.

    Parameters
    ----------
    record:
        The measurements.
    name:
        Secondary sensor name.
    observability:
        Optional pre-computed verdict. Passing it avoids recomputing the
        eigendecomposition; the result is identical either way.

    Returns
    -------
    ndarray
        Shape ``(24,)``, float64, in the order of :data:`FEATURE_NAMES`.
    """
    pair_sigma = record.pair_sigma(name)
    shares, _, energy = excitation_spectrum(record)
    scale = energy / (pair_sigma * pair_sigma)
    matrix = gramian(record, name)
    rhs = normal_equation_rhs(record, name)
    if observability is None:
        axis_source = _dominant_axis(record)
        entropy = _entropy(shares)
    else:
        axis_source = observability.dominant_axis
        entropy = observability.excitation_entropy
    normalised = matrix / scale if scale > 0.0 else np.zeros((3, 3))
    b_scaled = rhs / scale if scale > 0.0 else np.zeros(3)
    rates = record.reference_rates
    speed = np.linalg.norm(rates, axis=1)
    residual = residuals(record, name)
    residual_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))) / pair_sigma
    along = float(
        np.sqrt(np.mean(np.square(residual @ axis_source)))
    ) / pair_sigma
    return np.array(
        [
            b_scaled[0],
            b_scaled[1],
            b_scaled[2],
            normalised[0, 0],
            normalised[1, 1],
            normalised[2, 2],
            normalised[0, 1],
            normalised[0, 2],
            normalised[1, 2],
            np.log10(max(scale, 1.0e-300)),
            np.log10(float(len(record))),
            record.duration_s,
            float(np.mean(speed)),
            float(np.max(speed)),
            float(np.std(speed)),
            shares[0],
            shares[1],
            shares[2],
            entropy,
            axis_source[0],
            axis_source[1],
            axis_source[2],
            residual_rms,
            along,
        ],
        dtype=float,
    )


def _dominant_axis(record: AlignmentRecord) -> np.ndarray:
    _, vectors, _ = excitation_spectrum(record)
    axis = vectors[:, 0]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis


def _entropy(shares: np.ndarray) -> float:
    clipped = np.clip(shares, 1.0e-300, None)
    return float(-np.sum(clipped * np.log2(clipped)))


def build_population(
    population: list,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features and targets for a population from :func:`aligncal.dataset.training_population`.

    Returns
    -------
    (features, targets, profiles)
        ``features`` shape ``(m, 24)``, ``targets`` shape ``(m, 3)`` in radians,
        ``profiles`` the generating excitation profile of each row, so a score
        can be broken down by excitation richness.
    """
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    profiles: list[str] = []
    for generated in population:
        for name in generated.record.names:
            rows.append(build(generated.record, name))
            targets.append(np.asarray(generated.truth[name], dtype=float))
            profiles.append(generated.profile)
    if not rows:
        raise ValueError("build_population() was given an empty population")
    return np.vstack(rows), np.vstack(targets), profiles
