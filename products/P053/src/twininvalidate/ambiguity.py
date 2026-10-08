"""Twin invalidation against asset fault: why this monitor cannot tell them apart.

A residual monitor sees one thing: the residual. The residual is driven by the
*mismatch* between the model the filter carries and the process that produced
the measurements. It carries no information about which side of that mismatch
moved.

This module makes the claim concrete rather than rhetorical. Take a constant
offset ``b`` in the measurement path, appearing at sample ``onset``, and
realise it two ways:

**World A, asset fault.** The sensor develops a bias. The asset's measurement
becomes ``y_k = C x_k + b + v_k`` and the twin keeps its declared offset
``d = 0``, so

    e_k = y_k - (C x_hat_k + 0) = C (x_k - x_hat_k) + b + v_k.

**World B, twin invalidation.** The sensor is fine and the asset is unchanged,
but the twin's declared offset is revised to ``d = -b`` -- a re-identification
that went the wrong way, or a configuration error. Then
``y_k = C x_k + v_k`` and

    e_k = y_k - (C x_hat_k - b) = C (x_k - x_hat_k) + b + v_k.

The two expressions are the same function of the same quantities. The state
recursions are also the same: ``x`` is driven by the same ``A``, ``B``, ``u``
and ``w`` in both worlds, and ``x_hat`` is corrected by the same ``e``. So for
identical noise realisations the two residual streams are **algebraically
identical** -- not merely statistically similar, and not similar on average,
but the same number at every sample.

They are not identical bit for bit, because the two worlds add ``b`` at
different points in the expression and floating-point addition is not
associative. The measured largest absolute difference over the
default 40-run, 1200-sample pair is 6.17e-14 against residuals of order 1, a
few hundred units in the last place of a double. :func:`max_absolute_difference` returns that
number and ``tests/test_ambiguity.py`` asserts it stays below 1e-10. The
distinction matters because quoting "bitwise identical" would be a claim this
code does not support; quoting 1.1e-13 is a claim it does.

Consequence, stated plainly: an alarm from this package means *the twin and the
asset no longer agree*. It does not mean the asset is faulty, and it does not
mean the twin is wrong. Attribution needs information this monitor does not
have -- a second, independently-instrumented measurement path, a known-good
reference manoeuvre, a maintenance record. Anything in a tool of this kind that
claims to attribute cause from the residual alone is claiming something the
residual cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .twin import LinearGaussianTwin, reference_excitation, reference_twin


@dataclass(frozen=True)
class PairedStreams:
    """The two residual streams of the two worlds, with their shared settings.

    Attributes
    ----------
    asset_fault:
        Residual stream for World A (sensor bias on the asset), shape
        ``(n_runs, n_samples)``, dimensionless.
    twin_invalid:
        Residual stream for World B (wrong declared offset on the twin), same
        shape.
    onset:
        Sample at which the offset appears.
    offset:
        The offset ``b`` in radians.
    """

    asset_fault: np.ndarray
    twin_invalid: np.ndarray
    onset: int
    offset: float


def paired_streams(
    offset: float = 0.02,
    onset: int = 400,
    n_runs: int = 40,
    n_samples: int = 1200,
    seed: int = 53400,
    burn_in: int = 200,
    twin: LinearGaussianTwin | None = None,
) -> PairedStreams:
    """Simulate both worlds on one shared noise realisation.

    Parameters
    ----------
    offset:
        Measurement offset ``b`` in radians. The default, 0.02 rad, is twice
        the declared measurement-noise standard deviation and produces a
        measured post-onset residual mean of 0.81 standard deviations (the
        filter absorbs part of the offset into its state estimate, which is
        why the shift is 0.81 and not 0.02/sqrt(S) = 1.85).
    onset:
        Sample at which the offset appears, relative to sample 0 of the
        returned streams.
    n_runs, n_samples:
        Batch size.
    seed:
        Seed for ``numpy.random.default_rng``.
    burn_in:
        Samples run before sample 0.
    twin:
        Declared twin; defaults to :func:`twininvalidate.twin.reference_twin`.

    Returns
    -------
    A :class:`PairedStreams`.
    """
    if onset < 0 or onset >= n_samples:
        raise ValueError(f"onset must lie in [0, {n_samples}), got {onset}")
    if n_runs <= 0 or n_samples <= 0:
        raise ValueError("n_runs and n_samples must be positive")
    tw = reference_twin() if twin is None else twin
    filt = tw.steady_state()
    total = burn_in + n_samples
    rng = np.random.default_rng(seed)
    u = reference_excitation(total, dt=tw.dt)

    active = np.concatenate([np.zeros(burn_in), np.zeros(n_samples)])
    active[burn_in + onset :] = 1.0

    n = tw.n_states
    chol_q = np.linalg.cholesky(tw.Q)
    sigma_v = float(np.sqrt(tw.R[0, 0]))
    sqrt_s = float(np.sqrt(filt.S))
    c_row = tw.C[0]
    k_col = filt.K[:, 0]

    xa = np.zeros((n_runs, n))
    xha = np.zeros((n_runs, n))
    xb = np.zeros((n_runs, n))
    xhb = np.zeros((n_runs, n))
    za = np.empty((n_runs, n_samples))
    zb = np.empty((n_runs, n_samples))

    for t in range(total):
        v = sigma_v * rng.standard_normal(n_runs)
        w = rng.standard_normal((n_runs, n)) @ chol_q.T
        b_now = offset * active[t]

        # World A: offset is in the asset's measurement; twin declares 0.
        ea = (xa @ c_row + b_now + v) - xha @ c_row
        # World B: asset is clean; twin's declared offset is -b.
        eb = (xb @ c_row + v) - (xhb @ c_row - b_now)

        if t >= burn_in:
            za[:, t - burn_in] = ea / sqrt_s
            zb[:, t - burn_in] = eb / sqrt_s

        xa = xa @ tw.A.T + u[t] @ tw.B.T + w
        xha = xha @ tw.A.T + u[t] @ tw.B.T + np.outer(ea, k_col)
        xb = xb @ tw.A.T + u[t] @ tw.B.T + w
        xhb = xhb @ tw.A.T + u[t] @ tw.B.T + np.outer(eb, k_col)

    return PairedStreams(asset_fault=za, twin_invalid=zb, onset=onset, offset=float(offset))


def max_absolute_difference(pair: PairedStreams) -> float:
    """Largest absolute difference between the two worlds' residuals.

    For the construction of :func:`paired_streams` the two streams are
    algebraically identical, so this returns a floating-point rounding
    residual, of order 1e-13. A value of order 1 would mean the two worlds are
    distinguishable from the residual alone, which would make the README's
    attribution disclaimer too strong rather than too weak -- so this is
    measured, not assumed.
    """
    return float(np.max(np.abs(pair.asset_fault - pair.twin_invalid)))
