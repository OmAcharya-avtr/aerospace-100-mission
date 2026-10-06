"""Log-likelihood ratios for OOK and M-ary PPM over fading optical channels.

Sign convention, used everywhere in this package
------------------------------------------------
    L = log( P(bit = 0 | y) / P(bit = 1 | y) )

with equiprobable bits, so ``L = log p(y | 0) - log p(y | 1)``. ``L > 0``
favours a zero. A hard decision is ``bit_hat = (L < 0)``.

OOK with known channel state (the deterministic core)
-----------------------------------------------------
Under the thermal-limited model of :mod:`softdecode.detection`,
``p(y | b, h) = N(y; a h b, sigma**2)``, so

    L(y | h) = ( a**2 h**2 - 2 a h y ) / ( 2 sigma**2 )                    (1)

affine in ``y`` with slope ``-a h / sigma**2``. This is the textbook linear
LLR for an on-off keyed signal in signal-independent Gaussian noise (Proakis &
Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008, chapter 4: the
log-likelihood ratio of two equal-variance Gaussian hypotheses is affine in
the observation). Equation (1) at ``h = 1`` is the AWGN limit that
``validation/validate_awgn_limit.py`` checks the fading expressions against.

OOK with the channel state unknown (exact, by quadrature)
---------------------------------------------------------
Averaging the ``b = 1`` hypothesis over the fading law gives

    p(y | 1) = integral N(y; a h, sigma**2) f(h) dh
             ~= sum_i w_i N(y; a h_i, sigma**2)                            (2)

with ``(h_i, w_i)`` from :func:`softdecode.channel.amplitude_quadrature`. The
``b = 0`` hypothesis does not involve ``h``, so

    L_exact(y) = -y**2 / (2 sigma**2)
                 - logsumexp_i[ log w_i - (y - a h_i)**2 / (2 sigma**2) ]   (3)

The quadrature in (2) is checked against a Monte Carlo estimate of the same
integral in ``validation/validate_quadrature.py``.

Max-log approximation
---------------------
Replacing the ``logsumexp`` in (3) by a ``max`` is the max-log approximation
(the Laplace-type approximation of the integrand, weights included). Because
``logsumexp >= max`` termwise,

    L_maxlog(y) >= L_exact(y)   for every y,                               (4)

so the max-log error is one-signed for OOK. The same substitution applied to
the symbol sums of a PPM bit LLR is the classical max-log demapper of
Hagenauer & Hoeher (J. Hagenauer and P. Hoeher, "A Viterbi algorithm with
soft-decision outputs and its applications", *Proc. IEEE GLOBECOM 1989*,
1680-1686), whose soft-output formulation is where the ``max*`` / ``logsumexp``
distinction comes from.

M-ary PPM
---------
``M`` unit-duration slots, one pulsed. With known ``h`` the per-symbol metric,
after dropping the ``sum_m y_m**2`` term common to all symbols, is

    lambda_c = a h y_c / sigma**2 - a**2 h**2 / (2 sigma**2)                (5)

and the LLR of the ``k``-th label bit is

    L_k = logsumexp_{c: b_k(c)=0} lambda_c - logsumexp_{c: b_k(c)=1} lambda_c

With ``h`` unknown the same cancellation holds inside the fading average, so

    lambda_c = logsumexp_i[ log w_i + a h_i y_c / sigma**2
                            - a**2 h_i**2 / (2 sigma**2) ]                  (6)

Clipping
--------
Every fixed-point decoder saturates its LLRs. :func:`clip_llr` applies
``sign(L) * min(|L|, L_max)``; its cost in LLR error and, separately, in
decoded bit error rate is measured in
``validation/validate_maxlog_clipping.py``. The two are not the same quantity
and only the second one matters to a user.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = [
    "clip_llr",
    "gray_labels",
    "llr_ook_known_csi",
    "llr_ook_marginal",
    "llr_ook_maxlog",
    "llr_ppm_known_csi",
    "llr_ppm_marginal",
    "llr_ppm_maxlog",
    "rescale_llr",
]


def _as_array(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_sigma(sigma: float) -> float:
    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"sigma must be a finite positive number, got {sigma!r}")
    return s


def _check_amplitude(amplitude: float) -> float:
    a = float(amplitude)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"amplitude must be a finite positive number, got {amplitude!r}")
    return a


def _check_quadrature(nodes, weights) -> tuple[np.ndarray, np.ndarray]:
    h = _as_array("nodes", nodes).ravel()
    w = _as_array("weights", weights).ravel()
    if h.shape != w.shape:
        raise ValueError(f"nodes and weights must have equal size, got {h.size} and {w.size}")
    if h.size == 0:
        raise ValueError("quadrature must have at least one node")
    if np.any(h < 0.0):
        raise ValueError("fading nodes must be non-negative")
    if np.any(w <= 0.0):
        raise ValueError("quadrature weights must be strictly positive")
    return h, w


def llr_ook_known_csi(y, amplitude: float, h, sigma: float = 1.0) -> np.ndarray:
    """Exact OOK LLR with the channel state known, equation (1).

    Parameters
    ----------
    y:
        Matched-filter samples, electrical units.
    amplitude:
        Peak amplitude ``a`` at ``h = 1``, same units as ``y``.
    h:
        Normalised irradiance, dimensionless; broadcast against ``y``.
    sigma:
        Noise standard deviation of one sample.

    Returns
    -------
    LLR ``log P(0)/P(1)``, dimensionless, shape ``broadcast(y, h)``.
    """
    y = _as_array("y", y)
    hh = _as_array("h", h)
    if np.any(hh < 0.0):
        raise ValueError("h must be non-negative")
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    return (a * a * hh * hh - 2.0 * a * hh * y) / (2.0 * s2)


def llr_ook_marginal(y, amplitude: float, nodes, weights, sigma: float = 1.0) -> np.ndarray:
    """Exact OOK LLR with the channel state unknown, equation (3).

    ``nodes``/``weights`` come from
    :func:`softdecode.channel.amplitude_quadrature` or from
    :func:`softdecode.csi.posterior_quadrature`; in the latter case the result
    is the CSI-aware LLR conditioned on the estimate.
    """
    y = _as_array("y", y)
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    h, w = _check_quadrature(nodes, weights)
    terms = np.log(w) - (y[..., None] - a * h) ** 2 / (2.0 * s2)
    return -(y**2) / (2.0 * s2) - special.logsumexp(terms, axis=-1)


def llr_ook_maxlog(y, amplitude: float, nodes, weights, sigma: float = 1.0) -> np.ndarray:
    """Max-log OOK LLR: equation (3) with ``logsumexp`` replaced by ``max``.

    Satisfies ``llr_ook_maxlog >= llr_ook_marginal`` pointwise, equation (4).
    """
    y = _as_array("y", y)
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    h, w = _check_quadrature(nodes, weights)
    terms = np.log(w) - (y[..., None] - a * h) ** 2 / (2.0 * s2)
    return -(y**2) / (2.0 * s2) - np.max(terms, axis=-1)


def gray_labels(order: int) -> np.ndarray:
    """Reflected-binary (Gray) bit labels of ``M`` PPM symbols.

    Returns an ``(M, log2 M)`` array of 0/1, most significant bit first.
    """
    m = int(order)
    if m < 2 or (m & (m - 1)) != 0:
        raise ValueError(f"order must be a power of two and at least 2, got {order!r}")
    bits = int(np.log2(m))
    idx = np.arange(m)
    gray = idx ^ (idx >> 1)
    shifts = np.arange(bits - 1, -1, -1)
    return ((gray[:, None] >> shifts) & 1).astype(np.int8)


def _ppm_bit_llr(symbol_metric: np.ndarray, labels: np.ndarray, use_max: bool) -> np.ndarray:
    """Combine symbol metrics into bit LLRs under the stated labelling."""
    reduce = (lambda x: np.max(x, axis=-1)) if use_max else (
        lambda x: special.logsumexp(x, axis=-1)
    )
    bits = labels.shape[1]
    out = np.empty(symbol_metric.shape[:-1] + (bits,), dtype=float)
    for k in range(bits):
        zero = labels[:, k] == 0
        out[..., k] = reduce(symbol_metric[..., zero]) - reduce(symbol_metric[..., ~zero])
    return out


def _ppm_metric_known(y: np.ndarray, a: float, h: np.ndarray, s2: float) -> np.ndarray:
    return a * h[..., None] * y / s2 - (a * a * h[..., None] ** 2) / (2.0 * s2)


def _ppm_metric_marginal(
    y: np.ndarray, a: float, h: np.ndarray, w: np.ndarray, s2: float, use_max: bool
) -> np.ndarray:
    terms = (
        np.log(w)
        + a * h * y[..., None] / s2
        - (a * a * h * h) / (2.0 * s2)
    )
    return np.max(terms, axis=-1) if use_max else special.logsumexp(terms, axis=-1)


def llr_ppm_known_csi(
    y, amplitude: float, h, sigma: float = 1.0, labels: np.ndarray | None = None
) -> np.ndarray:
    """Exact M-PPM bit LLRs with the channel state known, equation (5).

    Parameters
    ----------
    y:
        Slot samples, shape ``(..., M)``.
    h:
        Normalised irradiance per symbol, shape ``(...)``; one value per PPM
        symbol, constant across its slots.
    labels:
        ``(M, log2 M)`` bit labelling; :func:`gray_labels` by default.

    Returns
    -------
    ``(..., log2 M)`` bit LLRs.
    """
    y = _as_array("y", y)
    order = y.shape[-1]
    lab = gray_labels(order) if labels is None else np.asarray(labels, dtype=np.int8)
    if lab.shape[0] != order:
        raise ValueError(f"labels has {lab.shape[0]} rows, expected {order}")
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    hh = np.asarray(h, dtype=float)
    if np.any(hh < 0.0):
        raise ValueError("h must be non-negative")
    metric = _ppm_metric_known(y, a, hh, s2)
    return _ppm_bit_llr(metric, lab, use_max=False)


def llr_ppm_marginal(
    y, amplitude: float, nodes, weights, sigma: float = 1.0, labels: np.ndarray | None = None
) -> np.ndarray:
    """Exact M-PPM bit LLRs with the channel state unknown, equation (6)."""
    y = _as_array("y", y)
    order = y.shape[-1]
    lab = gray_labels(order) if labels is None else np.asarray(labels, dtype=np.int8)
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    h, w = _check_quadrature(nodes, weights)
    metric = _ppm_metric_marginal(y, a, h, w, s2, use_max=False)
    return _ppm_bit_llr(metric, lab, use_max=False)


def llr_ppm_maxlog(
    y,
    amplitude: float,
    nodes,
    weights,
    sigma: float = 1.0,
    labels: np.ndarray | None = None,
    inner_max: bool = True,
) -> np.ndarray:
    """Max-log M-PPM bit LLRs.

    ``inner_max`` also replaces the fading-average ``logsumexp`` of equation
    (6) by a ``max``; set it to ``False`` to apply max-log only to the symbol
    sums, which is the classical demapper of Hagenauer & Hoeher (1989).
    """
    y = _as_array("y", y)
    order = y.shape[-1]
    lab = gray_labels(order) if labels is None else np.asarray(labels, dtype=np.int8)
    a = _check_amplitude(amplitude)
    s2 = _check_sigma(sigma) ** 2
    h, w = _check_quadrature(nodes, weights)
    metric = _ppm_metric_marginal(y, a, h, w, s2, use_max=bool(inner_max))
    return _ppm_bit_llr(metric, lab, use_max=True)


def clip_llr(llr, limit: float) -> np.ndarray:
    """Saturate LLR magnitude: ``sign(L) * min(|L|, limit)``.

    ``limit`` must be > 0. Zeros stay zero.
    """
    lim = float(limit)
    if not np.isfinite(lim) or lim <= 0.0:
        raise ValueError(f"limit must be a finite positive number, got {limit!r}")
    arr = np.asarray(llr, dtype=float)
    return np.clip(arr, -lim, lim)


def rescale_llr(llr, scale: float) -> np.ndarray:
    """Multiply LLRs by a positive scalar ``scale``.

    A positive rescaling leaves hard decisions and maximum-likelihood
    *block* decoding unchanged (both depend only on the sign pattern and on
    relative weights), but it changes the output of any decoder with a
    nonlinear message update, which is why it is a real competitor to a
    learned corrector. See ``validation/validate_corrector.py``.
    """
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return np.asarray(llr, dtype=float) * s
