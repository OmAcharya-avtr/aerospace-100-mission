r"""Classical process-noise adaptation — the baselines, implemented first.

Three schemes, in increasing order of sophistication. All of them adapt a
**scalar multiplier** ``s`` on a nominal process-noise matrix ``Q₀``, so they
are directly comparable with each other and with the learned adapter in
:mod:`navbench.ai`. Adapting a scalar rather than a full matrix is a deliberate
restriction: it keeps ``Q`` positive semi-definite and structurally correct by
construction, and it is the form in which the question "is my ``Q`` too small?"
is usually asked in practice.

1. :class:`FixedQ` — no adaptation
----------------------------------
Constant ``s``. Tuned offline by a grid search on a *training* set of
trajectories (``navbench.bench.tune_fixed_scale``) and then frozen. This is the
"hand-tuned Q" baseline the mission requires.

2. :class:`CovarianceMatching` — Mehra's covariance matching
------------------------------------------------------------
The innovation covariance predicted by the filter is **affine** in the process
noise scale for one step:

.. math::
    S(s) = H\left(F P^{+} F^{\mathsf T} + s Q_0\right)H^{\mathsf T} + R
         = \underbrace{H F P^{+}F^{\mathsf T}H^{\mathsf T} + R}_{S_0}
           + s\, H Q_0 H^{\mathsf T}

Matching the trace of ``S(s)`` to the empirical innovation covariance
``Ĉ_ν = (1/N)Σ ν ν^{\mathsf T}`` over a sliding window of length ``N`` gives a
closed-form estimate

.. math::
    \hat s = \frac{\operatorname{tr}\hat C_\nu - \operatorname{tr}S_0}
                  {\operatorname{tr}(H Q_0 H^{\mathsf T})}

clipped to ``[s_min, s_max]`` and smoothed with a first-order filter. This is
the covariance-matching idea of Mehra, R. K. (1970), "On the identification of
variances and adaptive Kalman filtering", *IEEE Transactions on Automatic
Control* **15**(2), 175–184, and Mehra (1972), "Approaches to adaptive
filtering", *IEEE Transactions on Automatic Control* **17**(5), 693–698,
specialised to a scalar unknown so that the estimate is closed form.

**Stated approximation.** ``Ĉ_ν`` is accumulated over a window during which
``P⁺`` changes; the affine relation above holds exactly only for a single step.
The error is second order in the variation of ``P⁺`` across the window and
shrinks as the filter approaches steady state.

3. :func:`mohamed_schwarz_q` — innovation-based adaptive estimation (IAE)
-------------------------------------------------------------------------
The full-matrix estimator

.. math:: \hat Q = K \hat C_\nu K^{\mathsf T}

of Mohamed, A. H. and Schwarz, K. P. (1999), "Adaptive Kalman filtering for
INS/GPS", *Journal of Geodesy* **73**(4), 193–203. It is provided as a utility
and as a *scalar-projected* adapter (:class:`IaeScaleAdapter`) that reports
``s = tr(Q̂)/tr(Q₀)``, so it can be scored on the same axis as the others.
Its known weakness — visible in the benchmark — is that ``K Ĉ_ν Kᵀ`` has rank
at most ``m``, so for ``n > m`` it cannot see the unobserved directions of
``Q``, and its estimate is biased low when ``R`` dominates ``S``.

Statistical floor common to all of them
---------------------------------------
``Ĉ_ν`` from ``N`` samples of an ``m``-dimensional innovation has relative
standard error ``≈ √(2/(N m))`` on its trace. With ``N = 30`` and ``m = 2``
that is 18 %, so no window-based scheme — classical or learned — can resolve a
``Q`` error smaller than about 20 % from a 30-sample window. This bound is
quoted in the model card and is the reason the benchmark reports the *spread*
of the estimates, not only their mean.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "QAdapter",
    "FixedQ",
    "CovarianceMatching",
    "IaeScaleAdapter",
    "mohamed_schwarz_q",
    "innovation_window_features",
]


class QAdapter:
    """Interface for a process-noise scale adapter.

    Subclasses maintain whatever internal statistics they need and expose the
    multiplier to apply to the nominal ``Q₀`` on the **next** prediction.
    """

    name: str = "base"

    def reset(self) -> None:
        """Clear internal state. Called once per trajectory."""

    def observe(
        self,
        innovation: ArrayLike,
        innovation_cov: ArrayLike,
        gain: ArrayLike,
        *,
        f: ArrayLike | None = None,
        h: ArrayLike | None = None,
        p_post: ArrayLike | None = None,
        q0: ArrayLike | None = None,
        r: ArrayLike | None = None,
    ) -> None:
        """Absorb one measurement update. Extra arguments are model context."""

    @property
    def scale(self) -> float:
        """Multiplier on ``Q₀`` for the next prediction (dimensionless, > 0)."""
        raise NotImplementedError

    @property
    def confidence(self) -> tuple[float, float] | None:
        """Optional interval on :attr:`scale`; ``None`` when not available."""
        return None


@dataclass
class FixedQ(QAdapter):
    """Constant process-noise scale (the hand-tuned baseline)."""

    value: float = 1.0
    name: str = "fixed"

    def __post_init__(self) -> None:
        if self.value <= 0.0:
            raise ValueError(f"value must be > 0, got {self.value}")

    @property
    def scale(self) -> float:
        """The frozen multiplier."""
        return float(self.value)


@dataclass
class CovarianceMatching(QAdapter):
    """Mehra-style covariance matching on a scalar ``Q`` multiplier.

    Parameters
    ----------
    window : int
        Sliding-window length ``N`` for ``Ĉ_ν``.
    smoothing : float
        First-order smoothing factor in ``(0, 1]``; the reported scale is
        ``s ← (1−λ)s + λ ŝ``. ``1`` means no smoothing.
    s_min, s_max : float
        Clipping bounds on the estimate.
    initial : float
        Scale used until the window is full.
    """

    window: int = 30
    smoothing: float = 0.3
    s_min: float = 1.0e-3
    s_max: float = 1.0e3
    initial: float = 1.0
    name: str = "covariance-matching"
    _buf: deque = field(default_factory=deque, repr=False)
    _scale: float = field(default=1.0, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {self.smoothing}")
        if not 0.0 < self.s_min < self.s_max:
            raise ValueError(f"need 0 < s_min < s_max, got {self.s_min}, {self.s_max}")
        self._buf = deque(maxlen=int(self.window))
        self._scale = float(self.initial)

    def reset(self) -> None:
        """Clear the innovation window and restore the initial scale."""
        self._buf.clear()
        self._scale = float(self.initial)

    def observe(
        self,
        innovation: ArrayLike,
        innovation_cov: ArrayLike,
        gain: ArrayLike,
        *,
        f: ArrayLike | None = None,
        h: ArrayLike | None = None,
        p_post: ArrayLike | None = None,
        q0: ArrayLike | None = None,
        r: ArrayLike | None = None,
    ) -> None:
        """Absorb one innovation; requires ``f``, ``h``, ``p_post``, ``q0``, ``r``."""
        del gain, innovation_cov
        nu = np.asarray(innovation, dtype=float).reshape(-1)
        self._buf.append(nu)
        if len(self._buf) < self._buf.maxlen:
            return
        if f is None or h is None or p_post is None or q0 is None or r is None:
            raise ValueError(
                "CovarianceMatching needs f, h, p_post, q0 and r to form the affine "
                "innovation-covariance model"
            )
        ff = np.atleast_2d(np.asarray(f, dtype=float))
        hh = np.atleast_2d(np.asarray(h, dtype=float))
        pp = np.atleast_2d(np.asarray(p_post, dtype=float))
        qq = np.atleast_2d(np.asarray(q0, dtype=float))
        rr = np.atleast_2d(np.asarray(r, dtype=float))
        c_nu = np.mean([np.outer(v, v) for v in self._buf], axis=0)
        s0 = hh @ (ff @ pp @ ff.T) @ hh.T + rr
        denom = float(np.trace(hh @ qq @ hh.T))
        if denom <= 0.0:
            return
        s_hat = (float(np.trace(c_nu)) - float(np.trace(s0))) / denom
        s_hat = float(np.clip(s_hat, self.s_min, self.s_max))
        self._scale = (1.0 - self.smoothing) * self._scale + self.smoothing * s_hat

    @property
    def scale(self) -> float:
        """Current smoothed multiplier."""
        return float(np.clip(self._scale, self.s_min, self.s_max))


def mohamed_schwarz_q(
    innovations: ArrayLike, gain: ArrayLike
) -> NDArray[np.float64]:
    r"""Innovation-based adaptive estimate ``Q̂ = K Ĉ_ν Kᵀ``.

    Parameters
    ----------
    innovations : array_like, shape (N, m)
        Sliding window of innovations.
    gain : array_like, shape (n, m)
        Most recent Kalman gain.

    Returns
    -------
    ndarray, shape (n, n)

    Notes
    -----
    Source: Mohamed & Schwarz (1999), *Journal of Geodesy* **73**(4), 193–203.
    Rank of the result is at most ``m``; see the module docstring for the
    consequence.
    """
    v = np.atleast_2d(np.asarray(innovations, dtype=float))
    k = np.atleast_2d(np.asarray(gain, dtype=float))
    if v.shape[0] < 1:
        raise ValueError("need at least one innovation")
    c = np.einsum("ij,ik->jk", v, v) / v.shape[0]
    return k @ c @ k.T


@dataclass
class IaeScaleAdapter(QAdapter):
    """Scalar projection of the Mohamed–Schwarz IAE estimate.

    Reports ``s = tr(K Ĉ_ν Kᵀ) / tr(Q₀)``, clipped and smoothed like
    :class:`CovarianceMatching` so the two are compared on equal terms.
    """

    window: int = 30
    smoothing: float = 0.3
    s_min: float = 1.0e-3
    s_max: float = 1.0e3
    initial: float = 1.0
    name: str = "iae"
    _buf: deque = field(default_factory=deque, repr=False)
    _scale: float = field(default=1.0, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {self.smoothing}")
        self._buf = deque(maxlen=int(self.window))
        self._scale = float(self.initial)

    def reset(self) -> None:
        """Clear the window and restore the initial scale."""
        self._buf.clear()
        self._scale = float(self.initial)

    def observe(
        self,
        innovation: ArrayLike,
        innovation_cov: ArrayLike,
        gain: ArrayLike,
        *,
        f: ArrayLike | None = None,
        h: ArrayLike | None = None,
        p_post: ArrayLike | None = None,
        q0: ArrayLike | None = None,
        r: ArrayLike | None = None,
    ) -> None:
        """Absorb one innovation; requires ``q0`` and the current ``gain``."""
        del innovation_cov, f, h, p_post, r
        self._buf.append(np.asarray(innovation, dtype=float).reshape(-1))
        if len(self._buf) < self._buf.maxlen or q0 is None:
            return
        qq = np.atleast_2d(np.asarray(q0, dtype=float))
        tr_q0 = float(np.trace(qq))
        if tr_q0 <= 0.0:
            return
        q_hat = mohamed_schwarz_q(np.array(self._buf), gain)
        s_hat = float(np.clip(float(np.trace(q_hat)) / tr_q0, self.s_min, self.s_max))
        self._scale = (1.0 - self.smoothing) * self._scale + self.smoothing * s_hat

    @property
    def scale(self) -> float:
        """Current smoothed multiplier."""
        return float(np.clip(self._scale, self.s_min, self.s_max))


def innovation_window_features(
    innovations: ArrayLike, innovation_covs: ArrayLike, applied_scale: float
) -> NDArray[np.float64]:
    r"""Feature vector for the learned adapter — computable online, no truth.

    Given a window of ``N`` innovations ``ν_j`` and their filter-predicted
    covariances ``S_j``, and the process-noise multiplier currently applied:

    ==  ==========================================================
    0   ``log(mean NIS / m)`` — 0 when the filter is consistent
    1   ``log(tr Ĉ_ν / mean tr S)`` — the covariance-matching ratio
    2   lag-1 autocorrelation of the whitened innovations
    3   lag-2 autocorrelation of the whitened innovations
    4   fraction of whitened components with ``|·| > 2``
    5   ``log10`` of the applied process-noise multiplier
    6   ``log(N)`` — tells the model how much evidence it has
    ==  ==========================================================

    Whitening uses ``S_j^{-1/2}ν_j`` via a Cholesky solve, so feature 0 is
    exactly the mean NIS. All features are dimensionless.
    """
    v = np.atleast_2d(np.asarray(innovations, dtype=float))
    s = np.asarray(innovation_covs, dtype=float)
    if s.ndim != 3 or s.shape[0] != v.shape[0]:
        raise ValueError(f"innovation_covs shape {s.shape} incompatible with {v.shape}")
    if applied_scale <= 0.0:
        raise ValueError(f"applied_scale must be > 0, got {applied_scale}")
    n, m = v.shape
    white = np.zeros_like(v)
    nis = np.zeros(n)
    tr_s = np.zeros(n)
    for j in range(n):
        lo = np.linalg.cholesky(0.5 * (s[j] + s[j].T))
        white[j] = np.linalg.solve(lo, v[j])
        nis[j] = float(np.dot(white[j], white[j]))
        tr_s[j] = float(np.trace(s[j]))
    c_nu_tr = float(np.mean(np.sum(v * v, axis=1)))
    eps = 1e-30
    f0 = np.log(max(float(np.mean(nis)) / m, eps))
    f1 = np.log(max(c_nu_tr, eps) / max(float(np.mean(tr_s)), eps))

    def _acf(lag: int) -> float:
        if n <= lag + 1:
            return 0.0
        a, b = white[:-lag], white[lag:]
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return float(np.sum(a * b) / den) if den > 0.0 else 0.0

    f2, f3 = _acf(1), _acf(2)
    f4 = float(np.mean(np.abs(white) > 2.0))
    f5 = float(np.log10(applied_scale))
    f6 = float(np.log(n))
    return np.array([f0, f1, f2, f3, f4, f5, f6])
