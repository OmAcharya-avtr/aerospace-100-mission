"""M-ary pulse-position modulation over the Poisson counting channel.

PPM is the modulation of choice for a photon-starved optical link because it
puts all of a symbol's energy into one of ``M`` slots, buying bits per photon
at the cost of bandwidth. The deep-space case and the modulation/code selection
that follows from it are treated in J. Hamkins and B. Moision, "Multipulse
Pulse-Position Modulation on Discrete Memoryless Channels", *IPN Progress
Report* 42-161, Jet Propulsion Laboratory, 15 May 2005. This module implements
the single-pulse (``M``-ary, one pulsed slot per symbol) case.

**Channel.** A symbol occupies ``M`` slots. The receiver reports a count per
slot. Conditioned on the signalled slot ``j``, the counts are independent
Poisson:

    K_i ~ Poisson(n_b + n_d + n_s * [i == j]),   i = 0 .. M-1              (P1)

with ``n_s`` signal counts in the pulsed slot, ``n_b`` background counts per
slot and ``n_d`` dark counts per slot, all in detected counts per slot. Write
``n_0 = n_b + n_d`` for the non-signalled slot mean.

**The soft metric, derived.** The likelihood of a count vector ``k`` given slot
``j`` is, from (P1),

    P(k | j) = prod_i exp(-(n_0 + n_s [i==j])) (n_0 + n_s [i==j])^k_i / k_i!

Taking logs and dropping everything independent of ``j``,

    ln P(k | j) = C(k) - n_s + k_j ln((n_0 + n_s) / n_0),                  (P2)

    C(k) = sum_i [ -n_0 + k_i ln n_0 - ln k_i! ].

So **the whole soft-decision metric for the Poisson PPM channel is a single
number per slot, the count scaled by one constant**:

    Lambda_i = k_i * ln(1 + n_s / n_0).                                    (P3)

Three consequences worth stating because they are routinely got wrong:

1. The maximum-likelihood decision is ``argmax_i k_i``. Thresholding, or
   weighting counts non-linearly, cannot beat it.
2. The *scale* ``ln(1 + n_s/n_0)`` does not change the hard decision but is
   required for correct soft output: it is what makes the posterior, and hence
   every bit LLR handed to a decoder, correctly calibrated. A decoder fed raw
   counts as LLRs is mis-scaled by exactly that factor.
3. At ``n_0 -> 0`` the scale diverges. That is not a numerical problem to be
   clipped away; it is the statement that a single count in a background-free
   channel is **certain** evidence. The background-free channel is an erasure
   channel, handled by :func:`erasure_probability` and
   :func:`erasure_channel_probabilities`, not by the LLR path.

**Hard-decision error probability.** With ties among the maximum broken
uniformly at random, the exact probability of a correct decision is

    P_c = sum_k p_s(k) * sum_{t=0}^{M-1} C(M-1, t) F_0(k-1)^(M-1-t) p_0(k)^t / (t+1)
                                                                          (P4)

where ``p_s`` is the Poisson pmf at mean ``n_0 + n_s``, ``p_0`` and ``F_0`` the
pmf and cdf at mean ``n_0``, ``t`` the number of non-signalled slots tied at
the same count ``k``, and ``1/(t+1)`` the chance the random tie-break picks the
right one. (P4) is exact up to the truncation of the sum over ``k``, which
:func:`symbol_error_probability` chooses from the Poisson tail and reports.

At ``n_0 = 0``, (P4) collapses to ``P_e = exp(-n_s) (M-1)/M``: either at least
one signal count arrives and the decision is certain, or none does and the
receiver guesses. This is the known-answer case used in the tests.

Units: all means are dimensionless counts per slot. Nothing in this module
models dead time or afterpulsing; a count vector produced by
:mod:`photoncount.simulate` through a detector with either will not satisfy
(P1), and that is the point of :mod:`photoncount.correction`.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = [
    "PPMConfig",
    "bit_llrs",
    "bits_per_symbol",
    "erasure_channel_probabilities",
    "erasure_probability",
    "hard_decision",
    "sample_counts",
    "slot_means",
    "slot_metric_scale",
    "slot_metrics",
    "symbol_error_probability",
    "symbol_error_probability_mc",
    "symbol_log_posterior",
]


class PPMConfig:
    """Validated PPM channel parameters.

    Parameters
    ----------
    order:
        ``M >= 2``, the number of slots per symbol (-).
    signal_counts:
        ``n_s > 0``, mean detected counts in the pulsed slot (-).
    background_counts:
        ``n_b >= 0``, mean detected background counts per slot (-).
    dark_counts:
        ``n_d >= 0``, mean detected dark counts per slot (-).
    """

    __slots__ = ("background_counts", "dark_counts", "order", "signal_counts")

    def __init__(
        self,
        order: int,
        signal_counts: float,
        background_counts: float = 0.0,
        dark_counts: float = 0.0,
    ) -> None:
        m = int(order)
        if m < 2:
            raise ValueError(f"order must be >= 2, got {order!r}")
        ns = float(signal_counts)
        nb = float(background_counts)
        nd = float(dark_counts)
        for name, val, allow_zero in (
            ("signal_counts", ns, False),
            ("background_counts", nb, True),
            ("dark_counts", nd, True),
        ):
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val!r}")
            if val < 0.0 or (val == 0.0 and not allow_zero):
                raise ValueError(
                    f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {val!r}"
                )
        self.order = m
        self.signal_counts = ns
        self.background_counts = nb
        self.dark_counts = nd

    @property
    def noise_counts(self) -> float:
        """``n_0 = n_b + n_d``, the non-signalled slot mean (-)."""
        return self.background_counts + self.dark_counts

    @property
    def is_background_free(self) -> bool:
        """True when ``n_0 == 0``, i.e. the channel is an erasure channel."""
        return self.noise_counts == 0.0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PPMConfig(order={self.order!r}, signal_counts={self.signal_counts!r}, "
            f"background_counts={self.background_counts!r}, dark_counts={self.dark_counts!r})"
        )


def bits_per_symbol(order: int) -> float:
    """``log2(M)`` bits per PPM symbol. Not required to be an integer."""
    m = int(order)
    if m < 2:
        raise ValueError(f"order must be >= 2, got {order!r}")
    return float(np.log2(m))


def slot_means(config: PPMConfig, signalled_slot: int) -> np.ndarray:
    """The ``M`` Poisson means of (P1) for a given signalled slot (-)."""
    j = int(signalled_slot)
    if not 0 <= j < config.order:
        raise ValueError(f"signalled_slot must be in [0, {config.order}), got {j!r}")
    means = np.full(config.order, config.noise_counts, dtype=float)
    means[j] += config.signal_counts
    return means


def sample_counts(
    config: PPMConfig,
    signalled_slots: np.ndarray | int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw slot counts for one or many symbols. Shape ``(n_symbols, M)``, int.

    Deterministic for a given ``rng`` state. Models (P1) exactly: an ideal
    counter with no dead time.
    """
    slots = np.atleast_1d(np.asarray(signalled_slots, dtype=int))
    if np.any(slots < 0) or np.any(slots >= config.order):
        raise ValueError(f"signalled_slots must all be in [0, {config.order})")
    means = np.full((slots.size, config.order), config.noise_counts, dtype=float)
    means[np.arange(slots.size), slots] += config.signal_counts
    return rng.poisson(means)


def slot_metric_scale(config: PPMConfig) -> float:
    """``ln(1 + n_s / n_0)`` from (P3) (nats per count).

    Raises ``ValueError`` on a background-free channel, where the scale is
    infinite and the erasure-channel description applies instead.
    """
    if config.is_background_free:
        raise ValueError(
            "slot_metric_scale is infinite for a background-free channel "
            "(n_b + n_d == 0); use erasure_channel_probabilities instead"
        )
    return float(np.log1p(config.signal_counts / config.noise_counts))


def slot_metrics(counts: np.ndarray, config: PPMConfig) -> np.ndarray:
    """Per-slot soft metric (P3), ``Lambda_i = k_i ln(1 + n_s/n_0)``, nats.

    Same shape as ``counts``. This is the sufficient statistic: everything a
    decoder can use about a slot is its count times this one scale.
    """
    k = np.asarray(counts, dtype=float)
    if np.any(k < 0):
        raise ValueError("counts must be >= 0")
    return k * slot_metric_scale(config)


def symbol_log_posterior(counts: np.ndarray, config: PPMConfig) -> np.ndarray:
    """Log posterior over the ``M`` symbols, assuming equiprobable symbols.

    ``counts`` has shape ``(..., M)``; the return has the same shape and each
    row log-sum-exps to 0. From (P2), the posterior depends on the counts only
    through (P3), so this is a softmax of the slot metrics.
    """
    k = np.asarray(counts, dtype=float)
    if k.shape[-1] != config.order:
        raise ValueError(f"counts last axis must be {config.order}, got {k.shape[-1]}")
    metrics = slot_metrics(k, config)
    return metrics - special.logsumexp(metrics, axis=-1, keepdims=True)


def hard_decision(counts: np.ndarray, config: PPMConfig, rng: np.random.Generator) -> np.ndarray:
    """Maximum-likelihood slot decision ``argmax_i k_i``, ties broken uniformly.

    ``counts`` has shape ``(..., M)``; returns integer slot indices of shape
    ``counts.shape[:-1]``. The tie-break uses ``rng`` and is therefore
    deterministic for a given state; it matters, because at low ``n_s`` most
    symbols are all-zero and every slot is tied.
    """
    k = np.asarray(counts)
    if k.shape[-1] != config.order:
        raise ValueError(f"counts last axis must be {config.order}, got {k.shape[-1]}")
    return _argmax_random_tie(k, rng)


def _argmax_random_tie(k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    maxima = k.max(axis=-1, keepdims=True)
    is_max = k == maxima
    keys = rng.random(k.shape)
    keys = np.where(is_max, keys, -1.0)
    return np.argmax(keys, axis=-1)


def erasure_probability(config: PPMConfig) -> float:
    """``P(all M slot counts are zero)`` (-).

    ``exp(-(n_s + M n_0))``. On a background-free channel this is the classical
    PPM erasure probability ``exp(-n_s)``: the only way to learn nothing is for
    the signal slot to produce no count.
    """
    return float(np.exp(-(config.signal_counts + config.order * config.noise_counts)))


def erasure_channel_probabilities(config: PPMConfig) -> dict[str, float]:
    """The background-free channel as an ``M``-ary erasure channel.

    Returns ``{"erasure": exp(-n_s), "error": 0.0, "correct": 1 - exp(-n_s)}``.
    Raises ``ValueError`` if ``n_0 > 0``, because then a false count can occur
    in a non-signalled slot and the channel is no longer an erasure channel.
    """
    if not config.is_background_free:
        raise ValueError(
            "erasure_channel_probabilities requires a background-free channel "
            f"(n_b + n_d == 0), got n_0 = {config.noise_counts!r}"
        )
    p_er = float(np.exp(-config.signal_counts))
    return {"erasure": p_er, "error": 0.0, "correct": 1.0 - p_er}


def _tail_truncation(mean: float, relative: float = 1e-14) -> int:
    """Smallest ``K`` with ``P(Poisson(mean) > K) < relative``."""
    k = int(stats.poisson.isf(relative, max(mean, 1e-12)))
    return max(k, 1)


def symbol_error_probability(
    config: PPMConfig,
    truncation: int | None = None,
) -> dict[str, float]:
    """Exact hard-decision symbol error probability (P4), with its truncation.

    Returns ``{"error": P_e, "correct": P_c, "truncation": K,
    "tail_mass": P(K_signal > K)}``. The tail mass is the bound on the
    truncation error and is reported so it is never assumed.
    """
    m = config.order
    n0 = config.noise_counts
    ns = config.signal_counts
    k_max = _tail_truncation(n0 + ns) if truncation is None else int(truncation)
    if k_max < 1:
        raise ValueError(f"truncation must be >= 1, got {k_max!r}")
    k = np.arange(0, k_max + 1)
    p_s = stats.poisson.pmf(k, n0 + ns)
    if n0 == 0.0:
        # A background-free slot is a point mass at zero: pmf is 1 at k = 0,
        # and P(K_0 <= k - 1) is 0 at k = 0 and 1 for every k >= 1.
        p_0 = np.where(k == 0, 1.0, 0.0)
        f_0_below = np.where(k >= 1, 1.0, 0.0)
    else:
        p_0 = stats.poisson.pmf(k, n0)
        f_0_below = stats.poisson.cdf(k - 1, n0)
    t = np.arange(0, m)
    log_binom = special.gammaln(m) - special.gammaln(t + 1.0) - special.gammaln(m - t)
    # A probability of zero raised to a positive power is zero, but to the
    # power zero is one. Carrying -inf through the arithmetic produces
    # 0 * -inf = nan, so the zero factors are tracked as a mask instead and
    # the logs are filled with 0 where they are undefined.
    valid_f = f_0_below > 0.0
    valid_0 = p_0 > 0.0
    log_f = np.where(valid_f, np.log(np.where(valid_f, f_0_below, 1.0)), 0.0)
    log_p0 = np.where(valid_0, np.log(np.where(valid_0, p_0, 1.0)), 0.0)
    exp_f = (m - 1 - t)[None, :]
    exp_0 = t[None, :]
    terms = log_binom[None, :] + exp_f * log_f[:, None] + exp_0 * log_p0[:, None]
    vanishes = ((exp_f > 0) & ~valid_f[:, None]) | ((exp_0 > 0) & ~valid_0[:, None])
    weights = np.where(vanishes, 0.0, np.exp(terms) / (t + 1.0)[None, :])
    p_correct = float(np.sum(p_s * weights.sum(axis=1)))
    tail = float(stats.poisson.sf(k_max, n0 + ns))
    return {
        "error": 1.0 - p_correct,
        "correct": p_correct,
        "truncation": float(k_max),
        "tail_mass": tail,
    }


def symbol_error_probability_mc(
    config: PPMConfig,
    n_symbols: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Monte Carlo symbol error rate, for checking (P4). Returns rate and SE.

    Keys ``error``, ``standard_error``, ``n_symbols``. The standard error is the
    binomial ``sqrt(p(1-p)/n)``; a disagreement with (P4) is a defect only if it
    exceeds a few of these.
    """
    n = int(n_symbols)
    if n < 1:
        raise ValueError(f"n_symbols must be >= 1, got {n!r}")
    sent = rng.integers(0, config.order, size=n)
    counts = sample_counts(config, sent, rng)
    decided = _argmax_random_tie(counts, rng)
    err = float(np.mean(decided != sent))
    return {
        "error": err,
        "standard_error": float(np.sqrt(max(err * (1.0 - err), 0.0) / n)),
        "n_symbols": float(n),
    }


def bit_llrs(counts: np.ndarray, config: PPMConfig) -> np.ndarray:
    """Bit log-likelihood ratios for a natural-binary PPM mapping, nats.

    Requires ``M`` to be a power of two. Symbol ``j`` carries ``log2(M)`` bits,
    most significant first, as the binary expansion of ``j``. The LLR of bit
    ``l`` is

        L_l = ln( sum_{j: bit_l(j)=0} P(j|k) / sum_{j: bit_l(j)=1} P(j|k) )

    so a **positive** LLR favours bit 0. Shape ``counts.shape[:-1] + (log2 M,)``.
    """
    m = config.order
    b = int(round(np.log2(m)))
    if 2**b != m:
        raise ValueError(f"bit_llrs requires a power-of-two order, got {m!r}")
    log_post = symbol_log_posterior(counts, config)
    symbols = np.arange(m)
    out = np.empty(log_post.shape[:-1] + (b,), dtype=float)
    for pos in range(b):
        bit = (symbols >> (b - 1 - pos)) & 1
        zero = log_post[..., bit == 0]
        one = log_post[..., bit == 1]
        out[..., pos] = special.logsumexp(zero, axis=-1) - special.logsumexp(one, axis=-1)
    return out
