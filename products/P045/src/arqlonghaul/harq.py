"""Hybrid ARQ: type-I chase combining and type-II incremental redundancy.

The question this module exists to answer is the crossover: on a link where the
round trip costs D channel symbol times, is it cheaper to send a high-rate
frame and retransmit when it fails, or to pay for redundancy up front and
mostly not retransmit?  The answer is a number, it depends on D, and it is
computed here three ways that are compared against each other.

Code model
----------
A frame carries ``k`` information symbols.  After ``m`` transmissions the
receiver holds a word of ``n_m`` symbols and a bounded-distance decoder that
corrects up to ``t_m`` symbol errors.  The frame is in error iff more than
``t_m`` of the ``n_m`` symbols are wrong:

    P_F(n, t, p) = 1 - sum_{i=0}^{t} C(n, i) p^i (1-p)^(n-i)                (7)

Equation (7) is the standard block error probability of a bounded-distance
decoder on a binary symmetric channel (S. Lin and D. J. Costello, Jr., *Error
Control Coding*, 2nd ed., Prentice Hall, 2004).  The correcting power is taken
from the Singleton bound, ``d <= n - k + 1``, hence

    t_m = floor(alpha (n_m - k) / 2),   alpha in (0, 1]                     (8)

with ``alpha`` an explicit code-family efficiency: ``alpha = 1`` is an
MDS-like, Reed-Solomon-equivalent rate-compatible family, which is the best any
family can do, and ``alpha < 1`` scales it down to a realistic one.  No
particular standardised code is claimed.  ``alpha`` is an input, it is reported
with every number computed from it, and the README says plainly that the
crossover's location depends on it.

Two schemes
-----------
``type_i`` (chase combining): every transmission is the same ``n_1``-symbol
word.  The receiver combines ``m`` copies coherently, so the effective symbol
energy is ``m`` times one copy's and the per-symbol error probability falls:

    p_b(m) = Q(sqrt(2 m Es/N0))                                            (9)

The code never gets stronger, only the channel does.  Modelled as independent
decoding trials at the combined SNR; the real combined words are nested and
correlated, which this ignores, and the README lists it as a limitation.

``type_ii`` (incremental redundancy): the first transmission is ``n_1`` symbols
at rate ``k/n_1``; each later transmission adds ``delta`` new parity symbols, so
``n_m = n_1 + (m-1) delta`` and ``t_m`` grows by equation (8) while the
per-symbol error probability stays put.  The code gets stronger, the channel
does not.  Symbol errors accumulate across the received word, so the decoding
attempts at rounds 1..m are *not* independent; this module computes both the
independent-round approximation and the exact nested answer, and
``validation/validate_harq.py`` measures the difference.

Throughput accounting
---------------------
One HARQ process, no pipelining: after each transmission the sender waits D
symbol times for feedback before it may send again.  Then

    E[elapsed] = sum_{m=1}^{M} q_{m-1} (inc_m + D)                        (10)
    goodput    = k (1 - q_M) / E[elapsed]                                  (11)

where ``q_m`` is the probability the frame is still undecoded after ``m``
transmissions (``q_0 = 1``) and ``inc_m`` is the symbols sent in round ``m``.
Goodput is information symbols per channel symbol time, dimensionless in [0, 1].
``q_M`` is the residual frame error rate handed to the outer ARQ layer and is
reported alongside, because a goodput figure with an unreported residual is not
an answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .channel import bpsk_ber

__all__ = [
    "block_fer",
    "singleton_t",
    "HarqConfig",
    "HarqResult",
    "harq_goodput",
    "harq_goodput_exact",
    "harq_simulate",
    "optimal_first_rate",
    "crossover_rtt",
    "schedule_metrics",
]


def block_fer(n: int, t: int, p: float) -> float:
    """Block error probability of a bounded-distance decoder, equation (7).

    Args:
        n: Codeword length, symbols.
        t: Guaranteed error-correcting power, symbols.
        p: Per-symbol error probability, dimensionless in [0, 1].

    Returns:
        Probability the decoder fails, dimensionless in [0, 1].

    Raises:
        ValueError: if ``n <= 0``, ``t < 0``, ``t > n`` or ``p`` is out of range.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if t > n:
        raise ValueError(f"t ({t}) cannot exceed n ({n})")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return float(binom.sf(t, n, p))


def singleton_t(n: int, k: int, alpha: float = 1.0) -> int:
    """Correctable symbol errors from the Singleton bound, equation (8).

    Args:
        n: Codeword length, symbols.
        k: Information symbols.
        alpha: Code-family efficiency, dimensionless in (0, 1].  1.0 is the
            MDS limit.

    Returns:
        ``floor(alpha (n - k) / 2)``, symbols, >= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return int(math.floor(alpha * (n - k) / 2.0))


@dataclass(frozen=True)
class HarqConfig:
    """One HARQ configuration.

    Attributes:
        k: Information symbols per frame.
        n1: Length of the first transmission, symbols; ``>= k``.
        delta: Symbols added by each later transmission (type-II only).
        max_rounds: Maximum transmissions per frame, ``M >= 1``.
        esn0_db: Per-symbol energy to noise density ratio of one copy, dB.
        rtt_symbols: Feedback latency D, symbol times.
        alpha: Code-family efficiency of equation (8).
        scheme: ``"type_i"`` or ``"type_ii"``.
    """

    k: int
    n1: int
    delta: int = 0
    max_rounds: int = 4
    esn0_db: float = 4.0
    rtt_symbols: float = 0.0
    alpha: float = 1.0
    scheme: str = "type_ii"

    def __post_init__(self) -> None:
        if self.scheme not in ("type_i", "type_ii"):
            raise ValueError(
                f"scheme must be 'type_i' or 'type_ii', got {self.scheme!r}"
            )
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if self.n1 < self.k:
            raise ValueError(f"n1 ({self.n1}) must be >= k ({self.k})")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.rtt_symbols < 0:
            raise ValueError(f"rtt_symbols must be >= 0, got {self.rtt_symbols}")
        if self.scheme == "type_ii" and self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    @property
    def first_rate(self) -> float:
        """Code rate of the first transmission, ``k/n1``, dimensionless."""
        return self.k / self.n1

    def length_after(self, m: int) -> int:
        """Accumulated codeword length after ``m`` transmissions, symbols."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if self.scheme == "type_i":
            return self.n1
        return self.n1 + (m - 1) * self.delta

    def increment(self, m: int) -> int:
        """Symbols transmitted in round ``m``."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if self.scheme == "type_i":
            return self.n1
        return self.n1 if m == 1 else self.delta

    def t_after(self, m: int) -> int:
        """Correctable errors after ``m`` transmissions, equation (8)."""
        return singleton_t(self.length_after(m), self.k, self.alpha)

    def symbol_ber(self, m: int) -> float:
        """Per-symbol error probability relevant to round ``m``'s decode.

        Type-I: equation (9), the combined SNR of ``m`` copies.  Type-II: the
        single-copy BER, unchanged, because IR adds parity rather than energy.
        """
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if self.scheme == "type_i":
            return bpsk_ber(self.esn0_db + 10.0 * math.log10(m))
        return bpsk_ber(self.esn0_db)

    def total_symbols(self) -> int:
        """Symbols sent if every round is used, symbols."""
        return sum(self.increment(m) for m in range(1, self.max_rounds + 1))


@dataclass(frozen=True)
class HarqResult:
    """Goodput and reliability of one HARQ configuration.

    Attributes:
        goodput: Information symbols delivered per channel symbol time,
            dimensionless in [0, 1].
        residual_fer: Probability a frame is still undecoded after
            ``max_rounds`` transmissions, dimensionless.
        mean_rounds: Mean transmissions per frame, dimensionless.
        mean_symbols: Mean channel symbols spent per frame, excluding feedback
            waits, symbols.
        elapsed_symbols: Mean elapsed symbol times per frame including feedback
            waits, equation (10).
        q: ``q_1 .. q_M``, the residual error probability after each round.
        method: Which computation produced this, for provenance.
    """

    goodput: float
    residual_fer: float
    mean_rounds: float
    mean_symbols: float
    elapsed_symbols: float
    q: tuple[float, ...]
    method: str

    def as_dict(self) -> dict[str, float | str]:
        """Flat dict of the scalar fields."""
        return {
            "method": self.method,
            "goodput": self.goodput,
            "residual_fer": self.residual_fer,
            "mean_rounds": self.mean_rounds,
            "mean_symbols": self.mean_symbols,
            "elapsed_symbols": self.elapsed_symbols,
        }


def _assemble(cfg: HarqConfig, q: list[float], method: str) -> HarqResult:
    """Build a :class:`HarqResult` from ``q_1..q_M`` via equations (10)-(11)."""
    m_max = cfg.max_rounds
    q_prev = [1.0] + q[:-1]
    elapsed = sum(
        q_prev[m - 1] * (cfg.increment(m) + cfg.rtt_symbols)
        for m in range(1, m_max + 1)
    )
    symbols = sum(q_prev[m - 1] * cfg.increment(m) for m in range(1, m_max + 1))
    rounds = sum(q_prev[m - 1] for m in range(1, m_max + 1))
    residual = q[-1]
    goodput = cfg.k * (1.0 - residual) / elapsed if elapsed > 0 else 0.0
    return HarqResult(
        goodput=goodput,
        residual_fer=residual,
        mean_rounds=rounds,
        mean_symbols=symbols,
        elapsed_symbols=elapsed,
        q=tuple(q),
        method=method,
    )


def harq_goodput(cfg: HarqConfig) -> HarqResult:
    """Goodput under the independent-round approximation.

    For type-I this is exact under the stated independent-trials model, because
    each round is modelled as a fresh decode:

        q_m = prod_{j=1}^{m} P_F(n_1, t_1, p_b(j))

    For type-II it is the approximation ``q_m = P_F(n_m, t_m, p_b)``, which
    ignores that a frame reaching round ``m`` is already known to have failed
    rounds 1..m-1.  Because ``{E_m > t_m}`` does not imply ``{E_j > t_j}`` for
    ``j < m``, this *overstates* the residual error rate; :func:`harq_goodput_exact`
    computes the nested answer and ``validation/validate_harq.py`` reports the
    gap.
    """
    q: list[float] = []
    if cfg.scheme == "type_i":
        running = 1.0
        for m in range(1, cfg.max_rounds + 1):
            running *= block_fer(cfg.n1, cfg.t_after(m), cfg.symbol_ber(m))
            q.append(running)
        return _assemble(cfg, q, "closed_form_type_i")
    p = cfg.symbol_ber(1)
    for m in range(1, cfg.max_rounds + 1):
        q.append(block_fer(cfg.length_after(m), cfg.t_after(m), p))
    q = list(np.minimum.accumulate(q))
    return _assemble(cfg, q, "closed_form_independent_rounds")


def harq_goodput_exact(cfg: HarqConfig) -> HarqResult:
    """Exact goodput for type-II IR by forward dynamic programming.

    Carries the sub-probability distribution of the accumulated symbol error
    count over frames that have *not* yet decoded.  After round ``m``, states
    with error count ``<= t_m`` are absorbed as successes and removed; the
    remaining mass is convolved with ``Binomial(delta, p)`` for the next round.
    Exact to floating-point accumulation, with cost ``O(M n_M delta)``.

    For type-I the independent-trials model is the definition of the scheme
    here, so this function returns :func:`harq_goodput` unchanged and labels it
    as such.
    """
    if cfg.scheme == "type_i":
        base = harq_goodput(cfg)
        return HarqResult(**{**base.__dict__, "method": "exact_type_i_same_as_closed"})
    p = cfg.symbol_ber(1)
    # dist[j] = P(accumulated errors == j and not yet decoded)
    dist = binom.pmf(np.arange(cfg.n1 + 1), cfg.n1, p)
    q: list[float] = []
    for m in range(1, cfg.max_rounds + 1):
        if m > 1:
            inc = binom.pmf(np.arange(cfg.delta + 1), cfg.delta, p)
            dist = np.convolve(dist, inc)
        t_m = cfg.t_after(m)
        if t_m >= 0:
            dist = dist.copy()
            dist[: min(t_m + 1, dist.size)] = 0.0
        q.append(float(dist.sum()))
    return _assemble(cfg, q, "exact_nested_dp")


def harq_simulate(
    cfg: HarqConfig, n_frames: int, rng: np.random.Generator
) -> HarqResult:
    """Monte Carlo over ``n_frames`` frames, as an independent check.

    Draws the symbol error count per round as a binomial and applies the same
    decoding rule the analytic paths use.  This exists to catch an error in the
    analytic bookkeeping, not to produce the published figure; the published
    figure is the exact DP, which the Monte Carlo is compared against in
    ``validation/validate_harq.py``.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    p1 = cfg.symbol_ber(1)
    rounds_used = np.zeros(n_frames, dtype=np.int64)
    symbols = np.zeros(n_frames, dtype=np.float64)
    failed = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        errs = 0
        ok = False
        used = 0
        spent = 0.0
        for m in range(1, cfg.max_rounds + 1):
            used = m
            spent += cfg.increment(m)
            if cfg.scheme == "type_i":
                errs = int(rng.binomial(cfg.n1, cfg.symbol_ber(m)))
                if errs <= cfg.t_after(m):
                    ok = True
                    break
            else:
                errs += int(rng.binomial(cfg.increment(m), p1))
                if errs <= cfg.t_after(m):
                    ok = True
                    break
        rounds_used[i] = used
        symbols[i] = spent
        failed[i] = not ok
    mean_rounds = float(rounds_used.mean())
    mean_symbols = float(symbols.mean())
    elapsed = mean_symbols + cfg.rtt_symbols * mean_rounds
    residual = float(failed.mean())
    goodput = cfg.k * (1.0 - residual) / elapsed if elapsed > 0 else 0.0
    return HarqResult(
        goodput=goodput,
        residual_fer=residual,
        mean_rounds=mean_rounds,
        mean_symbols=mean_symbols,
        elapsed_symbols=elapsed,
        q=(),
        method="monte_carlo",
    )


def optimal_first_rate(
    k: int,
    rtt_symbols: float,
    esn0_db: float,
    delta: int,
    max_rounds: int = 4,
    alpha: float = 1.0,
    n1_grid: np.ndarray | None = None,
    exact: bool = True,
) -> tuple[float, int, HarqResult]:
    """Best first-transmission rate for a given RTT, by grid search.

    Args:
        k: Information symbols per frame.
        rtt_symbols: Feedback latency D, symbol times.
        esn0_db: Per-symbol SNR, dB.
        delta: IR increment, symbols.
        max_rounds: M.
        alpha: Code-family efficiency.
        n1_grid: Candidate first-transmission lengths, symbols.  Defaults to
            ``k`` to ``3k`` in 32 steps.
        exact: Use the exact nested DP rather than the approximation.

    Returns:
        ``(best_rate, best_n1, result)``.
    """
    if n1_grid is None:
        n1_grid = np.unique(np.linspace(k, 3 * k, 32).astype(int))
    best: tuple[float, int, HarqResult] | None = None
    engine = harq_goodput_exact if exact else harq_goodput
    for n1 in n1_grid:
        cfg = HarqConfig(
            k=k,
            n1=int(n1),
            delta=delta,
            max_rounds=max_rounds,
            esn0_db=esn0_db,
            rtt_symbols=rtt_symbols,
            alpha=alpha,
            scheme="type_ii",
        )
        res = engine(cfg)
        if best is None or res.goodput > best[2].goodput:
            best = (cfg.first_rate, int(n1), res)
    assert best is not None
    return best


def crossover_rtt(
    k: int,
    esn0_db: float,
    delta: int,
    retransmit_rate: float,
    upfront_rate: float,
    max_rounds: int = 4,
    alpha: float = 1.0,
    d_grid: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Locate the RTT at which up-front redundancy overtakes retransmission.

    Two strategies are compared over a grid of feedback latencies D:

        retransmit: first transmission at ``retransmit_rate``, up to
            ``max_rounds`` IR increments,
        up front:   first transmission at ``upfront_rate`` (longer, stronger),
            same increment schedule available.

    Args:
        k: Information symbols per frame.
        esn0_db: Per-symbol SNR, dB.
        delta: IR increment, symbols.
        retransmit_rate: First-transmission code rate of the high-rate strategy.
        upfront_rate: First-transmission code rate of the low-rate strategy.
        max_rounds: M for both.
        alpha: Code-family efficiency.
        d_grid: Feedback latencies to evaluate, symbol times.  Defaults to a
            log grid from ``k/100`` to ``100 k``.

    Returns:
        A dict with ``d``, ``goodput_retransmit``, ``goodput_upfront`` and
        ``crossover_d``.  ``crossover_d`` is ``nan`` if no sign change occurs
        inside the grid, which is reported rather than extrapolated.
    """
    for name, rate in (("retransmit_rate", retransmit_rate), ("upfront_rate", upfront_rate)):
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {rate}")
    if upfront_rate >= retransmit_rate:
        raise ValueError(
            f"upfront_rate ({upfront_rate}) must be below retransmit_rate "
            f"({retransmit_rate}); the up-front strategy is the one that pays "
            "for redundancy before it knows whether it needs it"
        )
    if d_grid is None:
        d_grid = np.unique(np.round(np.logspace(np.log10(k / 100.0), np.log10(100.0 * k), 60)))
    g_re = np.empty(d_grid.size)
    g_up = np.empty(d_grid.size)
    for i, d in enumerate(d_grid):
        g_re[i] = harq_goodput_exact(
            HarqConfig(
                k=k,
                n1=int(round(k / retransmit_rate)),
                delta=delta,
                max_rounds=max_rounds,
                esn0_db=esn0_db,
                rtt_symbols=float(d),
                alpha=alpha,
            )
        ).goodput
        g_up[i] = harq_goodput_exact(
            HarqConfig(
                k=k,
                n1=int(round(k / upfront_rate)),
                delta=delta,
                max_rounds=max_rounds,
                esn0_db=esn0_db,
                rtt_symbols=float(d),
                alpha=alpha,
            )
        ).goodput
    diff = g_up - g_re
    sign_change = np.nonzero(np.diff(np.sign(diff)))[0]
    if sign_change.size == 0:
        cross = float("nan")
    else:
        i = int(sign_change[0])
        x0, x1 = d_grid[i], d_grid[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        cross = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return {
        "d": d_grid,
        "goodput_retransmit": g_re,
        "goodput_upfront": g_up,
        "crossover_d": cross,
    }


def schedule_metrics(
    k: int,
    increments: list[int] | tuple[int, ...] | np.ndarray,
    p: float,
    rtt_symbols: float,
    alpha: float = 1.0,
    drop_penalty: float = 0.0,
) -> dict[str, float]:
    """Exact metrics for an arbitrary type-II increment schedule at fixed BER.

    Generalises :func:`harq_goodput_exact` to a schedule whose increments differ
    from round to round, and to a per-symbol error probability supplied directly
    rather than derived from an Es/N0.  Both generalisations are needed by
    :mod:`arqlonghaul.policy`: the learned policy picks a different increment at
    every round, and the analytic baseline is evaluated at the stationary
    *mixture* error probability of a two-state fade, which is not any single
    Es/N0.

    The dynamic program is the one documented on :func:`harq_goodput_exact`:
    carry the sub-probability distribution of accumulated symbol errors over
    frames not yet decoded, absorb states at or below ``t_m`` after each round,
    convolve with the next increment's binomial.

    Args:
        k: Information symbols per frame.
        increments: Symbols sent in each round.  ``increments[0]`` is the total
            length of the first transmission's *redundancy*, so
            ``n_1 = k + increments[0]``; later entries are parity increments.
        p: Per-symbol error probability.
        rtt_symbols: Feedback latency D, symbol times, charged after every round.
        alpha: Code-family efficiency of equation (8).
        drop_penalty: Symbol times charged for a frame still undecoded after the
            last round, representing the cost of handing it to the outer ARQ
            layer.  0 leaves it uncharged.

    Returns:
        ``{"goodput", "residual_fer", "mean_rounds", "mean_symbols",
        "elapsed_symbols", "expected_cost"}``.  ``expected_cost`` is
        ``elapsed_symbols + drop_penalty * residual_fer``, the surrogate the
        learned policy minimises.
    """
    inc = [int(x) for x in increments]
    if not inc:
        raise ValueError("increments must be non-empty")
    if any(x < 0 for x in inc):
        raise ValueError(f"increments must all be >= 0, got {inc}")
    if inc[0] <= 0:
        raise ValueError("the first increment must be > 0 (n_1 must exceed k)")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if rtt_symbols < 0:
        raise ValueError(f"rtt_symbols must be >= 0, got {rtt_symbols}")
    if drop_penalty < 0:
        raise ValueError(f"drop_penalty must be >= 0, got {drop_penalty}")
    n1 = k + inc[0]
    dist = binom.pmf(np.arange(n1 + 1), n1, p)
    elapsed = 0.0
    symbols = 0.0
    rounds = 0.0
    n_acc = n1
    reach = 1.0
    for m, step in enumerate(inc, start=1):
        if m > 1:
            n_acc += step
            if step > 0:
                dist = np.convolve(dist, binom.pmf(np.arange(step + 1), step, p))
        sent = n1 if m == 1 else step
        elapsed += reach * (sent + rtt_symbols)
        symbols += reach * sent
        rounds += reach
        t_m = singleton_t(n_acc, k, alpha)
        if t_m >= 0:
            dist = dist.copy()
            dist[: min(t_m + 1, dist.size)] = 0.0
        reach = float(dist.sum())
    residual = reach
    goodput = k * (1.0 - residual) / elapsed if elapsed > 0 else 0.0
    return {
        "goodput": goodput,
        "residual_fer": residual,
        "mean_rounds": rounds,
        "mean_symbols": symbols,
        "elapsed_symbols": elapsed,
        "expected_cost": elapsed + drop_penalty * residual,
    }
