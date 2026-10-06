"""The learned-policy environment, and the three disjoint seed sets.

Environment
-----------
One HARQ process on a long link whose signal-to-noise ratio is governed by a
two-state fade.  Time is counted in *rounds*: one round is one transmission
plus one feedback wait of D symbol times.  The fade state follows the
Gilbert-Elliott chain of :mod:`arqlonghaul.channel`, advanced one step per
round, with a per-symbol error probability from :func:`arqlonghaul.channel.bpsk_ber`
at the Es/N0 of the current state.  Decoding follows the Singleton-bound
bounded-distance model of :mod:`arqlonghaul.harq`, equations (7) and (8).

The agent chooses, at every round, how many redundancy symbols to send next.
It does *not* observe the fade state.  That is the whole problem: on a long
link the feedback that could reveal the state is at least one round trip old by
the time it can be acted on, so a policy's only purchase on the channel is
through the statistics of what it has already seen.  Whether that is worth
anything depends on how the fade correlation time compares with the round
length, which is why :func:`long_burst_env` and :func:`short_burst_env` exist
and are both reported.

Seed discipline
---------------
Three disjoint seed sets, and nothing crosses between them:

    fit    seeds 1000-1199   train the learned model
    tune   seeds 2000-2199   choose every free parameter of every baseline
                             *and* of the learned policy
    report seeds 3000-3399   the only numbers that are published

A baseline whose parameter was chosen on the reporting seeds is not a baseline,
it is the learned model wearing a disguise, and the comparison is then rigged.
The tune set exists so that the fixed-rate baseline gets exactly the same
advantage the learned policy gets: one pass of parameter selection on data that
is not the data it is scored on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .channel import GilbertElliottChannel, bpsk_ber

__all__ = [
    "FadeHarqEnv",
    "SeedSplit",
    "SEED_SPLIT",
    "long_burst_env",
    "short_burst_env",
    "DEFAULT_ACTIONS",
]

DEFAULT_ACTIONS: tuple[int, ...] = (10, 25, 50, 100, 200)
"""Redundancy increments the agent may choose, symbols.

Five actions spanning a factor of twenty.  At ``k = 200`` information symbols
the first-round choices correspond to first-transmission code rates of 0.952,
0.889, 0.800, 0.667 and 0.500.
"""


@dataclass(frozen=True)
class SeedSplit:
    """The three disjoint seed sets, as explicit tuples.

    Attributes:
        fit: Seeds used to train the learned model.
        tune: Seeds used to select free parameters, for every policy.
        report: Seeds used for the published comparison, used once.
    """

    fit: tuple[int, ...]
    tune: tuple[int, ...]
    report: tuple[int, ...]

    def __post_init__(self) -> None:
        sets = [set(self.fit), set(self.tune), set(self.report)]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = sets[i] & sets[j]
                if overlap:
                    raise ValueError(
                        f"seed sets {i} and {j} overlap on {sorted(overlap)[:5]}; "
                        "the three sets must be disjoint"
                    )

    def summary(self) -> dict[str, int]:
        """Sizes of the three sets."""
        return {
            "n_fit": len(self.fit),
            "n_tune": len(self.tune),
            "n_report": len(self.report),
        }


SEED_SPLIT = SeedSplit(
    fit=tuple(range(1000, 1200)),
    tune=tuple(range(2000, 2200)),
    report=tuple(range(3000, 3400)),
)
"""The split used by every script in this repository."""


@dataclass(frozen=True)
class FadeHarqEnv:
    """A HARQ-over-fade environment.

    Attributes:
        k: Information symbols per frame.
        alpha: Code-family efficiency, equation (8).
        esn0_good_db: Per-symbol Es/N0 in the Good fade state, dB.
        esn0_bad_db: Per-symbol Es/N0 in the Bad (faded) state, dB.
        p_gb: P(Good -> Bad) per round.
        p_bg: P(Bad -> Good) per round.
        rtt_symbols: Feedback latency D, symbol times, charged every round.
        max_rounds: Transmissions allowed per frame, M.
        drop_penalty: Symbol times charged when a frame is still undecoded
            after M rounds, representing the cost of escalating it to the outer
            ARQ layer.  It is an explicit design input, not a tuned quantity,
            and every cost figure says which value produced it.
        actions: Redundancy increments the agent may choose, symbols.
        name: Label.
    """

    k: int = 200
    alpha: float = 0.5
    esn0_good_db: float = 2.5
    esn0_bad_db: float = -1.5
    p_gb: float = 0.004
    p_bg: float = 0.04
    rtt_symbols: float = 50.0
    max_rounds: int = 4
    drop_penalty: float = 2000.0
    actions: tuple[int, ...] = field(default=DEFAULT_ACTIONS)
    name: str = "fade-harq"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if not self.actions or any(a <= 0 for a in self.actions):
            raise ValueError("actions must be a non-empty tuple of positive integers")
        if self.esn0_bad_db > self.esn0_good_db:
            raise ValueError(
                f"esn0_bad_db ({self.esn0_bad_db}) must not exceed esn0_good_db "
                f"({self.esn0_good_db}); the Bad state is the faded one"
            )
        if self.drop_penalty < 0:
            raise ValueError(f"drop_penalty must be >= 0, got {self.drop_penalty}")
        # Validate the chain through the shared channel class.
        GilbertElliottChannel(p_gb=self.p_gb, p_bg=self.p_bg, eps_g=0.0, eps_b=1.0)

    @property
    def chain(self) -> GilbertElliottChannel:
        """The underlying two-state chain, for its stationary quantities."""
        return GilbertElliottChannel(
            p_gb=self.p_gb, p_bg=self.p_bg, eps_g=0.0, eps_b=1.0, name=self.name
        )

    @property
    def pi_bad(self) -> float:
        """Stationary probability of the faded state, dimensionless."""
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def mean_burst_rounds(self) -> float:
        """Mean faded-state sojourn, rounds."""
        return 1.0 / self.p_bg

    @property
    def ber_good(self) -> float:
        """Per-symbol error probability in the Good state."""
        return bpsk_ber(self.esn0_good_db)

    @property
    def ber_bad(self) -> float:
        """Per-symbol error probability in the Bad state."""
        return bpsk_ber(self.esn0_bad_db)

    @property
    def ber_mixture(self) -> float:
        """Stationary mean per-symbol error probability, dimensionless.

        ``(1 - pi_b) p_good + pi_b p_bad``.  This is the only channel knowledge
        the analytic baseline is given: the marginal, with no state information
        and no data.
        """
        pb = self.pi_bad
        return (1.0 - pb) * self.ber_good + pb * self.ber_bad

    def state_sequence(self, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
        """Boolean faded-state trace over ``n_rounds`` rounds, started stationary.

        Generated up front and shared between policies so that policy
        comparisons use common random numbers.
        """
        if n_rounds <= 0:
            raise ValueError(f"n_rounds must be > 0, got {n_rounds}")
        return self.chain.state_trace(n_rounds, rng)

    def describe(self) -> dict[str, float | int | str]:
        """Flat dict of the configuration and its derived statistics."""
        return {
            "name": self.name,
            "k": self.k,
            "alpha": self.alpha,
            "esn0_good_db": self.esn0_good_db,
            "esn0_bad_db": self.esn0_bad_db,
            "rtt_symbols": self.rtt_symbols,
            "max_rounds": self.max_rounds,
            "drop_penalty": self.drop_penalty,
            "pi_bad": self.pi_bad,
            "mean_burst_rounds": self.mean_burst_rounds,
            "ber_good": self.ber_good,
            "ber_bad": self.ber_bad,
            "ber_mixture": self.ber_mixture,
            "n_actions": len(self.actions),
        }


def _env_from_burst(
    mean_burst_rounds: float, pi_bad: float, name: str, **kwargs: float | int
) -> FadeHarqEnv:
    """Build an environment from a mean fade sojourn and a fade duty cycle."""
    if mean_burst_rounds < 1.0:
        raise ValueError(f"mean_burst_rounds must be >= 1, got {mean_burst_rounds}")
    if not 0.0 < pi_bad < 1.0:
        raise ValueError(f"pi_bad must be in (0, 1), got {pi_bad}")
    p_bg = 1.0 / mean_burst_rounds
    p_gb = p_bg * pi_bad / (1.0 - pi_bad)
    if p_gb > 1.0:
        raise ValueError(
            f"pi_bad={pi_bad} with mean_burst_rounds={mean_burst_rounds} implies "
            f"P(G->B)={p_gb:.3f} > 1"
        )
    return FadeHarqEnv(p_gb=p_gb, p_bg=p_bg, name=name, **kwargs)


def long_burst_env(**kwargs: float | int) -> FadeHarqEnv:
    """Fade sojourn long compared with one round: 25 rounds, 20 per cent duty.

    The regime where stale feedback is still informative, because the state that
    produced the feedback is probably the state that is still there.
    """
    return _env_from_burst(25.0, 0.20, "long-burst (25 rounds, 20% faded)", **kwargs)


def short_burst_env(**kwargs: float | int) -> FadeHarqEnv:
    """Fade sojourn comparable with one round: 1.5 rounds, 20 per cent duty.

    The regime where stale feedback is nearly worthless, because the state has
    almost certainly changed.  Included so that the learned policy is reported
    where it should not help as well as where it might.
    """
    return _env_from_burst(1.5, 0.20, "short-burst (1.5 rounds, 20% faded)", **kwargs)


def nominal_rtt_symbols(rate_bps: float, rtt_s: float) -> float:
    """Feedback latency in symbol times, ``rate_bps * rtt_s`` for one bit/symbol.

    BPSK carries one bit per channel symbol, so the symbol rate equals the bit
    rate and D is simply the bandwidth-delay product in bits.  Provided so that
    examples can say which physical link a value of D corresponds to.
    """
    if rate_bps <= 0 or rtt_s < 0:
        raise ValueError("rate_bps must be > 0 and rtt_s >= 0")
    return float(rate_bps * rtt_s)


def effective_memory_rounds(env: FadeHarqEnv) -> float:
    """Rounds over which the fade state stays correlated, ``-1/ln|lambda|``.

    ``lambda = 1 - p_gb - p_bg`` is the second eigenvalue of the chain, so the
    state autocorrelation decays as ``lambda**k`` and this is its 1/e time in
    rounds.  Compare it with 1: above 1 the previous round's feedback says
    something about this round, below 1 it does not.
    """
    lam = abs(1.0 - env.p_gb - env.p_bg)
    if lam <= 0.0:
        return 0.0
    if lam >= 1.0:
        return math.inf
    return -1.0 / math.log(lam)
