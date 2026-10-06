"""Frame-error channels: independent, and correlated (Gilbert-Elliott).

The whole point of this package is the difference between these two, so both
are expressed in the same interface and parameterised so that they can be given
the *same marginal frame error rate* and differ only in correlation.  Any
comparison that does not hold the marginal fixed is measuring two things at
once.

A channel produces a boolean array: ``errors(n_slots, rng)[t]`` is True if a
frame transmitted in slot ``t`` would arrive in error.  The array is generated
for the whole run before any protocol sees it, which has two consequences worth
stating.  First, the channel evolves in *time*, not per transmission, so a
protocol that idles still experiences the channel passing by -- which is
correct, and is what makes burst length comparable with the round-trip time.
Second, the same realisation can be fed to stop-and-wait, go-back-N and
selective repeat, so protocol comparisons use common random numbers and the
difference between two protocols is measured with far less Monte Carlo noise
than two independent runs would give.

Gilbert-Elliott model
---------------------
Two states, Good and Bad, a first-order Markov chain on the frame slot index,
with per-state error probabilities:

    P(G -> B) = p_gb,   P(B -> G) = p_bg
    P(error | G) = eps_g,   P(error | B) = eps_b

Stationary distribution and marginals:

    pi_b = p_gb / (p_gb + p_bg)                                            (2)
    p_bar = (1 - pi_b) eps_g + pi_b eps_b                                  (3)
    E[Bad sojourn]  = 1 / p_bg   slots                                     (4)
    E[Good sojourn] = 1 / p_gb   slots

Equations (2)-(4) are the standard two-state Markov results; the model itself is
E. N. Gilbert, "Capacity of a burst-noise channel", Bell System Technical
Journal, vol. 39, no. 5, pp. 1253-1265, 1960, extended by E. O. Elliott,
"Estimates of error rates for codes on burst-noise channels", Bell System
Technical Journal, vol. 42, no. 5, pp. 1977-1997, 1963, to the case eps_g > 0.
(Title and year of both were confirmed against the Nokia Bell Labs publication
pages; volume, issue and page numbers are as given in the reference list of
arXiv:2005.06921, read 2026-10-06.)

Lag-k autocorrelation of the error indicator, used as the burstiness summary:

    rho_k = pi_b (1 - pi_b) (eps_b - eps_g)**2 lambda**k / (p_bar (1 - p_bar))  (5)

with lambda = 1 - p_gb - p_bg the second eigenvalue of the transition matrix.
Equation (5) is derived in validation/VALIDATION.md section 2 and is checked
against the sample autocorrelation of a realisation.

Physical-layer mapping
----------------------
:func:`ber_to_fer` converts a raw bit error rate to a frame error rate under the
independent-bit assumption, and :func:`bpsk_ber` gives the raw BER of coherent
BPSK over AWGN.  Both are textbook; both are only used where the frame error
rate has to be tied to a link budget rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

__all__ = [
    "FrameChannel",
    "IndependentFrameChannel",
    "GilbertElliottChannel",
    "bpsk_ber",
    "ber_to_fer",
    "fer_to_ber",
]


class FrameChannel:
    """Interface: a frame-error process over slot index.

    Subclasses implement :meth:`errors` and expose :attr:`mean_fer`.
    """

    name: str = "channel"

    @property
    def mean_fer(self) -> float:
        """Stationary marginal frame error probability, dimensionless."""
        raise NotImplementedError

    def errors(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Boolean error flags for slots ``0 .. n_slots-1``."""
        raise NotImplementedError

    def autocorrelation(self, lag: int) -> float:
        """Analytic lag-``lag`` autocorrelation of the error indicator."""
        raise NotImplementedError


@dataclass(frozen=True)
class IndependentFrameChannel(FrameChannel):
    """Memoryless frame-error channel: each slot errs independently with ``fer``.

    This is the channel for which every expression in :mod:`closedform` is
    exact.  It is the reference, not the realistic case.
    """

    fer: float
    name: str = "independent"

    def __post_init__(self) -> None:
        if not 0.0 <= self.fer <= 1.0:
            raise ValueError(f"fer must be in [0, 1], got {self.fer}")

    @property
    def mean_fer(self) -> float:
        """Stationary marginal frame error probability."""
        return float(self.fer)

    @property
    def mean_burst_slots(self) -> float:
        """Mean run length of consecutive errors, slots: ``1/(1-fer)``."""
        if self.fer >= 1.0:
            return math.inf
        return 1.0 / (1.0 - self.fer)

    def errors(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Boolean error flags, shape ``(n_slots,)``."""
        if n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {n_slots}")
        return rng.random(n_slots) < self.fer

    def autocorrelation(self, lag: int) -> float:
        """1 at lag 0, 0 otherwise."""
        return 1.0 if lag == 0 else 0.0


@dataclass(frozen=True)
class GilbertElliottChannel(FrameChannel):
    """Two-state Markov frame-error channel with controllable burst length.

    Attributes:
        p_gb: P(Good -> Bad) per slot, dimensionless in (0, 1).
        p_bg: P(Bad -> Good) per slot, dimensionless in (0, 1).
        eps_g: Frame error probability in the Good state.
        eps_b: Frame error probability in the Bad state.
        name: Label.

    The construction that makes comparisons fair is
    :meth:`from_mean_and_burst`, which fixes the marginal frame error rate and
    the mean bad-state sojourn independently.
    """

    p_gb: float
    p_bg: float
    eps_g: float = 0.0
    eps_b: float = 0.5
    name: str = "gilbert-elliott"

    def __post_init__(self) -> None:
        for field in ("p_gb", "p_bg"):
            v = getattr(self, field)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{field} must be in (0, 1], got {v}")
        for field in ("eps_g", "eps_b"):
            v = getattr(self, field)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {v}")
        if self.eps_b < self.eps_g:
            raise ValueError(
                f"eps_b ({self.eps_b}) must be >= eps_g ({self.eps_g}); "
                "the Bad state is the worse one by definition"
            )

    @classmethod
    def from_mean_and_burst(
        cls,
        mean_fer: float,
        mean_burst_slots: float,
        eps_g: float = 0.0,
        eps_b: float = 1.0,
        name: str | None = None,
    ) -> GilbertElliottChannel:
        """Build a channel with a given marginal FER and bad-state sojourn.

        Inverts equations (2) and (4):

            p_bg = 1 / mean_burst_slots
            pi_b = (mean_fer - eps_g) / (eps_b - eps_g)
            p_gb = p_bg pi_b / (1 - pi_b)

        Args:
            mean_fer: Target stationary frame error probability, in
                ``(eps_g, eps_b)``.
            mean_burst_slots: Target mean bad-state sojourn, slots, >= 1.
                With ``eps_b = 1`` this is also the mean length of a run of
                errored frames, which is the usual operational definition of
                burst length.
            eps_g: Frame error probability in the Good state.
            eps_b: Frame error probability in the Bad state.
            name: Optional label.

        Raises:
            ValueError: if ``mean_fer`` is not strictly inside
                ``(eps_g, eps_b)`` or ``mean_burst_slots < 1``.
        """
        if mean_burst_slots < 1.0:
            raise ValueError(
                f"mean_burst_slots must be >= 1 slot, got {mean_burst_slots}"
            )
        if not eps_g < mean_fer < eps_b:
            raise ValueError(
                f"mean_fer={mean_fer} must lie strictly between eps_g={eps_g} and "
                f"eps_b={eps_b}; outside that range no stationary distribution exists"
            )
        p_bg = 1.0 / mean_burst_slots
        pi_b = (mean_fer - eps_g) / (eps_b - eps_g)
        p_gb = p_bg * pi_b / (1.0 - pi_b)
        if p_gb > 1.0:
            raise ValueError(
                f"the requested (mean_fer={mean_fer}, mean_burst_slots="
                f"{mean_burst_slots}) implies P(G->B)={p_gb:.3f} > 1; "
                "either lower the mean FER or lengthen the burst"
            )
        return cls(
            p_gb=p_gb,
            p_bg=p_bg,
            eps_g=eps_g,
            eps_b=eps_b,
            name=name or f"GE fer={mean_fer:g} burst={mean_burst_slots:g}",
        )

    @property
    def pi_bad(self) -> float:
        """Stationary probability of the Bad state, equation (2)."""
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def mean_fer(self) -> float:
        """Stationary marginal frame error probability, equation (3)."""
        pb = self.pi_bad
        return (1.0 - pb) * self.eps_g + pb * self.eps_b

    @property
    def mean_burst_slots(self) -> float:
        """Mean Bad-state sojourn, slots, equation (4)."""
        return 1.0 / self.p_bg

    @property
    def mean_good_slots(self) -> float:
        """Mean Good-state sojourn, slots."""
        return 1.0 / self.p_gb

    @property
    def lam(self) -> float:
        """Second eigenvalue of the transition matrix, ``1 - p_gb - p_bg``."""
        return 1.0 - self.p_gb - self.p_bg

    def autocorrelation(self, lag: int) -> float:
        """Lag-``lag`` autocorrelation of the error indicator, equation (5)."""
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        if lag == 0:
            return 1.0
        p = self.mean_fer
        if p <= 0.0 or p >= 1.0:
            return 0.0
        pb = self.pi_bad
        num = pb * (1.0 - pb) * (self.eps_b - self.eps_g) ** 2 * self.lam**lag
        return num / (p * (1.0 - p))

    def errors(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Boolean error flags, shape ``(n_slots,)``, started in stationarity.

        The state chain is generated with a vectorised inverse-transform step
        per slot: a single uniform per slot decides the transition, a second
        decides the error.  Both are drawn as whole arrays up front, so the cost
        is O(n_slots) with a Python loop only over the chain recursion, which
        numpy cannot vectorise.
        """
        if n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {n_slots}")
        u_state = rng.random(n_slots)
        u_err = rng.random(n_slots)
        states = np.empty(n_slots, dtype=bool)
        bad = rng.random() < self.pi_bad
        p_gb, p_bg = self.p_gb, self.p_bg
        for t in range(n_slots):
            states[t] = bad
            if bad:
                if u_state[t] < p_bg:
                    bad = False
            elif u_state[t] < p_gb:
                bad = True
        thresh = np.where(states, self.eps_b, self.eps_g)
        return u_err < thresh

    def state_trace(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Boolean Bad-state trace, for diagnostics and plots."""
        if n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {n_slots}")
        u_state = rng.random(n_slots)
        states = np.empty(n_slots, dtype=bool)
        bad = rng.random() < self.pi_bad
        for t in range(n_slots):
            states[t] = bad
            if bad:
                if u_state[t] < self.p_bg:
                    bad = False
            elif u_state[t] < self.p_gb:
                bad = True
        return states


def bpsk_ber(esn0_db: float) -> float:
    """Raw bit error rate of coherent BPSK over AWGN.

    ``p_b = Q(sqrt(2 Es/N0)) = 0.5 erfc(sqrt(Es/N0))``, the standard result for
    antipodal signalling with matched-filter detection and perfect phase
    reference (any digital communications text; S. Lin and D. J. Costello, Jr.,
    *Error Control Coding*, 2nd ed., 2004).

    Args:
        esn0_db: Symbol energy to noise density ratio, dB.

    Returns:
        Bit error probability, dimensionless in (0, 0.5].
    """
    esn0 = 10.0 ** (float(esn0_db) / 10.0)
    return float(0.5 * erfc(math.sqrt(esn0)))


def ber_to_fer(ber: float, frame_bits: int) -> float:
    """Frame error rate from a raw BER assuming independent bit errors.

    ``p = 1 - (1 - ber)**frame_bits``.  Valid only when bit errors within a
    frame are independent and the frame carries no error correction; with a code
    use :func:`arqlonghaul.harq.block_fer` instead.

    Args:
        ber: Bit error probability, in [0, 1].
        frame_bits: Frame length, bits.
    """
    if not 0.0 <= ber <= 1.0:
        raise ValueError(f"ber must be in [0, 1], got {ber}")
    if frame_bits <= 0:
        raise ValueError(f"frame_bits must be > 0, got {frame_bits}")
    return float(-np.expm1(frame_bits * np.log1p(-ber))) if ber < 1.0 else 1.0


def fer_to_ber(fer: float, frame_bits: int) -> float:
    """Inverse of :func:`ber_to_fer`: ``ber = 1 - (1 - fer)**(1/frame_bits)``."""
    if not 0.0 <= fer < 1.0:
        raise ValueError(f"fer must be in [0, 1), got {fer}")
    if frame_bits <= 0:
        raise ValueError(f"frame_bits must be > 0, got {frame_bits}")
    return float(-np.expm1(np.log1p(-fer) / frame_bits))
