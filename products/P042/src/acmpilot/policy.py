"""Rate-adaptation policies: two causal baselines and one acausal upper bound.

The problem these policies face
------------------------------
At slot ``n`` the transmitter must commit to a MODCOD index. The only channel
information it has is the receiver's report of the SNR at slot ``n - d``, where
``d`` is the round-trip feedback delay in slots. The channel then delivers
``snr_true[n]``, and the slot succeeds if and only if
``snr_true[n] >= threshold[chosen]``. Nothing in that structure is avoidable by
better engineering: the delay is a propagation fact, and the policy therefore
necessarily acts on a stale state.

The three policies here are **non-learned and implemented first**, per the
portfolio's baseline-before-ML rule. The learned predictor in
:mod:`acmpilot.predictor` is benchmarked against all three on the same seeded
sample paths.

Floor behaviour
---------------
Every causal policy returns an index in ``[0, K-1]``: when even the lowest
MODCOD looks unsupported the policy still selects index 0 rather than muting the
transmitter. Muting would make the outage accounting degenerate (a muted slot is
not an outage, it is zero goodput by choice) and the comparison between policies
less legible. A real terminal would mute; the consequence of not modelling that
is recorded in the README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .modcod import ModcodTable

#: The exact wording every plot legend, docstring and table must use for the
#: clairvoyant policy. It is not a policy anyone can run.
CLAIRVOYANT_LABEL: str = "clairvoyant (upper bound, no causal policy can achieve)"


@runtime_checkable
class Policy(Protocol):
    """A MODCOD selection rule.

    ``causal`` is False only for the clairvoyant upper bound.
    """

    name: str
    causal: bool

    def select(
        self, table: ModcodTable, observed_db: np.ndarray, true_db: np.ndarray
    ) -> np.ndarray:
        """Return one MODCOD index per slot, shape ``observed_db.shape``."""
        ...


def _highest_supported(table: ModcodTable, predicted_db: np.ndarray) -> np.ndarray:
    """Highest index with ``threshold <= predicted_db``, floored at 0."""
    idx = table.best_supported(predicted_db)
    return np.maximum(np.atleast_1d(idx), 0)


@dataclass(frozen=True)
class FixedMargin:
    """Select the MODCOD supported by (last known SNR) minus a fixed margin.

    The margin is the whole policy: it is the designer's standing bet on how far
    the channel can fall during one feedback round trip. Too small and the link
    spends its time in outage; too large and it never uses the capacity it has.
    There is no margin that is right for all ``(tau, tau_c)``, which is the
    point.

    Parameters
    ----------
    margin_db
        Subtracted from the delayed SNR report before selection, dB, >= 0.
    """

    margin_db: float = 3.0
    causal: bool = True

    def __post_init__(self) -> None:
        if self.margin_db < 0:
            raise ValueError(f"margin_db must be >= 0 dB, got {self.margin_db}")

    @property
    def name(self) -> str:
        """Label for tables and plot legends."""
        return f"fixed margin {self.margin_db:g} dB"

    def select(
        self, table: ModcodTable, observed_db: np.ndarray, true_db: np.ndarray
    ) -> np.ndarray:
        """Select per slot from ``observed_db - margin_db``. ``true_db`` unused."""
        del true_db
        return _highest_supported(table, np.asarray(observed_db, dtype=float) - self.margin_db)


@dataclass(frozen=True)
class ThresholdHysteresis:
    """Switching thresholds with a dead band, one step per slot.

    State is the currently selected index ``k``. Per slot, with ``x`` the
    delayed SNR report:

        upgrade   k -> k+1   if  x >= threshold[k+1] + up_margin_db
        downgrade k -> k-1   if  x <  threshold[k]   + down_margin_db
        otherwise hold

    The dead band around the boundary between ``k`` and ``k+1`` has width
    ``up_margin_db - down_margin_db`` dB, and that width is exactly the
    chatter-against-outage knob: widening it cuts the switch count and raises
    the time spent below capacity; narrowing it does the reverse. Single-step
    moves mean the policy needs ``K-1`` slots to climb the whole ladder, which
    is itself a source of conservatism at short correlation times.

    Parameters
    ----------
    up_margin_db
        Extra SNR required above the next mode's threshold to upgrade, dB.
    down_margin_db
        Margin below which the current mode is abandoned, dB. Must be
        ``<= up_margin_db`` for the dead band to be non-negative.
    """

    up_margin_db: float = 2.0
    down_margin_db: float = 0.5
    causal: bool = True

    def __post_init__(self) -> None:
        if self.up_margin_db < self.down_margin_db:
            raise ValueError(
                f"up_margin_db ({self.up_margin_db}) must be >= down_margin_db "
                f"({self.down_margin_db}) or the dead band is negative"
            )
        if self.down_margin_db < 0:
            raise ValueError(f"down_margin_db must be >= 0 dB, got {self.down_margin_db}")

    @property
    def dead_band_db(self) -> float:
        """Width of the dead band, dB."""
        return self.up_margin_db - self.down_margin_db

    @property
    def name(self) -> str:
        """Label for tables and plot legends."""
        return f"hysteresis +{self.up_margin_db:g}/-{self.down_margin_db:g} dB"

    def select(
        self, table: ModcodTable, observed_db: np.ndarray, true_db: np.ndarray
    ) -> np.ndarray:
        """Sequential selection; ``true_db`` unused."""
        del true_db
        x = np.asarray(observed_db, dtype=float)
        thr = table.thresholds_db
        top = table.n_modes - 1
        out = np.empty(x.size, dtype=int)
        k = 0
        for n in range(x.size):
            if k < top and x[n] >= thr[k + 1] + self.up_margin_db:
                k += 1
            elif k > 0 and x[n] < thr[k] + self.down_margin_db:
                k -= 1
            out[n] = k
        return out


@dataclass(frozen=True)
class ClairvoyantUpperBound:
    """Selects using the **true future** channel state. Not implementable.

    This policy reads ``snr_true[n]`` at slot ``n``. It is acausal: no terminal
    can do this, because the information does not exist at the transmitter until
    ``d`` slots later. It is included for exactly one reason --- it separates the
    part of the gap to capacity that is caused by *prediction error* from the
    part that is caused by the *granularity of the MODCOD ladder*. A causal
    policy that reaches the clairvoyant bound would be a contradiction; a causal
    policy that closes most of the gap to it has nothing left to gain from a
    better predictor.

    Use :data:`CLAIRVOYANT_LABEL` wherever this appears in a legend or table.
    """

    causal: bool = False

    @property
    def name(self) -> str:
        """The mandated label, :data:`CLAIRVOYANT_LABEL`."""
        return CLAIRVOYANT_LABEL

    def select(
        self, table: ModcodTable, observed_db: np.ndarray, true_db: np.ndarray
    ) -> np.ndarray:
        """Select the highest MODCOD the **true** SNR supports. ``observed_db`` unused."""
        del observed_db
        return _highest_supported(table, np.asarray(true_db, dtype=float))


def baseline_policies(
    *, margin_db: float = 3.0, up_margin_db: float = 2.0, down_margin_db: float = 0.5
) -> tuple[FixedMargin, ThresholdHysteresis, ClairvoyantUpperBound]:
    """The three non-learned policies, in the order the specification requires."""
    return (
        FixedMargin(margin_db=margin_db),
        ThresholdHysteresis(up_margin_db=up_margin_db, down_margin_db=down_margin_db),
        ClairvoyantUpperBound(),
    )
