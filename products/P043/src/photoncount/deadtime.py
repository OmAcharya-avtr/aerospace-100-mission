"""Dead-time models for a photon-counting detector, forward and inverse.

Every real counter is blind for a while after a count. Two idealisations
bracket real behaviour, and both are standard (G. F. Knoll, *Radiation
Detection and Measurement*, 4th ed., Wiley 2010, chapter 4, "Dead time"):

**Non-paralyzable.** An event during the dead period is simply lost and does
not extend the dead period. With true rate ``n`` (counts/s) and dead time
``tau`` (s) the observed rate is

    m = n / (1 + n tau)                                                   (D1)

which is monotonic, saturates at ``1/tau``, and inverts uniquely:

    n = m / (1 - m tau),       valid for m tau < 1.                       (D2)

**Paralyzable (extending).** An event during the dead period is lost *and*
restarts the dead period, so a burst can lock the detector out indefinitely:

    m = n exp(-n tau)                                                     (D3)

**(D3) is not monotonic.** It rises, peaks, and falls back to zero:

    dm/dn = 0  at  n = 1/tau,   giving  m_max = 1 / (e tau).              (D4)

So every observed rate below ``m_max`` has **two** true rates that produce it,
one below ``1/tau`` and one above, and an observed rate above ``m_max`` has
none. Inverting (D3) without saying which branch you are on is the single most
common error in this subject, and it is silent: a detector driven past its
maximum reports a *falling* count rate, which reads as a fading source.
Expressed with the Lambert W function,

    n = -W(-m tau) / tau                                                  (D5)

with the principal branch ``W_0`` giving the lower root (``n tau <= 1``) and
the ``W_{-1}`` branch giving the upper root (``n tau >= 1``). This module
requires the caller to choose, and :func:`paralyzable_true` has no default that
hides the choice from a reader of the call site.

Units: ``n`` and ``m`` in counts/s, ``tau`` in s. The equations depend only on
the product ``n tau``, so a dimensionless per-slot form works equally well:
pass ``n`` in counts/slot and ``tau`` in slots.

Validity: (D1) and (D3) assume a Poisson arrival process, a fixed dead time
with no jitter, and no afterpulsing. Afterpulsing is in
:mod:`photoncount.afterpulse`; the two together are the regime the learned
correction in :mod:`photoncount.correction` is for, because the inverses above
each assume the other effect is absent.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import special

__all__ = [
    "DeadTimeModel",
    "ParalyzableBranch",
    "dead_time_loss_fraction",
    "is_observable",
    "live_time_fraction",
    "nonparalyzable_observed",
    "nonparalyzable_true",
    "observed_rate",
    "paralyzable_maximum",
    "paralyzable_observed",
    "paralyzable_true",
    "true_rate",
]

DeadTimeModel = Literal["paralyzable", "nonparalyzable"]
ParalyzableBranch = Literal["lower", "upper"]


def _as_rate(name: str, value: np.ndarray | float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return arr


def _as_tau(value: float) -> float:
    tau = float(value)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"dead_time must be finite and > 0, got {value!r}")
    return tau


def nonparalyzable_observed(true_rate_hz: np.ndarray | float, dead_time_s: float) -> np.ndarray:
    """(D1) ``m = n / (1 + n tau)``, counts/s. Monotonic, saturates at ``1/tau``."""
    n = _as_rate("true_rate_hz", true_rate_hz)
    tau = _as_tau(dead_time_s)
    return n / (1.0 + n * tau)


def nonparalyzable_true(observed_rate_hz: np.ndarray | float, dead_time_s: float) -> np.ndarray:
    """(D2) ``n = m / (1 - m tau)``, counts/s.

    Raises ``ValueError`` when ``m tau >= 1``, which is physically unreachable
    for this model: a non-paralyzable counter cannot report at or above
    ``1/tau``.
    """
    m = _as_rate("observed_rate_hz", observed_rate_hz)
    tau = _as_tau(dead_time_s)
    if np.any(m * tau >= 1.0):
        raise ValueError(
            "observed_rate_hz * dead_time_s must be < 1 for the non-paralyzable "
            f"model; max observed rate is 1/tau = {1.0 / tau:.6g} counts/s"
        )
    return m / (1.0 - m * tau)


def paralyzable_observed(true_rate_hz: np.ndarray | float, dead_time_s: float) -> np.ndarray:
    """(D3) ``m = n exp(-n tau)``, counts/s. Non-monotonic; see (D4)."""
    n = _as_rate("true_rate_hz", true_rate_hz)
    tau = _as_tau(dead_time_s)
    return n * np.exp(-n * tau)


def paralyzable_maximum(dead_time_s: float) -> tuple[float, float]:
    """(D4) ``(n_max, m_max) = (1/tau, 1/(e tau))``, both counts/s.

    ``n_max`` is the true rate at which the observed rate peaks; ``m_max`` is
    that peak. No paralyzable detector with this dead time can ever report
    above ``m_max``, whatever the illumination.
    """
    tau = _as_tau(dead_time_s)
    return 1.0 / tau, 1.0 / (np.e * tau)


def paralyzable_true(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: float,
    branch: ParalyzableBranch,
) -> np.ndarray:
    """(D5) ``n = -W(-m tau) / tau``, counts/s. ``branch`` is mandatory.

    ``branch="lower"`` takes the principal Lambert branch ``W_0`` and returns
    the root with ``n tau <= 1``; ``branch="upper"`` takes ``W_{-1}`` and
    returns the root with ``n tau >= 1``. Raises ``ValueError`` if any observed
    rate exceeds ``m_max = 1/(e tau)``, where no root exists.
    """
    m = _as_rate("observed_rate_hz", observed_rate_hz)
    tau = _as_tau(dead_time_s)
    _, m_max = paralyzable_maximum(tau)
    if np.any(m > m_max * (1.0 + 1e-12)):
        raise ValueError(
            f"observed_rate_hz exceeds the paralyzable maximum 1/(e tau) = {m_max:.6g} "
            "counts/s, so no true rate reproduces it; the detector is saturated "
            "or the dead time is wrong"
        )
    if branch not in ("lower", "upper"):
        raise ValueError(f"branch must be 'lower' or 'upper', got {branch!r}")
    arg = np.clip(-m * tau, -1.0 / np.e, 0.0)
    k = 0 if branch == "lower" else -1
    w = special.lambertw(arg, k=k)
    root = -np.real(w) / tau
    return np.where(m == 0.0, 0.0 if branch == "lower" else np.inf, root)


def observed_rate(
    true_rate_hz: np.ndarray | float,
    dead_time_s: float,
    model: DeadTimeModel,
) -> np.ndarray:
    """Dispatch (D1)/(D3) on ``model``. counts/s."""
    if model == "nonparalyzable":
        return nonparalyzable_observed(true_rate_hz, dead_time_s)
    if model == "paralyzable":
        return paralyzable_observed(true_rate_hz, dead_time_s)
    raise ValueError(f"model must be 'paralyzable' or 'nonparalyzable', got {model!r}")


def true_rate(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: float,
    model: DeadTimeModel,
    branch: ParalyzableBranch = "lower",
) -> np.ndarray:
    """Dispatch (D2)/(D5) on ``model``. counts/s.

    ``branch`` is ignored for the non-paralyzable model, whose inverse is
    unique. It is **not** ignored for the paralyzable model and defaults to the
    lower root, which is the right choice only if you know the detector is
    operating below ``1/tau``. :func:`is_observable` is the check.
    """
    if model == "nonparalyzable":
        return nonparalyzable_true(observed_rate_hz, dead_time_s)
    if model == "paralyzable":
        return paralyzable_true(observed_rate_hz, dead_time_s, branch)
    raise ValueError(f"model must be 'paralyzable' or 'nonparalyzable', got {model!r}")


def is_observable(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: float,
    model: DeadTimeModel,
) -> np.ndarray:
    """Boolean: can ``model`` with this ``tau`` produce this observed rate at all?

    ``m tau < 1`` for non-paralyzable, ``m <= 1/(e tau)`` for paralyzable.
    """
    m = _as_rate("observed_rate_hz", observed_rate_hz)
    tau = _as_tau(dead_time_s)
    if model == "nonparalyzable":
        return m * tau < 1.0
    if model == "paralyzable":
        return m <= 1.0 / (np.e * tau) * (1.0 + 1e-12)
    raise ValueError(f"model must be 'paralyzable' or 'nonparalyzable', got {model!r}")


def dead_time_loss_fraction(
    true_rate_hz: np.ndarray | float,
    dead_time_s: float,
    model: DeadTimeModel,
) -> np.ndarray:
    """Fraction of true events not counted, ``1 - m/n`` (-).

    For the non-paralyzable model this equals ``n tau / (1 + n tau)``; for the
    paralyzable model ``1 - exp(-n tau)``. At ``n tau = 1`` the two are 0.5 and
    0.632 respectively.
    """
    n = _as_rate("true_rate_hz", true_rate_hz)
    m = observed_rate(n, dead_time_s, model)
    return np.where(n > 0.0, 1.0 - np.divide(m, n, out=np.ones_like(n), where=n > 0.0), 0.0)


def live_time_fraction(
    true_rate_hz: np.ndarray | float,
    dead_time_s: float,
    model: DeadTimeModel,
) -> np.ndarray:
    """Fraction of wall-clock time the detector is live (-).

    Non-paralyzable: ``1 - m tau`` (each *recorded* count costs one dead
    period). Paralyzable: ``exp(-n tau)`` (the probability that no true event
    occurred in the preceding ``tau``). These are different quantities for a
    reason and conflating them is how a live-time correction goes wrong.
    """
    n = _as_rate("true_rate_hz", true_rate_hz)
    tau = _as_tau(dead_time_s)
    if model == "nonparalyzable":
        return 1.0 - nonparalyzable_observed(n, tau) * tau
    if model == "paralyzable":
        return np.exp(-n * tau)
    raise ValueError(f"model must be 'paralyzable' or 'nonparalyzable', got {model!r}")
