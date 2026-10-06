"""Second-order timing loop: design, jitter variance and cycle-slip rate.

Everything in this module is **normalised to the symbol period** ``T``:
frequencies are cycles per symbol, the loop's natural frequency is radians per
symbol, and timing errors and jitter are symbol periods.  For a PPM slot clock
the same algebra applies with ``T`` read as the slot period and the loop update
period handled explicitly; see :mod:`slotsync.ppm`.

The chain, in the order the package measures it
-----------------------------------------------
1. ``K_d`` - the detector gain, the slope of the S-curve at the origin, measured
   by :func:`slotsync.scurve.scurve`.  Not assumed anywhere.
2. :class:`LoopDesign` - from a normalised loop noise bandwidth ``B_n = B_L T``
   and a damping factor ``zeta``, the two loop coefficients.
3. :func:`jitter_variance_closed_form` - the classical loop-noise prediction
   ``sigma_eps^2 = 2 B_n sigma_n^2 / K_d^2``.
4. :func:`jitter_variance_exact` - the same quantity from the exact discrete-time
   loop, with no small-bandwidth approximation.
5. :func:`cycle_slip_rate_rice` - the level-crossing estimate of the slip rate
   against loop SNR.

Steps 3 and 4 are independent routes to the same number and they disagree once
``B_n`` is not small; :mod:`slotsync.simulate` adds a third route, a Monte Carlo
of the actual loop driven by the actual detector.  The disagreements are the
product's result, not a defect to be tuned away.

Continuous prototype and the noise-bandwidth relation
-----------------------------------------------------
The analogue second-order loop has closed-loop transfer function

    ``H(s) = (2 zeta w_n s + w_n^2) / (s^2 + 2 zeta w_n s + w_n^2)``

and one-sided noise bandwidth, in hertz,

    ``B_L = w_n (1 + 4 zeta^2) / (8 zeta)``   [hertz, with w_n in rad/s]

equivalently ``B_L = (w_n / 2) (zeta + 1 / (4 zeta))``, which is the classical
second-order-loop result.  It is **not quoted from a page here**:
:func:`noise_bandwidth_numeric` computes ``int_0^inf |H(j 2 pi f)|^2 df`` by
quadrature and ``validation/validate_loop_coefficients.py`` checks the closed
form against it over a sweep of damping factors and natural frequencies.  The
first draft of this module carried the expression with a spurious factor of
``2 pi``; the quadrature check is what caught it, which is the reason the check
exists.  The derivation is in ``docs/TIMING_MODEL.md`` section 3.

Discrete coefficients: exact pole matching
------------------------------------------
With ``theta = w_n T`` (radians per symbol) the two loop coefficients are chosen
so that the discrete loop's poles are **exactly** ``exp(s_i T)`` for the two
poles ``s_i`` of the analogue prototype.  Writing ``alpha = k1 K_d`` and
``beta = k2 K_d``, the linearised loop's characteristic polynomial is
``z^2 - (2 - alpha - beta) z + (1 - alpha)``, and matching it term by term to
``z^2 - 2 exp(-zeta theta) cos(theta sqrt(1 - zeta^2)) z + exp(-2 zeta theta)``
gives

    ``alpha = 1 - exp(-2 zeta theta)``
    ``beta  = 2 - alpha - 2 exp(-zeta theta) cos(theta sqrt(1 - zeta^2))``

with ``cos`` replaced by ``cosh(theta sqrt(zeta^2 - 1))`` for an over-damped
loop.  The coefficients are applied as ``w <- w - k2 e`` and
``tau_hat <- tau_hat - k1 e + w`` with ``k1 = alpha / K_d``, ``k2 = beta / K_d``.
``tests/test_loop.py`` checks the pole identity directly, and the mapping is
exactly invertible (:meth:`LoopDesign.from_coefficients`).

Dividing by ``K_d`` makes the closed-loop dynamics independent of the detector,
which is the whole reason the gain has to be measured rather than guessed.  GNU
Radio's Symbol Sync block exposes the same three design quantities - loop
bandwidth, damping factor and expected TED gain, the last of which its
documentation defines as "the slope of the TED's S-curve at timing offset
tau = 0" - but its loop-bandwidth parameter is normalised differently, so a
number is not transferable between the two without checking the convention.

Cycle slips
-----------
A slip is the loop losing a whole symbol of timing.  It becomes possible once
the timing error reaches the S-curve's unstable zero crossing, at
``+-reversal_offset`` (half a symbol for a detector whose S-curve is
one-symbol-periodic).  :func:`cycle_slip_rate_rice` estimates the rate at which
a Gaussian timing error crosses that boundary using the standard level-crossing
argument for a stationary Gaussian process,

    ``nu = (1 / 2 pi) (sigma_dot / sigma) exp(-a^2 / (2 sigma^2))``

with the discrete-time step difference standing in for the derivative.  It is an
**estimate**, and it is compared against a measured slip count in
``validation/validate_cycle_slips.py``; the derivation and its three stated
approximations are in ``docs/TIMING_MODEL.md`` section 5.  Gardner's *Phaselock
Techniques*, 3rd edition (Wiley, 2005, ISBN 978-0-471-43063-6) is the standard
reference for loop noise and cycle slips in phase-locked loops; its
bibliographic details were verified against the publisher's record on
2026-10-06 and **no equation or page from it is quoted here**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_discrete_lyapunov

__all__ = [
    "LoopDesign",
    "cycle_slip_rate_rice",
    "error_autocorrelation_weights",
    "jitter_variance_closed_form",
    "jitter_variance_coloured",
    "jitter_variance_exact",
    "loop_snr_db",
    "noise_bandwidth_closed_form",
    "noise_bandwidth_numeric",
    "slip_free_seconds",
]


def noise_bandwidth_closed_form(natural_frequency: float, damping: float) -> float:
    """One-sided loop noise bandwidth of the analogue prototype, normalised.

    Parameters
    ----------
    natural_frequency
        ``theta = w_n T``, radians per symbol period.
    damping
        ``zeta``, dimensionless.

    Returns
    -------
    float
        ``B_n = B_L T``, cycles per symbol (dimensionless):
        ``theta (1 + 4 zeta^2) / (8 zeta)``.
    """
    if natural_frequency <= 0.0:
        raise ValueError(
            f"natural_frequency must be positive radians per symbol, got {natural_frequency!r}"
        )
    if damping <= 0.0:
        raise ValueError(f"damping must be positive, got {damping!r}")
    return natural_frequency * (1.0 + 4.0 * damping**2) / (8.0 * damping)


def noise_bandwidth_numeric(
    natural_frequency: float, damping: float, *, limit: float = 2000.0
) -> float:
    """``int_0^inf |H(j 2 pi f)|^2 df`` by quadrature, for the same prototype.

    An independent route to :func:`noise_bandwidth_closed_form`.  The integral is
    taken by quadrature up to ``f_max = limit * theta / (2 pi)`` and the tail
    beyond it is added analytically: for large ``w`` the integrand tends to
    ``(2 zeta theta / w)**2``, whose tail integrates to
    ``(2 zeta theta)**2 / (4 pi**2 f_max)``.  Without that correction the
    truncation alone leaves a relative error of ``16 zeta**3 / (pi * limit *
    (1 + 4 zeta**2))``, which at ``limit = 400`` and ``zeta = 2`` is 6.0e-03 -
    large enough to be mistaken for a wrong closed form.
    """
    theta = natural_frequency
    zeta = damping
    if theta <= 0.0 or zeta <= 0.0:
        raise ValueError("natural_frequency and damping must both be positive")

    def integrand(f: float) -> float:
        w = 2.0 * math.pi * f
        numer = (2.0 * zeta * theta * w) ** 2 + theta**4
        denom = (theta**2 - w**2) ** 2 + (2.0 * zeta * theta * w) ** 2
        return numer / denom

    upper = limit * theta / (2.0 * math.pi)
    value, _ = quad(integrand, 0.0, upper, limit=800)
    tail = (2.0 * zeta * theta) ** 2 / (4.0 * math.pi**2 * upper)
    return float(value + tail)


def _pole_matched_coefficients(theta: float, zeta: float) -> tuple[float, float]:
    """``(alpha, beta)`` placing the discrete poles at ``exp(s_i T)``.

    See the module docstring.  ``theta`` is ``w_n T`` in radians per symbol and
    ``zeta`` the damping factor.
    """
    if theta <= 0.0 or zeta <= 0.0:
        raise ValueError("theta and zeta must both be positive")
    alpha = 1.0 - math.exp(-2.0 * zeta * theta)
    decay = math.exp(-zeta * theta)
    if zeta < 1.0:
        oscillation = math.cos(theta * math.sqrt(1.0 - zeta * zeta))
    else:
        oscillation = math.cosh(theta * math.sqrt(zeta * zeta - 1.0))
    beta = 2.0 - alpha - 2.0 * decay * oscillation
    return alpha, beta


@dataclass(frozen=True)
class LoopDesign:
    """Coefficients of a second-order timing loop, and the design it came from.

    Attributes
    ----------
    noise_bandwidth
        ``B_n = B_L T``, cycles per symbol.
    damping
        ``zeta``, dimensionless.
    detector_gain
        ``K_d``, detector output per symbol period of timing error.  Measured.
    natural_frequency
        ``theta = w_n T``, radians per symbol.
    k_proportional, k_integral
        ``k1`` and ``k2`` as applied to the raw detector output.  Both already
        carry the ``1 / K_d`` normalisation.
    """

    noise_bandwidth: float
    damping: float
    detector_gain: float
    natural_frequency: float
    k_proportional: float
    k_integral: float

    @classmethod
    def from_bandwidth(
        cls, noise_bandwidth: float, damping: float, detector_gain: float
    ) -> LoopDesign:
        """Design a loop from ``B_n``, ``zeta`` and a **measured** ``K_d``.

        Raises
        ------
        ValueError
            If ``B_n`` is not in ``(0, 0.5)`` cycles per symbol, if ``zeta`` is
            not positive, or if ``detector_gain`` is zero.  A zero gain is a real
            outcome - the early-late gate on a rectangular pulse has
            ``K_d = 0`` - and it is reported as an error here rather than
            producing infinite coefficients.
        """
        if not 0.0 < noise_bandwidth < 0.5:
            raise ValueError(
                f"noise_bandwidth must lie in (0, 0.5) cycles per symbol, got "
                f"{noise_bandwidth!r}"
            )
        if damping <= 0.0:
            raise ValueError(f"damping must be positive, got {damping!r}")
        if detector_gain == 0.0 or not math.isfinite(detector_gain):
            raise ValueError(
                "detector_gain must be a non-zero finite number; a measured gain of "
                "zero means the detector carries no timing information for this "
                "pulse shape and no loop can be designed from it"
            )
        theta = 8.0 * damping * noise_bandwidth / (1.0 + 4.0 * damping**2)
        alpha, beta = _pole_matched_coefficients(theta, damping)
        k1 = alpha / detector_gain
        k2 = beta / detector_gain
        return cls(
            noise_bandwidth=float(noise_bandwidth),
            damping=float(damping),
            detector_gain=float(detector_gain),
            natural_frequency=float(theta),
            k_proportional=float(k1),
            k_integral=float(k2),
        )

    @classmethod
    def from_coefficients(
        cls, k_proportional: float, k_integral: float, detector_gain: float
    ) -> LoopDesign:
        """Recover ``B_n`` and ``zeta`` from the coefficients: the exact inverse.

        Inverts the pole-matching mapping in closed form:
        ``zeta theta = -log(1 - alpha) / 2`` from the constant term, then the
        middle term gives ``theta sqrt(|1 - zeta^2|)`` through an ``arccos`` or an
        ``arccosh``.  ``tests/test_loop.py`` round-trips a grid of designs through
        both directions.
        """
        if k_integral <= 0.0 or k_proportional <= 0.0:
            raise ValueError(
                f"both coefficients must be positive, got k1={k_proportional!r}, "
                f"k2={k_integral!r}"
            )
        alpha = k_proportional * detector_gain
        beta = k_integral * detector_gain
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"k1 * K_d must lie in (0, 1) for a pole-matched design, got {alpha!r}"
            )
        u = -0.5 * math.log1p(-alpha)  # u = zeta * theta
        c = (2.0 - alpha - beta) / (2.0 * math.exp(-u))
        if c >= 1.0:
            inner = math.acosh(c)
            squared = u * u - inner * inner
        else:
            inner = math.acos(max(c, -1.0))
            squared = u * u + inner * inner
        if squared <= 0.0:
            raise ValueError(
                "these coefficients do not correspond to a second-order "
                "pole-matched design; the inverse mapping has no positive solution"
            )
        theta = math.sqrt(squared)
        zeta = u / theta
        return cls(
            noise_bandwidth=noise_bandwidth_closed_form(theta, zeta),
            damping=zeta,
            detector_gain=float(detector_gain),
            natural_frequency=theta,
            k_proportional=float(k_proportional),
            k_integral=float(k_integral),
        )

    @property
    def normalised_coefficients(self) -> tuple[float, float]:
        """``(alpha, beta) = (k1 K_d, k2 K_d)``: the detector-independent pair."""
        return (
            self.k_proportional * self.detector_gain,
            self.k_integral * self.detector_gain,
        )

    def state_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, B)`` of the linearised discrete loop in state ``[eps, w]``.

        ``eps[k+1], w[k+1] = A @ [eps[k], w[k]] + B * n[k]`` where ``n`` is the
        additive detector noise.  Derived in ``docs/TIMING_MODEL.md`` section 4.
        """
        alpha, beta = self.normalised_coefficients
        a = np.array([[1.0 - alpha - beta, 1.0], [-beta, 1.0]])
        b = np.array([[-(self.k_proportional + self.k_integral)], [-self.k_integral]])
        return a, b

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of ``A``.  Both inside the unit circle for a stable loop."""
        a, _ = self.state_matrices()
        return np.linalg.eigvals(a)

    @property
    def is_stable(self) -> bool:
        """True when every closed-loop pole is strictly inside the unit circle."""
        return bool(np.all(np.abs(self.closed_loop_poles) < 1.0))

    def summary(self) -> dict[str, float | bool]:
        """Flat dictionary of the design, for printing and serialising."""
        alpha, beta = self.normalised_coefficients
        return {
            "B_n_cycles_per_symbol": self.noise_bandwidth,
            "zeta": self.damping,
            "K_d": self.detector_gain,
            "theta_rad_per_symbol": self.natural_frequency,
            "k1": self.k_proportional,
            "k2": self.k_integral,
            "alpha": alpha,
            "beta": beta,
            "max_pole_magnitude": float(np.max(np.abs(self.closed_loop_poles))),
            "stable": self.is_stable,
        }


def jitter_variance_closed_form(
    noise_bandwidth: float, detector_gain: float, ted_noise_variance: float
) -> float:
    """Classical loop-noise prediction of the timing jitter variance.

    ``sigma_eps^2 = 2 B_n sigma_n^2 / K_d^2``, in squared symbol periods.

    Parameters
    ----------
    noise_bandwidth
        ``B_n = B_L T``, cycles per symbol.
    detector_gain
        ``K_d``, measured.
    ted_noise_variance
        ``sigma_n^2``: the variance of the detector output at zero timing error,
        **measured** from the same detector and channel as the loop will see.

    Validity
    --------
    The expression assumes (i) the timing error stays inside the S-curve's linear
    range, (ii) the detector noise is white across loop updates, and (iii)
    ``B_n`` is small enough that the discrete loop behaves like the analogue
    prototype.  :func:`jitter_variance_exact` drops (iii) and
    :func:`slotsync.simulate.run_timing_loop` drops all three.
    """
    if ted_noise_variance < 0.0:
        raise ValueError(f"ted_noise_variance must be non-negative, got {ted_noise_variance!r}")
    if detector_gain == 0.0:
        raise ValueError("detector_gain must be non-zero")
    if not 0.0 < noise_bandwidth < 0.5:
        raise ValueError(
            f"noise_bandwidth must lie in (0, 0.5) cycles per symbol, got {noise_bandwidth!r}"
        )
    return 2.0 * noise_bandwidth * ted_noise_variance / detector_gain**2


def jitter_variance_exact(design: LoopDesign, ted_noise_variance: float) -> float:
    """Timing jitter variance of the exact discrete loop, squared symbol periods.

    Solves the discrete Lyapunov equation ``P = A P A' + B B' sigma_n^2`` for the
    stationary state covariance and returns ``P[0, 0]``.  No small-bandwidth
    approximation is made, so the ratio of this to
    :func:`jitter_variance_closed_form` measures how good that approximation is
    at a given ``B_n`` - which is one of the numbers this package reports.
    """
    if ted_noise_variance < 0.0:
        raise ValueError(f"ted_noise_variance must be non-negative, got {ted_noise_variance!r}")
    if not design.is_stable:
        raise ValueError(
            "the loop is unstable (a closed-loop pole lies on or outside the unit "
            f"circle, max |pole| = {float(np.max(np.abs(design.closed_loop_poles))):.6f}); "
            "no stationary jitter variance exists"
        )
    a, b = design.state_matrices()
    q = (b @ b.T) * ted_noise_variance
    p = solve_discrete_lyapunov(a, q)
    return float(p[0, 0])


def _step_difference_variance(design: LoopDesign, ted_noise_variance: float) -> float:
    """Variance of ``eps[k+1] - eps[k]``, the discrete stand-in for ``d eps / dk``."""
    a, b = design.state_matrices()
    q = (b @ b.T) * ted_noise_variance
    p = solve_discrete_lyapunov(a, q)
    alpha, beta = design.normalised_coefficients
    c = np.array([[-(alpha + beta), 1.0]])
    d = -(design.k_proportional + design.k_integral)
    return float((c @ p @ c.T)[0, 0] + d * d * ted_noise_variance)


def loop_snr_db(jitter_variance: float, boundary: float = 0.5) -> float:
    """Loop SNR in dB: ``10 log10(boundary^2 / sigma_eps^2)``.

    ``boundary`` is the timing error at which the detector's S-curve reverses, in
    symbol periods; half a symbol for a one-symbol-periodic S-curve.  Defining
    loop SNR against the actual reversal point rather than against a nominal
    value is what ties the slip rate to the measured S-curve.
    """
    if jitter_variance <= 0.0:
        raise ValueError(f"jitter_variance must be positive, got {jitter_variance!r}")
    if boundary <= 0.0:
        raise ValueError(f"boundary must be positive in symbol periods, got {boundary!r}")
    return float(10.0 * math.log10(boundary**2 / jitter_variance))


def cycle_slip_rate_rice(
    design: LoopDesign,
    ted_noise_variance: float,
    boundary: float = 0.5,
) -> float:
    """Level-crossing estimate of the cycle-slip rate, slips per symbol.

    ``2 * (1 / 2 pi) * (sigma_step / sigma_eps) * exp(-boundary^2 / (2
    sigma_eps^2))`` - the standard Gaussian level-crossing rate for the two
    boundaries ``+-boundary``, with the one-step difference of the timing error
    standing in for its derivative.

    Approximations, all three of them material
    ------------------------------------------
    1. The timing error is treated as a stationary **Gaussian** process, which it
       is only while the loop stays in the S-curve's linear range - exactly the
       condition that fails as a slip approaches.
    2. Every boundary crossing is counted as a slip.  A real loop can cross and
       come back, so this **overestimates** the slip rate.
    3. The discrete step difference replaces the continuous derivative, which is
       only accurate for a loop that moves little per symbol.

    ``validation/validate_cycle_slips.py`` measures the real rate against this
    estimate and reports the ratio, including where it is wrong by an order of
    magnitude.
    """
    sigma2 = jitter_variance_exact(design, ted_noise_variance)
    if sigma2 <= 0.0:
        return 0.0
    step2 = _step_difference_variance(design, ted_noise_variance)
    sigma = math.sqrt(sigma2)
    step = math.sqrt(max(step2, 0.0))
    exponent = -(boundary**2) / (2.0 * sigma2)
    return float(2.0 * (step / sigma) / (2.0 * math.pi) * math.exp(exponent))


def slip_free_seconds(slip_rate_per_symbol: float, symbol_rate_hz: float) -> float:
    """Mean time between slips in seconds, from a per-symbol rate and a symbol rate.

    Returns ``inf`` for a zero rate.  The symbol rate is the caller's; nothing in
    this package assumes one.
    """
    if slip_rate_per_symbol < 0.0:
        raise ValueError(f"slip_rate_per_symbol must be non-negative, got {slip_rate_per_symbol!r}")
    if symbol_rate_hz <= 0.0:
        raise ValueError(f"symbol_rate_hz must be positive, got {symbol_rate_hz!r}")
    if slip_rate_per_symbol == 0.0:
        return float("inf")
    return float(1.0 / (slip_rate_per_symbol * symbol_rate_hz))


def error_autocorrelation_weights(design: LoopDesign, max_lag: int) -> np.ndarray:
    """``rho[j] = sum_m h[m] h[m+j]`` for the loop's noise-to-error impulse response.

    The linearised loop maps the detector noise ``n`` to the timing error through
    ``eps[k] = sum_{m>=1} h[m] n[k-m]`` with ``h[m] = C A^(m-1) B``.  The sums of
    lagged products have a closed form,

        ``rho[j] = C P (A')^j C'``,

    where ``P`` solves the unit-variance Lyapunov equation ``P = A P A' + B B'``.
    Deriving it this way avoids truncating an impulse response that is thousands
    of samples long for a narrow loop.  Derivation in ``docs/TIMING_MODEL.md``
    section 4.2.

    Returns
    -------
    numpy.ndarray
        ``rho[0 .. max_lag]``, squared symbol periods per unit noise variance.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    if not design.is_stable:
        raise ValueError("the loop design is unstable; no stationary correlation exists")
    a, b = design.state_matrices()
    p = solve_discrete_lyapunov(a, b @ b.T)
    c = np.array([[1.0, 0.0]])
    weights = np.empty(max_lag + 1)
    power = np.eye(2)
    for j in range(max_lag + 1):
        weights[j] = float((c @ p @ power @ c.T)[0, 0])
        power = power @ a.T
    return weights


def jitter_variance_coloured(design: LoopDesign, autocovariance: np.ndarray) -> float:
    """Jitter variance for detector noise with a **measured** autocovariance.

    ``sigma_eps^2 = sum_j R_n[j] rho[j]`` over ``j = -J .. J``, i.e.
    ``R_n[0] rho[0] + 2 sum_{j>=1} R_n[j] rho[j]``.

    Parameters
    ----------
    design
        Loop design.
    autocovariance
        ``R_n[0 .. J]``, the autocovariance of the detector output at zero timing
        error, measured by
        :func:`slotsync.simulate.measure_ted_autocovariance`.

    Why this exists
    ---------------
    :func:`jitter_variance_closed_form` and :func:`jitter_variance_exact` both
    assume the detector output noise is **white** across loop updates.  Channel
    noise is close to white, but detector *self-noise* - the data-dependent part
    that survives when the channel noise is switched off - is not: it is
    correlated at the lags over which consecutive updates share samples and share
    data.  For the early-late gate and for Gardner's detector the self-noise
    dominates, and the white-noise prediction then over-estimates the jitter by a
    large factor.  This function is the same calculation without the whiteness
    assumption, and ``validation/validate_jitter_agreement.py`` reports all three
    against the Monte Carlo.

    Returns
    -------
    float
        Squared symbol periods.  Can come out slightly negative if a truncated
        measured autocovariance is not positive semi-definite; the caller is
        expected to report that rather than clip it, so no clipping is done here.
    """
    r = np.asarray(autocovariance, dtype=float)
    if r.ndim != 1 or r.size < 1:
        raise ValueError(
            f"autocovariance must be a 1-D array with at least one entry, got {r.shape}"
        )
    weights = error_autocorrelation_weights(design, r.size - 1)
    return float(r[0] * weights[0] + 2.0 * np.dot(r[1:], weights[1:]))
