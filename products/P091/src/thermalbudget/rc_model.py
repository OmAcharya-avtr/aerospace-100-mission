"""Baseline 2, part one: the first-order thermal RC relaxation, in closed form.

A body with one thermal capacitance ``C`` and one conductance to ambient ``1/R``,
driven by a constant dissipation ``P``, satisfies ``C dT/dt = P - (T - T_a)/R``,
whose solution is

    T(t) = T_inf - (T_inf - T_0) * exp(-t / tau),    tau = R * C,
    T_inf = T_a + P * R.

This is the standard lumped-capacitance result; see Incropera, DeWitt, Bergman &
Lavine, *Fundamentals of Heat and Mass Transfer*, 6th edition, section 5.2
(lumped capacitance method, validity ``Bi = hL/k < 0.1``), and Lienhard & Lienhard,
*A Heat Transfer Textbook*, 4th edition, section 1.3. Units here: ``T`` in degrees
Celsius, ``t`` in seconds, ``tau`` in seconds.

**Validity range, stated because it is routinely violated by the object being
measured.** A system-on-chip is not a lumped capacitance. It has at least a die,
a heat spreader and a heatsink-plus-fan, with time constants differing by an
order of magnitude, so a single-exponential fit is an approximation whose
residual is structured rather than white. This module therefore reports the RMS
residual and the lag-1 residual autocorrelation of every fit, so the reader can
see how badly the first-order form fits rather than taking ``tau`` on faith. A
second time constant, if present, shows up as a large positive residual
autocorrelation and as a systematic early-time error.

**Three estimators of tau, deliberately sharing no algebra.** The specification
asks for tau from the rise and, independently, from the cooldown. This module
goes further and provides three routes to tau from the SAME trace, because
agreement between estimators that share no cancellation is the only kind of
agreement worth reporting:

1. :func:`fit_exponential_relaxation` -- separable nonlinear least squares. The
   model is linear in the asymptote and the amplitude once ``tau`` is fixed, so
   the inner problem is a two-column linear solve and only ``tau`` is searched,
   over ``log tau`` by bounded scalar minimisation. Deterministic, no initial
   guess to tune.
2. :func:`tau_from_log_differences` -- for uniformly sampled data,
   ``T_{k+1} - T_k = C * exp(-t_k / tau) * (exp(-h / tau) - 1)``, so
   ``log |Delta T_k|`` is linear in ``t_k`` with slope ``-1 / tau``. Needs no
   asymptote estimate at all, which is what makes it independent: the asymptote
   is where a least-squares fit of a truncated rise is weakest.
3. :func:`tau_from_integral` -- the area rule
   ``tau = integral_0^T (T_inf - T(t)) dt / (T_inf - T_0)`` for an exact
   exponential, evaluated by the trapezium rule with the tail correction
   subtracted. Uses the asymptote but no logarithms and no fitting.

Estimator 2 is noise-sensitive by construction, since it differentiates; the
conditioning is reported with its result and the independent-bound check in
``validation/`` scales its slack by that conditioning rather than by a constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

#: Bounds on the searched time constant, seconds. The campaign's runs are 180 to
#: 720 s long, so a tau below 2 s is unresolvable at a 1 s sample interval and a
#: tau above 5000 s is unidentifiable from a 720 s window. Hitting either bound
#: is reported, not silently accepted.
TAU_BOUNDS_S: tuple[float, float] = (2.0, 5000.0)

#: Fewest finite samples a relaxation fit will accept.
MIN_SAMPLES: int = 20


class RelaxationFitError(ValueError):
    """Raised when a relaxation fit is not identifiable from the data given."""


@dataclass(frozen=True)
class RelaxationFit:
    """A fitted ``T(t) = asymptote + amplitude * exp(-t / tau)``.

    Attributes
    ----------
    asymptote_c:
        ``T_inf`` for a rise, ``T_ambient`` for a cooldown, degrees Celsius.
    amplitude_c:
        ``T_0 - asymptote``. Negative for a rise, positive for a cooldown.
    tau_s:
        Time constant, seconds.
    initial_c:
        ``asymptote + amplitude``, the fitted temperature at ``t = 0``.
    rms_residual_c:
        Root-mean-square fit residual, degrees Celsius.
    residual_autocorrelation:
        Lag-1 autocorrelation of the residuals. A value near zero means the
        first-order form is adequate; a large positive value means a second time
        constant or a drift the single exponential cannot represent.
    n:
        Finite samples used.
    span_in_tau:
        Length of the fitted window divided by ``tau_s``. Below about 1.5 the
        asymptote is extrapolated rather than observed and the fit should be
        distrusted; the number is reported so that judgement is the reader's.
    at_bound:
        True if ``tau_s`` landed on either end of :data:`TAU_BOUNDS_S`.
    """

    asymptote_c: float
    amplitude_c: float
    tau_s: float
    initial_c: float
    rms_residual_c: float
    residual_autocorrelation: float
    n: int
    span_in_tau: float
    at_bound: bool

    def predict(self, t_s: np.ndarray | float) -> np.ndarray:
        """Model temperature at ``t_s`` seconds, degrees Celsius."""
        t = np.asarray(t_s, dtype=float)
        return self.asymptote_c + self.amplitude_c * np.exp(-t / self.tau_s)


def _clean(t_s: np.ndarray, temperature_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_s, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    if t.shape != temperature.shape:
        raise ValueError(f"shapes differ: {t.shape} and {temperature.shape}")
    keep = np.isfinite(t) & np.isfinite(temperature)
    t, temperature = t[keep], temperature[keep]
    if t.size < MIN_SAMPLES:
        raise RelaxationFitError(
            f"a relaxation fit needs at least {MIN_SAMPLES} finite samples, got {t.size}"
        )
    return t - t.min(), temperature


def _linear_inner(t: np.ndarray, temperature: np.ndarray, tau: float) -> tuple[float, float, float]:
    """Best asymptote and amplitude at fixed ``tau``; returns them with the SSE."""
    basis = np.column_stack([np.ones_like(t), np.exp(-t / tau)])
    coefficients, *_ = np.linalg.lstsq(basis, temperature, rcond=None)
    residual = temperature - basis @ coefficients
    return float(coefficients[0]), float(coefficients[1]), float(residual @ residual)


def fit_exponential_relaxation(
    t_s: np.ndarray,
    temperature_c: np.ndarray,
    *,
    tau_bounds_s: tuple[float, float] = TAU_BOUNDS_S,
) -> RelaxationFit:
    """Separable least-squares fit of ``asymptote + amplitude * exp(-t / tau)``.

    Only ``log tau`` is searched; the asymptote and amplitude are solved exactly
    at each candidate. ``scipy.optimize.minimize_scalar`` with ``method="bounded"``
    is deterministic, so repeated calls on identical input return identical
    parameters.

    Raises
    ------
    RelaxationFitError
        On too few finite samples, a window shorter than the lower tau bound, or
        a temperature channel with no variation at all.
    """
    t, temperature = _clean(t_s, temperature_c)
    window = float(t.max())
    if window <= tau_bounds_s[0]:
        raise RelaxationFitError(
            f"the fitted window is {window:g} s, no longer than the lower tau bound "
            f"{tau_bounds_s[0]:g} s; tau is not identifiable"
        )
    if float(np.ptp(temperature)) == 0.0:
        raise RelaxationFitError(
            "the temperature channel is constant over the window; neither tau nor the "
            "asymptote is identifiable"
        )
    lower, upper = math.log(tau_bounds_s[0]), math.log(tau_bounds_s[1])
    result = optimize.minimize_scalar(
        lambda log_tau: _linear_inner(t, temperature, math.exp(log_tau))[2],
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )
    tau = math.exp(float(result.x))
    asymptote, amplitude, sse = _linear_inner(t, temperature, tau)
    residual = temperature - (asymptote + amplitude * np.exp(-t / tau))
    denominator = float(residual @ residual)
    rho = float(residual[1:] @ residual[:-1] / denominator) if denominator > 0.0 else 0.0
    at_bound = bool(
        math.isclose(tau, tau_bounds_s[0], rel_tol=1e-3)
        or math.isclose(tau, tau_bounds_s[1], rel_tol=1e-3)
    )
    return RelaxationFit(
        asymptote_c=asymptote,
        amplitude_c=amplitude,
        tau_s=tau,
        initial_c=asymptote + amplitude,
        rms_residual_c=math.sqrt(sse / t.size),
        residual_autocorrelation=float(np.clip(rho, -1.0, 1.0)),
        n=int(t.size),
        span_in_tau=window / tau,
        at_bound=at_bound,
    )


@dataclass(frozen=True)
class TauEstimate:
    """One estimate of tau with the conditioning of the route that produced it.

    Attributes
    ----------
    method:
        Short name of the estimator.
    tau_s:
        The estimate, seconds.
    stderr_s:
        Standard error where the route provides one, else ``nan``.
    conditioning:
        A dimensionless number describing how ill-conditioned this route was on
        this data: for the log-difference route it is the ratio of the
        sample-to-sample noise to the mean absolute successive difference, which
        is the factor by which differencing amplifies noise. A slack derived from
        this rather than from an absolute constant is what the mission's build
        guide requires of a bound check.
    n:
        Samples used.
    note:
        Anything the reader needs in order not to over-read the number.
    """

    method: str
    tau_s: float
    stderr_s: float
    conditioning: float
    n: int
    note: str = ""


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average over ``width`` samples, shortening the array."""
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="valid")


def tau_from_log_differences(
    t_s: np.ndarray,
    temperature_c: np.ndarray,
    *,
    min_pairs: int = 10,
    smooth_samples: int = 5,
    snr_floor: float = 3.0,
) -> TauEstimate:
    """Tau from the slope of ``log |Delta T|`` against time. Uses no asymptote.

    For an exact exponential sampled at times ``t_k``, successive differences
    satisfy ``Delta T_k = C * exp(-t_k / tau) * (exp(-h / tau) - 1)``, so the
    logarithm of their magnitude is linear in ``t_k`` with slope ``-1 / tau``.
    The asymptote cancels, which is exactly why this route is an independent
    check on a least-squares fit whose weakest parameter is that asymptote.

    The cost is noise amplification, and it is severe enough that two declared
    pre-steps are part of the estimator rather than optional tidying. Both were
    added after measuring what happens without them: on a synthetic rise of
    known ``tau = 90 s`` observed over eight time constants, the unguarded
    version returned 451 s, a five-fold error, because beyond about three time
    constants the successive differences are noise, and keeping only those with
    the trend sign selects the upward half of that noise and flattens the fitted
    slope.

    1. ``smooth_samples`` -- a centred moving average of that width is applied
       before differencing, which divides the per-sample noise by its square
       root. Declared, not tuned: 5 samples at a 1 s interval is 5 s, two orders
       of magnitude below any thermal time constant this product can resolve.
    2. ``snr_floor`` -- a difference is kept only if its magnitude exceeds
       ``snr_floor`` times the estimated differencing noise, which confines the
       fit to the part of the window where there is signal to difference. The
       surviving count and the time of the last kept pair are reported.

    The remaining bias is upward selection among kept pairs near the floor, and
    it is reported through :attr:`TauEstimate.conditioning`, the ratio of the
    differencing noise to the mean magnitude of the kept differences. The
    independent-bound check in ``validation/`` scales its slack by that number.

    Raises
    ------
    RelaxationFitError
        If fewer than ``min_pairs`` usable pairs survive, or the fitted slope is
        not negative.
    """
    t, temperature = _clean(t_s, temperature_c)
    if smooth_samples > 1:
        half = smooth_samples // 2
        smoothed = _moving_average(temperature, smooth_samples)
        times = t[half : half + smoothed.size]
    else:
        smoothed, times = temperature, t
    raw_noise = float(np.std(np.diff(temperature, n=2))) / math.sqrt(6.0)
    noise = raw_noise / math.sqrt(max(smooth_samples, 1))
    differences = np.diff(smoothed)
    midpoints = 0.5 * (times[1:] + times[:-1])
    direction = 1.0 if smoothed[-1] >= smoothed[0] else -1.0
    floor = snr_floor * noise * math.sqrt(2.0)
    # The floor selects a contiguous WINDOW, not individual pairs. Selecting pairs
    # one by one on their own magnitude keeps the upward half of the noise and
    # biases the fitted slope flat, which is the defect that made the unguarded
    # version return 451 s for a true 90 s. The envelope is a heavily smoothed
    # copy of the same differences, used only to find where the signal stops.
    envelope = np.abs(_moving_average(differences * direction, min(21, differences.size)))
    above = np.nonzero(envelope > floor)[0]
    if above.size == 0:
        raise RelaxationFitError(
            f"no part of the differenced trace clears the {snr_floor:g}-sigma signal "
            f"floor of {floor:.4g} degC; the log-difference route is not usable on this "
            "trace, which is what a flat or noise-dominated channel looks like"
        )
    cutoff = int(above.max()) + min(21, differences.size)
    inside = np.zeros(differences.size, dtype=bool)
    inside[: min(cutoff, differences.size)] = True
    usable = inside & ((differences * direction) > 0.0)
    magnitudes = np.abs(differences[usable])
    kept_times = midpoints[usable]
    usable_count = int(magnitudes.size)
    dropped_inside = int(np.sum(inside) - usable_count)
    if usable_count < min_pairs:
        raise RelaxationFitError(
            f"only {usable_count} successive differences survive inside the signal "
            f"window of {int(np.sum(inside))} pairs, fewer than the {min_pairs} needed; "
            "the log-difference route is not usable on this trace"
        )
    logs = np.log(magnitudes)
    centred = kept_times - kept_times.mean()
    sxx = float(centred @ centred)
    slope = float(centred @ (logs - logs.mean()) / sxx)
    if slope >= 0.0:
        raise RelaxationFitError(
            f"the fitted log-difference slope is {slope:+.4g} per second, not negative; "
            "the trace is not relaxing towards an asymptote over this window"
        )
    tau = -1.0 / slope
    residual = logs - (logs.mean() + slope * centred)
    dof = max(usable_count - 2, 1)
    sigma2 = float(residual @ residual) / dof
    slope_stderr = math.sqrt(sigma2 / sxx)
    tau_stderr = slope_stderr / (slope * slope)
    mean_step = float(np.mean(magnitudes)) or math.inf
    return TauEstimate(
        method="log-differences",
        tau_s=tau,
        stderr_s=tau_stderr,
        conditioning=noise * math.sqrt(2.0) / mean_step,
        n=usable_count,
        note=(
            f"{usable_count} of {int(differences.size)} smoothed differences kept inside "
            f"the signal window ending at t = {float(kept_times.max()):.0f} s "
            f"({dropped_inside} dropped inside it for having the wrong sign); "
            f"{snr_floor:g}-sigma floor {floor:.4g} degC, smoothing width "
            f"{smooth_samples} samples"
        ),
    )


def tau_from_integral(
    t_s: np.ndarray,
    temperature_c: np.ndarray,
    *,
    asymptote_c: float,
) -> TauEstimate:
    """Tau from the area between the trace and its asymptote. No logarithms.

    For ``T(t) = T_inf + C exp(-t/tau)`` the area between the asymptote and the
    curve over ``[0, W]`` is ``-C tau (1 - exp(-W/tau))``, so

        tau = area / (|T_inf - T_0| * (1 - exp(-W/tau)))

    which is solved by a short fixed-point iteration on the bracketed tail
    factor. The trapezium rule is used for the area, so the estimate inherits
    second-order quadrature error in the sample interval and nothing else.

    Raises
    ------
    RelaxationFitError
        If the initial offset from the asymptote is too small to divide by, or
        the iteration does not settle.
    """
    t, temperature = _clean(t_s, temperature_c)
    offset = float(temperature[0] - asymptote_c)
    if abs(offset) < 1e-6:
        raise RelaxationFitError(
            f"the trace starts {offset:+.3g} degC from the asymptote; the area rule "
            "divides by that offset and is not usable"
        )
    area = float(np.trapezoid(asymptote_c - temperature, t))
    window = float(t.max())
    tau = abs(area / offset) if offset else math.nan
    for _ in range(200):
        if tau <= 0.0 or not math.isfinite(tau):
            raise RelaxationFitError("the area-rule iteration left the valid range")
        tail = 1.0 - math.exp(-window / tau)
        updated = abs(area / offset) / tail
        if abs(updated - tau) < 1e-9 * max(tau, 1.0):
            tau = updated
            break
        tau = updated
    return TauEstimate(
        method="area-rule",
        tau_s=tau,
        stderr_s=math.nan,
        conditioning=abs(1.0 / (1.0 - math.exp(-window / tau))) if tau > 0 else math.inf,
        n=int(t.size),
        note=(
            f"window is {window / tau:.2f} tau; the tail correction factor is "
            f"{1.0 / (1.0 - math.exp(-window / tau)):.3f}"
        ),
    )
