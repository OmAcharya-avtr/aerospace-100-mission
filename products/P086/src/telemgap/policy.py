"""The gap policy: which gaps may be interpolated across, which may not, and why.

The whole product exists for this module. Every analysis downstream of a
telemetry archive interpolates; the interpolation is usually fine and
occasionally invents a signal across an outage. What is missing from the normal
practice is not better interpolation, it is a **statement of permission that
can be refused**, together with the bound the permission rests on and an
explicit account of whether that bound was declared or guessed.

Equations
---------
Linear interpolation between two samples separated by ``h`` seconds, for a
signal twice continuously differentiable on the interval with
``|d^2 v / d t^2| <= M2``:

    max |v(t) - L(t)| <= M2 h^2 / 8                                     (3)

Standard Lagrange remainder for a first-degree interpolating polynomial;
Hildebrand, *Introduction to Numerical Analysis*, 2nd ed., Dover 1987, §3.2, or
Atkinson, *An Introduction to Numerical Analysis*, 2nd ed., Wiley 1989, §3.2.
Units: ``M2`` in channel units per second squared, ``h`` in seconds, bound in
channel units. Validity: ``v`` in ``C^2`` on the closed gap interval, and the
two endpoint values **exact**. It is not valid across a discontinuity, and a
gap is precisely where a discontinuity cannot be ruled out, which is what the
``gap_interior`` determinability limit says.

Zero-order hold from the left sample, for ``|d v / d t| <= M1``:

    max |v(t) - v(t_0)| <= M1 h                                         (4)

Same sources, first-degree Taylor remainder. Units: ``M1`` in channel units per
second.

**Equations (3) and (4) assume exact endpoints, and a telemetry sample is not
exact.** An interpolation anchored on measured samples inherits their noise:
the realised error is bounded by the interpolation term plus the endpoint noise
contribution, and for Gaussian noise the second term has no deterministic
bound at all. The policy therefore acts on

    predicted = (interpolation bound) + k * sigma                       (3')

with ``sigma`` the channel's measurement noise standard deviation and ``k`` a
declared coverage factor, default 3. **Equation (3') is a coverage statement,
not a bound**, in the sense of the ISO/IEC *Guide to the Expression of
Uncertainty in Measurement* (JCGM 100:2008 §6.3), and it is labelled that way
everywhere it is reported. The deterministic part of it, equation (3) against
exact endpoints, is checked as a theorem in
``validation/validate_interpolation_policy.py``; the coverage part is checked
by measuring the exceedance rate.

An **independently derived** bound on ``M2``, used only as a cross-check and
never by the policy itself, comes from Bernstein's inequality for band-limited
functions: if ``v`` is band-limited to ``|f| <= B`` hertz and ``|v| <= A``,

    |d^k v / d t^k| <= (2 pi B)^k A                                     (8)

Boas, *Entire Functions*, Academic Press 1954, Theorem 11.1.2. Equations (3)
and (8) share no mathematics -- (3) is a Taylor remainder on a real interval,
(8) is a complex-analytic bound on an entire function of exponential type --
which is why validation uses (8) to check (3) rather than checking (3) against
another difference formula.

Estimating the curvature, and knowing when you cannot
-----------------------------------------------------
``M2`` is taken from the channel's declared ``max_curvature`` when that is
non-zero. When it is zero, which is the common operational case, it is
estimated from the record. A plain second difference is useless for this: for
white measurement noise of standard deviation ``sigma`` the second difference
has standard deviation ``sqrt(6) sigma``, so dividing by ``T^2`` yields an
estimate dominated by noise by orders of magnitude. The estimator here instead
fits a least-squares quadratic over a centred window of ``2m + 1`` samples and
takes its second derivative,

    M2_est = quantile_q( |2 a_2(i)| ) / T^2                             (9)

with ``a_2(i)`` the quadratic coefficient of the fit centred at sample ``i``.
This is the Savitzky-Golay derivative filter (Savitzky & Golay 1964,
*Analytical Chemistry* 36(8), 1627-1633). Outliers are replaced by their local
median first, using equation (5), so that the anomalies the record contains do
not set the curvature of the signal it carries.

Two properties of that estimator are computed and returned, because without
them the estimate is not interpretable:

**Its noise floor.** The second-derivative estimate of pure noise has standard
deviation ``2 sigma sqrt(sum c_j^2) / T^2`` with ``c_j`` the fit's quadratic
coefficients, so the ``q``-quantile of its absolute value is ``z_q`` times
that, with ``z_q = sqrt(2) erf^-1(q)``. An estimate below
:data:`MEASURABILITY_FACTOR` times that floor **is the floor**, and the policy
declares the curvature *not measurable from this record* and refuses the linear
method rather than acting on a number that is its own noise.

**Its bandwidth.** The filter attenuates curvature it cannot resolve. Its
transfer function relative to the exact ``-(2 pi f)^2`` is computed in closed
form and :attr:`CurvatureEstimate.bandwidth_hz` reports the frequency at which
it falls to one half. Curvature carried above that frequency is invisible to
equation (9), so **for a channel with near-Nyquist content the estimate
under-predicts, by a factor derivable in closed form, and equation (3) built on
it can be violated.** That is measured on a channel constructed for the
purpose, published as a headline result, and it is the reason this package's
position is: declare the bound, or accept the refusal.

The endpoint gate
-----------------
Equations (3) and (4) bound the error of an interpolation **anchored on the
values at the ends of the gap**. If one of those two frames is itself corrupt,
misparsed or undecided, the anchor is wrong by an unbounded amount and no
curvature bound can rescue it: a single bit error in the endpoint word moves
the whole chord. The policy therefore refuses a gap whose bounding frame
carries an unresolved integrity verdict on that channel, which is the point at
which the integrity classification of :mod:`telemgap.classify` stops being a
report and becomes an input. The size of the error this gate prevents is
measured in ``validation/validate_interpolation_policy.py``.

The Nyquist gate
----------------
The sampling theorem (Shannon 1949, *Proc. IRE* 37(1), 10-21) guarantees
reconstruction of a signal band-limited below ``1 / (2 T)`` from samples at
spacing ``T``. Inside a gap of duration ``h = k T`` the local spacing is
``k T``, supporting only ``1 / (2 k T)``; content between the two cannot be
recovered at any interpolation order. The policy carries a declared
``max_gap_slots`` gate that refuses on this ground alone. Its default of 4 is a
**policy choice, not a theorem**, is reported as such, and does not replace
equation (3') -- both gates must pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .classify import IntegrityReport
from .record import TelemetryRecord
from .screens import MAD_TO_SIGMA, Gap, robust_local_statistics

#: Default quantile of equation (9).
DEFAULT_CURVATURE_QUANTILE = 0.99

#: Default half-window ``m`` of the quadratic fit in equation (9). With
#: ``m = 7`` the noise gain ``2 sqrt(sum c_j^2)`` is 0.0311 and the filter's
#: half-power frequency is at 0.0962 of the sampling frequency; both numbers
#: are computed by :func:`quadratic_second_derivative_coefficients` and
#: :func:`estimator_bandwidth_fraction` rather than quoted from here.
DEFAULT_HALF_WINDOW = 7

#: Multiple of the quantile-matched noise floor an estimate must exceed before
#: the policy will treat it as a measurement of curvature rather than of noise.
MEASURABILITY_FACTOR = 2.0

#: Default coverage factor ``k`` of equation (3'). Three standard deviations.
DEFAULT_COVERAGE_K = 3.0

#: Default Nyquist gate, in cadence slots. A policy choice, not a theorem.
DEFAULT_MAX_GAP_SLOTS = 4.0

#: Share of a channel's variance that may sit above the equation (9)
#: estimator's half-power frequency before its curvature estimate is declared
#: unresolvable. A declared policy parameter, not a theorem. The default of 1
#: per cent was chosen because it refuses the one constructed channel in
#: telemgap.dataset whose curvature the estimator demonstrably under-predicts,
#: and admits the three it does not; the measurement is in
#: validation/validate_interpolation_policy.py and the number is reported with
#: the measurement rather than defended here.
DEFAULT_RESOLVABILITY_THRESHOLD = 0.01

#: Robust z above which a sample is replaced by its local median before
#: equation (9) is evaluated, so that the record's own anomalies do not set the
#: curvature of the signal it carries.
CURVATURE_OUTLIER_Z = 8.0

#: Interpolation methods the policy can permit, in preference order.
METHODS = ("linear", "hold")


@dataclass(frozen=True)
class ChannelTolerance:
    """Declared reconstruction tolerance for one channel.

    Attributes
    ----------
    channel_name
        Channel the tolerance applies to.
    tolerance
        Maximum reconstruction error the caller is willing to accept inside a
        gap, in the channel's declared unit. Must be positive and finite.
    """

    channel_name: str
    tolerance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError(
                f"tolerance for {self.channel_name!r} must be positive and finite, "
                f"got {self.tolerance!r}"
            )


@dataclass(frozen=True)
class CurvatureEstimate:
    """A curvature bound with everything needed to decide whether to believe it.

    Attributes
    ----------
    value
        ``M2`` the policy will act on, channel units per second squared.
    source
        ``"declared"`` or ``"estimated"``.
    noise_floor
        Quantile-matched noise floor of equation (9) at the same quantile,
        channel units per second squared. Zero for a declared value.
    sigma
        Measurement noise standard deviation used, channel units.
    sigma_source
        ``"declared"`` or ``"estimated"``.
    measurable
        Whether ``value`` exceeds :data:`MEASURABILITY_FACTOR` times
        ``noise_floor``. Always ``True`` for a declared value.
    bandwidth_hz
        Half-power frequency of the estimator, hertz. Curvature above it is
        attenuated and therefore under-reported. ``inf`` for a declared value.
    half_window
        ``m`` of equation (9).
    high_frequency_fraction
        Share of the channel's variance sitting above ``bandwidth_hz``,
        equation (17). Zero for a declared value.
    resolvable
        Whether that share is below the declared resolvability threshold. A
        curvature estimate on a channel whose content lives above the
        estimator's own bandwidth is **biased, not noisy**, so the noise floor
        does not catch it and ``measurable`` can be ``True`` while the estimate
        is wrong by two orders of magnitude. That failure is measured in
        ``validation/validate_interpolation_policy.py``; this flag is what
        stops the policy acting on it.
    """

    value: float
    source: str
    noise_floor: float
    sigma: float
    sigma_source: str
    measurable: bool
    bandwidth_hz: float
    half_window: int
    high_frequency_fraction: float = 0.0
    resolvable: bool = True


@dataclass(frozen=True)
class GateOutcome:
    """One gate's verdict for one gap and channel."""

    gate: str
    passed: bool
    value: float
    limit: float
    statement: str


@dataclass(frozen=True)
class GapDecision:
    """The policy's verdict for one gap and one channel.

    Attributes
    ----------
    decision
        ``"interpolate"`` or ``"refuse"``. The refusal is the product's output,
        not an error.
    method
        The permitted method chosen, or ``"none"`` when refused.
    predicted_error
        Equation (3') for the chosen method, channel units: the interpolation
        bound plus the coverage term. ``inf`` when refused.
    gates
        Every gate evaluated, passed or failed, for the audit trail.
    """

    gap_index: int
    channel: int
    channel_name: str
    duration_s: float
    unobserved_s: float
    slots: float
    tolerance: float
    curvature: CurvatureEstimate
    slew: float
    coverage: float
    bound_linear: float
    bound_hold: float
    predicted_linear: float
    predicted_hold: float
    decision: str
    method: str
    predicted_error: float
    margin: float
    reason: str
    gates: tuple[GateOutcome, ...]

    @property
    def refused(self) -> bool:
        """Whether the policy refused to interpolate this gap for this channel."""
        return self.decision == "refuse"

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the gates that did not pass."""
        return tuple(gate.gate for gate in self.gates if not gate.passed)


@dataclass
class PolicyPlan:
    """Every per-gap, per-channel decision, plus the declared policy parameters."""

    decisions: tuple[GapDecision, ...]
    max_gap_slots: float
    curvature_quantile: float
    coverage_k: float
    half_window: int
    curvatures: dict[str, CurvatureEstimate] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)

    @property
    def refusals(self) -> tuple[GapDecision, ...]:
        """Decisions that refused."""
        return tuple(decision for decision in self.decisions if decision.refused)

    @property
    def permissions(self) -> tuple[GapDecision, ...]:
        """Decisions that permitted interpolation."""
        return tuple(decision for decision in self.decisions if not decision.refused)

    def summary(self) -> dict[str, int]:
        """Counts for reports and regression tests."""
        tally = {
            "decisions": len(self.decisions),
            "interpolate": len(self.permissions),
            "refuse": len(self.refusals),
        }
        for method in METHODS:
            tally[f"method_{method}"] = sum(
                1 for decision in self.decisions if decision.method == method
            )
        for gate in (
            "endpoints_clean",
            "nyquist_slots",
            "curvature_measurable",
            "curvature_resolvable",
            "curvature_bound",
            "slew_bound",
            "tolerance_above_noise",
        ):
            tally[f"failed_{gate}"] = sum(
                1 for decision in self.decisions if gate in decision.failed_gates
            )
        return tally

    def for_channel(self, channel_name: str) -> tuple[GapDecision, ...]:
        """Decisions for one channel, in gap order."""
        return tuple(
            decision for decision in self.decisions if decision.channel_name == channel_name
        )


def linear_error_bound(curvature: float, duration_s: float) -> float:
    """Equation (3): ``M2 h^2 / 8``, channel units. Both arguments non-negative."""
    if curvature < 0.0:
        raise ValueError(f"curvature must be non-negative, got {curvature!r}")
    if duration_s < 0.0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s!r}")
    return float(curvature) * float(duration_s) ** 2 / 8.0


def hold_error_bound(slew: float, duration_s: float) -> float:
    """Equation (4): ``M1 h``, channel units. Both arguments non-negative."""
    if slew < 0.0:
        raise ValueError(f"slew must be non-negative, got {slew!r}")
    if duration_s < 0.0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s!r}")
    return float(slew) * float(duration_s)


def bernstein_curvature(amplitude: float, bandwidth_hz: float) -> float:
    """Equation (8) at ``k = 2``: ``(2 pi B)^2 A``, channel units per second squared.

    Independent of equations (3) and (9); used by validation as a cross-check
    on the curvature the policy acts on, never by the policy itself.
    """
    if amplitude < 0.0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude!r}")
    if bandwidth_hz < 0.0:
        raise ValueError(f"bandwidth_hz must be non-negative, got {bandwidth_hz!r}")
    return float(amplitude) * (2.0 * np.pi * float(bandwidth_hz)) ** 2


def half_normal_quantile(q: float) -> float:
    """``z`` such that ``P(|X| <= z) = q`` for a standard normal ``X``.

    Equals ``sqrt(2) erf^-1(q)``. Solved by bisection on :func:`math.erf` to
    35 iterations, which is below the resolution of a double, rather than by
    importing an inverse error function; ``scipy`` is not a dependency of this
    package and a closed-form rational approximation would need its own error
    analysis.
    """
    if not 0.0 <= q < 1.0:
        raise ValueError(f"q must lie in [0, 1), got {q!r}")
    low, high = 0.0, 40.0
    for _ in range(35):
        middle = 0.5 * (low + high)
        if math.erf(middle / math.sqrt(2.0)) < q:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def quadratic_second_derivative_coefficients(half_window: int) -> np.ndarray:
    """Coefficients ``c_j`` with ``a_2 = sum_j c_j v_j`` for a centred quadratic fit.

    The second derivative estimate is ``2 a_2 / T^2``. Derived from the
    pseudo-inverse of the Vandermonde matrix over offsets
    ``-m .. m``, which is the Savitzky-Golay construction.
    """
    if half_window < 2:
        raise ValueError(f"half_window must be at least 2, got {half_window!r}")
    offsets = np.arange(-half_window, half_window + 1, dtype=np.float64)
    design = np.column_stack((np.ones_like(offsets), offsets, offsets**2))
    return np.linalg.pinv(design)[2]


def estimator_transfer(half_window: int, frequency_times_cadence: float) -> float:
    """Magnitude of the equation (9) filter relative to the exact ``-(2 pi f)^2``.

    A value of 1 means the estimator reports the true curvature of a sinusoid
    at that frequency; a value below 1 means it under-reports it by that
    factor. ``frequency_times_cadence`` is the dimensionless ``f T``.
    """
    if frequency_times_cadence <= 0.0:
        return 1.0
    coefficients = quadratic_second_derivative_coefficients(half_window)
    offsets = np.arange(-half_window, half_window + 1, dtype=np.float64)
    response = 2.0 * float(
        np.sum(coefficients * np.cos(2.0 * np.pi * frequency_times_cadence * offsets))
    )
    exact = -((2.0 * np.pi * frequency_times_cadence) ** 2)
    return float(abs(response / exact))


def estimator_bandwidth_fraction(half_window: int) -> float:
    """Lowest ``f T`` at which :func:`estimator_transfer` falls to one half.

    Found by scanning a fine grid and bisecting the first crossing, so the
    answer is a measurement of the filter rather than a formula asserted about
    it. Returns 0.5 when no crossing exists below Nyquist.
    """
    grid = np.linspace(1e-4, 0.5, 4000)
    values = np.array([estimator_transfer(half_window, point) for point in grid])
    below = np.flatnonzero(values < 0.5)
    if below.size == 0:
        return 0.5
    index = int(below[0])
    low = grid[index - 1] if index > 0 else grid[0]
    high = grid[index]
    for _ in range(40):
        middle = 0.5 * (low + high)
        if estimator_transfer(half_window, middle) < 0.5:
            high = middle
        else:
            low = middle
    return float(0.5 * (low + high))


def estimate_noise_sigma(values: np.ndarray) -> float:
    """Robust measurement-noise standard deviation from second differences.

    For white noise of standard deviation ``sigma`` the second difference has
    variance ``6 sigma^2``, so

        sigma_hat = 1.4826 * MAD(second differences) / sqrt(6)           (16)

    with 1.4826 the Gaussian consistency constant (Rousseeuw & Croux 1993). The
    MAD rather than the standard deviation, so that the record's own anomalies
    do not set its noise scale.

    **This estimator is confounded with signal content near the sampling
    frequency**, because a second difference cannot tell a fast sinusoid from
    noise. The size of that confusion is measured per channel in
    ``validation/validate_interpolation_policy.py`` and is one reason the
    policy prefers a declared ``noise_sigma``.
    """
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.size < 3:
        return 0.0
    second = series[2:] - 2.0 * series[1:-1] + series[:-2]
    median = float(np.median(second))
    mad = float(np.median(np.abs(second - median)))
    return float(MAD_TO_SIGMA * mad / math.sqrt(6.0))


def high_frequency_variance_fraction(
    values: np.ndarray,
    cadence_s: float,
    cutoff_hz: float,
    span: float,
    half_window: int = DEFAULT_HALF_WINDOW,
    outlier_z: float = CURVATURE_OUTLIER_Z,
) -> float:
    """Share of a channel's variance above ``cutoff_hz``, equation (17).

        fraction = sum_{f > cutoff} |V(f)|^2 / sum_{f > 0} |V(f)|^2         (17)

    with ``V`` the discrete Fourier transform of the outlier-cleaned,
    linearly detrended, Hann-windowed series. The detrend removes the DC and
    ramp terms, which otherwise dominate the denominator and make every channel
    look low-frequency; the Hann window is applied because a rectangular window
    leaks a strong low-frequency term across the whole spectrum at a rate that
    swamps a small high-frequency component (Harris 1978, *Proc. IEEE* 66(1),
    51-83).

    This is an estimate with its own bias: the window broadens every line by
    about two bins, so a component just below the cutoff contributes a little
    above it. The consequence is that the fraction is slightly overstated,
    which makes the gate that uses it slightly conservative, which is the
    direction a refusal policy should err in. Returns ``0.0`` for a series too
    short or too flat to transform.
    """
    if cadence_s <= 0.0:
        raise ValueError(f"cadence_s must be positive, got {cadence_s!r}")
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz!r}")
    series = np.asarray(values, dtype=np.float64)
    if series.size < 4 * half_window:
        return 0.0
    medians, scaled = robust_local_statistics(series, half_window, span)
    with np.errstate(invalid="ignore"):
        z = np.abs(series - medians) / scaled
    cleaned = np.where(np.isfinite(series) & (z <= outlier_z), series, medians)
    index = np.arange(cleaned.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, cleaned, 1)
    detrended = cleaned - (slope * index + intercept)
    window = np.hanning(detrended.size)
    spectrum = np.fft.rfft(detrended * window)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(detrended.size, d=cadence_s)
    total = float(power[1:].sum())
    if total <= 0.0:
        return 0.0
    above = float(power[frequencies > cutoff_hz].sum())
    return float(above / total)


def estimate_curvature(
    values: np.ndarray,
    cadence_s: float,
    span: float,
    declared_sigma: float = 0.0,
    half_window: int = DEFAULT_HALF_WINDOW,
    quantile: float = DEFAULT_CURVATURE_QUANTILE,
    outlier_z: float = CURVATURE_OUTLIER_Z,
    resolvability_threshold: float | None = DEFAULT_RESOLVABILITY_THRESHOLD,
) -> CurvatureEstimate:
    """Equation (9) with its noise floor, its bandwidth and a measurability verdict.

    Parameters
    ----------
    values
        ``(n,)`` channel values in received order.
    cadence_s
        Declared cadence ``T``, seconds.
    span
        Declared range span of the channel, for the MAD floor of equation (5).
    declared_sigma
        Declared measurement noise standard deviation. When positive it is used
        for the noise floor and the coverage term; when zero, equation (16)
        estimates it and the result says so.
    half_window
        ``m`` of equation (9).
    quantile
        ``q`` of equation (9).
    outlier_z
        Robust z above which a sample is replaced by its local median first.
    resolvability_threshold
        Share of the channel's variance that may sit above the estimator's own
        half-power frequency before the estimate is declared unresolvable.
        ``None`` disables the check, which is what
        ``validation/validate_interpolation_policy.py`` does to measure what
        the check is worth.
    """
    if cadence_s <= 0.0:
        raise ValueError(f"cadence_s must be positive, got {cadence_s!r}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile!r}")
    series = np.asarray(values, dtype=np.float64)
    coefficients = quadratic_second_derivative_coefficients(half_window)
    bandwidth = estimator_bandwidth_fraction(half_window) / cadence_s
    if declared_sigma > 0.0:
        sigma, sigma_source = float(declared_sigma), "declared"
    else:
        sigma, sigma_source = estimate_noise_sigma(series), "estimated"
    gain = 2.0 * float(np.sqrt(np.sum(coefficients**2)))
    floor = half_normal_quantile(quantile) * gain * sigma / cadence_s**2

    width = 2 * half_window + 1
    if series.size < width or not np.any(np.isfinite(series)):
        return CurvatureEstimate(
            value=0.0,
            source="estimated",
            noise_floor=float(floor),
            sigma=sigma,
            sigma_source=sigma_source,
            measurable=False,
            bandwidth_hz=float(bandwidth),
            half_window=int(half_window),
            high_frequency_fraction=0.0,
            resolvable=False,
        )

    medians, scaled = robust_local_statistics(series, half_window, span)
    with np.errstate(invalid="ignore"):
        z = np.abs(series - medians) / scaled
    cleaned = np.where(np.isfinite(series) & (z <= outlier_z), series, medians)
    view = np.lib.stride_tricks.sliding_window_view(cleaned, width)
    second = 2.0 * (view @ coefficients) / cadence_s**2
    value = float(np.quantile(np.abs(second), quantile))
    fraction = high_frequency_variance_fraction(
        series, cadence_s, bandwidth, span, half_window=half_window, outlier_z=outlier_z
    )
    resolvable = (
        True if resolvability_threshold is None else bool(fraction <= resolvability_threshold)
    )
    return CurvatureEstimate(
        value=value,
        source="estimated",
        noise_floor=float(floor),
        sigma=sigma,
        sigma_source=sigma_source,
        measurable=bool(value > MEASURABILITY_FACTOR * floor),
        bandwidth_hz=float(bandwidth),
        half_window=int(half_window),
        high_frequency_fraction=float(fraction),
        resolvable=resolvable,
    )


def decide_gap(
    gap: Gap,
    gap_index: int,
    channel_index: int,
    channel_name: str,
    unit_tolerance: float,
    curvature: CurvatureEstimate,
    slew: float,
    cadence_s: float,
    coverage_k: float = DEFAULT_COVERAGE_K,
    max_gap_slots: float = DEFAULT_MAX_GAP_SLOTS,
    endpoint_verdicts: tuple[str, str] = ("clean", "clean"),
) -> GapDecision:
    """Evaluate every gate for one gap and channel and return the verdict.

    Gates. All applicable gates must pass for a method to be permitted.

    ``nyquist_slots``
        The gap in cadence slots against ``max_gap_slots``. Applies to both
        methods.
    ``tolerance_above_noise``
        ``k sigma`` against the declared tolerance. A tolerance below the
        channel's own measurement noise cannot be met by an interpolation, and
        cannot be met by a received sample either; refusing is the only honest
        answer and saying so is more useful than a number. Applies to both.
    ``curvature_measurable``
        Whether the curvature is declared, or estimated above its own noise
        floor. Applies to ``linear``.
    ``endpoints_clean``
        Whether both frames bounding the gap carry a clean integrity verdict on
        this channel. Applies to both methods; see the module docstring.
    ``curvature_resolvable``
        Whether the channel's content sits inside the estimator's own
        bandwidth. An estimate on a channel with content above it is biased
        rather than noisy, so ``curvature_measurable`` passes while the number
        is wrong by orders of magnitude. Applies to ``linear``.
    ``curvature_bound``
        Equation (3') against the declared tolerance. Applies to ``linear``.
    ``slew_bound``
        Equation (4) plus coverage against the declared tolerance. Applies to
        ``hold``.
    """
    if coverage_k < 0.0:
        raise ValueError(f"coverage_k must be non-negative, got {coverage_k!r}")
    slots = float(gap.duration_s / cadence_s) if cadence_s > 0.0 else float("inf")
    coverage = float(coverage_k * curvature.sigma)
    bound_linear = linear_error_bound(curvature.value, gap.duration_s)
    bound_hold = hold_error_bound(slew, gap.duration_s)
    predicted_linear = bound_linear + coverage
    predicted_hold = bound_hold + coverage

    unclean = tuple(verdict for verdict in endpoint_verdicts if verdict != "clean")
    endpoints = GateOutcome(
        gate="endpoints_clean",
        passed=not unclean,
        value=float(len(unclean)),
        limit=0.0,
        statement=(
            "both frames bounding the gap carry a clean integrity verdict on this channel"
            if not unclean
            else f"a frame bounding the gap is {' and '.join(unclean)} on this channel; "
            "equations (3) and (4) bound an interpolation anchored on correct endpoint "
            "values, and a wrong anchor moves the whole chord by an unbounded amount"
        ),
    )
    nyquist = GateOutcome(
        gate="nyquist_slots",
        passed=bool(slots <= max_gap_slots),
        value=slots,
        limit=float(max_gap_slots),
        statement=(
            f"gap spans {slots:.4f} cadence slots against a declared limit of "
            f"{max_gap_slots:.4f}; inside it the local sample spacing supports only "
            f"{1.0 / (2.0 * max(slots, 1e-12) * cadence_s):.6g} Hz where the record elsewhere "
            f"resolves {1.0 / (2.0 * cadence_s):.6g} Hz"
        ),
    )
    noise_gate = GateOutcome(
        gate="tolerance_above_noise",
        passed=bool(coverage <= unit_tolerance),
        value=coverage,
        limit=float(unit_tolerance),
        statement=(
            f"the coverage term {coverage_k:g} sigma = {coverage:.6g} ({curvature.sigma_source} "
            f"sigma = {curvature.sigma:.6g}) against a declared tolerance of "
            f"{unit_tolerance:.6g}; a tolerance below the channel's own measurement noise "
            "cannot be met by an interpolation and cannot be met by a received sample either"
        ),
    )
    measurable = GateOutcome(
        gate="curvature_measurable",
        passed=bool(curvature.source == "declared" or curvature.measurable),
        value=float(curvature.value),
        limit=float(MEASURABILITY_FACTOR * curvature.noise_floor),
        statement=(
            f"curvature {curvature.value:.6g} per s^2 ({curvature.source}) against "
            f"{MEASURABILITY_FACTOR:g} times its own noise floor "
            f"{curvature.noise_floor:.6g}; an estimate below that is its own noise, and the "
            f"estimator is in any case blind above {curvature.bandwidth_hz:.6g} Hz"
        ),
    )
    resolvable = GateOutcome(
        gate="curvature_resolvable",
        passed=bool(curvature.source == "declared" or curvature.resolvable),
        value=float(curvature.high_frequency_fraction),
        limit=float(DEFAULT_RESOLVABILITY_THRESHOLD),
        statement=(
            f"{curvature.high_frequency_fraction:.6g} of this channel's variance sits above "
            f"{curvature.bandwidth_hz:.6g} Hz, the half-power frequency of the equation (9) "
            "estimator; content above it is attenuated, so the estimate is biased rather than "
            "noisy and its noise floor does not catch the error"
        ),
    )
    curvature_gate = GateOutcome(
        gate="curvature_bound",
        passed=bool(predicted_linear <= unit_tolerance),
        value=float(predicted_linear),
        limit=float(unit_tolerance),
        statement=(
            f"equation (3') predicts {bound_linear:.6g} + {coverage:.6g} = "
            f"{predicted_linear:.6g} against a declared tolerance of {unit_tolerance:.6g}"
        ),
    )
    slew_gate = GateOutcome(
        gate="slew_bound",
        passed=bool(predicted_hold <= unit_tolerance),
        value=float(predicted_hold),
        limit=float(unit_tolerance),
        statement=(
            f"equation (4) plus coverage predicts {bound_hold:.6g} + {coverage:.6g} = "
            f"{predicted_hold:.6g} against a declared tolerance of {unit_tolerance:.6g}"
        ),
    )
    gates = (
        endpoints,
        nyquist,
        noise_gate,
        measurable,
        resolvable,
        curvature_gate,
        slew_gate,
    )

    common = endpoints.passed and nyquist.passed and noise_gate.passed
    linear_ok = common and measurable.passed and resolvable.passed and curvature_gate.passed
    hold_ok = common and slew_gate.passed
    if linear_ok and (not hold_ok or predicted_linear <= predicted_hold):
        method, predicted = "linear", predicted_linear
    elif hold_ok:
        method, predicted = "hold", predicted_hold
    else:
        method, predicted = "none", float("inf")

    if method == "none":
        decision = "refuse"
        margin = float("-inf")
        failed = [gate for gate in gates if not gate.passed]
        if failed:
            reason = "refused: " + "; ".join(gate.statement for gate in failed)
        else:  # pragma: no cover - defensive; the gate sets cover every path
            reason = "refused: no method was permitted and no gate failed"
    else:
        decision = "interpolate"
        chosen = "curvature_bound" if method == "linear" else "slew_bound"
        reason = (
            f"permitted by {method}: "
            + next(gate.statement for gate in gates if gate.gate == chosen)
            + f"; {nyquist.statement}"
        )
        margin = float(unit_tolerance - predicted)

    return GapDecision(
        gap_index=int(gap_index),
        channel=int(channel_index),
        channel_name=channel_name,
        duration_s=float(gap.duration_s),
        unobserved_s=float(gap.unobserved_s),
        slots=slots,
        tolerance=float(unit_tolerance),
        curvature=curvature,
        slew=float(slew),
        coverage=coverage,
        bound_linear=float(bound_linear),
        bound_hold=float(bound_hold),
        predicted_linear=float(predicted_linear),
        predicted_hold=float(predicted_hold),
        decision=decision,
        method=method,
        predicted_error=float(predicted),
        margin=margin,
        reason=reason,
        gates=gates,
    )


def channel_curvatures(
    record: TelemetryRecord,
    half_window: int = DEFAULT_HALF_WINDOW,
    quantile: float = DEFAULT_CURVATURE_QUANTILE,
    resolvability_threshold: float | None = DEFAULT_RESOLVABILITY_THRESHOLD,
) -> dict[str, CurvatureEstimate]:
    """One :class:`CurvatureEstimate` per channel, declared where declared."""
    out: dict[str, CurvatureEstimate] = {}
    for column, channel in enumerate(record.schema.channels):
        if channel.max_curvature > 0.0:
            sigma = channel.noise_sigma
            sigma_source = "declared"
            if sigma <= 0.0:
                sigma = estimate_noise_sigma(record.values[:, column])
                sigma_source = "estimated"
            out[channel.name] = CurvatureEstimate(
                value=float(channel.max_curvature),
                source="declared",
                noise_floor=0.0,
                sigma=float(sigma),
                sigma_source=sigma_source,
                measurable=True,
                bandwidth_hz=float("inf"),
                half_window=int(half_window),
            )
        else:
            out[channel.name] = estimate_curvature(
                record.values[:, column],
                record.schema.cadence_s,
                channel.span,
                declared_sigma=channel.noise_sigma,
                half_window=half_window,
                quantile=quantile,
                resolvability_threshold=resolvability_threshold,
            )
    return out


def endpoint_verdicts(
    report: IntegrityReport, n_frames: int, n_channels: int
) -> dict[tuple[int, int], str]:
    """Non-clean integrity verdict per ``(frame, channel)``, for the endpoint gate.

    A value-level anomaly marks its own channel. A frame-level anomaly other
    than a gap -- a repeated counter, a reorder or a clock reset -- marks every
    channel of that frame, because it is a property of the frame rather than of
    one word. A gap marks nothing: it is the thing being interpolated across.
    """
    out: dict[tuple[int, int], str] = {}
    for anomaly in report.anomalies:
        if anomaly.observation == "gap":
            continue
        if not 0 <= anomaly.index < n_frames:  # pragma: no cover - defensive
            continue
        if anomaly.channel >= 0:
            out[(anomaly.index, anomaly.channel)] = anomaly.verdict.value
        else:
            for column in range(n_channels):
                out.setdefault((anomaly.index, column), anomaly.verdict.value)
    return out


def plan(
    record: TelemetryRecord,
    report: IntegrityReport,
    tolerances: dict[str, float],
    max_gap_slots: float = DEFAULT_MAX_GAP_SLOTS,
    curvature_quantile: float = DEFAULT_CURVATURE_QUANTILE,
    coverage_k: float = DEFAULT_COVERAGE_K,
    half_window: int = DEFAULT_HALF_WINDOW,
    resolvability_threshold: float | None = DEFAULT_RESOLVABILITY_THRESHOLD,
) -> PolicyPlan:
    """Build the gap-handling policy for every gap in ``report`` and every channel.

    Parameters
    ----------
    record
        The record the gaps were found in.
    report
        An :class:`~telemgap.classify.IntegrityReport` over the same record.
    tolerances
        Declared reconstruction tolerance per channel name, in that channel's
        unit. Every channel must appear; a missing channel is an error rather
        than a silent default, because a silent default is the failure this
        product exists to remove.
    max_gap_slots
        The Nyquist gate, in cadence slots.
    curvature_quantile, half_window
        Parameters of equation (9).
    coverage_k
        ``k`` of equation (3').
    resolvability_threshold
        Passed to :func:`estimate_curvature`. ``None`` disables the
        resolvability gate.

    Returns
    -------
    PolicyPlan
    """
    missing = [name for name in record.schema.channel_names if name not in tolerances]
    if missing:
        raise ValueError(
            "a reconstruction tolerance must be declared for every channel; missing "
            f"{missing}. A silent default is the failure this policy exists to remove."
        )
    for name, value in tolerances.items():
        ChannelTolerance(channel_name=name, tolerance=float(value))
    if max_gap_slots <= 0.0:
        raise ValueError(f"max_gap_slots must be positive, got {max_gap_slots!r}")

    curvatures = channel_curvatures(
        record,
        half_window=half_window,
        quantile=curvature_quantile,
        resolvability_threshold=resolvability_threshold,
    )
    unclean = endpoint_verdicts(report, len(record), record.n_channels)
    decisions: list[GapDecision] = []
    for gap_index, gap in enumerate(report.findings.gaps):
        for column, channel in enumerate(record.schema.channels):
            decisions.append(
                decide_gap(
                    gap=gap,
                    gap_index=gap_index,
                    channel_index=column,
                    channel_name=channel.name,
                    unit_tolerance=float(tolerances[channel.name]),
                    curvature=curvatures[channel.name],
                    slew=float(channel.max_slew),
                    cadence_s=float(record.schema.cadence_s),
                    coverage_k=float(coverage_k),
                    max_gap_slots=float(max_gap_slots),
                    endpoint_verdicts=(
                        unclean.get((gap.after_index, column), "clean"),
                        unclean.get((gap.before_index, column), "clean"),
                    ),
                )
            )
    return PolicyPlan(
        decisions=tuple(decisions),
        max_gap_slots=float(max_gap_slots),
        curvature_quantile=float(curvature_quantile),
        coverage_k=float(coverage_k),
        half_window=int(half_window),
        curvatures=curvatures,
        tolerances={name: float(value) for name, value in tolerances.items()},
    )
