"""Streaming change detectors, each with a stated threshold-setting procedure.

Every detector implements the same interface so the benchmark can score them on
one footing:

* ``update(x) -> bool`` consumes one sample and returns ``True`` on an alarm.
* ``reset()`` returns the detector to its initial state.
* ``threshold`` is the single scalar the benchmark calibrates. Every other
  parameter is declared and fixed.
* ``default_threshold()`` returns the threshold the detector's own literature
  or implementation convention suggests, *without* reference to any target
  false-alarm rate. Comparing detectors at their default thresholds is the
  standard error this package exists to make visible, so the defaults are
  shipped in order to be measured, not in order to be used.

Harness convention, stated because it is not the only possible one
------------------------------------------------------------------
On an alarm the benchmark calls ``reset()``, so a long stationary stream yields
many independent run lengths. ADWIN as published shrinks its window instead of
discarding it; EWMA and CUSUM charts in practice are often restarted at a
headstart value rather than at zero. Full reset is used here for all five
analytic detectors because it makes the run lengths comparable, and it is the
convention under which every ARL figure in this repository was measured.

References, each verified in this session against a publisher-hosted or
author-hosted source (see validation/VALIDATION.md section 9 for how)
---------------------------------------------------------------------
* Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1-2),
  100-115. DOI 10.1093/biomet/41.1-2.100. The CUSUM recursion and the
  reference-value/decision-interval parameterisation.
* Hinkley, D. V. (1971). "Inference about the change-point from cumulative sum
  tests." *Biometrika* 58(3), 509-523. DOI 10.1093/biomet/58.3.509.
* Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving
  Averages." *Technometrics* 1(3). DOI 10.1080/00401706.1959.10489860. The page
  range could not be verified from a publisher-hosted page in this container
  and is therefore not quoted.
* Bifet, A. and Gavalda, R. (2007). "Learning from Time-Changing Data with
  Adaptive Windowing." *Proceedings of the 7th SIAM International Conference on
  Data Mining*, 443-448. DOI 10.1137/1.9781611972771.42. The cut rule
  implemented here is transcribed from the authors' own technical report
  "Adaptive Parameter-free Learning from Evolving Data Streams" (Universitat
  Politecnica de Catalunya, upcommons.upc.edu), section 4.1.1.
* Kolmogorov-Smirnov two-sample statistic: the implementation here is checked
  against ``scipy.stats.ks_2samp`` to machine precision in
  ``validation/validate_detector_reference.py``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "ADWIN",
    "alarm_ratio_trace",
    "first_alarm_at_or_after",
    "ANALYTIC_DETECTORS",
    "CUSUM",
    "EWMA",
    "Detector",
    "PageHinkley",
    "WindowedKS",
    "ks_two_sample_statistic",
    "make_detector",
]


class Detector(ABC):
    """Base class for a streaming univariate change detector."""

    #: Human-readable name used in every table and figure.
    name: str = "detector"
    #: Name of the single calibrated scalar, for error messages and reports.
    threshold_name: str = "threshold"
    #: When ``True`` the detector manages its own post-alarm state and the
    #: benchmark must **not** call :meth:`reset` after an alarm. Only ADWIN in
    #: ``shrink_on_detect`` mode sets this, because published ADWIN drops the
    #: older part of its window rather than discarding the window. The contract
    #: exists so that both conventions can be measured side by side instead of
    #: one being asserted to be the fair one.
    self_resetting: bool = False

    @abstractmethod
    def update(self, x: float) -> bool:
        """Consume one sample (dimensionless, pre-change sigma units).

        Returns
        -------
        bool
            ``True`` if this sample raises an alarm.
        """

    @abstractmethod
    def reset(self) -> None:
        """Return the detector to its initial state."""

    @property
    @abstractmethod
    def threshold(self) -> float:
        """The calibrated scalar."""

    @threshold.setter
    @abstractmethod
    def threshold(self, value: float) -> None: ...

    @staticmethod
    @abstractmethod
    def default_threshold() -> float:
        """The literature or convention default, for the 'wrong comparison' demo."""

    def is_armed(self) -> bool:
        """``True`` when an alarm is possible on the next sample.

        A detector with no warm-up is always armed. A windowed detector is
        un-armed until its windows are filled, and becomes un-armed again after
        a reset. :func:`telemdrift.scoring.blind_fraction` uses this to measure
        how much of a stream a detector spends unable to detect anything.
        """
        return True

    @abstractmethod
    def alarm_ratio(self) -> float:
        """Current evidence divided by its own alarm level, dimensionless.

        An alarm is raised exactly when this exceeds 1.0. The five analytic
        detectors compare five incomparable scalars against five incomparable
        thresholds; dividing by the threshold is what lets all of them share one
        axis in a figure. It returns 0.0 whenever the detector is still warming
        up and no alarm is possible.
        """

    def describe(self) -> str:
        """One-line description including the current threshold value."""
        return f"{self.name}({self.threshold_name}={self.threshold:.6g})"


class CUSUM(Detector):
    """Two-sided tabular CUSUM on standardised observations (Page 1954).

    With ``z_t = (x_t - mu0) / sigma0``:

    .. code-block:: text

        S_plus_t  = max(0, S_plus_{t-1}  + z_t - k)
        S_minus_t = max(0, S_minus_{t-1} - z_t - k)
        alarm     = max(S_plus_t, S_minus_t) > h

    ``k`` is the reference value in standard deviations and is **fixed**, not
    calibrated: ``k = delta/2`` is the standard choice for fastest detection of
    a shift of size ``delta``, so ``k = 0.5`` targets a one-sigma shift.
    ``h`` is the decision interval and is the calibrated scalar.

    Threshold-setting procedure
    ---------------------------
    ``h`` is found by bisection on a declared target ARL0 over seeded stationary
    streams (:func:`telemdrift.thresholds.calibrate_threshold`). No closed-form
    ARL0 is used: the Siegmund and Markov-chain approximations in the SPC
    literature are good but they are approximations, and the point of this
    package is that the operating point must be measured.

    Assumptions
    -----------
    ``mu0`` and ``sigma0`` are *known* and supplied. This is the assumption that
    separates CUSUM from Page-Hinkley in this package; see
    :class:`PageHinkley`.
    """

    name = "CUSUM"
    threshold_name = "h"

    def __init__(self, h: float = 4.0, k: float = 0.5, mu0: float = 0.0, sigma0: float = 1.0):
        if h <= 0.0:
            raise ValueError("CUSUM decision interval h must be > 0")
        if k < 0.0:
            raise ValueError("CUSUM reference value k must be >= 0")
        if sigma0 <= 0.0:
            raise ValueError("CUSUM sigma0 must be > 0")
        self._h = float(h)
        self.k = float(k)
        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.s_plus = 0.0
        self.s_minus = 0.0

    @property
    def threshold(self) -> float:
        return self._h

    @threshold.setter
    def threshold(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("CUSUM decision interval h must be > 0")
        self._h = float(value)

    @staticmethod
    def default_threshold() -> float:
        """``h = 4`` with ``k = 0.5``: the textbook SPC pair, chosen for ARL0 ~ 168."""
        return 4.0

    def reset(self) -> None:
        self.s_plus = 0.0
        self.s_minus = 0.0

    def alarm_ratio(self) -> float:
        return max(self.s_plus, self.s_minus) / self._h

    def update(self, x: float) -> bool:
        z = (float(x) - self.mu0) / self.sigma0
        self.s_plus = max(0.0, self.s_plus + z - self.k)
        self.s_minus = max(0.0, self.s_minus - z - self.k)
        return self.s_plus > self._h or self.s_minus > self._h


class PageHinkley(Detector):
    """Two-sided Page-Hinkley test with a running mean (Page 1954, Hinkley 1971).

    .. code-block:: text

        xbar_t = running sample mean of x_1..x_t
        m_t    = sum_{i<=t} (x_i - xbar_i - delta)
        PH_up  = m_t - min_{i<=t} m_i
        PH_dn  = max_{i<=t} m_i - m_t
        alarm  = max(PH_up, PH_dn) > lambda_

    ``delta`` is the magnitude allowance (fixed at 0.005, the value used by the
    common streaming implementations) and ``lambda_`` is the calibrated scalar.

    How this differs from CUSUM, and why it matters
    -----------------------------------------------
    CUSUM standardises against a *declared* ``mu0`` and ``sigma0``.
    Page-Hinkley subtracts the *running sample mean*, so it needs no declared
    nominal value -- and in exchange its statistic is not scale-free and its
    ARL0 depends on how fast the running mean adapts. The benchmark reports both
    because the practical choice between them is exactly this trade.
    """

    name = "Page-Hinkley"
    threshold_name = "lambda"

    def __init__(self, lambda_: float = 50.0, delta: float = 0.005):
        if lambda_ <= 0.0:
            raise ValueError("Page-Hinkley lambda must be > 0")
        if delta < 0.0:
            raise ValueError("Page-Hinkley delta must be >= 0")
        self._lambda = float(lambda_)
        self.delta = float(delta)
        self.n = 0
        self.mean = 0.0
        self.m = 0.0
        self.m_min = 0.0
        self.m_max = 0.0

    @property
    def threshold(self) -> float:
        return self._lambda

    @threshold.setter
    def threshold(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Page-Hinkley lambda must be > 0")
        self._lambda = float(value)

    @staticmethod
    def default_threshold() -> float:
        """``lambda = 50``, the default shipped by the common streaming implementations."""
        return 50.0

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m = 0.0
        self.m_min = 0.0
        self.m_max = 0.0

    def alarm_ratio(self) -> float:
        return max(self.m - self.m_min, self.m_max - self.m) / self._lambda

    def update(self, x: float) -> bool:
        xv = float(x)
        self.n += 1
        self.mean += (xv - self.mean) / self.n
        self.m += xv - self.mean - self.delta
        if self.m < self.m_min:
            self.m_min = self.m
        if self.m > self.m_max:
            self.m_max = self.m
        return (self.m - self.m_min) > self._lambda or (self.m_max - self.m) > self._lambda


class EWMA(Detector):
    """Two-sided EWMA control chart with exact time-varying limits (Roberts 1959).

    .. code-block:: text

        z_t     = (1 - r) z_{t-1} + r (x_t - mu0) / sigma0,   z_0 = 0
        var z_t = r / (2 - r) * (1 - (1 - r)^(2 t))
        alarm   = abs(z_t) > L * sqrt(var z_t)

    ``r`` is the smoothing constant, fixed at 0.1. ``L`` is the calibrated
    scalar. The *exact* finite-``t`` variance is used rather than the asymptotic
    ``r/(2-r)``, because the asymptotic limit is too wide in the first few dozen
    samples and that is precisely where a restarted chart spends its time in an
    ARL0 measurement -- using the asymptotic form inflates the measured ARL0 for
    reasons that have nothing to do with the detector.
    """

    name = "EWMA"
    threshold_name = "L"

    def __init__(self, L: float = 3.0, r: float = 0.1, mu0: float = 0.0, sigma0: float = 1.0):
        if L <= 0.0:
            raise ValueError("EWMA control limit L must be > 0")
        if not 0.0 < r <= 1.0:
            raise ValueError("EWMA smoothing constant r must satisfy 0 < r <= 1")
        if sigma0 <= 0.0:
            raise ValueError("EWMA sigma0 must be > 0")
        self._L = float(L)
        self.r = float(r)
        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.z = 0.0
        self.t = 0
        self._asym = self.r / (2.0 - self.r)

    @property
    def threshold(self) -> float:
        return self._L

    @threshold.setter
    def threshold(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("EWMA control limit L must be > 0")
        self._L = float(value)

    @staticmethod
    def default_threshold() -> float:
        """``L = 3``: three-sigma limits, the universal control-chart default."""
        return 3.0

    def reset(self) -> None:
        self.z = 0.0
        self.t = 0

    def _limit(self) -> float:
        if self.t == 0:
            return 0.0
        var = self._asym * (1.0 - (1.0 - self.r) ** (2 * self.t))
        return self._L * math.sqrt(var)

    def alarm_ratio(self) -> float:
        limit = self._limit()
        return abs(self.z) / limit if limit > 0.0 else 0.0

    def update(self, x: float) -> bool:
        self.t += 1
        z_in = (float(x) - self.mu0) / self.sigma0
        self.z = (1.0 - self.r) * self.z + self.r * z_in
        return abs(self.z) > self._limit()


def ks_two_sample_statistic(reference_sorted: np.ndarray, window: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup_x abs(F_ref(x) - F_win(x))``.

    Parameters
    ----------
    reference_sorted:
        Reference sample, **already sorted ascending**.
    window:
        Detection sample, any order.

    Returns
    -------
    float
        The two-sided KS statistic, in [0, 1].

    Notes
    -----
    Implemented directly rather than through ``scipy.stats.ks_2samp`` because
    the streaming benchmark evaluates it of order 10^5 times and the SciPy call
    measured 987 us against 26 us here (``validation/outputs/
    validate_detector_reference.txt``). The two agree to machine precision on
    the statistic; that equality is a committed check, not an assumption.
    ``scipy.stats.kstest(x, "norm", args=(loc, scale))`` is **not** used
    anywhere in this package: it raises ``TypeError`` on the installed SciPy.
    """
    ref = np.asarray(reference_sorted, dtype=float)
    win = np.sort(np.asarray(window, dtype=float))
    if ref.size == 0 or win.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.concatenate([ref, win])
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_win = np.searchsorted(win, grid, side="right") / win.size
    return float(np.max(np.abs(cdf_ref - cdf_win)))


class WindowedKS(Detector):
    """Sliding-window two-sample KS test against a fixed reference window.

    The first ``n_ref`` samples after a reset form the reference window. After
    that, the most recent ``n_det`` samples form the detection window and the
    two-sample KS statistic is evaluated every ``stride`` samples:

    .. code-block:: text

        alarm = D_ks(reference, last n_det samples) > c

    ``c`` is the calibrated scalar. ``stride`` exists for cost: evaluating every
    sample costs ``n_ref + n_det`` work per sample and buys nothing, because the
    statistic moves by at most ``1/n_det`` per sample. ``stride`` is declared
    (default 5) and it caps the achievable delay resolution at ``stride``
    samples, which the benchmark reports rather than hides.

    Threshold-setting procedure, and the default that exists to be wrong
    -------------------------------------------------------------------
    ``default_threshold`` returns the asymptotic two-sided critical value at
    ``alpha = 0.005``,

    .. code-block:: text

        c_alpha = sqrt(-0.5 ln(alpha / 2)) * sqrt((n_ref + n_det) / (n_ref n_det))

    which is what a practitioner reaches for. It controls the error rate of
    *one* test. The detector performs a test every ``stride`` samples forever,
    so the per-test level says nothing about the stream-level false-alarm rate.
    The measured gap is in README.md.
    """

    name = "Windowed KS"
    threshold_name = "c"

    N_REF = 200
    N_DET = 100
    ALPHA_DEFAULT = 0.005

    def __init__(
        self,
        c: float = 0.2,
        n_ref: int = N_REF,
        n_det: int = N_DET,
        stride: int = 5,
    ):
        if not 0.0 < c <= 1.0:
            raise ValueError("KS critical value c must satisfy 0 < c <= 1")
        if n_ref < 2 or n_det < 2:
            raise ValueError("n_ref and n_det must both be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._c = float(c)
        self.n_ref = int(n_ref)
        self.n_det = int(n_det)
        self.stride = int(stride)
        self._ref: np.ndarray | None = None
        self._ref_buf: list[float] = []
        self._win: np.ndarray = np.empty(self.n_det)
        self._filled = 0
        self._pos = 0
        self._since = 0
        self._last_stat = 0.0

    @property
    def threshold(self) -> float:
        return self._c

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("KS critical value c must satisfy 0 < c <= 1")
        self._c = float(value)

    @staticmethod
    def default_threshold() -> float:
        """Asymptotic KS critical value at alpha = 0.005 for n_ref=200, n_det=100."""
        n, m = WindowedKS.N_REF, WindowedKS.N_DET
        return math.sqrt(-0.5 * math.log(WindowedKS.ALPHA_DEFAULT / 2.0)) * math.sqrt(
            (n + m) / (n * m)
        )

    def is_armed(self) -> bool:
        return self._ref is not None and self._filled >= self.n_det

    def alarm_ratio(self) -> float:
        return self._last_stat / self._c

    @property
    def last_statistic(self) -> float:
        """Most recently evaluated KS statistic, in [0, 1]. 0.0 during warm-up."""
        return self._last_stat

    def reset(self) -> None:
        self._ref = None
        self._ref_buf = []
        self._filled = 0
        self._pos = 0
        self._since = 0
        self._last_stat = 0.0

    def update(self, x: float) -> bool:
        xv = float(x)
        if self._ref is None:
            self._ref_buf.append(xv)
            if len(self._ref_buf) >= self.n_ref:
                self._ref = np.sort(np.asarray(self._ref_buf, dtype=float))
                self._ref_buf = []
            return False
        self._win[self._pos] = xv
        self._pos = (self._pos + 1) % self.n_det
        if self._filled < self.n_det:
            self._filled += 1
            return False
        self._since += 1
        if self._since < self.stride:
            return False
        self._since = 0
        self._last_stat = ks_two_sample_statistic(self._ref, self._win)
        return self._last_stat > self._c

    @property
    def warmup(self) -> int:
        """Samples consumed after a reset before an alarm is possible."""
        return self.n_ref + self.n_det


class ADWIN(Detector):
    """ADWIN-style adaptive window, cut rule implemented as published.

    Exponential histogram of ``M`` buckets per row (the ADWIN2 data structure).
    After every ``check_every`` samples, every bucket boundary is tried as a cut
    point ``W = W0 . W1`` (``W0`` the older part) and the rule applied is, with
    ``n0 = |W0|``, ``n1 = |W1|``, ``n = |W|``:

    .. code-block:: text

        m        = 2 / (1/n0 + 1/n1)                  # harmonic mean
        eps_cut  = sqrt( (1 / (2 m)) * ln(4 n / delta) )
        cut      = abs(mean(W0) - mean(W1)) > eps_cut

    transcribed from Bifet and Gavalda's own technical report "Adaptive
    Parameter-free Learning from Evolving Data Streams" section 4.1.1
    (upcommons.upc.edu), which is the author-hosted source this session could
    read. **The SDM 2007 paper is widely quoted with** ``m = 1/(1/n0 + 1/n1)``
    **, a factor of two smaller**, giving an ``eps_cut`` larger by ``sqrt(2)``.
    This package implements the harmonic-mean form above and calibrates
    ``delta`` to a measured ARL0, which absorbs the factor exactly: a constant
    multiplier on ``eps_cut`` is equivalent to a rescaling of ``ln(4n/delta)``.
    The discrepancy is therefore recorded for honesty, not papered over --
    nothing in this repository's numbers depends on which of the two is meant,
    because none of them use the nominal ``delta`` as a false-alarm rate.

    ``min_sub`` is the smallest admissible subwindow and ships at **30**, not at
    the 5 the first version of this package used. The reason is measured, not
    stylistic. ``eps_cut`` grows only as ``sqrt(log(1/delta))``, so ``delta`` is a
    very weak control on the false-alarm rate: at ``min_sub = 5`` ten orders of
    magnitude of ``delta`` (1e-2 to 1e-12) move the measured ARL0 only from 45 to
    600 samples, and reaching a 500-sample ARL0 needs ``delta`` near 1e-11 --
    which the calibration's bracket search does not reach from the shipped
    default of 0.002, so it reports a bracketing failure. At ``min_sub = 30`` the
    same target is reached near ``delta = 1e-7``. The measured table is in
    ``validation/outputs/validate_robustness.txt`` section 2A. ``min_sub = 30``
    is declared before any ARL1 is measured, so it is a configuration choice and
    not a result that was tuned.

    ``delta`` is the calibrated scalar and is **not** a false-alarm probability
    in this harness: the published bound is per cut test, there are
    ``O(log n)`` cut tests per check and a check every ``check_every`` samples,
    so the nominal ``delta`` and the measured ARL0 are different quantities.
    README.md gives the measured gap.

    Harness convention
    ------------------
    Published ADWIN drops ``W0`` and continues with ``W1``. The benchmark resets
    the whole window, for the reason given in this module's docstring. The ARL1
    figures are unaffected (they end at the first alarm); the ARL0 figures are
    measured under full reset and are labelled as such.
    """

    name = "ADWIN"
    threshold_name = "delta"

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_sub: int = 30,
        check_every: int = 10,
        shrink_on_detect: bool = False,
    ):
        if not 0.0 < delta < 1.0:
            raise ValueError("ADWIN delta must satisfy 0 < delta < 1")
        if max_buckets < 2:
            raise ValueError("ADWIN max_buckets must be >= 2")
        if min_sub < 1:
            raise ValueError("ADWIN min_sub must be >= 1")
        if check_every < 1:
            raise ValueError("ADWIN check_every must be >= 1")
        self._delta = float(delta)
        self.max_buckets = int(max_buckets)
        self.min_sub = int(min_sub)
        self.check_every = int(check_every)
        self.shrink_on_detect = bool(shrink_on_detect)
        self.self_resetting = bool(shrink_on_detect)
        self._rows: list[list[list[float]]] = []
        self.n = 0
        self.total = 0.0
        self._since = 0
        self._last_evidence = 0.0
        self._best_cut_index = -1

    @property
    def threshold(self) -> float:
        return self._delta

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("ADWIN delta must satisfy 0 < delta < 1")
        self._delta = float(value)

    @staticmethod
    def default_threshold() -> float:
        """``delta = 0.002``, the default shipped by the common ADWIN implementations."""
        return 0.002

    def reset(self) -> None:
        self._rows = []
        self.n = 0
        self.total = 0.0
        self._since = 0
        self._last_evidence = 0.0
        self._best_cut_index = -1

    def _compress(self) -> None:
        i = 0
        while i < len(self._rows):
            if len(self._rows[i]) > self.max_buckets:
                b1 = self._rows[i].pop(0)
                b2 = self._rows[i].pop(0)
                if i + 1 == len(self._rows):
                    self._rows.append([])
                self._rows[i + 1].append([b1[0] + b2[0], b1[1] + b2[1]])
            i += 1

    def buckets_oldest_first(self) -> list[list[float]]:
        """Flattened ``[count, sum]`` buckets, oldest first. Exposed for tests."""
        out: list[list[float]] = []
        for row in reversed(self._rows):
            out.extend(row)
        return out

    def cut_evidence(self) -> float:
        """Largest ``abs(mean(W0) - mean(W1)) / eps_cut`` over all cut points.

        Dimensionless. A cut is declared when this exceeds 1.0, which is exactly
        the published rule rewritten so it can be plotted beside the other four
        detectors. Returns 0.0 when no admissible cut point exists yet.
        """
        buckets = self.buckets_oldest_first()
        n0 = 0.0
        s0 = 0.0
        best = 0.0
        best_idx = -1
        for idx in range(len(buckets) - 1):
            n0 += buckets[idx][0]
            s0 += buckets[idx][1]
            n1 = self.n - n0
            s1 = self.total - s0
            if n0 < self.min_sub or n1 < self.min_sub:
                continue
            m = 2.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * self.n / self._delta))
            ratio = abs(s0 / n0 - s1 / n1) / eps_cut
            if ratio > best:
                best = ratio
                best_idx = idx
        self._last_evidence = best
        self._best_cut_index = best_idx
        return best

    def cut_detected(self) -> bool:
        """Apply the published cut rule at every bucket boundary. Exposed for tests."""
        return self.cut_evidence() > 1.0

    def is_armed(self) -> bool:
        return self.n >= 2 * self.min_sub

    def alarm_ratio(self) -> float:
        return self._last_evidence

    def _drop_oldest(self, n_buckets: int) -> None:
        """Discard the ``n_buckets`` oldest buckets, keeping the recent window.

        This is published ADWIN's response to a detected cut: ``W0`` goes, ``W1``
        stays, so the detector keeps the evidence it has just gathered about the
        new regime instead of starting blind.
        """
        remaining = n_buckets
        for ri in range(len(self._rows) - 1, -1, -1):
            row = self._rows[ri]
            while row and remaining > 0:
                b = row.pop(0)
                self.n -= int(b[0])
                self.total -= b[1]
                remaining -= 1
            if remaining == 0:
                break
        while self._rows and not self._rows[-1] and len(self._rows) > 1:
            self._rows.pop()

    def update(self, x: float) -> bool:
        if not self._rows:
            self._rows.append([])
        xv = float(x)
        self._rows[0].append([1.0, xv])
        self.n += 1
        self.total += xv
        self._since += 1
        self._compress()
        if self._since < self.check_every:
            return False
        self._since = 0
        fired = self.cut_detected()
        if fired and self.shrink_on_detect and self._best_cut_index >= 0:
            self._drop_oldest(self._best_cut_index + 1)
        return fired


#: The five analytic detectors, in the order every table in this repository uses.
ANALYTIC_DETECTORS: tuple[str, ...] = ("cusum", "page_hinkley", "ewma", "ks", "adwin")

_FACTORIES = {
    "cusum": CUSUM,
    "page_hinkley": PageHinkley,
    "ewma": EWMA,
    "ks": WindowedKS,
    "adwin": ADWIN,
}


def make_detector(key: str, threshold: float | None = None, **kwargs: float) -> Detector:
    """Construct an analytic detector by key.

    Parameters
    ----------
    key:
        One of :data:`ANALYTIC_DETECTORS`.
    threshold:
        The calibrated scalar. ``None`` uses the detector's declared default,
        which is the comparison this package argues against making.
    """
    if key not in _FACTORIES:
        raise ValueError(f"unknown detector {key!r}; expected one of {ANALYTIC_DETECTORS}")
    cls = _FACTORIES[key]
    det = cls(**kwargs)  # type: ignore[arg-type]
    det.threshold = cls.default_threshold() if threshold is None else threshold
    return det


def alarm_ratio_trace(
    detector: Detector, stream: np.ndarray, reset_on_alarm: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``detector`` over ``stream`` and return its alarm ratios and alarms.

    Parameters
    ----------
    detector:
        Any detector. It is reset before the run.
    stream:
        1-D array of standardised samples.
    reset_on_alarm:
        ``True`` (the default) runs the detector as the benchmark does and as it
        would run in service: an alarm resets it and the stream continues, so a
        trace over a long episode shows every alarm, not only the first. Set
        ``False`` to watch a single statistic grow past its threshold without
        interruption, which is what a textbook figure shows and is not what a
        detector does. A detector declaring ``self_resetting`` is never reset
        externally regardless of this flag.

    Returns
    -------
    ratios:
        ``detector.alarm_ratio()`` after each sample, same length as ``stream``.
    alarms:
        Integer indices of every alarm, in order. Empty if the detector never
        alarmed.
    """
    detector.reset()
    external = reset_on_alarm and not getattr(detector, "self_resetting", False)
    out = np.empty(len(stream))
    alarms: list[int] = []
    for i, value in enumerate(stream):
        fired = detector.update(value)
        out[i] = detector.alarm_ratio()
        if fired:
            alarms.append(i)
            if external:
                detector.reset()
    return out, np.asarray(alarms, dtype=int)


def first_alarm_at_or_after(alarms: np.ndarray, index: int) -> int:
    """First alarm index at or after ``index``, or ``-1`` if there is none.

    The benchmark's detection event: an alarm before the change is a false
    alarm, not a detection, and must not be reported as one.
    """
    arr = np.asarray(alarms, dtype=int)
    later = arr[arr >= index]
    return int(later[0]) if later.size else -1
