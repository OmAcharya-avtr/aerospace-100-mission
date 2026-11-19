"""The acquisition-time distribution: a lognormal body with a retry tail.

WHAT THIS IS A MODEL OF
-----------------------

The time from the start of a pointing-acquisition-tracking (PAT) sequence at the
incoming ground station to a stable fine-tracking lock, in seconds. Two
structural facts about that quantity drive the form chosen here.

1. **The body is right-skewed and positive.** Acquisition is a search over a
   solid angle whose duration is a product of effects — uncertainty-cone size,
   beacon signal-to-noise ratio, scan rate, atmospheric scintillation deepening
   fades during the search — rather than a sum of them. A product of positive
   factors is lognormal in the limit, which is the standard argument for using a
   lognormal for a positive, product-generated duration (Aitchison, J. and
   Brown, J. A. C., *The Lognormal Distribution*, Cambridge University Press,
   1957, chapter 1). The physical setting of the search, and the role of
   uncertainty cone, beacon power and scintillation in it, is described in
   Hemmati, H. (editor), *Deep Space Optical Communications*, JPL Deep-Space
   Communications and Navigation Series, Wiley, 2006, in the chapter on
   acquisition, tracking and pointing, and in Kaushal, H. and Kaddoum, G.,
   "Optical Communication in Space: Challenges and Mitigation Techniques",
   *IEEE Communications Surveys and Tutorials*, volume 19, number 1, pages
   57-96, 2017.

2. **The tail is a retry tail, not a lognormal tail.** A PAT sequence that does
   not lock within its allotted attempt window aborts and restarts. The
   resulting duration is the sum of the failed attempts' windows and a
   successful attempt, so the upper tail is a geometric mixture of shifted
   bodies and is heavier than the body alone. Deep fades that cause such an
   abort are governed by the scintillation statistics of the turbulent
   atmosphere; Andrews, L. C. and Phillips, R. L., *Laser Beam Propagation
   through Random Media*, second edition, SPIE Press, 2005, is the reference for
   the scintillation index and the fade statistics that make a long fade
   possible, and Kim, I. I., McArthur, B. and Korevaar, E., "Comparison of laser
   beam propagation at 785 nm and 1550 nm in fog and haze for optical wireless
   communications", *Proceedings of SPIE*, volume 4214, pages 26-37, 2001, is
   the reference for the attenuation regimes that set a link's fade margin.

WHAT IS NOT CLAIMED
-------------------

**No numeric value in this module is a measurement reported by any of those
sources.** The sources establish the form of the distribution and the physical
reason for the retry tail. The parameters are declared inputs, supplied by the
caller, and the values used in this repository's corpus are declared in
``data/generate_corpus.py``. A reader wanting measured acquisition times for a
specific terminal must take them from that terminal's own test campaign.

MODEL
-----

With median ``m`` seconds, log-scale shape ``sigma``, per-attempt abort
probability ``p`` and attempt window ``w`` seconds, the acquisition time is

.. math::

    T_{acq} = k w + L, \\quad k \\sim \\mathrm{Geometric_0}(1-p),
              \\quad L \\sim \\mathrm{Lognormal}(\\ln m, \\sigma)

truncated at ``w`` from above for the successful attempt's own duration, so that
a success cannot take longer than the window that defines a success. ``k`` is
the number of FAILED attempts.

Mean and variance follow from the geometric-plus-truncated-lognormal structure.
The truncated body's first two moments have an exact closed form in the standard
normal distribution function,

.. math::

    E[L \\mid L \\le w] = e^{\\mu + \\sigma^2/2}\\,
        \\frac{\\Phi\\!\\left(\\frac{\\ln w - \\mu}{\\sigma} - \\sigma\\right)}
              {\\Phi\\!\\left(\\frac{\\ln w - \\mu}{\\sigma}\\right)}, \\quad
    E[L^2 \\mid L \\le w] = e^{2\\mu + 2\\sigma^2}\\,
        \\frac{\\Phi\\!\\left(\\frac{\\ln w - \\mu}{\\sigma} - 2\\sigma\\right)}
              {\\Phi\\!\\left(\\frac{\\ln w - \\mu}{\\sigma}\\right)}

(the standard moment-generating identity for the lognormal restricted to a
left tail; see Aitchison and Brown 1957, chapter 2). :meth:`AcquisitionModel.moments`
uses that form. :meth:`AcquisitionModel.moments_quadrature` evaluates the same
two integrals by Gauss-Legendre quadrature of the truncated density, which
touches the probability density function and not the distribution function, and
``tests/test_acquisition.py`` requires the two paths to agree. The closed form
is the one used everywhere, because the quadrature's node computation is an
eigenvalue problem and is far too slow to call per event.

UNITS AND VALIDITY RANGE
------------------------

All times seconds. Valid for ``m > 0``, ``sigma > 0``, ``0 <= p < 1``,
``w > 0``. The model is not valid where ``m`` is comparable to or larger than
``w``: then almost every attempt aborts, the geometric count diverges in
practice, and a terminal in that regime does not acquire at all. :func:`validate`
rejects ``m >= w``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats


@lru_cache(maxsize=8)
def _legendre_nodes(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes and weights. The node computation is an
    eigenvalue problem whose cost grows as the cube of ``points``, so it is
    computed once per distinct size rather than once per call."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return nodes, weights


@dataclass(frozen=True)
class AcquisitionModel:
    """Declared acquisition-time distribution. Seconds throughout.

    Parameters
    ----------
    median_s:
        Median of the per-attempt lognormal body, seconds, strictly positive.
    log_sigma:
        Shape of the lognormal body on the natural-log scale, strictly positive.
    abort_probability:
        Probability that one attempt aborts without lock, in ``[0, 1)``.
    attempt_window_s:
        Duration of one attempt before it aborts, seconds, strictly greater than
        ``median_s``.
    """

    median_s: float
    log_sigma: float
    abort_probability: float
    attempt_window_s: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any declared bound is violated."""
        if not np.isfinite(
            [
                self.median_s,
                self.log_sigma,
                self.abort_probability,
                self.attempt_window_s,
            ]
        ).all():
            raise ValueError("every AcquisitionModel field must be finite")
        if self.median_s <= 0.0:
            raise ValueError(
                f"median_s must be strictly positive, got {self.median_s!r}"
            )
        if self.log_sigma <= 0.0:
            raise ValueError(
                f"log_sigma must be strictly positive, got {self.log_sigma!r}"
            )
        if not 0.0 <= self.abort_probability < 1.0:
            raise ValueError(
                "abort_probability must lie in [0, 1); at 1 no attempt ever locks. "
                f"Got {self.abort_probability!r}"
            )
        if self.attempt_window_s <= self.median_s:
            raise ValueError(
                f"attempt_window_s ({self.attempt_window_s!r}) must exceed median_s "
                f"({self.median_s!r}). At or below the median almost every attempt "
                "aborts and the model is outside its declared validity range."
            )

    @property
    def body(self) -> stats.rv_continuous:
        """The untruncated per-attempt lognormal body, as a frozen scipy object."""
        return stats.lognorm(s=self.log_sigma, scale=self.median_s)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` acquisition times, seconds, as a float64 array.

        The successful attempt's own duration is drawn from the body truncated to
        ``(0, attempt_window_s]`` by inverse transform, so the draw is exact
        rather than rejection-based and the stream consumption does not depend
        on the realised values.
        """
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size!r}")
        body = self.body
        upper = float(body.cdf(self.attempt_window_s))
        uniform = rng.random(size)
        success = body.ppf(uniform * upper)
        failures = rng.geometric(1.0 - self.abort_probability, size=size) - 1
        return np.asarray(failures * self.attempt_window_s + success, dtype=float)

    def _truncated_body_moments(self) -> tuple[float, float]:
        """``(E[L|L<=w], E[L^2|L<=w])`` of the truncated body, closed form."""
        mu = float(np.log(self.median_s))
        sigma = float(self.log_sigma)
        standardised = (float(np.log(self.attempt_window_s)) - mu) / sigma
        denominator = float(stats.norm.cdf(standardised))
        if denominator <= 0.0:
            raise ValueError(
                "the attempt window lies so far below the body's median that the "
                "truncation has zero probability; the model is outside its declared "
                "validity range"
            )
        first = (
            float(np.exp(mu + 0.5 * sigma * sigma))
            * float(stats.norm.cdf(standardised - sigma))
            / denominator
        )
        second = (
            float(np.exp(2.0 * mu + 2.0 * sigma * sigma))
            * float(stats.norm.cdf(standardised - 2.0 * sigma))
            / denominator
        )
        return first, second

    def moments(self) -> tuple[float, float]:
        """``(mean, standard deviation)`` of the acquisition time, seconds.

        The failed-attempt count is geometric with success probability
        ``1 - p``, so its mean is ``p / (1 - p)`` and its variance
        ``p / (1 - p) ** 2``. The successful attempt's duration is the body
        truncated at the attempt window, whose first two moments are the closed
        form in the module docstring. Count and duration are independent by
        construction, so means add and variances add.
        """
        first, second = self._truncated_body_moments()
        p = self.abort_probability
        count_mean = p / (1.0 - p)
        count_variance = p / (1.0 - p) ** 2
        mean = count_mean * self.attempt_window_s + first
        variance = count_variance * self.attempt_window_s**2 + second - first * first
        return mean, float(np.sqrt(max(variance, 0.0)))

    def moments_quadrature(self, quadrature_points: int = 256) -> tuple[float, float]:
        """:meth:`moments` by Gauss-Legendre quadrature of the truncated density.

        An independent path to the same two numbers: it touches the probability
        DENSITY function where :meth:`moments` touches the DISTRIBUTION function
        and the lognormal moment identity. Used by the test suite as a
        cross-check and not in any hot path, because computing the nodes is an
        eigenvalue problem whose cost grows as the cube of
        ``quadrature_points``.
        """
        if quadrature_points < 16:
            raise ValueError(
                f"quadrature_points must be at least 16, got {quadrature_points!r}"
            )
        body = self.body
        upper = float(body.cdf(self.attempt_window_s))
        nodes, weights = _legendre_nodes(int(quadrature_points))
        # Map the Gauss-Legendre nodes from [-1, 1] onto (0, attempt_window_s].
        half = 0.5 * self.attempt_window_s
        grid = half * (nodes + 1.0)
        density = body.pdf(grid) / upper
        first = float(np.sum(weights * half * grid * density))
        second = float(np.sum(weights * half * grid * grid * density))
        p = self.abort_probability
        count_mean = p / (1.0 - p)
        count_variance = p / (1.0 - p) ** 2
        mean = count_mean * self.attempt_window_s + first
        variance = count_variance * self.attempt_window_s**2 + second - first * first
        return mean, float(np.sqrt(max(variance, 0.0)))

    def quantile(self, level: float, size: int = 200_000, seed: int = 96_096) -> float:
        """Monte Carlo quantile of the acquisition time at ``level``, seconds.

        The mixture has no closed-form quantile. A fixed seed is used so the
        value a caller gets is reproducible; the sampling error at the default
        size is reported by ``validation/validate_acquisition_model.py``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}")
        draws = self.sample(size, np.random.default_rng(seed))
        return float(np.quantile(draws, level))
