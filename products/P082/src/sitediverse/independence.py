"""Baseline 1: the independence assumption, implemented in order to be refuted.

This module is the thing this product exists to measure the error of. It is
implemented carefully and correctly, because an under-strength version of the
claim being refuted would inflate the apparent size of the refutation, and
because it is what site-diversity practice actually does: the product of
per-site unavailabilities is the textbook first estimate of a diversity
network's joint unavailability, and ITU-R P.1815-1 and the P.618 site-diversity
material both start from a joint statistic and warn that an independence
assumption is optimistic, which is precisely the direction measured here.

**(14) Naive product of marginals, the specified baseline.** With long-term
marginal cloud probabilities ``qbar_i``,

    P_naive = prod_i qbar_i

computed as ``exp( sum_i log qbar_i )``. The exponential-of-sum form is not
decoration. For eight stations at a marginal rate of 0.3 the product is 6.6e-05
and the direct product of eight floats loses about three units in the last place
to repeated rounding, which is immaterial; what is NOT immaterial is the
complement. A reader who wants the AVAILABILITY of the network writes
``1 - prod q_i``, and for a well-separated network where the product is 1e-12
that expression returns a number whose error is 1e-16 absolute -- fine -- while
the reader who then asks for the number of nines gets ``-log10(1 - A)`` of the
subtraction, which has lost every digit of the product. This module therefore
reports the UNAVAILABILITY as primary everywhere and derives availability from
it, never the other way round. ``availability_nines`` works from the
unavailability directly.

**(15) Conditional independence, honouring the marginal climatology.** The
naive product uses one long-term rate per station and therefore assumes the
cloud climatology is stationary as well as spatially independent. Those are two
assumptions and they fail in opposite directions. The second baseline here
removes only the stationarity assumption,

    P_conditional = (1/T) sum_t prod_i q_it

the epoch-by-epoch product averaged over the record. By Jensen's inequality on
the product, ``P_conditional >= P_naive`` whenever the ``q_it`` co-vary across
stations and ``P_conditional <= P_naive`` when they anti-vary. Two stations with
seasonal cloud maxima six months apart are JOINTLY clear more often than their
annual means suggest, and the naive product therefore overstates their joint
unavailability. Separating this from the spatial-correlation effect is what
makes the headline decomposition of equation (16) exact rather than
approximate.

**(16) The overstatement decomposition.** With ``P_measured`` the directly
observed joint cloud rate of ``sitediverse.empirical``,

    P_measured - P_naive  =  (P_conditional - P_naive)   [co-modulation]
                          +  (P_measured - P_conditional) [spatial correlation]

an exact identity by construction, with no cross term. The first bracket is
what the stationarity assumption costs and the second is what the independence
assumption costs. Both are reported, because a single "overstatement" number
that mixes them is not attributable to anything a network designer can change,
whereas the second term is, and it is the one the correlation models of
``sitediverse.correlation`` and ``sitediverse.learned`` are trying to predict.

The headline figure of this product is the pair ``(P_naive, P_measured)`` and
the ratio and difference between them, with the climatology sampling
uncertainty of ``sitediverse.uncertainty`` attached to each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .climatology import Climatology


def product_of_marginals(rates: np.ndarray) -> float:
    """Equation (14): the product of marginal unavailabilities, in log space.

    Parameters
    ----------
    rates:
        ``(n,)`` marginal cloud probabilities, each strictly inside ``(0, 1]``.
        A rate of exactly zero makes the product zero and is accepted; a
        negative rate or one above one raises.

    Returns
    -------
    float
        The product, in ``[0, 1]``.
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"rates must be a non-empty one-dimensional array, got shape "
            f"{values.shape}"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            "every marginal rate must lie in [0, 1]; got a value of "
            f"{float(np.min(values)):.6e} to {float(np.max(values)):.6e}"
        )
    if np.any(values == 0.0):
        return 0.0
    return float(np.exp(np.sum(np.log(values))))


def conditional_product(rate_series: np.ndarray) -> float:
    """Equation (15): the epoch-by-epoch product, averaged over the record.

    Parameters
    ----------
    rate_series:
        ``(epochs, n)`` marginal cloud probability per epoch per station.

    Returns
    -------
    float
    """
    values = np.asarray(rate_series, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"rate_series must be two-dimensional, got {values.ndim}D")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"rate_series must not be empty, got shape {values.shape}")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("every marginal rate must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        log_rate = np.log(values)
    product = np.exp(np.sum(log_rate, axis=1))
    product[np.any(values == 0.0, axis=1)] = 0.0
    return float(np.mean(product))


def availability_nines(unavailability: float) -> float:
    """Number of nines of availability, ``-log10(unavailability)``.

    Derived from the unavailability directly and never from ``1 - A``, which is
    the subtraction equation (14) exists to avoid. An unavailability of exactly
    zero has unbounded nines and returns ``inf``; a negative value raises.
    """
    value = float(unavailability)
    if value < 0.0:
        raise ValueError(f"unavailability must be non-negative, got {value!r}")
    if value == 0.0:
        return float("inf")
    if value >= 1.0:
        return 0.0 if value == 1.0 else float(-np.log10(value))
    return float(-np.log10(value))


@dataclass(frozen=True)
class Overstatement:
    """The equation (16) decomposition for one station set.

    Attributes
    ----------
    naive:
        ``P_naive`` of equation (14).
    conditional:
        ``P_conditional`` of equation (15).
    measured:
        The directly observed joint cloud rate.
    sites:
        Indices of the stations, in the order they were supplied.
    """

    naive: float
    conditional: float
    measured: float
    sites: tuple[int, ...]

    @property
    def comodulation_term(self) -> float:
        """``P_conditional - P_naive``: what the stationarity assumption costs."""
        return self.conditional - self.naive

    @property
    def correlation_term(self) -> float:
        """``P_measured - P_conditional``: what independence costs."""
        return self.measured - self.conditional

    @property
    def total_difference(self) -> float:
        """``P_measured - P_naive``: the headline absolute overstatement."""
        return self.measured - self.naive

    @property
    def ratio(self) -> float:
        """``P_measured / P_naive``: the headline overstatement factor.

        Returns ``inf`` when the naive product is exactly zero and the measured
        rate is not, and ``nan`` when both are zero, because no finite ratio
        describes either case and returning a number would invent one.
        """
        if self.naive == 0.0:
            return float("inf") if self.measured > 0.0 else float("nan")
        return self.measured / self.naive

    @property
    def availability_deficit(self) -> float:
        """``A_naive - A_measured``, the availability the assumption invents.

        Equal to ``total_difference`` in exact arithmetic, and computed from the
        unavailabilities rather than from a difference of two availabilities
        near one, which would lose the leading digits.
        """
        return self.total_difference

    @property
    def nines_lost(self) -> float:
        """Nines of availability the assumption claims and the record denies."""
        return availability_nines(self.naive) - availability_nines(self.measured)

    def identity_residual(self) -> float:
        """``total - (comodulation + correlation)``. Zero in exact arithmetic.

        Equation (16) is exact by construction, so this is a pure rounding
        residual and is the right quantity to put a tolerance on. It is NOT
        asserted to be exactly zero anywhere: it is a sum of three
        floating-point differences and cannot be.
        """
        return self.total_difference - (self.comodulation_term + self.correlation_term)


def overstatement(
    climatology: Climatology,
    cloud: np.ndarray,
    sites: tuple[int, ...] | list[int],
) -> Overstatement:
    """Equation (16) for one station set, from an estimated climatology.

    Parameters
    ----------
    climatology:
        Estimated from the record by ``sitediverse.climatology``.
    cloud:
        ``(epochs, S)`` boolean record.
    sites:
        Indices of the stations in the network, at least two, distinct.

    Returns
    -------
    Overstatement
    """
    index = tuple(int(site) for site in sites)
    if len(index) < 2:
        raise ValueError(
            f"a diversity network needs at least 2 stations, got {len(index)}"
        )
    if len(set(index)) != len(index):
        raise ValueError(f"station indices must be distinct, got {index}")
    indicator = np.asarray(cloud)
    if indicator.ndim != 2:
        raise ValueError(f"cloud must be two-dimensional, got {indicator.ndim}D")
    count = indicator.shape[1]
    for site in index:
        if not 0 <= site < count:
            raise ValueError(
                f"station index {site} is outside [0, {count}) for this record"
            )
    selected = list(index)
    return Overstatement(
        naive=product_of_marginals(climatology.long_term_rate[selected]),
        conditional=conditional_product(climatology.rate_series()[:, selected]),
        measured=float(np.mean(np.all(indicator[:, selected], axis=1))),
        sites=index,
    )
