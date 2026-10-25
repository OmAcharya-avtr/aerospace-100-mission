"""Exceptions raised by :mod:`orbitdrift`.

Every one carries an actionable message. ``CovarianceRefused`` is not an error
condition in the usual sense -- it is a deliberate, tested refusal, and
:mod:`orbitdrift.guard` documents why.
"""

from __future__ import annotations


class OrbitDriftError(Exception):
    """Base class for every exception raised by this package."""


class InvalidTleError(OrbitDriftError, ValueError):
    """A two-line element set could not be parsed or initialised."""


class PropagationError(OrbitDriftError, RuntimeError):
    """A propagator reported an error code, or produced a non-finite state."""


class InsufficientDataError(OrbitDriftError, ValueError):
    """Too few samples to fit the requested growth model."""


class CovarianceRefused(OrbitDriftError, PermissionError):
    """A caller asked this package for a covariance, or for a conjunction use.

    This package measures *model divergence* between two propagators. It has
    no access to tracking residuals, so nothing it produces is an orbit
    determination covariance, and nothing it produces may be used as the
    input covariance of a conjunction assessment. See
    :mod:`orbitdrift.guard`.
    """
