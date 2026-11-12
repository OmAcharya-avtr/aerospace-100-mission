"""Origin of a number, and the rule that a synthetic figure can never be measured.

This product reports figures measured on an NVIDIA Jetson Orin Nano Engineering
Reference Developer Kit Super beside figures computed from synthetic fixtures
written by this repository's own generator. The two must never be confused, and
the mechanism here is the one that stops it: every trace, and every result
derived from one, carries an :class:`Origin`, and an aggregate is no better than
its worst input.

``MEASURED ∧ SYNTHETIC = SYNTHETIC``. The lattice is two elements, so the meet
is the minimum under ``SYNTHETIC < MEASURED``; it is monotone, associative,
commutative and idempotent, with identity ``MEASURED`` and absorbing element
``SYNTHETIC``. P095 ``boardprobe`` of this mission carries a three-element
version of the same lattice over individual sensor readings and is the
conceptual layer below this one; this package does not import it, depend on it
or compare any number against it. The relationship is prose only.

The one-way rule is deliberate: there is no function here that promotes a
``SYNTHETIC`` result to ``MEASURED``. Labelling a figure as measured goes
through :func:`require_measured`, which raises rather than returning a label.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable


class Origin(enum.Enum):
    """Where a number came from.

    ``MEASURED``
        Read from a CSV trace produced by ``data/campaign.py`` running on the
        board. The only origin a figure in a validation document may carry.
    ``SYNTHETIC``
        Produced by ``tests/fixtures/synthetic_trace.py``. Schema-identical to a
        measured trace and physically plausible, but not a measurement of
        anything. Every file that generator writes is named ``SYNTHETIC_*``.
    """

    SYNTHETIC = "SYNTHETIC"
    MEASURED = "MEASURED"

    @property
    def rank(self) -> int:
        """Position in the lattice; ``SYNTHETIC`` is 0 and absorbing."""
        return 0 if self is Origin.SYNTHETIC else 1

    def __str__(self) -> str:
        return self.value


class NotMeasuredError(RuntimeError):
    """Raised when a figure would be labelled measured without measured inputs."""


def meet(*origins: Origin | Iterable[Origin]) -> Origin:
    """Lattice meet of the given origins; ``SYNTHETIC`` if any input is synthetic.

    Parameters
    ----------
    *origins:
        :class:`Origin` values, or iterables of them, in any mixture.

    Returns
    -------
    Origin
        ``MEASURED`` only if every input is ``MEASURED``. The meet of no inputs
        is the identity ``MEASURED``, which is why :func:`require_measured` is a
        separate call and an empty input set is rejected there rather than being
        silently labelled a measurement.

    Raises
    ------
    TypeError
        If any element is not an :class:`Origin`.
    """
    flat: list[Origin] = []
    for item in origins:
        if isinstance(item, Origin):
            flat.append(item)
        elif isinstance(item, str) or not isinstance(item, Iterable):
            raise TypeError(f"meet() takes Origin values, got {type(item).__name__}")
        else:
            for element in item:
                if not isinstance(element, Origin):
                    raise TypeError(
                        f"meet() takes Origin values, got {type(element).__name__}"
                    )
                flat.append(element)
    worst = Origin.MEASURED
    for origin in flat:
        if origin.rank < worst.rank:
            worst = origin
    return worst


def require_measured(origin: Origin, what: str, n_inputs: int | None = None) -> Origin:
    """Return ``origin`` if it is ``MEASURED``, else raise :class:`NotMeasuredError`.

    Parameters
    ----------
    origin:
        The origin of the figure about to be labelled a measurement.
    what:
        Name of the figure, quoted in the error message.
    n_inputs:
        Number of inputs the figure was derived from, when the caller knows it.
        Zero is rejected: the meet of an empty set is the lattice identity
        ``MEASURED``, so a check run over no input files would otherwise emit a
        vacuous measured verdict. This is the one place that is caught.

    Raises
    ------
    NotMeasuredError
        If ``origin`` is ``SYNTHETIC``, or if ``n_inputs`` is zero.
    """
    if n_inputs is not None and n_inputs <= 0:
        raise NotMeasuredError(
            f"{what}: refusing to label a measurement derived from {n_inputs} inputs; "
            "an empty input set cannot produce a measured figure"
        )
    if origin is not Origin.MEASURED:
        raise NotMeasuredError(
            f"{what}: origin is {origin}, not MEASURED; a figure derived from "
            "synthetic input may not be reported as a measurement"
        )
    return origin


def label(origin: Origin) -> str:
    """Short human-readable tag for a figure of this origin, for report headers."""
    if origin is Origin.MEASURED:
        return "MEASURED on the board (MAXN_SUPER, one ambient)"
    return "SYNTHETIC fixture, not a measurement of anything"
