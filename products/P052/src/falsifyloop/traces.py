"""Uniformly sampled multi-signal traces, the only object the requirement
language is defined over.

A :class:`Trace` is a finite, uniformly sampled record of named real-valued
signals produced by one simulation. Everything in :mod:`falsifyloop.requirements`
is defined inductively over a trace, following the standard robustness-degree
construction for signal temporal logic (Fainekos & Pappas 2009; Donze & Maler
2010), restricted here to the fragment documented in that module.

Units
-----
``times`` is in seconds. Each signal carries whatever unit the caller's
simulator produced; this module does not convert and does not guess. The unit of
a robustness value is the unit of the term it was computed from, which is why
:class:`falsifyloop.requirements.Predicate` accepts an explicit ``scale``.

Restrictions, and why they are enforced rather than documented
--------------------------------------------------------------
* **Uniform sampling.** The time-bounded operators convert a window in seconds
  to an integer sample offset. That conversion is only exact for a uniform grid,
  so a non-uniform ``times`` is rejected instead of being silently rounded.
* **At least two samples.** The first-difference term needs two samples to be
  defined at all.
* **Finite values.** ``nan`` would break the sign agreement between the
  robustness semantics and the Boolean semantics, which is the property this
  package is built on, so it is rejected at construction.

References
----------
Fainekos, G. E. and Pappas, G. J. (2009), "Robustness of temporal logic
specifications for continuous-time signals", Theoretical Computer Science
410(42). The robustness-degree semantics.

Donze, A. and Maler, O. (2010), "Robust satisfaction of temporal logic over
real-valued signals", FORMATS 2010. Time-bounded operators over sampled signals.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: Relative tolerance used when checking that ``times`` is uniformly spaced.
UNIFORM_RTOL = 1e-9


class Trace:
    """A uniformly sampled, finite, multi-signal trace.

    Parameters
    ----------
    times:
        Strictly increasing sample times in seconds, shape ``(T,)``, ``T >= 2``,
        uniformly spaced to within :data:`UNIFORM_RTOL` relative tolerance.
    signals:
        Mapping from signal name to a finite real array of shape ``(T,)``. The
        unit of each signal is the caller's; it is recorded nowhere and
        converted nowhere.

    Raises
    ------
    TypeError
        If ``signals`` is not a mapping, or a signal is not array-convertible.
    ValueError
        On fewer than two samples, non-monotonic or non-uniform ``times``, a
        length mismatch, a non-finite value, or an empty signal set.
    """

    __slots__ = ("_dt", "_names", "_signals", "_times")

    def __init__(self, times: np.ndarray, signals: Mapping[str, np.ndarray]) -> None:
        t = np.asarray(times, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got shape {t.shape}")
        if t.size < 2:
            raise ValueError(f"a trace needs at least 2 samples, got {t.size}")
        if not np.all(np.isfinite(t)):
            raise ValueError("times contains a non-finite value")
        steps = np.diff(t)
        if not np.all(steps > 0.0):
            raise ValueError("times must be strictly increasing")
        dt = float(steps[0])
        if not np.allclose(steps, dt, rtol=UNIFORM_RTOL, atol=0.0):
            raise ValueError(
                "times must be uniformly spaced; the time-bounded operators convert "
                f"a window in seconds to an integer sample offset. Observed step range "
                f"[{steps.min():.6g}, {steps.max():.6g}] s against dt = {dt:.6g} s."
            )
        if not isinstance(signals, Mapping):
            raise TypeError(f"signals must be a mapping of name to array, got {type(signals)!r}")
        if not signals:
            raise ValueError("a trace needs at least one signal")

        stored: dict[str, np.ndarray] = {}
        for name, values in signals.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"signal names must be non-empty strings, got {name!r}")
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"signal {name!r} must be one-dimensional, got shape {arr.shape}")
            if arr.size != t.size:
                raise ValueError(
                    f"signal {name!r} has {arr.size} samples but times has {t.size}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"signal {name!r} contains a non-finite value; nan or inf would break the "
                    "sign agreement between the robustness and Boolean semantics"
                )
            stored[name] = arr

        self._times = t
        self._signals = stored
        self._dt = dt
        self._names = tuple(sorted(stored))

    @property
    def times(self) -> np.ndarray:
        """Sample times in seconds, shape ``(T,)``."""
        return self._times

    @property
    def dt(self) -> float:
        """Sample interval in seconds."""
        return self._dt

    @property
    def length(self) -> int:
        """Number of samples ``T``."""
        return int(self._times.size)

    @property
    def names(self) -> tuple[str, ...]:
        """Signal names, sorted."""
        return self._names

    @property
    def duration(self) -> float:
        """``times[-1] - times[0]`` in seconds."""
        return float(self._times[-1] - self._times[0])

    def signal(self, name: str) -> np.ndarray:
        """Return the signal named ``name``, shape ``(T,)``.

        Raises
        ------
        KeyError
            If the trace has no such signal; the message lists what it does have.
        """
        try:
            return self._signals[name]
        except KeyError:
            raise KeyError(
                f"trace has no signal {name!r}; available signals are {list(self._names)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._signals

    def __repr__(self) -> str:
        return (
            f"Trace(T={self.length}, dt={self._dt:.6g} s, "
            f"duration={self.duration:.6g} s, signals={list(self._names)})"
        )
