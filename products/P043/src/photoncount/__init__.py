"""photoncount: a photon-counting optical receiver chain for photon-starved links.

Six layers, each usable on its own:

===========================  ========================================================
:mod:`photoncount.poisson`   Poisson counting statistics for signal, background, dark
:mod:`photoncount.webb`      Webb distribution for APD excess-noise counting statistics
:mod:`photoncount.ppm`       M-ary PPM slot statistics, exact error probability, soft metrics
:mod:`photoncount.deadtime`  Paralyzable and non-paralyzable dead time, forward and inverse
:mod:`photoncount.afterpulse` Afterpulsing as an excess-count process
:mod:`photoncount.capacity`  Rates available on the PPM Poisson channel
===========================  ========================================================

and three layers for getting it onto hardware:

=============================  ======================================================
:mod:`photoncount.simulate`    Event-level detector simulator, the ground truth
:mod:`photoncount.hal`         One backend contract: simulated, and a device stub
:mod:`photoncount.ops`         Preflight, capture and backout as executable checks
=============================  ======================================================

plus :mod:`photoncount.dataset` and :mod:`photoncount.correction`, the learned
rate correction and the closed-form inversions it is measured against.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use. Validation level 3,
hardware-pending: every timing and resource number in this repository came from
a shared cloud container, not from a flight-representative board.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "afterpulse",
    "capacity",
    "correction",
    "dataset",
    "deadtime",
    "hal",
    "ops",
    "poisson",
    "ppm",
    "simulate",
    "webb",
]
