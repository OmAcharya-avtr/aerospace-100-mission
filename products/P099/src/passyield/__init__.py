"""passyield -- realised versus booked data yield of a ground-station contact.

Research-grade software. **Not flight-qualified, not certified and not
approved for operational aerospace use.**

WHAT IT COMPUTES
----------------
The nominal-versus-realised data-yield shortfall of a scheduled ground-station
contact, decomposed into acquisition, re-synchronisation and idle time, and
returned **as a distribution with its quantiles** rather than as a point
number. A contact scheduler books a window and a rate; the shortfall against
that booking is a random variable, and a schedule can only be written against
a random variable by booking a quantile of it.

THE PIECES
----------
- :mod:`passyield.geometry` -- pass windows, elevation, azimuth and slant
  range from an SGP4 propagation of a published element set.
- :mod:`passyield.link` -- achievable rate against elevation, from a cited
  link budget.
- :mod:`passyield.decompose` -- the per-pass accounting identity and its
  reported residual.
- :mod:`passyield.baselines` -- baseline 1 (the booked figure), baseline 2
  (the analytic law, by numerical convolution) and baseline 3 (ground truth,
  with a pass-level bootstrap).
- :mod:`passyield.bounds` -- brackets from mathematics the estimators do not
  use.
- :mod:`passyield.distribution` -- the headline output object.
- :mod:`passyield.predictor` -- the learned per-pass predictor and its
  uncertainty output.
- :mod:`passyield.scoring` -- CRPS, PIT calibration and interval coverage.
- :mod:`passyield.dataset` -- the committed per-pass record and the two
  held-out splits.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "baselines",
    "bounds",
    "dataset",
    "decompose",
    "distribution",
    "geometry",
    "link",
    "predictor",
    "report",
    "scoring",
]
