"""twindiverge -- attribute a twin-versus-reality residual, after refusing what cannot be.

The order of operations is the product. Identifiability analysis comes first and
gates attribution: a parameter that the supplied telemetry cannot distinguish is
**refused**, as a reported outcome rather than an error, and the identifiable
parameter *combinations* are reported in its place.

Modules, in the order they run
------------------------------
:mod:`twindiverge.plant`
    The declared digital twin: a reaction-wheel assembly, equation (1), with the
    forward sensitivity equations and the on-paper derivation of its exact
    unidentifiable direction.
:mod:`twindiverge.telemetry`
    Telemetry sets -- which channels, over which campaign, at what noise level.
    An identifiability question has no answer without one.
:mod:`twindiverge.identifiability`
    **Baseline 1.** Numerical rank, conditioning, Fisher information,
    Cramer-Rao bounds with their finiteness clause, identifiable combinations,
    and practical identifiability separated from structural.
:mod:`twindiverge.attribution`
    **Baseline 2.** Generalised least squares on the identifiable subspace with
    confidence regions, the specified first-order sensitivity attribution, the
    stronger constrained single-parameter maximum likelihood, and the
    equivalence family the telemetry cannot distinguish.
:mod:`twindiverge.profile`
    **Baseline 3.** Nonlinear profile likelihood: the ground truth the
    linearised covariance is scored against, with a flat profile overriding
    whatever the covariance claims.
:mod:`twindiverge.features`, :mod:`twindiverge.learned`
    The learned residual-signature classifier and its confidence output.
:mod:`twindiverge.synthetic`, :mod:`twindiverge.pipeline`,
:mod:`twindiverge.scoring`, :mod:`twindiverge.report`
    Deterministic case generation, the ordered pipeline, scores with exact
    intervals, and plain-text rendering.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
