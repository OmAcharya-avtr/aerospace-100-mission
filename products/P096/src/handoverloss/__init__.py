"""handoverloss: which mechanism dominates the data loss across an optical handover.

End-to-end data loss across a ground-station handover on an optical downlink,
attributed separately to **acquisition time**, **protocol restart** and **buffer
drain**, with a bootstrap interval on each. The headline output is WHICH of the
three dominates, not the total, because that is the number that tells an
operator whether to buy faster acquisition, a different protocol or more buffer.

Research-grade software. Not flight-qualified, not certified, and not approved
for operational aerospace use.

The three baselines and the learned component, in the order they must be read:

* :func:`~handoverloss.baselines.acquisition_only_loss_bits` -- baseline 1, the
  assumption that acquisition time is the whole story, implemented at full
  strength as the thing being refuted.
* :func:`~handoverloss.baselines.fluid_decomposition` -- baseline 2, the
  analytic fluid buffer-drain and restart model, closed form.
* :func:`~handoverloss.accounting.attribute` -- baseline 3, ground truth by
  direct accounting over a simulated trace, with event-level bootstrap
  intervals. This is the reference the other two and the model are scored
  against.
* :class:`~handoverloss.predictor.HandoverLossPredictor` -- the learned
  component, with an uncertainty output, scored on held-out station pairs and
  held-out time separately.

:func:`~handoverloss.bounds.loss_bracket` is an independently derived bracket on
the total, from conservation and capacity counting alone, which uses none of the
models above.
"""

from __future__ import annotations

from .accounting import AttributionReport, MechanismShare, attribute
from .acquisition import AcquisitionModel
from .baselines import (
    FluidDecomposition,
    acquisition_only_loss_bits,
    fluid_decomposition,
    fluid_decomposition_array,
)
from .bootstrap import RESAMPLING_UNIT, BootstrapInterval, event_bootstrap
from .bounds import LossBracket, bracket_slack_bits, loss_bracket, max_service_bits
from .corpus import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    Corpus,
    CorpusAbsentError,
    Split,
    load_corpus,
    station_pair_split,
    time_split,
)
from .predictor import HandoverLossPredictor, Prediction, PredictorScore, score_baselines
from .sequence import MECHANISMS, HandoverSequence, Phase
from .trace import EventParameters, TraceResult, simulate

__version__ = "0.1.0"

__all__ = [
    "AcquisitionModel",
    "AttributionReport",
    "BootstrapInterval",
    "Corpus",
    "CorpusAbsentError",
    "EventParameters",
    "FEATURE_COLUMNS",
    "FluidDecomposition",
    "HandoverLossPredictor",
    "HandoverSequence",
    "LossBracket",
    "MECHANISMS",
    "MechanismShare",
    "Phase",
    "Prediction",
    "PredictorScore",
    "RESAMPLING_UNIT",
    "Split",
    "TARGET_COLUMNS",
    "TraceResult",
    "__version__",
    "acquisition_only_loss_bits",
    "attribute",
    "bracket_slack_bits",
    "event_bootstrap",
    "fluid_decomposition",
    "fluid_decomposition_array",
    "load_corpus",
    "loss_bracket",
    "max_service_bits",
    "score_baselines",
    "simulate",
    "station_pair_split",
    "time_split",
]
