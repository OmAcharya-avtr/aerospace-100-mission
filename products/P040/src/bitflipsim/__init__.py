"""bitflipsim - single-event-upset effects on small scikit-learn / ONNX inference.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use. This package models upsets; it does not qualify
parts, and nothing in it substitutes for radiation testing of real hardware.

Public surface, grouped by module:

* :mod:`bitflipsim.bitlayout`  - IEEE 754 and two's-complement layouts,
  discovered from numpy at runtime, plus exact single-bit-flip predictions.
* :mod:`bitflipsim.flux`       - ``lambda = flux * cross_section * bits``, the
  Poisson count model and its sampling errors.
* :mod:`bitflipsim.injection`  - uniform bit-site sampling and exact bitwise
  injection into numpy arrays.
* :mod:`bitflipsim.network`    - a bit-addressable two-layer MLP whose forward
  pass reproduces ``MLPClassifier.predict_proba``.
* :mod:`bitflipsim.datasets`   - the deterministic synthetic problem and the
  reference trained model.
* :mod:`bitflipsim.criticality`- the ground-truth per-bit sweep, the two
  baselines, and degradation avoided per protected byte.
* :mod:`bitflipsim.predictor`  - the learned criticality predictor with an
  ensemble uncertainty output.
* :mod:`bitflipsim.mitigation` - clamping with a derived bound, triplication
  voters, periodic reload, and their costs.
* :mod:`bitflipsim.campaign`   - Poisson-sized campaigns with standard errors.
* :mod:`bitflipsim.onnx_io`    - optional float32 initializer extraction from an
  ONNX model, so the injection target can come from a real exported file.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .bitlayout import (
    FlipPrediction,
    FloatLayout,
    IntLayout,
    bit_site_count,
    exponent_field,
    exponent_scale_factor,
    flip_bit,
    float_layout,
    from_bits,
    int_layout,
    mantissa_field,
    predict_flip,
    to_bits,
    verify_float_layout,
)
from .campaign import (
    CampaignResult,
    activation_campaign,
    flux_campaign,
    parameter_campaign,
    single_upset_sites,
    trials_for_standard_error,
)
from .criticality import (
    CriticalitySweep,
    ProtectionResult,
    accuracy,
    evaluate_protection,
    exponent_bit_baseline_scores,
    magnitude_baseline_scores,
    mean_total_variation,
    oracle_scores,
    random_scores,
    sweep_bit_criticality,
    sweep_quantized_bit_criticality,
    total_variation,
)
from .datasets import Problem, make_problem, reference_parameters, train_reference_model
from .flux import (
    ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT,
    ILLUSTRATIVE_FLUX_PER_CM2_S,
    CountStatistics,
    UpsetRate,
    count_statistics,
    poisson_validity,
    sample_upset_counts,
    upset_rate,
)
from .injection import (
    BitUpset,
    apply_upsets,
    distinct_site_count,
    net_flipped_bits,
    sample_upsets,
    upset_site_histogram,
)
from .mitigation import (
    MitigationCost,
    bit_majority_vote,
    clamp_logit_bound,
    clamp_parameters,
    clamping_cost,
    expected_live_upsets,
    measure_clamp_latency,
    reload_cost,
    triplication_cost,
    word_majority_vote,
)
from .network import (
    Int8Quantization,
    MlpParameters,
    ParameterLayout,
    from_sklearn,
    make_layout,
    quantize_int8,
    softmax,
)
from .predictor import (
    FEATURE_NAMES,
    CriticalityPrediction,
    CriticalityPredictor,
    ParameterSplit,
    build_features,
    split_parameters,
    uncertainty_calibration,
)

__all__ = [
    "FEATURE_NAMES",
    "ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT",
    "ILLUSTRATIVE_FLUX_PER_CM2_S",
    "BitUpset",
    "CampaignResult",
    "CountStatistics",
    "CriticalityPrediction",
    "CriticalityPredictor",
    "CriticalitySweep",
    "FlipPrediction",
    "FloatLayout",
    "Int8Quantization",
    "IntLayout",
    "MitigationCost",
    "MlpParameters",
    "ParameterLayout",
    "ParameterSplit",
    "Problem",
    "ProtectionResult",
    "UpsetRate",
    "__version__",
    "accuracy",
    "activation_campaign",
    "apply_upsets",
    "bit_majority_vote",
    "bit_site_count",
    "build_features",
    "clamp_logit_bound",
    "clamp_parameters",
    "clamping_cost",
    "count_statistics",
    "distinct_site_count",
    "evaluate_protection",
    "expected_live_upsets",
    "exponent_bit_baseline_scores",
    "exponent_field",
    "exponent_scale_factor",
    "flip_bit",
    "float_layout",
    "flux_campaign",
    "from_bits",
    "from_sklearn",
    "int_layout",
    "magnitude_baseline_scores",
    "make_layout",
    "make_problem",
    "mantissa_field",
    "mean_total_variation",
    "measure_clamp_latency",
    "net_flipped_bits",
    "oracle_scores",
    "parameter_campaign",
    "poisson_validity",
    "predict_flip",
    "quantize_int8",
    "random_scores",
    "reference_parameters",
    "reload_cost",
    "sample_upset_counts",
    "sample_upsets",
    "single_upset_sites",
    "softmax",
    "split_parameters",
    "sweep_bit_criticality",
    "sweep_quantized_bit_criticality",
    "to_bits",
    "total_variation",
    "train_reference_model",
    "trials_for_standard_error",
    "triplication_cost",
    "uncertainty_calibration",
    "upset_rate",
    "upset_site_histogram",
    "verify_float_layout",
    "word_majority_vote",
]
