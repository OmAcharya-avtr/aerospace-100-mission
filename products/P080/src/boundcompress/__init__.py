"""boundcompress: certify that a declared error bound held, and price the guarantee.

Not a codec. ``numcodecs`` and ``zfpy`` are the right tools for compressing an
array, and CCSDS 121.0-B-3 and 122.0-B-2 are the standards a flight
implementation should use. What they do not answer is the flight-software
question this package exists for: *can I prove that no sample exceeded my
declared error bound, and what did I pay in rate to be able to prove it?*

The answer is a :class:`~boundcompress.certificate.Certificate` -- the
measured maximum per-sample error, the index where it occurred, the realised
rate in actual encoded bytes, and a rate-distortion comparison at exactly
matched distortion -- so the cost of the guarantee is visible rather than
asserted.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

Quick use
---------

>>> import numpy as np
>>> from boundcompress import encode, decode
>>> values = 27.0 + 1.6 * np.sin(np.linspace(0.0, 6.0, 512))
>>> result = encode(values, "abs", 1e-3, predictor="linear")
>>> result.certificate.bound_holds
True
>>> bool(np.max(np.abs(values - decode(result.blob))) <= 1e-3)
True
"""

from __future__ import annotations

from .baselines import BaselineResult, measure_all
from .benchmark import Cell, Verdict, break_even_arrays, summarise
from .certificate import Certificate, certify, digest_bytes
from .codec import EncodeResult, decode, encode, encode_decode
from .container import Container, ContainerError, block_sizes, pack, unpack
from .dataset import SIGNAL_NAMES, Signal, generate, generate_all, training_signals
from .entropy import shannon_entropy_bits
from .learned import LearnedPredictor
from .predictors import CLASSICAL_NAMES, ClassicalPredictor, get_predictor, noise_gain
from .quantise import MODES, BoundNotAchievable, QuantiserSpec, make_spec

# ``boundcompress.quantise`` is the module, and the function of the same name
# inside it is deliberately NOT re-exported here. Binding the function to the
# package attribute shadows the submodule, so ``from boundcompress import
# quantise`` would hand a caller the function and ``quantise.MODES`` would
# raise AttributeError. The function is reached as
# ``boundcompress.quantise.quantise``.
from .ratedistortion import SweepRow, geometric_bounds, sweep_signal
from .report import format_certificate, format_table

__version__ = "0.1.0"

__all__ = [
    "BaselineResult",
    "BoundNotAchievable",
    "CLASSICAL_NAMES",
    "Cell",
    "Certificate",
    "ClassicalPredictor",
    "Container",
    "ContainerError",
    "EncodeResult",
    "LearnedPredictor",
    "MODES",
    "QuantiserSpec",
    "SIGNAL_NAMES",
    "Signal",
    "SweepRow",
    "Verdict",
    "__version__",
    "block_sizes",
    "break_even_arrays",
    "certify",
    "decode",
    "digest_bytes",
    "encode",
    "encode_decode",
    "format_certificate",
    "format_table",
    "generate",
    "generate_all",
    "geometric_bounds",
    "get_predictor",
    "make_spec",
    "measure_all",
    "noise_gain",
    "pack",
    "shannon_entropy_bits",
    "summarise",
    "sweep_signal",
    "training_signals",
    "unpack",
]
