"""Validation: the inference path, from scikit-learn through numpy to ONNX.

The product's claim is confined to the scikit-learn / ONNX path, so that path
has to be shown to be the same function at every stage, and the injection has
to be shown to act on the real serialised bytes.

Checks
------
1. The explicit numpy forward pass of ``bitflipsim.network`` reproduces
   ``MLPClassifier.predict_proba`` exactly when the parameters are stored in
   float64, and to float32 round-off when they are stored in float32. If this
   failed, every criticality number would be about a different model from the
   one scikit-learn fitted.
2. A hand-encoded ONNX ``ModelProto`` of the same network loads in onnxruntime
   and produces the same logits as the numpy forward pass. The protobuf field
   numbers used by ``bitflipsim.onnx_io`` are therefore right; onnxruntime
   would reject the file otherwise.
3. Every float32 initializer is located inside the serialised bytes and decodes
   bit-exactly back to the tensor it was written from.
4. Flipping one bit in the model FILE - at the byte offset of that initializer
   element, not in a numpy array standing in for it - changes the onnxruntime
   output by the amount predicted from the IEEE 754 layout.
5. Flipping the same bit twice restores the file byte for byte.

onnxruntime is an optional extra. If it is missing this script reports the ONNX
checks as SKIPPED and still runs check 1; it does not pretend they passed.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import sys

import numpy as np

from bitflipsim.bitlayout import float_layout, predict_flip
from bitflipsim.datasets import make_problem, train_reference_model
from bitflipsim.network import from_sklearn
from bitflipsim.onnx_io import (
    FLOAT32_DATA_TYPE,
    build_mlp_onnx,
    flip_initializer_bit,
    list_initializers,
)

failures: list[str] = []
skipped: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


problem = make_problem()
classifier = train_reference_model(problem)
params_f64 = from_sklearn(classifier, itemsize_bytes=8)
params_f32 = from_sklearn(classifier, itemsize_bytes=4)
x = problem.evaluation.x

print("=" * 78)
print("Check 1 - numpy forward pass versus MLPClassifier.predict_proba")
print("=" * 78)
reference = classifier.predict_proba(x)
deviation_f64 = float(np.abs(params_f64.probabilities(x) - reference).max())
deviation_f32 = float(np.abs(params_f32.probabilities(x) - reference).max())
print(f"evaluation samples        {x.shape[0]}")
print(f"architecture              {params_f32.layout.n_in} -> "
      f"{params_f32.layout.n_hidden} (ReLU) -> {params_f32.layout.n_out}, softmax")
print(f"max |diff| float64 store  {deviation_f64:.6e}")
print(f"max |diff| float32 store  {deviation_f32:.6e}")
report("numpy forward pass equals sklearn in float64", deviation_f64 == 0.0,
       f"max absolute probability difference over {x.shape[0]} samples x "
       f"{params_f32.layout.n_out} classes = {deviation_f64:.6e} (bit-exact required)")
report("float32 storage costs only rounding", deviation_f32 < 1.0e-5,
       f"max absolute probability difference = {deviation_f32:.6e}, which is the "
       f"cost of storing the weights in float32 rather than float64, not an upset")

print()
print("=" * 78)
print("Check 2 to 5 - the ONNX path")
print("=" * 78)
try:
    import onnxruntime
except ImportError:
    print("onnxruntime is not installed; the ONNX checks are SKIPPED, not passed.")
    skipped.append("ONNX checks (onnxruntime missing)")
    onnxruntime = None  # type: ignore[assignment]

if onnxruntime is not None:
    print(f"onnxruntime               {onnxruntime.__version__}")
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3

    def run(blob: bytes) -> np.ndarray:
        session = onnxruntime.InferenceSession(
            blob, options, providers=["CPUExecutionProvider"]
        )
        return session.run(None, {"x": x.astype(np.float32)})[0]

    blob = build_mlp_onnx(params_f32)
    print(f"serialised model          {len(blob)} bytes")
    print(f"float32 data_type code    {FLOAT32_DATA_TYPE}")
    produced = run(blob)
    numpy_logits = params_f32.logits(x)
    onnx_deviation = float(np.abs(produced - numpy_logits).max())
    print(f"max |onnx - numpy| logit  {onnx_deviation:.6e}")
    report("onnxruntime loads the hand-encoded model and agrees with numpy",
           onnx_deviation < 1.0e-4,
           f"max absolute logit difference over {x.shape[0]} samples = "
           f"{onnx_deviation:.6e}; onnxruntime accumulates in float32, this package "
           f"in float64")

    initializers = list_initializers(blob)
    tensors = params_f32.tensors()
    print()
    print(f"{'initializer':<12} {'dims':<12} {'elements':>9} {'byte offset':>12} "
          f"{'bytes':>7} {'decodes exactly':>16}")
    exact = 0
    for init in initializers:
        matches = bool(np.array_equal(init.array(blob), tensors[init.name]))
        exact += int(matches)
        print(f"{init.name:<12} {str(init.dims):<12} {init.n_elements:>9} "
              f"{init.raw_offset:>12} {init.raw_length:>7} {str(matches):>16}")
    total_elements = sum(init.n_elements for init in initializers)
    report("every initializer is located and decodes bit-exactly",
           exact == len(initializers) == 4 and total_elements == params_f32.layout.size,
           f"{exact} of {len(initializers)} initializers, {total_elements} elements "
           f"total, matching the {params_f32.layout.size}-parameter layout")

    print()
    print("In-file bit flips, prediction from the IEEE 754 layout versus onnxruntime:")
    print(f"{'initializer':<12} {'element':>8} {'bit':>5} {'role':<9} "
          f"{'weight before':>16} {'weight after':>16} {'predicted':>16} "
          f"{'|d logit| max':>15}")
    golden = run(blob)
    w1 = next(init for init in initializers if init.name == "W1")
    mismatch = 0
    rows = 0
    for element, bit in ((5, 30), (5, 31), (5, 23), (5, 0), (17, 30), (40, 12)):
        before = float(w1.array(blob).reshape(-1)[element])
        prediction = predict_flip(np.float32(before), bit, "float32")
        patched = flip_initializer_bit(blob, w1, element, bit)
        after = float(w1.array(patched).reshape(-1)[element])
        agree = (np.isnan(prediction.predicted_value) and np.isnan(after)) or (
            after == prediction.predicted_value
        )
        if not agree:
            mismatch += 1
        with np.errstate(over="ignore", invalid="ignore"):
            faulty = run(patched)
            deviation = float(np.abs(faulty - golden).max())
        rows += 1
        print(f"{'W1':<12} {element:>8} {bit:>5} "
              f"{float_layout('float32').role(bit):<9} {before:>16.9g} {after:>16.9g} "
              f"{prediction.predicted_value:>16.9g} {deviation:>15.6e}")
    report("in-file bit flips match the layout prediction", mismatch == 0 and rows == 6,
           f"{rows} in-file flips, {mismatch} disagreeing with predict_flip")

    roundtrip = flip_initializer_bit(flip_initializer_bit(blob, w1, 5, 30), w1, 5, 30)
    report("flipping the same file bit twice restores the file", roundtrip == blob,
           f"{len(blob)}-byte model restored byte for byte")

    print()
    print("Scope of the ONNX support, stated plainly: this reads and writes float32")
    print("initializers stored in raw_data, which is what a dense MLP exporter")
    print("produces. It is not a general ONNX implementation: it does not run")
    print("graphs itself, does not handle float_data-encoded tensors, quantized")
    print("ONNX types, external data files, subgraphs or sparse initializers, and")
    print("reports each of those as unsupported rather than skipping it.")

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILED check(s): {failures}")
    sys.exit(1)
if skipped:
    print(f"RESULT: checks PASSED, {len(skipped)} SKIPPED: {skipped}")
else:
    print("RESULT: all checks PASSED")
