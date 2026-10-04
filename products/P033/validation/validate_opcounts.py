#!/usr/bin/env python3
"""Validation 1 --- analytic operation counts against hand counts.

Checks the analytic cost model in :mod:`edgeinfer.ops` against arithmetic done
by hand for two small networks, and against ``onnxruntime``'s own numerical
output for the same two graphs.

Hand counts are reproduced in full in this script's output so that a reader can
redo them without opening the source. The same counts appear in
``tests/test_ops.py`` as test-comment derivations.

References for the formulas under test:
  - Golub, G. H. & Van Loan, C. F. (2013), Matrix Computations, 4th ed.,
    section 1.1.11: an (M, K) x (K, N) dense product costs 2*M*K*N flops.
  - Sze, V., Chen, Y.-H., Yang, T.-J. & Emer, J. S. (2017), "Efficient
    Processing of Deep Neural Networks: A Tutorial and Survey", Proceedings of
    the IEEE 105(12), 2295-2329, section II-A: a convolutional layer costs
    N * M * prod(S_out) * (C/group) * prod(K) multiply-accumulates.
  - ONNX Standard, operator set 17, Conv: output spatial size
    floor((in + pad_begin + pad_end - dilation*(kernel-1) - 1)/stride) + 1.
  - Aho, A. V., Lam, M. S., Sethi, R. & Ullman, J. D. (2006), Compilers:
    Principles, Techniques, and Tools, 2nd ed., section 8.4: liveness analysis.

Runtime: under 5 s on one CPU core.
"""

from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort

from edgeinfer.analytic import analytic_estimate, peak_activation_bytes
from edgeinfer.dataset import hand_counted_cnn, hand_counted_mlp
from edgeinfer.environment import describe_environment
from edgeinfer.ops import node_cost
from edgeinfer.roofline import DeviceModel

TOLERANCE = 0  # operation and byte counts are integers: the check is exact

HAND_MLP = {
    "label": "X(1,4) -> Gemm(4,6)+b -> Relu -> Gemm(6,3)+b -> Y, float32",
    "macs": 42,
    "flops": 99,
    "traffic_bytes": 328,
    "weight_bytes": 204,
    "peak_activation_bytes": 48,
    "derivation": [
        "tensors (4 B/element): X 16, W0 96, B0 24, G0 24, H0 24, W1 72, B1 12, Y 12",
        "MACs  gemm0 = 1*4*6 = 24; gemm1 = 1*6*3 = 18; Relu = 0        -> 42",
        "FLOPs gemm0 = 2*24 + 6 = 54; act0 = 6; gemm1 = 2*18 + 3 = 39  -> 99",
        "traffic gemm0 = (16+96+24)+24 = 160; act0 = 24+24 = 48;",
        "        gemm1 = (24+72+12)+12 = 120                           -> 328 B",
        "weights = 96 + 24 + 72 + 12                                   -> 204 B",
        "liveness: entry 16; gemm0 16+24 = 40; act0 24+24 = 48; gemm1 24+12 = 36",
        "peak activations                                              -> 48 B",
    ],
}

HAND_CNN = {
    "label": (
        "X(1,1,6,6) -> Conv(1->2,3x3,pad1) -> Relu -> MaxPool(2x2,s2) "
        "-> Reshape(1,18) -> Gemm(18,2)+b -> Y, float32"
    ),
    "macs": 684,
    "flops": 1514,
    "traffic_bytes": 1832,
    "weight_bytes": 240,
    "peak_activation_bytes": 576,
    "derivation": [
        "shapes: conv out floor((6+1+1-3)/1)+1 = 6, so C0 = (1,2,6,6);",
        "        pool out floor((6-2)/2)+1 = 3, so P0 = (1,2,3,3)",
        "tensors: X 144, CW0 72, C0 288, R0 288, P0 72, RS 16 (int64), F 72,",
        "         FW 144, FB 8, Y 8",
        "MACs  conv0 = 1*2*36*1*9 = 648; fc = 1*18*2 = 36               -> 684",
        "FLOPs conv0 = 1296; crelu0 = 72; pool0 = 18*4 = 72; flat = 0;",
        "      fc = 2*36 + 2 = 74                                       -> 1514",
        "traffic conv0 = (144+72)+288 = 504; crelu0 = 288+288 = 576;",
        "        pool0 = 288+72 = 360; flat = (72+16)+72 = 160;",
        "        fc = (72+144+8)+8 = 232                                -> 1832 B",
        "weights = 72 + 16 + 144 + 8                                    -> 240 B",
        "liveness: entry 144; conv0 144+288 = 432; crelu0 288+288 = 576;",
        "          pool0 288+72 = 360; flat 72+72 = 144; fc 72+8 = 80",
        "peak activations                                               -> 576 B",
    ],
}


def _check(label: str, computed: int, expected: int, unit: str) -> bool:
    ok = abs(computed - expected) <= TOLERANCE
    status = "PASS" if ok else "FAIL"
    print(
        f"  {label:<28} computed = {computed:>10} {unit:<2}  "
        f"hand = {expected:>10} {unit:<2}  {status}"
    )
    return ok


def validate_network(model, hand: dict, device: DeviceModel) -> list[bool]:
    print(f"\n{hand['label']}")
    print("-" * 78)
    print("  hand derivation:")
    for line in hand["derivation"]:
        print(f"    {line}")
    print()

    estimate = analytic_estimate(model.graph, device)
    peak_act, live_by_step = peak_activation_bytes(model.graph)
    results = [
        _check("total MACs", estimate.total_macs, hand["macs"], "-"),
        _check("total FLOPs", estimate.total_flops, hand["flops"], "-"),
        _check(
            "compulsory traffic", estimate.total_traffic_bytes, hand["traffic_bytes"], "B"
        ),
        _check("resident weight bytes", estimate.weight_bytes, hand["weight_bytes"], "B"),
        _check(
            "peak activation bytes", peak_act, hand["peak_activation_bytes"], "B"
        ),
    ]
    print("\n  per-node breakdown (analytic):")
    print(
        f"    {'node':<10}{'op':<20}{'MACs':>10}{'FLOPs':>10}{'bytes':>10}"
        f"{'live after':>12}"
    )
    for node in model.graph.nodes:
        cost = node_cost(node, model.graph)
        print(
            f"    {node.name:<10}{node.op_type:<20}{cost.macs:>10}{cost.flops:>10}"
            f"{cost.bytes_total:>10}{live_by_step[node.name]:>12}"
        )
    return results


def validate_runtime_numerics(model, expected_fn, label: str) -> bool:
    """Run the graph in onnxruntime and compare against NumPy."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model.model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )
    feed = model.input_feed(7)
    got = session.run(None, feed)[0]
    expected = expected_fn(feed, model.initializer_arrays)
    error = float(np.max(np.abs(got - expected)))
    tolerance = 1e-5
    ok = error <= tolerance
    print(
        f"  {label:<28} max |onnxruntime - numpy| = {error:.3e}  "
        f"tolerance = {tolerance:.0e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def mlp_reference(feed: dict, arrays: dict) -> np.ndarray:
    hidden = np.maximum(feed["X"] @ arrays["W0"] + arrays["B0"], 0.0)
    return hidden @ arrays["W1"] + arrays["B1"]


def cnn_reference(feed: dict, arrays: dict) -> np.ndarray:
    """Direct convolution written out, so the reference shares no code with
    either onnxruntime or the cost model."""
    x = feed["X"][0, 0]
    weight = arrays["CW0"]
    padded = np.zeros((8, 8), dtype=np.float32)
    padded[1:7, 1:7] = x
    conv = np.zeros((2, 6, 6), dtype=np.float32)
    for channel in range(2):
        for row in range(6):
            for col in range(6):
                patch = padded[row : row + 3, col : col + 3]
                conv[channel, row, col] = float(np.sum(patch * weight[channel, 0]))
    relu = np.maximum(conv, 0.0)
    pooled = np.zeros((2, 3, 3), dtype=np.float32)
    for channel in range(2):
        for row in range(3):
            for col in range(3):
                pooled[channel, row, col] = relu[
                    channel, 2 * row : 2 * row + 2, 2 * col : 2 * col + 2
                ].max()
    flat = pooled.reshape(1, 18)
    return flat @ arrays["FW"] + arrays["FB"]


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 1: analytic operation counts vs hand counts")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "NOTE: this check is exact integer arithmetic and is independent of the "
        "host; it is the one validation in this repository that a different "
        "machine cannot change."
    )

    device = DeviceModel(
        "counting-only", 1e9, 1e9, source="declared; irrelevant to an operation count"
    )
    results: list[bool] = []
    results += validate_network(hand_counted_mlp(), HAND_MLP, device)
    results += validate_network(hand_counted_cnn(), HAND_CNN, device)

    print("\nonnxruntime numerical agreement (the graphs must also be runnable)")
    print("-" * 78)
    results.append(
        validate_runtime_numerics(hand_counted_mlp(), mlp_reference, "hand MLP")
    )
    results.append(
        validate_runtime_numerics(hand_counted_cnn(), cnn_reference, "hand CNN")
    )

    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
