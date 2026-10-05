"""ONNX serialisation, initializer location and in-file bit patching.

The onnxruntime-dependent tests skip cleanly if onnxruntime is absent, which is
the documented optional dependency; the pure-protobuf tests always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.onnx_io import (
    FLOAT32_DATA_TYPE,
    build_mlp_onnx,
    flip_initializer_bit,
    list_initializers,
)

onnxruntime = pytest.importorskip("onnxruntime", reason="optional ONNX extra not installed")


def _session(blob: bytes):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(blob, options, providers=["CPUExecutionProvider"])


def test_float32_data_type_code():
    # onnx.TensorProto.DataType.FLOAT == 1
    assert FLOAT32_DATA_TYPE == 1


def test_initializers_are_located_and_decode_exactly(params):
    blob = build_mlp_onnx(params)
    initializers = {init.name: init for init in list_initializers(blob)}
    assert set(initializers) == {"W1", "b1", "W2", "b2"}
    tensors = params.tensors()
    for name, init in initializers.items():
        assert init.n_elements == tensors[name].size
        assert np.array_equal(init.array(blob), tensors[name])
    total = sum(init.n_elements for init in initializers.values())
    assert total == params.layout.size


def test_onnxruntime_loads_the_hand_encoded_model(params, problem):
    blob = build_mlp_onnx(params)
    session = _session(blob)
    assert [i.name for i in session.get_inputs()] == ["x"]
    assert [o.name for o in session.get_outputs()] == ["logits"]
    x = problem.evaluation.x.astype(np.float32)
    produced = session.run(None, {"x": x})[0]
    reference = params.logits(problem.evaluation.x)
    # float32 accumulation in onnxruntime versus float64 accumulation here.
    assert np.abs(produced - reference).max() < 1e-4


def test_patching_a_file_bit_changes_the_onnxruntime_output(params, problem):
    blob = build_mlp_onnx(params)
    w1 = next(init for init in list_initializers(blob) if init.name == "W1")
    x = problem.evaluation.x.astype(np.float32)
    golden = _session(blob).run(None, {"x": x})[0]
    patched = flip_initializer_bit(blob, w1, element_index=5, bit_position=30)
    assert len(patched) == len(blob)
    assert sum(a != b for a, b in zip(patched, blob, strict=True)) <= 4
    faulty = _session(patched).run(None, {"x": x})[0]
    deviation = float(np.abs(faulty - golden).max())
    assert deviation > 1e30, deviation


def test_patching_a_mantissa_bit_is_a_small_change(params, problem):
    blob = build_mlp_onnx(params)
    w1 = next(init for init in list_initializers(blob) if init.name == "W1")
    x = problem.evaluation.x.astype(np.float32)
    golden = _session(blob).run(None, {"x": x})[0]
    patched = flip_initializer_bit(blob, w1, element_index=5, bit_position=0)
    faulty = _session(patched).run(None, {"x": x})[0]
    assert float(np.abs(faulty - golden).max()) < 1e-4


def test_patch_roundtrip_restores_the_original_bytes(params):
    blob = build_mlp_onnx(params)
    w2 = next(init for init in list_initializers(blob) if init.name == "W2")
    once = flip_initializer_bit(blob, w2, 3, 17)
    twice = flip_initializer_bit(once, w2, 3, 17)
    assert twice == blob


def test_patch_validation(params):
    blob = build_mlp_onnx(params)
    b2 = next(init for init in list_initializers(blob) if init.name == "b2")
    with pytest.raises(ValueError, match="element_index"):
        flip_initializer_bit(blob, b2, 99, 0)
    with pytest.raises(ValueError, match="bit_position"):
        flip_initializer_bit(blob, b2, 0, 32)


def test_no_graph_is_reported():
    with pytest.raises(ValueError, match="no GraphProto"):
        list_initializers(b"\x08\x08")
