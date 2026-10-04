"""Build and read ONNX ModelProto bytes without the ``onnx`` package.

Why: ``onnxruntime`` is available in the target environment but ``onnx`` is
not, and ``onnxruntime.InferenceSession`` accepts a serialised ModelProto as
``bytes``. The field numbers below are taken from the ONNX Standard's
``onnx.proto`` (ONNX IR, github.com/onnx/onnx, ``onnx/onnx.proto``) and are
**verified empirically** against ``onnxruntime/datasets/mul_1.onnx``, a model
shipped inside the installed ``onnxruntime`` wheel, by
``tests/test_onnx_io.py::TestFieldNumbersAgainstShippedModel``. If a field
number here were wrong, that test would read the wrong value and fail, and
:func:`build_model` would produce a model ``onnxruntime`` refuses to load.

Scope: enough of the IR to express the graphs this package characterises ---
float tensors with static shapes, initialisers held as ``raw_data``, and
integer/int-list operator attributes. Sparse tensors, subgraphs, functions,
training fields, maps and sequences are not implemented; :func:`parse_model`
raises on a construct it cannot represent rather than returning a graph that
silently omits it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgeinfer import pbwire as pw
from edgeinfer.graph import GraphError, ModelGraph, Node, TensorSpec

__all__ = [
    "DEFAULT_IR_VERSION",
    "DEFAULT_OPSET",
    "IR_ONLY_ATTRIBUTES",
    "OnnxModel",
    "build_model",
    "parse_model",
    "shipped_model_path",
]

#: ONNX IR version written by :func:`build_model`. IR 9 corresponds to ONNX
#: release 1.14 and is accepted by every ``onnxruntime`` 1.x this package
#: supports; a newer IR would be rejected by older runtimes for no benefit.
DEFAULT_IR_VERSION: int = 9

#: Default operator-set version for the ``ai.onnx`` domain.
DEFAULT_OPSET: int = 17

# -- ModelProto ---------------------------------------------------------
_MODEL_IR_VERSION = 1
_MODEL_PRODUCER_NAME = 2
_MODEL_GRAPH = 7
_MODEL_OPSET_IMPORT = 8

# -- GraphProto ---------------------------------------------------------
_GRAPH_NODE = 1
_GRAPH_NAME = 2
_GRAPH_INITIALIZER = 5
_GRAPH_INPUT = 11
_GRAPH_OUTPUT = 12

# -- NodeProto ----------------------------------------------------------
_NODE_INPUT = 1
_NODE_OUTPUT = 2
_NODE_NAME = 3
_NODE_OP_TYPE = 4
_NODE_ATTRIBUTE = 5

# -- AttributeProto -----------------------------------------------------
_ATTR_NAME = 1
_ATTR_I = 3
_ATTR_INTS = 8
_ATTR_TYPE = 20
_ATTR_TYPE_INT = 2
_ATTR_TYPE_INTS = 7

# -- ValueInfoProto / TypeProto / TensorShapeProto ----------------------
_VI_NAME = 1
_VI_TYPE = 2
_TYPE_TENSOR_TYPE = 1
_TT_ELEM_TYPE = 1
_TT_SHAPE = 2
_SHAPE_DIM = 1
_DIM_VALUE = 1

# -- TensorProto --------------------------------------------------------
_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_FLOAT_DATA = 4
_TENSOR_NAME = 8
_TENSOR_RAW_DATA = 9

# -- OperatorSetIdProto -------------------------------------------------
_OPSET_DOMAIN = 1
_OPSET_VERSION = 2

#: ONNX ``TensorProto.DataType`` codes, ONNX Standard ``onnx.proto``.
_ELEM_TYPE = {"float32": 1, "uint8": 2, "int8": 3, "int32": 6, "int64": 7, "float64": 11}
_ELEM_TYPE_INV = {v: k for k, v in _ELEM_TYPE.items()}
_NUMPY_DTYPE = {"float32": np.float32, "float64": np.float64, "int32": np.int32,
                "int64": np.int64, "int8": np.int8, "uint8": np.uint8}


@dataclass(frozen=True)
class OnnxModel:
    """A serialised ONNX model with the :class:`ModelGraph` it corresponds to.

    Attributes
    ----------
    graph
        The IR view used by the analytic cost model.
    model_bytes
        Serialised ModelProto, ready for ``onnxruntime.InferenceSession``.
    initializer_arrays
        The weight arrays, keyed by tensor name, so a test can recompute a
        reference output without re-parsing the model.
    """

    graph: ModelGraph
    model_bytes: bytes
    initializer_arrays: dict[str, np.ndarray]

    def input_feed(self, seed: int = 0) -> dict[str, np.ndarray]:
        """Deterministic random inputs for every graph input.

        Values are standard normal from ``numpy.random.default_rng(seed)``,
        cast to the input dtype. Input *values* do not change the analytic
        cost, and ``onnxruntime`` kernel timing for the operators used here is
        data-independent, so a fixed arbitrary feed is sufficient; a model
        with data-dependent control flow would need otherwise and this package
        does not support one.
        """
        rng = np.random.default_rng(seed)
        feed: dict[str, np.ndarray] = {}
        for spec in self.graph.inputs:
            dtype = _NUMPY_DTYPE.get(spec.dtype, np.float32)
            if np.issubdtype(dtype, np.integer):
                feed[spec.name] = rng.integers(0, 4, size=spec.shape).astype(dtype)
            else:
                feed[spec.name] = rng.standard_normal(spec.shape).astype(dtype)
        return feed


def _encode_tensor(name: str, array: np.ndarray) -> bytes:
    dtype_name = str(array.dtype)
    if dtype_name not in _ELEM_TYPE:
        raise GraphError(
            f"initialiser {name!r}: dtype {dtype_name!r} not supported by edgeinfer's "
            f"ONNX writer (supported: {sorted(_ELEM_TYPE)})"
        )
    contiguous = np.ascontiguousarray(array)
    body = b"".join(pw.field_varint(_TENSOR_DIMS, int(d)) for d in contiguous.shape)
    body += pw.field_varint(_TENSOR_DATA_TYPE, _ELEM_TYPE[dtype_name])
    body += pw.field_bytes(_TENSOR_RAW_DATA, contiguous.tobytes())
    body += pw.field_string(_TENSOR_NAME, name)
    return body


def _encode_value_info(spec: TensorSpec) -> bytes:
    if spec.dtype not in _ELEM_TYPE:
        raise GraphError(f"tensor {spec.name!r}: dtype {spec.dtype!r} not supported")
    dims = b"".join(pw.field_message(_SHAPE_DIM, pw.field_varint(_DIM_VALUE, int(d)))
                    for d in spec.shape)
    tensor_type = pw.field_varint(_TT_ELEM_TYPE, _ELEM_TYPE[spec.dtype])
    tensor_type += pw.field_message(_TT_SHAPE, dims)
    type_proto = pw.field_message(_TYPE_TENSOR_TYPE, tensor_type)
    return pw.field_string(_VI_NAME, spec.name) + pw.field_message(_VI_TYPE, type_proto)


def _encode_attribute(name: str, value: object) -> bytes:
    body = pw.field_string(_ATTR_NAME, name)
    if isinstance(value, bool):
        raise GraphError(f"attribute {name!r}: pass an int, not a bool")
    if isinstance(value, int):
        body += pw.field_varint(_ATTR_I, value)
        body += pw.field_varint(_ATTR_TYPE, _ATTR_TYPE_INT)
        return body
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, int) or isinstance(item, bool):
                raise GraphError(
                    f"attribute {name!r}: only int and list-of-int attributes are "
                    f"supported, got element {item!r}"
                )
            body += pw.field_varint(_ATTR_INTS, int(item))
        body += pw.field_varint(_ATTR_TYPE, _ATTR_TYPE_INTS)
        return body
    raise GraphError(
        f"attribute {name!r}: only int and list-of-int attributes are supported, "
        f"got {type(value).__name__}"
    )


#: Attributes that exist in edgeinfer's IR but not in the ONNX operator
#: schema, and must therefore not be serialised. ``Reshape`` takes its target
#: shape as an *input tensor* from opset 5 on (ONNX Standard, ``Reshape-5``),
#: while edgeinfer's analytic shape rules read shapes from attributes because
#: the IR does not carry constant tensor values. Serialising such an attribute
#: makes ``onnxruntime`` reject the model with "Unrecognized attribute";
#: :func:`parse_model` recovers it from the constant initialiser instead.
IR_ONLY_ATTRIBUTES: dict[str, frozenset[str]] = {
    "Reshape": frozenset({"shape"}),
    "Flatten": frozenset({"shape"}),
}


def _encode_node(node: Node) -> bytes:
    body = b"".join(pw.field_string(_NODE_INPUT, name) for name in node.inputs)
    body += b"".join(pw.field_string(_NODE_OUTPUT, name) for name in node.outputs)
    body += pw.field_string(_NODE_NAME, node.name)
    body += pw.field_string(_NODE_OP_TYPE, node.op_type)
    skip = IR_ONLY_ATTRIBUTES.get(node.op_type, frozenset())
    for key, value in node.attrs.items():
        if key in skip:
            continue
        body += pw.field_message(_NODE_ATTRIBUTE, _encode_attribute(key, value))
    return body


def build_model(
    graph: ModelGraph,
    initializer_arrays: dict[str, np.ndarray],
    *,
    ir_version: int = DEFAULT_IR_VERSION,
    opset: int = DEFAULT_OPSET,
    producer: str = "edgeinfer",
) -> OnnxModel:
    """Serialise a :class:`ModelGraph` plus its weights into ONNX bytes.

    Parameters
    ----------
    graph
        Graph to serialise. Shape inference is run if needed, because graph
        outputs need declared shapes in the ModelProto.
    initializer_arrays
        One array per entry of ``graph.initializers``, keyed by tensor name.
        Shapes and dtypes must match the declared specs exactly; a mismatch
        raises rather than being silently reshaped, because a silently
        reshaped weight changes the operation count the analytic model
        reported.
    ir_version, opset
        ONNX IR and ``ai.onnx`` operator-set versions.
    producer
        ``producer_name`` field.

    Returns
    -------
    OnnxModel

    Raises
    ------
    GraphError
        On a missing or mismatched initialiser, or an unsupported dtype or
        attribute type.
    """
    graph.infer_shapes()
    for spec in graph.initializers:
        if spec.name not in initializer_arrays:
            raise GraphError(f"no array supplied for initialiser {spec.name!r}")
        array = initializer_arrays[spec.name]
        if tuple(array.shape) != spec.shape:
            raise GraphError(
                f"initialiser {spec.name!r}: array shape {tuple(array.shape)} does not "
                f"match declared shape {spec.shape}"
            )
        if str(array.dtype) != spec.dtype:
            raise GraphError(
                f"initialiser {spec.name!r}: array dtype {array.dtype} does not match "
                f"declared dtype {spec.dtype}"
            )

    body = b"".join(pw.field_message(_GRAPH_NODE, _encode_node(n)) for n in graph.nodes)
    body += pw.field_string(_GRAPH_NAME, graph.name)
    body += b"".join(
        pw.field_message(_GRAPH_INITIALIZER, _encode_tensor(s.name, initializer_arrays[s.name]))
        for s in graph.initializers
    )
    body += b"".join(pw.field_message(_GRAPH_INPUT, _encode_value_info(s)) for s in graph.inputs)
    body += b"".join(
        pw.field_message(_GRAPH_OUTPUT, _encode_value_info(graph.spec(name)))
        for name in graph.outputs
    )

    model = pw.field_varint(_MODEL_IR_VERSION, ir_version)
    model += pw.field_string(_MODEL_PRODUCER_NAME, producer)
    model += pw.field_message(_MODEL_GRAPH, body)
    model += pw.field_message(
        _MODEL_OPSET_IMPORT,
        pw.field_string(_OPSET_DOMAIN, "") + pw.field_varint(_OPSET_VERSION, opset),
    )
    return OnnxModel(
        graph=graph,
        model_bytes=model,
        initializer_arrays=dict(initializer_arrays),
    )


# -- parsing ------------------------------------------------------------


def _first(fields: list[tuple[int, int, object]], number: int) -> object | None:
    for f, _wt, value in fields:
        if f == number:
            return value
    return None


def _all(fields: list[tuple[int, int, object]], number: int) -> list[object]:
    return [value for f, _wt, value in fields if f == number]


def _parse_value_info(buf: bytes) -> TensorSpec:
    fields = list(pw.iter_fields(buf))
    name = bytes(_first(fields, _VI_NAME) or b"").decode()
    type_proto = _first(fields, _VI_TYPE)
    if type_proto is None:
        raise GraphError(f"value_info {name!r} has no type")
    tt = _first(list(pw.iter_fields(bytes(type_proto))), _TYPE_TENSOR_TYPE)
    if tt is None:
        raise GraphError(
            f"value_info {name!r}: only tensor_type is supported; maps, sequences and "
            "optionals are not representable in edgeinfer's IR"
        )
    tt_fields = list(pw.iter_fields(bytes(tt)))
    elem = int(_first(tt_fields, _TT_ELEM_TYPE) or 0)
    shape_buf = _first(tt_fields, _TT_SHAPE)
    dims: list[int] = []
    if shape_buf is not None:
        for dim in _all(list(pw.iter_fields(bytes(shape_buf))), _SHAPE_DIM):
            value = _first(list(pw.iter_fields(bytes(dim))), _DIM_VALUE)
            if value is None:
                raise GraphError(
                    f"value_info {name!r} has a symbolic (dim_param) axis; make shapes "
                    "static before characterising the model"
                )
            dims.append(int(value))
    if elem not in _ELEM_TYPE_INV:
        raise GraphError(f"value_info {name!r}: unsupported ONNX elem_type {elem}")
    return TensorSpec(name, tuple(dims), _ELEM_TYPE_INV[elem])


def _parse_tensor(buf: bytes) -> tuple[TensorSpec, np.ndarray]:
    fields = list(pw.iter_fields(buf))
    name = bytes(_first(fields, _TENSOR_NAME) or b"").decode()
    dims = tuple(int(v) for v in _all(fields, _TENSOR_DIMS))  # type: ignore[arg-type]
    elem = int(_first(fields, _TENSOR_DATA_TYPE) or 0)
    if elem not in _ELEM_TYPE_INV:
        raise GraphError(f"initialiser {name!r}: unsupported ONNX elem_type {elem}")
    dtype_name = _ELEM_TYPE_INV[elem]
    raw = _first(fields, _TENSOR_RAW_DATA)
    if raw is not None:
        array = np.frombuffer(bytes(raw), dtype=_NUMPY_DTYPE[dtype_name]).reshape(dims)
    else:
        packed = _first(fields, _TENSOR_FLOAT_DATA)
        if packed is None:
            raise GraphError(
                f"initialiser {name!r} carries neither raw_data nor float_data; "
                "edgeinfer's parser supports those two encodings only"
            )
        array = np.frombuffer(bytes(packed), dtype="<f4").reshape(dims).astype(
            _NUMPY_DTYPE[dtype_name]
        )
    return TensorSpec(name, dims, dtype_name), np.array(array)


def _parse_attributes(node_fields: list[tuple[int, int, object]]) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for buf in _all(node_fields, _NODE_ATTRIBUTE):
        fields = list(pw.iter_fields(bytes(buf)))
        name = bytes(_first(fields, _ATTR_NAME) or b"").decode()
        kind = _first(fields, _ATTR_TYPE)
        if kind == _ATTR_TYPE_INT:
            attrs[name] = int(_first(fields, _ATTR_I) or 0)
        elif kind == _ATTR_TYPE_INTS:
            attrs[name] = [int(v) for v in _all(fields, _ATTR_INTS)]  # type: ignore[arg-type]
        # Other attribute kinds are skipped: they do not enter the cost model,
        # and a skipped attribute cannot change an operation count for the
        # operators edgeinfer supports.
    return attrs


def parse_model(model_bytes: bytes) -> tuple[ModelGraph, dict[str, np.ndarray]]:
    """Parse ONNX bytes into a :class:`ModelGraph` and its initialiser arrays.

    Only the subset documented in the module docstring is understood.

    Raises
    ------
    GraphError
        On a symbolic axis, an unsupported element type, or a tensor encoding
        other than ``raw_data``/``float_data``.
    """
    fields = list(pw.iter_fields(model_bytes))
    graph_buf = _first(fields, _MODEL_GRAPH)
    if graph_buf is None:
        raise GraphError("ModelProto has no graph field")
    gfields = list(pw.iter_fields(bytes(graph_buf)))
    name = bytes(_first(gfields, _GRAPH_NAME) or b"onnx_graph").decode() or "onnx_graph"

    arrays: dict[str, np.ndarray] = {}
    init_specs: list[TensorSpec] = []
    for buf in _all(gfields, _GRAPH_INITIALIZER):
        spec, array = _parse_tensor(bytes(buf))
        init_specs.append(spec)
        arrays[spec.name] = array
    init_names = {s.name for s in init_specs}

    all_inputs = [_parse_value_info(bytes(b)) for b in _all(gfields, _GRAPH_INPUT)]
    # An IR-3 model lists initialisers among the graph inputs; a later one does
    # not. Exclude them either way so graph.inputs means "fed at run time".
    inputs = tuple(s for s in all_inputs if s.name not in init_names)
    outputs = tuple(_parse_value_info(bytes(b)).name for b in _all(gfields, _GRAPH_OUTPUT))

    nodes: list[Node] = []
    for i, buf in enumerate(_all(gfields, _GRAPH_NODE)):
        nfields = list(pw.iter_fields(bytes(buf)))
        node_name = bytes(_first(nfields, _NODE_NAME) or b"").decode() or f"node_{i}"
        op_type = bytes(_first(nfields, _NODE_OP_TYPE) or b"").decode()
        nodes.append(
            Node(
                name=node_name,
                op_type=op_type,
                inputs=tuple(bytes(v).decode() for v in _all(nfields, _NODE_INPUT)),
                outputs=tuple(bytes(v).decode() for v in _all(nfields, _NODE_OUTPUT)),
                attrs=_parse_attributes(nfields),
            )
        )
    # ONNX carries a Reshape's target shape as a second input tensor, while
    # edgeinfer's IR carries it as a 'shape' attribute (the IR holds no
    # constant tensor values). Recover it here when that input is a constant
    # initialiser, so that a build -> parse -> build round trip is exact. When
    # it is not constant the attribute stays absent and shape inference
    # refuses, which is the honest outcome: such a graph cannot be
    # characterised statically.
    recovered: list[Node] = []
    for node in nodes:
        if node.op_type in ("Reshape", "Flatten") and len(node.inputs) > 1:
            target = arrays.get(node.inputs[1])
            if target is not None:
                attrs = dict(node.attrs)
                attrs["shape"] = [int(v) for v in np.asarray(target).ravel()]
                node = Node(
                    name=node.name,
                    op_type=node.op_type,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    attrs=attrs,
                )
        recovered.append(node)

    graph = ModelGraph(
        name=name,
        inputs=inputs,
        outputs=outputs,
        nodes=tuple(recovered),
        initializers=tuple(init_specs),
    )
    return graph, arrays


def shipped_model_path(filename: str = "mul_1.onnx") -> str:
    """Absolute path of a model shipped inside the installed ``onnxruntime``.

    ``onnxruntime`` bundles three tiny models under ``onnxruntime/datasets``
    (``mul_1.onnx``, ``sigmoid.onnx``, ``logreg_iris.onnx``). They are used
    here as an independently produced ONNX file to validate this module's
    field numbers against, so that nothing has to be downloaded.

    Raises
    ------
    FileNotFoundError
        If the installed ``onnxruntime`` does not ship that file.
    """
    import os

    import onnxruntime

    path = os.path.join(os.path.dirname(onnxruntime.__file__), "datasets", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{filename} is not present in the installed onnxruntime at {path}"
        )
    return path
