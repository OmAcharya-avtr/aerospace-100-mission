"""Backend-independent model-graph intermediate representation.

`edgeinfer` characterises a model from its graph, not from its weights. The IR
here is deliberately small: a topologically ordered node list, named tensors
with static shapes and element types, and the attributes the cost model in
:mod:`edgeinfer.ops` needs (kernel shape, strides, pads, dilations, group).

Shape conventions follow the ONNX operator specification (ONNX Standard,
operator set 17, `onnx/docs/Operators.md`): activations are NCHW, convolution
weights are (M, C/group, kH, kW), and `Gemm` operates on 2-D matrices with
optional transposition flags.

Element sizes follow ONNX `TensorProto.DataType`; only float32, float64,
float16, int64, int32 and int8 are sized here because the cost model is only
validated for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, prod

__all__ = [
    "DTYPE_BYTES",
    "GraphError",
    "ModelGraph",
    "Node",
    "TensorSpec",
    "conv_output_spatial",
]

#: Bytes per element, keyed by the dtype names this package accepts.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


class GraphError(ValueError):
    """Raised when a graph is structurally invalid or shapes are inconsistent."""


@dataclass(frozen=True)
class TensorSpec:
    """A named tensor with a fully static shape.

    Parameters
    ----------
    name
        Tensor name, unique within a graph.
    shape
        Dimensions, all strictly positive. Dimensionless.
    dtype
        A key of :data:`DTYPE_BYTES`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.name:
            raise GraphError("tensor name must be non-empty")
        if self.dtype not in DTYPE_BYTES:
            raise GraphError(
                f"tensor {self.name!r}: dtype {self.dtype!r} is not one of "
                f"{sorted(DTYPE_BYTES)}"
            )
        if len(self.shape) == 0:
            raise GraphError(f"tensor {self.name!r}: scalar (rank-0) tensors are not supported")
        for dim in self.shape:
            if not isinstance(dim, int) or dim <= 0:
                raise GraphError(
                    f"tensor {self.name!r}: every dimension must be a positive int, "
                    f"got shape {self.shape}; dynamic axes must be made static first"
                )

    @property
    def elements(self) -> int:
        """Element count [dimensionless]."""
        return int(prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Size in bytes, dense row-major, no padding."""
        return self.elements * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class Node:
    """One operator instance.

    Parameters
    ----------
    name
        Node name, unique within a graph.
    op_type
        ONNX operator type string, e.g. ``"Conv"``, ``"Gemm"``, ``"Relu"``.
    inputs, outputs
        Tensor names. Inputs may name graph inputs, initialisers (weights) or
        the outputs of earlier nodes.
    attrs
        Operator attributes using ONNX attribute names and semantics.
    """

    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.op_type:
            raise GraphError(f"node {self.name!r} has an empty op_type")
        if not self.outputs:
            raise GraphError(f"node {self.name!r} ({self.op_type}) produces no output")


def conv_output_spatial(
    in_dims: tuple[int, ...],
    kernel: tuple[int, ...],
    strides: tuple[int, ...],
    pads: tuple[int, ...],
    dilations: tuple[int, ...],
) -> tuple[int, ...]:
    """Spatial output size of ONNX ``Conv``/``Pool`` with explicit pads.

    Implements the ONNX Standard operator-set-17 formula for ``auto_pad =
    NOTSET`` (`onnx/docs/Operators.md`, ``Conv``)::

        out[i] = floor((in[i] + pad_begin[i] + pad_end[i]
                        - dilation[i] * (kernel[i] - 1) - 1) / stride[i]) + 1

    Parameters
    ----------
    in_dims
        Spatial input dimensions, length ``n``. Dimensionless.
    kernel, strides, dilations
        Length ``n`` each, all strictly positive.
    pads
        Length ``2 * n``, begin pads for every axis then end pads for every
        axis, each non-negative, following the ONNX pads ordering.

    Returns
    -------
    tuple of int
        Spatial output dimensions.

    Raises
    ------
    GraphError
        If lengths are inconsistent or the result would be non-positive (a
        kernel larger than the padded input).
    """
    n = len(in_dims)
    if not (len(kernel) == len(strides) == len(dilations) == n):
        raise GraphError(
            f"kernel/strides/dilations must each have length {n}, got "
            f"{len(kernel)}/{len(strides)}/{len(dilations)}"
        )
    if len(pads) != 2 * n:
        raise GraphError(f"pads must have length {2 * n} for {n} spatial axes, got {len(pads)}")
    out: list[int] = []
    for i in range(n):
        if strides[i] <= 0 or dilations[i] <= 0 or kernel[i] <= 0:
            raise GraphError("kernel, stride and dilation must all be strictly positive")
        padded = in_dims[i] + pads[i] + pads[i + n]
        effective = dilations[i] * (kernel[i] - 1) + 1
        value = floor((padded - effective) / strides[i]) + 1
        if value <= 0:
            raise GraphError(
                f"axis {i}: effective kernel extent {effective} exceeds padded input "
                f"{padded}; output size would be {value}"
            )
        out.append(int(value))
    return tuple(out)


@dataclass
class ModelGraph:
    """A static, topologically ordered inference graph.

    Parameters
    ----------
    name
        Graph name.
    inputs
        Graph input tensors, supplied at inference time.
    outputs
        Graph output tensor names.
    nodes
        Nodes in execution order. Every node input must already be defined by
        an earlier node, a graph input or an initialiser.
    initializers
        Constant tensors resident for the lifetime of the session (weights,
        biases). Their bytes count towards resident memory, not activation
        memory.
    tensors
        Specs for every intermediate tensor, keyed by name. Filled by
        :meth:`infer_shapes` when not supplied.
    """

    name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[str, ...]
    nodes: tuple[Node, ...]
    initializers: tuple[TensorSpec, ...] = ()
    tensors: dict[str, TensorSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.inputs = tuple(self.inputs)
        self.nodes = tuple(self.nodes)
        self.initializers = tuple(self.initializers)
        self.outputs = tuple(self.outputs)
        if not self.nodes:
            raise GraphError(f"graph {self.name!r} has no nodes")
        if not self.inputs:
            raise GraphError(f"graph {self.name!r} has no inputs")
        for spec in (*self.inputs, *self.initializers):
            self.tensors.setdefault(spec.name, spec)
        self._check_topological_order()

    def _check_topological_order(self) -> None:
        defined = {spec.name for spec in self.inputs} | {s.name for s in self.initializers}
        seen_names: set[str] = set()
        for node in self.nodes:
            if node.name in seen_names:
                raise GraphError(f"duplicate node name {node.name!r} in graph {self.name!r}")
            seen_names.add(node.name)
            for tensor_name in node.inputs:
                if tensor_name and tensor_name not in defined:
                    raise GraphError(
                        f"node {node.name!r} ({node.op_type}) consumes {tensor_name!r}, "
                        "which no earlier node, graph input or initialiser defines; "
                        "nodes must be given in topological order"
                    )
            defined.update(o for o in node.outputs if o)
        missing = [o for o in self.outputs if o not in defined]
        if missing:
            raise GraphError(f"graph outputs {missing} are never produced")

    # -- queries ---------------------------------------------------------
    def spec(self, tensor_name: str) -> TensorSpec:
        """Return the :class:`TensorSpec` for ``tensor_name``."""
        try:
            return self.tensors[tensor_name]
        except KeyError:
            raise GraphError(
                f"no shape known for tensor {tensor_name!r}; call infer_shapes() first"
            ) from None

    @property
    def weight_bytes(self) -> int:
        """Total resident initialiser bytes."""
        return sum(spec.nbytes for spec in self.initializers)

    @property
    def op_types(self) -> tuple[str, ...]:
        """Op types in execution order (with repeats)."""
        return tuple(node.op_type for node in self.nodes)

    def consumers(self) -> dict[str, int]:
        """Number of node inputs each tensor feeds, graph outputs counted once."""
        counts: dict[str, int] = dict.fromkeys(self.tensors, 0)
        for node in self.nodes:
            for tensor_name in node.inputs:
                if tensor_name:
                    counts[tensor_name] = counts.get(tensor_name, 0) + 1
        for name in self.outputs:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def infer_shapes(self) -> ModelGraph:
        """Fill :attr:`tensors` for every node output. Returns ``self``.

        Shape rules are implemented in :mod:`edgeinfer.ops` so that cost and
        shape stay in one place per operator.
        """
        from edgeinfer.ops import infer_node_output_specs

        for node in self.nodes:
            for spec in infer_node_output_specs(node, self):
                self.tensors[spec.name] = spec
        return self
