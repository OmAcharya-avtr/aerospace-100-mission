"""Per-operator analytic cost: operation counts, memory traffic and shapes.

This module is the deterministic core of the analytic baseline. For each
supported operator it returns:

* ``macs`` --- multiply-accumulate operations [dimensionless]
* ``flops`` --- floating-point operations [dimensionless], counting one
  multiply-accumulate as two flops
* ``bytes_read`` --- bytes that must be read at least once [B]
* ``bytes_written`` --- bytes written [B]

Counting conventions and sources
--------------------------------
* **Dense matrix product.** A product of an ``(M, K)`` and a ``(K, N)`` matrix
  costs ``M*K*N`` multiply-accumulates, i.e. ``2*M*K*N`` flops. Golub & Van
  Loan 2013, *Matrix Computations*, 4th ed., §1.1.11 ("Matrix-Matrix
  Multiplication"), which gives ``2mnk`` flops. Valid for dense operands; a
  Strassen-type algorithm would be asymptotically cheaper and is not modelled.
* **Convolution.** ``Conv`` with input ``(N, C, *S_in)``, weight
  ``(M, C/group, *K)`` and output ``(N, M, *S_out)`` costs
  ``N * M * prod(S_out) * (C/group) * prod(K)`` multiply-accumulates. Sze,
  Chen, Yang & Emer 2017, "Efficient Processing of Deep Neural Networks: A
  Tutorial and Survey", *Proceedings of the IEEE* 105(12), 2295-2329, §II-A,
  which counts exactly this product for a convolutional layer. Valid for
  direct (non-Winograd, non-FFT) convolution; a Winograd implementation
  reduces multiplies and the model will then overestimate.
* **Metadata operators.** ``Reshape`` and ``Flatten`` are charged zero flops;
  only their traffic is counted. ``Squeeze``, ``Unsqueeze`` and ``Transpose``
  are not supported, because their output-shape rules are not implemented and
  a wrong output shape would corrupt every downstream count.
* **Elementwise activations and binary ops.** One operation per output
  element. ``Relu`` is one comparison, which is counted as one flop so that
  activation cost is not silently zero; ``Sigmoid`` and ``Tanh`` are counted
  at :data:`TRANSCENDENTAL_FLOPS_PER_ELEMENT` flops per element, an explicit
  modelling assumption rather than a measured figure (see the limitation note
  in ``README.md``).
* **Pooling.** ``MaxPool`` and ``AveragePool`` perform
  ``prod(kernel)`` comparisons or additions per output element; counted as
  flops, zero MACs.
* **Softmax.** Counted at :data:`SOFTMAX_FLOPS_PER_ELEMENT` flops per element
  covering max, subtract, exp, sum and divide. An assumption, stated as such.
* **BatchNormalization** (inference). Two flops per element: the four
  parameters fold into one scale and one shift, Ioffe & Szegedy 2015,
  "Batch Normalization", *ICML 2015*, §3.1 (inference-time folding).

Memory traffic
--------------
``bytes_read`` counts every input tensor once and ``bytes_written`` counts
every output tensor once. This is the *compulsory* traffic in the sense used
by the roofline model: the minimum DRAM traffic if every byte is fetched
exactly once and caches are perfect. Real traffic is higher whenever a tile
is re-fetched, so the analytic latency from this traffic is a lower bound on
the memory-bound side. Williams, Waterman & Patterson 2009 make the same
simplification and call it out (see :mod:`edgeinfer.roofline`).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from edgeinfer.graph import GraphError, ModelGraph, Node, TensorSpec, conv_output_spatial

__all__ = [
    "SOFTMAX_FLOPS_PER_ELEMENT",
    "SUPPORTED_OPS",
    "TRANSCENDENTAL_FLOPS_PER_ELEMENT",
    "OpCost",
    "UnsupportedOperatorError",
    "infer_node_output_specs",
    "node_cost",
]

#: Flops charged per element for ``Sigmoid``/``Tanh``. A stated assumption:
#: an exp or tanh evaluation is counted as this many flops.
TRANSCENDENTAL_FLOPS_PER_ELEMENT: int = 4

#: Flops charged per element for ``Softmax`` (max, subtract, exp, sum, divide).
SOFTMAX_FLOPS_PER_ELEMENT: int = 5

_ELEMENTWISE_UNARY = ("Relu", "LeakyRelu", "Clip", "Abs", "Neg", "Identity")
_TRANSCENDENTAL_UNARY = ("Sigmoid", "Tanh", "Exp", "Erf")
_ELEMENTWISE_BINARY = ("Add", "Sub", "Mul", "Div")
#: Shape-changing metadata operators. Their output shape comes from the
#: ``shape`` attribute of edgeinfer's IR, which :func:`edgeinfer.onnx_io.parse_model`
#: recovers from the constant target-shape initialiser. ``Squeeze``,
#: ``Unsqueeze`` and ``Transpose`` are deliberately NOT supported: their shape
#: rules are not implemented, and listing them would mean reporting a wrong
#: output shape rather than refusing.
_RESHAPE_LIKE = ("Reshape", "Flatten")

#: Operator types this package can cost and shape-infer.
SUPPORTED_OPS: frozenset[str] = frozenset(
    (
        "Conv",
        "Gemm",
        "MatMul",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Softmax",
        "BatchNormalization",
        "Concat",
        *_ELEMENTWISE_UNARY,
        *_TRANSCENDENTAL_UNARY,
        *_ELEMENTWISE_BINARY,
        *_RESHAPE_LIKE,
    )
)


class UnsupportedOperatorError(GraphError):
    """Raised when a graph contains an operator with no cost model.

    The analytic estimate is refused rather than silently charged zero: an
    unknown operator with zero cost turns a budget overrun into a budget pass,
    which is the failure mode this package exists to prevent.
    """

    def __init__(self, op_type: str, node_name: str = "") -> None:
        where = f" (node {node_name!r})" if node_name else ""
        super().__init__(
            f"operator {op_type!r}{where} has no cost model in edgeinfer. "
            f"Supported operators: {', '.join(sorted(SUPPORTED_OPS))}. "
            "Add a cost rule in edgeinfer.ops before characterising this graph; "
            "edgeinfer will not charge an unknown operator zero cost."
        )
        self.op_type = op_type
        self.node_name = node_name


@dataclass(frozen=True)
class OpCost:
    """Analytic cost of one node.

    Attributes
    ----------
    macs
        Multiply-accumulate count [dimensionless].
    flops
        Floating-point operation count [dimensionless].
    bytes_read, bytes_written
        Compulsory memory traffic [B].
    """

    macs: int
    flops: int
    bytes_read: int
    bytes_written: int

    @property
    def bytes_total(self) -> int:
        """Total compulsory traffic [B]."""
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        """Flops per byte of compulsory traffic [FLOP/B]; ``inf`` if no traffic."""
        if self.bytes_total == 0:
            return float("inf")
        return self.flops / self.bytes_total


def _ints(attrs: dict[str, object], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = attrs.get(key)
    if value is None:
        return default
    return tuple(int(v) for v in value)  # type: ignore[union-attr]


def _attr_int(attrs: dict[str, object], key: str, default: int) -> int:
    value = attrs.get(key)
    return default if value is None else int(value)  # type: ignore[arg-type]


# -- shape inference ----------------------------------------------------


def infer_node_output_specs(node: Node, graph: ModelGraph) -> list[TensorSpec]:
    """Return specs for ``node``'s outputs, given shapes already known.

    Raises
    ------
    UnsupportedOperatorError
        If ``node.op_type`` has no rule.
    GraphError
        If declared attributes and input shapes are inconsistent.
    """
    op = node.op_type
    if op not in SUPPORTED_OPS:
        raise UnsupportedOperatorError(op, node.name)
    x = graph.spec(node.inputs[0])

    if op == "Conv":
        w = graph.spec(node.inputs[1])
        if len(w.shape) != len(x.shape):
            raise GraphError(
                f"node {node.name!r}: Conv weight rank {len(w.shape)} does not match "
                f"input rank {len(x.shape)}"
            )
        spatial = len(x.shape) - 2
        kernel = _ints(node.attrs, "kernel_shape", w.shape[2:])
        strides = _ints(node.attrs, "strides", (1,) * spatial)
        dilations = _ints(node.attrs, "dilations", (1,) * spatial)
        pads = _ints(node.attrs, "pads", (0,) * (2 * spatial))
        group = _attr_int(node.attrs, "group", 1)
        if x.shape[1] % group or w.shape[0] % group:
            raise GraphError(
                f"node {node.name!r}: group={group} does not divide input channels "
                f"{x.shape[1]} and output channels {w.shape[0]}"
            )
        if w.shape[1] != x.shape[1] // group:
            raise GraphError(
                f"node {node.name!r}: Conv weight expects {w.shape[1]} input channels per "
                f"group but input has {x.shape[1]} channels with group={group}"
            )
        out_spatial = conv_output_spatial(x.shape[2:], kernel, strides, pads, dilations)
        return [TensorSpec(node.outputs[0], (x.shape[0], w.shape[0], *out_spatial), x.dtype)]

    if op in ("MaxPool", "AveragePool"):
        spatial = len(x.shape) - 2
        kernel = _ints(node.attrs, "kernel_shape", ())
        if len(kernel) != spatial:
            raise GraphError(
                f"node {node.name!r}: {op} requires kernel_shape of length {spatial}"
            )
        strides = _ints(node.attrs, "strides", (1,) * spatial)
        dilations = _ints(node.attrs, "dilations", (1,) * spatial)
        pads = _ints(node.attrs, "pads", (0,) * (2 * spatial))
        out_spatial = conv_output_spatial(x.shape[2:], kernel, strides, pads, dilations)
        return [TensorSpec(node.outputs[0], (*x.shape[:2], *out_spatial), x.dtype)]

    if op == "GlobalAveragePool":
        return [
            TensorSpec(node.outputs[0], (*x.shape[:2], *((1,) * (len(x.shape) - 2))), x.dtype)
        ]

    if op == "Gemm":
        w = graph.spec(node.inputs[1])
        trans_a = _attr_int(node.attrs, "transA", 0)
        trans_b = _attr_int(node.attrs, "transB", 0)
        m, ka = (x.shape[1], x.shape[0]) if trans_a else (x.shape[0], x.shape[1])
        kb, n = (w.shape[1], w.shape[0]) if trans_b else (w.shape[0], w.shape[1])
        if ka != kb:
            raise GraphError(
                f"node {node.name!r}: Gemm inner dimensions disagree, {ka} vs {kb} "
                f"(transA={trans_a}, transB={trans_b})"
            )
        return [TensorSpec(node.outputs[0], (m, n), x.dtype)]

    if op == "MatMul":
        w = graph.spec(node.inputs[1])
        if x.shape[-1] != w.shape[-2]:
            raise GraphError(
                f"node {node.name!r}: MatMul inner dimensions disagree, "
                f"{x.shape[-1]} vs {w.shape[-2]}"
            )
        return [TensorSpec(node.outputs[0], (*x.shape[:-1], w.shape[-1]), x.dtype)]

    if op == "Concat":
        axis = _attr_int(node.attrs, "axis", 0) % len(x.shape)
        total = 0
        for name in node.inputs:
            spec = graph.spec(name)
            if len(spec.shape) != len(x.shape):
                raise GraphError(f"node {node.name!r}: Concat inputs have differing ranks")
            total += spec.shape[axis]
        shape = list(x.shape)
        shape[axis] = total
        return [TensorSpec(node.outputs[0], tuple(shape), x.dtype)]

    if op in _RESHAPE_LIKE:
        shape = node.attrs.get("shape")
        if shape is None:
            raise GraphError(
                f"node {node.name!r}: {op} needs a static target shape. In ONNX the "
                "target shape is a second input tensor; edgeinfer's IR carries it as "
                "a 'shape' attribute, which parse_model recovers from the constant "
                "initialiser. A Reshape whose target shape is computed at run time "
                "cannot be characterised statically and is refused rather than "
                "guessed."
            )
        new_shape = tuple(int(v) for v in shape)  # type: ignore[union-attr]
        if prod(new_shape) != x.elements:
            raise GraphError(
                f"node {node.name!r}: {op} to {new_shape} changes element count "
                f"from {x.elements} to {prod(new_shape)}"
            )
        return [TensorSpec(node.outputs[0], new_shape, x.dtype)]

    if op in _ELEMENTWISE_BINARY:
        other = graph.spec(node.inputs[1])
        shape = _broadcast(x.shape, other.shape, node.name, op)
        return [TensorSpec(node.outputs[0], shape, x.dtype)]

    # Remaining supported ops are shape-preserving: elementwise unary,
    # transcendental unary, Softmax, BatchNormalization.
    return [TensorSpec(node.outputs[0], x.shape, x.dtype)]


def _broadcast(
    a: tuple[int, ...], b: tuple[int, ...], node_name: str, op: str
) -> tuple[int, ...]:
    """NumPy/ONNX multidirectional broadcasting of two shapes."""
    rank = max(len(a), len(b))
    pa = (1,) * (rank - len(a)) + a
    pb = (1,) * (rank - len(b)) + b
    out: list[int] = []
    for da, db in zip(pa, pb, strict=True):
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise GraphError(
                f"node {node_name!r}: {op} cannot broadcast shapes {a} and {b}"
            )
    return tuple(out)


# -- cost ---------------------------------------------------------------


def node_cost(node: Node, graph: ModelGraph) -> OpCost:
    """Analytic cost of one node. See the module docstring for every formula.

    Raises
    ------
    UnsupportedOperatorError
        If ``node.op_type`` has no cost rule.
    """
    op = node.op_type
    if op not in SUPPORTED_OPS:
        raise UnsupportedOperatorError(op, node.name)

    in_specs = [graph.spec(name) for name in node.inputs if name]
    out_specs = [graph.spec(name) for name in node.outputs if name]
    bytes_read = sum(spec.nbytes for spec in in_specs)
    bytes_written = sum(spec.nbytes for spec in out_specs)
    out_elements = sum(spec.elements for spec in out_specs)

    macs = 0
    flops = 0

    if op == "Conv":
        x, w = in_specs[0], in_specs[1]
        y = out_specs[0]
        group = _attr_int(node.attrs, "group", 1)
        kernel = _ints(node.attrs, "kernel_shape", w.shape[2:])
        # N * M * prod(S_out) * (C/group) * prod(K)  -- Sze et al. 2017 §II-A
        macs = y.shape[0] * y.shape[1] * prod(y.shape[2:]) * (x.shape[1] // group) * prod(kernel)
        flops = 2 * macs
        if len(node.inputs) > 2 and node.inputs[2]:
            flops += y.elements  # bias add, one flop per output element

    elif op == "Gemm":
        x, w = in_specs[0], in_specs[1]
        y = out_specs[0]
        k = x.shape[0] if _attr_int(node.attrs, "transA", 0) else x.shape[1]
        macs = y.shape[0] * k * y.shape[1]  # 2*M*K*N flops, Golub & Van Loan §1.1.11
        flops = 2 * macs
        if len(node.inputs) > 2 and node.inputs[2]:
            flops += y.elements

    elif op == "MatMul":
        x = in_specs[0]
        y = out_specs[0]
        macs = prod(y.shape) * x.shape[-1]
        flops = 2 * macs

    elif op in ("MaxPool", "AveragePool"):
        kernel = _ints(node.attrs, "kernel_shape", ())
        flops = out_elements * prod(kernel)

    elif op == "GlobalAveragePool":
        flops = in_specs[0].elements + out_elements  # one add per input, one divide per output

    elif op == "Softmax":
        flops = SOFTMAX_FLOPS_PER_ELEMENT * out_elements

    elif op == "BatchNormalization":
        flops = 2 * out_elements  # folded scale and shift, Ioffe & Szegedy 2015 §3.1

    elif op in _TRANSCENDENTAL_UNARY:
        flops = TRANSCENDENTAL_FLOPS_PER_ELEMENT * out_elements

    elif op in _ELEMENTWISE_UNARY or op in _ELEMENTWISE_BINARY:
        flops = out_elements

    elif op in _RESHAPE_LIKE or op == "Concat":
        flops = 0  # metadata or copy only; the traffic term carries the cost

    return OpCost(
        macs=int(macs),
        flops=int(flops),
        bytes_read=int(bytes_read),
        bytes_written=int(bytes_written),
    )
