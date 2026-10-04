"""Model-graph features for the learned latency and memory predictors.

The features are deliberately the same quantities the analytic model uses,
plus the composition of the graph. That is the point of the comparison: if the
learned model wins, it won by combining those quantities better than a
roofline does; if the analytic model wins, the learned model had the same
information and did not improve on it.

No feature requires running the model. Everything here is derived from shapes
and operator types, so a prediction is available before a candidate is ever
loaded onto a device --- which is the use case.

Feature list (:data:`FEATURE_NAMES`, in order):

=======================  ==========================================
``log10_flops``          log10 of total operation count + 1
``log10_bytes``          log10 of total compulsory traffic [B] + 1
``log10_macs``           log10 of total multiply-accumulates + 1
``log10_intensity``      log10 of flops per byte + 1
``n_nodes``              node count
``log10_max_node_flops`` log10 of the largest single node's flops + 1
``log10_weight_bytes``   log10 of resident weight bytes + 1
``log10_peak_act_bytes`` log10 of peak live activation bytes + 1
``n_distinct_ops``       number of distinct operator types
``frac_<op>``            fraction of total flops in each of the eight
                         operator groups in :data:`OP_GROUPS`
=======================  ==========================================

``log10(x + 1)`` is used rather than ``log10(x)`` so that a zero-flop graph
(all reshapes) maps to 0 instead of ``-inf``. Logarithms are used because
latency spans orders of magnitude across the population, and a linear feature
would make the fit a function of the largest graph alone.
"""

from __future__ import annotations

import numpy as np

from edgeinfer.analytic import peak_activation_bytes
from edgeinfer.graph import ModelGraph
from edgeinfer.ops import node_cost

__all__ = ["FEATURE_NAMES", "OP_GROUPS", "graph_features", "population_features"]

#: Operator groups whose flop share becomes a feature. Grouped rather than
#: one-hot per operator so the feature count stays fixed when a new operator
#: is added to :data:`edgeinfer.ops.SUPPORTED_OPS`.
OP_GROUPS: dict[str, tuple[str, ...]] = {
    "conv": ("Conv",),
    "matmul": ("Gemm", "MatMul"),
    "pool": ("MaxPool", "AveragePool", "GlobalAveragePool"),
    "act_cheap": ("Relu", "LeakyRelu", "Clip", "Abs", "Neg", "Identity"),
    "act_trans": ("Sigmoid", "Tanh", "Exp", "Erf"),
    "binary": ("Add", "Sub", "Mul", "Div"),
    "norm": ("BatchNormalization", "Softmax"),
    "shape": ("Reshape", "Flatten", "Concat"),
}

FEATURE_NAMES: tuple[str, ...] = (
    "log10_flops",
    "log10_bytes",
    "log10_macs",
    "log10_intensity",
    "n_nodes",
    "log10_max_node_flops",
    "log10_weight_bytes",
    "log10_peak_act_bytes",
    "n_distinct_ops",
    *(f"frac_{group}" for group in OP_GROUPS),
)

_GROUP_OF: dict[str, str] = {
    op: group for group, ops in OP_GROUPS.items() for op in ops
}


def graph_features(graph: ModelGraph) -> np.ndarray:
    """Feature vector for one graph, in :data:`FEATURE_NAMES` order.

    Parameters
    ----------
    graph
        Graph to featurise; shape-inferred in place if needed.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(FEATURE_NAMES),)``, dtype float64.

    Raises
    ------
    edgeinfer.ops.UnsupportedOperatorError
        If any node has no cost model. Featurising an unknown operator as
        zero would hand the predictor a graph it has not seen as if it were
        empty.
    """
    graph.infer_shapes()
    total_flops = 0
    total_macs = 0
    total_bytes = 0
    max_node_flops = 0
    group_flops = dict.fromkeys(OP_GROUPS, 0)
    for node in graph.nodes:
        cost = node_cost(node, graph)
        total_flops += cost.flops
        total_macs += cost.macs
        total_bytes += cost.bytes_total
        max_node_flops = max(max_node_flops, cost.flops)
        group_flops[_GROUP_OF[node.op_type]] += cost.flops

    intensity = total_flops / total_bytes if total_bytes else 0.0
    peak_act, _ = peak_activation_bytes(graph)
    denom = total_flops if total_flops else 1
    return np.asarray(
        [
            np.log10(total_flops + 1),
            np.log10(total_bytes + 1),
            np.log10(total_macs + 1),
            np.log10(intensity + 1),
            float(len(graph.nodes)),
            np.log10(max_node_flops + 1),
            np.log10(graph.weight_bytes + 1),
            np.log10(peak_act + 1),
            float(len(set(graph.op_types))),
            *(group_flops[group] / denom for group in OP_GROUPS),
        ],
        dtype=float,
    )


def population_features(graphs: list[ModelGraph]) -> np.ndarray:
    """Feature matrix for a list of graphs.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(graphs), len(FEATURE_NAMES))``.
    """
    if not graphs:
        raise ValueError("population_features needs at least one graph")
    return np.vstack([graph_features(g) for g in graphs])
