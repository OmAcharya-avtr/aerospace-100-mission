"""The analytic cost model: the deterministic baseline of this package.

Given a :class:`~edgeinfer.graph.ModelGraph` and a
:class:`~edgeinfer.roofline.DeviceModel`, this module produces an estimate of
inference latency and peak memory **without running anything**. It is
implemented and validated before the learned predictor in
:mod:`edgeinfer.predictor`, and it is the thing that predictor has to beat.

Latency
-------
Per node, the roofline time bound of Williams, Waterman & Patterson 2009 (see
:mod:`edgeinfer.roofline`); the graph estimate is the sum over nodes plus the
device's per-node dispatch overhead and its constant per-inference overhead.
Summing node times assumes sequential execution with no inter-node overlap.

Peak memory
-----------
Classic liveness analysis over the topologically ordered node list: a tensor
is live from the step that produces it to the last step that consumes it, and
peak activation memory is the largest total live size over all steps. Liveness
analysis is the standard compiler formulation --- Aho, Lam, Sethi & Ullman
2006, *Compilers: Principles, Techniques, and Tools*, 2nd ed., §8.4
("Liveness analysis"). The activation/weight split follows the memory
accounting of Sze, Chen, Yang & Emer 2017, *Proceedings of the IEEE* 105(12),
§II-B.

Assumptions, which make this a *lower bound* on real peak memory:

1. A tensor's storage is freed the instant its last consumer finishes. A real
   allocator with a caching arena holds more.
2. No workspace or scratch buffer. ``onnxruntime`` allocates per-kernel
   scratch (im2col buffers for some convolution kernels, for instance) that
   this model does not see.
3. No allocator alignment or page granularity.
4. Weights are resident for the whole session and counted once.

The peak-memory validation in ``validation/validate_memory.py`` checks this
model against a *known allocation pattern* driven through the measurement path
in :mod:`edgeinfer.memtrace`, not against a runtime, precisely because of
assumptions 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgeinfer.graph import ModelGraph
from edgeinfer.ops import OpCost, node_cost
from edgeinfer.roofline import DeviceModel, roofline_time_s

__all__ = [
    "AnalyticEstimate",
    "NodeEstimate",
    "analytic_estimate",
    "node_cost_arrays",
    "peak_activation_bytes",
]


@dataclass(frozen=True)
class NodeEstimate:
    """Analytic cost and roofline time for one node."""

    name: str
    op_type: str
    cost: OpCost
    roofline_s: float
    memory_bound: bool

    @property
    def total_s(self) -> float:
        """Roofline time only; overhead is accounted at graph level [s]."""
        return self.roofline_s


@dataclass(frozen=True)
class AnalyticEstimate:
    """Analytic latency and memory estimate for a whole graph.

    Attributes
    ----------
    graph_name
        Name of the graph estimated.
    device_name, device_source
        Device the estimate is for, and where its peaks came from. Carried so
        no latency figure can be quoted without its device.
    latency_s
        Estimated latency [s]: summed roofline node times plus per-node
        dispatch overhead plus the constant per-inference overhead. A lower
        bound under the assumptions in the module docstring when both overhead
        terms are zero.
    roofline_only_s
        The same sum with the overhead term removed [s]: the pure bound.
    total_flops, total_macs
        Operation counts over the graph [dimensionless].
    total_traffic_bytes
        Compulsory memory traffic over the graph [B].
    peak_memory_bytes
        Resident weights plus peak live activation bytes [B].
    weight_bytes, peak_activation_bytes
        The two components of ``peak_memory_bytes`` [B].
    memory_bound_fraction
        Fraction of nodes whose arithmetic intensity is below the device ridge
        point [dimensionless, 0-1].
    nodes
        Per-node estimates in execution order.
    """

    graph_name: str
    device_name: str
    device_source: str
    latency_s: float
    roofline_only_s: float
    total_flops: int
    total_macs: int
    total_traffic_bytes: int
    peak_memory_bytes: int
    weight_bytes: int
    peak_activation_bytes: int
    memory_bound_fraction: float
    nodes: tuple[NodeEstimate, ...]

    @property
    def arithmetic_intensity(self) -> float:
        """Graph-level flops per byte of compulsory traffic [FLOP/B]."""
        if self.total_traffic_bytes == 0:
            return float("inf")
        return self.total_flops / self.total_traffic_bytes

    def summary_lines(self) -> list[str]:
        """Human-readable lines for a report, every number with its unit."""
        return [
            f"graph                   : {self.graph_name}",
            f"device                  : {self.device_name} ({self.device_source})",
            f"analytic latency        : {self.latency_s * 1e3:.4f} ms "
            f"(roofline bound + dispatch and fixed overheads)",
            f"roofline bound only     : {self.roofline_only_s * 1e3:.4f} ms",
            f"total MACs              : {self.total_macs}",
            f"total FLOPs             : {self.total_flops}",
            f"compulsory traffic      : {self.total_traffic_bytes} B",
            f"arithmetic intensity    : {self.arithmetic_intensity:.4g} FLOP/B",
            f"memory-bound nodes      : {self.memory_bound_fraction * 100:.1f} %",
            f"analytic peak memory    : {self.peak_memory_bytes} B "
            f"({self.weight_bytes} B weights + {self.peak_activation_bytes} B activations)",
        ]


def peak_activation_bytes(graph: ModelGraph) -> tuple[int, dict[str, int]]:
    """Peak live activation bytes by liveness analysis.

    Weights (initialisers) and graph inputs that are also initialisers are
    excluded; graph inputs are counted as live from step 0 until their last
    consumer, because the caller has to hold them.

    Parameters
    ----------
    graph
        Shape-inferred graph.

    Returns
    -------
    (peak_bytes, live_by_step)
        ``peak_bytes`` [B] is the largest live total at any point, taken while
        a node's outputs and its inputs are both allocated. ``live_by_step``
        [B] is the live total *after* each node has run and its dead inputs
        have been released, keyed by node name; it is therefore smaller than
        the peak reached during that node. The per-step map makes the analysis
        auditable rather than a single opaque number.

    Notes
    -----
    Aho et al. 2006 §8.4. A lower bound on real peak memory: see the module
    docstring, assumptions 1-3.
    """
    weight_names = {spec.name for spec in graph.initializers}
    remaining = {k: v for k, v in graph.consumers().items() if k not in weight_names}
    live: dict[str, int] = {}
    for spec in graph.inputs:
        if spec.name not in weight_names:
            live[spec.name] = spec.nbytes

    peak = sum(live.values())
    live_by_step: dict[str, int] = {}
    for node in graph.nodes:
        # Outputs are allocated before the inputs are released: a kernel needs
        # both ends of the copy at once.
        for name in node.outputs:
            if name:
                live[name] = graph.spec(name).nbytes
        peak = max(peak, sum(live.values()))
        for name in node.inputs:
            if not name or name in weight_names:
                continue
            remaining[name] = remaining.get(name, 0) - 1
            if remaining[name] <= 0 and name not in graph.outputs:
                live.pop(name, None)
        live_by_step[node.name] = sum(live.values())
    return int(peak), live_by_step


def analytic_estimate(graph: ModelGraph, device: DeviceModel) -> AnalyticEstimate:
    """Analytic latency and peak-memory estimate for ``graph`` on ``device``.

    The graph is shape-inferred in place if it has not been already.

    Raises
    ------
    edgeinfer.ops.UnsupportedOperatorError
        If any node has no cost model. The estimate is refused rather than
        under-counted.
    """
    if any(name not in graph.tensors for node in graph.nodes for name in node.outputs if name):
        graph.infer_shapes()

    estimates: list[NodeEstimate] = []
    total_flops = 0
    total_macs = 0
    total_bytes = 0
    roofline_sum = 0.0
    memory_bound = 0
    for node in graph.nodes:
        cost = node_cost(node, graph)
        t = roofline_time_s(cost.flops, cost.bytes_total, device)
        is_mem_bound = device.is_memory_bound(cost.arithmetic_intensity)
        memory_bound += int(is_mem_bound)
        estimates.append(
            NodeEstimate(
                name=node.name,
                op_type=node.op_type,
                cost=cost,
                roofline_s=t,
                memory_bound=is_mem_bound,
            )
        )
        total_flops += cost.flops
        total_macs += cost.macs
        total_bytes += cost.bytes_total
        roofline_sum += t

    peak_act, _ = peak_activation_bytes(graph)
    overhead = device.overhead_per_node_s * len(graph.nodes) + device.fixed_overhead_s
    return AnalyticEstimate(
        graph_name=graph.name,
        device_name=device.name,
        device_source=device.source,
        latency_s=roofline_sum + overhead,
        roofline_only_s=roofline_sum,
        total_flops=total_flops,
        total_macs=total_macs,
        total_traffic_bytes=total_bytes,
        peak_memory_bytes=graph.weight_bytes + peak_act,
        weight_bytes=graph.weight_bytes,
        peak_activation_bytes=peak_act,
        memory_bound_fraction=memory_bound / len(graph.nodes),
        nodes=tuple(estimates),
    )


def node_cost_arrays(
    graphs: list[ModelGraph],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of graphs into the arrays :func:`calibrate_device` needs.

    Parameters
    ----------
    graphs
        Shape-inferable graphs, at least one.

    Returns
    -------
    (node_flops, node_bytes, graph_index)
        ``node_flops`` [dimensionless] and ``node_bytes`` [B] are flat
        per-node arrays over every graph in order; ``graph_index`` gives each
        node's graph position in ``graphs``.

    Raises
    ------
    edgeinfer.ops.UnsupportedOperatorError
        If any node has no cost model.
    """
    if not graphs:
        raise ValueError("node_cost_arrays needs at least one graph")
    flops: list[int] = []
    traffic: list[int] = []
    index: list[int] = []
    for g_i, graph in enumerate(graphs):
        graph.infer_shapes()
        for node in graph.nodes:
            cost = node_cost(node, graph)
            flops.append(cost.flops)
            traffic.append(cost.bytes_total)
            index.append(g_i)
    return (
        np.asarray(flops, dtype=float),
        np.asarray(traffic, dtype=float),
        np.asarray(index, dtype=int),
    )
