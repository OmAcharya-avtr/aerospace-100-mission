"""Roofline performance bound and the device model it needs.

Source
------
Williams, S., Waterman, A. & Patterson, D. (2009), "Roofline: An Insightful
Visual Performance Model for Multicore Architectures", *Communications of the
ACM* 52(4), 65-76. The model bounds attainable performance by the smaller of
compute peak and the product of memory bandwidth and arithmetic intensity:

.. code-block:: text

    attainable [FLOP/s] = min( peak_flops [FLOP/s],
                               peak_bandwidth [B/s] * I [FLOP/B] )

where ``I`` is arithmetic intensity, flops performed per byte of DRAM traffic.
The *ridge point* is ``peak_flops / peak_bandwidth`` [FLOP/B]; a kernel with
intensity below it is memory bound, above it compute bound.

Converting the bound to a time, for a kernel of ``F`` flops moving ``B`` bytes:

.. code-block:: text

    t >= max( F / peak_flops, B / peak_bandwidth )   [s]

which is the same statement as the ``min`` on rate. This package uses the
time form because latency, not throughput, is what a control-loop budget is
written against.

Assumptions and validity
------------------------
1. **It is a bound, not a prediction.** Williams et al. 2009 §3 present the
   roofline as an upper bound on attainable performance; real kernels sit
   below the roof. The latency it gives is therefore a *lower* bound, and the
   measured latency should exceed it. A measured latency *below* the roofline
   bound means the device model's peaks are wrong, and
   :func:`roofline_time_s` callers should treat that as a calibration fault.
2. **No overlap between nodes.** Node times are summed, so a runtime that
   overlaps one node's memory traffic with the next node's compute will beat
   the sum.
3. **Compulsory traffic only.** Traffic comes from :mod:`edgeinfer.ops`,
   which counts each tensor once. Cache-miss re-fetches raise real traffic and
   lower real performance.
4. **Fixed overheads.** A real runtime pays a dispatch cost per kernel and a
   constant cost per inference call (argument marshalling, output allocation,
   binding). The roofline models neither. :class:`DeviceModel` carries
   explicit ``overhead_per_node_s`` and ``fixed_overhead_s`` terms for them;
   both are zero unless calibrated or declared, and when either is non-zero
   the result is no longer a pure roofline bound. This is stated wherever such
   a number is reported. For the small graphs this repository measures, the
   constant term is the *dominant* one, which is a finding reported in
   ``README.md`` rather than a detail.
5. **Peaks are declared, not measured.** ``peak_flops`` and ``peak_bandwidth``
   are properties of the device the caller declares. Nothing in this module
   derives them; :func:`calibrate_device` fits them from measurements and
   labels the result as a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["DeviceModel", "calibrate_device", "roofline_time_s"]


@dataclass(frozen=True)
class DeviceModel:
    """Declared peak compute and peak memory bandwidth of one target.

    Parameters
    ----------
    name
        Human-readable target name, carried into every report so that a number
        can never be read without its device.
    peak_flops
        Peak floating-point rate [FLOP/s], strictly positive.
    peak_bandwidth_bytes_s
        Peak main-memory bandwidth [B/s], strictly positive.
    overhead_per_node_s
        Fixed per-node dispatch cost [s], non-negative. Zero keeps the
        estimate a pure roofline bound.
    fixed_overhead_s
        Constant per-inference cost [s], non-negative: the part of a call that
        does not scale with the graph at all.
    source
        Where the peaks came from: ``"declared"``, ``"calibrated-fit"``, or a
        citation. Reports print this verbatim.
    """

    name: str
    peak_flops: float
    peak_bandwidth_bytes_s: float
    overhead_per_node_s: float = 0.0
    fixed_overhead_s: float = 0.0
    source: str = "declared"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DeviceModel.name must be non-empty")
        if not np.isfinite(self.peak_flops) or self.peak_flops <= 0:
            raise ValueError(f"peak_flops must be finite and > 0, got {self.peak_flops}")
        if not np.isfinite(self.peak_bandwidth_bytes_s) or self.peak_bandwidth_bytes_s <= 0:
            raise ValueError(
                f"peak_bandwidth_bytes_s must be finite and > 0, "
                f"got {self.peak_bandwidth_bytes_s}"
            )
        if not np.isfinite(self.overhead_per_node_s) or self.overhead_per_node_s < 0:
            raise ValueError(
                f"overhead_per_node_s must be finite and >= 0, got {self.overhead_per_node_s}"
            )
        if not np.isfinite(self.fixed_overhead_s) or self.fixed_overhead_s < 0:
            raise ValueError(
                f"fixed_overhead_s must be finite and >= 0, got {self.fixed_overhead_s}"
            )

    @property
    def ridge_point_flops_per_byte(self) -> float:
        """Arithmetic intensity at which the roof changes slope [FLOP/B].

        Williams et al. 2009 §3: ``peak_flops / peak_bandwidth``.
        """
        return self.peak_flops / self.peak_bandwidth_bytes_s

    def attainable_flops(self, intensity_flops_per_byte: float) -> float:
        """Roofline attainable rate at a given arithmetic intensity [FLOP/s]."""
        if intensity_flops_per_byte < 0:
            raise ValueError("arithmetic intensity cannot be negative")
        return float(
            min(self.peak_flops, self.peak_bandwidth_bytes_s * intensity_flops_per_byte)
        )

    def is_memory_bound(self, intensity_flops_per_byte: float) -> bool:
        """True when intensity is below the ridge point."""
        return intensity_flops_per_byte < self.ridge_point_flops_per_byte

    def scaled(self, factor: float) -> DeviceModel:
        """Return a copy with both peaks multiplied by ``factor``.

        Used to express a throttled device (``factor < 1``). The returned
        model's ``source`` records the scaling so a throttled number can never
        be mistaken for a nominal one.
        """
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"scale factor must be finite and > 0, got {factor}")
        return replace(
            self,
            peak_flops=self.peak_flops * factor,
            peak_bandwidth_bytes_s=self.peak_bandwidth_bytes_s * factor,
            source=f"{self.source}; peaks scaled by {factor:.4g}",
        )


def roofline_time_s(flops: float, traffic_bytes: float, device: DeviceModel) -> float:
    """Roofline lower bound on the time for one kernel [s].

    ``t = max(flops / peak_flops, traffic_bytes / peak_bandwidth)``, the time
    form of Williams, Waterman & Patterson 2009 §3.

    Parameters
    ----------
    flops
        Floating-point operations [dimensionless], non-negative.
    traffic_bytes
        Memory traffic [B], non-negative.
    device
        Target device.

    Returns
    -------
    float
        Time [s], not including ``device.overhead_per_node_s``.
    """
    if flops < 0 or traffic_bytes < 0:
        raise ValueError(
            f"flops and traffic_bytes must be non-negative, got {flops}, {traffic_bytes}"
        )
    return max(flops / device.peak_flops, traffic_bytes / device.peak_bandwidth_bytes_s)


def calibrate_device(
    node_flops: np.ndarray,
    node_bytes: np.ndarray,
    graph_index: np.ndarray,
    measured_s: np.ndarray,
    name: str,
    grid: int = 20,
) -> DeviceModel:
    """Fit a device model's four parameters to measured graph latencies.

    The fitted parameters are ``peak_flops``, ``peak_bandwidth_bytes_s``,
    ``overhead_per_node_s`` and ``fixed_overhead_s``. This is a
    **four-parameter fit**, not a device specification. It exists so that the
    analytic baseline competes with the learned predictor on equal footing:
    both see the same training set, the baseline with four free parameters.
    The returned model's ``source`` says ``calibrated-fit`` so that no report
    can present it as a datasheet figure.

    Method
    ------
    The predicted latency of graph ``g`` is exactly what
    :func:`edgeinfer.analytic.analytic_estimate` computes:

    .. code-block:: text

        t_pred(g) = sum_{i in g} max(F_i / peak_flops, B_i / peak_bw)
                    + overhead_per_node * n_nodes(g)
                    + fixed_overhead

    The first term contains a ``max`` and is not differentiable at the ridge,
    so the two peaks are searched on a ``grid x grid`` log-spaced grid over
    four decades around a least-squares seed. For each grid point the two
    overhead terms enter linearly and are solved by non-negative least squares
    (Lawson & Hanson 1974, *Solving Least Squares Problems*, ch. 23, as
    implemented by :func:`scipy.optimize.nnls`) on rows weighted by
    ``1 / t_measured``. That weighting makes the linear solve approximate a
    relative-error fit, which is the right objective when latencies span
    orders of magnitude; an unweighted fit would be decided almost entirely by
    the largest graph. The grid point with the lowest mean squared error of
    ``log(t)`` wins.

    Non-negativity is imposed because a negative overhead has no physical
    reading and would let the fit cancel an over-large peak against it.

    Parameters
    ----------
    node_flops, node_bytes
        Flat per-node arrays over every calibration graph: operation count
        [dimensionless] and compulsory traffic [B]. Same length.
    graph_index
        For each entry of ``node_flops``, the index of the graph it belongs
        to. Values must cover ``0 .. n_graphs - 1`` with no gaps.
    measured_s
        Measured latency per graph [s], length ``n_graphs``, strictly
        positive. Which statistic (p50, p99, mean) is the caller's choice and
        must be recorded in the caller's report.
    name
        Device name for the returned model.
    grid
        Points per axis in the peak search, >= 4.

    Returns
    -------
    DeviceModel
        With ``source="calibrated-fit"``.
    """
    from scipy.optimize import nnls

    node_flops = np.asarray(node_flops, dtype=float)
    node_bytes = np.asarray(node_bytes, dtype=float)
    graph_index = np.asarray(graph_index, dtype=int)
    measured_s = np.asarray(measured_s, dtype=float)
    if not (len(node_flops) == len(node_bytes) == len(graph_index)):
        raise ValueError("node_flops, node_bytes and graph_index must have the same length")
    if len(node_flops) == 0:
        raise ValueError("calibration needs at least one node")
    n_graphs = len(measured_s)
    if n_graphs < 4:
        raise ValueError(f"need at least 4 calibration graphs for 4 parameters, got {n_graphs}")
    if graph_index.min() < 0 or graph_index.max() != n_graphs - 1:
        raise ValueError(
            f"graph_index must cover 0..{n_graphs - 1} with no gaps, got range "
            f"{graph_index.min()}..{graph_index.max()}"
        )
    if np.any(measured_s <= 0):
        raise ValueError("measured_s must be strictly positive")
    if np.any(node_flops < 0) or np.any(node_bytes < 0):
        raise ValueError("node_flops and node_bytes must be non-negative")
    if grid < 4:
        raise ValueError(f"grid must be >= 4, got {grid}")

    node_counts = np.bincount(graph_index, minlength=n_graphs).astype(float)
    graph_flops = np.bincount(graph_index, weights=node_flops, minlength=n_graphs)
    graph_bytes = np.bincount(graph_index, weights=node_bytes, minlength=n_graphs)

    # Seed: the rate that would explain the median graph if it were purely
    # compute bound, and likewise for bandwidth.
    seed_flops = max(float(np.median(graph_flops / measured_s)), 1.0)
    seed_bw = max(float(np.median(graph_bytes / measured_s)), 1.0)
    log_target = np.log(measured_s)
    weight = 1.0 / measured_s
    design = np.column_stack([node_counts, np.ones_like(node_counts)]) * weight[:, None]

    best: tuple[float, float, float, float, float] | None = None
    for pf in np.logspace(np.log10(seed_flops) - 2, np.log10(seed_flops) + 2, grid):
        compute_t = node_flops / pf
        for bw in np.logspace(np.log10(seed_bw) - 2, np.log10(seed_bw) + 2, grid):
            per_node_t = np.maximum(compute_t, node_bytes / bw)
            base = np.bincount(graph_index, weights=per_node_t, minlength=n_graphs)
            coeffs, _ = nnls(design, (measured_s - base) * weight)
            per_node, fixed = float(coeffs[0]), float(coeffs[1])
            pred = np.maximum(base + per_node * node_counts + fixed, 1e-15)
            err = float(np.mean((np.log(pred) - log_target) ** 2))
            if best is None or err < best[0]:
                best = (err, float(pf), float(bw), per_node, fixed)

    assert best is not None
    _, pf, bw, per_node, fixed = best
    return DeviceModel(
        name=name,
        peak_flops=pf,
        peak_bandwidth_bytes_s=bw,
        overhead_per_node_s=per_node,
        fixed_overhead_s=fixed,
        source="calibrated-fit",
    )
