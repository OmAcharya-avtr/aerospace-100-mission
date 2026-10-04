"""Deterministic synthetic model-graph population, and its hand-built members.

The learned predictor in :mod:`edgeinfer.predictor` is trained on a population
of small ONNX graphs generated here. Two things matter about that population
and both are limitations as much as features:

1. **It is synthetic and narrow.** Random MLPs and small 2-D CNNs over a
   bounded shape range. A predictor fitted on it has no claim to generalise to
   a transformer, a quantised graph, or a model an order of magnitude larger.
   That is stated in ``MODEL_CARD.md`` and ``DATASET_CARD.md``.
2. **It is reproducible from a seed.** ``generate_population(seed=...)``
   returns byte-identical models for a given seed, NumPy version and this
   package's version, so a pinned regression output is meaningful. The
   generator uses ``numpy.random.default_rng``, whose PCG64 stream is
   guaranteed stable across NumPy releases by NumPy's own compatibility policy
   for ``Generator`` bit streams (NumPy documentation, "Random sampling ---
   Compatibility policy").

Nothing is downloaded. The two hand-countable networks used in the
operation-count validation (:func:`hand_counted_mlp`,
:func:`hand_counted_cnn`) are fixed, tiny, and chosen so that every operation
count can be written out by hand in a test comment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.onnx_io import OnnxModel, build_model

__all__ = [
    "GraphSpecSummary",
    "generate_population",
    "hand_counted_cnn",
    "hand_counted_mlp",
    "random_cnn",
    "random_mlp",
]


@dataclass(frozen=True)
class GraphSpecSummary:
    """A short description of a generated graph, for a dataset listing."""

    name: str
    family: str
    n_nodes: int
    weight_bytes: int
    detail: str


def _weights(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Standard-normal weights scaled by ``1/sqrt(fan_in)``.

    The scaling follows the Glorot/He convention (He, Zhang, Ren & Sun 2015,
    "Delving Deep into Rectifiers", ICCV 2015, §2.2) so that activations in a
    deeper generated network do not overflow float32. Weight *values* do not
    change the analytic cost and do not change ``onnxruntime`` kernel timing
    for these operators; the scaling exists only to keep the runtime's own
    outputs finite.
    """
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    return (rng.standard_normal(shape) / np.sqrt(max(fan_in, 1))).astype(np.float32)


def random_mlp(
    rng: np.random.Generator,
    name: str,
    *,
    batch: int = 1,
    n_in: int = 32,
    widths: tuple[int, ...] = (64, 32),
    n_out: int = 8,
    activation: str = "Relu",
) -> OnnxModel:
    """A fully connected network as ``Gemm``/activation pairs plus a final ``Gemm``.

    Parameters
    ----------
    rng
        Generator for the weights.
    name
        Graph name.
    batch, n_in, n_out
        Batch size, input features, output features; all >= 1.
    widths
        Hidden layer widths, each >= 1. May be empty for a single ``Gemm``.
    activation
        ``"Relu"``, ``"Sigmoid"`` or ``"Tanh"``.
    """
    if min(batch, n_in, n_out) < 1 or any(w < 1 for w in widths):
        raise ValueError("batch, n_in, n_out and every width must be >= 1")
    if activation not in ("Relu", "Sigmoid", "Tanh"):
        raise ValueError(f"activation must be Relu, Sigmoid or Tanh, got {activation!r}")

    dims = [n_in, *widths, n_out]
    nodes: list[Node] = []
    inits: list[TensorSpec] = []
    arrays: dict[str, np.ndarray] = {}
    current = "X"
    for i in range(len(dims) - 1):
        w_name, b_name = f"W{i}", f"B{i}"
        out = f"H{i}" if i < len(dims) - 2 else "Y"
        arrays[w_name] = _weights(rng, (dims[i], dims[i + 1]))
        arrays[b_name] = _weights(rng, (dims[i + 1],))
        inits += [
            TensorSpec(w_name, (dims[i], dims[i + 1])),
            TensorSpec(b_name, (dims[i + 1],)),
        ]
        gemm_out = out if i == len(dims) - 2 else f"G{i}"
        nodes.append(Node(f"gemm{i}", "Gemm", (current, w_name, b_name), (gemm_out,)))
        if i < len(dims) - 2:
            nodes.append(Node(f"act{i}", activation, (gemm_out,), (out,)))
        current = out
    graph = ModelGraph(
        name=name,
        inputs=(TensorSpec("X", (batch, n_in)),),
        outputs=("Y",),
        nodes=tuple(nodes),
        initializers=tuple(inits),
    )
    return build_model(graph, arrays)


def random_cnn(
    rng: np.random.Generator,
    name: str,
    *,
    batch: int = 1,
    in_channels: int = 1,
    spatial: int = 16,
    channels: tuple[int, ...] = (4, 8),
    kernel: int = 3,
    pool_every: bool = True,
    n_out: int = 4,
) -> OnnxModel:
    """A small 2-D CNN: ``Conv``/``Relu``[/``MaxPool``] blocks then ``Gemm``.

    Parameters
    ----------
    batch, in_channels, spatial
        Input ``(batch, in_channels, spatial, spatial)``; ``spatial`` >= kernel.
    channels
        Output channels of each ``Conv``, at least one entry.
    kernel
        Square kernel edge, odd and >= 1.
    pool_every
        Insert a 2x2 stride-2 ``MaxPool`` after each block.
    n_out
        Final ``Gemm`` output width.
    """
    if not channels:
        raise ValueError("channels must have at least one entry")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"kernel must be odd and >= 1, got {kernel}")
    if spatial < kernel:
        raise ValueError(f"spatial ({spatial}) must be >= kernel ({kernel})")

    nodes: list[Node] = []
    inits: list[TensorSpec] = []
    arrays: dict[str, np.ndarray] = {}
    current = "X"
    c_in, h = in_channels, spatial
    pad = kernel // 2
    for i, c_out in enumerate(channels):
        w_name = f"CW{i}"
        arrays[w_name] = _weights(rng, (c_out, c_in, kernel, kernel))
        inits.append(TensorSpec(w_name, (c_out, c_in, kernel, kernel)))
        nodes.append(
            Node(
                f"conv{i}",
                "Conv",
                (current, w_name),
                (f"C{i}",),
                {
                    "kernel_shape": [kernel, kernel],
                    "strides": [1, 1],
                    "pads": [pad, pad, pad, pad],
                    "group": 1,
                },
            )
        )
        nodes.append(Node(f"crelu{i}", "Relu", (f"C{i}",), (f"R{i}",)))
        current = f"R{i}"
        if pool_every and h >= 2:
            nodes.append(
                Node(
                    f"pool{i}",
                    "MaxPool",
                    (current,),
                    (f"P{i}",),
                    {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]},
                )
            )
            current = f"P{i}"
            h //= 2
        c_in = c_out

    flat = c_in * h * h
    # ONNX Reshape takes its target shape as a second *input* from opset 5 on
    # (ONNX Standard, Reshape-5 and later), not as an attribute. The IR node
    # carries the shape as an attribute as well, because the analytic shape
    # rule in edgeinfer.ops reads shapes from attributes, not from constant
    # tensor values.
    arrays["RS"] = np.asarray([batch, flat], dtype=np.int64)
    inits.append(TensorSpec("RS", (2,), "int64"))
    nodes.append(
        Node("flat", "Reshape", (current, "RS"), ("F",), {"shape": [batch, flat]})
    )
    arrays["FW"] = _weights(rng, (flat, n_out))
    arrays["FB"] = _weights(rng, (n_out,))
    inits += [TensorSpec("FW", (flat, n_out)), TensorSpec("FB", (n_out,))]
    nodes.append(Node("fc", "Gemm", ("F", "FW", "FB"), ("Y",)))
    graph = ModelGraph(
        name=name,
        inputs=(TensorSpec("X", (batch, in_channels, spatial, spatial)),),
        outputs=("Y",),
        nodes=tuple(nodes),
        initializers=tuple(inits),
    )
    return build_model(graph, arrays)


def hand_counted_mlp() -> OnnxModel:
    """A fixed two-layer MLP whose cost is counted by hand in the tests.

    Shape: ``X (1, 4) -> Gemm(4, 6) + bias -> Relu -> Gemm(6, 3) + bias -> Y``.
    All tensors float32 (4 B per element). Hand counts appear in
    ``tests/test_ops.py::TestHandCountedMlp``.
    """
    rng = np.random.default_rng(20260101)
    return random_mlp(rng, "hand_mlp", batch=1, n_in=4, widths=(6,), n_out=3)


def hand_counted_cnn() -> OnnxModel:
    """A fixed one-block CNN whose cost is counted by hand in the tests.

    Shape: ``X (1, 1, 6, 6) -> Conv(1->2, 3x3, pad 1) -> Relu ->
    MaxPool(2x2, stride 2) -> Reshape(1, 18) -> Gemm(18, 2) + bias -> Y``.
    Hand counts appear in ``tests/test_ops.py::TestHandCountedCnn``.
    """
    rng = np.random.default_rng(20260102)
    return random_cnn(
        rng,
        "hand_cnn",
        batch=1,
        in_channels=1,
        spatial=6,
        channels=(2,),
        kernel=3,
        pool_every=True,
        n_out=2,
    )


def generate_population(
    n_models: int = 120, seed: int = 20260401
) -> tuple[list[OnnxModel], list[GraphSpecSummary]]:
    """Generate a reproducible population of small graphs.

    Roughly half MLPs and half CNNs, with widths, depths and spatial sizes
    drawn from the ranges below. The ranges are chosen so that every model
    runs in well under a millisecond on one CPU core, because the whole
    population has to be measured inside this repository's three-minute
    per-script compute budget.

    ===========  =======================================
    Family       Range
    ===========  =======================================
    MLP          1-3 hidden layers, width 8-512, inputs
                 8-512, outputs 2-64, Relu/Sigmoid/Tanh
    CNN          1-3 conv blocks, 2-48 channels, spatial
                 8-48, kernel 1/3/5
    ===========  =======================================

    Parameters
    ----------
    n_models
        Population size, >= 2.
    seed
        Seed for every random choice, including the family split.

    Returns
    -------
    (models, summaries)
    """
    if n_models < 2:
        raise ValueError(f"n_models must be >= 2, got {n_models}")
    rng = np.random.default_rng(seed)
    models: list[OnnxModel] = []
    summaries: list[GraphSpecSummary] = []
    for i in range(n_models):
        if i % 2 == 0:
            depth = int(rng.integers(1, 4))
            widths = tuple(int(rng.integers(8, 513)) for _ in range(depth))
            n_in = int(rng.integers(8, 513))
            n_out = int(rng.integers(2, 65))
            activation = str(rng.choice(["Relu", "Sigmoid", "Tanh"]))
            model = random_mlp(
                rng,
                f"mlp_{i:03d}",
                n_in=n_in,
                widths=widths,
                n_out=n_out,
                activation=activation,
            )
            detail = f"in={n_in} widths={widths} out={n_out} act={activation}"
            family = "mlp"
        else:
            blocks = int(rng.integers(1, 4))
            channels = tuple(int(rng.integers(2, 49)) for _ in range(blocks))
            spatial = int(rng.integers(8, 49))
            kernel = int(rng.choice([1, 3, 5]))
            model = random_cnn(
                rng,
                f"cnn_{i:03d}",
                in_channels=1,
                spatial=spatial,
                channels=channels,
                kernel=kernel,
                n_out=int(rng.integers(2, 33)),
            )
            detail = f"spatial={spatial} channels={channels} kernel={kernel}"
            family = "cnn"
        models.append(model)
        summaries.append(
            GraphSpecSummary(
                name=model.graph.name,
                family=family,
                n_nodes=len(model.graph.nodes),
                weight_bytes=model.graph.weight_bytes,
                detail=detail,
            )
        )
    return models, summaries
