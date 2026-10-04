"""Inference backends behind one contract.

Three backends implement :class:`InferenceBackend`:

:class:`SimulatedBackend`
    Consumes a sampled per-stage duration from an **injected synthetic cost
    model**. The validation in ``validation/validate_backend_cost_model.py``
    uses it to check that the measurement path in :mod:`edgeinfer.harness`
    recovers a cost model it was given, which is the only way to test a
    measurement instrument without a reference clock. Nothing in the
    simulated backend may be reported as a device measurement, and every
    profile it produces carries ``backend="simulated"``.

:class:`OnnxRuntimeBackend`
    A real ``onnxruntime`` ``InferenceSession``. Latency from this backend is
    a measurement of the host this package runs on --- in this repository, a
    shared single-core cloud container, which is recorded in every output
    file.

:class:`SklearnBackend`
    A fitted scikit-learn estimator's ``predict``. Present because a
    classical estimator is a legitimate candidate for an edge inference slot
    and because it needs no ONNX conversion (``skl2onnx`` is not available in
    the target environment).

The contract is deliberately narrow: ``prepare()`` does everything that may be
done once per deployment, ``infer()`` does exactly what the deadline covers,
and ``close()`` releases resources. The harness times ``infer()`` only, so
session construction and weight loading never leak into a latency figure.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DRY_RUN_SINK",
    "InferenceBackend",
    "OnnxRuntimeBackend",
    "PipelineStage",
    "SimulatedBackend",
    "SklearnBackend",
    "spin_wait",
]

#: Dry-run mode discards outputs here. A dry run exercises the real command
#: path and the real timing while guaranteeing that nothing downstream can
#: consume a result, which is how a deployment is rehearsed without acting on
#: an inference.
DRY_RUN_SINK: list[Any] = []


class InferenceBackend(ABC):
    """One candidate's inference path.

    Subclasses must not do lazy work inside :meth:`infer`: anything that can
    be hoisted into :meth:`prepare` belongs there, or the first timed repeat
    measures initialisation rather than inference. The ``warmup`` repeats in
    :func:`edgeinfer.harness.benchmark` exist as a second line of defence,
    not as a licence to initialise lazily.
    """

    #: Short backend identifier written into every profile and report.
    kind: str = "abstract"

    @abstractmethod
    def prepare(self) -> None:
        """Do all one-off work: build sessions, allocate buffers, load weights."""

    @abstractmethod
    def infer(self) -> Any:
        """Run exactly one inference and return its output."""

    def dry_run(self) -> None:
        """Run one inference and discard the output.

        The real command path and the real timing, with the result guaranteed
        unused. Overridden only if a backend needs to suppress a side effect.
        """
        DRY_RUN_SINK.clear()
        self.infer()

    def close(self) -> None:
        """Release resources. Safe to call more than once.

        The default is a no-op: a backend that holds nothing has nothing to
        release, and making this abstract would force every such backend to
        write an empty override.
        """
        return None

    def __enter__(self) -> InferenceBackend:
        self.prepare()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def spin_wait(duration_s: float) -> None:
    """Busy-wait for ``duration_s`` on ``perf_counter`` [s].

    Used instead of :func:`time.sleep` because ``sleep`` on Linux yields to
    the scheduler and returns late by a timer-slack interval of order 50 us to
    1 ms, which would swamp the microsecond-scale costs being injected. A spin
    overshoots by one loop iteration, a bias of order 100 ns, which the
    validation script measures and reports rather than assuming away.

    Parameters
    ----------
    duration_s
        Non-negative wait [s]. Zero returns immediately.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if duration_s == 0:
        return
    counter = time.perf_counter
    deadline = counter() + duration_s
    while counter() < deadline:
        pass


@dataclass(frozen=True)
class PipelineStage:
    """One stage of a synthetic pipeline's injected cost model.

    Attributes
    ----------
    name
        Stage name.
    mean_s
        Mean duration [s], strictly positive.
    std_s
        Standard deviation [s], non-negative. For ``dist="lognormal"`` this is
        the standard deviation of the *duration*, not of its logarithm; the
        lognormal parameters are derived from the mean and standard deviation
        by the standard moment relations (Johnson, Kotz & Balakrishnan 1994,
        *Continuous Univariate Distributions*, vol. 1, ch. 14):
        ``sigma^2 = ln(1 + (std/mean)^2)``, ``mu = ln(mean) - sigma^2 / 2``.
    dist
        ``"constant"``, ``"normal"`` or ``"lognormal"``. ``"lognormal"`` is the
        default for a latency stage because a duration is positive and
        right-skewed; a normal draw can go negative and is clipped at zero,
        which distorts the mean it was given.
    """

    name: str
    mean_s: float
    std_s: float = 0.0
    dist: str = "lognormal"

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_s) or self.mean_s <= 0:
            raise ValueError(f"stage {self.name!r}: mean_s must be finite and > 0")
        if not np.isfinite(self.std_s) or self.std_s < 0:
            raise ValueError(f"stage {self.name!r}: std_s must be finite and >= 0")
        if self.dist not in ("constant", "normal", "lognormal"):
            raise ValueError(
                f"stage {self.name!r}: dist must be 'constant', 'normal' or 'lognormal', "
                f"got {self.dist!r}"
            )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one duration [s], never negative."""
        if self.dist == "constant" or self.std_s == 0.0:
            return float(self.mean_s)
        if self.dist == "normal":
            return float(max(0.0, rng.normal(self.mean_s, self.std_s)))
        sigma_sq = float(np.log1p((self.std_s / self.mean_s) ** 2))
        mu = float(np.log(self.mean_s) - sigma_sq / 2.0)
        return float(rng.lognormal(mu, np.sqrt(sigma_sq)))

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable description, for a cross-check file."""
        return {"name": self.name, "mean_s": self.mean_s, "std_s": self.std_s,
                "dist": self.dist}


class SimulatedBackend(InferenceBackend):
    """A pipeline whose per-stage cost is injected, not measured.

    Parameters
    ----------
    stages
        The injected cost model, at least one stage.
    seed
        Seed for the duration sampler. The whole pipeline is reproducible from
        ``(stages, seed, consume_time)``.
    consume_time
        When ``True`` (the default) each stage busy-waits for its sampled
        duration, so that the timing harness measures something real and the
        injected cost model can be recovered from measurements. When
        ``False`` the backend only accumulates the sampled durations, which is
        what a unit test of the sampler wants.
    work_bytes
        Optional per-stage array size [B]. When non-zero each stage allocates
        and touches an array of that size, so that a memory measurement has
        something to see. Zero by default.
    """

    kind = "simulated"

    def __init__(
        self,
        stages: tuple[PipelineStage, ...],
        seed: int = 0,
        consume_time: bool = True,
        work_bytes: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("SimulatedBackend needs at least one stage")
        if work_bytes < 0:
            raise ValueError(f"work_bytes must be non-negative, got {work_bytes}")
        self.stages = tuple(stages)
        self.seed = int(seed)
        self.consume_time = bool(consume_time)
        self.work_bytes = int(work_bytes)
        self._rng: np.random.Generator | None = None
        self.last_sampled_s: tuple[float, ...] = ()

    @property
    def injected_mean_s(self) -> float:
        """Sum of the stages' declared means [s]: the cost model's own mean."""
        return float(sum(s.mean_s for s in self.stages))

    @property
    def injected_std_s(self) -> float:
        """Standard deviation of the stage sum [s], stages independent.

        Independent stages' variances add (standard result for the variance of
        a sum of independent random variables), so this is
        ``sqrt(sum(std_i^2))``.
        """
        return float(np.sqrt(sum(s.std_s**2 for s in self.stages)))

    def prepare(self) -> None:
        """Seed the sampler. Resets the stream, so a run is reproducible."""
        self._rng = np.random.default_rng(self.seed)

    def infer(self) -> float:
        """Run one pipeline pass and return the total sampled duration [s]."""
        if self._rng is None:
            raise RuntimeError("SimulatedBackend.prepare() must be called before infer()")
        sampled: list[float] = []
        total = 0.0
        for stage in self.stages:
            duration = stage.sample(self._rng)
            sampled.append(duration)
            total += duration
            if self.work_bytes:
                buffer = np.empty(self.work_bytes, dtype=np.uint8)
                buffer[::4096] = 1
            if self.consume_time:
                spin_wait(duration)
        self.last_sampled_s = tuple(sampled)
        return total

    def pipeline_dict(self, n_samples: int) -> dict[str, object]:
        """JSON-serialisable pipeline definition, reproducible from the seed."""
        return {
            "stages": [s.as_dict() for s in self.stages],
            "n_samples": int(n_samples),
            "seed": self.seed,
        }


class OnnxRuntimeBackend(InferenceBackend):
    """An ``onnxruntime`` ``InferenceSession`` over a serialised model.

    Parameters
    ----------
    model_bytes
        Serialised ONNX ModelProto.
    feed
        Input dict; the same object is reused for every call so that feed
        construction never enters a timed repeat.
    providers
        Execution providers, in priority order. Defaults to CPU only, because
        a figure from a provider the reader does not have is not a useful
        figure.
    intra_op_num_threads
        Threads inside one operator. Defaults to 1: on a shared single-core
        host, more threads produce contention and a longer tail, and a
        single-thread number is the one that transfers to an edge target with
        a declared core allocation.
    """

    kind = "onnxruntime"

    def __init__(
        self,
        model_bytes: bytes,
        feed: dict[str, np.ndarray],
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        intra_op_num_threads: int = 1,
    ) -> None:
        if intra_op_num_threads < 1:
            raise ValueError(
                f"intra_op_num_threads must be >= 1, got {intra_op_num_threads}"
            )
        self.model_bytes = model_bytes
        self.feed = feed
        self.providers = tuple(providers)
        self.intra_op_num_threads = int(intra_op_num_threads)
        self._session: Any = None
        self._output_names: list[str] = []

    def prepare(self) -> None:
        """Build the session. Everything expensive happens here, not in infer."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_num_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3  # errors only; warnings pollute a report
        self._session = ort.InferenceSession(
            self.model_bytes, sess_options=options, providers=list(self.providers)
        )
        self._output_names = [o.name for o in self._session.get_outputs()]

    def infer(self) -> list[np.ndarray]:
        """One ``session.run`` over the stored feed."""
        if self._session is None:
            raise RuntimeError("OnnxRuntimeBackend.prepare() must be called before infer()")
        return self._session.run(self._output_names, self.feed)

    def close(self) -> None:
        """Drop the session."""
        self._session = None

    @property
    def session(self) -> Any:
        """The underlying ``InferenceSession``, or ``None`` before prepare."""
        return self._session


class SklearnBackend(InferenceBackend):
    """A fitted scikit-learn estimator's ``predict`` as the inference path.

    Parameters
    ----------
    estimator
        Any fitted object with ``predict``.
    x
        Input batch, reused for every call.
    """

    kind = "sklearn"

    def __init__(self, estimator: Any, x: np.ndarray) -> None:
        if not hasattr(estimator, "predict"):
            raise TypeError(
                f"{type(estimator).__name__} has no predict(); SklearnBackend needs a "
                "fitted scikit-learn-style estimator"
            )
        self.estimator = estimator
        self.x = np.asarray(x)

    def prepare(self) -> None:
        """One untimed call, so that any lazy allocation inside the estimator
        happens before the timed repeats."""
        self.estimator.predict(self.x)

    def infer(self) -> np.ndarray:
        """One ``predict`` call."""
        return self.estimator.predict(self.x)
