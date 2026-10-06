"""Hardware abstraction layer for a modem/codec device, with deployment checks.

What this is and is not
-----------------------
This module defines **one** interface, :class:`ModemBackend`, for the device that
would encode, interleave and decode on real hardware, and two implementations:

* :class:`SimulatedModemBackend` -- fully implements the interface against the
  models in this package. Deterministic and seeded.
* :class:`DeviceModemBackend` -- carries the documented contract for each method
  and raises :class:`NotImplementedError` on every one. It exists so that the
  contract is written down *before* the hardware, and so that one shared test
  suite (``tests/test_hal_contract.py``, parameterised over both backends) holds
  the device implementation to the same contract the day it is written.

**No number produced by** :class:`SimulatedModemBackend` **is hardware
evidence.** This product is validation level 3 with ``hardware_pending: true``.
What is missing for level 4 is measured timing and resource use from a Jetson
Orin Nano; no simulated backend, extrapolation, vendor datasheet or workstation
run substitutes for it.

Run modes
---------
:class:`RunMode` has three values:

``SIMULATION``
    The full loop against the models. Results are returned.
``DRY_RUN``
    The real command path with real timing, outputs discarded. Used to rehearse
    a deployment: every preflight check runs, every command is issued, every
    duration is measured, and the payload is thrown away. A dry run on the
    device backend exercises the device's command path; a dry run on the
    simulated backend exercises only this package's.
``LIVE``
    Reserved for real hardware. :class:`SimulatedModemBackend` refuses it, so a
    simulated result can never be mislabelled as a live one.

Deployment and recovery, as executable checks
---------------------------------------------
* :func:`preflight_checks` returns a list of named, runnable checks. Each returns
  a :class:`CheckResult`. :func:`run_preflight` runs them all and reports; a run
  that does not pass preflight is refused by :meth:`ModemSession.start`.
* :class:`RunCapture` states exactly what is recorded during a run: the
  configuration hash, the backend identity, the run mode, the per-stage
  durations, the counters, and the preflight report. It is serialisable to a
  plain dict of JSON-safe types, with **no filesystem paths in it**.
* :meth:`ModemSession.backout` is the recovery path for a half-finished run: it
  resets the backend, marks the capture ``aborted`` with a stated reason, and
  returns the capture so the partial record survives. It is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .channel import ChannelConfig
from .interleave import BlockInterleaver
from .reedsolomon import ReedSolomon

#: Largest interleaver block, in symbols, that the ``round_trip`` preflight check
#: will attempt. A preflight must be cheap enough to run before every run; a block
#: larger than this is reported as un-preflightable instead of being encoded.
ROUND_TRIP_SYMBOL_LIMIT = 200_000


class RunMode(Enum):
    """Execution mode of a modem session."""

    SIMULATION = "simulation"
    DRY_RUN = "dry_run"
    LIVE = "live"


@dataclass(frozen=True)
class ModemConfig:
    """Configuration a backend is asked to realise.

    Attributes
    ----------
    n:
        Codeword length, symbols.
    k:
        Message length, symbols.
    m:
        Field extension degree; bits per symbol.
    depth:
        Block-interleaver depth, symbols.
    symbol_rate_hz:
        Code symbol rate, Hz.
    """

    n: int
    k: int
    m: int
    depth: int
    symbol_rate_hz: float

    def __post_init__(self) -> None:
        if not 1 <= self.k < self.n:
            raise ValueError(f"require 1 <= k < n, got k={self.k}, n={self.n}")
        if (self.n - self.k) % 2:
            raise ValueError(f"n - k must be even, got {self.n - self.k}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth!r}")
        if not self.symbol_rate_hz > 0:
            raise ValueError(f"symbol_rate_hz must be > 0 Hz, got {self.symbol_rate_hz!r}")

    @property
    def identity(self) -> str:
        """Stable short hash of the configuration, for the run capture."""
        payload = json.dumps(
            {
                "n": self.n,
                "k": self.k,
                "m": self.m,
                "depth": self.depth,
                "symbol_rate_hz": self.symbol_rate_hz,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one preflight check.

    Attributes
    ----------
    name:
        Check identifier.
    passed:
        Whether the check passed.
    detail:
        One-line human-readable result. Must contain no filesystem path.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    """All preflight results for one intended run."""

    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """``True`` only if every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """The checks that did not pass."""
        return tuple(r for r in self.results if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "passed": self.passed,
            "checks": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
        }


@dataclass
class RunCapture:
    """Everything recorded during a run.

    Nothing here is a filesystem path, by design: the release gate for this
    portfolio fails on an absolute path anywhere in tracked content, including
    inside a saved capture.
    """

    config_identity: str
    backend_name: str
    mode: str
    environment: dict[str, str]
    stage_durations_s: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    preflight: dict[str, Any] | None = None
    status: str = "open"
    abort_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the whole capture."""
        return {
            "config_identity": self.config_identity,
            "backend_name": self.backend_name,
            "mode": self.mode,
            "environment": dict(self.environment),
            "stage_durations_s": {k: float(v) for k, v in self.stage_durations_s.items()},
            "counters": {k: int(v) for k, v in self.counters.items()},
            "preflight": self.preflight,
            "status": self.status,
            "abort_reason": self.abort_reason,
        }


def environment_record() -> dict[str, str]:
    """Environment facts worth recording with a measurement. No paths."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "numpy": np.__version__,
        "maxsize_bits": str(sys.maxsize.bit_length() + 1),
    }


class ModemBackend(ABC):
    """One interface for the modem/codec device.

    Every method's contract is stated here and is enforced by the shared test
    suite for both the simulated and the device backend.
    """

    name: str = "abstract"

    @abstractmethod
    def open(self, config: ModemConfig, mode: RunMode) -> None:
        """Acquire the device and apply ``config``.

        Contract: idempotent for a repeated identical ``(config, mode)``; raises
        ``RuntimeError`` if called with a different configuration while already
        open; must leave the backend in a state where :meth:`is_open` is ``True``
        on success and ``False`` on any failure.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the device.

        Contract: safe to call when already closed; after it returns
        :meth:`is_open` is ``False``.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Whether the backend currently holds the device."""

    @abstractmethod
    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode and interleave ``depth`` codewords' worth of message symbols.

        Contract: ``message`` has shape ``(depth, k)`` with values in
        ``[0, 2**m)``; the return has shape ``(depth * n,)``, the same dtype
        family (integer), and is the interleaved codeword block. Raises
        ``RuntimeError`` if not open, ``ValueError`` on a shape or range
        violation.
        """

    @abstractmethod
    def decode(self, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """De-interleave and decode one interleaved block.

        Contract: ``received`` has shape ``(depth * n,)``; returns
        ``(messages, ok)`` with ``messages`` of shape ``(depth, k)`` and ``ok``
        a boolean array of shape ``(depth,)`` that is ``True`` where the decoder
        reported success. Raises ``RuntimeError`` if not open, ``ValueError`` on
        a shape or range violation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Return the device to its post-open state without releasing it.

        Contract: safe to call at any time, including when closed; clears any
        partial state so that a subsequent :meth:`encode` behaves as if no
        previous call had happened.
        """

    @abstractmethod
    def self_test(self) -> CheckResult:
        """Run the backend's own confidence check.

        Contract: never raises; returns a :class:`CheckResult` whose ``detail``
        contains no filesystem path.
        """


class SimulatedModemBackend(ModemBackend):
    """Full implementation of :class:`ModemBackend` against this package's models.

    Deterministic: given the same configuration and the same input it returns
    the same output, with no internal randomness at all.
    """

    name = "simulated"

    def __init__(self) -> None:
        self._config: ModemConfig | None = None
        self._mode: RunMode | None = None
        self._code: ReedSolomon | None = None
        self._interleaver: BlockInterleaver | None = None
        self._blocks_encoded = 0

    def open(self, config: ModemConfig, mode: RunMode) -> None:
        if mode is RunMode.LIVE:
            raise ValueError(
                "SimulatedModemBackend refuses RunMode.LIVE: a simulated result must "
                "never be labelled as a live hardware measurement"
            )
        if self._config is not None:
            if self._config == config and self._mode is mode:
                return
            raise RuntimeError(
                "backend already open with a different configuration; close() first"
            )
        try:
            code = ReedSolomon(config.n, config.k, config.m)
            interleaver = BlockInterleaver(config.depth, config.n)
        except Exception:
            self._config = None
            self._mode = None
            raise
        self._config = config
        self._mode = mode
        self._code = code
        self._interleaver = interleaver
        self._blocks_encoded = 0

    def close(self) -> None:
        self._config = None
        self._mode = None
        self._code = None
        self._interleaver = None
        self._blocks_encoded = 0

    def is_open(self) -> bool:
        return self._config is not None

    def reset(self) -> None:
        self._blocks_encoded = 0

    def _require_open(self) -> tuple[ModemConfig, ReedSolomon, BlockInterleaver]:
        if self._config is None or self._code is None or self._interleaver is None:
            raise RuntimeError("backend is not open; call open() first")
        return self._config, self._code, self._interleaver

    def encode(self, message: np.ndarray) -> np.ndarray:
        config, code, interleaver = self._require_open()
        msg = np.asarray(message, dtype=np.int64)
        if msg.shape != (config.depth, config.k):
            raise ValueError(
                f"message must have shape ({config.depth}, {config.k}), got {msg.shape}"
            )
        if msg.size and (msg.min() < 0 or msg.max() >= (1 << config.m)):
            raise ValueError(f"message symbols must lie in [0, {(1 << config.m) - 1}]")
        block = np.concatenate([code.encode(row) for row in msg])
        self._blocks_encoded += 1
        return interleaver.interleave(block)

    def decode(self, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        config, code, interleaver = self._require_open()
        r = np.asarray(received, dtype=np.int64)
        if r.shape != (config.depth * config.n,):
            raise ValueError(
                f"received must have shape ({config.depth * config.n},), got {r.shape}"
            )
        if r.size and (r.min() < 0 or r.max() >= (1 << config.m)):
            raise ValueError(f"received symbols must lie in [0, {(1 << config.m) - 1}]")
        words = interleaver.deinterleave(r).reshape(config.depth, config.n)
        messages = np.empty((config.depth, config.k), dtype=np.int64)
        ok = np.empty(config.depth, dtype=bool)
        for i, word in enumerate(words):
            result = code.decode(word)
            messages[i] = result.message
            ok[i] = result.success
        return messages, ok

    def self_test(self) -> CheckResult:
        try:
            config, code, _ = self._require_open()
            msg = np.arange(config.k, dtype=np.int64) % (1 << config.m)
            cw = code.encode(msg)
            out = code.decode(cw)
            ok = bool(out.success and np.array_equal(out.message, msg))
            return CheckResult(
                "backend_self_test",
                ok,
                f"simulated RS({config.n},{config.k}) round trip ok={ok}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            return CheckResult("backend_self_test", False, f"{type(exc).__name__}")


class DeviceModemBackend(ModemBackend):
    """Contract-only backend for real hardware. Every method raises.

    The target is a Jetson Orin Nano carrying the modem/codec. Each method below
    repeats the contract from :class:`ModemBackend` and adds what the device
    implementation must additionally guarantee. Nothing here is implemented and
    nothing here has been measured.
    """

    name = "device"

    def __init__(self, device_id: str = "jetson-orin-nano") -> None:
        self.device_id = str(device_id)

    _MISSING = (
        "DeviceModemBackend is a contract, not an implementation. Level 4 for this "
        "product requires measured timing and resource use from a Jetson Orin Nano; "
        "no simulated backend, extrapolation or workstation run substitutes for it."
    )

    def open(self, config: ModemConfig, mode: RunMode) -> None:
        """Contract as :meth:`ModemBackend.open`, plus: must verify that the
        device firmware reports the same ``(n, k, m, depth)`` it was asked for and
        fail rather than silently reconfigure; must not hold the device if it
        raises."""
        raise NotImplementedError(self._MISSING)

    def close(self) -> None:
        """Contract as :meth:`ModemBackend.close`, plus: must release the device
        even if the last operation errored, so a crashed run does not strand it."""
        raise NotImplementedError(self._MISSING)

    def is_open(self) -> bool:
        """Contract as :meth:`ModemBackend.is_open`. Returns ``False`` here
        because the device is never opened."""
        return False

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Contract as :meth:`ModemBackend.encode`, plus: must be bit-exact
        against :class:`SimulatedModemBackend` for the same input, which is the
        acceptance test the day the device exists."""
        raise NotImplementedError(self._MISSING)

    def decode(self, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Contract as :meth:`ModemBackend.decode`, plus: must agree with
        :class:`SimulatedModemBackend` on both the decoded messages and the
        success flags for every input within the correction radius."""
        raise NotImplementedError(self._MISSING)

    def reset(self) -> None:
        """Contract as :meth:`ModemBackend.reset`, plus: must clear device FIFOs
        so that a backed-out run leaves no symbols in flight."""
        raise NotImplementedError(self._MISSING)

    def self_test(self) -> CheckResult:
        """Contract as :meth:`ModemBackend.self_test`. Never raises; reports the
        missing implementation as a failed check."""
        return CheckResult(
            "backend_self_test", False, "device backend not implemented (hardware pending)"
        )


def preflight_checks(
    config: ModemConfig,
    backend: ModemBackend,
    channel: ChannelConfig | None = None,
    mode: RunMode = RunMode.DRY_RUN,
) -> list[tuple[str, Callable[[], CheckResult]]]:
    """Named, runnable preflight checks for one intended run.

    Each entry is ``(name, thunk)``; the thunk returns a :class:`CheckResult` and
    does not raise. The set is deliberately small and all of it is checkable
    without hardware:

    ``code_parameters``
        ``n``, ``k`` and ``m`` form a constructible Reed-Solomon code.
    ``interleaver_capacity``
        The interleaver block fits in a stated memory budget and its latency is
        reported, so a run cannot start with a depth nobody costed.
    ``depth_against_fade``
        If a channel is supplied, the depth is at least the mean fade length in
        symbols. Fails loudly when it is not, which is the mistake this whole
        product exists to prevent.
    ``backend_open``
        The backend opens with this configuration **in the mode the run will use**,
        so that the check leaves the backend in the state the run needs and
        :meth:`ModemSession.start` finds its own ``open`` call idempotent.
    ``backend_self_test``
        The backend's own confidence check.
    ``round_trip``
        An encode/decode round trip through the backend returns the input.
    """

    def code_parameters() -> CheckResult:
        try:
            code = ReedSolomon(config.n, config.k, config.m)
        except Exception as exc:
            return CheckResult("code_parameters", False, f"{type(exc).__name__}: {exc}")
        return CheckResult(
            "code_parameters",
            True,
            f"RS({code.n},{code.k}) over GF(2^{code.m}), t={code.t}, rate={code.rate:.4f}",
        )

    def interleaver_capacity() -> CheckResult:
        il = BlockInterleaver(config.depth, config.n)
        cost = il.cost(config.symbol_rate_hz, bits_per_symbol=config.m)
        ok = cost.memory_bytes <= 64e6
        return CheckResult(
            "interleaver_capacity",
            ok,
            f"depth={config.depth} costs {cost.memory_bytes:.0f} B and "
            f"{cost.latency_ms:.3f} ms end to end (budget 64 MB)",
        )

    def depth_against_fade() -> CheckResult:
        if channel is None:
            return CheckResult(
                "depth_against_fade", True, "no channel supplied; check skipped"
            )
        lc = channel.samples_per_correlation_time
        ok = config.depth >= lc
        return CheckResult(
            "depth_against_fade",
            ok,
            f"depth={config.depth} vs correlation length Lc={lc:.1f} symbols; "
            f"ratio={config.depth / lc:.3f}",
        )

    def backend_open() -> CheckResult:
        try:
            backend.open(config, mode)
        except Exception as exc:
            return CheckResult("backend_open", False, f"{type(exc).__name__}: {exc}")
        return CheckResult("backend_open", backend.is_open(), f"backend={backend.name}")

    def backend_self_test() -> CheckResult:
        return backend.self_test()

    def round_trip() -> CheckResult:
        # A preflight check must be cheap enough to run before every run. One
        # interleaver block is depth*n symbols and each codeword costs a software
        # encode and decode, so a configuration whose block exceeds this guard is
        # reported as un-preflightable rather than attempted; such a depth has
        # already failed interleaver_capacity above.
        block_symbols = config.depth * config.n
        if block_symbols > ROUND_TRIP_SYMBOL_LIMIT:
            return CheckResult(
                "round_trip",
                False,
                f"one block is {block_symbols} symbols, above the preflight limit of "
                f"{ROUND_TRIP_SYMBOL_LIMIT}; reduce depth before running",
            )
        try:
            msg = np.tile(
                np.arange(config.k, dtype=np.int64) % (1 << config.m), (config.depth, 1)
            )
            block = backend.encode(msg)
            out, ok = backend.decode(block)
            good = bool(np.array_equal(out, msg) and ok.all())
            return CheckResult("round_trip", good, f"encode/decode identity={good}")
        except Exception as exc:
            return CheckResult("round_trip", False, f"{type(exc).__name__}: {exc}")

    return [
        ("code_parameters", code_parameters),
        ("interleaver_capacity", interleaver_capacity),
        ("depth_against_fade", depth_against_fade),
        ("backend_open", backend_open),
        ("backend_self_test", backend_self_test),
        ("round_trip", round_trip),
    ]


def run_preflight(
    config: ModemConfig,
    backend: ModemBackend,
    channel: ChannelConfig | None = None,
    mode: RunMode = RunMode.DRY_RUN,
) -> PreflightReport:
    """Run every check from :func:`preflight_checks` and collect the results."""
    results = []
    for _name, thunk in preflight_checks(config, backend, channel, mode):
        results.append(thunk())
    return PreflightReport(tuple(results))


class ModemSession:
    """A run of the modem through a backend, with preflight, capture and backout.

    Parameters
    ----------
    backend:
        The backend to drive.
    config:
        The configuration to apply.
    mode:
        :class:`RunMode`. ``DRY_RUN`` issues every command and measures every
        duration but discards the payloads.
    channel:
        Optional channel configuration, used by the ``depth_against_fade``
        preflight check.
    """

    def __init__(
        self,
        backend: ModemBackend,
        config: ModemConfig,
        mode: RunMode = RunMode.SIMULATION,
        channel: ChannelConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.mode = mode
        self.channel = channel
        self.capture = RunCapture(
            config_identity=config.identity,
            backend_name=backend.name,
            mode=mode.value,
            environment=environment_record(),
        )
        self._started = False

    def start(self) -> PreflightReport:
        """Run preflight and open the backend. Refuses to start if preflight fails."""
        report = run_preflight(self.config, self.backend, self.channel, self.mode)
        self.capture.preflight = report.to_dict()
        if not report.passed:
            self.capture.status = "refused"
            self.capture.abort_reason = "preflight failed: " + ", ".join(
                r.name for r in report.failures
            )
            raise RuntimeError(self.capture.abort_reason)
        self.backend.open(self.config, self.mode)
        self.backend.reset()
        self._started = True
        self.capture.status = "running"
        return report

    def transfer(self, blocks: int, seed: int = 0) -> dict[str, Any] | None:
        """Push ``blocks`` interleaved blocks through the backend.

        Returns a summary dict in ``SIMULATION`` mode and ``None`` in
        ``DRY_RUN`` mode, where the payloads are discarded by design. Durations
        and counters are recorded in :attr:`capture` in both modes.
        """
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks!r}")
        rng = np.random.default_rng(seed)
        enc_s = 0.0
        dec_s = 0.0
        symbols = 0
        frames_ok = 0
        frames = 0
        for _ in range(blocks):
            msg = rng.integers(
                0, 1 << self.config.m, size=(self.config.depth, self.config.k)
            ).astype(np.int64)
            t0 = time.perf_counter()
            block = self.backend.encode(msg)
            enc_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            out, ok = self.backend.decode(block)
            dec_s += time.perf_counter() - t0
            symbols += int(block.size)
            frames += int(self.config.depth)
            frames_ok += int(np.count_nonzero(ok & np.all(out == msg, axis=1)))
        self.capture.stage_durations_s["encode"] = enc_s
        self.capture.stage_durations_s["decode"] = dec_s
        self.capture.counters["symbols"] = symbols
        self.capture.counters["frames"] = frames
        self.capture.counters["frames_ok"] = frames_ok
        if self.mode is RunMode.DRY_RUN:
            return None
        return {
            "blocks": blocks,
            "frames": frames,
            "frames_ok": frames_ok,
            "encode_s": enc_s,
            "decode_s": dec_s,
            "symbol_throughput_per_s": symbols / (enc_s + dec_s) if enc_s + dec_s else 0.0,
        }

    def finish(self) -> RunCapture:
        """Close the backend and mark the capture complete."""
        self.backend.close()
        self._started = False
        if self.capture.status == "running":
            self.capture.status = "complete"
        return self.capture

    def backout(self, reason: str) -> RunCapture:
        """Recovery path for a half-finished run.

        Resets and closes the backend, marks the capture ``aborted`` with
        ``reason``, and returns it so the partial record survives. Idempotent:
        calling it twice leaves the first reason in place.
        """
        if self.capture.status != "aborted":
            try:
                self.backend.reset()
            except NotImplementedError:
                pass
            try:
                self.backend.close()
            except NotImplementedError:
                pass
            self._started = False
            self.capture.status = "aborted"
            self.capture.abort_reason = str(reason)
        return self.capture
