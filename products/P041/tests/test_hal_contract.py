"""One contract suite, parameterised over the simulated and the device backend.

The point of this file is that the device backend is held to the same contract
the day it is implemented. Until then, every contract test asserts the documented
failure: :class:`NotImplementedError` with a message that says what is missing.

Exercises REQ-23, REQ-24, REQ-25, REQ-26, REQ-27 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from codedfade.channel import ChannelConfig
from codedfade.hal import (
    DeviceModemBackend,
    ModemBackend,
    ModemConfig,
    ModemSession,
    RunMode,
    SimulatedModemBackend,
    environment_record,
    preflight_checks,
    run_preflight,
)

CONFIG = ModemConfig(n=31, k=21, m=5, depth=8, symbol_rate_hz=1.0e6)
CHANNEL = ChannelConfig(0.6, 2.0e-4, 1.0e6, seed=0)


@pytest.fixture(params=["simulated", "device"])
def backend(request: pytest.FixtureRequest) -> ModemBackend:
    return (
        SimulatedModemBackend() if request.param == "simulated" else DeviceModemBackend()
    )


def _is_device(backend: ModemBackend) -> bool:
    return isinstance(backend, DeviceModemBackend)


class TestSharedContract:
    """Every test here applies to both backends."""

    def test_starts_closed(self, backend: ModemBackend) -> None:
        assert backend.is_open() is False

    def test_open_then_is_open(self, backend: ModemBackend) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError, match="Jetson Orin Nano"):
                backend.open(CONFIG, RunMode.SIMULATION)
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        assert backend.is_open() is True

    def test_open_is_idempotent_for_the_same_configuration(
        self, backend: ModemBackend
    ) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.open(CONFIG, RunMode.SIMULATION)
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        backend.open(CONFIG, RunMode.SIMULATION)
        assert backend.is_open() is True

    def test_open_with_a_different_configuration_raises(
        self, backend: ModemBackend
    ) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.open(CONFIG, RunMode.SIMULATION)
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        other = ModemConfig(n=31, k=21, m=5, depth=16, symbol_rate_hz=1.0e6)
        with pytest.raises(RuntimeError, match="different configuration"):
            backend.open(other, RunMode.SIMULATION)

    def test_close_is_safe_when_already_closed(self, backend: ModemBackend) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.close()
            return
        backend.close()
        backend.close()
        assert backend.is_open() is False

    def test_encode_requires_open(self, backend: ModemBackend) -> None:
        msg = np.zeros((CONFIG.depth, CONFIG.k), dtype=np.int64)
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.encode(msg)
            return
        with pytest.raises(RuntimeError, match="not open"):
            backend.encode(msg)

    def test_decode_requires_open(self, backend: ModemBackend) -> None:
        block = np.zeros(CONFIG.depth * CONFIG.n, dtype=np.int64)
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.decode(block)
            return
        with pytest.raises(RuntimeError, match="not open"):
            backend.decode(block)

    def test_encode_shape_contract(self, backend: ModemBackend) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.encode(np.zeros((1, 1), dtype=np.int64))
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        with pytest.raises(ValueError, match="message must have shape"):
            backend.encode(np.zeros((CONFIG.depth, CONFIG.k + 1), dtype=np.int64))
        with pytest.raises(ValueError, match="symbols must lie in"):
            backend.encode(np.full((CONFIG.depth, CONFIG.k), 32, dtype=np.int64))

    def test_decode_shape_contract(self, backend: ModemBackend) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.decode(np.zeros(3, dtype=np.int64))
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        with pytest.raises(ValueError, match="received must have shape"):
            backend.decode(np.zeros(3, dtype=np.int64))
        with pytest.raises(ValueError, match="symbols must lie in"):
            backend.decode(np.full(CONFIG.depth * CONFIG.n, 32, dtype=np.int64))

    def test_round_trip_identity(self, backend: ModemBackend) -> None:
        msg = np.tile(np.arange(CONFIG.k, dtype=np.int64), (CONFIG.depth, 1))
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.encode(msg)
            return
        backend.open(CONFIG, RunMode.SIMULATION)
        out, ok = backend.decode(backend.encode(msg))
        assert np.array_equal(out, msg)
        assert bool(ok.all())

    def test_reset_is_safe_when_closed(self, backend: ModemBackend) -> None:
        if _is_device(backend):
            with pytest.raises(NotImplementedError):
                backend.reset()
            return
        backend.reset()
        assert backend.is_open() is False

    def test_self_test_never_raises(self, backend: ModemBackend) -> None:
        result = backend.self_test()
        assert result.name == "backend_self_test"
        assert isinstance(result.passed, bool)
        assert "/" not in result.detail or not result.detail.startswith("/")

    def test_name_is_set(self, backend: ModemBackend) -> None:
        assert backend.name in ("simulated", "device")


class TestSimulatedOnly:
    def test_refuses_live_mode(self) -> None:
        """A simulated result must never be labelled a live hardware measurement."""
        with pytest.raises(ValueError, match="refuses RunMode.LIVE"):
            SimulatedModemBackend().open(CONFIG, RunMode.LIVE)

    def test_is_deterministic(self) -> None:
        msg = np.tile(np.arange(CONFIG.k, dtype=np.int64), (CONFIG.depth, 1))
        a, b = SimulatedModemBackend(), SimulatedModemBackend()
        a.open(CONFIG, RunMode.SIMULATION)
        b.open(CONFIG, RunMode.SIMULATION)
        assert np.array_equal(a.encode(msg), b.encode(msg))

    def test_corrects_up_to_t_symbol_errors_through_the_backend(self) -> None:
        backend = SimulatedModemBackend()
        backend.open(CONFIG, RunMode.SIMULATION)
        msg = np.tile(np.arange(CONFIG.k, dtype=np.int64), (CONFIG.depth, 1))
        block = backend.encode(msg)
        rng = np.random.default_rng(0)
        # a burst of depth*t symbols is exactly what depth t-error correction
        # survives after de-interleaving
        start = 100
        length = CONFIG.depth * 5
        block[start : start + length] ^= rng.integers(
            1, 1 << CONFIG.m, size=length
        ).astype(np.int64)
        out, ok = backend.decode(block)
        assert bool(ok.all())
        assert np.array_equal(out, msg)

    def test_open_failure_leaves_the_backend_closed(self) -> None:
        backend = SimulatedModemBackend()
        bad = ModemConfig(n=31, k=21, m=4, depth=4, symbol_rate_hz=1e6)
        with pytest.raises(ValueError):
            backend.open(bad, RunMode.SIMULATION)
        assert backend.is_open() is False


class TestDeviceOnly:
    def test_every_operational_method_raises_with_the_hardware_message(self) -> None:
        device = DeviceModemBackend()
        for call in (
            lambda: device.open(CONFIG, RunMode.LIVE),
            lambda: device.close(),
            lambda: device.encode(np.zeros((CONFIG.depth, CONFIG.k), dtype=np.int64)),
            lambda: device.decode(np.zeros(CONFIG.depth * CONFIG.n, dtype=np.int64)),
            lambda: device.reset(),
        ):
            with pytest.raises(NotImplementedError, match="Jetson Orin Nano"):
                call()

    def test_self_test_reports_the_missing_implementation(self) -> None:
        result = DeviceModemBackend().self_test()
        assert result.passed is False
        assert "hardware pending" in result.detail

    def test_device_id_is_recorded(self) -> None:
        assert DeviceModemBackend().device_id == "jetson-orin-nano"


class TestModemConfig:
    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            (dict(n=31, k=31, m=5, depth=1, symbol_rate_hz=1e6), "1 <= k < n"),
            (dict(n=31, k=20, m=5, depth=1, symbol_rate_hz=1e6), "n - k must be even"),
            (dict(n=31, k=21, m=5, depth=0, symbol_rate_hz=1e6), "depth must be >= 1"),
            (dict(n=31, k=21, m=5, depth=1, symbol_rate_hz=0.0), "symbol_rate_hz"),
        ],
    )
    def test_rejects_bad_parameters(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ModemConfig(**kwargs)

    def test_identity_is_stable_and_short(self) -> None:
        assert CONFIG.identity == ModemConfig(**vars(CONFIG)).identity
        assert len(CONFIG.identity) == 16

    def test_identity_changes_with_the_configuration(self) -> None:
        other = ModemConfig(n=31, k=21, m=5, depth=16, symbol_rate_hz=1e6)
        assert other.identity != CONFIG.identity


class TestPreflight:
    def test_all_checks_pass_with_a_sufficient_depth(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        report = run_preflight(config, SimulatedModemBackend(), CHANNEL)
        assert report.passed, report.to_dict()
        assert len(report.results) == 6

    def test_depth_against_fade_fails_when_the_depth_is_too_small(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=4, symbol_rate_hz=1.0e6)
        report = run_preflight(config, SimulatedModemBackend(), CHANNEL)
        assert not report.passed
        assert [r.name for r in report.failures] == ["depth_against_fade"]

    def test_depth_against_fade_is_skipped_without_a_channel(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=4, symbol_rate_hz=1.0e6)
        report = run_preflight(config, SimulatedModemBackend(), None)
        assert report.passed

    def test_interleaver_capacity_fails_on_an_absurd_depth(self) -> None:
        """A depth nobody costed must be caught before anything is encoded.

        The round-trip check must also refuse rather than attempt it: one block at
        this depth is 510 million symbols, which would exhaust the host.
        """
        config = ModemConfig(n=255, k=223, m=8, depth=2_000_000, symbol_rate_hz=1.0e6)
        report = run_preflight(config, SimulatedModemBackend(), None)
        names = [r.name for r in report.failures]
        assert "interleaver_capacity" in names
        assert "round_trip" in names
        detail = next(r.detail for r in report.results if r.name == "round_trip")
        assert "above the preflight limit" in detail

    def test_device_backend_fails_preflight(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        report = run_preflight(config, DeviceModemBackend(), CHANNEL)
        assert not report.passed
        assert {"backend_open", "backend_self_test", "round_trip"} <= {
            r.name for r in report.failures
        }

    def test_checks_are_named_and_callable(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        checks = preflight_checks(config, SimulatedModemBackend(), CHANNEL)
        assert [name for name, _ in checks] == [
            "code_parameters",
            "interleaver_capacity",
            "depth_against_fade",
            "backend_open",
            "backend_self_test",
            "round_trip",
        ]
        for _name, thunk in checks:
            thunk()  # must not raise

    def test_no_check_detail_contains_an_absolute_path(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        report = run_preflight(config, SimulatedModemBackend(), CHANNEL)
        for r in report.results:
            assert "/home/" not in r.detail
            assert "/Users/" not in r.detail

    def test_code_parameters_check_fails_on_an_impossible_code(self) -> None:
        config = ModemConfig.__new__(ModemConfig)
        object.__setattr__(config, "n", 31)
        object.__setattr__(config, "k", 21)
        object.__setattr__(config, "m", 4)
        object.__setattr__(config, "depth", 8)
        object.__setattr__(config, "symbol_rate_hz", 1e6)
        report = run_preflight(config, SimulatedModemBackend(), None)
        assert "code_parameters" in {r.name for r in report.failures}


class TestSession:
    def test_simulation_mode_returns_a_summary(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(
            SimulatedModemBackend(), config, RunMode.SIMULATION, CHANNEL
        )
        session.start()
        summary = session.transfer(1, seed=0)
        assert summary is not None
        assert summary["frames"] == 256
        assert summary["frames_ok"] == 256
        assert summary["symbol_throughput_per_s"] > 0
        capture = session.finish()
        assert capture.status == "complete"
        assert capture.counters["symbols"] == 256 * 31

    def test_dry_run_discards_the_payload_but_keeps_the_timing(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(SimulatedModemBackend(), config, RunMode.DRY_RUN, CHANNEL)
        session.start()
        assert session.transfer(1, seed=0) is None
        capture = session.finish()
        assert capture.mode == "dry_run"
        assert capture.stage_durations_s["encode"] > 0
        assert capture.stage_durations_s["decode"] > 0
        assert capture.counters["frames"] == 256

    def test_start_refuses_when_preflight_fails(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=4, symbol_rate_hz=1.0e6)
        session = ModemSession(
            SimulatedModemBackend(), config, RunMode.SIMULATION, CHANNEL
        )
        with pytest.raises(RuntimeError, match="preflight failed"):
            session.start()
        assert session.capture.status == "refused"
        assert "depth_against_fade" in (session.capture.abort_reason or "")

    def test_transfer_before_start_raises(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(SimulatedModemBackend(), config)
        with pytest.raises(RuntimeError, match="not started"):
            session.transfer(1)

    def test_transfer_rejects_zero_blocks(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(
            SimulatedModemBackend(), config, RunMode.SIMULATION, CHANNEL
        )
        session.start()
        with pytest.raises(ValueError, match="blocks must be >= 1"):
            session.transfer(0)

    def test_backout_is_idempotent_and_keeps_the_first_reason(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(
            SimulatedModemBackend(), config, RunMode.SIMULATION, CHANNEL
        )
        session.start()
        capture = session.backout("link dropped mid block")
        assert capture.status == "aborted"
        assert capture.abort_reason == "link dropped mid block"
        again = session.backout("something else")
        assert again.abort_reason == "link dropped mid block"
        assert session.backend.is_open() is False

    def test_backout_survives_a_backend_that_raises(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(DeviceModemBackend(), config, RunMode.SIMULATION, CHANNEL)
        capture = session.backout("device never came up")
        assert capture.status == "aborted"

    def test_capture_is_json_serialisable_and_has_no_absolute_path(self) -> None:
        config = ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6)
        session = ModemSession(SimulatedModemBackend(), config, RunMode.DRY_RUN, CHANNEL)
        session.start()
        session.transfer(1)
        text = json.dumps(session.finish().to_dict(), sort_keys=True)
        assert "/home/" not in text
        assert "/Users/" not in text
        json.loads(text)

    def test_environment_record_has_no_paths(self) -> None:
        env = environment_record()
        assert set(env) == {
            "python",
            "implementation",
            "machine",
            "system",
            "numpy",
            "maxsize_bits",
        }
        for value in env.values():
            assert "/" not in value
