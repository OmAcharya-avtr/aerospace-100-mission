"""One contract test suite, parameterised over every backend.

This file is the reason the device backend exists before the device does. Each
test below runs against **both** :class:`SimulatedBackend` and
:class:`DeviceBackend`. Where an operation needs hardware, the shared
expectation is not "it works" but "it either works or raises
``NotImplementedError`` naming what is missing" --- which is a testable contract
and the only one honest to write without a detector on the bench.
"""

from __future__ import annotations

import numpy as np
import pytest

from photoncount.hal import (
    REQUIRED_DESCRIPTION_KEYS,
    Acquisition,
    AcquisitionRequest,
    BackendMode,
    DeviceBackend,
    PhotonCountingBackend,
    SimulatedBackend,
)
from photoncount.simulate import DetectorSpec

SPEC = DetectorSpec(
    dead_time_s=1e-7,
    model="paralyzable",
    afterpulse_probability=0.04,
    afterpulse_mean_delay_s=4e-7,
)


def _simulated() -> PhotonCountingBackend:
    return SimulatedBackend(SPEC, 2e5, np.random.default_rng(1234))


def _device() -> PhotonCountingBackend:
    return DeviceBackend(
        dead_time_s=1e-7,
        dead_time_model="paralyzable",
        afterpulse_probability=0.04,
        afterpulse_mean_delay_s=4e-7,
        max_sustained_rate_hz=3.6e6,
        timestamp_resolution_s=1e-10,
        serial_number="unset",
    )


BACKENDS = [pytest.param(_simulated, id="simulated"), pytest.param(_device, id="device")]


@pytest.fixture(params=BACKENDS)
def backend(request) -> PhotonCountingBackend:
    return request.param()


# --- contract clause 1: describe() works without hardware -------------------


def test_describe_returns_every_required_key(backend):
    desc = backend.describe()
    missing = [k for k in REQUIRED_DESCRIPTION_KEYS if k not in desc]
    assert missing == []


def test_describe_does_not_open_the_backend(backend):
    backend.describe()
    assert backend.is_open is False


def test_describe_simulated_flag_matches_the_class(backend):
    assert backend.describe()["simulated"] is backend.simulated
    assert backend.describe()["hardware_required"] is backend.hardware_required


def test_describe_is_a_copy(backend):
    first = backend.describe()
    first["dead_time_s"] = -999.0
    assert backend.describe()["dead_time_s"] != -999.0


def test_declared_dead_time_is_non_negative(backend):
    assert float(backend.describe()["dead_time_s"]) >= 0.0


# --- contract clause 2: requests are validated before hardware --------------


@pytest.mark.parametrize("bad", ["not-a-request", 3, None])
def test_bad_request_raises_value_error_not_notimplemented(backend, bad):
    with pytest.raises(ValueError):
        backend.acquire(bad, BackendMode.SIMULATION)


def test_bad_mode_raises_value_error_not_notimplemented(backend):
    with pytest.raises(ValueError):
        backend.acquire(AcquisitionRequest(1e-3, 2), "simulation")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0.0, "n_windows": 2},
        {"window_s": -1e-3, "n_windows": 2},
        {"window_s": 1e-3, "n_windows": 0},
        {"window_s": 1e-3, "n_windows": 2, "label": "a/b"},
        {"window_s": 1e-3, "n_windows": 2, "label": ""},
        {"window_s": 1e-3, "n_windows": 2, "label": ".."},
    ],
)
def test_invalid_acquisition_request_is_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        AcquisitionRequest(**kwargs)


def test_request_total_seconds():
    assert AcquisitionRequest(1e-3, 50).total_seconds == pytest.approx(5e-2)


# --- contract clause 3 and 4: open/acquire behaviour ------------------------


def test_acquire_either_works_or_refuses_for_want_of_hardware(backend):
    request = AcquisitionRequest(1e-3, 5, "contract")
    try:
        backend.open()
    except NotImplementedError as exc:
        assert "hardware" in str(exc) or "physical" in str(exc)
        assert backend.hardware_required is True
        return
    try:
        acq = backend.acquire(request, BackendMode.SIMULATION)
    except NotImplementedError as exc:  # pragma: no cover - simulated backend works
        assert "physical" in str(exc) or "hardware" in str(exc)
        return
    finally:
        backend.close()
    assert isinstance(acq, Acquisition)
    assert acq.counts.size == 5
    assert acq.wall_seconds > 0.0


def test_acquire_before_open_raises_runtime_or_notimplemented(backend):
    request = AcquisitionRequest(1e-3, 2, "contract")
    with pytest.raises((RuntimeError, NotImplementedError)):
        backend.acquire(request, BackendMode.SIMULATION)


def test_hardware_operations_refuse_only_when_hardware_is_required(backend):
    """Either the backend implements open/self_test, or it names what is missing."""
    if backend.hardware_required:
        for call in (backend.open, backend.self_test):
            with pytest.raises(NotImplementedError) as info:
                call()
            assert any(w in str(info.value) for w in ("hardware", "physical", "device"))
    else:
        backend.open()
        assert backend.is_open is True
        assert backend.self_test()["passed"] is True
        backend.close()


# --- contract clause 5 and 6: close and context manager ---------------------


def test_close_is_idempotent_and_safe_before_open(backend):
    backend.close()
    backend.close()
    assert backend.is_open is False


def test_context_manager_closes(backend):
    try:
        with backend as entered:
            assert entered is backend
            assert backend.is_open is True
    except NotImplementedError:
        assert backend.hardware_required is True
        return
    assert backend.is_open is False


# --- provenance: a simulated number must never look like a measurement ------


def test_is_measurement_is_false_for_simulated_counts():
    backend = _simulated()
    with backend:
        acq = backend.acquire(AcquisitionRequest(1e-3, 4), BackendMode.SIMULATION)
    assert acq.simulated is True
    assert acq.is_measurement is False


def test_dry_run_discards_counts_but_keeps_real_timing():
    backend = _simulated()
    with backend:
        acq = backend.acquire(AcquisitionRequest(1e-3, 6, "dry"), BackendMode.DRY_RUN)
    assert acq.discarded is True
    assert acq.counts.size == 0
    assert acq.wall_seconds > 0.0
    assert np.isnan(acq.observed_rate_hz)
    assert acq.is_measurement is False


def test_device_backend_never_returns_counts():
    backend = _device()
    with pytest.raises(NotImplementedError):
        backend.acquire(AcquisitionRequest(1e-3, 4, "dev"), BackendMode.SIMULATION)


# --- simulated-backend specifics --------------------------------------------


def test_simulated_backend_records_the_true_rate():
    backend = _simulated()
    with backend:
        acq = backend.acquire(AcquisitionRequest(1e-3, 20))
    assert acq.metadata["incident_rate_hz"] == pytest.approx(2e5)
    # At n*tau = 0.02 with p = 0.04 and t_ap = 4*tau, afterpulsing adds more
    # counts than dead time removes, so the observed rate is ABOVE the true one.
    # Anyone who assumes dead time always lowers the count rate is wrong here.
    assert acq.observed_rate_hz > 2e5
    assert acq.metadata["afterpulse_fraction"] > 0.0


def test_simulated_backend_is_deterministic():
    a = SimulatedBackend(SPEC, 2e5, np.random.default_rng(99))
    b = SimulatedBackend(SPEC, 2e5, np.random.default_rng(99))
    with a, b:
        ca = a.acquire(AcquisitionRequest(1e-3, 10)).counts
        cb = b.acquire(AcquisitionRequest(1e-3, 10)).counts
    assert np.array_equal(ca, cb)


def test_simulated_backend_self_test_passes():
    backend = _simulated()
    assert backend.self_test()["passed"] is True


def test_simulated_backend_max_rate_matches_the_model():
    par = SimulatedBackend(SPEC, 1e5, np.random.default_rng(0))
    assert par.describe()["max_sustained_rate_hz"] == pytest.approx(1.0 / (np.e * 1e-7))
    nonpar = SimulatedBackend(
        DetectorSpec(dead_time_s=1e-7, model="nonparalyzable"), 1e5, np.random.default_rng(0)
    )
    assert nonpar.describe()["max_sustained_rate_hz"] == pytest.approx(1e7)
    ideal = SimulatedBackend(DetectorSpec(), 1e5, np.random.default_rng(0))
    assert ideal.describe()["max_sustained_rate_hz"] == float("inf")


def test_simulated_backend_rejects_bad_construction():
    with pytest.raises(TypeError):
        SimulatedBackend("spec", 1e5, np.random.default_rng(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SimulatedBackend(SPEC, -1.0, np.random.default_rng(0))


def test_fano_factor_needs_two_windows():
    backend = _simulated()
    with backend:
        acq = backend.acquire(AcquisitionRequest(1e-3, 1))
    assert np.isnan(acq.fano_factor)
