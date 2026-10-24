"""The HAL contract: protocols, identity, and the hardware flag."""

from __future__ import annotations

import pytest

from timetransfer.backends import (
    AbsentDriver,
    DeviceBackend,
    GuardedDriver,
    LoopbackDriver,
    RecordedDriver,
    SimulatedBackend,
    SimulationSpec,
    make_backend_pair,
)
from timetransfer.errors import ConfigurationError, DeviceAbsentError, ExchangeLostError
from timetransfer.exchange import Exchange, ExchangeSet
from timetransfer.hal import (
    Backend,
    BackendInfo,
    ExchangeOutcome,
    ExchangeTruth,
    TimestampDriver,
)

from conftest import make_spec


def test_backend_info_validates() -> None:
    with pytest.raises(ConfigurationError, match="backend kind"):
        BackendInfo(kind="emulated", name="x")
    with pytest.raises(ConfigurationError, match="name"):
        BackendInfo(kind="device", name="")


@pytest.mark.parametrize(
    ("kind", "driver", "expected"),
    [
        ("simulated", "", False),
        ("simulated", "loopback", False),
        ("device", "", False),
        ("device", "loopback", False),
        ("device", "recorded", False),
        ("device", "absent", False),
        ("device", "guarded-loopback", False),
        ("device", "ixgbe-ptp", True),
    ],
)
def test_is_hardware_only_for_a_device_with_a_real_driver(
    kind: str, driver: str, expected: bool
) -> None:
    assert BackendInfo(kind=kind, name="n", driver=driver).is_hardware is expected


def test_no_backend_in_this_repository_is_hardware() -> None:
    spec = make_spec()
    sim, dev = make_backend_pair(spec)
    assert not sim.info.is_hardware
    assert not dev.info.is_hardware
    recorded = DeviceBackend(
        RecordedDriver(
            ExchangeSet(
                t1_ns=[0, 1],
                t2_ns=[5, 6],
                t3_ns=[6, 7],
                t4_ns=[11, 12],
                seq=[0, 1],
            )
        )
    )
    assert not recorded.info.is_hardware


def test_exchange_truth_bias_is_half_the_asymmetry() -> None:
    truth = ExchangeTruth(
        offset_s=1e-3, asymmetry_s=8e-4, forward_delay_s=5.4e-3, reverse_delay_s=4.6e-3
    )
    assert truth.bias_s == pytest.approx(4e-4)


def test_exchange_outcome_rejects_inconsistent_states() -> None:
    ex = Exchange(seq=0, t1_ns=0, t2_ns=1, t3_ns=2, t4_ns=3)
    with pytest.raises(ConfigurationError, match="lost outcome carries timestamps"):
        ExchangeOutcome(seq=0, exchange=ex, lost=True)
    with pytest.raises(ConfigurationError, match="carries no timestamps"):
        ExchangeOutcome(seq=0, exchange=None, lost=False)


def test_backends_satisfy_the_backend_protocol() -> None:
    spec = make_spec()
    sim, dev = make_backend_pair(spec)
    assert isinstance(sim, Backend)
    assert isinstance(dev, Backend)


def test_drivers_satisfy_the_driver_protocol() -> None:
    spec = make_spec()
    assert isinstance(LoopbackDriver(spec), TimestampDriver)
    assert isinstance(AbsentDriver(), TimestampDriver)
    assert isinstance(GuardedDriver(LoopbackDriver(spec)), TimestampDriver)


def test_simulated_backend_open_close_is_idempotent() -> None:
    backend = SimulatedBackend(make_spec())
    assert not backend.is_open
    backend.open()
    backend.open()
    assert backend.is_open
    backend.close()
    backend.close()
    assert not backend.is_open


def test_device_backend_refuses_to_exchange_before_open() -> None:
    backend = DeviceBackend(LoopbackDriver(make_spec()))
    with pytest.raises(DeviceAbsentError, match="before open"):
        backend.next_exchange(0)


def test_absent_driver_fails_the_open() -> None:
    backend = DeviceBackend(AbsentDriver(reason="no board on this bench"))
    with pytest.raises(DeviceAbsentError, match="no board on this bench"):
        backend.open()


def test_loopback_driver_refuses_before_open() -> None:
    driver = LoopbackDriver(make_spec())
    with pytest.raises(DeviceAbsentError, match="before open"):
        driver.exchange(0)


def test_loopback_driver_raises_exchange_lost_on_a_dropped_exchange() -> None:
    driver = LoopbackDriver(make_spec(loss_probability=0.999, seed=5))
    driver.open()
    with pytest.raises(ExchangeLostError, match="lost in the link model"):
        for seq in range(50):
            driver.exchange(seq)


def test_device_backend_converts_a_loss_into_a_lost_outcome() -> None:
    backend = DeviceBackend(LoopbackDriver(make_spec(loss_probability=0.999, seed=5)))
    backend.open()
    outcomes = [backend.next_exchange(seq) for seq in range(30)]
    assert any(o.lost for o in outcomes)
    assert backend.lost_count == sum(o.lost for o in outcomes)
    backend.close()


def test_recorded_driver_replays_gaps_as_losses() -> None:
    recorded = ExchangeSet(
        t1_ns=[0, 2_000_000_000],
        t2_ns=[5_000_000, 2_005_000_000],
        t3_ns=[5_100_000, 2_005_100_000],
        t4_ns=[10_200_000, 2_010_200_000],
        seq=[0, 2],
    )
    driver = RecordedDriver(recorded)
    driver.open()
    assert driver.exchange(0)[0] == 0
    with pytest.raises(ExchangeLostError, match="not in the recording"):
        driver.exchange(1)
    assert driver.touch_count == 2


def test_recorded_driver_rejects_an_empty_recording() -> None:
    empty = ExchangeSet(t1_ns=[], t2_ns=[], t3_ns=[], t4_ns=[], seq=[])
    with pytest.raises(ConfigurationError, match="at least one exchange"):
        RecordedDriver(empty)


def test_guarded_driver_counts_and_can_refuse() -> None:
    inner = LoopbackDriver(make_spec())
    driver = GuardedDriver(inner, allow_exchange=False)
    driver.open()
    assert driver.name == "guarded-loopback"
    with pytest.raises(DeviceAbsentError, match="forbidden"):
        driver.exchange(0)
    assert driver.refused_exchanges == 1
    assert driver.touch_count == 0
    assert driver.probe()["guard_allow_exchange"] == "false"


def test_guarded_driver_can_refuse_the_open() -> None:
    driver = GuardedDriver(LoopbackDriver(make_spec()), allow_open=False)
    with pytest.raises(DeviceAbsentError, match="open\\(\\) is forbidden"):
        driver.open()
    assert driver.refused_opens == 1


def test_probe_reports_that_the_stubs_are_not_hardware() -> None:
    spec = make_spec()
    for driver in (LoopbackDriver(spec), AbsentDriver()):
        assert driver.probe()["is_hardware"] == "false"


def test_backend_info_detail_carries_the_declared_parameters() -> None:
    spec = SimulationSpec(period_s=0.5, resolution_ns=8, seed=99)
    info = SimulatedBackend(spec).info
    assert info.detail["period_s"] == "0.5"
    assert info.detail["resolution_ns"] == "8"
    assert info.detail["seed"] == "99"
