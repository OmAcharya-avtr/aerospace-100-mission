"""Named regression tests with literal inputs, one per defect found in this build.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Every counterexample found during this build is pinned
here with its literal inputs and a note saying what went wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.backends import DeviceBackend, GuardedDriver, RecordedDriver
from timetransfer.dryrun import dry_run
from timetransfer.errors import ConfigurationError
from timetransfer.exchange import ExchangeSet
from timetransfer.hal import STUB_DRIVERS, BackendInfo
from timetransfer.skew import fit_least_squares


def _recording(n: int = 12) -> ExchangeSet:
    base = np.arange(n, dtype=np.int64) * 1_000_000_000
    return ExchangeSet(
        t1_ns=base,
        t2_ns=base + 5_000_000,
        t3_ns=base + 5_100_000,
        t4_ns=base + 10_200_000,
        seq=np.arange(n, dtype=np.int64),
        backend_kind="simulated",
    )


def test_regression_guarded_stub_is_not_hardware() -> None:
    """A stub wrapped in ``GuardedDriver`` reported ``is_hardware=True``.

    Found by ``tests/test_dryrun.py::test_dry_run_touches_no_hardware`` during
    this build. ``BackendInfo.is_hardware`` compared the driver name against a
    literal list of stub names that contained ``"loopback"``, ``"recorded"``
    and ``"guarded-loopback"`` but **not** ``"guarded-recorded"``. Dry-run mode
    wraps the recorded driver in a guard, so the dry-run backend declared
    itself hardware and the dry-run result reported 240 hardware operations.
    The property is now that a ``guarded-`` prefix is stripped before the
    comparison.

    Literal inputs: a device backend over ``GuardedDriver(RecordedDriver(...))``.
    """
    driver = GuardedDriver(RecordedDriver(_recording()))
    backend = DeviceBackend(driver, name="dry-run", period_s=1.0, virtual_timebase=True)
    assert driver.name == "guarded-recorded"
    assert backend.info.is_hardware is False
    # And every stub, guarded or not.
    for name in sorted(STUB_DRIVERS):
        assert BackendInfo(kind="device", name="x", driver=name).is_hardware is False
        assert (
            BackendInfo(kind="device", name="x", driver=f"guarded-{name}").is_hardware is False
        )
    # A real driver name is still hardware, guarded or not, so the strip does
    # not create a hole.
    assert BackendInfo(kind="device", name="x", driver="ixgbe-ptp").is_hardware is True
    assert (
        BackendInfo(kind="device", name="x", driver="guarded-ixgbe-ptp").is_hardware is True
    )


def test_regression_dry_run_reports_zero_hardware_operations() -> None:
    """The end-to-end form of the defect above, on the dry-run result itself."""
    result = dry_run(_recording(40), min_exchanges=8)
    assert result.is_hardware is False
    assert result.hardware_operations == 0
    assert result.driver_operations == 40


def test_regression_identical_epochs_are_refused() -> None:
    """A fit over identical epochs returned a zero skew with zero uncertainty.

    Found by a Hypothesis probe during this build: with ``t = [0, 0, 0, 0]``
    the design matrix is rank deficient, ``numpy.linalg.pinv`` returns the
    minimum-norm solution, and the reported covariance has a zero on the skew
    diagonal. The function therefore claimed a skew of exactly zero with an
    uncertainty of exactly zero, which is the most confident possible statement
    about the least identifiable quantity. It is now refused.

    Literal inputs: ``t_s = [0.0, 0.0, 0.0, 0.0]``, ``offset_s = [1e-3] * 4``.
    """
    t = np.zeros(4, dtype=np.float64)
    y = np.full(4, 1.0e-3)
    with pytest.raises(ConfigurationError, match="every epoch is identical"):
        fit_least_squares(t, y, order=1)
    # A span of one nanosecond is enough to be identifiable and must still work.
    t_ok = np.asarray([0.0, 1e-9, 2e-9, 3e-9])
    fit = fit_least_squares(t_ok, y, order=1)
    assert np.isfinite(fit.skew_stderr_s_per_s)


def test_regression_flicker_fm_edf_at_m_one_is_clipped_not_fractional() -> None:
    """The simplified flicker-FM edf formula is below one at ``m = 1`` for every N.

    ``2(N-2)/(2.3N-4.9)`` tends to ``2/2.3 = 0.8696`` and never exceeds one, so
    a chi-squared interval built on it directly would be built on fewer than
    one degree of freedom. The clip to ``1.0`` is deliberate and is pinned here
    with literal N values so it cannot be removed as a rounding detail.
    """
    from timetransfer.allan import equivalent_degrees_of_freedom

    for n in (10, 100, 1000, 100_000):
        assert equivalent_degrees_of_freedom("flicker_fm", n, 1) == 1.0
