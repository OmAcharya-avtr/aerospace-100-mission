"""Inject each of the six failure modes and show what the harness does.

Produces ``screenshots/failure_modes.png``: how far each run got before it
aborted, which fault signalled, whether recovery verified, and — for the
cascade case — the compounding lateness that defines it.

Run: ``python examples/failure_modes.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hilforge.backends import make_backend_pair
from hilforge.backends.device import AbsentDriver, DeviceBackend
from hilforge.backends.stubs import (
    DisconnectingDriver,
    DroppingDriver,
    RejectingDriver,
    SteppingBackDriver,
)
from hilforge.deploy import RunGuard, snapshot_backend
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
N = 200
SEED = 20261004
FAULT_AT = 60
OUT = ROOT / "screenshots" / "failure_modes.png"


def _cfg(**kwargs) -> LoopConfig:
    kwargs.setdefault("injected_durations_s", tuple(np.full(N, 0.004)))
    kwargs.setdefault("period", PeriodSpec(period_s=PERIOD))
    return LoopConfig(n_iterations=N, **kwargs)


def _run(backend, cfg):
    """Run under a guard; return (fault name, completed, recovered)."""
    backend.open()
    snapshot_backend(backend)
    guard = RunGuard(backend)
    fault = "none"
    completed = N
    try:
        with guard:
            record = HilLoop(backend, cfg).run()
            completed = record.n_completed
            if record.aborted:
                fault = record.abort_reason
    except Exception as exc:  # noqa: BLE001 - the case is the exception
        fault = type(exc).__name__
        record = getattr(exc, "record", None)
        completed = record.n_completed if record is not None else 0
    backend.close()
    return fault, completed, guard.recovered


def main() -> int:
    cases = []

    backend = DeviceBackend(AbsentDriver(detail="nothing on the bus"))
    try:
        backend.open()
        fault = "none"
    except Exception as exc:  # noqa: BLE001
        fault = type(exc).__name__
    cases.append(("device absent", fault, 0, None))

    for label, driver, cfg in (
        (
            "disconnect mid-run",
            DisconnectingDriver(seed=SEED, fail_at_read=FAULT_AT),
            _cfg(),
        ),
        (
            "sample dropped",
            DroppingDriver(seed=SEED, drop_at_reads=(FAULT_AT,)),
            _cfg(on_dropped_sample="abort"),
        ),
        (
            "write rejected",
            RejectingDriver(seed=SEED, reject_at_writes=(FAULT_AT,)),
            _cfg(on_write_rejected="abort"),
        ),
        (
            "timebase steps back",
            SteppingBackDriver(seed=SEED, step_back_at_read=FAULT_AT, step_back_s=0.05),
            _cfg(),
        ),
    ):
        fault, completed, recovered = _run(
            DeviceBackend(driver, sample_dt_s=PERIOD), cfg
        )
        cases.append((label, fault, completed, recovered))

    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    cascade_durations = np.full(N, 0.004)
    cascade_durations[FAULT_AT:] = 1.25 * PERIOD
    cascade_cfg = _cfg(
        period=PeriodSpec(period_s=PERIOD, cascade_limit=5),
        injected_durations_s=tuple(cascade_durations),
        on_cascade="abort",
    )
    fault, completed, recovered = _run(sim, cascade_cfg)
    cases.append(("overrun cascade", fault, completed, recovered))

    print(f"{'failure mode':<22} {'signal':<28} {'completed':>10} {'recovered':>10}")
    for label, fault, completed, recovered in cases:
        print(f"{label:<22} {fault:<28} {completed:>10} "
              f"{'n/a' if recovered is None else str(recovered):>10}")

    # The lateness profile of the cascade case, for the right-hand panel.
    sim2, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    record = HilLoop(
        sim2,
        _cfg(
            period=PeriodSpec(period_s=PERIOD),
            injected_durations_s=tuple(cascade_durations),
        ),
    ).run()
    lateness = np.asarray(record.overruns.lateness_s) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    fig.suptitle(
        "HilForge: the six failure modes, injected at the driver and accounted by the loop",
        fontsize=12,
    )

    ax = axes[0]
    labels = [c[0] for c in cases]
    completed = [c[2] for c in cases]
    colours = [
        "#2f6f4e" if c[3] else ("#9a7b2f" if c[3] is None else "#b3412c") for c in cases
    ]
    bars = ax.barh(labels, completed, color=colours)
    ax.axvline(FAULT_AT, color="k", ls="--", lw=1.1,
               label=f"fault injected at iteration {FAULT_AT}")
    ax.axvline(N, color="k", ls=":", lw=1.1, label=f"requested {N} iterations")
    for bar, case in zip(bars, cases, strict=True):
        tag = "recovery verified" if case[3] else ("no run" if case[3] is None else "NOT recovered")
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2, tag,
                va="center", fontsize=8)
    ax.set_xlabel("iterations completed before the abort")
    ax.set_xlim(0, N * 1.45)
    ax.set_title("each fault stops the run; the guard puts the rig back")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    idx = np.arange(lateness.size)
    ax.fill_between(idx, 0, np.maximum(lateness, 0), color="#b3412c", alpha=0.8)
    ax.fill_between(idx, np.minimum(lateness, 0), 0, color="#44618c", alpha=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(FAULT_AT, color="k", ls="--", lw=1.1,
               label=f"durations jump to 1.25 T at {FAULT_AT}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("lateness c[i] − D[i] [ms]")
    ax.set_title(
        "the cascade: once lateness carries forward it grows without bound\n"
        f"cascade overruns = {record.overruns.cascade_count}, "
        f"direct = {record.overruns.direct_count}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print()
    print(f"figure: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
