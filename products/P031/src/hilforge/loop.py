"""The fixed-rate HIL loop: four stages, deadline accounting, dry run.

Two clocks, on purpose
----------------------
A HIL harness has two independent notions of time and conflating them is how
a "reproducible" run stops being reproducible.

* The **model clock** is the backend's timebase. The loop advances it by one
  period per iteration (:meth:`Backend.advance`) and reads it to timestamp the
  iteration. It drives the plant. On real hardware it is the device's clock.
* The **measurement clock** times the host's execution of each stage. It never
  touches the data path.

Because the two are separate, a run can be bit-identical in its data path
while its measured latencies differ, and a run with
:attr:`LoopConfig.injected_durations_s` set reads no wall clock at all and is
bit-identical in *both*.

Stages
------
=============  ==========================================================
Stage          Work
=============  ==========================================================
``sense``      read the ``ahrs`` channel
``estimate``   one complementary-filter update of the attitude estimate
``control``    PD torque command from the estimate
``actuate``    write the ``torque`` channel (or rehearse it, in a dry run)
=============  ==========================================================

The complementary filter is

    theta_hat[k] = a (theta_hat[k-1] + omega_meas[k] dt) + (1 - a) theta_meas[k]

with ``a`` in ``[0, 1)`` the gyro weight and ``dt`` the period [s]. It is the
single-pole complementary filter of Higgins, "A Comparison of Complementary
and Kalman Filtering", *IEEE Transactions on Aerospace and Electronic
Systems* AES-11(3):321-325, 1975, §II: high-pass on the integrated rate,
low-pass on the direct angle, the two weights summing to one. Validity: the
gyro bias must be small compared with ``(1-a)/a`` times the angle noise, or
the estimate inherits the bias; this loop makes no attempt to estimate bias.

Deadline accounting
-------------------
Both overrun definitions of :mod:`hilforge.timing` are accumulated as the run
proceeds. The cascade definition is the one that can abort a run, because it
is the one that compounds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .dryrun import DryRunActuator
from .errors import (
    ConfigurationError,
    DeviceDisconnectedError,
    OverrunCascadeError,
    SampleDroppedError,
    TimebaseRegressionError,
    WriteRejectedError,
)
from .hal import WriteAck
from .plant import PDController, PDGains
from .timebase import MonotonicTimebase, Timebase
from .timing import (
    LatencyHistogram,
    MonotonicGuard,
    OverrunAccount,
    PeriodSpec,
    overrun_report,
)

__all__ = [
    "STAGES",
    "HilLoop",
    "IterationRecord",
    "LoopConfig",
    "RunRecord",
]

STAGES: tuple[str, ...] = ("sense", "estimate", "control", "actuate")
_DEFAULT_FRACTIONS = (0.30, 0.20, 0.15, 0.35)


@dataclass(frozen=True)
class LoopConfig:
    """Everything that decides what a run does.

    Attributes
    ----------
    period: PeriodSpec
        Period, deadline and cascade limit.
    n_iterations:
        Number of iterations, >= 1.
    gains:
        PD gains for the control stage.
    theta_cmd_rad:
        Attitude command [rad].
    filter_gyro_weight:
        Complementary-filter gyro weight ``a`` in ``[0, 1)``.
    dry_run:
        If ``True`` every actuator is wrapped in
        :class:`hilforge.dryrun.DryRunActuator` and no write reaches the
        backend.
    on_dropped_sample:
        ``"hold"`` reuses the previous sample and counts the drop;
        ``"abort"`` ends the run with ``SampleDroppedError`` recorded. A drop
        on the first iteration always aborts: there is nothing to hold.
    on_write_rejected:
        ``"hold"`` counts the refusal and leaves the previous command in
        force; ``"abort"`` ends the run.
    on_cascade:
        ``"abort"`` raises :class:`~hilforge.errors.OverrunCascadeError` when
        ``period.cascade_limit`` consecutive cascade overruns occur;
        ``"record"`` only counts them.
    on_disconnect:
        ``"raise"`` (default) re-raises
        :class:`~hilforge.errors.DeviceDisconnectedError` with the partial
        :class:`RunRecord` attached as its ``record`` attribute;
        ``"record"`` returns that partial record instead. The default is
        ``"raise"`` because a returned record looks like a completed run to
        anything that only checks for an exception — including
        :class:`hilforge.deploy.RunGuard`, which would then not recover. That
        was found by ``validation/recovery_abort.py`` and is recorded in
        ``validation/VALIDATION.md`` §5.
    injected_durations_s:
        Optional per-iteration total durations [s]. When given, no wall clock
        is read: stage durations are these totals split by
        ``injected_stage_fractions``, so the entire run including its timing
        is deterministic. Length must equal ``n_iterations``.
    injected_stage_fractions:
        Fractions of the injected total assigned to the four stages; must be
        positive and sum to 1 within 1e-12.
    guard_tolerance_s:
        Backwards-step tolerance of the model-clock
        :class:`~hilforge.timing.MonotonicGuard` [s].
    record_signals:
        Keep per-iteration sample/command arrays in the record. Off for long
        benchmark runs.
    overrun_flagger:
        Optional ``f(recent_durations_s) -> bool`` consulted before each
        iteration. Its flag is recorded, and if ``shed_factor < 1`` the
        ``estimate`` stage's injected duration is multiplied by it on a
        flagged iteration. Only has a timing effect in injected mode.
    shed_factor:
        Multiplier applied to the ``estimate`` stage duration on a flagged
        iteration, in ``(0, 1]``. ``1.0`` disables shedding.
    flagger_history:
        Number of past iteration durations handed to ``overrun_flagger``.
    """

    period: PeriodSpec
    n_iterations: int
    gains: PDGains = field(default_factory=PDGains)
    theta_cmd_rad: float = 0.0
    filter_gyro_weight: float = 0.98
    dry_run: bool = False
    on_dropped_sample: str = "hold"
    on_write_rejected: str = "hold"
    on_cascade: str = "record"
    on_disconnect: str = "raise"
    injected_durations_s: tuple[float, ...] | None = None
    injected_stage_fractions: tuple[float, ...] = _DEFAULT_FRACTIONS
    guard_tolerance_s: float = 0.0
    record_signals: bool = True
    overrun_flagger: Callable[[np.ndarray], bool] | None = None
    shed_factor: float = 1.0
    flagger_history: int = 8

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ConfigurationError(
                f"n_iterations must be >= 1, got {self.n_iterations!r}"
            )
        if not (0.0 <= self.filter_gyro_weight < 1.0):
            raise ConfigurationError(
                f"filter_gyro_weight must be in [0, 1), got {self.filter_gyro_weight!r}"
            )
        for name, value in (
            ("on_dropped_sample", self.on_dropped_sample),
            ("on_write_rejected", self.on_write_rejected),
        ):
            if value not in ("hold", "abort"):
                raise ConfigurationError(
                    f"{name} must be 'hold' or 'abort', got {value!r}"
                )
        if self.on_cascade not in ("abort", "record"):
            raise ConfigurationError(
                f"on_cascade must be 'abort' or 'record', got {self.on_cascade!r}"
            )
        if self.on_disconnect not in ("raise", "record"):
            raise ConfigurationError(
                f"on_disconnect must be 'raise' or 'record', got {self.on_disconnect!r}"
            )
        if self.injected_durations_s is not None:
            if len(self.injected_durations_s) != self.n_iterations:
                raise ConfigurationError(
                    f"injected_durations_s has {len(self.injected_durations_s)} entries "
                    f"but n_iterations is {self.n_iterations}"
                )
            if any((not np.isfinite(d)) or d < 0.0 for d in self.injected_durations_s):
                raise ConfigurationError(
                    "injected_durations_s must all be finite and >= 0 s"
                )
        fr = np.asarray(self.injected_stage_fractions, dtype=np.float64)
        if fr.shape != (len(STAGES),):
            raise ConfigurationError(
                f"injected_stage_fractions must have {len(STAGES)} entries, got {fr.shape}"
            )
        if np.any(fr <= 0.0) or abs(float(fr.sum()) - 1.0) > 1e-12:
            raise ConfigurationError(
                f"injected_stage_fractions must be positive and sum to 1, got "
                f"{self.injected_stage_fractions!r} (sum {float(fr.sum())!r})"
            )
        if not (0.0 < self.shed_factor <= 1.0):
            raise ConfigurationError(
                f"shed_factor must be in (0, 1], got {self.shed_factor!r}"
            )
        if self.flagger_history < 1:
            raise ConfigurationError(
                f"flagger_history must be >= 1, got {self.flagger_history!r}"
            )

    @property
    def deterministic_timing(self) -> bool:
        """``True`` when no wall clock is read, i.e. injected durations given."""
        return self.injected_durations_s is not None


@dataclass
class IterationRecord:
    """One iteration's data and timing.

    All durations are seconds. ``sample`` and ``command`` are present only
    when :attr:`LoopConfig.record_signals` is ``True``.
    """

    index: int
    model_time_s: float
    stage_s: dict[str, float]
    total_s: float
    completion_s: float
    deadline_s: float
    lateness_s: float
    direct_overrun: bool
    cascade_overrun: bool
    flagged: bool
    shed: bool
    dropped: bool
    rejected: bool
    saturated: bool
    theta_hat_rad: float
    sample: np.ndarray | None = None
    command: np.ndarray | None = None
    applied: np.ndarray | None = None


@dataclass
class RunRecord:
    """Everything a run produced, and everything needed to judge it.

    Attributes
    ----------
    backend_kind, backend_name, backend_driver:
        Backend identity, copied from :class:`hilforge.hal.BackendInfo`.
    is_hardware:
        ``False`` for a simulated backend and for a device backend driven by
        a stub. A number from a record with ``is_hardware is False`` is not a
        hardware measurement.
    deterministic_timing:
        ``True`` if the run read no wall clock.
    iterations:
        Per-iteration records.
    overruns:
        :class:`~hilforge.timing.OverrunAccount` over the completed
        iterations.
    stage_histograms:
        One :class:`~hilforge.timing.LatencyHistogram` per stage, plus
        ``"total"``.
    aborted, abort_reason, abort_index:
        Whether the run ended early, why, and at which iteration.
    """

    backend_kind: str
    backend_name: str
    backend_driver: str
    is_hardware: bool
    deterministic_timing: bool
    period_s: float
    deadline_s: float
    n_requested: int
    dry_run: bool
    iterations: list[IterationRecord] = field(default_factory=list)
    overruns: OverrunAccount | None = None
    stage_histograms: dict[str, LatencyHistogram] = field(default_factory=dict)
    dropped_samples: int = 0
    rejected_writes: int = 0
    saturated_writes: int = 0
    writes_issued: int = 0
    rehearsed_writes: int = 0
    flagged_iterations: int = 0
    shed_iterations: int = 0
    aborted: bool = False
    abort_reason: str = ""
    abort_index: int = -1

    @property
    def n_completed(self) -> int:
        """Iterations that ran to completion."""
        return len(self.iterations)

    def durations_s(self) -> np.ndarray:
        """Per-iteration total durations [s]."""
        return np.array([it.total_s for it in self.iterations], dtype=np.float64)

    def stage_durations_s(self, stage: str) -> np.ndarray:
        """Per-iteration durations [s] of one stage."""
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; known: {STAGES}")
        return np.array([it.stage_s[stage] for it in self.iterations], dtype=np.float64)

    def signal_matrix(self) -> np.ndarray:
        """``(n, 5)`` array ``[theta_meas, omega_meas, theta_hat, cmd, applied]``.

        The data path, with no timing in it. This is what parity is checked
        on. Raises if the run did not record signals.
        """
        if not self.iterations or self.iterations[0].sample is None:
            raise ValueError("run did not record signals; set record_signals=True")
        rows = []
        for it in self.iterations:
            assert it.sample is not None and it.command is not None
            applied = it.applied if it.applied is not None else np.array([np.nan])
            rows.append(
                [
                    float(it.sample[0]),
                    float(it.sample[1]),
                    it.theta_hat_rad,
                    float(it.command[0]),
                    float(applied[0]),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def data_digest(self) -> str:
        """SHA-256 over the data path only, as a hex string.

        Covers the signal matrix bytes, the per-iteration flags and the
        counters. Deliberately excludes every timing field, so two runs with
        the same seed on different hosts give the same digest.
        """
        h = hashlib.sha256()
        h.update(self.signal_matrix().tobytes())
        flags = np.array(
            [
                [it.dropped, it.rejected, it.saturated]
                for it in self.iterations
            ],
            dtype=np.uint8,
        )
        h.update(flags.tobytes())
        h.update(
            f"{self.dropped_samples}|{self.rejected_writes}|{self.saturated_writes}|"
            f"{self.writes_issued}|{self.rehearsed_writes}".encode()
        )
        return h.hexdigest()

    def full_digest(self) -> str:
        """SHA-256 over the data path **and** the timing fields.

        Only reproducible when :attr:`deterministic_timing` is ``True``.
        """
        h = hashlib.sha256()
        h.update(self.data_digest().encode())
        timing = np.array(
            [[it.stage_s[s] for s in STAGES] + [it.total_s, it.completion_s]
             for it in self.iterations],
            dtype=np.float64,
        )
        h.update(timing.tobytes())
        return h.hexdigest()

    def summary(self) -> dict[str, object]:
        """Compact dictionary for printing and for the benchmark record."""
        out: dict[str, object] = {
            "backend": self.backend_name,
            "driver": self.backend_driver,
            "is_hardware": self.is_hardware,
            "deterministic_timing": self.deterministic_timing,
            "dry_run": self.dry_run,
            "period_s": self.period_s,
            "n_requested": self.n_requested,
            "n_completed": self.n_completed,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "abort_index": self.abort_index,
            "dropped_samples": self.dropped_samples,
            "rejected_writes": self.rejected_writes,
            "saturated_writes": self.saturated_writes,
            "writes_issued": self.writes_issued,
            "rehearsed_writes": self.rehearsed_writes,
            "flagged_iterations": self.flagged_iterations,
            "shed_iterations": self.shed_iterations,
        }
        if self.overruns is not None:
            out["direct_overruns"] = self.overruns.direct_count
            out["cascade_overruns"] = self.overruns.cascade_count
            out["max_consecutive_cascade"] = self.overruns.max_consecutive_cascade
        if "total" in self.stage_histograms and len(self.stage_histograms["total"]):
            out.update(
                {f"total_{k}": v for k, v in self.stage_histograms["total"].summary().items()}
            )
        return out


class HilLoop:
    """A fixed-rate loop over a HAL backend.

    Parameters
    ----------
    backend:
        An open or unopened backend. :meth:`run` opens it if needed and leaves
        it in the state it found it.
    config:
        See :class:`LoopConfig`.
    measurement_clock:
        Timebase used to time stages. Defaults to
        :class:`~hilforge.timebase.MonotonicTimebase`. Unused when
        ``config.injected_durations_s`` is set.
    """

    def __init__(
        self,
        backend,
        config: LoopConfig,
        *,
        measurement_clock: Timebase | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._clock = measurement_clock or MonotonicTimebase()

    def run(self) -> RunRecord:
        """Execute the configured number of iterations and return the record.

        Raises
        ------
        OverrunCascadeError
            If ``on_cascade="abort"`` and the cascade limit is reached.
        SampleDroppedError, WriteRejectedError
            If the corresponding policy is ``"abort"``.
        DeviceDisconnectedError
            If ``on_disconnect="raise"`` (the default).
        TimebaseRegressionError
            If the model clock went backwards by more than
            ``guard_tolerance_s``. There is no policy for this: a loop that
            continues on a backwards clock under-counts its own overruns.

        Every exception raised here carries the partial :class:`RunRecord` as
        its ``record`` attribute, so an aborted run is still inspectable.
        """
        cfg = self.config
        spec = cfg.period
        was_open = self.backend.is_open
        if not was_open:
            self.backend.open()
        info = self.backend.info
        record = RunRecord(
            backend_kind=info.kind,
            backend_name=info.name,
            backend_driver=info.driver,
            is_hardware=info.is_hardware,
            deterministic_timing=cfg.deterministic_timing,
            period_s=spec.period_s,
            deadline_s=spec.effective_deadline_s,
            n_requested=cfg.n_iterations,
            dry_run=cfg.dry_run,
        )
        record.stage_histograms = {s: LatencyHistogram() for s in STAGES}
        record.stage_histograms["total"] = LatencyHistogram()

        sensor = self.backend.sensors["ahrs"]
        raw_actuator = self.backend.actuators["torque"]
        actuator = DryRunActuator(raw_actuator) if cfg.dry_run else raw_actuator
        controller = PDController(cfg.gains, theta_cmd_rad=cfg.theta_cmd_rad)
        guard = MonotonicGuard(tolerance_s=cfg.guard_tolerance_s)

        dt = spec.period_s
        a = cfg.filter_gyro_weight
        theta_hat = 0.0
        theta_hat_init = False
        last_sample: np.ndarray | None = None
        prev_completion = 0.0
        consecutive = 0
        durations: list[float] = []

        try:
            for i in range(cfg.n_iterations):
                flagged = False
                if cfg.overrun_flagger is not None and durations:
                    hist = np.asarray(durations[-cfg.flagger_history :], dtype=np.float64)
                    flagged = bool(cfg.overrun_flagger(hist))
                shed = bool(flagged and cfg.shed_factor < 1.0)
                if flagged:
                    record.flagged_iterations += 1
                if shed:
                    record.shed_iterations += 1

                model_time = guard.check(self.backend.timebase.now())
                budget = self._stage_budget(i, shed)
                stage_s: dict[str, float] = {}

                # --- sense -------------------------------------------------
                dropped = False
                t0 = self._mark()
                try:
                    sample = sensor.read()
                    last_sample = sample
                except SampleDroppedError:
                    dropped = True
                    record.dropped_samples += 1
                    if cfg.on_dropped_sample == "abort" or last_sample is None:
                        record.aborted = True
                        record.abort_index = i
                        record.abort_reason = (
                            "sample dropped on first iteration, nothing to hold"
                            if last_sample is None
                            else "sample dropped, policy=abort"
                        )
                        raise
                    sample = last_sample
                stage_s["sense"] = self._elapsed(t0, budget["sense"])

                # --- estimate ----------------------------------------------
                t0 = self._mark()
                if not theta_hat_init:
                    theta_hat = float(sample[0])
                    theta_hat_init = True
                else:
                    theta_hat = a * (theta_hat + float(sample[1]) * dt) + (1.0 - a) * float(
                        sample[0]
                    )
                stage_s["estimate"] = self._elapsed(t0, budget["estimate"])

                # --- control -----------------------------------------------
                t0 = self._mark()
                command = controller.command(np.array([theta_hat, sample[1]]))
                command = np.clip(command, actuator.spec.lower, actuator.spec.upper)
                stage_s["control"] = self._elapsed(t0, budget["control"])

                # --- actuate -----------------------------------------------
                rejected = False
                ack: WriteAck | None = None
                t0 = self._mark()
                try:
                    ack = actuator.write(command)
                except WriteRejectedError:
                    rejected = True
                    record.rejected_writes += 1
                    if cfg.on_write_rejected == "abort":
                        record.aborted = True
                        record.abort_index = i
                        record.abort_reason = "actuator write rejected, policy=abort"
                        raise
                stage_s["actuate"] = self._elapsed(t0, budget["actuate"])

                if ack is not None:
                    if cfg.dry_run:
                        record.rehearsed_writes += 1
                    else:
                        record.writes_issued += 1
                    if ack.saturated:
                        record.saturated_writes += 1

                total = float(sum(stage_s.values()))
                durations.append(total)
                start = max(spec.release_time_s(i), prev_completion)
                completion = start + total
                deadline = spec.absolute_deadline_s(i)
                cascade = completion > deadline
                prev_completion = completion

                record.iterations.append(
                    IterationRecord(
                        index=i,
                        model_time_s=model_time,
                        stage_s=stage_s,
                        total_s=total,
                        completion_s=completion,
                        deadline_s=deadline,
                        lateness_s=completion - deadline,
                        direct_overrun=total > spec.effective_deadline_s,
                        cascade_overrun=cascade,
                        flagged=flagged,
                        shed=shed,
                        dropped=dropped,
                        rejected=rejected,
                        saturated=bool(ack.saturated) if ack is not None else False,
                        theta_hat_rad=theta_hat,
                        sample=sample.copy() if cfg.record_signals else None,
                        command=command.copy() if cfg.record_signals else None,
                        applied=(
                            ack.applied.copy()
                            if (cfg.record_signals and ack is not None)
                            else None
                        ),
                    )
                )
                for name, value in stage_s.items():
                    record.stage_histograms[name].add(value)
                record.stage_histograms["total"].add(total)

                consecutive = consecutive + 1 if cascade else 0
                if (
                    spec.cascade_limit
                    and consecutive >= spec.cascade_limit
                    and cfg.on_cascade == "abort"
                ):
                    record.aborted = True
                    record.abort_index = i
                    record.abort_reason = (
                        f"{consecutive} consecutive cascade overruns "
                        f"(limit {spec.cascade_limit})"
                    )
                    raise OverrunCascadeError(
                        record.abort_reason,
                        run_length=consecutive,
                        first_index=i - consecutive + 1,
                    )

                self.backend.advance(dt)
        except DeviceDisconnectedError as exc:
            record.aborted = True
            if record.abort_index < 0:
                record.abort_index = len(record.iterations)
            record.abort_reason = "device disconnected mid-run"
            record.overruns = overrun_report(
                record.durations_s(), spec.period_s, deadline_s=spec.deadline_s
            )
            if not was_open:
                self.backend.close()
            if cfg.on_disconnect == "raise":
                exc.record = record  # type: ignore[attr-defined]
                raise
            return record
        except (
            SampleDroppedError,
            WriteRejectedError,
            OverrunCascadeError,
            TimebaseRegressionError,
        ) as exc:
            record.overruns = overrun_report(
                record.durations_s(), spec.period_s, deadline_s=spec.deadline_s
            )
            if not was_open:
                self.backend.close()
            exc.record = record  # type: ignore[attr-defined]
            raise
        record.overruns = overrun_report(
            record.durations_s(), spec.period_s, deadline_s=spec.deadline_s
        )
        if not was_open:
            self.backend.close()
        return record

    # -- timing helpers -------------------------------------------------
    def _stage_budget(self, index: int, shed: bool) -> dict[str, float]:
        """Injected per-stage durations [s] for iteration ``index``.

        Empty values when timing is measured rather than injected.
        """
        cfg = self.config
        if cfg.injected_durations_s is None:
            return dict.fromkeys(STAGES, -1.0)
        total = float(cfg.injected_durations_s[index])
        budget = {
            s: total * f for s, f in zip(STAGES, cfg.injected_stage_fractions, strict=True)
        }
        if shed:
            budget["estimate"] *= cfg.shed_factor
        return budget

    def _mark(self) -> float:
        """Measurement-clock timestamp [s], or 0.0 in injected mode."""
        if self.config.injected_durations_s is None:
            return self._clock.now()
        return 0.0

    def _elapsed(self, t0: float, injected: float) -> float:
        """Stage duration [s]: measured, or the injected value."""
        if self.config.injected_durations_s is None:
            return max(self._clock.now() - t0, 0.0)
        return float(injected)
