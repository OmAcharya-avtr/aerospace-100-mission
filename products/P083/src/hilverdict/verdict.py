"""The run-level verdict: is this hardware-in-the-loop result worth anything?

The order matters
-----------------
1. **The clock first.** A duration measured across a device-clock reset is not
   biased, it is meaningless, so a run with a detected reset is
   :data:`UNDECIDABLE` and nothing else is computed for it. A run whose clock
   drifts beyond the declared allowance has its durations corrected before
   anything reads them, because the drift changes which iterations appear to
   have missed at all.
2. **The declared conditions next.** Actuator saturation, incidence beyond the
   declared validity limit, or mean utilisation at or above 1 each break a step
   of the certificate's derivation, so such a run is :data:`UNDECIDABLE` rather
   than analysed with a bound that does not apply to it.
3. **The certificate.** Every miss whose closed-form bound falls below the
   tolerance is dismissed, with a proof.
4. **Re-simulation, on the residual only.** The misses the certificate leaves
   undecided, and only those, are paid for.
5. **The aggregate.** Even when every miss is dismissed individually, the
   superposition bound of equation (15) is reported, because "no single miss
   mattered" and "the misses together did not matter" are different claims.

What the verdict is not
-----------------------
:data:`TIMING_FAITHFUL` is an evidence statement about the declared plant,
the declared controller, the declared metric and the declared tolerance. It is
not a certification, not an airworthiness statement and not a claim about a
real rig. The words used for it are deliberately drab.
"""

from __future__ import annotations

from dataclasses import dataclass

from .certificate import RunCertificate, certify_run
from .clock import CLOCK_DRIFTING, CLOCK_RESET, ClockEstimate, drift_corrected_log, estimate_clock
from .resimulate import Resimulation, resimulate_run
from .timeline import RunLog, miss_events, overrun_counts, utilisation

__all__ = [
    "NOT_TIMING_FAITHFUL",
    "RunVerdict",
    "TIMING_FAITHFUL",
    "UNDECIDABLE",
    "UNDECIDED_RUN",
    "run_verdict",
]

#: No miss could have moved the trajectory past the declared tolerance, and
#: the timing trace itself is believable. An evidence statement, not a
#: certification.
TIMING_FAITHFUL = "TIMING_FAITHFUL"

#: At least one miss measurably moved the trajectory past the tolerance.
NOT_TIMING_FAITHFUL = "NOT_TIMING_FAITHFUL"

#: Misses remain that the certificate could not settle and that were not
#: re-simulated. The run is not condemned; it is unfinished.
UNDECIDED_RUN = "UNDECIDED"

#: The timing trace or the run itself is outside what any method here covers.
UNDECIDABLE = "UNDECIDABLE"


@dataclass(frozen=True)
class RunVerdict:
    """The whole answer for one run.

    Attributes
    ----------
    label
        The run's declared label.
    verdict
        One of the four module-level constants.
    clock
        What the clock check found.
    certificate
        The run certificate, or ``None`` when the clock made the run
        undecidable before it was computed.
    resimulations
        Re-simulated answers for the residual undecided misses.
    reasons
        Human-readable reasons, in the order they were found. Always populated
        for a non-faithful or undecidable verdict, and carries the drift
        correction note when one was applied.
    misses
        Number of recorded misses after any drift correction.
    dismissed
        Number of misses the certificate disposed of in closed form.
    resimulated
        Number of misses re-simulated.
    consequential
        Iteration indices of misses that measurably mattered.
    aggregate_bound
        Equation (15), the superposition bound over every miss.
    aggregate_dismissed
        True when (15) itself falls below the tolerance, which is the stronger
        statement that the misses together cannot have mattered.
    utilisation
        Mean execution time over the period.
    cascade_ratio
        Cascade overruns divided by direct overruns. Reported because the
        non-cascading miss model is a declaration, and a large ratio says the
        run is further from that model than the verdict's wording suggests.
    """

    label: str
    verdict: str
    clock: ClockEstimate
    certificate: RunCertificate | None
    resimulations: tuple[Resimulation, ...]
    reasons: tuple[str, ...]
    misses: int
    dismissed: int
    resimulated: int
    consequential: tuple[int, ...]
    aggregate_bound: float
    aggregate_dismissed: bool
    utilisation: float
    cascade_ratio: float

    @property
    def faithful(self) -> bool:
        """True only for :data:`TIMING_FAITHFUL`."""
        return self.verdict == TIMING_FAITHFUL

    @property
    def resimulation_saving(self) -> float:
        """Fraction of misses the certificate saved a re-simulation on."""
        if not self.misses:
            return 1.0
        return self.dismissed / self.misses


def run_verdict(
    log: RunLog,
    tolerance: float = 1.0,
    *,
    resimulate_undecided: bool = True,
    allowance_ppm: float = 100.0,
    correct_drift: bool = True,
) -> RunVerdict:
    """Decide whether ``log`` is timing-faithful, in the order above."""
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    clock = estimate_clock(log.host_time_s, log.device_time_s, allowance_ppm)
    reasons: list[str] = []
    if clock.verdict == CLOCK_RESET:
        reasons.append(
            f"the device clock reset at sample(s) {list(clock.resets)}; durations "
            "spanning a reset are meaningless, not merely biased, so no timing "
            "verdict can be taken on this run"
        )
        counts = overrun_counts(log)
        return RunVerdict(
            label=log.label,
            verdict=UNDECIDABLE,
            clock=clock,
            certificate=None,
            resimulations=(),
            reasons=tuple(reasons),
            misses=len(miss_events(log)),
            dismissed=0,
            resimulated=0,
            consequential=(),
            aggregate_bound=float("nan"),
            aggregate_dismissed=False,
            utilisation=utilisation(log),
            cascade_ratio=counts.ratio,
        )
    working = log
    if clock.verdict == CLOCK_DRIFTING and correct_drift:
        working = drift_corrected_log(log, clock.drift_ppm)
        reasons.append(
            f"the device clock drifts by {clock.drift_ppm:.4f} ppm, beyond the "
            f"declared allowance of {allowance_ppm:.4f} ppm; the durations were "
            "divided by 1 + rho before the misses were extracted"
        )
    counts = overrun_counts(working)
    certificate = certify_run(working, tolerance)
    if certificate.refusals:
        reasons.extend(certificate.refusals)
        return RunVerdict(
            label=log.label,
            verdict=UNDECIDABLE,
            clock=clock,
            certificate=certificate,
            resimulations=(),
            reasons=tuple(reasons),
            misses=len(certificate.certificates),
            dismissed=0,
            resimulated=0,
            consequential=(),
            aggregate_bound=certificate.aggregate_bound,
            aggregate_dismissed=certificate.aggregate_bound < tolerance,
            utilisation=utilisation(working),
            cascade_ratio=counts.ratio,
        )
    undecided = tuple(c.event.index for c in certificate.undecided)
    resimulations: tuple[Resimulation, ...] = ()
    if undecided and resimulate_undecided:
        resimulations = resimulate_run(working, tolerance, subset=undecided)
    consequential = tuple(
        result.event.index for result in resimulations if result.consequential
    )
    if consequential:
        verdict = NOT_TIMING_FAITHFUL
        reasons.append(
            f"{len(consequential)} of {len(certificate.certificates)} misses "
            f"moved the trajectory past the declared tolerance; iterations "
            f"{list(consequential)}"
        )
    elif undecided and not resimulate_undecided:
        verdict = UNDECIDED_RUN
        reasons.append(
            f"{len(undecided)} misses were left undecided by the certificate and "
            "were not re-simulated"
        )
    else:
        verdict = TIMING_FAITHFUL
    return RunVerdict(
        label=log.label,
        verdict=verdict,
        clock=clock,
        certificate=certificate,
        resimulations=resimulations,
        reasons=tuple(reasons),
        misses=len(certificate.certificates),
        dismissed=len(certificate.dismissed),
        resimulated=len(resimulations),
        consequential=consequential,
        aggregate_bound=certificate.aggregate_bound,
        aggregate_dismissed=certificate.aggregate_bound < tolerance,
        utilisation=utilisation(working),
        cascade_ratio=counts.ratio,
    )
