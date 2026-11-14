"""The headline: by how much energy per inference understates energy per decision.

The factor
----------

``F = (energy per useful decision) / (energy per inference)``

Both numbers are measured on the same run, and both are shares of the same two
integrals, which is what makes the decomposition exact:

* the numerator is ``E_total / A`` -- the trapezoidal integral of the measured
  rail power over the **run**, ``[t0_mono, t1_mono]``, sleep windows included,
  divided by the counted accepted decisions;
* the denominator is ``E_active / N`` -- the same integral restricted to the
  run's **declared** active phases, divided by the counted inferences.

``F`` is reported whatever its value, including a value near 1, which would say
the quoted unit is adequate for this pipeline on this board.

The decomposition, which is exact
---------------------------------

.. code-block:: text

    F = (E_total / A) / (E_active_quoted / N)
      = (N / A)                          rejection factor, 1/a
      * (E_total / E_active)             duty-cycle factor
      * (E_active / E_active_quoted)     nominal-versus-measured rail error

Those are the three factors the specification names and there is **no residual
term**. The product equals ``F`` identically in exact arithmetic;
:attr:`Decomposition.closure_error` reports how far it is from ``F`` in floating
point, which is a rounding residual and nothing else.

This is a correction. An earlier version of this module integrated the
**sampler's span** rather than the run, and partitioned the run by the merged
inference intervals rather than by the declared phases. Both choices put a
fourth, residual factor into the decomposition -- the ratio of the measured
active span to the summed inference latency -- which quietly absorbed the
difference between what the pipeline declared and what it did. Integrating the
run and partitioning by the declared phases removes that term, and what it was
measuring becomes a separate, reported diagnostic:
:class:`powerenvelope.energy.PhaseOccupancy`, carried on
:attr:`Decomposition.occupancy`, which says how much of the declared active
phase the inferences actually occupied. On the measured campaign that is 0.9997
of the declared time, a shortfall of 10.5 ms per 38-cycle run, because the loop
checks the clock before starting an inference and does not start one that would
run past the boundary.

For the ``measured-active`` variant the third factor is exactly 1 by
construction, because the quoted active energy IS the measured active energy.
It is carried anyway so that the two variants are reported in the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import (
    VARIANTS,
    EnergyPerInference,
    GroundTruth,
    energy_per_inference,
    ground_truth,
)
from .energy import (
    PhaseOccupancy,
    declared_active_energy,
    declared_sleep_energy,
    phase_occupancy,
    run_energy,
)
from .trace import Cell


@dataclass(frozen=True)
class Decomposition:
    """The headline factor and the three factors that account for it exactly.

    Attributes
    ----------
    factor
        ``F``, dimensionless. Greater than 1 means energy per inference
        understates energy per useful decision by that factor.
    rejection_factor
        ``N / A``, which is ``1 / a``.
    duty_cycle_factor
        ``E_total / E_active``, both over the run.
    rail_voltage_factor
        ``E_active_measured / E_active_quoted``: 1 for the ``measured-active``
        variant, and the nominal-versus-measured active-power error for the
        ``nominal-rail`` variant.
    closure_error
        ``product of the three factors / F - 1``. A floating-point rounding
        residual; it is reported so that the decomposition is checked rather
        than claimed.
    occupancy
        The phase-occupancy diagnostic. NOT a factor of the decomposition: see
        this module's docstring for why it used to be one and no longer is.
    partition_residual
        ``(E_active + E_sleep) / E_total - 1``. The declared active and sleep
        phases tile the run's cycles exactly, but the run may extend a few
        milliseconds past the last cycle, and that tail is the whole of this
        residual. Reported per cell by
        ``validation/validate_measured_baselines.py``.
    """

    cell: str
    variant: str
    factor: float
    rejection_factor: float
    duty_cycle_factor: float
    rail_voltage_factor: float
    closure_error: float
    occupancy: PhaseOccupancy
    partition_residual: float
    energy_per_useful_decision_mj: float
    energy_per_inference_mj: float
    acceptance_rate: float
    total_energy_mj: float
    active_energy_mj: float
    sleep_energy_mj: float
    active_duration_s: float
    mean_latency_s: float
    n_inferences: int
    n_accepted: int
    baseline1: EnergyPerInference
    truth: GroundTruth

    def product_of_named_factors(self) -> float:
        """Product of the three factors the specification names."""
        return self.rejection_factor * self.duty_cycle_factor * self.rail_voltage_factor

    def lines(self) -> list[str]:
        """Report lines, one per quantity, for a CLI or a validation output."""
        return [
            f"cell                              : {self.cell}",
            f"variant of energy per inference   : {self.variant}",
            f"inferences counted                : {self.n_inferences}",
            f"useful decisions counted          : {self.n_accepted}",
            f"acceptance rate a                 : {self.acceptance_rate:.6f}",
            f"run energy E_total                : {self.total_energy_mj:.6f} mJ",
            f"declared-active energy E_active   : {self.active_energy_mj:.6f} mJ",
            f"declared-sleep energy E_sleep     : {self.sleep_energy_mj:.6f} mJ",
            f"partition residual                : {self.partition_residual:+.3e}",
            f"active duration                   : {self.active_duration_s:.6f} s",
            f"mean inference latency L          : {self.mean_latency_s:.9f} s",
            f"energy per inference (baseline 1) : {self.energy_per_inference_mj:.9f} mJ",
            f"  spec-literal P_act * L          : {self.baseline1.spec_literal_energy_mj:.9f} mJ",
            f"energy per useful decision        : {self.energy_per_useful_decision_mj:.6f} mJ",
            f"HEADLINE understatement factor F  : {self.factor:.6f}",
            f"  rejection factor N/A            : {self.rejection_factor:.6f}",
            f"  duty-cycle factor E_tot/E_act   : {self.duty_cycle_factor:.6f}",
            f"  rail-voltage factor             : {self.rail_voltage_factor:.6f}",
            f"  closure error of the three      : {self.closure_error:.3e}",
            "DIAGNOSTIC, not a factor:",
            f"  declared active time            : {self.occupancy.declared_s:.6f} s",
            f"  occupied by inferences          : {self.occupancy.occupied_s:.6f} s "
            f"(occupancy {self.occupancy.occupancy:.6f})",
            f"  spanned by merged intervals     : {self.occupancy.spanned_s:.6f} s",
            f"  shortfall / overshoot           : {self.occupancy.shortfall_s * 1e3:.3f} ms"
            f" / {self.occupancy.overshoot_s * 1e3:.3f} ms",
        ]


def decompose(cell: Cell, variant: str = "measured-active") -> Decomposition:
    """The headline factor for one cell, with its three exact factors.

    Parameters
    ----------
    cell
        A cell that ran the pipeline.
    variant
        Which Baseline-1 variant to put in the denominator. See
        :data:`powerenvelope.baselines.VARIANTS`.

    Raises
    ------
    ValueError
        On an unknown variant, or if the cell accepted no decision.
    TraceDataAbsentError
        For the background floor cell.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    decisions = cell.require_decisions()
    truth = ground_truth(cell)
    baseline = energy_per_inference(cell, variant=variant)
    active_measured = declared_active_energy(cell, column="vdd_in_mw")
    active_quoted = declared_active_energy(cell, column=VARIANTS[variant])
    sleep = declared_sleep_energy(cell)
    whole = run_energy(cell, column="vdd_in_mw")
    factor = truth.energy_per_useful_decision_mj / baseline.energy_mj
    rejection = decisions.inferences / truth.accepted
    duty_cycle = whole.energy_mj / active_measured.energy_mj
    rail_voltage = active_measured.energy_mj / active_quoted.energy_mj
    product = rejection * duty_cycle * rail_voltage
    return Decomposition(
        cell=cell.label,
        variant=variant,
        factor=factor,
        rejection_factor=rejection,
        duty_cycle_factor=duty_cycle,
        rail_voltage_factor=rail_voltage,
        closure_error=product / factor - 1.0,
        occupancy=phase_occupancy(cell),
        partition_residual=(active_measured.energy_mj + sleep.energy_mj)
        / whole.energy_mj
        - 1.0,
        energy_per_useful_decision_mj=truth.energy_per_useful_decision_mj,
        energy_per_inference_mj=baseline.energy_mj,
        acceptance_rate=truth.acceptance_rate,
        total_energy_mj=whole.energy_mj,
        active_energy_mj=active_measured.energy_mj,
        sleep_energy_mj=sleep.energy_mj,
        active_duration_s=active_measured.duration_s,
        mean_latency_s=baseline.mean_latency_s,
        n_inferences=truth.inferences,
        n_accepted=truth.accepted,
        baseline1=baseline,
        truth=truth,
    )
