"""The headline: by how much energy per inference understates energy per decision.

The factor
----------

``F = (energy per useful decision) / (energy per inference)``

Both numbers are measured on the same run. The numerator is Baseline 3: the
trapezoidal integral of the measured rail power over the whole run, sleep
windows included, divided by the counted accepted decisions. The denominator is
Baseline 1: the mean active rail power times the mean inference latency.

``F`` is reported whatever its value, including a value near 1, which would say
the quoted unit is adequate for this pipeline on this board.

The decomposition
-----------------

Write ``a`` for the counted acceptance rate, ``E_total`` for the whole-run
energy, ``E_active`` for the energy restricted to the active windows, ``T_act``
for the measured active duration, ``N`` for the inference count and ``L`` for
the mean inference latency. With the mean active power defined as the
time-weighted ``P_act = E_active / T_act``:

.. code-block:: text

    F = (E_total / (N*a)) / (P_act * L)
      = (1/a)                  rejection factor
      * (E_total / E_active)   duty-cycle factor
      * (P_act / P_quoted)     nominal-versus-measured active-power error
      * (T_act / (N * L))      within-active accounting factor

The first three are the factors the specification names. The fourth is the
**residual** and it is reported as one, rather than the three being asserted to
account for ``F``.

What the residual is, and why it is not zero. ``N * L`` is the summed inference
latency. ``T_act`` is the measured duration of the merged active windows. They
differ by the time inside an active window that no inference occupied: the
pipeline's own loop overhead between ``t_end`` of one inference and ``t_start``
of the next, and the tail of each active phase after the last inference that
fitted. So the residual is a measurable property of the pipeline's loop, it is
computed here in closed form as ``T_act / (N * L)``, and
:attr:`Decomposition.closure_error` reports how far the product of the four
factors is from ``F``. That last number is a floating-point rounding residual
and nothing else; if it is ever large, one of the four factors is not what this
docstring says it is.

For the ``measured-active`` variant the third factor is exactly 1 by
construction, because the quoted power IS the measured power. It is carried
anyway so that the two variants are reported in the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import (
    VARIANTS,
    EnergyPerInference,
    GroundTruth,
    energy_per_inference,
    ground_truth,
)
from .energy import active_windows, energy_over_windows, whole_run_energy
from .trace import Cell


@dataclass(frozen=True)
class Decomposition:
    """The headline factor and the four factors that account for it.

    Attributes
    ----------
    factor
        ``F``, dimensionless. Greater than 1 means energy per inference
        understates energy per useful decision by that factor.
    rejection_factor
        ``1 / a``.
    duty_cycle_factor
        ``E_total / E_active``.
    rail_voltage_factor
        ``P_act_measured / P_act_quoted``: 1 for the ``measured-active``
        variant, and the nominal-versus-measured active-power error for the
        ``nominal-rail`` variant.
    residual_factor
        ``F`` divided by the product of the three named factors.
    accounting_factor
        ``T_act / (N * L)``, computed independently of ``F``. The residual
        should equal this.
    closure_error
        ``residual_factor / accounting_factor - 1``, dimensionless. A
        floating-point rounding residual; it is reported so that the
        decomposition is checked rather than claimed.
    """

    cell: str
    variant: str
    factor: float
    rejection_factor: float
    duty_cycle_factor: float
    rail_voltage_factor: float
    residual_factor: float
    accounting_factor: float
    closure_error: float
    energy_per_useful_decision_mj: float
    energy_per_inference_mj: float
    acceptance_rate: float
    total_energy_mj: float
    active_energy_mj: float
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
            f"whole-run energy E_total          : {self.total_energy_mj:.6f} mJ",
            f"active-window energy E_active     : {self.active_energy_mj:.6f} mJ",
            f"active duration T_act             : {self.active_duration_s:.6f} s",
            f"mean inference latency L          : {self.mean_latency_s:.9f} s",
            f"energy per inference (baseline 1) : {self.energy_per_inference_mj:.9f} mJ",
            f"energy per useful decision        : {self.energy_per_useful_decision_mj:.6f} mJ",
            f"HEADLINE understatement factor F  : {self.factor:.6f}",
            f"  rejection factor 1/a            : {self.rejection_factor:.6f}",
            f"  duty-cycle factor E_tot/E_act   : {self.duty_cycle_factor:.6f}",
            f"  rail-voltage factor P_m/P_q     : {self.rail_voltage_factor:.6f}",
            f"  residual factor                 : {self.residual_factor:.6f}",
            f"  accounting factor T_act/(N*L)   : {self.accounting_factor:.6f}",
            f"  closure error of the above      : {self.closure_error:.3e}",
        ]


def decompose(cell: Cell, variant: str = "measured-active") -> Decomposition:
    """The headline factor for one cell, with its four factors.

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
    starts, ends = active_windows(decisions)
    active_measured = energy_over_windows(cell.power, starts, ends, column="vdd_in_mw")
    whole = whole_run_energy(cell.power, column="vdd_in_mw")
    factor = truth.energy_per_useful_decision_mj / baseline.energy_mj
    rejection = 1.0 / truth.acceptance_rate
    duty_cycle = whole.energy_mj / active_measured.energy_mj
    rail_voltage = active_measured.mean_power_mw / baseline.mean_active_power_mw
    named = rejection * duty_cycle * rail_voltage
    residual = factor / named
    latency_total = float(np.sum(decisions.latency_s))
    accounting = active_measured.duration_s / latency_total
    return Decomposition(
        cell=cell.label,
        variant=variant,
        factor=factor,
        rejection_factor=rejection,
        duty_cycle_factor=duty_cycle,
        rail_voltage_factor=rail_voltage,
        residual_factor=residual,
        accounting_factor=accounting,
        closure_error=residual / accounting - 1.0,
        energy_per_useful_decision_mj=truth.energy_per_useful_decision_mj,
        energy_per_inference_mj=baseline.energy_mj,
        acceptance_rate=truth.acceptance_rate,
        total_energy_mj=whole.energy_mj,
        active_energy_mj=active_measured.energy_mj,
        active_duration_s=active_measured.duration_s,
        mean_latency_s=baseline.mean_latency_s,
        n_inferences=truth.inferences,
        n_accepted=truth.accepted,
        baseline1=baseline,
        truth=truth,
    )
