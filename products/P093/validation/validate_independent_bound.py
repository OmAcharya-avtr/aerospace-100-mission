"""Bounds derived from mathematics this product's integrator does not use.

INPUT ORIGIN: SYNTHETIC. The same brackets are applied to the measured campaign
by ``validate_measured_headline.py``, which refuses until the traces exist.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot: a property test that shares an expression with the implementation
passes on exactly the cases the expression gets wrong. An earlier product of
this mission had 777 tests, property tests included, and none of them could
catch catastrophic cancellation in its closed form, because every test compared
the series against expressions sharing the cancellation.

Four bounds, four different arguments, none of them a trapezoid:

1. the left and right **Riemann sums**, whose mean the trapezoid is;
2. the **mean value theorem for integrals**, which knows about no quadrature
   rule at all;
3. **positivity of the sleep-window power**, which bounds the duty-cycle factor
   from below using only the measured floor minimum and the sleep duration;
4. **extreme values of the measured power and latency**, which bound the
   headline factor from below without any integral, any acceptance rate as a
   multiplier, or any part of the decomposition.

A fifth, independent, number is reported beside them: the **quantiser noise
floor**. A uniform quantiser of width ``q`` has standard deviation
``q / sqrt(12)``; the rail current quantises at 8 mA, which at the measured bus
voltage is a power quantum of about 40 mW. That is a statement about the
instrument and not about this code, and it is the figure that actually limits
every energy number here.

Every slack **follows the conditioning of its own expression** and has no
absolute floor. Section 5 demonstrates why, by showing that the floored form
admits a value seven orders of magnitude outside the bracket it is supposed to
guard.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.bounds import (  # noqa: E402
    Bracket,
    duty_cycle_lower_bracket,
    extreme_value_bracket,
    headline_lower_bracket,
    riemann_bracket,
    slack_from_conditioning,
)
from powerenvelope.decomposition import decompose  # noqa: E402
from powerenvelope.energy import active_windows, cell_energy, trapezoid_energy_mj  # noqa: E402
from powerenvelope.predictor import target_quantisation_sd  # noqa: E402
from powerenvelope.schema import CURRENT_LSB_MA  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_independent_bound_output.txt")
    out.rule("INDEPENDENT BOUNDS -- brackets from other mathematics")
    out("INPUT ORIGIN: SYNTHETIC. The measured application is in validate_measured_headline.py.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    campaign = _bootstrap.synthetic_campaign()

    out.rule("1  the Riemann bracket, whose mean the trapezoid is")
    out(f"  {'cell':<22}{'trapezoid mJ':>15}{'width mJ':>13}{'slack mJ':>12}{'inside':>8}")
    out("  " + "-" * 70)
    riemann_ok = True
    for label, cell in sorted(campaign.cells.items()):
        value = trapezoid_energy_mj(cell.power.t, cell.power.vdd_in_mw)
        bracket = riemann_bracket(cell.power.t, cell.power.vdd_in_mw)
        inside = bracket.contains(value)
        riemann_ok &= inside
        out(
            f"  {label:<22}{value:>15.6f}{bracket.width:>13.6f}{bracket.slack:>12.3e}"
            f"{str(inside):>8}"
        )
    out.check(
        "every cell's trapezoidal energy lies inside its Riemann bracket",
        riemann_ok,
        f"{len(campaign)} cell(s)",
    )

    out.rule("2  the mean-value bracket, which uses no quadrature rule")
    out(f"  {'cell':<22}{'low mJ':>14}{'trapezoid mJ':>15}{'high mJ':>14}{'inside':>8}")
    out("  " + "-" * 73)
    mean_value_ok = True
    for label, cell in sorted(campaign.cells.items()):
        value = trapezoid_energy_mj(cell.power.t, cell.power.vdd_in_mw)
        bracket = extreme_value_bracket(cell.power.t, cell.power.vdd_in_mw)
        inside = bracket.contains(value)
        mean_value_ok &= inside
        out(
            f"  {label:<22}{bracket.low:>14.3f}{value:>15.3f}{bracket.high:>14.3f}"
            f"{str(inside):>8}"
        )
    out.check(
        "every cell's trapezoidal energy lies inside its mean-value bracket",
        mean_value_ok,
        "wide by construction; a value outside it is a mistake about which samples or "
        "which span were integrated, not a quadrature disagreement",
    )

    out.rule("3  the duty-cycle factor against the sleep window's own power")
    floor_min = float(campaign.require("floor").power.vdd_in_mw.min())
    out(f"  measured floor minimum: {floor_min:.6f} mW")
    out(f"  {'cell':<22}{'bound':>12}{'measured':>12}{'margin':>12}{'inside':>8}")
    out("  " + "-" * 68)
    duty_ok = True
    for cell in campaign.pipeline_cells():
        whole, active = cell_energy(cell)
        sleep_s = max(0.0, whole.duration_s - active.duration_s)
        bracket = duty_cycle_lower_bracket(active.energy_mj, floor_min, sleep_s)
        measured = whole.energy_mj / active.energy_mj
        inside = bracket.contains(measured)
        duty_ok &= inside
        out(
            f"  {cell.label:<22}{bracket.low:>12.6f}{measured:>12.6f}"
            f"{measured - bracket.low:>12.3e}{str(inside):>8}"
        )
    out.check(
        "every cell's duty-cycle factor satisfies the sleep-power bound",
        duty_ok,
        f"{len(campaign.pipeline_cells())} pipeline cell(s)",
    )
    out(
        "  THIS BOUND HOLDS ON SYNTHETIC DATA AND FAILS ON THE MEASURED CAMPAIGN, on "
        "nine of nineteen cells. Its premise is that a sleep window draws at least the "
        "minimum power ever seen on the no-pipeline floor run, and on the measured "
        "campaign that premise is false: the run order was not randomised and the "
        "board's foreign load fell throughout, so the floor was measured at the highest "
        "load of the campaign. validate_measured_headline.py section 6 publishes the "
        "violation with the per-cell table. The bound is not weakened here to make the "
        "measured case pass."
    )

    out.rule("4  the headline factor against extreme values alone")
    out(f"  {'cell':<22}{'bound':>12}{'measured F':>13}{'ratio':>9}{'inside':>8}")
    out("  " + "-" * 66)
    headline_ok = True
    ratios = []
    for cell in campaign.pipeline_cells():
        result = decompose(cell)
        inside_mask = np.zeros(cell.power.t.size, dtype=bool)
        for lo, hi in zip(*active_windows(cell.require_decisions()), strict=True):
            inside_mask |= (cell.power.t >= lo) & (cell.power.t <= hi)
        bracket = headline_lower_bracket(
            run_duration_s=float(cell.power.t[-1] - cell.power.t[0]),
            min_power_mw=float(cell.power.vdd_in_mw.min()),
            max_active_power_mw=float(cell.power.vdd_in_mw[inside_mask].max()),
            max_latency_s=float(cell.require_decisions().latency_s.max()),
            n_accepted=result.n_accepted,
        )
        inside = bracket.contains(result.factor)
        headline_ok &= inside
        ratios.append(result.factor / bracket.low)
        out(
            f"  {cell.label:<22}{bracket.low:>12.6f}{result.factor:>13.6f}"
            f"{result.factor / bracket.low:>9.3f}{str(inside):>8}"
        )
    out.check(
        "every cell's headline factor satisfies the extreme-value bound",
        headline_ok,
        f"{len(ratios)} cell(s)",
    )
    out(
        f"  BOUND TIGHTNESS, stated rather than left to be worked out: the measured "
        f"factor exceeds the bound by {min(ratios):.2f} to {max(ratios):.2f} times. "
        "The bound is a LOWER bound only -- its upper end is infinite -- so it can "
        "catch a factor that is too small and cannot catch one that is too large. A "
        "ratio near 1 means it would catch a small error; a large ratio means it would "
        "only catch a gross one. Both are reported as whatever they are."
    )

    out.rule("5  why the slack follows conditioning and has no absolute floor")
    narrow = Bracket(low=1e-23, high=1e-23 + 4.5e-23, slack=1.8e-15, source="absolute floor")
    correct = Bracket(
        low=1e-23,
        high=1e-23 + 4.5e-23,
        slack=slack_from_conditioning(np.asarray([1e-23, 4.5e-23])),
        source="conditioning",
    )
    out(f"  bracket width                         : {correct.width:.3e}")
    out(f"  slack from conditioning               : {correct.slack:.3e}")
    out(f"  slack floored at an absolute 1.8e-15  : {narrow.slack:.3e}")
    out.check(
        "the conditioning slack is far below the bracket width",
        correct.slack < 0.01 * correct.width,
        f"{correct.slack / correct.width:.3e} of the width",
    )
    out.check(
        "the absolute-floor slack would exceed the bracket width by orders of magnitude",
        narrow.slack > 1e6 * narrow.width,
        f"{narrow.slack / narrow.width:.3e} of the width, which makes the bracket vacuous",
    )
    out.check(
        "the conditioning slack rejects a value 1e7 widths outside the bracket",
        not correct.contains(1e-16),
        "and the floored form admits it",
    )

    out.rule("6  the quantiser noise floor, which is what actually limits these numbers")
    cell = campaign.require(_bootstrap.HEADLINE_CELL)
    quantum = float(np.mean(cell.power.quantisation_floor_mw))
    per_sample = float(target_quantisation_sd(quantum, 1.0))
    whole, _active = cell_energy(cell)
    n_samples = len(cell.power)
    energy_sd = per_sample * whole.duration_s / np.sqrt(n_samples)
    out(f"  current quantum                 : {CURRENT_LSB_MA:.1f} mA")
    out(f"  power quantum at the bus voltage: {quantum:.3f} mW")
    out(f"  per-sample quantiser sd         : {per_sample:.3f} mW  (q / sqrt(12))")
    out(f"  samples in the run              : {n_samples}")
    out(f"  implied energy sd               : {energy_sd:.3f} mJ on {whole.energy_mj:.3f} mJ")
    out(f"  relative                        : {energy_sd / whole.energy_mj:.3e}")
    out(
        "  The sqrt(n) reduction above is a LOWER BOUND on the noise and is named as "
        "one: the INA3221 averages 512 conversions in hardware, so neighbouring 20 Hz "
        "samples overlap in the instrument and their quantisation errors are not "
        "independent. The honest statement is that the true figure is somewhere "
        "between this and the unreduced per-sample value."
    )
    out.check(
        "the quantiser floor is far above the timestamp-resolution floor of about 1e-8",
        energy_sd / whole.energy_mj > 1e-8,
        "so the 40 mW quantum, not the floating-point arithmetic, is the limit on every "
        "energy figure in this product",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
