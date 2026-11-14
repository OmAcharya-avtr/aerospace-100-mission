"""Negative controls, each anchored on the TRUE value and asserted to FIRE.

INPUT ORIGIN: SYNTHETIC for every control over the arithmetic, plus two controls
over the MEASURED-data refusal path that use temporary directories and no
measured data at all.

Two rules this mission learned the hard way govern every control below.

**A control is anchored on the true value, not on an output.** Offsetting a
reconstruction does not guarantee a violation, because the reconstruction
already sits inside the band: only an offset strictly greater than the band's
own width plus twice its slack is guaranteed to leave it. Every injected
violation below is computed from the bracket's own width and slack.

**A control is asserted to fire, and in its declared direction.** A control that
does not fire is a failure of the control, not a success of the product. Both
directions are declared and counted: the controls that must fire, and the
controls that must NOT fire, which are the ones that would catch a check that
rejects everything.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

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
from powerenvelope.predictor import (  # noqa: E402
    LICENSED_DESCRIPTORS,
    DescriptorLeakError,
    assert_descriptors_licensed,
)
from powerenvelope.trace import TraceDataAbsentError  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_negative_controls_output.txt")
    out.rule("NEGATIVE CONTROLS -- anchored on the true value, verified to fire")
    out("INPUT ORIGIN: SYNTHETIC arithmetic, plus the measured-data REFUSAL path.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    campaign = _bootstrap.synthetic_campaign()
    cell = campaign.require(_bootstrap.HEADLINE_CELL)
    power = cell.power

    out.rule("1  the energy integral against two independent brackets")
    truth = trapezoid_energy_mj(power.t, power.vdd_in_mw)
    out(f"  true whole-run energy: {truth:.6f} mJ")
    for name, bracket in (
        ("Riemann", riemann_bracket(power.t, power.vdd_in_mw)),
        ("mean-value", extreme_value_bracket(power.t, power.vdd_in_mw)),
    ):
        out(
            f"  {name} bracket: [{bracket.low:.6f}, {bracket.high:.6f}] mJ, "
            f"width {bracket.width:.6e}, slack {bracket.slack:.6e}"
        )
        out.control(
            f"the true energy is inside the {name} bracket",
            not bracket.contains(truth),
            must_fire=False,
            detail=f"excess {bracket.excess(truth):.3e} mJ",
        )
        # Anchored on the TRUE value: the smallest offset guaranteed to leave a
        # closed bracket of width w with slack s is w + 2s. A relative margin is
        # added so the injected value is not on the boundary.
        offset = bracket.width + 2.0 * bracket.slack + abs(truth) * 1e-9
        out.control(
            f"an energy offset by the {name} bracket's own width leaves it",
            not bracket.contains(truth + offset),
            must_fire=True,
            detail=f"offset {offset:.6e} mJ, injected {truth + offset:.6f} mJ",
        )
        out.control(
            f"the same offset downwards leaves the {name} bracket",
            not bracket.contains(truth - offset),
            must_fire=True,
            detail=f"injected {truth - offset:.6f} mJ",
        )

    out.rule("2  the duty-cycle factor against a bound from the sleep window's power")
    floor_min = float(campaign.require("floor").power.vdd_in_mw.min())
    whole, active = cell_energy(cell)
    sleep_s = whole.duration_s - active.duration_s
    bracket = duty_cycle_lower_bracket(active.energy_mj, floor_min, sleep_s)
    true_factor = whole.energy_mj / active.energy_mj
    out(f"  measured floor minimum: {floor_min:.6f} mW over {sleep_s:.3f} s asleep")
    out(f"  lower bound: {bracket.low:.9f}; true factor: {true_factor:.9f}")
    out.control(
        "the true duty-cycle factor satisfies the bound",
        not bracket.contains(true_factor),
        must_fire=False,
        detail=f"margin {true_factor - bracket.low:.6e}",
    )
    # Anchored on the TRUE value: a factor below the bound by twice the bound's
    # own slack plus a relative margin must be rejected.
    violation = bracket.low - 2.0 * bracket.slack - abs(bracket.low) * 1e-9
    out.control(
        "a duty-cycle factor below the bound is rejected",
        not bracket.contains(violation),
        must_fire=True,
        detail=f"injected {violation:.9f} against a bound of {bracket.low:.9f}",
    )

    out.rule("3  the headline factor against a bound from extreme values alone")
    result = decompose(cell)
    inside = np.zeros(power.t.size, dtype=bool)
    for lo, hi in zip(*active_windows(cell.require_decisions()), strict=True):
        inside |= (power.t >= lo) & (power.t <= hi)
    bracket = headline_lower_bracket(
        run_duration_s=float(power.t[-1] - power.t[0]),
        min_power_mw=float(power.vdd_in_mw.min()),
        max_active_power_mw=float(power.vdd_in_mw[inside].max()),
        max_latency_s=float(cell.require_decisions().latency_s.max()),
        n_accepted=result.n_accepted,
    )
    out(f"  lower bound: {bracket.low:.9f}; measured factor: {result.factor:.9f}")
    out(f"  bound source: {bracket.source}")
    out.control(
        "the measured headline factor satisfies the independent bound",
        not bracket.contains(result.factor),
        must_fire=False,
        detail=f"margin {result.factor - bracket.low:.6f}",
    )
    violation = bracket.low - 2.0 * bracket.slack - abs(bracket.low) * 1e-9
    out.control(
        "a headline factor below the independent bound is rejected",
        not bracket.contains(violation),
        must_fire=True,
        detail=f"injected {violation:.9f}",
    )

    out.rule("4  the decomposition's own closure")
    product = result.product_of_named_factors()
    out(f"  F = {result.factor:.12f}; product of the three named factors = {product:.12f}")
    out.control(
        "the undamaged decomposition closes",
        abs(product / result.factor - 1.0) > 1e-12,
        must_fire=False,
        detail=f"closure error {product / result.factor - 1.0:.3e}",
    )
    # Anchored on the TRUE value: scale one factor by 1 + 1e-6, which is six
    # orders above the closure tolerance, and the closure must fail.
    damaged = (
        result.rejection_factor * (1.0 + 1e-6)
        * result.duty_cycle_factor
        * result.rail_voltage_factor
    )
    out.control(
        "a decomposition with one factor perturbed by 1e-6 does not close",
        abs(damaged / result.factor - 1.0) > 1e-12,
        must_fire=True,
        detail=f"closure error {damaged / result.factor - 1.0:.3e} against a 1e-12 tolerance",
    )

    out.rule("5  the permissive-gate control: the rejection half of the headline")
    strict = decompose(campaign.require(_bootstrap.HEADLINE_CELL))
    permissive = decompose(campaign.require(_bootstrap.PERMISSIVE_CELL))
    out(
        f"  strict: a = {strict.acceptance_rate:.6f}, 1/a = {strict.rejection_factor:.6f}, "
        f"F = {strict.factor:.6f}"
    )
    out(
        f"  permissive: a = {permissive.acceptance_rate:.6f}, 1/a = "
        f"{permissive.rejection_factor:.6f}, F = {permissive.factor:.6f}"
    )
    out.control(
        "the permissive gate nearly removes the rejection factor",
        permissive.rejection_factor < 0.5 * strict.rejection_factor,
        must_fire=True,
        detail=f"ratio {strict.rejection_factor / permissive.rejection_factor:.4f}",
    )
    out.control(
        "the permissive gate leaves the duty-cycle factor alone",
        abs(permissive.duty_cycle_factor / strict.duty_cycle_factor - 1.0) > 0.02,
        must_fire=False,
        detail=f"{permissive.duty_cycle_factor:.6f} vs {strict.duty_cycle_factor:.6f}; the "
        "duty-cycle factor is a property of the duty, not of the gate",
    )
    full = decompose(campaign.require("w64_d100"))
    out.control(
        "at duty 1.00 the duty-cycle factor is 1",
        abs(full.duty_cycle_factor - 1.0) > 1e-9,
        must_fire=False,
        detail=f"{full.duty_cycle_factor:.12f}; there is no sleep window to charge",
    )

    out.rule("6  the descriptor licence check")
    out.control(
        "the licensed descriptor set passes",
        _raises(lambda: assert_descriptors_licensed(LICENSED_DESCRIPTORS)),
        must_fire=False,
        detail=f"{len(LICENSED_DESCRIPTORS)} descriptor(s)",
    )
    for name in (
        "vdd_in_mw",
        "mean_latency_s",
        "cpu_busy_fraction",
        "inferences",
        "acceptance_rate",
        "moon_phase",
    ):
        out.control(
            f"an unlicensed descriptor {name!r} is refused",
            _raises(lambda name=name: assert_descriptors_licensed((*LICENSED_DESCRIPTORS, name))),
            must_fire=True,
            detail="DescriptorLeakError",
        )

    out.rule("7  the MEASURED-data refusal path")
    scratch = Path(tempfile.mkdtemp(prefix="powerenvelope-refusal-"))
    empty = scratch / "empty"
    empty.mkdir()
    previous = os.environ.get(_bootstrap.TRACES_ENVIRONMENT_VARIABLE)
    try:
        os.environ[_bootstrap.TRACES_ENVIRONMENT_VARIABLE] = str(empty)
        fired = _raises(_bootstrap.require_measured_campaign, TraceDataAbsentError)
        out.control(
            "an empty directory is refused rather than substituted",
            fired,
            must_fire=True,
            detail="require_measured_campaign raises TraceDataAbsentError",
        )
        synthetic_directory = scratch / "synthetic"
        _bootstrap.synthetic_campaign(synthetic_directory, cycles=3)
        os.environ[_bootstrap.TRACES_ENVIRONMENT_VARIABLE] = str(synthetic_directory)
        fired = _raises(_bootstrap.require_measured_campaign, TraceDataAbsentError)
        out.control(
            "a directory that declares itself synthetic is refused as measured",
            fired,
            must_fire=True,
            detail="the manifest's synthetic flag is read and honoured",
        )
    finally:
        if previous is None:
            os.environ.pop(_bootstrap.TRACES_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[_bootstrap.TRACES_ENVIRONMENT_VARIABLE] = previous

    out.rule("8  the control on the control: a slack with an absolute floor is vacuous")
    out(
        "  A bound-check slack must follow the CONDITIONING of its own expression. A "
        "slack floored at an absolute constant makes a narrow bracket admit anything, "
        "which is the failure this product's slack rule exists to avoid."
    )
    narrow = Bracket(low=1e-23, high=1e-23 + 4.5e-23, slack=1.8e-15, source="absolute floor")
    correct = Bracket(
        low=1e-23,
        high=1e-23 + 4.5e-23,
        slack=slack_from_conditioning(np.asarray([1e-23, 4.5e-23])),
        source="conditioning",
    )
    out(f"  bracket width {narrow.width:.3e}; wrong slack {narrow.slack:.3e}; "
        f"right slack {correct.slack:.3e}")
    out.control(
        "the conditioning-based slack rejects an intruder 1e7 times the bracket width",
        not correct.contains(1e-16),
        must_fire=True,
        detail="1e-16 is outside the correctly widened bracket",
    )
    out.control(
        "the absolute-floor slack ADMITS that intruder, which is why it is not used",
        narrow.contains(1e-16),
        must_fire=True,
        detail="this control fires on the WRONG form, demonstrating it is vacuous",
    )

    out.rule("summary")
    out(
        f"  controls declared {out.controls_expected}; must-fire "
        f"{out.must_fire_fired}/{out.must_fire}; must-NOT-fire silent "
        f"{out.must_not_fire_silent}/{out.must_not_fire}"
    )
    out(
        "  A control that did not fire in its declared direction is reported as a FAILED "
        "check above and is not removed."
    )
    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


def _raises(call, exception=DescriptorLeakError) -> bool:
    """Whether ``call`` raises ``exception``. Any other exception is not a fire."""
    try:
        call()
    except exception:
        return True
    except Exception:  # noqa: BLE001 -- a different error is not the control firing
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
