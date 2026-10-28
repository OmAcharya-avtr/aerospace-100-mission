"""The worked example quoted in README.md, run so that the quoted output is real.

This script is the source of the README's "A worked example" section. It is
re-executed as the last step before hand-off and the README text is replaced
with whatever it printed, because two products of this mission caught their own
wrong README numbers that way.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

import telemgap  # noqa: E402
from telemgap import dataset, interpolate, policy  # noqa: E402


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("telemgap -- the worked example from README.md")
    for line in environment_lines():
        out(line)
    out("")

    # --- the example, exactly as README.md quotes it -----------------------
    generated = dataset.generate(samples=4000, seed=515, n_clock_resets=0)
    record = generated.record

    report = telemgap.check(record)

    out(f"frames received            : {len(record)}")
    out(f"gaps                       : {len(report.findings.gaps)}")
    out(f"frames the gaps imply lost : {report.missing_frame_total}")
    out(f"verdict counts             : {report.counts}")

    limit = report.determinability["in_range_corruption"]
    out("")
    out("what this record cannot determine, first output of every call:")
    for key in report.determinability.undeterminable_keys:
        out(f"  {key}")
    out(f"worst-channel share of single-bit errors a range screen cannot see: {limit.quantity:.6f}")

    plan = policy.plan(
        record,
        report,
        tolerances={
            "battery_voltage": 0.05,
            "valve_current": 0.01,
            "gyro_rate": 0.02,
            "tank_pressure": 0.02,
        },
    )
    out("")
    out(f"policy decisions           : {len(plan.decisions)}")
    out(f"interpolation permitted    : {len(plan.permissions)}")
    out(f"interpolation refused      : {len(plan.refusals)}")

    first = plan.refusals[0]
    out("")
    out(f"first refusal: gap {first.gap_index}, channel {first.channel_name}")
    out(f"  gap duration   : {first.duration_s:.3f} s ({first.slots:.2f} cadence slots)")
    out(f"  curvature used : {first.curvature.value:.6g} per s^2 ({first.curvature.source})")
    out(f"  failed gates   : {list(first.failed_gates)}")

    reconstruction = interpolate.reconstruct(record, report.findings.gaps, plan)
    out("")
    out(f"grid slots                 : {reconstruction.n_slots}")
    out(f"filled share               : {reconstruction.fill_fraction:.6f}")
    out(f"refused share              : {reconstruction.refused_fraction:.6f}")

    column = record.schema.channel_index("battery_voltage")
    error, count = interpolate.measured_error(
        reconstruction, generated.truth_times, generated.truth_values[:, column], column
    )
    predicted = max(
        decision.predicted_error
        for decision in plan.decisions
        if decision.channel == column and not decision.refused
    )
    out("")
    out(f"battery_voltage synthesised slots : {count}")
    out(f"  realised max absolute error     : {error:.6g} V")
    out(f"  worst predicted by equation (3'): {predicted:.6g} V")
    # --- end of the example -----------------------------------------------

    out("")
    out.check(
        "the example's realised error stays inside what the policy predicted",
        error <= predicted,
        f"{error:.6g} against {predicted:.6g}",
    )
    out.check(
        "the example refuses at least one gap and permits at least one",
        bool(plan.refusals) and bool(plan.permissions),
        f"{len(plan.refusals)} refused, {len(plan.permissions)} permitted",
    )
    out.check(
        "the determinability verdict names at least three undeterminable limits",
        len(report.determinability.undeterminable_keys) >= 3,
        f"{list(report.determinability.undeterminable_keys)}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
