"""The README's worked example, run verbatim.

Run:
    PYTHONPATH=src python validation/worked_example.py
"""

from __future__ import annotations

from safemodetree import (
    DetectionAssumption,
    FaultResponsePolicy,
    analyse,
    load_spec,
    packaged_spec_path,
    reachability_report,
)

spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
assumption = DetectionAssumption(latency_budget_s=90.0)

report = analyse(spec, assumption)
print(spec.summary())
print(f"assumption: {assumption.describe()}")
print(f"findings: {report.total_findings}  ok: {report.ok}")

reach = reachability_report(spec, assumption)
print(f"every fault state reaches a safe mode: {reach.all_reachable}")
print(f"worst cheapest route: {reach.worst_cost_min:.1f} min")
for item in report.conflicting_ambiguity:
    print(
        f"ambiguous in {item.mode}: {' vs '.join(item.faults)} -> "
        f"{item.preferred[item.faults[0]]} vs {item.preferred[item.faults[1]]}; "
        f"conservative {item.conservative_response} at "
        f"{item.conservative_cost_min:.1f} min "
        f"(+{item.conservatism_cost_min:.1f} min over the mean specific route)"
    )

policy = FaultResponsePolicy(spec, assumption, confidence_threshold=0.7)
for confidence in (0.96, 0.48):
    decision = policy.decide("fine_pointing", "st_latchup", confidence)
    print(
        f"conf {confidence:.2f} -> {decision.response_id}"
        f"{' (conservative)' if decision.routed_conservative else ''}; "
        f"cost if the fault really is st_latchup "
        f"{policy.cost_of(decision, 'st_latchup'):.1f} min, "
        f"if it is st_blinding {policy.cost_of(decision, 'st_blinding'):.1f} min"
    )
