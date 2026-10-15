"""What routing on confidence costs, and what it buys, measured.

Four policies on the same held-out test split:

- ``oracle``: perfect identification, cheaper of the fault-specific and the
  conservative route.
- ``always_specific``: act on the detector's identification, never route.
- ``confidence_routed``: act on the identification when the confidence reaches
  the declared threshold, otherwise take the class-conservative route.
- ``always_conservative``: never act on an identification.

Cost is the declared resolution cost in minutes of mission outage. Missed
detections are counted and NOT costed, because any number put there would be
invented. Mis-routes are counted separately: a mis-route is a fault-specific
response commanded for the wrong fault.

Run:
    PYTHONPATH=src python validation/validate_conservatism.py
"""

from __future__ import annotations

import math
import time

from safemodetree import (
    DetectionAssumption,
    analyse,
    compare_detectors,
    compare_policies,
    conservatism_sweep,
    load_spec,
    packaged_spec_path,
)

SEED = 63001
N_PER_FAULT = 400
NOISE_SCALE = 2.0
THRESHOLD = 0.7
MODES = ("fine_pointing", "coarse_pointing")
DETECTORS = ("learned", "baseline")
POLICIES = ("oracle", "always_specific", "confidence_routed", "always_conservative")

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> None:
    """Score the four policies in both modes with both detectors."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    print("safemodetree cost of conservatism")
    print("=" * 100)
    print(f"declaration: {spec.summary()}")
    print(f"seed {SEED}, {N_PER_FAULT} per fault, noise scale {NOISE_SCALE:g}, "
          f"routing threshold {THRESHOLD:g}")
    print("cost unit: minutes of mission outage per COMMANDED fault event")
    print()

    start = time.perf_counter()
    comparison = compare_detectors(
        spec, n_per_fault=N_PER_FAULT, noise_scale=NOISE_SCALE, seed=SEED
    )

    print("1. the declaration's own arithmetic, before any detector")
    print("-" * 100)
    report = analyse(spec)
    header = (
        f"{'class':<36} {'mode':<16} {'specific_min':>26} {'cons_min':>9} {'cost_of_cons':>13}"
    )
    print(header)
    print("-" * len(header))
    for item in report.ambiguity:
        specific = ", ".join(
            f"{fault_id.split('_')[0]}={value:.1f}"
            for fault_id, value in sorted(item.best_specific_cost_min.items())
        )
        print(
            f"{'+'.join(item.faults):<36} {item.mode:<16} {specific:>26} "
            f"{item.conservative_cost_min:>9.3f} {item.conservatism_cost_min:>13.3f}"
        )
    print()

    results: dict[tuple[str, str], dict[str, object]] = {}
    print("2. measured policy cost")
    print("-" * 100)
    header = (
        f"{'detector':<9} {'mode':<16} {'policy':<22} {'thr':>6} {'cmded':>6} {'missed':>7} "
        f"{'cost_min':>9} {'misrout':>8} {'unsafe':>7} {'cons_fr':>8} {'fa_cost':>8}"
    )
    print(header)
    print("-" * len(header))
    for detector in DETECTORS:
        for mode in MODES:
            scores = compare_policies(
                spec,
                comparison,
                mode_id=mode,
                confidence_threshold=THRESHOLD,
                detector=detector,
            )
            results[(detector, mode)] = scores
            for name in POLICIES:
                score = scores[name]
                print(
                    f"{detector:<9} {mode:<16} {score.name:<22} "
                    f"{score.confidence_threshold:>6.3f} {score.n_commanded:>6} "
                    f"{score.n_missed:>7} {score.mean_cost_min:>9.4f} "
                    f"{score.n_misrouted:>8} {score.n_unsafe:>7} "
                    f"{score.conservative_fraction:>8.4f} {score.false_alarm_cost_min:>8.4f}"
                )
    print()

    print("3. the cost of conservatism, per commanded fault event")
    print("-" * 100)
    header = (
        f"{'detector':<9} {'mode':<16} {'vs_oracle':>11} {'vs_specific':>13} "
        f"{'misroutes_avoided':>19} {'min_per_misroute_avoided':>26}"
    )
    print(header)
    print("-" * len(header))
    for key, scores in results.items():
        avoided = int(scores["misroutes_avoided"])
        delta = float(scores["cost_of_conservatism_vs_specific"])
        per = (
            delta * scores["confidence_routed"].n_commanded / avoided
            if avoided > 0
            else math.nan
        )
        print(
            f"{key[0]:<9} {key[1]:<16} "
            f"{float(scores['cost_of_conservatism_vs_oracle']):>+11.4f} "
            f"{delta:>+13.4f} {avoided:>19} {per:>26.4f}"
        )
    print()
    print("FINDING: the sign of the cost of conservatism against the always-specific")
    print("policy is not the same in every configuration. With the declared threshold")
    print("logic in coarse_pointing it is NEGATIVE: routing conservatively is cheaper")
    print("AND eliminates mis-routes. With the learned detector in fine_pointing it is")
    print("positive, so conservatism is bought with outage minutes. Both are measured")
    print("on the same test split and both are reported.")
    print()

    print("4. the routing-threshold sweep, learned detector, fine_pointing")
    print("-" * 100)
    header = (
        f"{'threshold':<12} {'cmded':>6} {'missed':>7} {'cost_min':>9} {'misrout':>8} "
        f"{'unsafe':>7} {'cons_fr':>8} {'fa_cost':>8}"
    )
    print(header)
    print("-" * len(header))
    rows = conservatism_sweep(spec, comparison, mode_id="fine_pointing", detector="learned")
    for row in rows:
        print(
            f"{row.confidence_threshold:<12.2f} {row.n_commanded:>6} {row.n_missed:>7} "
            f"{row.mean_cost_min:>9.4f} {row.n_misrouted:>8} {row.n_unsafe:>7} "
            f"{row.conservative_fraction:>8.4f} {row.false_alarm_cost_min:>8.4f}"
        )
    print()

    print("5. the four policies under a 60 s detection latency budget")
    print("-" * 100)
    tight_header = (
        f"{'policy':<22} {'cmded':>6} {'missed':>7} {'cost_min':>9} {'misrout':>8} "
        f"{'unsafe':>7} {'cons_fr':>8} {'fa_cost':>8}"
    )
    print(tight_header)
    print("-" * len(tight_header))
    tight = DetectionAssumption(latency_budget_s=60.0)
    tight_scores = compare_policies(
        spec,
        comparison,
        mode_id="fine_pointing",
        confidence_threshold=THRESHOLD,
        assumption=tight,
        detector="learned",
    )
    for name in POLICIES:
        score = tight_scores[name]
        print(
            f"{score.name:<22} {score.n_commanded:>6} {score.n_missed:>7} "
            f"{score.mean_cost_min:>9.4f} {score.n_misrouted:>8} {score.n_unsafe:>7} "
            f"{score.conservative_fraction:>8.4f} {score.false_alarm_cost_min:>8.4f}"
        )
    print()
    print("With the two 120 s power faults no longer actionable, the always-specific")
    print("policy can no longer command their fault-specific responses and falls back")
    print("to the conservative route for them, so its cost rises and its mis-route")
    print("count falls without any change to the detector.")
    print()
    elapsed = time.perf_counter() - start

    print("6. checks")
    print("-" * 100)
    learned_fine = results[("learned", "fine_pointing")]
    baseline_coarse = results[("baseline", "coarse_pointing")]
    record(
        "the always-conservative policy never mis-routes, in every configuration",
        all(scores["always_conservative"].n_misrouted == 0 for scores in results.values()),
    )
    record(
        "the always-conservative policy costs exactly the declared route cost of "
        "53.0 min from fine_pointing",
        all(
            abs(results[(d, "fine_pointing")]["always_conservative"].mean_cost_min - 53.0) <= 1e-9
            for d in DETECTORS
        ),
    )
    record(
        "confidence routing reduces the mis-route count against always-specific "
        "in every configuration",
        all(
            scores["confidence_routed"].n_misrouted < scores["always_specific"].n_misrouted
            for scores in results.values()
        ),
    )
    record(
        "the oracle is the cheapest policy in every configuration",
        all(
            scores["oracle"].mean_cost_min
            <= min(scores[name].mean_cost_min for name in POLICIES[1:]) + 1e-9
            for scores in results.values()
        ),
    )
    record(
        f"the learned detector in fine_pointing pays "
        f"{float(learned_fine['cost_of_conservatism_vs_specific']):+.4f} min per event "
        f"for {int(learned_fine['misroutes_avoided'])} avoided mis-routes",
        float(learned_fine["cost_of_conservatism_vs_specific"]) > 0.0,
    )
    record(
        f"the declared threshold logic in coarse_pointing pays "
        f"{float(baseline_coarse['cost_of_conservatism_vs_specific']):+.4f} min per event, "
        "which is negative",
        float(baseline_coarse["cost_of_conservatism_vs_specific"]) < 0.0,
    )
    record(
        "no policy leaves a commanded event with no reachable safe mode",
        all(scores[name].n_unsafe == 0 for scores in results.values() for name in POLICIES),
    )
    record(
        "the conservative fraction is non-decreasing along the threshold sweep",
        all(
            rows[i].conservative_fraction <= rows[i + 1].conservative_fraction + 1e-12
            for i in range(len(rows) - 1)
        ),
    )
    record(
        "the mis-route count reaches zero at the top of the sweep",
        rows[-1].n_misrouted == 0,
    )
    print()
    print(f"elapsed: {elapsed:.2f} s")
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("CONSERVATISM MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
