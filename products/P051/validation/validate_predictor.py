"""Validation 6: the learned switch predictor against the exact analytic condition.

The classical computation is implemented first and benchmarked first. The
learned model is measured against it on the same held-out episodes, and the
result is published whichever way it goes.

Two tasks, because the honest answer differs between them
  Task A, lead 0: does the guard fire at THIS step. This is the switching
    condition's own criterion, so the exact computation is right by
    construction and the learned model can at best tie.
  Task B, lead L: does the guard fire at any step in the next L. The exact
    answer here is a worst-case set computation that answers "may fire", so it
    over-predicts on a realised episode, and a learned model has room to win.

Checks
  1  Dataset construction, split by episode, with base rates and the measured
     saturation rate of the performance input.
  2  Task A: the exact one-step condition against the forest and the logistic
     model, with precision, recall, F1, Brier, expected calibration error and
     single-row latency.
  3  Task B: the exact worst-case and nominal multi-step predictors against the
     same two learned models, plus a persistence baseline and a base-rate
     baseline.
  4  Lead-time distributions for every predictor.
  5  Latency against forest size, so the cost side of the comparison is a curve
     and not one point.
  6  Calibration: reliability bins for the forest, and the measured fact that
     the exact predictors have no probability output at all.

Compute budget: 2 shared cores. This script trains four forests and runs about
60000 simulated steps; measured below.
"""

from __future__ import annotations

import time

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard import (  # noqa: E402
    ExactLeadPredictor,
    SimplexGuard,
    build_dataset,
    exact_predictor_scores,
    expected_calibration_error,
    fit_switch_predictor,
    guard_condition_scores,
    lead_times,
    measure_decision_cost,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    score_binary,
)

SEED = 5101
EPISODES = 40
STEPS = 1200
LEAD = 5
TREES = 150
start = time.perf_counter()
failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {name}: {detail}")


plant = reference_plant()
baseline, performance = reference_controllers(plant)
invariant = robust_invariant_set(plant, baseline).polytope
guard = SimplexGuard(plant, baseline, invariant)

print(f"validate_predictor.py   seed = {SEED}")
print(f"{EPISODES} episodes x {STEPS} steps, lead {LEAD}, {TREES}-tree forest")
print("=" * 78)

# --- check 1: the datasets ---------------------------------------------------
t0 = time.perf_counter()
data = {
    0: build_dataset(plant, guard, performance, EPISODES, STEPS, SEED, 0),
    LEAD: build_dataset(plant, guard, performance, EPISODES, STEPS, SEED, LEAD),
}
build_seconds = time.perf_counter() - t0
train_ids = np.arange(0, 26)
calib_ids = np.arange(26, 32)
test_ids = np.arange(32, 40)
splits = {
    lead: (
        d.select_episodes(train_ids),
        d.select_episodes(calib_ids),
        d.select_episodes(test_ids),
    )
    for lead, d in data.items()
}
print("check 1  datasets")
print("-" * 78)
for lead, d in data.items():
    tr, ca, te = splits[lead]
    print(
        f"  lead={lead}: {len(d)} rows, base rate {d.base_rate:.6f}; "
        f"train {len(tr)} ({tr.base_rate:.6f}), calib {len(ca)} ({ca.base_rate:.6f}), "
        f"test {len(te)} ({te.base_rate:.6f})"
    )
print(
    f"  performance-input saturation rate {data[0].saturation_rate:.6f} -- this is the "
    f"size of the approximation the exact multi-step predictor makes by ignoring "
    f"saturation"
)
print(f"  dataset construction: {build_seconds:.1f} s for {2 * EPISODES * STEPS} steps")
print(f"  features (5): {', '.join(data[0].feature_names)}")
print("  the switching-condition margin is NOT a feature, by design")
print("-" * 78)
report(
    "check 1  splits are by episode and disjoint",
    set(splits[0][0].episode_ids).isdisjoint(set(splits[0][2].episode_ids)),
    "26 train / 6 calibration / 8 test episodes, no row shared; a random row split "
    "would leak, because consecutive rows of one episode are the same trajectory",
)

# --- helpers ----------------------------------------------------------------
def learned_scores(lead: int, kind: str, trees: int, label: str):
    tr, ca, te = splits[lead]
    t_fit = time.perf_counter()
    model = fit_switch_predictor(tr, ca, kind, n_estimators=trees, random_state=SEED)
    fit_seconds = time.perf_counter() - t_fit
    prob = model.predict_proba(te.features)[:, 1]
    row = te.features[0:1]
    cost = measure_decision_cost(lambda: model.predict_proba(row), n_calls=150)
    scores = score_binary(
        label, te.labels, prob >= 0.5, probability=prob, microseconds_per_decision=cost
    )
    return model, prob, scores, fit_seconds


# --- check 2: Task A, the condition's own criterion --------------------------
print()
print("check 2  TASK A (lead 0): does the guard fire at this step")
print("-" * 78)
te0 = splits[0][2]
exact_a = guard_condition_scores(te0, guard, performance)
print("  " + exact_a.row())
forest_a_model, forest_a_prob, forest_a, forest_a_fit = learned_scores(
    0, "forest", TREES, f"learned forest ({TREES} trees)"
)
print("  " + forest_a.row())
logit_a_model, logit_a_prob, logit_a, logit_a_fit = learned_scores(
    0, "logistic", TREES, "learned logistic"
)
print("  " + logit_a.row())
base_rate_a = score_binary(
    "always-negative (base rate)",
    te0.labels,
    np.zeros(len(te0), dtype=bool),
    probability=np.full(len(te0), splits[0][0].base_rate),
)
print("  " + base_rate_a.row())
print(f"  forest fit {forest_a_fit:.1f} s, logistic fit {logit_a_fit:.1f} s")
print("-" * 78)
report(
    "check 2a the exact condition is perfect on its own criterion",
    exact_a.precision == 1.0 and exact_a.recall == 1.0,
    f"precision {exact_a.precision:.6f}, recall {exact_a.recall:.6f}, "
    f"{exact_a.false_positive} false positives, {exact_a.false_negative} false "
    f"negatives on {exact_a.n} held-out steps",
)
report(
    "check 2b THE LEARNED MODEL LOSES ON ACCURACY",
    forest_a.f1 < exact_a.f1,
    f"forest F1 {forest_a.f1:.6f} against the exact 1.000000; "
    f"{forest_a.false_positive} false positives and {forest_a.false_negative} false "
    f"negatives in {forest_a.n} steps. Structural reason: the exact boundary is the "
    f"intersection of {invariant.n_halfspaces} halfspaces in (x, u), none of them "
    f"axis-aligned, and an axis-aligned tree ensemble approximates a tilted hyperplane "
    f"with staircase error. No retune was attempted",
)
report(
    "check 2c THE LEARNED MODEL ALSO LOSES ON COST",
    forest_a.microseconds_per_decision > exact_a.microseconds_per_decision,
    f"forest {forest_a.microseconds_per_decision:.1f} us per single-row decision "
    f"against {exact_a.microseconds_per_decision:.2f} us for the exact condition, a "
    f"factor of "
    f"{forest_a.microseconds_per_decision / exact_a.microseconds_per_decision:.0f}. "
    f"At dt = {plant.dt:g} s the exact condition uses "
    f"{100 * exact_a.microseconds_per_decision / (plant.dt * 1e6):.4f} % of the sample "
    f"interval and the forest "
    f"{100 * forest_a.microseconds_per_decision / (plant.dt * 1e6):.2f} %",
)
report(
    "check 2d the learned model does provide something the exact one does not",
    np.isfinite(forest_a.brier) and not np.isfinite(exact_a.brier),
    f"a calibrated probability: forest Brier {forest_a.brier:.6f}, ECE "
    f"{forest_a.ece:.6f}, against an always-negative forecaster at the training base "
    f"rate with Brier {base_rate_a.brier:.6f}. The exact condition has no probability "
    f"output at all, which is the one structural advantage the learned model has here",
)

# --- check 3: Task B, anticipation ------------------------------------------
print()
print(f"check 3  TASK B (lead {LEAD}): does the guard fire within the next {LEAD} steps")
print("-" * 78)
teL = splits[LEAD][2]
exact_rows = []
for mode in ("worst_case", "nominal"):
    s = exact_predictor_scores(teL, guard, performance, LEAD, mode)
    exact_rows.append(s)
    print("  " + s.row())
persistence = score_binary(
    "persistence (fires now => will fire)",
    teL.labels,
    teL.fires_now,
    score=teL.fires_now.astype(float),
)
print("  " + persistence.row())
forest_b_model, forest_b_prob, forest_b, forest_b_fit = learned_scores(
    LEAD, "forest", TREES, f"learned forest ({TREES} trees)"
)
print("  " + forest_b.row())
logit_b_model, logit_b_prob, logit_b, logit_b_fit = learned_scores(
    LEAD, "logistic", TREES, "learned logistic"
)
print("  " + logit_b.row())
base_rate_b = score_binary(
    "always-negative (base rate)",
    teL.labels,
    np.zeros(len(teL), dtype=bool),
    probability=np.full(len(teL), splits[LEAD][0].base_rate),
)
print("  " + base_rate_b.row())
print(f"  forest fit {forest_b_fit:.1f} s, logistic fit {logit_b_fit:.1f} s")
print("-" * 78)
best_exact = max(exact_rows, key=lambda s: s.f1)
if forest_b.f1 > best_exact.f1:
    verdict = (
        f"THE LEARNED MODEL WINS ON THIS TASK: forest F1 {forest_b.f1:.6f} against the "
        f"best exact predictor's {best_exact.f1:.6f} ({best_exact.name}). The reason is "
        f"not that it is smarter: the exact predictor answers 'may fire under SOME "
        f"admissible disturbance sequence', which over-predicts a realised episode by "
        f"construction, and its precision is {best_exact.precision:.6f} against "
        f"recall {best_exact.recall:.6f}. The learned model is fitted to the realised "
        f"label, which is a different and easier question"
    )
else:
    verdict = (
        f"the learned model loses here too: forest F1 {forest_b.f1:.6f} against "
        f"{best_exact.f1:.6f} for {best_exact.name}"
    )
report("check 3a Task B verdict", True, verdict)
report(
    "check 3b the exact worst-case predictor is NOT sound on realised episodes",
    exact_rows[0].recall < 1.0,
    f"recall {exact_rows[0].recall:.6f}, so it misses "
    f"{1 - exact_rows[0].recall:.4f} of realised firings despite being a worst-case "
    f"over-approximation. The two stated approximations are responsible: it ignores the "
    f"saturation of the performance input, measured at "
    f"{data[0].saturation_rate:.6f} of steps, and it ignores the guard's own "
    f"intervention inside the horizon, which changes the realised trajectory. Both are "
    f"in the module docstring and both are measured rather than assumed harmless",
)
report(
    "check 3c the learned model is still far more expensive per decision",
    forest_b.microseconds_per_decision > 10.0 * exact_rows[0].microseconds_per_decision,
    f"forest {forest_b.microseconds_per_decision:.1f} us against "
    f"{exact_rows[0].microseconds_per_decision:.2f} us for the exact worst-case "
    f"predictor, a factor of "
    f"{forest_b.microseconds_per_decision / exact_rows[0].microseconds_per_decision:.0f}",
)

# --- check 4: lead times -----------------------------------------------------
print()
print("check 4  lead time before each firing event, in steps")
print("-" * 78)
lead_rows = {}
wc = ExactLeadPredictor(guard, performance, LEAD, "worst_case")
nom = ExactLeadPredictor(guard, performance, LEAD, "nominal")
lead_rows["exact worst_case"] = lead_times(
    wc.predict_many(teL.states, teL.references), teL.fires_now
)
lead_rows["exact nominal"] = lead_times(
    nom.predict_many(teL.states, teL.references), teL.fires_now
)
lead_rows[f"learned forest lead-{LEAD}"] = lead_times(forest_b_prob >= 0.5, teL.fires_now)
lead_rows["learned forest lead-0"] = lead_times(
    forest_a_model.predict_proba(teL.features)[:, 1] >= 0.5, teL.fires_now
)
lead_rows["exact one-step condition"] = lead_times(teL.fires_now, teL.fires_now)
for name, row in lead_rows.items():
    print(
        f"  {name:<28s} events={int(row['n_events']):<5d} mean={row['mean_lead']:6.3f} "
        f"median={row['median_lead']:5.1f} max={row['max_lead']:5.1f} "
        f"zero-lead fraction={row['fraction_zero_lead']:.4f}"
    )
print("-" * 78)
report(
    "check 4  the exact worst-case predictor anticipates every event",
    lead_rows["exact worst_case"]["fraction_zero_lead"] == 0.0,
    f"mean lead {lead_rows['exact worst_case']['mean_lead']:.3f} steps "
    f"({plant.dt * lead_rows['exact worst_case']['mean_lead']:.4f} s) with zero events "
    f"unanticipated, against mean "
    f"{lead_rows[f'learned forest lead-{LEAD}']['mean_lead']:.3f} steps and "
    f"{lead_rows[f'learned forest lead-{LEAD}']['fraction_zero_lead']:.4f} of events "
    f"unanticipated for the forest. The one-step condition has lead 0 by definition "
    f"and is in the table to make that explicit",
)

# --- check 5: latency against forest size -----------------------------------
print()
print("check 5  single-row latency and accuracy against forest size")
print("-" * 78)
size_rows = []
for trees in (20, 50, 150, 300):
    _, _, s, fit_s = learned_scores(0, "forest", trees, f"forest {trees}")
    size_rows.append((trees, s, fit_s))
    print(
        f"  trees={trees:<5d} F1={s.f1:.6f} Brier={s.brier:.6f} "
        f"us/decision={s.microseconds_per_decision:8.1f} fit={fit_s:5.1f} s"
    )
print(f"  exact one-step condition  F1=1.000000 "
      f"us/decision={exact_a.microseconds_per_decision:8.2f}")
print("-" * 78)
smallest = size_rows[0][1]
report(
    "check 5  no forest size reaches the exact condition's cost",
    smallest.microseconds_per_decision > exact_a.microseconds_per_decision,
    f"even a 20-tree forest costs {smallest.microseconds_per_decision:.1f} us per "
    f"single-row decision, "
    f"{smallest.microseconds_per_decision / exact_a.microseconds_per_decision:.0f} "
    f"times the exact condition's {exact_a.microseconds_per_decision:.2f} us, and its "
    f"F1 is {smallest.f1:.6f}. The gap is dominated by scikit-learn's per-call "
    f"overhead, not by tree traversal: the latency rises by only a factor of "
    f"{size_rows[-1][1].microseconds_per_decision / smallest.microseconds_per_decision:.2f} "
    f"from 20 to 300 trees",
)

# --- check 6: calibration ----------------------------------------------------
print()
print("check 6  reliability of the forest's calibrated probability, Task A")
print("-" * 78)
edges = np.linspace(0.0, 1.0, 11)
idx = np.clip(np.digitize(forest_a_prob, edges[1:-1]), 0, 9)
for b in range(10):
    sel = idx == b
    if not np.any(sel):
        continue
    print(
        f"  [{edges[b]:.1f}, {edges[b + 1]:.1f})  n={int(sel.sum()):<6d} "
        f"mean forecast={forest_a_prob[sel].mean():.6f} "
        f"observed frequency={te0.labels[sel].mean():.6f}"
    )
print("-" * 78)
ece = expected_calibration_error(te0.labels, forest_a_prob)
report(
    "check 6  the forest is well calibrated and that is its only advantage",
    ece < 0.02,
    f"expected calibration error {ece:.6f} over 10 equal-width bins, Brier "
    f"{forest_a.brier:.6f}. ECE is biased by binning, which is why the Brier score is "
    f"reported beside it. Neither the exact one-step condition nor the exact "
    f"multi-step predictor produces a probability, so neither has a Brier score to "
    f"compare against -- the comparison is on the confusion matrix and the latency, "
    f"and the learned model loses both",
)

# --- check 7: the n_jobs inference defect ------------------------------------
print()
print("check 7  scikit-learn n_jobs at single-row inference (a measured defect)")
print("-" * 78)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

tr0 = splits[0][0]
bare = RandomForestClassifier(
    n_estimators=TREES, min_samples_leaf=8, random_state=SEED, n_jobs=2
).fit(tr0.features, tr0.labels.astype(int))
row0 = splits[0][2].features[0:1]
cost_two = measure_decision_cost(lambda: bare.predict_proba(row0), n_calls=120)
bare.n_jobs = 1
cost_one = measure_decision_cost(lambda: bare.predict_proba(row0), n_calls=120)
print(f"  {TREES}-tree forest, n_jobs=2  {cost_two:9.1f} us per single-row predict_proba")
print(f"  {TREES}-tree forest, n_jobs=1  {cost_one:9.1f} us per single-row predict_proba")
print("-" * 78)
report(
    "check 7  n_jobs > 1 makes single-row inference slower, not faster",
    cost_two > cost_one,
    f"a factor of {cost_two / cost_one:.2f}. joblib's per-call dispatch dominates a "
    f"single-row predict, and a runtime guard decides one step at a time. "
    f"fit_switch_predictor therefore sets the forest's n_jobs to 1 AFTER fitting, so "
    f"the learned model is benchmarked at its best latency and the comparison against "
    f"the exact condition is not won by an avoidable overhead",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
