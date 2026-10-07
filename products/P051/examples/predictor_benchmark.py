"""The learned switch predictor against the exact analytic condition.

Writes ``../screenshots/predictor_benchmark.png``.

Top left: Task A, the switching condition's own criterion -- does the guard fire
at this step. The exact computation is right by construction, so precision,
recall and F1 are all exactly 1. The 150-tree random forest, given only raw
state and reference and never the support-function margin, reaches F1 0.98 and
no higher. This is the published loss.

Top right: single-row decision latency on a log scale, against the same bar
order. The exact condition costs single-digit microseconds; the forest costs
milliseconds. The learned model loses on cost by roughly three orders of
magnitude and it loses at every forest size, so the loss is not a
hyperparameter.

Bottom left: Task B, anticipating the switch five steps ahead. Here the learned
model WINS on F1, and the reason is stated rather than claimed: the exact
predictor answers "may fire under some admissible disturbance sequence", which
over-predicts a realised episode by construction. Its recall is high and its
precision is poor. That is a different question, not a worse answer.

Bottom right: the reliability diagram of the forest's isotonic-calibrated
probability on Task A. This is the one thing the exact computation cannot
produce at all: it has no probability output, so it has no Brier score and no
calibration curve. If a learned predictor earns a place in a runtime-assurance
architecture, this panel is where the case has to be made, not in the
accuracy bars.

Runtime: about 35 s on two contended cores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from simplexguard import (  # noqa: E402
    SimplexGuard,
    build_dataset,
    exact_predictor_scores,
    fit_switch_predictor,
    guard_condition_scores,
    measure_decision_cost,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    score_binary,
)

OUT = ROOT / "screenshots" / "predictor_benchmark.png"
SEED = 5101
EPISODES = 40
STEPS = 1200
LEAD = 5
TREES = 150


def main() -> None:
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    invariant = robust_invariant_set(plant, baseline).polytope
    guard = SimplexGuard(plant, baseline, invariant)

    data = {
        lead: build_dataset(plant, guard, performance, EPISODES, STEPS, SEED, lead)
        for lead in (0, LEAD)
    }
    splits = {
        lead: (
            d.select_episodes(np.arange(0, 26)),
            d.select_episodes(np.arange(26, 32)),
            d.select_episodes(np.arange(32, 40)),
        )
        for lead, d in data.items()
    }

    def learned(lead, kind, trees, label):
        tr, ca, te = splits[lead]
        model = fit_switch_predictor(tr, ca, kind, n_estimators=trees, random_state=SEED)
        prob = model.predict_proba(te.features)[:, 1]
        row = te.features[0:1]
        cost = measure_decision_cost(lambda: model.predict_proba(row), n_calls=120)
        return prob, score_binary(
            label, te.labels, prob >= 0.5, probability=prob,
            microseconds_per_decision=cost
        )

    te0 = splits[0][2]
    exact_a = guard_condition_scores(te0, guard, performance)
    prob_a, forest_a = learned(0, "forest", TREES, f"forest {TREES}")
    _, logit_a = learned(0, "logistic", TREES, "logistic")

    teL = splits[LEAD][2]
    exact_wc = exact_predictor_scores(teL, guard, performance, LEAD, "worst_case")
    exact_nm = exact_predictor_scores(teL, guard, performance, LEAD, "nominal")
    persistence = score_binary(
        "persistence", teL.labels, teL.fires_now, score=teL.fires_now.astype(float)
    )
    _, forest_b = learned(LEAD, "forest", TREES, f"forest {TREES}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # Task A bars
    ax = axes[0, 0]
    names_a = ["exact\ncondition", f"forest\n{TREES} trees", "logistic"]
    rows_a = [exact_a, forest_a, logit_a]
    width = 0.26
    pos = np.arange(len(names_a))
    for i, (metric, colour) in enumerate(
        (("precision", "tab:blue"), ("recall", "tab:orange"), ("f1", "tab:green"))
    ):
        values = [np.nan_to_num(getattr(r, metric), nan=0.0) for r in rows_a]
        ax.bar(pos + (i - 1) * width, values, width, color=colour, label=metric)
        for p, v in zip(pos + (i - 1) * width, values, strict=True):
            ax.text(p, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(pos)
    ax.set_xticklabels(names_a)
    ax.text(
        pos[-1], 0.08,
        "no positive predictions\nat threshold 0.5:\nprecision undefined,\nplotted as 0",
        ha="center", fontsize=7, color="0.3",
    )
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("score on the held-out episodes")
    ax.set_title(
        f"TASK A (lead 0): the condition's own criterion\n"
        f"test base rate {te0.base_rate:.4f}, {len(te0)} steps, 8 held-out episodes",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Latency
    ax = axes[0, 1]
    sizes = [20, 50, 150, 300]
    latencies, f1s = [], []
    for trees in sizes:
        _, s = learned(0, "forest", trees, f"forest {trees}")
        latencies.append(s.microseconds_per_decision)
        f1s.append(s.f1)
    ax.plot(sizes, latencies, marker="o", color="tab:red", label="forest, single row")
    ax.axhline(
        exact_a.microseconds_per_decision,
        color="tab:blue",
        ls="--",
        label=f"exact one-step condition ({exact_a.microseconds_per_decision:.1f} us)",
    )
    ax.axhline(
        exact_wc.microseconds_per_decision,
        color="tab:cyan",
        ls=":",
        label=f"exact {LEAD}-step predictor "
        f"({exact_wc.microseconds_per_decision:.1f} us)",
    )
    ax.axhline(
        plant.dt * 1e6, color="0.3", ls="-.", label=f"one sample interval ({plant.dt:g} s)"
    )
    ax.set_yscale("log")
    ax.set_xlabel("trees in the forest")
    ax.set_ylabel("microseconds per single-row decision")
    ax.set_title(
        "latency: the learned model loses at every size\n"
        f"20 trees already costs {latencies[0] / exact_a.microseconds_per_decision:.0f}x "
        f"the exact condition, at F1 {f1s[0]:.4f}",
        fontsize=10,
    )
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Task B bars
    ax = axes[1, 0]
    names_b = [
        "exact\nworst case",
        "exact\nnominal",
        "persistence",
        f"forest\n{TREES} trees",
    ]
    rows_b = [exact_wc, exact_nm, persistence, forest_b]
    pos = np.arange(len(names_b))
    for i, (metric, colour) in enumerate(
        (("precision", "tab:blue"), ("recall", "tab:orange"), ("f1", "tab:green"))
    ):
        values = [np.nan_to_num(getattr(r, metric), nan=0.0) for r in rows_b]
        ax.bar(pos + (i - 1) * width, values, width, color=colour, label=metric)
        for p, v in zip(pos + (i - 1) * width, values, strict=True):
            ax.text(p, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(pos)
    ax.set_xticklabels(names_b)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("score on the held-out episodes")
    ax.set_title(
        f"TASK B (lead {LEAD}): will the guard fire within {LEAD} steps\n"
        f"test base rate {teL.base_rate:.4f}; the learned model wins on F1 here, "
        f"on a different question",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Reliability
    ax = axes[1, 1]
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(prob_a, edges[1:-1]), 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        sel = idx == b
        if not np.any(sel):
            continue
        xs.append(float(prob_a[sel].mean()))
        ys.append(float(te0.labels[sel].mean()))
        ns.append(int(sel.sum()))
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", label="perfect calibration")
    ax.plot(xs, ys, marker="o", color="tab:purple", label="forest, isotonic")
    for x, y, n in zip(xs, ys, ns, strict=True):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5, -9), fontsize=7)
    ax.set_xlabel("mean forecast probability in bin")
    ax.set_ylabel("observed frequency in bin")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(
        f"Task A reliability: Brier {forest_a.brier:.6f}, ECE {forest_a.ece:.6f}\n"
        "the exact predictors produce no probability at all",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "simplexguard: the learned switch predictor against the exact condition "
        "(research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("  TASK A  " + exact_a.row())
    print("  TASK A  " + forest_a.row())
    print("  TASK A  " + logit_a.row())
    print("  TASK B  " + exact_wc.row())
    print("  TASK B  " + exact_nm.row())
    print("  TASK B  " + persistence.row())
    print("  TASK B  " + forest_b.row())
    for trees, lat, f1 in zip(sizes, latencies, f1s, strict=True):
        print(f"  latency  trees={trees:<4d} us/decision={lat:9.1f} F1={f1:.6f}")


if __name__ == "__main__":
    main()
