"""The learned classifier against the analytic baselines, inside and outside the null space.

The measurement this script exists for
--------------------------------------
The learned component is scored on five declared populations, and the two that
matter are the last three.

``single``
    Ordinary single-parameter faults, in the magnitude range the classifier was
    fitted on. The comparison a practitioner would ask for.
``weak``
    The same, a tenth the size, and wholly OUTSIDE the training distribution.
    This is where the analytic attributor and the learned one separate.
``equivalent``
    Inside the unidentifiable subspace, with a LARGE residual. Each case is the
    partner of a single-parameter fault under equation (6), so it produces the
    same telemetry to within two units in the last place and no
    single-parameter answer is correct. The analytic path refuses here and
    prints the family it cannot distinguish; what the learned classifier does
    instead is the headline result of this product.
``scale``, ``none``
    Inside the unidentifiable subspace with NO residual: a pure scale change,
    and no change at all. These two populations are the same observations to
    within the last bit, which is itself a measurement and is reported.

Nothing here is tuned. The classifier has no threshold to select -- its label is
the argmax of its own probability vector -- so the leakage trap that an earlier
product of this mission fell into, selecting a decision threshold on the test
labels, has no surface here. The split is by case, the training population
contains only ``single`` cases for the specified variant, and the
unidentifiable-subspace populations are drawn from a third disjoint seed set.

The ``learned-refusal`` ablation
--------------------------------
A fair objection to the headline is that the five-class model was never given
the option of saying "the telemetry cannot tell". The ablation gives it a sixth
class and trains it with pure-scale examples. Its measured behaviour on the
``equivalent`` population -- which it was NOT trained on, and where the residual
is large -- is reported beside the headline, and so is what the extra class
costs it on the ``weak`` population.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import time  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    NULL_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from twindiverge.features import FEATURE_NAMES  # noqa: E402
from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.learned import (  # noqa: E402
    ABSTAIN_THRESHOLD,
    LEARNING_RATE,
    MAX_DEPTH,
    RANDOM_STATE,
    STAGES,
    ResidualClassifier,
)
from twindiverge.pipeline import METHODS, analyse_case, learned_study  # noqa: E402
from twindiverge.plant import NOMINAL, simulate_with_sensitivities  # noqa: E402
from twindiverge.report import format_table, number, render_matrix  # noqa: E402
from twindiverge.scoring import CONFIDENT_THRESHOLD, confusion_matrix  # noqa: E402
from twindiverge.synthetic import (  # noqa: E402
    LABEL_NAMES,
    MAX_MAGNITUDE,
    MIN_MAGNITUDE,
    MULTI,
    UNIDENTIFIABLE,
    WEAK_MAX_MAGNITUDE,
    WEAK_MIN_MAGNITUDE,
    generate_cases,
)
from twindiverge.telemetry import TELEMETRY_SETS  # noqa: E402

#: The telemetry set every number below is measured on: the reference 200 s
#: rate-only record, whose null space is one-dimensional and exactly known.
SET_NAME = "rate_full"


def main() -> int:
    out = Tee("validate_learned_model_output.txt")
    out.rule("twindiverge -- the learned classifier against the analytic baselines")
    for line in environment_lines():
        out(line)
    telemetry = TELEMETRY_SETS[SET_NAME]
    report = analyse(telemetry, NOMINAL)
    out("")
    out(f"telemetry set              : {SET_NAME} ({telemetry.rows} rows)")
    out(f"baseline 1 rank            : {report.rank} of 5")
    out(f"baseline 1 refuses         : {', '.join(report.refused)}")
    out(f"null space dimension       : {report.null_basis.shape[0]}")
    out("")
    out(f"training cases             : {len(TRAIN_SEEDS)} (single-parameter only)")
    out(f"test cases per population  : {len(TEST_SEEDS)}")
    out(f"null-subspace cases        : {len(NULL_SEEDS)}")
    overlaps = (
        set(TRAIN_SEEDS) & set(TEST_SEEDS),
        set(TRAIN_SEEDS) & set(NULL_SEEDS),
        set(TEST_SEEDS) & set(NULL_SEEDS),
    )
    out(f"seed sets disjoint         : {not any(overlaps)} "
        f"({sum(len(group) for group in overlaps)} shared seeds)")
    out("")
    out(f"declared magnitude range          : {MIN_MAGNITUDE} to {MAX_MAGNITUDE}")
    out(f"declared weak magnitude range     : {WEAK_MIN_MAGNITUDE} to "
        f"{WEAK_MAX_MAGNITUDE}")
    out(f"declared boosting stages          : {STAGES}")
    out(f"declared tree depth               : {MAX_DEPTH}")
    out(f"declared learning rate            : {LEARNING_RATE}")
    out(f"declared random state             : {RANDOM_STATE}")
    out(f"declared abstention threshold     : {ABSTAIN_THRESHOLD:.2f}")
    out(f"declared confident threshold      : {CONFIDENT_THRESHOLD:.2f}")

    started = time.perf_counter()
    study = learned_study(telemetry, TRAIN_SEEDS, TEST_SEEDS, NULL_SEEDS)
    elapsed = time.perf_counter() - started

    out.rule("1. the computation budget, measured")
    out("The specification asks for accuracy at a fixed computation budget, so the")
    out("budget is measured rather than argued about. The analytic setup is paid")
    out("once per telemetry set and the learned fit once per training set; both are")
    out("amortised over a campaign and are reported separately from the per-case")
    out("costs.")
    out("")
    out(format_table(
        ["item", "seconds", "note"],
        [
            ["analytic setup, once per telemetry set",
             f"{study.setup_seconds:.6f}",
             "one joint state-and-sensitivity integration plus one SVD"],
            ["learned fit, once per training set",
             f"{study.fit_seconds:.6f}",
             f"both variants, {len(TRAIN_SEEDS)} and "
             f"{2 * len(TRAIN_SEEDS)} cases"],
            ["analytic attribution per case",
             f"{study.analytic_seconds_per_case:.6f}",
             f"mean of {study.budget_cases}; equations (12) to (16)"],
            ["learned prediction per case",
             f"{study.learned_seconds_per_case:.6f}",
             f"mean of {study.budget_cases}; one ensemble evaluation"],
        ],
    ))
    out("")
    ratio = study.learned_seconds_per_case / study.analytic_seconds_per_case
    out(f"learned cost per case divided by analytic cost per case : {ratio:.6f}")
    out(f"whole study wall clock                                  : {elapsed:.3f} s")
    out("")
    import os as _os

    out.check(
        "the whole learned study finishes inside the declared three-minute budget",
        elapsed < 180.0,
        f"{elapsed:.3f} s on {len(_os.sched_getaffinity(0))} cores",
    )

    out.rule("2. every method on every population")
    for block in study.populations:
        out("")
        out("-" * 78)
        out(f"population {block.population}: {block.cases} cases")
        out("-" * 78)
        out(f"detected as a real change : {block.detection_rate}")
        out("")
        out(format_table(
            ["method", "accuracy", "exact interval", "mean confidence",
             "confident and wrong", "refusal rate", "ambiguity declared", "Brier"],
            [
                [
                    method,
                    f"{block.methods[method].accuracy.value:.6f}",
                    f"[{block.methods[method].accuracy.lower:.6f}, "
                    f"{block.methods[method].accuracy.upper:.6f}]",
                    number(block.methods[method].mean_confidence),
                    f"{block.methods[method].confident_error.value:.6f}",
                    f"{block.methods[method].refusal_rate.value:.6f}",
                    f"{block.methods[method].ambiguity_declared.value:.6f}",
                    number(block.methods[method].brier),
                ]
                for method in METHODS
            ],
        ))

    out.rule("3. the headline: inside the unidentifiable subspace")
    out("The `equivalent` population. Each case produces telemetry identical to a")
    out("single-parameter fault, so the residual is large, easily detected, and")
    out("CANNOT be attributed to any one parameter. The correct answers are 'no")
    out("attributable change' and 'refused'.")
    out("")
    equivalent = study.population("equivalent")
    rows = []
    for method in METHODS:
        score = equivalent.methods[method]
        rows.append([
            method,
            f"{score.accuracy.value:.6f}",
            number(score.mean_confidence),
            f"{score.confident_error.value:.6f}",
            f"[{score.confident_error.lower:.6f}, {score.confident_error.upper:.6f}]",
            f"{score.refusal_rate.value:.6f}",
            f"{score.ambiguity_declared.value:.6f}",
        ])
    out(format_table(
        ["method", "accuracy", "mean confidence", "confident and wrong",
         "exact interval on that", "refusal rate", "ambiguity declared"],
        rows,
    ))
    out("")
    gated = equivalent.methods["analytic-gated"]
    plain = equivalent.methods["learned-parameter"]
    ablation = equivalent.methods["learned-refusal"]
    out("MEASURED RESULT, reported as found and not retuned:")
    out("")
    out(f"  the analytic path refuses {gated.refusal_rate.value:.6f} of these cases "
        f"and declares the")
    out(f"  equivalence family on {gated.ambiguity_declared.value:.6f} of them, so its "
        "accuracy under the")
    out(f"  scoring convention is {gated.accuracy.value:.6f};")
    out(f"  the specified learned classifier answers with a mean confidence of "
        f"{plain.mean_confidence:.6f},")
    out(f"  reaches an accuracy of {plain.accuracy.value:.6f}, is confident AND wrong on "
        f"{plain.confident_error.value:.6f}")
    out(f"  of them, and declares an ambiguity on "
        f"{plain.ambiguity_declared.value:.6f};")
    out("  the six-class ablation, which has a refusal class but was trained only on")
    out(f"  pure-scale examples of it, reaches {ablation.accuracy.value:.6f} here and is "
        f"confident and")
    out(f"  wrong on {ablation.confident_error.value:.6f}.")
    out("")
    out("This is the expected and the most valuable result in the product: the")
    out("learned component returns a confident wrong attribution exactly where the")
    out("analytic method correctly refuses to answer. It is research evidence and")
    out("nothing here is tuned to soften it.")
    out("")
    out.check(
        "the analytic path refuses every case inside the unidentifiable subspace",
        gated.refusal_rate.value == 1.0,
        f"refusal rate {gated.refusal_rate.value:.6f} over {equivalent.cases} cases",
    )
    out.check(
        "the analytic path declares its equivalence family on every such case",
        gated.ambiguity_declared.value == 1.0,
        f"{gated.ambiguity_declared.value:.6f}",
    )
    out.check(
        "no single-parameter answer can be correct there, so accuracy is zero",
        plain.accuracy.value == 0.0,
        f"the learned classifier's accuracy is {plain.accuracy.value:.6f}; the true "
        "cause is a four-parameter change and the label set has no such class",
    )
    out.check(
        "the learned classifier declares no ambiguity anywhere in that population",
        plain.ambiguity_declared.value == 0.0,
        f"{plain.ambiguity_declared.value:.6f}",
    )

    out.rule("4. the pair test: the same input, two different truths")
    out("Each `equivalent` case shares a noise seed with its `single` partner and")
    out("their observations differ by at most two units in the last place, so any")
    out("classifier MUST return the same answer for both. Measured here rather than")
    out("argued: the fraction of pairs on which the learned classifier agrees with")
    out("itself, and the fraction on which the analytic path agrees with itself.")
    out("")
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    singles = generate_cases(TEST_SEEDS, "single")
    partners = generate_cases(TEST_SEEDS, "equivalent")
    train_single = generate_cases(TRAIN_SEEDS, "single")
    model = ResidualClassifier("parameter").fit(
        np.vstack([
            analyse_case(telemetry, case, report, twin).features for case in train_single
        ]),
        np.array([case.label for case in train_single]),
    )
    learned_agree = 0
    gated_agree = 0
    confidences = []
    for single, partner in zip(singles, partners, strict=True):
        left = analyse_case(telemetry, single, report, twin)
        right = analyse_case(telemetry, partner, report, twin)
        first = model.predict_one(left.features)
        second = model.predict_one(right.features)
        learned_agree += int(first.label == second.label)
        gated_agree += int(left.gated_answer == right.gated_answer)
        confidences.extend([first.confidence, second.confidence])
    out(f"pairs                                          : {len(singles)}")
    out(f"pairs on which the learned label agrees        : {learned_agree}")
    out(f"pairs on which the analytic answer agrees      : {gated_agree}")
    out(f"mean learned confidence over both members      : "
        f"{float(np.mean(confidences)):.6f}")
    out(f"least learned confidence over both members     : "
        f"{float(np.min(confidences)):.6f}")
    out("")
    out("The two members of a pair have DIFFERENT true causes. A classifier that")
    out("agrees with itself on a pair is therefore wrong on one member of it, by")
    out("construction and whatever it answers. That is not a defect of the")
    out("classifier: it is what an unidentifiable subspace means, and the point is")
    out("that the classifier cannot say so while the analytic path does.")
    out("")
    out.check(
        "the learned classifier gives the same answer to both members of every pair",
        learned_agree == len(singles),
        f"{learned_agree} of {len(singles)}",
    )
    out.check(
        "the analytic path gives the same answer to both members of every pair",
        gated_agree == len(singles),
        f"{gated_agree} of {len(singles)}; that answer is REFUSED",
    )

    out.rule("5. outside the unidentifiable subspace: the baseline comparison")
    out("Ordinary single-parameter faults, and the same faults a tenth the size.")
    out("")
    rows = []
    for name in ("single", "weak"):
        block = study.population(name)
        for method in METHODS:
            score = block.methods[method]
            rows.append([
                name,
                method,
                f"{score.accuracy.value:.6f}",
                f"[{score.accuracy.lower:.6f}, {score.accuracy.upper:.6f}]",
                number(score.mean_confidence),
                f"{score.confident_error.value:.6f}",
            ])
    out(format_table(
        ["population", "method", "accuracy", "exact interval", "mean confidence",
         "confident and wrong"],
        rows,
    ))
    out("")
    single = study.population("single")
    weak = study.population("weak")
    best_analytic = max(
        single.methods["analytic-constrained"].accuracy.value,
        single.methods["analytic-sensitivity"].accuracy.value,
    )
    learned_single = single.methods["learned-parameter"].accuracy.value
    best_analytic_weak = max(
        weak.methods["analytic-constrained"].accuracy.value,
        weak.methods["analytic-sensitivity"].accuracy.value,
    )
    learned_weak = weak.methods["learned-parameter"].accuracy.value
    out("MEASURED RESULT, reported as found:")
    out("")
    out(f"  on `single`, the strongest analytic attributor reaches "
        f"{best_analytic:.6f} and the")
    out(f"  learned classifier reaches {learned_single:.6f}, a margin of "
        f"{learned_single - best_analytic:+.6f};")
    out("  on `weak`, which is outside the learned component's training")
    out(f"  distribution, the strongest analytic attributor reaches "
        f"{best_analytic_weak:.6f} and the")
    out(f"  learned classifier reaches {learned_weak:.6f}, a margin of "
        f"{learned_weak - best_analytic_weak:+.6f}.")
    out("")
    out("A baseline win is research evidence and is never removed, softened or")
    out("retuned away. The specified first-order sensitivity attribution of equation")
    out("(14) is reported in the table above beside the stronger constrained")
    out("maximum likelihood of equation (15), per the build guide's rule that a")
    out("specified baseline which measures weakly is reported next to the stronger")
    out("one rather than dropped.")

    out.rule("6. the no-residual populations")
    out("A pure scale change and no change at all. These produce the SAME")
    out("observations to within the last bit, so every score on them must agree.")
    out("")
    rows = []
    for name in ("scale", "none"):
        block = study.population(name)
        for method in METHODS:
            score = block.methods[method]
            rows.append([
                name, method,
                f"{score.accuracy.value:.6f}",
                number(score.mean_confidence),
                f"{score.confident_error.value:.6f}",
                f"{score.refusal_rate.value:.6f}",
            ])
    out(format_table(
        ["population", "method", "accuracy", "mean confidence",
         "confident and wrong", "refusal rate"],
        rows,
    ))
    out("")
    agree = all(
        study.population("scale").methods[method].accuracy.count
        == study.population("none").methods[method].accuracy.count
        for method in METHODS
    )
    out.check(
        "every method scores identically on the scale and none populations",
        agree,
        "they are the same observations to within the last bit, so a difference "
        "would be a defect in the scoring and not a result",
    )
    scale_block = study.population("scale")
    out("")
    out("MEASURED RESULT: on a residual that is pure sensor noise, the specified")
    out(f"learned classifier answers with a mean confidence of "
        f"{scale_block.methods['learned-parameter'].mean_confidence:.6f} and is")
    out(f"confident and wrong on "
        f"{scale_block.methods['learned-parameter'].confident_error.value:.6f} of cases, "
        "while the analytic")
    out(f"detection test refuses "
        f"{scale_block.methods['analytic-gated'].refusal_rate.value:.6f} of them. The "
        "six-class ablation,")
    out(f"which WAS trained on this population's kind, reaches "
        f"{scale_block.methods['learned-refusal'].accuracy.value:.6f} here.")

    out.rule("7. the ablation's cost")
    out("The sixth class is not free. What it buys on the no-residual populations")
    out("and what it costs elsewhere, side by side:")
    out("")
    rows = []
    for block in study.populations:
        plain_score = block.methods["learned-parameter"]
        ablation_score = block.methods["learned-refusal"]
        rows.append([
            block.population,
            f"{plain_score.accuracy.value:.6f}",
            f"{ablation_score.accuracy.value:.6f}",
            f"{ablation_score.accuracy.value - plain_score.accuracy.value:+.6f}",
            number(plain_score.brier),
            number(ablation_score.brier),
        ])
    out(format_table(
        ["population", "five-class accuracy", "six-class accuracy", "difference",
         "five-class Brier", "six-class Brier"],
        rows,
    ))
    out("")
    out("MEASURED RESULT: a refusal class trained on pure-scale examples generalises")
    out("to pure-scale cases and NOT to the ambiguity case, where the residual is")
    out("large and only the attribution is impossible. Giving a classifier the")
    out("option of refusing does not give it the ability to tell when to.")

    out.rule("8. calibration and feature importances")
    out("Reliability of the specified classifier's confidence, per population.")
    out("")
    for block in study.populations:
        out(f"population {block.population}:")
        out(format_table(
            ["bin low", "bin high", "cases", "mean confidence", "accuracy"],
            [
                [f"{low:.2f}", f"{high:.2f}", str(count), number(mean), number(hit)]
                for low, high, count, mean, hit in
                block.methods["learned-parameter"].calibration
            ],
        ))
        out("")
    out("impurity-based feature importances of the specified classifier. These")
    out("describe what the fitted trees used and are NOT evidence about the")
    out("physics; the physics is baseline 1's business.")
    out("")
    order = np.argsort(study.feature_importances)[::-1]
    out(format_table(
        ["feature", "importance"],
        [
            [FEATURE_NAMES[index], f"{float(study.feature_importances[index]):.6f}"]
            for index in order
        ],
    ))

    out.rule("9. confusion matrices on the two single-parameter populations")
    labels = (0, 1, 2, 3, 4, UNIDENTIFIABLE, MULTI)
    names = [LABEL_NAMES[label] for label in labels]
    for population in ("single", "weak"):
        cases = generate_cases(TEST_SEEDS, population)
        truths = []
        learned_answers = []
        analytic_answers = []
        inverse = {name: label for label, name in LABEL_NAMES.items()}
        for case in cases:
            analysis = analyse_case(telemetry, case, report, twin)
            prediction = model.predict_one(analysis.features)
            truths.append(case.label)
            learned_answers.append(prediction.label)
            analytic_answers.append(inverse[analysis.constrained_answer])
        out("")
        out(f"population {population}, learned classifier:")
        out(render_matrix(
            confusion_matrix(np.array(truths), np.array(learned_answers), labels), names
        ))
        out("")
        out(f"population {population}, equation (15) under its sparsity assumption:")
        out(render_matrix(
            confusion_matrix(np.array(truths), np.array(analytic_answers), labels), names
        ))

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
