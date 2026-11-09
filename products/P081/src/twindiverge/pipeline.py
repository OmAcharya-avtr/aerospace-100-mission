"""The ordered pipeline, and the learned study. One code path for every report.

The order of operations is the product
-------------------------------------
:func:`analyse_case` runs, in this order and no other:

1. **Baseline 1**, :mod:`twindiverge.identifiability`: which parameters can
   this telemetry set distinguish at all? Its verdict is binding on everything
   below.
2. **Detection**: is the residual distinguishable from sensor noise? A residual
   that is not gets the answer "no attributable change" and nothing else.
3. **Baseline 2**, :mod:`twindiverge.attribution`: estimate the identifiable
   combinations; refuse the parameters baseline 1 refused; and, under a declared
   and non-verifiable sparsity assumption, offer a single-parameter attribution
   together with the family of perturbations it cannot be distinguished from.
4. **The learned classifier**, only where it is being measured, never as a
   substitute for 1 to 3.

**Baseline 3**, :mod:`twindiverge.profile`, is the reference the linearised
covariance of baselines 1 and 2 is scored against. It is not in the per-case
path because it costs a nonlinear optimisation per grid point per direction;
``validation/validate_profile.py`` runs it and reports the scores.

The five answer sources compared in :func:`learned_study`
---------------------------------------------------------
``analytic-gated``
    What this product actually delivers. The detection test, then baseline 1's
    verdict. On a telemetry set with a null space it refuses every
    per-parameter attribution, so its accuracy on single-parameter cases is
    **zero by construction** -- and that is the honest answer, not a failure.
    It is tabulated at zero rather than omitted.
``analytic-sensitivity``
    Equation (14), the first-order sensitivity attribution the specification
    names as baseline 2's attributor. Always answers.
``analytic-constrained``
    Equation (15), the constrained single-parameter maximum likelihood, gated
    by the detection test. The strongest honest analytic attributor on this
    population, and the one the learned component is benchmarked against. The
    specified estimator of equation (14) is reported beside it, per the build
    guide's rule that a specified baseline which measures weakly is reported
    next to the stronger one rather than dropped.
``learned-parameter``
    The specified learned component. Five classes, no way to refuse.
``learned-refusal``
    The ablation with a sixth class, trained with pure-scale cases.

Scoring convention
------------------
An answer is correct when it names the parameter that moved; for a case whose
true cause is in the unidentifiable subspace (:data:`twindiverge.synthetic.
UNIDENTIFIABLE`) or spread over several parameters
(:data:`twindiverge.synthetic.MULTI`), the only correct answers are "no
attributable change" and "refused". This is the convention a user would apply:
being told "k_t" when the truth is "either J alone or the other four together,
and this telemetry cannot tell" is wrong, and being told "this telemetry cannot
tell" is right.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .attribution import Attribution, attribute
from .features import extract
from .identifiability import IdentifiabilityReport, analyse
from .learned import Prediction, ResidualClassifier
from .plant import (
    NOMINAL,
    PARAMETER_NAMES,
    WheelParameters,
    simulate_with_sensitivities,
)
from .profile import fit
from .scoring import (
    CONFIDENT_THRESHOLD,
    Proportion,
    brier_score,
    calibration_table,
    confident_error_rate,
    proportion,
)
from .synthetic import MULTI, UNIDENTIFIABLE, Case, generate_cases, observe
from .telemetry import TelemetrySet, design_matrix, residual_vector

__all__ = [
    "METHODS",
    "REFUSAL",
    "CaseAnalysis",
    "LearnedStudy",
    "MethodScore",
    "PopulationScore",
    "analyse_case",
    "learned_study",
    "montecarlo_directions",
]

#: The five answer sources, in the order they are tabulated.
METHODS: Final[tuple[str, ...]] = (
    "analytic-gated",
    "analytic-sensitivity",
    "analytic-constrained",
    "learned-parameter",
    "learned-refusal",
)

#: Printed name of the refusal answer.
REFUSAL: Final[str] = "REFUSED"


@dataclass(frozen=True)
class CaseAnalysis:
    """One case taken through steps 1 to 3 of the ordered pipeline."""

    case: Case
    identifiability: IdentifiabilityReport
    attribution: Attribution
    features: np.ndarray
    attribution_residual: np.ndarray

    @property
    def gated_answer(self) -> str:
        """What the product delivers: a parameter name, ``REFUSED``, or no change."""
        if not self.attribution.detected:
            return "UNIDENTIFIABLE"
        if self.identifiability.refused:
            return REFUSAL
        return self.attribution.constrained_choice

    @property
    def constrained_answer(self) -> str:
        """Equation (15), gated by detection only."""
        if not self.attribution.detected:
            return "UNIDENTIFIABLE"
        return self.attribution.constrained_choice

    @property
    def sensitivity_answer(self) -> str:
        """Equation (14). Ungated: it always names a parameter."""
        return self.attribution.sensitivity_choice

    @property
    def ambiguity_declared(self) -> bool:
        """Whether the report lists perturbations it cannot distinguish from its answer."""
        return self.attribution.equivalence_family().shape[0] > 1

    @property
    def truth_name(self) -> str:
        """The true label, in the same vocabulary as the answers."""
        if self.case.label == UNIDENTIFIABLE:
            return "UNIDENTIFIABLE"
        if self.case.label == MULTI:
            return "MULTI"
        return PARAMETER_NAMES[self.case.label]


def analyse_case(
    telemetry: TelemetrySet,
    case: Case,
    identifiability: IdentifiabilityReport | None = None,
    twin=None,
) -> CaseAnalysis:
    """Run steps 1 to 3 on one case.

    ``identifiability`` and ``twin`` may be supplied to avoid recomputing what
    does not depend on the case: baseline 1's report and the nominal twin
    trajectory are the same for every case on a given telemetry set, and
    recomputing them per case is the difference between a study that takes
    seconds and one that takes minutes.
    """
    report = analyse(telemetry, NOMINAL) if identifiability is None else identifiability
    reference = (
        simulate_with_sensitivities(NOMINAL, telemetry.campaign) if twin is None else twin
    )
    observed = observe(telemetry, case)
    residual = residual_vector(telemetry, observed, reference)
    return CaseAnalysis(
        case=case,
        identifiability=report,
        attribution=attribute(telemetry, report, residual, twin=reference),
        features=extract(telemetry, observed, reference),
        attribution_residual=residual,
    )


def _correct(answer: str, truth: str) -> bool:
    """The scoring convention of the module docstring."""
    if truth in ("UNIDENTIFIABLE", "MULTI"):
        return answer in ("UNIDENTIFIABLE", REFUSAL)
    return answer == truth


@dataclass(frozen=True)
class MethodScore:
    """One method's score on one population."""

    method: str
    accuracy: Proportion
    mean_confidence: float
    confident_error: Proportion
    refusal_rate: Proportion
    ambiguity_declared: Proportion
    brier: float
    calibration: tuple[tuple[float, float, int, float, float], ...] = ()


@dataclass(frozen=True)
class PopulationScore:
    """Every method's score on one case population."""

    population: str
    cases: int
    detection_rate: Proportion
    methods: dict[str, MethodScore] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedStudy:
    """The whole learned study: fitted models, scores, and what it cost."""

    telemetry_name: str
    train_cases: int
    test_cases: int
    populations: tuple[PopulationScore, ...]
    feature_importances: np.ndarray
    fit_seconds: float
    total_seconds: float
    refusal_variant_labels: tuple[int, ...]
    setup_seconds: float = float("nan")
    analytic_seconds_per_case: float = float("nan")
    learned_seconds_per_case: float = float("nan")
    budget_cases: int = 0

    def population(self, name: str) -> PopulationScore:
        """The score block for one population, by name."""
        for block in self.populations:
            if block.population == name:
                return block
        raise KeyError(f"no population named {name!r} in this study")


def _score_method(
    method: str,
    answers: list[str],
    truths: list[str],
    confidences: list[float],
    probabilities: np.ndarray | None,
    truth_columns: list[int] | None,
    declared: list[bool],
    refusals: list[bool],
    level: float,
) -> MethodScore:
    correct = np.array(
        [_correct(answer, truth) for answer, truth in zip(answers, truths, strict=True)]
    )
    confidence = np.array(confidences, dtype=float)
    brier = (
        brier_score(probabilities, np.array(truth_columns, dtype=int))
        if probabilities is not None and truth_columns is not None
        else float("nan")
    )
    return MethodScore(
        method=method,
        accuracy=proportion(int(correct.sum()), len(correct), level),
        mean_confidence=(
            float(np.nanmean(confidence))
            if confidence.size and not np.all(np.isnan(confidence))
            else float("nan")
        ),
        confident_error=confident_error_rate(
            confidence, correct, CONFIDENT_THRESHOLD, level
        ),
        refusal_rate=proportion(int(np.sum(refusals)), len(refusals), level),
        ambiguity_declared=proportion(int(np.sum(declared)), len(declared), level),
        brier=brier,
        calibration=calibration_table(confidence, correct),
    )


def learned_study(
    telemetry: TelemetrySet,
    train_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    null_seeds: tuple[int, ...],
    level: float = 0.95,
) -> LearnedStudy:
    """Fit both learned variants on ``train_seeds`` and score everything on the rest.

    ``train_seeds``, ``test_seeds`` and ``null_seeds`` must be disjoint. The
    split is by SEED, which here is the same as by case, because each case is
    generated from its own seed and no two cases share a trajectory or a noise
    draw.
    """
    for first, second, name in (
        (train_seeds, test_seeds, "train/test"),
        (train_seeds, null_seeds, "train/null"),
        (test_seeds, null_seeds, "test/null"),
    ):
        overlap = set(first) & set(second)
        if overlap:
            raise ValueError(
                f"the {name} seed sets overlap on {sorted(overlap)[:5]}; a split that "
                "leaks is not a split"
            )
    started = time.perf_counter()
    report = analyse(telemetry, NOMINAL)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)

    def analysed(seeds: tuple[int, ...], kind: str) -> tuple[CaseAnalysis, ...]:
        return tuple(
            analyse_case(telemetry, case, report, twin)
            for case in generate_cases(seeds, kind)
        )

    train_single = analysed(train_seeds, "single")
    train_scale = analysed(train_seeds, "scale")
    populations = {
        "single": analysed(test_seeds, "single"),
        "weak": analysed(test_seeds, "weak"),
        "equivalent": analysed(test_seeds, "equivalent"),
        "scale": analysed(null_seeds, "scale"),
        "none": analysed(null_seeds, "none"),
    }

    fit_started = time.perf_counter()
    plain = ResidualClassifier("parameter").fit(
        np.vstack([item.features for item in train_single]),
        np.array([item.case.label for item in train_single]),
    )
    with_refusal = ResidualClassifier("with_refusal").fit(
        np.vstack([item.features for item in (*train_single, *train_scale)]),
        np.array([item.case.label for item in (*train_single, *train_scale)]),
    )
    fit_seconds = time.perf_counter() - fit_started

    blocks = []
    for name, items in populations.items():
        truths = [item.truth_name for item in items]
        declared = [item.ambiguity_declared for item in items]
        features = np.vstack([item.features for item in items])
        scores: dict[str, MethodScore] = {}

        scores["analytic-gated"] = _score_method(
            "analytic-gated",
            [item.gated_answer for item in items],
            truths,
            [float("nan")] * len(items),
            None,
            None,
            declared,
            [item.gated_answer in ("UNIDENTIFIABLE", REFUSAL) for item in items],
            level,
        )
        scores["analytic-sensitivity"] = _score_method(
            "analytic-sensitivity",
            [item.sensitivity_answer for item in items],
            truths,
            [float(item.attribution.sensitivity_share.max()) for item in items],
            None,
            None,
            declared,
            [False] * len(items),
            level,
        )
        scores["analytic-constrained"] = _score_method(
            "analytic-constrained",
            [item.constrained_answer for item in items],
            truths,
            [item.attribution.constrained_confidence for item in items],
            None,
            None,
            declared,
            [item.constrained_answer == "UNIDENTIFIABLE" for item in items],
            level,
        )
        for method, model in (("learned-parameter", plain), ("learned-refusal", with_refusal)):
            predictions: tuple[Prediction, ...] = model.predict(features)
            columns = [
                model.labels.index(item.case.label) if item.case.label in model.labels else -1
                for item in items
            ]
            scores[method] = _score_method(
                method,
                [prediction.label_name for prediction in predictions],
                truths,
                [prediction.confidence for prediction in predictions],
                np.vstack([prediction.probabilities for prediction in predictions]),
                columns,
                [False] * len(items),
                [prediction.abstains for prediction in predictions],
                level,
            )
        blocks.append(
            PopulationScore(
                population=name,
                cases=len(items),
                detection_rate=proportion(
                    int(sum(item.attribution.detected for item in items)), len(items), level
                ),
                methods=scores,
            )
        )

    # The computation budget, measured rather than argued about. The analytic
    # setup -- one integration of the joint state-and-sensitivity system plus one
    # singular value decomposition -- is paid once per telemetry set, and the
    # learned fit is paid once per training set; both are amortised over a
    # campaign and are reported separately from the per-case costs.
    sample = populations["single"]
    probe = sample[: min(len(sample), 40)]
    setup_started = time.perf_counter()
    reference_twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    probe_matrix, _ = design_matrix(telemetry, NOMINAL, trajectory=reference_twin)
    np.linalg.svd(probe_matrix, full_matrices=True)
    setup_seconds = time.perf_counter() - setup_started

    analytic_started = time.perf_counter()
    for item in probe:
        attribute(telemetry, report, item.attribution_residual, twin=reference_twin)
    analytic_per_case = (time.perf_counter() - analytic_started) / len(probe)

    learned_started = time.perf_counter()
    for item in probe:
        plain.predict_one(item.features)
    learned_per_case = (time.perf_counter() - learned_started) / len(probe)

    return LearnedStudy(
        telemetry_name=telemetry.name,
        train_cases=len(train_single),
        test_cases=len(populations["single"]),
        populations=tuple(blocks),
        feature_importances=plain.feature_importances,
        fit_seconds=fit_seconds,
        total_seconds=time.perf_counter() - started,
        refusal_variant_labels=with_refusal.labels,
        setup_seconds=setup_seconds,
        analytic_seconds_per_case=analytic_per_case,
        learned_seconds_per_case=learned_per_case,
        budget_cases=len(probe),
    )


def montecarlo_directions(
    telemetry: TelemetrySet,
    truth: WheelParameters,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear-fit amplitudes along each identifiable direction, per noise realisation.

    Returns ``(amplitudes, crb)`` where ``amplitudes`` has shape
    ``(len(seeds), rank)`` and ``crb[i] = 1 / sigma_i`` is the Cramer-Rao bound
    on the standard deviation of direction ``i``, equation (13).

    The fit is the full nonlinear least-squares fit of :func:`twindiverge.
    profile.fit`, not the linear estimator of equation (12), so the empirical
    spread this returns is the spread of a real estimator and the comparison
    against the bound is a real Cramer-Rao test rather than an algebraic
    identity. On a telemetry set with a null space the fit itself is not unique,
    but its PROJECTIONS onto the identifiable directions are, and those are
    what is returned.
    """
    from .telemetry import measure

    report = analyse(telemetry, NOMINAL)
    matrix, _ = design_matrix(telemetry, NOMINAL)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    basis = right[: report.rank]
    rows = []
    for seed in seeds:
        observed = measure(telemetry, truth, seed=seed)[0]
        offset, _ = fit(telemetry, observed)
        rows.append(basis @ offset)
    return np.vstack(rows), 1.0 / singular_values[: report.rank]
