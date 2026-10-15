"""The held-out benchmark: physics alone against physics plus learned residual.

One function, :func:`run_benchmark`, does the whole comparison for one
experiment design and returns every number the README and the model card quote.
It is deliberately rigid about the protocol, because the protocol is what makes
the comparison worth anything.

Protocol
--------
1. **One calibration record.** The physics parameters are fitted to a single
   training trajectory, because that is what a calibration campaign gives you.
   The fit never sees a test trajectory.
2. **Several training trajectories for the discrepancy.** The residual model is
   trained on the residuals of all training trajectories at that one fitted
   parameter vector. Telemetry is cheaper than calibration campaigns, so this is
   the realistic asymmetry.
3. **Held-out trajectories, by phase and by seed.** Test trajectories differ
   from training trajectories in the input phase (so the dissipation and eclipse
   timing differ) *and* in the noise seed. No test sample appears in training in
   any form.
4. **Both models scored on the same test observations**, with the same metric,
   root-mean-square error in kelvin, reported per trajectory and pooled.
5. **The residual model's uncertainty is scored too**, by the empirical coverage
   of its predictive interval on the test observations.
6. **The absorption measurements of :mod:`twincalib.absorption` are run on the
   same fit**, so that the improvement and the absorption number come from one
   experiment and cannot drift apart.

If the physics-only model wins, that is the result and it is reported as the
result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorption import DEFAULT_STRONG_CUT, DriftReport, SubspaceSplit, decompose_residual
from .absorption import parameter_drift as _parameter_drift
from .fitting import CalibrationProblem, FitResult, fit
from .residual import ResidualModel, build_features
from .scenarios import Dataset, ExperimentDesign, calibration_model, make_dataset, truth_model
from .sensitivity import log_sensitivity

__all__ = [
    "BenchmarkResult",
    "TrajectoryScore",
    "coverage_fraction",
    "run_benchmark",
]


def coverage_fraction(
    error: np.ndarray, std: np.ndarray, z: float = 1.959963984540054
) -> float:
    """Fraction of errors inside ``+- z * std``, dimensionless.

    Parameters
    ----------
    error:
        Signed prediction errors, K.
    std:
        Predicted standard deviations, K, strictly positive.
    z:
        Interval half width in standard deviations. 1.0 corresponds to a nominal
        0.682689 for a Gaussian, 1.959964 to 0.95.
    """
    e = np.asarray(error, dtype=float).ravel()
    s = np.asarray(std, dtype=float).ravel()
    if e.shape != s.shape:
        raise ValueError("error and std must have the same shape")
    if e.size == 0:
        raise ValueError("at least one sample is required")
    if np.any(s <= 0.0):
        raise ValueError("std must be strictly positive")
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("z must be finite and positive")
    return float(np.mean(np.abs(e) <= z * s))


@dataclass(frozen=True)
class TrajectoryScore:
    """Per-trajectory scores on one held-out record.

    Attributes
    ----------
    phase, seed:
        Which trajectory.
    rmse_physics, rmse_combined:
        Root-mean-square error in K of the physics-only and the
        physics-plus-residual prediction.
    improvement:
        ``1 - rmse_combined / rmse_physics``, dimensionless. Negative when the
        residual model makes things worse.
    coverage_95, coverage_68:
        Empirical coverage of the residual model's predictive interval,
        dimensionless.
    mean_predicted_std:
        Mean predictive standard deviation, K.
    """

    phase: float
    seed: int
    rmse_physics: float
    rmse_combined: float
    improvement: float
    coverage_95: float
    coverage_68: float
    mean_predicted_std: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything the held-out benchmark produced.

    Attributes
    ----------
    design:
        The design it was run on.
    physics_fit:
        The physics-only fit on the single training trajectory.
    train_scores, test_scores:
        Per-trajectory scores.
    rmse_physics_pooled, rmse_combined_pooled:
        Pooled over all test observations, K.
    improvement_pooled:
        ``1 - combined/physics`` on the pooled test set, dimensionless.
    winner:
        ``"physics-only"`` or ``"physics+residual"``, decided on
        ``rmse_*_pooled`` with a stated margin -- see
        :attr:`decision_margin`.
    decision_margin:
        Relative margin the improvement must exceed to declare a winner. Inside
        the margin the winner is reported as ``"tie"``, because a comparison of
        two floating-point root-mean-square errors that differ in the fourth
        decimal is not a result.
    coverage_95_pooled, coverage_68_pooled:
        Pooled coverage of the predictive interval, dimensionless.
    discrepancy_rms:
        Root-mean-square size of the true model-form discrepancy on the test
        observations, K: the signal there was to find. Available only because
        the data is synthetic.
    noise_sigma:
        Thermometer noise standard deviation, K.
    splits:
        Subspace decomposition of the learned residual field on the first test
        trajectory, at several cuts, keyed by cut.
    drift:
        Parameter drift report.
    residual_kernel:
        The fitted Gaussian-process kernel, as a string.
    n_training_samples, n_training_used:
        Residual-model training-set size offered and used after the cap.
    """

    design: ExperimentDesign
    physics_fit: FitResult
    train_scores: tuple[TrajectoryScore, ...]
    test_scores: tuple[TrajectoryScore, ...]
    rmse_physics_pooled: float
    rmse_combined_pooled: float
    improvement_pooled: float
    winner: str
    decision_margin: float
    coverage_95_pooled: float
    coverage_68_pooled: float
    discrepancy_rms: float
    noise_sigma: float
    splits: dict[float, SubspaceSplit]
    drift: DriftReport
    residual_kernel: str
    n_training_samples: int
    n_training_used: int

    def summary(self) -> str:
        """One block of text, the verdict first."""
        lines = [
            f"held-out benchmark on design {self.design.name}",
            f"  VERDICT: {self.winner} "
            f"(decision margin {self.decision_margin:.3g} relative)",
            "",
            f"  true discrepancy RMS on the test set : "
            f"{self.discrepancy_rms:.6f} K",
            f"  thermometer noise                    : {self.noise_sigma:.6f} K",
            f"  physics-only pooled test RMSE        : "
            f"{self.rmse_physics_pooled:.6f} K",
            f"  physics+residual pooled test RMSE    : "
            f"{self.rmse_combined_pooled:.6f} K",
            f"  pooled improvement                   : "
            f"{self.improvement_pooled:+.6f}",
            f"  predictive interval coverage, 95 / 68 : "
            f"{self.coverage_95_pooled:.4f} / {self.coverage_68_pooled:.4f} "
            "(nominal 0.9500 / 0.6827)",
            "",
            "  per held-out trajectory",
            f"    {'phase':>7}{'seed':>7}{'rmse phys':>12}{'rmse comb':>12}"
            f"{'improve':>10}{'cov95':>8}",
        ]
        for s in self.test_scores:
            lines.append(
                f"    {s.phase:>7.3f}{s.seed:>7d}{s.rmse_physics:>12.6f}"
                f"{s.rmse_combined:>12.6f}{s.improvement:>+10.4f}"
                f"{s.coverage_95:>8.3f}"
            )
        return "\n".join(lines)


def _score(
    dataset: Dataset,
    design: ExperimentDesign,
    theta: np.ndarray,
    residual_model: ResidualModel,
) -> tuple[TrajectoryScore, np.ndarray, np.ndarray]:
    model = calibration_model()
    features, prediction = build_features(
        model,
        theta,
        dataset.inputs,
        design.initial_state,
        design.channels,
        design.sample_stride,
    )
    pred = residual_model.predict(features)
    err_physics = dataset.y - prediction
    err_combined = dataset.y - (prediction + pred.mean)
    score = TrajectoryScore(
        phase=dataset.phase,
        seed=dataset.seed,
        rmse_physics=float(np.sqrt(np.mean(err_physics**2))),
        rmse_combined=float(np.sqrt(np.mean(err_combined**2))),
        improvement=float(
            1.0
            - np.sqrt(np.mean(err_combined**2)) / np.sqrt(np.mean(err_physics**2))
        ),
        coverage_95=coverage_fraction(
            err_physics - pred.mean, np.sqrt(pred.std**2 + design.sigma_k**2)
        ),
        coverage_68=coverage_fraction(
            err_physics - pred.mean, np.sqrt(pred.std**2 + design.sigma_k**2), z=1.0
        ),
        mean_predicted_std=float(np.mean(pred.std)),
    )
    return score, err_physics, err_combined


def run_benchmark(
    design: ExperimentDesign,
    n_train: int = 4,
    n_test: int = 3,
    base_seed: int = 4200,
    residual_seed: int = 0,
    max_training_points: int = 600,
    decision_margin: float = 0.02,
    strong_cuts: tuple[float, ...] = (1e-1, DEFAULT_STRONG_CUT, 1e-3),
    max_nfev: int | None = 2000,
) -> BenchmarkResult:
    """Run the held-out physics-only against physics-plus-residual comparison.

    Parameters
    ----------
    design:
        The experiment design. Training and test trajectories are generated from
        it with different input phases and different noise seeds.
    n_train, n_test:
        Number of training and held-out trajectories. Phases are spread evenly
        over ``[0, 0.5)`` for training and ``[0.5, 1)`` for test, so the two
        sets do not overlap in input phase either.
    base_seed:
        Noise seeds are ``base_seed + i`` for training and
        ``base_seed + 1000 + i`` for test.
    residual_seed:
        Seed of the Gaussian process's restart sampling.
    max_training_points:
        Cap on the residual model's training-set size.
    decision_margin:
        Relative improvement the combined model must exceed to be declared the
        winner.
    strong_cuts:
        Relative singular-value cuts for the subspace decomposition.
    max_nfev:
        Residual-evaluation budget for each physics fit.

    Returns
    -------
    BenchmarkResult

    Raises
    ------
    ValueError
        If ``n_train`` or ``n_test`` is below 1, or ``decision_margin`` is
        negative.
    """
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must both be at least 1")
    if decision_margin < 0.0:
        raise ValueError("decision_margin must not be negative")
    model = calibration_model()
    generator = truth_model()
    train_phases = np.linspace(0.0, 0.5, n_train, endpoint=False)
    test_phases = np.linspace(0.5, 1.0, n_test, endpoint=False)
    train_sets = [
        make_dataset(design, seed=base_seed + i, phase=float(p), model=generator)
        for i, p in enumerate(train_phases)
    ]
    test_sets = [
        make_dataset(design, seed=base_seed + 1000 + i, phase=float(p), model=generator)
        for i, p in enumerate(test_phases)
    ]

    calib = train_sets[0]
    problem = CalibrationProblem(
        model=model,
        inputs=calib.inputs,
        initial_state=design.initial_state,
        channels=design.channels,
        sample_stride=design.sample_stride,
        y=calib.y,
        sigma=calib.sigma,
    )
    physics_fit = fit(problem, np.asarray(design.theta_true, dtype=float), max_nfev=max_nfev)
    theta_hat = physics_fit.theta

    feature_blocks = []
    target_blocks = []
    for ds in train_sets:
        features, prediction = build_features(
            model,
            theta_hat,
            ds.inputs,
            design.initial_state,
            design.channels,
            design.sample_stride,
        )
        feature_blocks.append(features)
        target_blocks.append(ds.y - prediction)
    x_train = np.vstack(feature_blocks)
    y_train = np.concatenate(target_blocks)
    residual_model = ResidualModel(
        noise_level=float(design.sigma_k) ** 2,
        random_state=residual_seed,
        max_training_points=max_training_points,
    ).fit(x_train, y_train)

    train_scores = tuple(_score(ds, design, theta_hat, residual_model)[0] for ds in train_sets)
    test_results = [_score(ds, design, theta_hat, residual_model) for ds in test_sets]
    test_scores = tuple(r[0] for r in test_results)
    pooled_physics = np.concatenate([r[1] for r in test_results])
    pooled_combined = np.concatenate([r[2] for r in test_results])
    rmse_physics = float(np.sqrt(np.mean(pooled_physics**2)))
    rmse_combined = float(np.sqrt(np.mean(pooled_combined**2)))
    improvement = float(1.0 - rmse_combined / rmse_physics)
    if improvement > decision_margin:
        winner = "physics+residual"
    elif improvement < -decision_margin:
        winner = "physics-only"
    else:
        winner = "tie"

    pooled_std = []
    for ds in test_sets:
        features, _pred = build_features(
            model,
            theta_hat,
            ds.inputs,
            design.initial_state,
            design.channels,
            design.sample_stride,
        )
        pr = residual_model.predict(features)
        pooled_std.append(np.sqrt(pr.std**2 + design.sigma_k**2))
    std_all = np.concatenate(pooled_std)
    coverage95 = coverage_fraction(pooled_combined, std_all)
    coverage68 = coverage_fraction(pooled_combined, std_all, z=1.0)

    clean_discrepancy = []
    for ds in test_sets:
        physics_clean = model.observe(
            design.theta_true,
            ds.inputs,
            design.initial_state,
            design.channels,
            design.sample_stride,
        )
        clean_discrepancy.append(ds.y_clean - physics_clean)
    discrepancy_rms = float(np.sqrt(np.mean(np.concatenate(clean_discrepancy) ** 2)))

    first_test = test_sets[0]
    sens_test = log_sensitivity(
        model,
        theta_hat,
        first_test.inputs,
        design.initial_state,
        design.channels,
        first_test.sigma,
        design.sample_stride,
    )
    features_test, _p = build_features(
        model,
        theta_hat,
        first_test.inputs,
        design.initial_state,
        design.channels,
        design.sample_stride,
    )
    learned_field = residual_model.predict(features_test).mean
    splits = {
        float(cut): decompose_residual(sens_test, learned_field, strong_cut=cut)
        for cut in strong_cuts
    }

    train_features, _tp = build_features(
        model,
        theta_hat,
        calib.inputs,
        design.initial_state,
        design.channels,
        design.sample_stride,
    )
    drift = _parameter_drift(
        problem,
        physics_fit,
        residual_model.predict(train_features).mean,
        theta_true=design.theta_true,
        max_nfev=max_nfev,
    )
    return BenchmarkResult(
        design=design,
        physics_fit=physics_fit,
        train_scores=train_scores,
        test_scores=test_scores,
        rmse_physics_pooled=rmse_physics,
        rmse_combined_pooled=rmse_combined,
        improvement_pooled=improvement,
        winner=winner,
        decision_margin=float(decision_margin),
        coverage_95_pooled=coverage95,
        coverage_68_pooled=coverage68,
        discrepancy_rms=discrepancy_rms,
        noise_sigma=float(design.sigma_k),
        splits=splits,
        drift=drift,
        residual_kernel=residual_model.kernel_description,
        n_training_samples=residual_model.n_training_offered,
        n_training_used=residual_model.n_training_used,
    )
