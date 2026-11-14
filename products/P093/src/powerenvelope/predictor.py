"""The learned active-power predictor, what it may see, and what it is scored against.

The learned component predicts a window's **mean active rail power** in mW from
the licensed descriptors of :mod:`powerenvelope.windows` -- the configuration
and the platform's own environment at the window start -- and nothing else.

Why a random forest. Three measured constraints, not preferences: the dataset
is a few hundred windows from one campaign; the build environment has two cores
and a three-minute budget for any fit; and a forest's spread across trees gives
an uncertainty output without a second model. Nothing here trains for longer
than a second on two cores, and the fit time is reported by
``validation/validate_measured_learned.py``.

**Uncertainty output.** :meth:`ActivePowerPredictor.predict` returns a point
estimate and a per-sample standard deviation across the ensemble's trees. That
is the ensemble's internal disagreement, which is a LOWER BOUND on the
predictive uncertainty: it excludes the noise in the target. The target's own
noise floor is the instrument's, and :func:`target_quantisation_sd` computes it
from the 8 mA current quantum -- a uniform quantiser of width ``q`` has
standard deviation ``q / sqrt(12)`` -- which is combined in quadrature when the
caller supplies the per-window quantisation step. That total is still a lower
bound, because it does not include run-to-run variation; the between-run
interval of :func:`powerenvelope.baselines.between_run_spread` is the measure
of that.

**What it is scored against, on the same splits.**

- :class:`AnalyticActivePowerModel` -- the duty-cycle model of Baseline 2 used
  in reverse. It takes the measured background floor, the training cells'
  whole-run mean power and their DECLARED duty, estimates the active excess as
  ``mean((P_run - P_floor) / duty)`` over the training cells only, and predicts
  ``P_floor + excess`` for any cell. It is fitted to nothing and sees no test
  run. This is the model the specification's pre-registered expectation names.
- :class:`ConstantPowerPredictor` -- the training mean. A learned model that
  cannot beat this has learned nothing from the descriptors.

**The exclusion is enforced in code, by a named check.**
:func:`assert_descriptors_licensed` raises :class:`DescriptorLeakError` for any
descriptor outside :data:`LICENSED_DESCRIPTORS`, and names the excluded class a
rejected descriptor belongs to. It is called by
:meth:`ActivePowerPredictor.fit`, so a caller cannot reach the estimator with an
unlicensed column. ``tests/test_predictor.py`` asserts that it fires, and
``validation/validate_descriptor_leak.py`` refits on narrower sets and reports
what the restriction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .energy import whole_run_energy
from .trace import Campaign
from .windows import FEATURE_NAMES

#: Forest size. Large enough for a stable across-tree spread, small enough that
#: a fit on a few hundred rows is a fraction of a second on two cores.
N_ESTIMATORS = 300

#: Depth cap. A few hundred rows and ten descriptors; an unbounded tree
#: memorises single windows.
MAX_DEPTH = 6

#: Declared seed for every fit in this product.
RANDOM_STATE = 930093

#: The descriptors the specification licenses. Exactly the columns of
#: :data:`powerenvelope.windows.FEATURE_NAMES`.
LICENSED_DESCRIPTORS: tuple[str, ...] = FEATURE_NAMES

#: The classes of quantity the specification excludes, with the name fragments
#: that identify them. Used only to word a refusal: the membership test is
#: exact against :data:`LICENSED_DESCRIPTORS`, so a quantity does not need to be
#: anticipated here to be rejected.
EXCLUDED_DESCRIPTOR_CLASSES: dict[str, tuple[str, ...]] = {
    "the pipeline's own measured power": (
        "vdd_in",
        "power_mw",
        "p_active",
        "mean_active_power",
        "active_mw",
        "soc_mw",
        "cpu_gpu_cv",
    ),
    "the pipeline's own measured latency": ("latency", "t_end", "t_start", "wall_s"),
    "the pipeline's own CPU occupancy": ("cpu_busy_fraction", "active_cpu", "occupancy"),
    "the pipeline's realised inference count": ("inference", "n_inf", "frame_idx"),
    "the pipeline's realised acceptance rate": ("accept", "reject", "n_candidates"),
}


class DescriptorLeakError(ValueError):
    """A descriptor outside the licensed set reached the learned predictor.

    Raised by :func:`assert_descriptors_licensed`. The message names every
    offending descriptor and, where it can, the excluded class it belongs to.
    """


def assert_descriptors_licensed(names: tuple[str, ...]) -> None:
    """Raise unless every name is in :data:`LICENSED_DESCRIPTORS`.

    This is the named check the specification requires. It is exact-membership
    and not pattern-based, so a quantity nobody anticipated is still rejected;
    the patterns in :data:`EXCLUDED_DESCRIPTOR_CLASSES` only make the message
    say which excluded class the caller tripped over.

    Raises
    ------
    DescriptorLeakError
        Naming every unlicensed descriptor.
    TypeError
        If ``names`` is not a sequence of strings.
    """
    if isinstance(names, str) or not all(isinstance(name, str) for name in names):
        raise TypeError("names must be a sequence of descriptor name strings")
    offending: list[str] = []
    for name in names:
        if name in LICENSED_DESCRIPTORS:
            continue
        lowered = name.lower()
        matched = [
            label
            for label, fragments in EXCLUDED_DESCRIPTOR_CLASSES.items()
            if any(fragment in lowered for fragment in fragments)
        ]
        reason = matched[0] if matched else "not in the licensed set"
        offending.append(f"{name!r} ({reason})")
    if offending:
        raise DescriptorLeakError(
            "unlicensed descriptor(s) reached the learned predictor: "
            + "; ".join(offending)
            + f". The specification licenses exactly {list(LICENSED_DESCRIPTORS)} and "
            "excludes the pipeline's own measured power, its measured latency, its CPU "
            "occupancy, its realised inference count and its realised acceptance rate."
        )


def target_quantisation_sd(
    step_mw: np.ndarray | float, n_samples: np.ndarray | float
) -> np.ndarray:
    """Standard deviation a uniform quantiser of width ``step_mw`` contributes, mW.

    A uniform quantiser of width ``q`` has variance ``q**2 / 12``. Averaging
    ``n`` readings reduces that by ``n`` only if the quantisation errors are
    independent, which they are not here: the INA3221 averages 512 conversions
    in hardware, so neighbouring 20 Hz samples overlap in the instrument. The
    ``sqrt(n)`` reduction is therefore applied as a LOWER BOUND on the noise and
    named as one, rather than as a correct error propagation.

    Raises
    ------
    ValueError
        On a non-positive step or sample count.
    """
    q = np.asarray(step_mw, dtype=np.float64)
    n = np.asarray(n_samples, dtype=np.float64)
    if np.any(q <= 0.0):
        raise ValueError("step_mw must be positive")
    if np.any(n <= 0.0):
        raise ValueError("n_samples must be positive")
    return q / np.sqrt(12.0) / np.sqrt(n)


@dataclass(frozen=True)
class Prediction:
    """A predicted mean active power with its uncertainty, all in mW."""

    power_mw: np.ndarray
    sigma_mw: np.ndarray
    sigma_model_mw: np.ndarray
    sigma_target_mw: np.ndarray


class ConstantPowerPredictor:
    """Predict the training mean active power for every window. The null model."""

    def __init__(self) -> None:
        self.power_mw_: float | None = None
        self.sd_mw_: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> ConstantPowerPredictor:
        """Record the training mean and standard deviation of ``targets``."""
        values = np.asarray(targets, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot fit a constant power to an empty target vector")
        self.power_mw_ = float(values.mean())
        self.sd_mw_ = float(values.std(ddof=1)) if values.size > 1 else 0.0
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        """Return the training mean, with the training spread as uncertainty."""
        if self.power_mw_ is None:
            raise RuntimeError("ConstantPowerPredictor.predict called before fit")
        n = int(np.asarray(features).shape[0])
        sigma = np.full(n, float(self.sd_mw_ or 0.0), dtype=np.float64)
        return Prediction(
            power_mw=np.full(n, self.power_mw_, dtype=np.float64),
            sigma_mw=sigma,
            sigma_model_mw=sigma,
            sigma_target_mw=np.zeros(n, dtype=np.float64),
        )


class AnalyticActivePowerModel:
    """Baseline 2's duty-cycle model, inverted to predict mean active power.

    The duty-cycle model says a run's mean total power is
    ``P_floor + duty * (P_active - P_floor)`` plus the wake transient. Dropping
    the transient -- which is a small part of a cycle's energy and is reported
    separately by :func:`powerenvelope.baselines.ramp_energy` -- and solving for
    the active excess gives, per training cell,
    ``excess = (P_run_mean - P_floor) / duty``. The model takes the mean of
    those over the TRAINING cells only and predicts ``P_floor + excess`` for
    any cell, including one whose duty it has never seen.

    It is fitted to nothing in the learned sense: it has one parameter, and that
    parameter never sees a test run.
    """

    def __init__(self) -> None:
        self.floor_mw_: float | None = None
        self.excess_mw_: float | None = None
        self.excess_sd_mw_: float | None = None
        self.cells_: tuple[str, ...] = ()
        self.per_cell_excess_mw_: tuple[float, ...] = ()

    def fit(
        self,
        campaign: Campaign,
        cell_labels: tuple[str, ...] | np.ndarray,
        floor_mw: float | None = None,
    ) -> AnalyticActivePowerModel:
        """Estimate the active excess from the named training cells.

        Parameters
        ----------
        campaign
            The loaded campaign.
        cell_labels
            Training cell labels. Duplicates are collapsed.
        floor_mw
            Measured background floor power, mW. Taken from the campaign's
            floor cell when omitted.

        Raises
        ------
        ValueError
            If no training cell has a positive duty, so the inversion has no
            data.
        """
        from .baselines import floor_power

        floor = float(floor_mw) if floor_mw is not None else floor_power(campaign).mean_mw
        labels = tuple(dict.fromkeys(str(label) for label in np.atleast_1d(cell_labels)))
        estimates: list[float] = []
        used: list[str] = []
        for label in labels:
            cell = campaign.require(label)
            if cell.duty <= 0.0 or not cell.has_pipeline:
                continue
            mean_power = whole_run_energy(cell.power).mean_power_mw
            estimates.append((mean_power - floor) / cell.duty)
            used.append(label)
        if not estimates:
            raise ValueError(
                "the analytic model has no training cell with a positive duty, so the "
                "active excess cannot be estimated. Nothing is predicted."
            )
        values = np.asarray(estimates, dtype=np.float64)
        self.floor_mw_ = floor
        self.excess_mw_ = float(values.mean())
        self.excess_sd_mw_ = float(values.std(ddof=1)) if values.size > 1 else 0.0
        self.cells_ = tuple(used)
        self.per_cell_excess_mw_ = tuple(float(value) for value in values)
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict ``P_floor + excess`` for every row of ``features``."""
        if self.excess_mw_ is None or self.floor_mw_ is None:
            raise RuntimeError("AnalyticActivePowerModel.predict called before fit")
        n = int(np.asarray(features).shape[0])
        sigma = np.full(n, float(self.excess_sd_mw_ or 0.0), dtype=np.float64)
        return Prediction(
            power_mw=np.full(n, self.floor_mw_ + self.excess_mw_, dtype=np.float64),
            sigma_mw=sigma,
            sigma_model_mw=sigma,
            sigma_target_mw=np.zeros(n, dtype=np.float64),
        )


class LinearActivePowerModel:
    """Least squares in the two configuration descriptors. The parametric classical.

    ``P_active = a + b * duty + c * frame_dim``, fitted by ordinary least
    squares over the TRAINING cells' measured mean active power. Three
    parameters, no tuning, no hyperparameter, and it sees no test cell.

    It is implemented and reported BEFORE the learned model because the build
    guide requires the classical alternative to be validated first: an
    under-strength classical baseline inflates a learned win, and an earlier
    product of this mission inflated one by about 50 per cent that way. On this
    campaign it is the strongest of the three classical predictors on held-out
    duty cycles, and the specification's own named analytic duty-inversion
    predictor is the weakest of the three; both facts are reported in
    ``validation/validate_measured_learned.py``.

    The uncertainty output is the residual standard error of the fit, which is
    a single number for every prediction: a linear model with three parameters
    has nothing per-sample to say.
    """

    #: The descriptors this model reads, by name. Both are licensed.
    DESCRIPTORS: tuple[str, ...] = ("duty", "frame_dim")

    def __init__(self) -> None:
        self.coefficients_: np.ndarray | None = None
        self.residual_sd_mw_: float | None = None
        self.cells_: tuple[str, ...] = ()

    def fit(
        self,
        campaign: Campaign,
        cell_labels: tuple[str, ...] | np.ndarray,
        targets: dict[str, float] | None = None,
    ) -> LinearActivePowerModel:
        """Fit on the named training cells' measured mean active power.

        Parameters
        ----------
        campaign
            The loaded campaign.
        cell_labels
            Training cell labels. Duplicates are collapsed.
        targets
            Optional mapping from cell label to the active power to fit. When
            omitted, each cell's measured mean active power is used.

        Raises
        ------
        ValueError
            With fewer than four training cells, where a three-parameter fit
            has no residual to report.
        """
        from .baselines import mean_active_power_mw

        labels = tuple(dict.fromkeys(str(label) for label in np.atleast_1d(cell_labels)))
        usable = [label for label in labels if campaign.require(label).has_pipeline]
        if len(usable) < 4:
            raise ValueError(
                f"a three-parameter fit over {len(usable)} training cell(s) has no "
                "residual; at least four are needed."
            )
        design = np.asarray(
            [
                [1.0, campaign.require(label).duty, float(campaign.require(label).frame_dim)]
                for label in usable
            ],
            dtype=np.float64,
        )
        observed = np.asarray(
            [
                (
                    targets[label]
                    if targets is not None
                    else mean_active_power_mw(campaign.require(label))
                )
                for label in usable
            ],
            dtype=np.float64,
        )
        coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
        residual = observed - design @ coefficients
        degrees = max(1, observed.size - design.shape[1])
        self.coefficients_ = coefficients
        self.residual_sd_mw_ = float(np.sqrt(float(residual @ residual) / degrees))
        self.cells_ = tuple(usable)
        return self

    def predict(
        self, features: np.ndarray, feature_names: tuple[str, ...] = LICENSED_DESCRIPTORS
    ) -> Prediction:
        """Predict from the ``duty`` and ``frame_dim`` columns of ``features``."""
        if self.coefficients_ is None:
            raise RuntimeError("LinearActivePowerModel.predict called before fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got shape {x.shape}")
        try:
            duty = x[:, feature_names.index("duty")]
            dimension = x[:, feature_names.index("frame_dim")]
        except ValueError:
            raise ValueError(
                "features must carry 'duty' and 'frame_dim' columns; got "
                f"{list(feature_names)}"
            ) from None
        power = (
            self.coefficients_[0]
            + self.coefficients_[1] * duty
            + self.coefficients_[2] * dimension
        )
        sigma = np.full(power.shape, float(self.residual_sd_mw_ or 0.0), dtype=np.float64)
        return Prediction(
            power_mw=power,
            sigma_mw=sigma,
            sigma_model_mw=sigma,
            sigma_target_mw=np.zeros_like(power),
        )

    def predict_cell(self, campaign: Campaign, cell_label: str) -> float:
        """Predicted mean active power for one cell, mW."""
        cell = campaign.require(cell_label)
        if self.coefficients_ is None:
            raise RuntimeError("LinearActivePowerModel.predict_cell called before fit")
        return float(
            self.coefficients_[0]
            + self.coefficients_[1] * cell.duty
            + self.coefficients_[2] * float(cell.frame_dim)
        )


class ActivePowerPredictor:
    """Random-forest mean-active-power predictor with an across-tree uncertainty."""

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int = MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(
                f"n_estimators must be >= 2 for an across-tree spread, got {n_estimators}"
            )
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.forest_: RandomForestRegressor | None = None
        self.feature_names_: tuple[str, ...] = ()

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> ActivePowerPredictor:
        """Fit the forest, after checking every descriptor is licensed.

        ``feature_names`` is required, not optional: an unnamed design matrix
        cannot be checked against the licensed set, and this product does not
        fit a model it cannot audit.

        Raises
        ------
        DescriptorLeakError
            If any descriptor is outside :data:`LICENSED_DESCRIPTORS`.
        ValueError
            On mismatched shapes, fewer than two training rows, a name count
            that disagrees with the column count, or a non-positive target --
            the target is a rail power in mW and a non-positive value means the
            caller built the dataset wrongly.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got shape {x.shape}")
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError(f"targets shape {y.shape} does not match features {x.shape}")
        if len(feature_names) != x.shape[1]:
            raise ValueError(
                f"{len(feature_names)} descriptor name(s) for {x.shape[1]} column(s); the "
                "names are what the licence check reads, so they must correspond"
            )
        assert_descriptors_licensed(tuple(feature_names))
        if x.shape[0] < 2:
            raise ValueError(f"need at least 2 training windows, got {x.shape[0]}")
        if np.any(y <= 0.0):
            raise ValueError("targets must be positive rail powers in mW")
        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(x, y)
        self.feature_names_ = tuple(feature_names)
        return self

    def predict(
        self,
        features: np.ndarray,
        quantisation_step_mw: np.ndarray | float | None = None,
        n_samples: np.ndarray | float | None = None,
    ) -> Prediction:
        """Predict mean active power with uncertainty.

        When both ``quantisation_step_mw`` and ``n_samples`` are supplied, the
        instrument's quantisation noise from :func:`target_quantisation_sd` is
        combined in quadrature with the across-tree spread.
        """
        if self.forest_ is None:
            raise RuntimeError("ActivePowerPredictor.predict called before fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got shape {x.shape}")
        per_tree = np.stack([tree.predict(x) for tree in self.forest_.estimators_], axis=0)
        power = per_tree.mean(axis=0)
        sigma_model = per_tree.std(axis=0, ddof=0)
        if quantisation_step_mw is None or n_samples is None:
            sigma_target = np.zeros_like(power)
        else:
            sigma_target = np.broadcast_to(
                target_quantisation_sd(quantisation_step_mw, n_samples), power.shape
            ).astype(np.float64)
        return Prediction(
            power_mw=power,
            sigma_mw=np.sqrt(sigma_model**2 + sigma_target**2),
            sigma_model_mw=sigma_model,
            sigma_target_mw=sigma_target,
        )

    def importances(self) -> dict[str, float]:
        """Impurity-based feature importances, by name."""
        if self.forest_ is None:
            raise RuntimeError("ActivePowerPredictor.importances called before fit")
        values = self.forest_.feature_importances_
        return {
            name: float(value)
            for name, value in zip(self.feature_names_, values, strict=True)
        }


@dataclass(frozen=True)
class Score:
    """Scores of one predictor on one split.

    ``mae`` and ``rmse`` are in mW over the window mean active powers.
    ``coverage`` is the fraction of test windows whose true power lies within
    one reported sigma of the prediction, which is the only check that the
    uncertainty output means anything; it is ``nan`` when no sigma was reported.
    """

    label: str
    n_test: int
    mae_mw: float
    rmse_mw: float
    bias_mw: float
    coverage: float


def score(
    label: str,
    truth: np.ndarray,
    prediction: Prediction | np.ndarray,
    sigma: np.ndarray | None = None,
) -> Score:
    """Score a prediction against measured window mean active powers."""
    y = np.asarray(truth, dtype=np.float64)
    if isinstance(prediction, Prediction):
        yhat = np.asarray(prediction.power_mw, dtype=np.float64)
        s: np.ndarray | None = np.asarray(prediction.sigma_mw, dtype=np.float64)
    else:
        yhat = np.asarray(prediction, dtype=np.float64)
        s = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if y.shape != yhat.shape:
        raise ValueError(f"truth shape {y.shape} does not match prediction {yhat.shape}")
    if y.size == 0:
        raise ValueError("cannot score an empty split")
    residual = yhat - y
    coverage = (
        float("nan")
        if s is None or not np.any(s > 0.0)
        else float(np.mean(np.abs(residual) <= s))
    )
    return Score(
        label=label,
        n_test=int(y.size),
        mae_mw=float(np.mean(np.abs(residual))),
        rmse_mw=float(np.sqrt(np.mean(residual**2))),
        bias_mw=float(np.mean(residual)),
        coverage=coverage,
    )
