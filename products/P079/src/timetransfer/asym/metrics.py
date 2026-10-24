"""Scoring for the asymmetry correctors, on one held-out split, same metrics.

Every estimator -- the symmetric assumption, the median filter, the mean
difference, ridge and the gradient-boosted model -- is scored on the **same**
held-out scenarios with the **same** metrics, and the table is printed with
the baselines in it rather than described afterwards.

Metrics
-------
* ``rmse_s``      root mean squared error on the asymmetry bias [s]
* ``mae_s``       mean absolute error [s]
* ``median_abs_s``median absolute error [s]; reported because the error
                  distribution is not symmetric and the median moves
                  independently of the mean
* ``bias_s``      mean signed error [s]; a corrector that is right on average
                  and wrong everywhere is a different failure from one that is
                  consistently high
* ``p95_abs_s``   95th percentile absolute error [s]
* ``skill``       ``1 - rmse / rmse_of_symmetric_assumption``; positive means
                  it beat doing nothing. The symmetric assumption's own skill
                  is 0 by construction.

A note on what "beating the baseline" would have to mean
--------------------------------------------------------
The decision this product supports is whether to apply a correction to a time
offset. The metric that decides it is the RMSE on the bias, because the bias
subtracts directly off the offset (equation (5) of
:mod:`timetransfer.twoway`). A corrector with a lower RMSE than the symmetric
assumption reduces the offset error; one with a higher RMSE makes it worse, and
a corrector that merely ties with a closed-form median filter is not worth the
training step, the model artefact or the version pin. That judgement is stated
here so that it is not reinvented after seeing the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ConfigurationError

__all__ = ["ScoreTable", "Scores", "score_predictions"]


@dataclass(frozen=True)
class Scores:
    """Scores of one estimator on one held-out split.

    Attributes
    ----------
    name:
        Estimator name.
    n:
        Held-out rows scored.
    rmse_s, mae_s, median_abs_s, bias_s, p95_abs_s:
        Error statistics [s]; see the module docstring.
    skill:
        ``1 - rmse / reference_rmse``.
    residual_std_s:
        Standard deviation of the held-out residuals [s]. This is the number
        :mod:`timetransfer.uncertainty` consumes as its asymmetry term, and it
        is measured on data the estimator did not see.
    """

    name: str
    n: int
    rmse_s: float
    mae_s: float
    median_abs_s: float
    bias_s: float
    p95_abs_s: float
    skill: float
    residual_std_s: float


def score_predictions(
    name: str,
    predicted_s: np.ndarray,
    target_s: np.ndarray,
    *,
    reference_rmse_s: float | None = None,
) -> Scores:
    """Score one estimator's predictions against the held-out target.

    Parameters
    ----------
    name:
        Estimator name, recorded in the result.
    predicted_s, target_s:
        Equal-length arrays [s], at least 2 entries.
    reference_rmse_s:
        RMSE of the reference estimator, used for ``skill``. ``None`` makes
        ``skill`` ``nan`` rather than silently self-referential.
    """
    pred = np.asarray(predicted_s, dtype=np.float64).reshape(-1)
    truth = np.asarray(target_s, dtype=np.float64).reshape(-1)
    if pred.size != truth.size:
        raise ConfigurationError(
            f"predicted and target must match in length: {pred.size} vs {truth.size}"
        )
    if pred.size < 2:
        raise ConfigurationError(f"need at least 2 rows to score, got {pred.size}")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ConfigurationError("predictions and targets must be finite")
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err * err)))
    skill = (
        float("nan")
        if reference_rmse_s is None or not (float(reference_rmse_s) > 0.0)
        else float(1.0 - rmse / float(reference_rmse_s))
    )
    return Scores(
        name=str(name),
        n=int(pred.size),
        rmse_s=rmse,
        mae_s=float(np.mean(np.abs(err))),
        median_abs_s=float(np.median(np.abs(err))),
        bias_s=float(np.mean(err)),
        p95_abs_s=float(np.percentile(np.abs(err), 95.0)),
        skill=skill,
        residual_std_s=float(np.std(err, ddof=1)),
    )


@dataclass(frozen=True)
class ScoreTable:
    """A set of :class:`Scores` with a renderer and a verdict.

    Attributes
    ----------
    rows:
        Scores in the order they were added.
    reference_name:
        Name of the estimator whose RMSE defines skill.
    """

    rows: tuple[Scores, ...]
    reference_name: str

    def best(self) -> Scores:
        """The row with the lowest RMSE."""
        return min(self.rows, key=lambda row: row.rmse_s)

    def by_name(self, name: str) -> Scores:
        """The row with this name."""
        for row in self.rows:
            if row.name == name:
                return row
        raise ConfigurationError(f"no scored estimator named {name!r}")

    def ratio(self, name_a: str, name_b: str) -> float:
        """``rmse(name_a) / rmse(name_b)``."""
        return self.by_name(name_a).rmse_s / self.by_name(name_b).rmse_s

    def as_text(self) -> str:
        """Fixed-width table, baselines included."""
        head = (
            f"{'estimator':>22} {'n':>5} {'rmse [s]':>12} {'mae [s]':>12} "
            f"{'med|e| [s]':>12} {'bias [s]':>12} {'p95|e| [s]':>12} {'skill':>8}"
        )
        rows = [
            f"{row.name:>22} {row.n:>5d} {row.rmse_s:>12.6e} {row.mae_s:>12.6e} "
            f"{row.median_abs_s:>12.6e} {row.bias_s:>+12.5e} {row.p95_abs_s:>12.6e} "
            f"{row.skill:>8.4f}"
            for row in self.rows
        ]
        return "\n".join(
            [head, *rows, f"skill is relative to {self.reference_name}; higher is better"]
        )
