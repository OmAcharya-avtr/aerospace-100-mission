"""Declared scenarios and the seeded dataset builders for the learned model.

Every array used anywhere in this repository comes from one of the functions
below, from an explicit seed. No data file is committed; regeneration is
deterministic.

Declared scenarios
------------------
The three change magnitudes were chosen **before any detector was run**, to put
the parameter step in the moderate-shift regime where a sequential test has a
delay of tens of samples rather than one or thousands. They are stated here,
once, and are not revisited after a delay is measured:

=================  =========================  ==============================
scenario           magnitude                  effect on the residual
=================  =========================  ==============================
parameter_step     actuator gain -1 %         mean shift of -0.404 sigma,
                                              variance unchanged
slow_ramp          actuator gain -1 % over    the same mean shift reached
                   400 samples (20 s)         linearly over 20 s
noise_variance     process noise x 2          mean unchanged at 0, residual
                                              variance x 1.06
=================  =========================  ==============================

The mean shift of the parameter step has a closed form, checked in
``validation/validate_twin.py``: with a relative gain error ``delta`` and the
constant part ``u0`` of the excitation,

    E[z] = C (I - (A - K C))^{-1} B delta u0 / sqrt(S).                  (10)

Seed allocation
---------------
Disjoint seeds, so that no two roles ever see the same noise realisation:

==============  ==========================================================
seed            role
==============  ==========================================================
53001           in-control streams for threshold calibration
53002           in-control streams for classifier training
53010-53012     changed streams for classifier training, one per scenario
53020           in-control streams for probability calibration
53030-53032     changed streams for probability calibration
53100-53102     changed streams for evaluation, one per scenario
53110-53123     in-control and changed streams for evaluation
53200-53202     changed streams for the matched-ARL0 benchmark
53300-53303     changed streams outside the training distribution
53400           paired streams for the attribution demonstration
53500-53501     streams for the figures
==============  ==========================================================

The families are spaced rather than consecutive because
:func:`build_labelled_set` consumes ``seed_changed + i`` for each of the three
scenarios. An earlier version of this module used 53003 for the training
changed streams and 53004/53005 for probability calibration, so the training
family's second and third scenarios reused the calibration family's seed
numbers. The arrays did not in fact collide, because the two families are
generated at different ``n_runs`` and ``numpy.random.default_rng`` therefore
draws different values; but the allocation did not support the claim of
disjoint families that the test-split strategy rests on, so it was widened.
The episode is recorded in ``validation/VALIDATION.md``, error 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asset import AssetChange, StreamSpec, simulate_residuals
from .features import DEFAULT_WINDOW, N_FEATURES, window_features

SCENARIOS: dict[str, AssetChange] = {
    "parameter_step": AssetChange(kind="parameter_step", onset=0, magnitude=-0.01),
    "slow_ramp": AssetChange(kind="slow_ramp", onset=0, magnitude=-0.01, ramp_samples=400),
    "noise_variance": AssetChange(kind="noise_variance", onset=0, magnitude=2.0),
}
"""The three declared change scenarios, with the change present from sample 0."""

SCENARIO_LABELS: dict[str, str] = {
    "parameter_step": "parameter step (actuator gain -1 %)",
    "slow_ramp": "slow ramp (gain -1 % over 400 samples)",
    "noise_variance": "noise variance (process noise x 2)",
}
"""Human-readable scenario labels for figures and tables."""

SEED_THRESHOLD_CALIBRATION = 53001
SEED_TRAIN_IN_CONTROL = 53002
SEED_TRAIN_CHANGED = 53010
SEED_PROBCAL_IN_CONTROL = 53020
SEED_PROBCAL_CHANGED = 53030
SEED_EVAL_CHANGED = 53100
SEED_EVAL_IN_CONTROL = 53110


def in_control_streams(
    n_runs: int = 400, n_samples: int = 4000, seed: int = SEED_THRESHOLD_CALIBRATION
) -> np.ndarray:
    """In-control normalised residual streams, shape ``(n_runs, n_samples)``.

    The default size, 400 x 4000 = 1.6e6 samples, resolves an ARL0 of 1000
    samples with about 400 alarms, i.e. a 5 % relative standard error.
    """
    return simulate_residuals(
        StreamSpec(change=AssetChange(), n_runs=n_runs, n_samples=n_samples, seed=seed)
    )


def changed_streams(
    scenario: str,
    n_runs: int = 400,
    n_samples: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Residual streams with the declared change present from sample 0.

    Parameters
    ----------
    scenario:
        One of the keys of :data:`SCENARIOS`.
    n_runs, n_samples:
        Batch size.
    seed:
        Defaults to ``SEED_EVAL_CHANGED + index of the scenario``, so the three
        scenarios never share a noise realisation.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    if seed is None:
        seed = SEED_EVAL_CHANGED + list(SCENARIOS).index(scenario)
    return simulate_residuals(
        StreamSpec(change=SCENARIOS[scenario], n_runs=n_runs, n_samples=n_samples, seed=seed)
    )


@dataclass(frozen=True)
class LabelledWindows:
    """A feature matrix with labels and the run index each row came from.

    ``run`` is kept so that any further split can be made by run rather than
    by window; see the test-split discussion in
    :mod:`twininvalidate.classifier`.
    """

    x: np.ndarray
    y: np.ndarray
    run: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[1] != N_FEATURES:
            raise ValueError(f"x must have shape (n, {N_FEATURES}), got {self.x.shape}")
        if self.y.shape != (self.x.shape[0],) or self.run.shape != (self.x.shape[0],):
            raise ValueError("x, y and run must agree on the number of rows")

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _labelled(
    z: np.ndarray, label: int, window: int, stride: int, run_offset: int
) -> LabelledWindows:
    feats = window_features(z, window)
    n_runs, n_windows, _ = feats.shape
    keep = np.arange(0, n_windows, stride)
    sub = feats[:, keep, :]
    x = sub.reshape(-1, N_FEATURES)
    y = np.full(x.shape[0], label, dtype=int)
    run = np.repeat(np.arange(n_runs) + run_offset, keep.size)
    return LabelledWindows(x=x, y=y, run=run)


def _concat(parts: list[LabelledWindows]) -> LabelledWindows:
    return LabelledWindows(
        x=np.concatenate([p.x for p in parts], axis=0),
        y=np.concatenate([p.y for p in parts], axis=0),
        run=np.concatenate([p.run for p in parts], axis=0),
    )


def build_labelled_set(
    seed_in_control: int,
    seed_changed: int,
    *,
    n_runs_in_control: int = 120,
    n_runs_per_scenario: int = 40,
    n_samples: int = 600,
    window: int = DEFAULT_WINDOW,
    stride: int = 10,
) -> LabelledWindows:
    """Build a balanced labelled window set from seeded streams.

    Label 0 comes from in-control streams; label 1 from streams in which one of
    the three declared scenarios is present from sample 0, so every window is
    post-onset. For ``slow_ramp`` that includes early windows in which the
    change has barely begun and which are close to indistinguishable from
    in-control: those are labelled 1 anyway. The alternative -- dropping them,
    or labelling by an effect-size threshold -- would make the training
    problem easier than the monitoring problem actually is, so it is not done,
    and the resulting irreducible label noise is why the Brier score reported
    in ``validation/validate_classifier.py`` cannot approach zero.

    Windows are subsampled with ``stride`` to keep the training set small
    enough for the compute budget; consecutive windows overlap in
    ``window - 1`` samples, so a stride of 10 still leaves substantial
    correlation within a run, which is why every split is by run.

    Parameters
    ----------
    seed_in_control, seed_changed:
        Seeds for the two stream families. Must differ.
    n_runs_in_control:
        Runs of in-control data.
    n_runs_per_scenario:
        Runs per changed scenario; three scenarios, so the positive class has
        ``3 * n_runs_per_scenario`` runs.
    n_samples:
        Stream length in samples.
    window:
        Feature window length.
    stride:
        Window subsampling stride.

    Returns
    -------
    A :class:`LabelledWindows`.
    """
    if seed_in_control == seed_changed:
        raise ValueError("in-control and changed streams must use different seeds")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    negatives = _labelled(
        simulate_residuals(
            StreamSpec(
                change=AssetChange(),
                n_runs=n_runs_in_control,
                n_samples=n_samples,
                seed=seed_in_control,
            )
        ),
        0,
        window,
        stride,
        run_offset=0,
    )
    parts = [negatives]
    offset = n_runs_in_control
    for i, name in enumerate(SCENARIOS):
        z = simulate_residuals(
            StreamSpec(
                change=SCENARIOS[name],
                n_runs=n_runs_per_scenario,
                n_samples=n_samples,
                seed=seed_changed + i,
            )
        )
        parts.append(_labelled(z, 1, window, stride, run_offset=offset))
        offset += n_runs_per_scenario
    return _concat(parts)


def training_set(**kwargs: int) -> LabelledWindows:
    """The declared training set (seeds 53002 / 53003)."""
    return build_labelled_set(SEED_TRAIN_IN_CONTROL, SEED_TRAIN_CHANGED, **kwargs)


def calibration_set(**kwargs: int) -> LabelledWindows:
    """The declared probability-calibration set (seeds 53004 / 53005)."""
    return build_labelled_set(
        SEED_PROBCAL_IN_CONTROL,
        SEED_PROBCAL_CHANGED,
        n_runs_in_control=kwargs.pop("n_runs_in_control", 60),
        n_runs_per_scenario=kwargs.pop("n_runs_per_scenario", 20),
        **kwargs,
    )
