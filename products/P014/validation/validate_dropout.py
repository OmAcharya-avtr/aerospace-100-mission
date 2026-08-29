#!/usr/bin/env python3
"""Validation 3: learned reconstructor vs regularized least-squares baseline,
across photon flux and subaperture-dropout rate.

Per the mission rule, the classical regularized least-squares baseline
(`wavelab.modal.ModalReconstructor`, Tikhonov-regularized) is implemented and
validated (see `validate_noise_free.py`, `validate_photon_noise.py`) before
this comparison. The learned reconstructor
(`wavelab.ml.ZernikeSlopeEnsemble`) is trained once on a mixed-condition
training set (moderate flux, moderate dropout, per `DATASET_CARD.md`), then
both are evaluated on identical held-out batches at every (flux, dropout)
operating point below. **Whichever model wins at a given point is reported as
measured** -- no tolerance is loosened and no point is dropped to make the
comparison look better for either side.

Run: ``python validation/validate_dropout.py`` from ``products/P014``
(~30-60 s). Output saved to ``validation/dropout_output.txt``.
"""

from __future__ import annotations

import time

import numpy as np

from wavelab.dataset import build_modal_geometry, generate_batch
from wavelab.ml import ZernikeSlopeEnsemble
from wavelab.modal import ModalReconstructor

NOLL = list(range(2, 16))
N_SIDE = 8
N_TRAIN = 1800
N_TEST = 400
TRAIN_FLUX = 800.0
TRAIN_DROPOUT = 0.25
FLUX_LEVELS = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
DROPOUT_LEVELS = (0.0, 0.15, 0.3, 0.45, 0.6)
FIXED_FLUX_FOR_DROPOUT_SWEEP = 800.0
FIXED_DROPOUT_FOR_FLUX_SWEEP = 0.0


def rms(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def evaluate(baseline: ModalReconstructor, model: ZernikeSlopeEnsemble, test) -> tuple[float, float, float]:
    base_pred = np.array(
        [baseline.reconstruct(test.slopes[i], active=test.active[i]) for i in range(len(test))]
    )
    ml_pred, ml_std = model.predict(test.slopes, test.active, return_std=True)
    base_err = base_pred - test.coeffs
    ml_err = ml_pred - test.coeffs
    base_rms = rms(base_err)
    ml_rms = rms(ml_err)
    calibration = float(np.mean(ml_std) / np.sqrt(np.mean(ml_err**2)))
    return base_rms, ml_rms, calibration


def main() -> None:
    t0 = time.time()
    geometry = build_modal_geometry(NOLL, N_SIDE)
    print(f"n_sub = {geometry.n_sub}, n_modes = {geometry.n_modes}")

    baseline = ModalReconstructor(NOLL, geometry.sub_x, geometry.sub_y, method="tikhonov", reg=3e-3)
    model = ZernikeSlopeEnsemble(
        geometry.n_sub, geometry.n_modes, n_estimators=5,
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=0,
    )
    train = generate_batch(
        geometry, N_TRAIN, photon_flux=TRAIN_FLUX, dropout_rate=TRAIN_DROPOUT, seed=100
    )
    t_train0 = time.time()
    model.fit(train.slopes, train.active, train.coeffs)
    train_time = time.time() - t_train0
    print(
        f"trained on {N_TRAIN} samples at flux={TRAIN_FLUX}, dropout={TRAIN_DROPOUT} "
        f"in {train_time:.1f} s"
    )
    print()

    print("=== Sweep A: reconstruction error vs photon flux (dropout fixed at "
          f"{FIXED_DROPOUT_FOR_FLUX_SWEEP}) ===")
    print(f"{'flux':>8}  {'baseline RMS':>13}  {'ML RMS':>10}  {'ML/base':>8}  {'winner':>9}  {'calib.':>7}")
    flux_rows = []
    for flux in FLUX_LEVELS:
        test = generate_batch(
            geometry, N_TEST, photon_flux=flux, dropout_rate=FIXED_DROPOUT_FOR_FLUX_SWEEP, seed=9000
        )
        base_rms, ml_rms, calib = evaluate(baseline, model, test)
        winner = "baseline" if base_rms < ml_rms else "ML"
        flux_rows.append((flux, base_rms, ml_rms, winner))
        print(f"{flux:8.0f}  {base_rms:13.6f}  {ml_rms:10.6f}  {ml_rms / base_rms:8.3f}  {winner:>9}  {calib:7.3f}")
    print()

    print("=== Sweep B: reconstruction error vs subaperture-dropout rate "
          f"(flux fixed at {FIXED_FLUX_FOR_DROPOUT_SWEEP}) ===")
    print(f"{'dropout':>8}  {'baseline RMS':>13}  {'ML RMS':>10}  {'ML/base':>8}  {'winner':>9}  {'calib.':>7}")
    dropout_rows = []
    for dropout in DROPOUT_LEVELS:
        test = generate_batch(
            geometry, N_TEST, photon_flux=FIXED_FLUX_FOR_DROPOUT_SWEEP, dropout_rate=dropout, seed=9500
        )
        base_rms, ml_rms, calib = evaluate(baseline, model, test)
        winner = "baseline" if base_rms < ml_rms else "ML"
        dropout_rows.append((dropout, base_rms, ml_rms, winner))
        print(f"{dropout:8.2f}  {base_rms:13.6f}  {ml_rms:10.6f}  {ml_rms / base_rms:8.3f}  {winner:>9}  {calib:7.3f}")
    print()

    base_wins = sum(1 for *_r, w in flux_rows + dropout_rows if w == "baseline")
    ml_wins = sum(1 for *_r, w in flux_rows + dropout_rows if w == "ML")
    print(f"Summary: baseline wins {base_wins}/{len(flux_rows) + len(dropout_rows)} operating points, "
          f"ML wins {ml_wins}/{len(flux_rows) + len(dropout_rows)}.")
    print("This is the measured result and is reported as-is (README, MODEL_CARD.md) "
          "regardless of which model wins where.")
    print(f"\ntotal wall time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
