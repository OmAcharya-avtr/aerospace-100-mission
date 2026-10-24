"""Check 8 -- seeded reproducibility at every level that has a seed.

Requirement (BATCH_08_SPEC, P079, L4 scaffolding item 2): "Simulation mode with
a seeded clock model (declared drift, jitter, asymmetry, outages) -- fully
reproducible, no hardware", and in the validation list, "seeded
reproducibility".

Every level is checked separately, because a reproducibility failure at one
level is invisible at another: the clock model, the link model, the synthetic
noise processes, the acquisition, the estimators, the dataset generator and the
learned models all carry a seed, and each is checked by digest or by byte
comparison rather than by eye.

Run: ``python validation/validate_reproducibility.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.asym import (  # noqa: E402
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    RidgeAsymmetryCorrector,
    TrainingConfig,
    build_dataset,
)
from timetransfer.backends import (  # noqa: E402
    SimulatedBackend,
    SimulationSpec,
    make_backend_pair,
)
from timetransfer.clockmodel import ClockModel, ClockSpec  # noqa: E402
from timetransfer.linkmodel import LinkModel, LinkSpec  # noqa: E402
from timetransfer.noise import NOISE_KINDS, synthetic_phase  # noqa: E402
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402
from timetransfer.uncertainty import AsymmetrySource, offset_uncertainty  # noqa: E402

SEED = 20261024
SPEC = SimulationSpec(
    clock=ClockSpec(
        offset0_s=1.250e-3,
        skew_s_per_s=1.800e-6,
        drift_per_s=2.000e-12,
        q1_s2_per_s=1.0e-20,
        q2_per_s=1.0e-26,
    ),
    link=LinkSpec(
        floor_ab_s=5.4e-3,
        floor_ba_s=4.6e-3,
        jitter_shape_ab=2.0,
        jitter_shape_ba=1.6,
        jitter_scale_ab_s=2.4e-4,
        jitter_scale_ba_s=1.8e-4,
        loss_probability=0.03,
        outage_windows=((118.0, 126.0),),
    ),
    period_s=1.0,
    timestamp_noise_s=2.0e-8,
    resolution_ns=1,
    seed=SEED,
)
N = scaled(300, minimum=30)


def _pipeline(seed: int) -> dict[str, object]:
    spec = SimulationSpec(
        clock=SPEC.clock,
        link=SPEC.link,
        period_s=SPEC.period_s,
        timestamp_noise_s=SPEC.timestamp_noise_s,
        resolution_ns=SPEC.resolution_ns,
        seed=seed,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, N, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    kal = kalman_skew(
        estimate.midpoint_s,
        estimate.offset_s,
        config=KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26),
    )
    unc = offset_uncertainty(
        fit,
        resolution_ns=spec.resolution_ns,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.6e-3,
    )
    return {
        "digest": exchanges.digest(),
        "n": exchanges.n,
        "n_lost": exchanges.n_lost,
        "offset_bytes": estimate.offset_s.tobytes(),
        "ls_params": fit.params.tobytes(),
        "kf_offset": kal.final_offset_s,
        "kf_skew": kal.final_skew_s_per_s,
        "u_combined": unc.u_combined_s,
        "n_effective": unc.n_effective,
    }


def main() -> int:
    out = Tee("validate_reproducibility_output.txt")
    out("timetransfer validation 8 -- seeded reproducibility at every level")

    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0
    out("")
    out(f"base seed : {SEED}")
    out(f"exchanges : {N}")

    out.rule("8a. the whole pipeline, twice, same seed")
    first = _pipeline(SEED)
    second = _pipeline(SEED)
    out(f"{'quantity':>20} {'first':>34} {'identical':>10}")
    for key in first:
        value = first[key]
        shown = (
            value[:16].hex() + "..."
            if isinstance(value, bytes)
            else (value if isinstance(value, str) else f"{value}")
        )
        same = first[key] == second[key]
        out(f"{key:>20} {str(shown)[:34]:>34} {str(same):>10}")
        if not same:
            failures += 1
    out("")
    out(f"verdict : {'PASS' if first == second else 'FAIL'}")
    if first != second:
        failures += 1

    out.rule("8b. a different seed gives a different realisation")
    other = _pipeline(SEED + 1)
    differ = other["digest"] != first["digest"]
    out(f"digest at seed {SEED}     : {first['digest']}")
    out(f"digest at seed {SEED + 1} : {other['digest']}")
    out(f"differ : {differ}")
    out(f"verdict : {'PASS' if differ else 'FAIL'}")
    if not differ:
        failures += 1

    out.rule("8c. the clock and link models separately")
    def clock_trace(seed: int) -> bytes:
        model = ClockModel(SPEC.clock, seed=seed)
        return np.asarray([model.offset_at(float(t)) for t in range(60)]).tobytes()

    def link_trace(seed: int) -> bytes:
        model = LinkModel(SPEC.link, seed=seed)
        rows = [model.draw(float(t)) for t in range(60)]
        return np.asarray([[a, b, float(c)] for a, b, c in rows]).tobytes()

    clock_same = clock_trace(7) == clock_trace(7)
    clock_diff = clock_trace(7) != clock_trace(8)
    link_same = link_trace(7) == link_trace(7)
    link_diff = link_trace(7) != link_trace(8)
    out(f"clock model identical on the same seed : {clock_same}")
    out(f"clock model differs on another seed    : {clock_diff}")
    out(f"link model identical on the same seed  : {link_same}")
    out(f"link model differs on another seed     : {link_diff}")
    if not (clock_same and clock_diff and link_same and link_diff):
        failures += 1
        out("FAIL")
    else:
        out("PASS")

    out.rule("8d. snapshot and restore reproduce an acquisition exactly")
    backend = SimulatedBackend(SPEC)
    backend.open()
    snapshot = backend.snapshot()
    run_a = acquire(backend, N, min_exchanges=8).exchanges
    backend.restore(snapshot)
    run_b = acquire(backend, N, min_exchanges=8).exchanges
    backend.close()
    restore_ok = run_a.timestamp_bytes() == run_b.timestamp_bytes()
    out(f"first digest  : {run_a.digest()}")
    out(f"second digest : {run_b.digest()}")
    out(f"identical     : {restore_ok}")
    out(f"verdict       : {'PASS' if restore_ok else 'FAIL'}")
    if not restore_ok:
        failures += 1

    out.rule("8e. the synthetic noise processes")
    out(f"{'process':>18} {'same seed':>12} {'other seed differs':>20}")
    levels = {"white_pm": 1.0e-9, "white_fm": 1.0e-20, "random_walk_fm": 1.0e-24}
    for kind in NOISE_KINDS:
        a = synthetic_phase(kind, 500, tau0_s=1.0, level=levels[kind], seed=5).tobytes()
        b = synthetic_phase(kind, 500, tau0_s=1.0, level=levels[kind], seed=5).tobytes()
        c = synthetic_phase(kind, 500, tau0_s=1.0, level=levels[kind], seed=6).tobytes()
        out(f"{kind:>18} {str(a == b):>12} {str(a != c):>20}")
        if not (a == b and a != c):
            failures += 1

    out.rule("8f. the simulated and device backends built twice each")
    sim_a, dev_a = make_backend_pair(SPEC)
    sim_b, dev_b = make_backend_pair(SPEC)
    for backend in (sim_a, dev_a, sim_b, dev_b):
        backend.open()
    traces = [
        acquire(backend, N, min_exchanges=8).exchanges.timestamp_bytes()
        for backend in (sim_a, dev_a, sim_b, dev_b)
    ]
    for backend in (sim_a, dev_a, sim_b, dev_b):
        backend.close()
    all_equal = len(set(traces)) == 1
    out("four acquisitions: simulated A, device A, simulated B, device B")
    out(f"all byte-identical : {all_equal}")
    out(f"verdict            : {'PASS' if all_equal else 'FAIL'}")
    if not all_equal:
        failures += 1

    out.rule("8g. the dataset generator and the learned models")
    config = GeneratorConfig(window=60)
    n_data = scaled(200, minimum=40)
    data_a = build_dataset(n_data, seed=SEED, config=config)
    data_b = build_dataset(n_data, seed=SEED, config=config)
    features_same = data_a.features.tobytes() == data_b.features.tobytes()
    target_same = data_a.target_s.tobytes() == data_b.target_s.tobytes()
    out(f"features identical : {features_same}")
    out(f"targets identical  : {target_same}")
    shifted = build_dataset(20, seed=SEED + 5, config=config)
    scenario_match = data_a.features[5].tobytes() == shifted.features[0].tobytes()
    out(f"scenario i depends only on base seed + i : {scenario_match}")
    train, test = data_a.split(train_fraction=0.7)
    training = TrainingConfig(n_estimators=60, min_samples_leaf=4)
    predictions = []
    sigmas = []
    for _ in range(2):
        gbr = LearnedAsymmetryCorrector(training).fit(
            data_a.features[train], data_a.target_s[train]
        )
        ridge = RidgeAsymmetryCorrector().fit(data_a.features[train], data_a.target_s[train])
        predictions.append(
            (
                gbr.predict(data_a.features[test]).tobytes(),
                ridge.predict(data_a.features[test]).tobytes(),
            )
        )
        sigmas.append(gbr.predict_with_uncertainty(data_a.features[test])[1].tobytes())
    models_same = predictions[0] == predictions[1] and sigmas[0] == sigmas[1]
    out(f"learned predictions identical on a refit : {predictions[0] == predictions[1]}")
    out(f"learned sigmas identical on a refit      : {sigmas[0] == sigmas[1]}")
    all_ok = features_same and target_same and scenario_match and models_same
    out(f"verdict : {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        failures += 1

    out.rule("summary")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
