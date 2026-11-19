"""Generate the committed handover corpus, deterministically, from fixed seeds.

Run from the repository root::

    python3 data/generate_corpus.py

Writes ``data/corpus/handovers.csv`` and ``data/corpus/manifest.json``. Both are
committed: the CSV is under 1 MB and the manifest carries its sha256, which
``handoverloss.corpus.load_corpus`` checks on every load, so a reader can tell
whether the file on disk is the one the published numbers came from.

WHAT IS SYNTHETIC HERE, STATED PLAINLY
--------------------------------------

Everything. There is no measured handover in this repository. The corpus is a
deterministic simulation of the declared handover sequence of
``handoverloss.sequence``, driven by the parameter ranges declared below. The
ranges were chosen to span a plausible operating envelope for an optical
ground-station handover and to make all three loss mechanisms reachable, so that
the question "which dominates" has a non-degenerate answer across the corpus.
**No value below is a measurement reported by any cited source**, and
``DATASET_CARD.md`` says the same thing at more length.

The one piece of dataset design worth declaring explicitly: the ranges of the
utilisation ``rho = R_s / R_d``, the ramp duration and the buffer headroom were
widened after a first pass in which the acquisition mechanism dominated almost
every event, because a corpus in which one answer is nearly always right cannot
distinguish a predictor from a constant. The widening changed the GENERATOR, not
any baseline and not the model, and the resulting dominance mix is reported in
``DATASET_CARD.md`` and by ``validation/validate_decomposition_residual.py``.

STRUCTURE
---------

``PAIRS`` station pairs by ``EPOCHS`` time epochs, ``EVENTS_PER_CELL`` handovers
in each cell. Station labels are synthetic (``GS01`` and upward) and carry no
relation to any real ground station. Each pair has its own fixed configuration
(rates, buffer, protocol class, nominal geometry); each epoch shifts the
atmospheric conditions; each event draws its own occupancy at the break, its own
acquisition time, its own restart jitter and its own burst and service noise.

SEEDS
-----

One root seed, :data:`ROOT_SEED`. Per-pair and per-epoch streams are derived
from it by ``numpy.random.SeedSequence.spawn``, so adding an epoch does not
change any earlier epoch's draws.

COMPUTE
-------

The simulation is vectorised over events inside each epoch. Measured wall time
is printed at the end and recorded in ``validation/VALIDATION.md``; the declared
budget is 60 s of CPU.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from handoverloss.acquisition import AcquisitionModel  # noqa: E402
from handoverloss.corpus import (  # noqa: E402
    CORPUS_CSV,
    CORPUS_MANIFEST,
    FEATURE_COLUMNS,
    PROVENANCE_COLUMNS,
    TARGET_COLUMNS,
)
from handoverloss.sequence import MECHANISMS  # noqa: E402
from handoverloss.trace import EventParameters, simulate  # noqa: E402

#: Root seed. Every stream in this file derives from it.
ROOT_SEED = 96_096_2026

#: Corpus shape. 16 x 8 x 12 = 1536 events, which keeps the CSV under 1 MB and the
#: whole generation inside the declared 60 s CPU budget on the 2-core build
#: machine. 16 pairs so that holding out 4 of them is a quarter of the corpus.
PAIRS = 16
EPOCHS = 8
EVENTS_PER_CELL = 12

#: Stratification grid for the utilisation ``rho = R_s / R_d``, one value per
#: pair, cycled. Declared rather than drawn, because the drain mechanism's
#: closed-form term scales as ``rho`` and an independent uniform draw over the
#: two rates left only a handful of pairs anywhere near the top of the range.
#: A first pass without this grid produced a corpus in which the acquisition
#: mechanism dominated 1016 of 1536 events and the drain mechanism 90, which
#: cannot distinguish a predictor from a constant. The grid changes the
#: GENERATOR only; no baseline and no model was touched. The resulting mix is
#: reported in DATASET_CARD.md.
UTILISATION_GRID = (0.04, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.74, 0.82, 0.88, 0.92, 0.95)

#: Stratification grid for the declared egress ramp, seconds, cycled over pairs
#: out of phase with the utilisation grid so the two are not collinear.
RAMP_GRID_S = (4.0, 14.0, 32.0, 58.0, 86.0, 120.0, 170.0)

#: Stratification grid for the declared protocol-restart class, seconds, cycled
#: out of phase with both grids above.
RESTART_GRID_S = (2.0, 6.0, 13.0, 24.0, 38.0, 56.0, 78.0)

#: Accounting window, seconds. Long enough to contain the longest excursion any
#: declared parameter combination can produce: the longest declared restart
#: (78 * 1.15 s), plus the longest declared ramp (170 * 1.15 s), plus room for a
#: multi-attempt acquisition tail. Acquisition draws beyond that room are capped
#: to the window and the number capped is recorded in the manifest.
HORIZON_S = 700.0

#: Safety margin, seconds, subtracted when capping a duration to the window.
#: Capping to exactly the remaining budget let a three-term sum round 6e-14
#: above the window and the EventParameters validator refused it, correctly.
CAP_MARGIN_S = 1.0e-6

#: Slot width, seconds. 0.25 s over a 420 s window is 1680 slots.
SLOT_SECONDS = 0.25

#: Declared per-slot burstiness and service jitter, log-scale shapes.
BURST_LOG_SIGMA = 0.45
SERVICE_LOG_SIGMA = 0.12


def _pair_configuration(index: int, rng: np.random.Generator) -> dict:
    """Fixed configuration of one station pair. Declared ranges, SI units.

    The configuration space spans, deliberately:

    * steady downlink rate 0.45 to 2.6 Gbit/s (an optical ground station);
    * utilisation ``rho = R_s / R_d`` taken from :data:`UTILISATION_GRID` with a
      small jitter, so the source rate is ``rho * R_d``. Stratified rather than
      drawn independently, for the reason stated at the grid's definition;
    * buffer 4 to 70 Gbit allocated to this stream;
    * protocol restart class from :data:`RESTART_GRID_S`, 2 to 78 s, spanning a
      fast link-layer resynchronisation and a slow store-and-forward session
      re-establishment;
    * egress ramp from :data:`RAMP_GRID_S`, 4 to 170 s to full rate;
    * acquisition median 6 to 58 s with log-scale shape 0.25 to 0.80 and an
      attempt window 70 to 150 s.

    Every bound here is a declared generator parameter. None is a measurement
    reported by any source cited anywhere in this repository.
    """
    downlink = float(rng.uniform(0.45e9, 2.6e9))
    utilisation = UTILISATION_GRID[index % len(UTILISATION_GRID)]
    utilisation = float(np.clip(utilisation * rng.uniform(0.92, 1.08), 0.02, 0.97))
    source = utilisation * downlink
    return {
        "station_from": f"GS{2 * index + 1:02d}",
        "station_to": f"GS{2 * index + 2:02d}",
        "source_rate_bps": source,
        "downlink_rate_bps": downlink,
        "buffer_capacity_bits": float(rng.uniform(4e9, 70e9)),
        "declared_restart_s": float(
            RESTART_GRID_S[(3 * index + 1) % len(RESTART_GRID_S)]
            * rng.uniform(0.85, 1.15)
        ),
        "declared_ramp_s": float(
            RAMP_GRID_S[(5 * index + 2) % len(RAMP_GRID_S)] * rng.uniform(0.85, 1.15)
        ),
        "acquisition_median_s": float(rng.uniform(6.0, 58.0)),
        "acquisition_log_sigma": float(rng.uniform(0.25, 0.80)),
        "acquisition_attempt_window_s": float(rng.uniform(70.0, 150.0)),
        "elevation_deg": float(rng.uniform(12.0, 78.0)),
        "source_burst_cv": BURST_LOG_SIGMA,
    }


def _epoch_conditions(rng: np.random.Generator) -> dict:
    """Atmospheric and operational conditions of one epoch. Declared ranges.

    ``turbulence_cn2_log10`` is the base-10 logarithm of a near-ground
    refractive-index structure parameter in m^(-2/3); the span -16.5 to -13.0
    covers weak to strong turbulence as those regimes are described in Andrews,
    L. C. and Phillips, R. L., *Laser Beam Propagation through Random Media*,
    second edition, SPIE Press, 2005. The value here is a declared covariate
    that drives the abort probability; it is not a measurement.

    ``beacon_margin_db`` is the link's fade margin in dB and ``sky_cover_index``
    a dimensionless 0-to-1 obscuration index; both are declared covariates.
    """
    return {
        "turbulence_cn2_log10": float(rng.uniform(-16.5, -13.0)),
        "beacon_margin_db": float(rng.uniform(1.5, 12.0)),
        "sky_cover_index": float(rng.uniform(0.0, 0.85)),
    }


def _abort_probability(
    turbulence_cn2_log10: float,
    beacon_margin_db: float,
    sky_cover_index: float,
    elevation_deg: float,
) -> float:
    """Per-attempt abort probability from the declared covariates, dimensionless.

    A declared monotone link function, not a fit: abort probability rises with
    turbulence strength and sky cover, falls with fade margin and elevation.
    Logistic in a weighted sum of the standardised covariates, clipped to
    ``[0.01, 0.55]`` so the acquisition model stays inside its own declared
    validity range.

    The DIRECTION of each term is what the physics supports -- stronger
    turbulence and lower elevation mean deeper and longer fades (Andrews and
    Phillips 2005), a larger fade margin means fewer aborts, and obscuration
    blocks the beacon outright (Kim, McArthur and Korevaar, SPIE 4214, 2001).
    The COEFFICIENTS are declared and arbitrary.
    """
    score = (
        1.30 * (turbulence_cn2_log10 + 14.75) / 1.75
        + 1.05 * (sky_cover_index - 0.4) / 0.4
        - 0.95 * (beacon_margin_db - 6.75) / 5.25
        - 0.60 * (elevation_deg - 45.0) / 33.0
    )
    return float(np.clip(1.0 / (1.0 + np.exp(-score)) * 0.62 - 0.03, 0.01, 0.55))


def main() -> int:
    started = time.perf_counter()
    root_sequence = np.random.SeedSequence(ROOT_SEED)
    pair_sequences = root_sequence.spawn(PAIRS)
    configurations = []
    for index in range(PAIRS):
        pair_rng = np.random.default_rng(pair_sequences[index])
        configurations.append(_pair_configuration(index, pair_rng))

    epoch_sequences = np.random.SeedSequence(ROOT_SEED + 1).spawn(EPOCHS)
    conditions = [
        _epoch_conditions(np.random.default_rng(seed)) for seed in epoch_sequences
    ]

    rows: list[dict] = []
    event_id = 0
    cell_sequences = np.random.SeedSequence(ROOT_SEED + 2).spawn(PAIRS * EPOCHS)

    for pair_index, configuration in enumerate(configurations):
        for epoch_index, condition in enumerate(conditions):
            rng = np.random.default_rng(cell_sequences[pair_index * EPOCHS + epoch_index])
            abort = _abort_probability(
                condition["turbulence_cn2_log10"],
                condition["beacon_margin_db"],
                condition["sky_cover_index"],
                configuration["elevation_deg"],
            )
            model = AcquisitionModel(
                median_s=configuration["acquisition_median_s"],
                log_sigma=configuration["acquisition_log_sigma"],
                abort_probability=abort,
                attempt_window_s=configuration["acquisition_attempt_window_s"],
            )
            count = EVENTS_PER_CELL

            # Occupancy at the break: a Beta draw over the whole range, so the
            # headroom spans "nearly empty" to "nearly full". An operator hands
            # over when the buffer already holds a pass's worth of data, which
            # is why the distribution is skewed towards full.
            occupancy_fraction = rng.beta(4.0, 1.6, size=count)
            capacity = configuration["buffer_capacity_bits"]
            occupancy = occupancy_fraction * capacity

            acquisition = model.sample(count, rng)
            # Restart latency: the declared class plus a gamma jitter with mean
            # 12 per cent of the class, so the realised value is never the
            # declared one and a predictor cannot read it off.
            declared_restart = configuration["declared_restart_s"]
            restart = declared_restart + rng.gamma(
                2.0, 0.06 * declared_restart, size=count
            )
            ramp = np.full(count, configuration["declared_ramp_s"])

            # The window must contain the whole excursion for every event. Cap
            # the acquisition tail rather than lengthening the window for one
            # outlier: a capped event is recorded as capped in the manifest.
            budget = (
                HORIZON_S
                - declared_restart
                - configuration["declared_ramp_s"]
                - CAP_MARGIN_S
            )
            capped = int(np.sum(acquisition > budget))
            acquisition = np.minimum(acquisition, max(budget, 0.0))
            restart = np.minimum(
                restart,
                HORIZON_S
                - acquisition
                - configuration["declared_ramp_s"]
                - CAP_MARGIN_S,
            )

            parameters = EventParameters(
                source_rate_bps=np.full(count, configuration["source_rate_bps"]),
                downlink_rate_bps=np.full(count, configuration["downlink_rate_bps"]),
                buffer_capacity_bits=np.full(count, capacity),
                initial_occupancy_bits=occupancy,
                acquisition_s=acquisition,
                restart_s=restart,
                ramp_s=ramp,
                horizon_s=HORIZON_S,
            )
            result = simulate(
                parameters,
                rng,
                slot_seconds=SLOT_SECONDS,
                burst_log_sigma=BURST_LOG_SIGMA,
                service_log_sigma=SERVICE_LOG_SIGMA,
            )
            dominant = result.dominant_names()

            for row_index in range(count):
                rows.append(
                    {
                        "event_id": event_id,
                        "pair_id": pair_index,
                        "station_from": configuration["station_from"],
                        "station_to": configuration["station_to"],
                        "epoch": epoch_index,
                        "source_rate_bps": configuration["source_rate_bps"],
                        "downlink_rate_bps": configuration["downlink_rate_bps"],
                        "buffer_capacity_bits": capacity,
                        "initial_occupancy_bits": float(occupancy[row_index]),
                        "occupancy_fraction": float(occupancy_fraction[row_index]),
                        "headroom_s": float(
                            (capacity - occupancy[row_index])
                            / configuration["source_rate_bps"]
                        ),
                        "utilisation": configuration["source_rate_bps"]
                        / configuration["downlink_rate_bps"],
                        "declared_ramp_s": configuration["declared_ramp_s"],
                        "declared_restart_s": declared_restart,
                        "acquisition_median_s": configuration["acquisition_median_s"],
                        "acquisition_log_sigma": configuration["acquisition_log_sigma"],
                        "acquisition_abort_probability": abort,
                        "acquisition_attempt_window_s": configuration[
                            "acquisition_attempt_window_s"
                        ],
                        "elevation_deg": configuration["elevation_deg"],
                        "turbulence_cn2_log10": condition["turbulence_cn2_log10"],
                        "beacon_margin_db": condition["beacon_margin_db"],
                        "sky_cover_index": condition["sky_cover_index"],
                        "source_burst_cv": configuration["source_burst_cv"],
                        "loss_acquisition_bits": float(result.dropped_bits[row_index, 0]),
                        "loss_restart_bits": float(result.dropped_bits[row_index, 1]),
                        "loss_drain_bits": float(result.dropped_bits[row_index, 2]),
                        "realised_acquisition_s": float(acquisition[row_index]),
                        "realised_restart_s": float(restart[row_index]),
                        "horizon_s": HORIZON_S,
                        "produced_bits": float(result.produced_bits[row_index]),
                        "delivered_bits": float(result.delivered_bits[row_index]),
                        "final_occupancy_bits": float(
                            result.final_occupancy_bits[row_index]
                        ),
                        "conservation_total_bits": float(
                            result.conservation_total_bits[row_index]
                        ),
                        "dominant": dominant[row_index],
                        "_capped": capped,
                    }
                )
                event_id += 1

    header = [
        "event_id",
        "pair_id",
        "station_from",
        "station_to",
        "epoch",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
        "realised_acquisition_s",
        "realised_restart_s",
        "horizon_s",
        "produced_bits",
        "delivered_bits",
        "final_occupancy_bits",
        "conservation_total_bits",
        "dominant",
    ]
    missing = [name for name in (*FEATURE_COLUMNS, *TARGET_COLUMNS, *PROVENANCE_COLUMNS)
               if name not in header]
    if missing:
        raise SystemExit(f"header is missing declared columns: {missing}")

    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                row[name] if isinstance(row[name], str) else repr(row[name])
                for name in header
            )
        )
    text = "\n".join(lines) + "\n"
    target = ROOT / CORPUS_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    counts = {name: 0 for name in MECHANISMS}
    for row in rows:
        counts[row["dominant"]] += 1
    capped_total = sum({(r["pair_id"], r["epoch"]): r["_capped"] for r in rows}.values())
    elapsed = time.perf_counter() - started

    manifest = {
        "generator": "data/generate_corpus.py",
        "root_seed": ROOT_SEED,
        "pairs": PAIRS,
        "epochs": EPOCHS,
        "events_per_cell": EVENTS_PER_CELL,
        "events": len(rows),
        "horizon_s": HORIZON_S,
        "slot_seconds": SLOT_SECONDS,
        "slots_per_event": int(np.ceil(HORIZON_S / SLOT_SECONDS)),
        "burst_log_sigma": BURST_LOG_SIGMA,
        "service_log_sigma": SERVICE_LOG_SIGMA,
        "acquisition_times_capped_to_window": capped_total,
        "dominance_counts": counts,
        "csv": CORPUS_CSV,
        "sha256": digest,
        "bytes": len(text.encode("utf-8")),
        "origin": "SYNTHETIC. No measured handover appears in this repository.",
        "generation_seconds": round(elapsed, 3),
    }
    (ROOT / CORPUS_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {CORPUS_CSV}  ({len(rows)} events, {len(text.encode('utf-8'))} bytes)")
    print(f"wrote {CORPUS_MANIFEST}")
    print(f"sha256 {digest}")
    print(f"dominance counts {counts}")
    print(f"acquisition times capped to the window: {capped_total}")
    print(f"generation seconds {elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
