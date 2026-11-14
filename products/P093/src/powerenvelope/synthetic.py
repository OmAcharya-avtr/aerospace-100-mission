"""A deterministic SYNTHETIC campaign generator that writes the real CSV schema.

What this is for, and what it is not for
----------------------------------------

It exists so that the arithmetic of this product can be checked against a truth
that is known in closed form, and so that the unit tests and the ``--simulate``
path of the CLI have a campaign to run on. Every number it produces is
SYNTHETIC: the campaign it writes carries
:attr:`~powerenvelope.provenance.Origin.SYNTHETIC`, its ``manifest.json``
carries ``"synthetic": true``, and it drops a ``SYNTHETIC`` marker file beside
the traces.

**It is never a fallback for a missing measured file.** Nothing in
:mod:`powerenvelope.trace`, in ``validation/`` or in ``examples/`` calls it when
a measured trace is absent: those paths refuse and exit non-zero. A synthetic
figure presented where a measured one belongs would look like a measurement and
not be one, which is the single failure mode this product exists to make
impossible.

The model it writes, so the truth is computable without this code
----------------------------------------------------------------

Rail current, in whole multiples of the 8 mA quantum, against a constant bus
voltage :data:`SYNTHETIC_V_MV`:

.. code-block:: text

    I(t) = I_floor
         + I_active   when t is inside an active phase
         + I_ramp     when t is additionally within RAMP_S of that phase's start

Power is ``V * I / 1000`` in mW, exactly as the campaign's sampler computes it.
Samples are on a uniform grid of :data:`SYNTHETIC_SAMPLE_PERIOD_S` starting at
:data:`SYNTHETIC_T0`, so for the default parameters every quantity this product
computes has a closed form that
``validation/validate_baselines_synthetic.py`` writes out independently from
the declared constants. The active phase of cycle ``c`` is
``[c*CYCLE_S, c*CYCLE_S + duty*CYCLE_S)`` and the per-cell latency divides
``duty*CYCLE_S`` exactly for every duty in the plan, so inferences tile the
active phase with no remainder and the within-active accounting factor of the
decomposition is exactly 1.

Acceptance is a declared deterministic rule, :func:`gate_outcome`, applied in
exactly the order ``data/campaign.py`` applies its gate: no candidate, then
border, then area, then score against ``tau``. The synthetic score distribution
is uniform on a declared grid, and the strict and permissive taus are set at its
65th and 5th percentiles, so the synthetic acceptance rate responds to ``tau``
the way the campaign's does and the permissive control cell genuinely accepts
more. The rate is still computed by COUNTING the rows, never asserted.

Two optional perturbations exist for tests that need a non-ideal trace:
``current_jitter_lsb`` moves the current by whole quanta from a seeded
generator, and ``time_jitter_s`` makes the sample grid non-uniform. Both default
to zero, because the closed-form checks require the ideal grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .provenance import Origin
from .schema import (
    AREA_MAX_DECLARED,
    AREA_MIN_DECLARED,
    CURRENT_LSB_MA,
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    N_CORES,
    POWER_COLUMNS,
)
from .trace import Campaign, load_campaign

#: Bus voltage the generator writes on every channel-1 sample, mV. Not a
#: measurement: a declared constant of the synthetic model.
SYNTHETIC_V_MV = 5088

#: Background current, in 8 mA quanta.
SYNTHETIC_I_FLOOR_LSB = 47

#: Additional current while an inference is running, in 8 mA quanta.
SYNTHETIC_I_ACTIVE_LSB = 15

#: Additional current during the settling interval after a wake, in 8 mA quanta.
SYNTHETIC_I_RAMP_LSB = 5

#: Settling interval of the synthetic model, seconds. Three sample periods.
SYNTHETIC_RAMP_S = 0.15

#: Sampler period of the synthetic model, seconds.
SYNTHETIC_SAMPLE_PERIOD_S = 0.05

#: Cycle length of the synthetic model, seconds. The same 2.0 s the campaign
#: declares, because the loader refuses any other value.
SYNTHETIC_CYCLE_S = 2.0

#: Monotonic origin of every synthetic run. A plausible uptime, so that nothing
#: downstream can accidentally depend on timestamps near zero.
SYNTHETIC_T0 = 100000.0

#: Per-inference latency by frame dimension, seconds. Chosen so that each one
#: divides ``duty * CYCLE_S`` exactly for every duty in the plan.
SYNTHETIC_LATENCY_S: dict[int, float] = {48: 0.005, 64: 0.010, 96: 0.020}

#: The synthetic score grid: ``SCORE_BASE + ((idx * SCORE_MULT) % SCORE_MOD) *
#: SCORE_STEP``, which is uniform on 100 levels from 20.0 to 69.5 because
#: ``SCORE_MULT`` is coprime with ``SCORE_MOD``.
SCORE_BASE = 20.0
SCORE_STEP = 0.5
SCORE_MULT = 7
SCORE_MOD = 100

#: Moduli of the three geometric rejection causes, and the two areas used.
#: ``AREA_BAD`` is above the campaign's declared ``area_max`` of 60.
NO_CANDIDATE_MOD = 25
BORDER_MOD = 17
AREA_OUT_MOD = 23
AREA_OK = 9
AREA_BAD = 95

#: Gate threshold by frame dimension in the strict configuration, at the 65th
#: percentile of the score grid or just off it, and the permissive value at its
#: 5th percentile. Declared constants of the synthetic model, chosen before any
#: rate was counted, so that the strict rate lands near 0.35 of the
#: geometry-passing inferences the way the campaign's rule does.
SYNTHETIC_TAU_STRICT: dict[int, float] = {48: 50.0, 64: 52.5, 96: 55.0}
SYNTHETIC_TAU_PERMISSIVE = 22.5

#: Foreign CPU occupancy of the synthetic board, as a fraction. The generator
#: adds the pipeline's own occupancy of one core out of :data:`N_CORES` on top
#: of this, so a monitor second with no inference in it reads exactly this.
SYNTHETIC_FOREIGN_BUSY = 0.08

#: The default plan: the same twenty cell names the campaign writes.
DEFAULT_PLAN: tuple[tuple[str, int, float, float, bool], ...] = (
    ("floor", 64, 0.0, 0.0, False),
    *(
        (f"w{dim}_d{int(round(duty * 100)):03d}", dim, duty, SYNTHETIC_TAU_STRICT[dim], True)
        for dim in (48, 64, 96)
        for duty in (0.10, 0.25, 0.50, 0.75, 1.00)
    ),
    ("w64_d050_rep2", 64, 0.50, SYNTHETIC_TAU_STRICT[64], True),
    ("w64_d050_rep3", 64, 0.50, SYNTHETIC_TAU_STRICT[64], True),
    ("w64_d050_rep4", 64, 0.50, SYNTHETIC_TAU_STRICT[64], True),
    ("w64_d050_permissive", 64, 0.50, SYNTHETIC_TAU_PERMISSIVE, True),
)

#: Default number of cycles per synthetic run. Ten cycles is 20 s, a quarter of
#: the campaign's 38, which keeps a whole synthetic campaign under a second to
#: write and still gives nine or more windows per run at the default length.
DEFAULT_CYCLES = 10


def synthetic_score(frame_idx: int) -> float:
    """The declared synthetic score of one frame index, dimensionless.

    Zero when the frame has no candidate at all, which the campaign's gate reads
    as ``no_candidate``. Otherwise a point of the uniform grid declared above.
    """
    if frame_idx % NO_CANDIDATE_MOD == 0:
        return 0.0
    return SCORE_BASE + ((frame_idx * SCORE_MULT) % SCORE_MOD) * SCORE_STEP


def gate_outcome(frame_idx: int, tau: float) -> tuple[int, str, float, int, int, int]:
    """Apply the declared synthetic gate, in the campaign's own order.

    Returns ``(accepted, reason, score, n_candidates, area, touches_border)``.
    The order of the tests is the order of ``gate()`` in ``data/campaign.py``:
    no candidate, then border, then area outside the declared bounds, then score
    below ``tau``. Deterministic in the frame index and ``tau`` alone.
    """
    score = synthetic_score(frame_idx)
    candidates = 0 if score <= 0.0 else 2
    border = 1 if (candidates and frame_idx % BORDER_MOD == 0) else 0
    area = AREA_BAD if frame_idx % AREA_OUT_MOD == 0 else AREA_OK
    if score <= 0.0:
        return 0, "no_candidate", 0.0, 0, 0, 0
    if border:
        return 0, "border", score, candidates, area, border
    if area < AREA_MIN_DECLARED or area > AREA_MAX_DECLARED:
        return 0, "area_out", score, candidates, area, border
    if score < tau:
        return 0, "low_score", score, candidates, area, border
    return 1, "accept", score, candidates, area, border


def _active_phase_s(duty: float, cycle_s: float) -> float:
    return duty * cycle_s


def inference_rows(
    dim: int, duty: float, cycles: int, tau: float, cycle_s: float = SYNTHETIC_CYCLE_S
) -> list[tuple]:
    """The decision rows of one synthetic cell, in campaign column order."""
    latency = SYNTHETIC_LATENCY_S[dim]
    active = _active_phase_s(duty, cycle_s)
    per_cycle = int(round(active / latency))
    rows: list[tuple] = []
    index = 0
    for cycle in range(cycles):
        base = SYNTHETIC_T0 + cycle * cycle_s
        for slot in range(per_cycle):
            t_start = base + slot * latency
            accepted, reason, score, candidates, area, border = gate_outcome(index, tau)
            rows.append(
                (
                    t_start,
                    t_start + latency,
                    cycle,
                    index,
                    score,
                    candidates,
                    area,
                    border,
                    accepted,
                    reason,
                )
            )
            index += 1
    return rows


def _write_csv(path: Path, header: tuple[str, ...], rows: list[str]) -> None:
    path.write_text(",".join(header) + "\n" + "".join(rows))


def _write_power(
    path: Path,
    duty: float,
    cycles: int,
    has_pipeline: bool,
    cycle_s: float,
    current_jitter_lsb: int,
    time_jitter_s: float,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Write one synthetic ``power_<cell>.csv`` and return ``(t0, t1, n_samples)``.

    The active and ramp masks are decided by INTEGER sample index, not by a
    floating-point phase. Computing ``offset - floor(offset / cycle_s) * cycle_s``
    and comparing it with a boundary puts a sample exactly on a boundary on
    whichever side the rounding of that particular offset happens to fall, and
    the counts then differ between otherwise identical cycles: measured here,
    cycle 0 got three ramp samples and cycle 1 got four. An integer comparison
    cannot do that, and the closed form in
    ``validation/validate_baselines_synthetic.py`` depends on it not doing it.

    ``time_jitter_s`` perturbs the RECORDED timestamp only, which is what makes
    the integration grid non-uniform; it does not move the model's phase
    boundaries, so a jittered trace is still a trace of the same power model.
    """
    period = SYNTHETIC_SAMPLE_PERIOD_S
    per_cycle = int(round(cycle_s / period))
    n = cycles * per_cycle
    active_slots = int(round(_active_phase_s(duty, cycle_s) / period)) if has_pipeline else 0
    ramp_slots = int(round(SYNTHETIC_RAMP_S / period))
    lines: list[str] = []
    for index in range(n):
        offset = index * period
        t = SYNTHETIC_T0 + offset
        if time_jitter_s:
            t += float(rng.uniform(-time_jitter_s, time_jitter_s))
        slot = index % per_cycle
        lsb = SYNTHETIC_I_FLOOR_LSB
        if slot < active_slots:
            lsb += SYNTHETIC_I_ACTIVE_LSB
            if slot < ramp_slots:
                lsb += SYNTHETIC_I_RAMP_LSB
        if current_jitter_lsb:
            lsb += int(rng.integers(-current_jitter_lsb, current_jitter_lsb + 1))
        current_ma = int(max(1, lsb) * CURRENT_LSB_MA)
        cv_ma = int(round(current_ma * 0.45 / CURRENT_LSB_MA)) * int(CURRENT_LSB_MA)
        soc_ma = int(round(current_ma * 0.18 / CURRENT_LSB_MA)) * int(CURRENT_LSB_MA)
        power = SYNTHETIC_V_MV * current_ma / 1000.0
        lines.append(
            f"{t:.6f},{SYNTHETIC_V_MV:d},{current_ma:d},760,{cv_ma:d},1080,"
            f"{soc_ma:d},{power:.3f}\n"
        )
    _write_csv(path, POWER_COLUMNS, lines)
    return SYNTHETIC_T0, SYNTHETIC_T0 + (n - 1) * period, n


def _write_monitor(
    path: Path, duty: float, cycles: int, has_pipeline: bool, cycle_s: float
) -> None:
    """Write one synthetic ``monitor_<cell>.csv``.

    The occupancy a monitor sample reports is the fraction of the preceding
    second that the pipeline occupied, divided by :data:`N_CORES` because the
    pipeline is single-threaded, plus :data:`SYNTHETIC_FOREIGN_BUSY`. With a
    2.0 s cycle the overlap has a closed form -- an even-numbered second
    contains the start of an active phase, an odd-numbered one contains only
    whatever of it ran past the one-second mark -- so a second with no inference
    in it reads exactly the foreign figure. That is what makes
    :func:`powerenvelope.windows.pipeline_free_monitor_mask` testable.
    """
    seconds = int(round(cycles * cycle_s))
    active = _active_phase_s(duty, cycle_s) if has_pipeline else 0.0
    lines: list[str] = []
    for index in range(1, seconds + 1):
        t = SYNTHETIC_T0 + index
        start = index - 1
        occupied = min(active, 1.0) if start % 2 == 0 else max(0.0, active - 1.0)
        busy = SYNTHETIC_FOREIGN_BUSY + occupied / N_CORES
        tj = 44.5 + 6.0 * duty + 0.05 * index
        load1 = 1.20 + 0.40 * duty
        clocks = ",".join(["1728000"] * N_CORES)
        lines.append(f"{t:.6f},{busy:.6f},{tj:.3f},{load1:.2f},{clocks}\n")
    _write_csv(path, MONITOR_COLUMNS, lines)


def write_campaign(
    directory: Path | str,
    plan: tuple[tuple[str, int, float, float, bool], ...] = DEFAULT_PLAN,
    cycles: int = DEFAULT_CYCLES,
    seed: int = 930093,
    current_jitter_lsb: int = 0,
    time_jitter_s: float = 0.0,
    complete: bool = True,
) -> Path:
    """Write a SYNTHETIC campaign into ``directory`` and return its path.

    Parameters
    ----------
    directory
        Target directory. Created if absent. Anything already in it with a
        colliding name is overwritten.
    plan
        ``(cell, frame_dim, duty, tau, has_pipeline)`` per cell.
    cycles
        Cycles per run.
    seed
        Seed for the optional perturbations. Fixed by default so that a
        regenerated campaign is byte-identical.
    current_jitter_lsb
        Perturb the current by up to this many whole 8 mA quanta. Zero keeps the
        closed form exact.
    time_jitter_s
        Perturb each sample timestamp by up to this much. Zero keeps the sample
        grid uniform.
    complete
        Whether to write the ``CAMPAIGN_COMPLETE`` marker. Pass False to build
        the fixture a refusal test needs.

    Raises
    ------
    ValueError
        On a non-positive cycle count, or a plan entry whose frame dimension has
        no declared latency, or a duty whose active phase is not an exact
        multiple of that latency.
    """
    if cycles < 1:
        raise ValueError(f"cycles must be at least 1, got {cycles}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cycle_s = SYNTHETIC_CYCLE_S
    runs: list[dict] = []
    for label, dim, duty, tau, has_pipeline in plan:
        if dim not in SYNTHETIC_LATENCY_S:
            raise ValueError(
                f"cell {label!r} has frame dimension {dim}, which has no declared "
                f"synthetic latency; declared: {sorted(SYNTHETIC_LATENCY_S)}"
            )
        latency = SYNTHETIC_LATENCY_S[dim]
        active = _active_phase_s(duty, cycle_s)
        if has_pipeline and abs(active / latency - round(active / latency)) > 1e-12:
            raise ValueError(
                f"cell {label!r}: active phase {active} s is not an exact multiple of "
                f"the latency {latency} s, so inferences would not tile it and the "
                "closed form would not hold"
            )
        t0, t1, n_samples = _write_power(
            root / f"power_{label}.csv",
            duty,
            cycles,
            has_pipeline,
            cycle_s,
            current_jitter_lsb,
            time_jitter_s,
            rng,
        )
        _write_monitor(root / f"monitor_{label}.csv", duty, cycles, has_pipeline, cycle_s)
        inferences = 0
        accepted = 0
        active_cpu = 0.0
        if has_pipeline:
            rows = inference_rows(dim, duty, cycles, tau, cycle_s)
            lines = [
                f"{row[0]:.6f},{row[1]:.6f},{row[2]:d},{row[3]:d},{row[4]:.6f},"
                f"{row[5]:d},{row[6]:d},{row[7]:d},{row[8]:d},{row[9]}\n"
                for row in rows
            ]
            _write_csv(root / f"decisions_{label}.csv", DECISION_COLUMNS, lines)
            inferences = len(rows)
            accepted = sum(row[8] for row in rows)
            active_cpu = inferences * latency
        runs.append(
            {
                "cell": label,
                "n": dim,
                "duty": duty,
                "tau": tau,
                "cycles": cycles,
                "t0_mono": t0,
                "t1_mono": t1,
                "epoch_at_t0": 1763000000.0,
                "inferences": inferences,
                "accepted": accepted,
                "active_cpu_s": active_cpu,
                "wall_s": cycles * cycle_s,
                "sampler_rc": 0,
                "monitor_rc": 0,
                "pipeline": has_pipeline,
                "rail_samples": n_samples,
            }
        )
    manifest = {
        "product": "P093 PowerEnvelope",
        "role": "campaign",
        "synthetic": True,
        "origin": "synthetic",
        "generator": "src/powerenvelope/synthetic.py",
        "note": (
            "SYNTHETIC. Written by the committed generator, not measured on any "
            "board. Every figure derived from this directory is synthetic and the "
            "scripts that use it say so at the point of use."
        ),
        "cycle_s": cycle_s,
        "cycles_per_run": cycles,
        "sample_period_s": SYNTHETIC_SAMPLE_PERIOD_S,
        "grad_thr": 90,
        "area_ref": 6.0,
        "area_min": AREA_MIN_DECLARED,
        "area_max": AREA_MAX_DECLARED,
        "tau_rule": (
            "SYNTHETIC: tau is a declared constant per frame dimension, "
            "SYNTHETIC_TAU_STRICT, with SYNTHETIC_TAU_PERMISSIVE for the control "
            "cell. The campaign's own rule is a quantile of a calibration probe."
        ),
        "ina3221": {"update_interval": "0", "samples": "512"},
        "env": {
            "uname": "synthetic",
            "python": "synthetic",
            "ncpu_affinity": N_CORES,
            "governor": "synthetic",
        },
        "calibration": [
            {
                "n": dim,
                "latency_median_s": SYNTHETIC_LATENCY_S[dim],
                "tau_strict": SYNTHETIC_TAU_STRICT[dim],
                "tau_permissive": SYNTHETIC_TAU_PERMISSIVE,
                "synthetic": True,
            }
            for dim in sorted(SYNTHETIC_LATENCY_S)
        ],
        "plan": [entry[0] for entry in plan],
        "runs": runs,
        "aborted": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (root / "campaign.log").write_text(
        "[synthetic] written by src/powerenvelope/synthetic.py; "
        f"{len(plan)} cell(s), {cycles} cycle(s) each. Not a measurement.\n"
    )
    (root / "SYNTHETIC").write_text(
        "Every file in this directory was written by the committed synthetic "
        "generator. Nothing here was measured on hardware.\n"
    )
    if complete:
        (root / "CAMPAIGN_COMPLETE").write_text("synthetic\n")
    return root


def load_synthetic_campaign(directory: Path | str, **kwargs) -> Campaign:
    """Write a synthetic campaign into ``directory`` and load it as SYNTHETIC."""
    root = write_campaign(directory, **kwargs)
    return load_campaign(root, origin=Origin.SYNTHETIC, directory_label=root.name)
