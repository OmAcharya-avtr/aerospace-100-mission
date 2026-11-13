"""A committed, deterministic synthetic campaign generator.

Everything in Stage A of this product's build -- the loader, the schema, the
accounting, all three baselines, the bootstrap, the bounds, the windowing and
the learned predictor -- is tested against traces this module writes. Nothing
derived from it is a measurement of any board, and every script and test that
uses it says so.

The generator is faithful in the one respect that matters: it reproduces the
campaign loop's **release and drop logic** exactly, rather than writing
plausible-looking columns. Given a sequence of execution times it runs

    release = i*T;  start = max(release, previous_end);  end = start + exec
    response = end - release;  missed = response > D
    next_index = floor(end / T) + 1;  dropped_before = next_index - i - 1

which is the loop in ``data/campaign.py``. A trace it writes therefore satisfies
``releases = sum(1 + dropped_before)`` and ``misses = sum(missed +
dropped_before)`` by construction and not by assertion.

Execution times are drawn as

    exec = base * contention(n_workers) * (1 + heavy_tail_spike)

with ``contention`` increasing in the worker count and the spike a Pareto draw
fired with a declared probability. ``base`` is lognormal by default and Gaussian
when ``distribution="normal"``.

Both knobs exist for the negative controls, and which direction each pushes the
mean-plus-margin rule was MEASURED here rather than assumed:

- a lognormal base with no Pareto spikes is right-skewed, so the Gaussian rule
  **understates** the tail at the deadline;
- a Gaussian base makes the rule approximately right by construction, which is
  the must-NOT-fire control;
- a Pareto spike fired often enough inflates the sample standard deviation more
  than it inflates the exceedance count, so the Gaussian rule **overstates**.
  That is the opposite of the direction one might expect of a heavy tail and it
  is why the controls assert a direction rather than a magnitude.

Determinism is by fixed seed through ``numpy.random.Generator`` (PCG64), and
``tests/test_fixtures.py`` asserts the written bytes are identical across two
calls at the same seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deadlinerisk.provenance import Origin
from deadlinerisk.schema import JOB_COLUMNS, MONITOR_COLUMNS, N_CORES
from deadlinerisk.trace import Campaign, load_campaign

#: Period and implicit deadline of the synthetic loop, seconds. Same as the
#: measured campaign's, so a test's numbers are the same order of magnitude.
PERIOD_S = 0.010
DEADLINE_S = PERIOD_S

#: Nominal median execution time as a fraction of the deadline on an idle board.
IDLE_FRACTION = 0.45

#: Declared default seed for every fixture in this product.
SEED = 920092


@dataclass(frozen=True)
class RunSpec:
    """One synthetic run: label, stressor counts and duration."""

    run: str
    n_cpu: int
    n_mem: int
    duration_s: float
    tail_probability: float = 0.004
    tail_alpha: float = 1.6
    distribution: str = "lognormal"
    idle_fraction: float = IDLE_FRACTION
    spread: float = 0.09


#: The default synthetic campaign's run plan. It mirrors the measured
#: campaign's shape -- one long ground-truth run plus CPU levels 0..6, a
#: memory-only mix and a combined mix -- at a tenth of the duration, so a test
#: suite runs in seconds.
DEFAULT_PLAN: tuple[RunSpec, ...] = (
    RunSpec("gt_long", 4, 0, 60.0),
    *(RunSpec(f"mix_c{level}", level, 0, 20.0) for level in range(7)),
    RunSpec("mix_mem2", 0, 2, 20.0),
    RunSpec("mix_c2mem1", 2, 1, 20.0),
)


def contention_factor(n_cpu: int, n_mem: int) -> float:
    """Multiplier on the idle execution time, dimensionless.

    A declared synthetic relationship, not a model of any hardware: each CPU
    worker adds 14 per cent and each memory worker 26 per cent, the memory
    worker costing more because it also competes for bandwidth. Monotone in both
    arguments, which is the only property any test relies on.
    """
    if n_cpu < 0 or n_mem < 0:
        raise ValueError(f"worker counts must be >= 0, got ({n_cpu}, {n_mem})")
    return 1.0 + 0.14 * n_cpu + 0.26 * n_mem


def execution_times(
    n_jobs: int,
    n_cpu: int,
    n_mem: int,
    rng: np.random.Generator,
    tail_probability: float = 0.004,
    tail_alpha: float = 1.6,
    idle_fraction: float = IDLE_FRACTION,
    deadline_s: float = DEADLINE_S,
    distribution: str = "lognormal",
    spread: float = 0.09,
) -> np.ndarray:
    """Draw ``n_jobs`` synthetic execution times, seconds.

    Parameters
    ----------
    distribution
        ``"lognormal"`` for a right-skewed base, ``"normal"`` for a Gaussian
        base clipped below at a tenth of the median so no draw is negative.
    spread
        Lognormal sigma, or the Gaussian standard deviation as a fraction of
        the median, dimensionless.

    Raises
    ------
    ValueError
        On a non-positive job count, a tail probability outside ``[0, 1]``, a
        non-positive Pareto exponent, a non-positive spread, or an unknown
        distribution name.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    if not 0.0 <= tail_probability <= 1.0:
        raise ValueError(f"tail_probability must be in [0, 1], got {tail_probability!r}")
    if tail_alpha <= 0.0:
        raise ValueError(f"tail_alpha must be > 0, got {tail_alpha!r}")
    if spread <= 0.0:
        raise ValueError(f"spread must be > 0, got {spread!r}")
    median = idle_fraction * deadline_s
    if distribution == "lognormal":
        base = median * np.exp(rng.normal(0.0, spread, size=n_jobs))
    elif distribution == "normal":
        base = np.maximum(
            0.1 * median, median * (1.0 + rng.normal(0.0, spread, size=n_jobs))
        )
    else:
        raise ValueError(
            f"distribution must be 'lognormal' or 'normal', got {distribution!r}"
        )
    scaled = base * contention_factor(n_cpu, n_mem)
    fires = rng.random(n_jobs) < tail_probability
    spike = (rng.pareto(tail_alpha, size=n_jobs) + 1.0) * 0.55
    return scaled * np.where(fires, 1.0 + spike, 1.0)


def simulate_loop(
    exec_s: np.ndarray, period_s: float = PERIOD_S, deadline_s: float = DEADLINE_S
) -> dict[str, np.ndarray]:
    """Run the campaign loop's release and drop logic over given exec times.

    Returns a dict of the eight :data:`deadlinerisk.schema.JOB_COLUMNS` arrays.
    The number of ROWS is at most ``len(exec_s)``: the loop consumes one
    execution time per executed job and stops when the next release index would
    run past the supplied draws.
    """
    if period_s <= 0.0 or deadline_s <= 0.0:
        raise ValueError("period_s and deadline_s must both be > 0")
    draws = np.asarray(exec_s, dtype=np.float64)
    index: list[int] = []
    release: list[float] = []
    delay: list[float] = []
    end: list[float] = []
    execution: list[float] = []
    response: list[float] = []
    missed: list[int] = []
    dropped: list[int] = []
    i = 0
    previous_end = 0.0
    for draw in draws.tolist():
        rel = i * period_s
        start = max(rel, previous_end)
        finish = start + draw
        resp = finish - rel
        nxt = int(finish // period_s) + 1
        drop = max(0, nxt - i - 1)
        index.append(i)
        release.append(rel)
        delay.append(start - rel)
        end.append(finish)
        execution.append(draw)
        response.append(resp)
        missed.append(1 if resp > deadline_s else 0)
        dropped.append(drop)
        previous_end = finish
        i = nxt
    return {
        "job_index": np.asarray(index, dtype=np.int64),
        "release_s": np.asarray(release, dtype=np.float64),
        "start_delay_s": np.asarray(delay, dtype=np.float64),
        "end_s": np.asarray(end, dtype=np.float64),
        "exec_s": np.asarray(execution, dtype=np.float64),
        "response_s": np.asarray(response, dtype=np.float64),
        "missed": np.asarray(missed, dtype=np.int64),
        "dropped_before": np.asarray(dropped, dtype=np.int64),
    }


def _write_jobs(path: Path, columns: dict[str, np.ndarray]) -> None:
    lines = [",".join(JOB_COLUMNS)]
    for row in range(columns["job_index"].size):
        lines.append(
            f"{int(columns['job_index'][row])},"
            f"{columns['release_s'][row]:.6f},"
            f"{columns['start_delay_s'][row]:.9f},"
            f"{columns['end_s'][row]:.6f},"
            f"{columns['exec_s'][row]:.9f},"
            f"{columns['response_s'][row]:.9f},"
            f"{int(columns['missed'][row])},"
            f"{int(columns['dropped_before'][row])}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_monitor(
    path: Path, duration_s: float, n_cpu: int, n_mem: int, rng: np.random.Generator
) -> None:
    """Write a 1 Hz monitor file whose descriptors move with the worker count.

    The relationships are declared synthetic ones. Foreign occupancy is a small
    non-zero baseline plus noise, because the measured campaign shares its board
    and a synthetic fixture that asserted zero foreign load would be testing a
    situation the product never sees.
    """
    samples = int(duration_s)
    lines = [",".join(MONITOR_COLUMNS)]
    workers = n_cpu + n_mem
    for sample in range(samples):
        foreign = max(0.0, 1.4 + 0.55 * float(rng.normal()))
        task = 92.0 / (1.0 + 0.1 * workers) / N_CORES
        stress = min(99.0, 15.5 * workers + 0.8 * float(rng.normal()))
        clk_mean = 1728000.0 - 9000.0 * workers - 1500.0 * float(rng.random())
        clk_min = clk_mean - 12000.0 - 4000.0 * float(rng.random())
        tj = 41000.0 + 950.0 * workers + 160.0 * float(rng.normal())
        load1 = 0.35 + 0.97 * workers + 0.05 * float(rng.normal())
        clocks = [int(clk_mean) for _ in range(N_CORES)]
        lines.append(
            f"{float(sample) + 1.0:.3f},"
            f"{int(600 * (1 + workers))},{92},{int(540 * workers)},"
            f"{int(8 + 2 * workers)},"
            f"{foreign:.4f},{task:.4f},{stress:.4f},"
            + ",".join(str(value) for value in clocks)
            + f",{clk_mean:.1f},{int(clk_min)},{int(tj)},{load1:.2f}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_synthetic_campaign(
    directory: Path | str,
    plan: tuple[RunSpec, ...] = DEFAULT_PLAN,
    seed: int = SEED,
    period_s: float = PERIOD_S,
    deadline_s: float = DEADLINE_S,
    complete: bool = True,
) -> Path:
    """Write a whole synthetic campaign into ``directory``. Returns the directory.

    The manifest has the same keys the measured campaign writes, including the
    probe table and the declared ground-truth-level rule, because the loader and
    the CLI read them and a fixture that omitted them would not exercise that.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    runs: list[dict] = []
    for spec in plan:
        n_draws = int(spec.duration_s / period_s) + 1
        draws = execution_times(
            n_draws,
            spec.n_cpu,
            spec.n_mem,
            rng,
            tail_probability=spec.tail_probability,
            tail_alpha=spec.tail_alpha,
            idle_fraction=spec.idle_fraction,
            deadline_s=deadline_s,
            distribution=spec.distribution,
            spread=spec.spread,
        )
        columns = simulate_loop(draws, period_s, deadline_s)
        keep = columns["release_s"] <= spec.duration_s
        columns = {name: values[keep] for name, values in columns.items()}
        _write_jobs(root / f"jobs_{spec.run}.csv", columns)
        _write_monitor(
            root / f"mon_{spec.run}.csv", spec.duration_s, spec.n_cpu, spec.n_mem, rng
        )
        releases = int((1 + columns["dropped_before"]).sum())
        misses = int((columns["missed"] + columns["dropped_before"]).sum())
        runs.append(
            {
                "run": spec.run,
                "n_cpu_stress": spec.n_cpu,
                "n_mem_stress": spec.n_mem,
                "duration_s": spec.duration_s,
                "payload_w": 0,
                "executed_jobs": int(columns["job_index"].size),
                "releases": releases,
                "misses": misses,
                "miss_rate": misses / releases if releases else -1,
                "median_exec_s": float(np.median(columns["exec_s"])),
            }
        )
    manifest = {
        "campaign": "P092 DeadlineRisk -- SYNTHETIC FIXTURE, not a measurement",
        "env": {
            "model": "synthetic fixture; no board was measured",
            "uname": "synthetic",
            "nproc": N_CORES,
            "governor": "synthetic",
            "clk_max_khz": "1728000",
            "sudo_n_true": 1,
            "python": "synthetic",
        },
        "period_s": period_s,
        "deadline_s": deadline_s,
        "payload_w": 0,
        "target_frac_of_deadline": IDLE_FRACTION,
        "calibration": [
            {"attempt": 0, "w": 0, "median_exec_s": IDLE_FRACTION * deadline_s,
             "frac_of_deadline": IDLE_FRACTION}
        ],
        "probes": [
            {"n_cpu_stress": level, "releases": 1500, "misses": level * 3,
             "miss_rate": level * 3 / 1500}
            for level in range(7)
        ],
        "gt_level": 4,
        "gt_reason": "synthetic fixture: the level is declared, not chosen from probes",
        "gt_rate_band": [0.0005, 0.25],
        "gt_target_rate": 0.02,
        "runs": runs,
        "complete": complete,
        "seed": seed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if complete:
        (root / "CAMPAIGN_COMPLETE").write_text("")
    return root


def load_synthetic_campaign(
    directory: Path | str,
    plan: tuple[RunSpec, ...] = DEFAULT_PLAN,
    seed: int = SEED,
    **kwargs,
) -> Campaign:
    """Write and then load a synthetic campaign, tagged :attr:`Origin.SYNTHETIC`."""
    root = write_synthetic_campaign(directory, plan=plan, seed=seed, **kwargs)
    return load_campaign(root, origin=Origin.SYNTHETIC, directory_label="synthetic")
