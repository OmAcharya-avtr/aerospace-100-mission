"""Generator of schema-identical SYNTHETIC traces, for unit tests and examples.

Nothing this module writes is a measurement. Every file it produces is named
``SYNTHETIC_<label>.csv`` and the manifest it writes is
``SYNTHETIC_manifest.json`` with a ``model`` field that says in the file that it
is not a board, so that
:meth:`thermalbudget.trace.Campaign.board_matches_declared_target` is False for
anything loaded from here. The loader requires an explicit
:class:`~thermalbudget.provenance.Origin` argument and
:func:`load_synthetic_campaign` passes ``SYNTHETIC``; there is no path in this
repository that loads these files as measured.

Why a generator at all. The measured traces arrive from a campaign running on
the board and are not available while the package is being written. Unit tests
must not wait for them and must not be written against them, because a unit test
that needs a 12-minute soak is not a unit test. So the generator emits the full
declared run plan with a plausible first-order thermal response, and the
measured traces are used only by the validation scripts, which refuse to run
without them.

**Two regimes, and neither is a default.** ``regime`` is a required argument.
``"no_throttle"`` holds the CPU clock at its maximum for the whole run;
``"throttled"`` rolls the clock off above a declared knee. Both exist because
the product must not presuppose that throttling happens: the code paths for a
degenerate clock-versus-temperature relation and for an identifiable one are
both exercised by tests, and no test asserts that a gap exists on the real
board.

The measured campaign turned out to be in the ``"no_throttle"`` regime -- die
rise 10.94 degC at duty 1.0, clock constant at 1728000 kHz throughout -- but
that was not known when these fixtures were written, and the ``"throttled"``
regime is kept because a product that only exercised the regime its one board
happened to be in would be untested for the other.

The thermal model used to synthesise is the same first-order form baseline 2
fits, which makes these fixtures unsuitable as evidence for baseline 2 and
suitable only for checking that the estimator recovers parameters it was given.
Every test that uses them says so.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Prefix on every file written here.
PREFIX = "SYNTHETIC_"

#: Default seed. Fixed so regeneration is deterministic.
SEED = 910091

#: Declared regimes. ``regime`` is required at every entry point.
REGIMES = ("no_throttle", "throttled")

#: Clock ceiling of the measured board, kHz (``cpuinfo_max_freq``).
MAX_KHZ = 1728000.0

#: GPU devfreq frequency held by the measured cold-burst trace, Hz.
GPU_HZ = 306000000.0

#: Idle die temperature of the synthetic board, degrees Celsius.
IDLE_C = 45.0

#: Degrees of steady-state rise per unit duty cycle, synthetic.
RISE_PER_DUTY_C = 23.0

#: Rise time constant, seconds, synthetic. Chosen so that every soak in the
#: declared plan reaches the default equilibrium criterion and the NOT_EQUILIBRATED
#: branch has to be exercised by a deliberately short run rather than by a
#: borderline one, which would make the test a coin toss on the noise draw.
TAU_RISE_S = 90.0

#: Cooldown time constant, seconds, synthetic. Deliberately NOT equal to
#: TAU_RISE_S: the two are estimated independently by baseline 2 and reported
#: side by side, and a fixture in which they agree exactly would hide a defect
#: in whichever estimator was wrong.
TAU_COOL_S = 185.0

#: Knee of the synthetic clock roll-off, degrees Celsius, used only in the
#: ``"throttled"`` regime.
THROTTLE_KNEE_C = 58.0

#: kHz lost per degree above the knee, ``"throttled"`` regime only.
THROTTLE_SLOPE_KHZ_PER_C = 42000.0

#: Iterations per second per worker at the clock ceiling and duty 1.0, synthetic.
ITERS_PER_S_AT_MAX = 29000.0

#: The declared run plan, label for label and second for second as
#: ``data/campaign.py`` writes it. Generating all seventeen runs is string
#: formatting over about 4800 rows and takes well under a second, so there was no
#: reason to shorten it and every reason not to: a fixture whose runs are shorter
#: than the measured ones would exercise different branches of the equilibrium
#: test from the ones the measured data will take.
PLAN: tuple[tuple[str, float, int], ...] = (
    ("idle_pre", 0.0, 120),
    ("burst_1.00_a", 1.0, 20),
    ("cool_a", 0.0, 240),
    ("burst_1.00_b", 1.0, 20),
    ("cool_b", 0.0, 240),
    ("soak_1.00_r1", 1.0, 720),
    ("cool_r1", 0.0, 300),
    ("soak_0.50", 0.5, 480),
    ("cool_d50", 0.0, 240),
    ("soak_0.75", 0.75, 480),
    ("cool_d75", 0.0, 240),
    ("soak_0.25", 0.25, 420),
    ("cool_d25", 0.0, 180),
    ("soak_1.00_r2", 1.0, 480),
    ("cool_r2", 0.0, 300),
    ("soak_1.00_r3", 1.0, 480),
    ("cool_r3", 0.0, 180),
)

ZONES = ("cpu-thermal", "gpu-thermal", "soc0-thermal", "soc1-thermal", "soc2-thermal",
         "tj-thermal")
N_CORES = 6

#: Per-zone offset from the die channel, degrees Celsius, so the six zones are
#: distinguishable and the die channel is not merely duplicated.
ZONE_OFFSET_C = {
    "cpu-thermal": 0.0,
    "gpu-thermal": 0.12,
    "soc0-thermal": -0.06,
    "soc1-thermal": -0.75,
    "soc2-thermal": -1.34,
    "tj-thermal": 0.0,
}


@dataclass(frozen=True)
class SyntheticParameters:
    """The parameters a fixture was generated with, so a test can compare.

    A test that checks an estimator recovers ``tau_rise_s`` reads it from here
    rather than restating the literal, which keeps the test honest when the
    fixture changes.
    """

    regime: str
    seed: int
    idle_c: float = IDLE_C
    rise_per_duty_c: float = RISE_PER_DUTY_C
    tau_rise_s: float = TAU_RISE_S
    tau_cool_s: float = TAU_COOL_S
    throttle_knee_c: float = THROTTLE_KNEE_C
    throttle_slope_khz_per_c: float = THROTTLE_SLOPE_KHZ_PER_C
    max_khz: float = MAX_KHZ
    iters_per_s_at_max: float = ITERS_PER_S_AT_MAX

    def steady_state_c(self, duty: float) -> float:
        """Synthetic steady-state die temperature at a commanded duty, deg C."""
        return self.idle_c + self.rise_per_duty_c * duty

    def clock_khz_at(self, temperature_c: float) -> float:
        """Synthetic clock at a die temperature, kHz, under this regime."""
        if self.regime == "no_throttle":
            return self.max_khz
        excess = max(0.0, temperature_c - self.throttle_knee_c)
        return max(0.3 * self.max_khz, self.max_khz - self.throttle_slope_khz_per_c * excess)


def label_offset(label: str) -> int:
    """A stable per-run offset derived from the run label.

    ``hash()`` is NOT used, and the reason is a defect this generator actually
    shipped for part of its build: CPython salts the hash of a string per
    process unless ``PYTHONHASHSEED`` is fixed, so a generator seeded with
    ``hash(label)`` produces different traces in two different interpreters
    while looking perfectly deterministic inside one. A test that writes two
    campaigns in the same process cannot see it; two runs of the same validation
    script, minutes apart, printed different headline numbers. ``zlib.crc32`` is
    a fixed function of the bytes and has no such behaviour.
    """
    return zlib.crc32(label.encode("utf-8")) % 10_000


def _check_regime(regime: str) -> str:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    return regime


def synthesise_run(
    label: str,
    duty: float,
    seconds: int,
    *,
    regime: str,
    seed: int = SEED,
    start_c: float | None = None,
    inject_missing: Sequence[int] = (),
) -> tuple[tuple[str, ...], list[list[str]], SyntheticParameters]:
    """Rows of one synthetic run, as strings exactly as they appear in a CSV.

    Parameters
    ----------
    label, duty, seconds:
        Run identity, commanded duty in [0, 1], and length in seconds.
    regime:
        ``"no_throttle"`` or ``"throttled"``. Required.
    seed:
        Seed for the per-run noise stream; combined with a hash of the label so
        two runs of the same duty are not identical.
    start_c:
        Initial die temperature. Defaults to the synthetic idle floor for a
        duty-0 run and to the idle floor for a loaded run, which is what a cold
        start is.
    inject_missing:
        Row indices at which every sentinel-coded field is written as ``-1``,
        so the loader's missing-data path is exercised by the fixture rather
        than only by hand-built rows.

    Returns
    -------
    (header, rows, parameters)
    """
    _check_regime(regime)
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"duty must lie in [0, 1], got {duty}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    parameters = SyntheticParameters(regime=regime, seed=seed)
    rng = np.random.default_rng(seed + label_offset(label))

    header = (
        "t_wall", "t_rel", "iters_cum", "iters_per_s", "achieved_duty", "loadavg1",
        "foreign_cpu_pct",
        *(f"temp_{zone}" for zone in ZONES),
        *(f"cpu{index}_khz" for index in range(N_CORES)),
        "gpu_hz",
    )

    t0_wall = 1794457188.963
    initial = parameters.idle_c if start_c is None else start_c
    if duty > 0.0:
        target = parameters.steady_state_c(duty)
        tau = parameters.tau_rise_s
    else:
        target = parameters.idle_c
        tau = parameters.tau_cool_s

    rows: list[list[str]] = []
    iters_cum = 0
    for index in range(seconds):
        t_rel = index + 1.0 + float(rng.normal(0.0, 0.004))
        die = target - (target - initial) * math.exp(-t_rel / tau)
        die += float(rng.normal(0.0, 0.035))
        clock = parameters.clock_khz_at(die)
        if duty > 0.0:
            per_s = parameters.iters_per_s_at_max * duty * (clock / parameters.max_khz)
            per_s *= 1.0 + float(rng.normal(0.0, 0.035))
            per_s = max(per_s, 0.0)
        else:
            per_s = 0.0
        iters_cum += int(round(per_s))
        achieved = 0.0 if duty == 0.0 else max(0.0, duty * (1.0 + float(rng.normal(0.0, 0.05))))
        loadavg = 0.30 + 5.6 * duty * (1.0 - math.exp(-t_rel / 45.0))
        foreign = 20.1 + float(rng.normal(0.0, 0.4))

        temps_milli = [
            int(round(1000.0 * (die + ZONE_OFFSET_C[zone] + float(rng.normal(0.0, 0.02)))))
            for zone in ZONES
        ]
        clocks = [int(round(clock)) for _ in range(N_CORES)]
        gpu = int(GPU_HZ)
        if index in set(inject_missing):
            temps_milli = [-1] * len(temps_milli)
            clocks = [-1] * len(clocks)
            gpu = -1
            foreign = -1.0

        row = [
            f"{t0_wall + t_rel:.3f}",
            f"{t_rel:.3f}",
            str(iters_cum),
            f"{per_s:.2f}",
            f"{achieved:.4f}",
            f"{loadavg:.2f}",
            f"{foreign:.2f}",
            *[str(value) for value in temps_milli],
            *[str(value) for value in clocks],
            str(gpu),
        ]
        rows.append(row)
    return header, rows, parameters


def write_run(
    directory: str | Path,
    label: str,
    duty: float,
    seconds: int,
    *,
    regime: str,
    seed: int = SEED,
    start_c: float | None = None,
    inject_missing: Sequence[int] = (),
) -> Path:
    """Write one run to ``directory/SYNTHETIC_<label>.csv`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    header, rows, _ = synthesise_run(
        label, duty, seconds, regime=regime, seed=seed, start_c=start_c,
        inject_missing=inject_missing,
    )
    path = directory / f"{PREFIX}{label}.csv"
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_campaign(
    directory: str | Path,
    *,
    regime: str,
    seed: int = SEED,
    plan: Sequence[tuple[str, float, int]] = PLAN,
    inject_missing_in: Sequence[str] = ("soak_0.50",),
) -> Path:
    """Write the whole synthetic run plan plus ``SYNTHETIC_manifest.json``.

    The manifest's ``model`` field states in the file that this is not a board.
    ``inject_missing_in`` names runs that get three sentinel-coded rows, so a
    campaign-wide check meets missing data somewhere rather than nowhere.

    Returns
    -------
    Path
        The directory written to.
    """
    _check_regime(regime)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    parameters = SyntheticParameters(regime=regime, seed=seed)
    # Thread the die temperature across runs, so a cooldown starts from where the
    # run before it ended and actually decays. A cooldown trace that starts at the
    # idle floor carries no decay to estimate tau from, and the independent
    # cooldown estimate of tau is one of the things these fixtures exist to
    # exercise.
    current_c = parameters.idle_c
    for label, duty, seconds in plan:
        missing = (5, 6, 7) if label in set(inject_missing_in) else ()
        path = write_run(
            directory, label, duty, seconds, regime=regime, seed=seed,
            start_c=current_c, inject_missing=missing,
        )
        target = parameters.steady_state_c(duty) if duty > 0.0 else parameters.idle_c
        tau = parameters.tau_rise_s if duty > 0.0 else parameters.tau_cool_s
        current_c = target - (target - current_c) * math.exp(-seconds / tau)
        runs.append(
            {
                "label": label,
                "duty": duty,
                "seconds": seconds,
                "csv": path.name,
                "synthetic": True,
                "start_die_c": round(current_c, 4),
            }
        )
    manifest = {
        "model": "SYNTHETIC fixture written by tests/fixtures/synthetic_trace.py; "
                 "not a board and not a measurement",
        "synthetic": True,
        "regime": regime,
        "seed": seed,
        "generator": "tests/fixtures/synthetic_trace.py",
        "zones_readable": list(ZONES),
        "zones_present_but_unreadable": ["cv0-thermal", "cv1-thermal", "cv2-thermal"],
        "nworkers": N_CORES,
        "cpuinfo_max_freq_khz": int(MAX_KHZ),
        "governor": "schedutil",
        "window_s": 0.05,
        "sample_s": 1.0,
        "runs": runs,
    }
    (directory / f"{PREFIX}manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_synthetic_campaign(directory: str | Path, *, regime: str, seed: int = SEED):
    """Write the synthetic plan into ``directory`` and load it as SYNTHETIC.

    The import is local so that this module can be used by a test that only
    wants raw rows without the package on ``sys.path``.
    """
    from thermalbudget.provenance import Origin
    from thermalbudget.trace import load_campaign

    write_campaign(directory, regime=regime, seed=seed)
    directory = Path(directory)
    manifest = directory / f"{PREFIX}manifest.json"
    target = directory / "manifest.json"
    if not target.exists():
        target.write_text(manifest.read_text())
    return load_campaign(directory, origin=Origin.SYNTHETIC)


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate a synthetic campaign into a named directory."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="synthetic_trace",
        description="Write a SYNTHETIC, schema-identical campaign. Not a measurement.",
    )
    parser.add_argument("directory", help="directory to write SYNTHETIC_*.csv into")
    parser.add_argument("--regime", required=True, choices=REGIMES)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args(argv)
    written = write_campaign(arguments.directory, regime=arguments.regime, seed=arguments.seed)
    print(f"wrote {len(PLAN)} SYNTHETIC runs into {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
