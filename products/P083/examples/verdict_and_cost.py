"""What the whole pipeline returns over a population, and what each policy costs.

What to notice: the certificate's cost does not grow with the number of misses
in a run while re-simulation's does, so the saving widens exactly where it is
needed. The verdict bar chart is the product's actual output: one word per
run, with the undecidable cases separated out rather than guessed at.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import time  # noqa: E402

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    CONSEQUENTIAL,
    NEUTRAL,
    REFERENCE,
    UNDECIDED,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.certificate import CertificateModel, certify_run  # noqa: E402
from hilverdict.plant import LoopModel  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, ClockSpec, generate_run  # noqa: E402
from hilverdict.verdict import (  # noqa: E402
    NOT_TIMING_FAITHFUL,
    TIMING_FAITHFUL,
    UNDECIDABLE,
    run_verdict,
)

STEPS = 200
SEEDS = tuple(range(8500, 8518))


def main() -> int:
    style()
    model = LoopModel()
    verdicts: list[str] = []
    misses: list[int] = []
    full_calls: list[int] = []
    residual_calls: list[int] = []
    certificate_ms: list[float] = []
    reference_ms: list[float] = []
    for position, seed in enumerate(SEEDS):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
        )
        started = time.perf_counter()
        CertificateModel.build(model, STEPS)
        certificate = certify_run(log, 1.0)
        certificate_ms.append((time.perf_counter() - started) * 1e3)
        started = time.perf_counter()
        everything = resimulate_run(log, 1.0)
        reference_ms.append((time.perf_counter() - started) * 1e3)
        subset = tuple(item.event.index for item in certificate.undecided)
        partial = resimulate_run(log, 1.0, subset=subset)
        verdicts.append(run_verdict(log, 1.0).verdict)
        misses.append(len(certificate.certificates))
        full_calls.append(sum(result.plant_calls for result in everything))
        residual_calls.append(sum(result.plant_calls for result in partial))

    # Two deliberately inadmissible runs, so the undecidable column is not empty
    # by accident: a saturating manoeuvre and a device-clock reset.
    saturating = generate_run(seed=SEEDS[0], profile="loaded", amplitude_deg=14.0,
                              steps=STEPS)
    reset_spec = ClockSpec(drift_ppm=0.0, jitter_s=2.0e-7, tick_s=1.0e-7,
                           reset_fractions=(0.45,), name="reset")
    reset_run = generate_run(seed=SEEDS[1], profile="loaded", amplitude_deg=6.0,
                             steps=STEPS, clock=reset_spec)
    verdicts.append(run_verdict(saturating, 1.0).verdict)
    verdicts.append(run_verdict(reset_run, 1.0).verdict)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    order = [TIMING_FAITHFUL, NOT_TIMING_FAITHFUL, UNDECIDABLE]
    colours = [CERTIFIED, CONSEQUENTIAL, NEUTRAL]
    counts = [verdicts.count(name) for name in order]
    bars = axes[0].bar(
        [name.replace("_", "\n") for name in order], counts, color=colours
    )
    for bar, count in zip(bars, counts, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
            fontsize=9,
        )
    axes[0].set_ylabel("runs")
    axes[0].set_title(
        f"{len(SEEDS)} declared campaigns plus two deliberately inadmissible runs"
    )
    axes[0].tick_params(axis="x", labelsize=8)

    axes[1].scatter(misses, full_calls, s=28, c=REFERENCE, marker="o",
                    label="re-simulate every miss")
    axes[1].scatter(misses, residual_calls, s=28, c=CERTIFIED, marker="s",
                    label="certificate, then the residual")
    axes[1].scatter(misses, np.full(len(misses), 1.0), s=28, c=UNDECIDED,
                    marker="^", label="certificate alone (no plant evaluation)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("misses recorded in the run")
    axes[1].set_ylabel("plant right-hand-side evaluations")
    axes[1].set_title("the reference's cost grows with the misses; the proof's does not")
    axes[1].legend(loc="center right", fontsize=7.5)

    figure.suptitle(
        "hilverdict: verdicts over a population, and the cost of each policy",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "verdict_and_cost")
    print(f"written: {path}")
    for name, count in zip(order, counts, strict=True):
        print(f"{name}: {count}")
    print(f"misses total {sum(misses)}")
    print(f"plant evaluations, every miss : {sum(full_calls)}")
    print(f"plant evaluations, residual   : {sum(residual_calls)}")
    print(f"saved {1.0 - sum(residual_calls) / sum(full_calls):.6f}")
    print(f"certificate wall clock, median {float(np.median(certificate_ms)):.3f} ms; "
          f"reference, median {float(np.median(reference_ms)):.3f} ms "
          "(UNSTABLE on shared hardware)")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
