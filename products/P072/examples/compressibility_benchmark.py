"""Size estimation: the entropy floor, the instrument constant, and a learned predictor.

What to notice: the error on the compression ratio is not the quantity that
matters. The left panel shows the ratio error, where the entropy floor is
visibly biased high -- it is a bound for a memoryless source and the generator,
like a real codec, exploits correlation the first-order entropy cannot see. The
right panel shows what that costs: a window planned with an under-estimated size
overflows, the pass runs out of time, and everything queued behind the product
that did not fit is lost. The conservative arm uses the forest's own ensemble
dispersion to over-reserve, and is the only arm that both avoids every overflow
and leaves little capacity unspent.
"""

from __future__ import annotations

import numpy as np

from _common import ANALYTIC, EXACT, GREEDY, LEARNED, LEFT_BEHIND, ORACLE, bootstrap, save, style

bootstrap()

from downlinkpick.benchmark import benchmark_ratio_sources  # noqa: E402
from downlinkpick.compress import (  # noqa: E402
    CompressibilityModel,
    entropy_floor_ratios,
    global_mean_ratios,
    instrument_constant_ratios,
    instrument_constant_table,
)
from downlinkpick.dataset import generate_instance, generate_training_table  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402

SEEDS = tuple(range(6001, 6021))
SPLIT = 4200

ARM_COLOURS = {
    "learned-mean": LEARNED,
    "learned+2sigma": EXACT,
    "instrument-constant": GREEDY,
    "global-mean": LEFT_BEHIND,
    "entropy-floor": ANALYTIC,
}


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    table = generate_training_table(6000, 1002)
    model = CompressibilityModel().fit(
        table.compression_features[:SPLIT], table.compression_target[:SPLIT]
    )
    constant_table = instrument_constant_table(
        table.instrument[:SPLIT], table.compression_target[:SPLIT]
    )
    global_mean = float(table.compression_target[:SPLIT].mean())

    test_x = table.compression_features[SPLIT:]
    truth = table.compression_target[SPLIT:]
    floor = table.entropy_floor[SPLIT:]
    predicted, dispersion = model.predict_with_dispersion(test_x)
    conservative = np.clip(predicted + 2.0 * dispersion, 1.0e-3, 1.0)
    instrument = np.array([constant_table[name] for name in table.instrument[SPLIT:]])

    def learned_arm(instance, safe: bool = False):
        prediction = model.predict_inventory(instance.inventory, instance.autocorrelation)
        return prediction.conservative_mapping() if safe else prediction.as_mapping()

    arms = {
        "learned-mean": learned_arm,
        "learned+2sigma": lambda instance: learned_arm(instance, safe=True),
        "instrument-constant": lambda instance: instrument_constant_ratios(
            instance.inventory, constant_table
        ),
        "global-mean": lambda instance: global_mean_ratios(
            instance.inventory, np.array([global_mean])
        ),
        "entropy-floor": lambda instance: entropy_floor_ratios(instance.inventory),
    }
    instances = [
        generate_instance(seed, n_products=24, n_windows=3, capacity_fraction=0.45)
        for seed in SEEDS
    ]
    results = {
        result.name: result
        for result in benchmark_ratio_sources(instances, arms, selector=exact_select)
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    figure.subplots_adjust(wspace=0.3, bottom=0.17, top=0.88)

    for label, values, colour in (
        ("learned-mean", predicted, LEARNED),
        ("learned+2sigma", conservative, EXACT),
        ("instrument-constant", instrument, GREEDY),
        ("entropy-floor", floor, ANALYTIC),
    ):
        error = values - truth
        left.hist(
            error,
            bins=60,
            histtype="step",
            linewidth=1.5,
            color=colour,
            label=f"{label} (bias {error.mean():+.4f})",
        )
    left.axvline(0.0, color=ORACLE, linewidth=0.9, linestyle="--")
    left.set_xlabel("estimated ratio minus realised ratio")
    left.set_ylabel("held-out products")
    left.set_title(f"ratio error on {truth.size} held-out products")
    left.legend(fontsize=7.5)

    for name, colour in ARM_COLOURS.items():
        result = results[name]
        right.scatter(
            result.unspent_fraction,
            result.delivered_fraction,
            s=90,
            color=colour,
            edgecolor=ORACLE,
            linewidth=0.6,
            zorder=3,
        )
        right.annotate(
            f"{name}\noverflow {result.overflow_rate:.1%}",
            xy=(result.unspent_fraction, result.delivered_fraction),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=7.5,
        )
    oracle = results["oracle"]
    right.scatter(
        oracle.unspent_fraction,
        oracle.delivered_fraction,
        s=110,
        marker="*",
        color=ORACLE,
        zorder=4,
    )
    right.annotate(
        "oracle ratios",
        xy=(oracle.unspent_fraction, oracle.delivered_fraction),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.5,
    )
    right.set_xlabel("fraction of plan capacity left unspent")
    right.set_ylabel("value delivered, against the oracle")
    right.set_title(f"what the estimate costs, {len(SEEDS)} instances")
    right.set_ylim(0.6, 1.03)

    figure.text(
        0.5,
        0.01,
        "an under-estimated size is paid for in lost products; an over-estimated one in "
        "unspent capacity, and the dispersion output lets the caller choose which",
        ha="center",
        fontsize=7,
    )
    return save(figure, "compressibility_benchmark")


if __name__ == "__main__":
    print(main())
