"""Command-line interface: ``python -m bitflipsim <subcommand>``.

Subcommands
-----------
``layout``       discovered IEEE 754 / two's-complement layout and its re-derivation
``flip``         predicted versus actual effect of flipping one bit
``flux``         upset rate from flux, cross-section and bit count
``campaign``     Poisson-sized upset campaign with its standard error
``criticality``  the two baselines and the learned predictor, head to head
``mitigate``     clamping bound, triplication and reload costs
``onnx``         export the reference model as ONNX and list its initializers

Every number printed comes from the library functions, not from a literal in
this file.
"""

from __future__ import annotations

import argparse
import functools
import sys

import numpy as np

from . import __version__
from .bitlayout import (
    exponent_field,
    flip_bit,
    float_layout,
    int_layout,
    predict_flip,
    to_bits,
    verify_float_layout,
)
from .campaign import parameter_campaign
from .criticality import (
    evaluate_protection,
    exponent_bit_baseline_scores,
    magnitude_baseline_scores,
    oracle_scores,
    random_scores,
    sweep_bit_criticality,
)
from .datasets import make_problem, reference_parameters
from .flux import poisson_validity, upset_rate
from .mitigation import (
    clamp_logit_bound,
    clamping_cost,
    measure_clamp_latency,
    reload_cost,
    triplication_cost,
)
from .network import quantize_int8
from .predictor import (
    CriticalityPredictor,
    build_features,
    split_parameters,
    uncertainty_calibration,
)

FLOAT_DTYPES = ("float16", "float32", "float64")
INT_DTYPES = ("int8", "int16", "int32")


@functools.lru_cache(maxsize=1)
def _reference() -> tuple[object, object]:
    """Problem and trained parameters, built once per process.

    Fitting the reference model costs about two seconds on one core, so the
    subcommands share one instance. A CLI process runs one subcommand, so this
    only matters when several subcommands are driven from the same interpreter.
    """
    problem = make_problem()
    return problem, reference_parameters(problem)


def _cmd_layout(args: argparse.Namespace) -> int:
    if args.dtype in FLOAT_DTYPES:
        layout = float_layout(args.dtype)
        print(f"dtype                 {layout.dtype_name}")
        print(f"storage bits          {layout.total_bits}")
        print(f"sign bit              {layout.sign_bit}")
        print(f"exponent field        bits {layout.exponent_lsb}..{layout.exponent_msb} "
              f"({layout.exponent_bits} bits)")
        print(f"mantissa field        bits 0..{layout.mantissa_bits - 1} "
              f"({layout.mantissa_bits} bits)")
        print(f"exponent bias         {layout.exponent_bias}")
        print(f"reserved exponent     {layout.max_biased_exponent} (inf / NaN)")
        print()
        report = verify_float_layout(args.dtype)
        print("independent re-derivation from probe encodings:")
        for key in ("mantissa_bits", "exponent_bits", "exponent_bias", "sign_bit", "total_bits"):
            declared, probed, agree = report[key]  # type: ignore[misc]
            print(f"  {key:<16} finfo={declared:<6} probe={probed:<6} agree={agree}")
        print(f"  overall agree    {report['agree']}")
        return 0 if report["agree"] else 1
    layout = int_layout(args.dtype)
    print(f"dtype                 {layout.dtype_name}")
    print(f"storage bits          {layout.total_bits}")
    print(f"sign bit              {layout.sign_bit} (two's complement)")
    print(f"value bits            bits 0..{layout.total_bits - 2}")
    return 0


def _cmd_flip(args: argparse.Namespace) -> int:
    dtype = args.dtype
    if dtype in FLOAT_DTYPES:
        value = np.asarray(args.value, dtype=dtype)[()]
        prediction = predict_flip(value, args.bit, dtype)
        actual = flip_bit(value, args.bit, dtype)
        layout = float_layout(dtype)
        print(f"value                 {float(value)!r}  ({dtype})")
        print(f"bit pattern           0x{to_bits(value, dtype):0{layout.total_bits // 4}x}")
        print(f"biased exponent       {exponent_field(value, dtype)}")
        print(f"bit {args.bit:<17} role {layout.role(args.bit)}")
        print(f"predicted value       {prediction.predicted_value!r}")
        print(f"actual value          {float(actual)!r}")
        print(f"regime                {prediction.regime}")
        print(f"predicted ratio       {prediction.predicted_ratio!r}")
        print(f"predicted delta       {prediction.predicted_delta!r}")
        agree = (
            np.isnan(prediction.predicted_value) and np.isnan(float(actual))
        ) or float(actual) == prediction.predicted_value
        print(f"prediction matches    {agree}")
        return 0 if agree else 1
    layout = int_layout(dtype)
    value = int(np.asarray(int(args.value), dtype=dtype)[()])
    actual = int(flip_bit(value, args.bit, dtype))
    delta = layout.flip_delta(value, args.bit)
    print(f"value                 {value}  ({dtype})")
    print(f"bit {args.bit:<17} role {layout.role(args.bit)}")
    print(f"predicted delta       {delta:+d}")
    print(f"actual value          {actual}")
    print(f"prediction matches    {actual - value == delta}")
    return 0 if actual - value == delta else 1


def _cmd_flux(args: argparse.Namespace) -> int:
    rate = upset_rate(args.flux, args.cross_section, args.bits)
    print(f"flux                  {rate.flux_per_cm2_s:.6e} particles cm^-2 s^-1")
    print(f"cross-section         {rate.cross_section_cm2_per_bit:.6e} cm^2 bit^-1")
    print(f"exposed bits          {rate.bit_count}")
    print(f"upset rate            {rate.rate_per_s:.6e} upsets s^-1")
    print(f"upset rate            {rate.rate_fit:.6e} FIT")
    print(f"mean time between     {rate.mean_time_between_upsets_s:.6e} s")
    if args.exposure is not None:
        validity = poisson_validity(rate, args.exposure)
        print(f"exposure              {args.exposure:.6e} s")
        print(f"expected upsets       {validity['expected_upsets']:.6e}")
        print(f"per-bit probability   {validity['per_bit_probability']:.6e}")
        print(f"Poisson validity      {validity['within_validity']} "
              f"(threshold {validity['threshold']:.1e})")
    return 0


def _cmd_campaign(args: argparse.Namespace) -> int:
    problem, params = _reference()
    rng = np.random.default_rng(args.seed)
    limit = float(np.abs(params.values).max()) if args.clamp else None
    result = parameter_campaign(
        params,
        problem.evaluation.x,
        problem.evaluation.y,
        args.expected,
        args.trials,
        rng,
        clamp_limit=limit,
    )
    print(f"expected upsets       {result.expected_upsets:.4f}")
    print(f"trials                {result.trials}")
    print(f"drawn counts          mean {result.upset_counts.mean():.4f} "
          f"var {result.upset_counts.var(ddof=1):.4f}")
    print(f"clamp limit           {'none' if limit is None else f'{limit:.6f}'}")
    print(f"golden accuracy       {result.golden_accuracy:.6f}")
    print(f"mean degradation      {result.mean_degradation:.6f} "
          f"+- {result.degradation_standard_error:.6f} (standard error)")
    print(f"mean accuracy         {result.mean_accuracy:.6f} "
          f"+- {result.accuracy_standard_error:.6f}")
    print(f"mean accuracy drop    {result.mean_accuracy_drop:.6f}")
    return 0


def _cmd_criticality(args: argparse.Namespace) -> int:
    problem, params = _reference()
    layout = float_layout("float32")
    sweep_cal = sweep_bit_criticality(params, problem.calibration.x, problem.calibration.y)
    sweep_eval = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
    features = build_features(params, problem.calibration.x, layout)
    split = split_parameters(params.layout.size, layout.total_bits, seed=args.seed)
    predictor = CriticalityPredictor(seed=args.seed).fit(
        features[split.site_mask("train")],
        sweep_cal.flat_degradation()[split.site_mask("train")],
    )
    prediction = predictor.predict(features[split.site_mask("test")])
    n_test = split.test_parameters.size
    degradation = sweep_eval.degradation[split.test_parameters]
    methods = {
        "magnitude_baseline": magnitude_baseline_scores(params)[split.test_parameters],
        "exponent_heuristic": exponent_bit_baseline_scores(params.layout.size, layout)[
            split.test_parameters
        ],
        "learned_predictor": prediction.expected.reshape(n_test, layout.total_bits),
        "oracle": oracle_scores(degradation),
        "random": random_scores(degradation.shape, np.random.default_rng(args.seed)),
    }
    budget = n_test * params.layout.itemsize_bytes * args.budget_fraction
    print(f"test parameters       {n_test} of {params.layout.size} (grouped split, "
          f"seed {split.seed})")
    print(f"cost model            {args.cost_model}")
    print(f"budget                {budget:.2f} bytes "
          f"({args.budget_fraction:.3f} of the test region)")
    print(f"{'method':<20} {'avoided':>12} {'per byte':>12} {'fraction':>10}")
    for name, scores in methods.items():
        result = evaluate_protection(
            scores, degradation, budget, params.layout.itemsize_bytes, args.cost_model, name
        )
        print(f"{name:<20} {result.avoided:12.6f} {result.avoided_per_byte:12.6f} "
              f"{result.avoided_fraction:10.6f}")
    calibration = uncertainty_calibration(
        prediction, sweep_eval.flat_degradation()[split.site_mask("test")]
    )
    print()
    print(f"uncertainty coverage at k=2  {calibration['coverage']:.6f} "
          f"(Gaussian reference {calibration['gaussian_reference']:.6f})")
    print(f"mean abs error (log1p)       {calibration['mean_abs_error_log1p']:.6e}")
    print("top features:")
    for name, importance in predictor.importance_table()[:5]:
        print(f"  {name:<32} {importance:.6f}")
    return 0


def _cmd_mitigate(args: argparse.Namespace) -> int:
    problem, params = _reference()
    limit = float(np.abs(params.values).max())
    max_input = float(np.abs(problem.evaluation.x).max())
    bound = clamp_logit_bound(limit, params.layout.n_in, max_input)
    latency = measure_clamp_latency(params.layout.size)
    inference = _time_inference(params, problem.evaluation.x)
    print(f"parameters            {params.layout.size} ({params.layout.total_bytes} bytes)")
    print(f"clamp limit C         {limit:.6f} (= max |w|, so the golden output is unchanged)")
    print(f"max |x|               {max_input:.6f}")
    print(f"single-upset bound    {bound:.6f} on max_k |d logits_k|")
    print()
    costs = [
        clamping_cost(params.layout.total_bytes, latency, inference),
        triplication_cost(params.layout.total_bytes, inference, latency["median_s"]),
        reload_cost(params.layout.total_bytes, args.scrub_interval, args.reload_time, 1e-3)[0],
    ]
    print(f"{'scheme':<26} {'extra bytes':>12} {'mem factor':>11} {'latency frac':>13}")
    for cost in costs:
        print(f"{cost.scheme:<26} {cost.extra_memory_bytes:12d} {cost.memory_factor:11.3f} "
              f"{cost.latency_overhead_fraction:13.6f}")
    print()
    quantized = quantize_int8(params)
    print(f"int8 quantized block  {quantized.codes.nbytes} bytes "
          f"({params.layout.total_bytes / quantized.codes.nbytes:.2f}x smaller)")
    return 0


def _time_inference(params: object, x: np.ndarray, repeats: int = 50) -> float:
    import time

    samples = np.empty(repeats, dtype=np.float64)
    for r in range(repeats):
        t0 = time.perf_counter()
        params.probabilities(x)  # type: ignore[attr-defined]
        samples[r] = time.perf_counter() - t0
    return float(np.median(samples))


def _cmd_onnx(args: argparse.Namespace) -> int:
    from .onnx_io import build_mlp_onnx, list_initializers

    problem, params = _reference()
    blob = build_mlp_onnx(params)
    initializers = list_initializers(blob)
    if args.out:
        with open(args.out, "wb") as handle:
            handle.write(blob)
        print(f"wrote {args.out} ({len(blob)} bytes)")
    else:
        print(f"serialised model      {len(blob)} bytes")
    print(f"{'initializer':<14} {'dims':<12} {'elements':>9} {'raw offset':>11} {'bytes':>7}")
    for init in initializers:
        print(f"{init.name:<14} {str(init.dims):<12} {init.n_elements:9d} "
              f"{init.raw_offset:11d} {init.raw_length:7d}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bitflipsim",
        description=(
            "Single-event-upset effects on small scikit-learn / ONNX inference. "
            "Research-grade; not flight-qualified, not certified, not approved "
            "for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"bitflipsim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_layout = sub.add_parser("layout", help="discovered storage layout and its re-derivation")
    p_layout.add_argument("--dtype", default="float32", choices=FLOAT_DTYPES + INT_DTYPES)
    p_layout.set_defaults(func=_cmd_layout)

    p_flip = sub.add_parser("flip", help="predicted versus actual single-bit flip")
    p_flip.add_argument("--value", type=float, required=True)
    p_flip.add_argument("--bit", type=int, required=True)
    p_flip.add_argument("--dtype", default="float32", choices=FLOAT_DTYPES + INT_DTYPES)
    p_flip.set_defaults(func=_cmd_flip)

    p_flux = sub.add_parser("flux", help="upset rate from flux, cross-section and bit count")
    p_flux.add_argument("--flux", type=float, required=True,
                        help="particles cm^-2 s^-1")
    p_flux.add_argument("--cross-section", type=float, required=True,
                        help="cm^2 bit^-1")
    p_flux.add_argument("--bits", type=int, required=True, help="exposed bits")
    p_flux.add_argument("--exposure", type=float, default=None, help="seconds")
    p_flux.set_defaults(func=_cmd_flux)

    p_campaign = sub.add_parser("campaign", help="Poisson-sized upset campaign")
    p_campaign.add_argument("--expected", type=float, default=4.0,
                            help="Poisson mean upset count")
    p_campaign.add_argument("--trials", type=int, default=200)
    p_campaign.add_argument("--clamp", action="store_true",
                            help="read parameters back through a clamp at max |w|")
    p_campaign.add_argument("--seed", type=int, default=0)
    p_campaign.set_defaults(func=_cmd_campaign)

    p_crit = sub.add_parser("criticality", help="baselines versus the learned predictor")
    p_crit.add_argument("--budget-fraction", type=float, default=0.05,
                        help="protection budget as a fraction of the test region")
    p_crit.add_argument("--cost-model", default="bit", choices=("bit", "word"))
    p_crit.add_argument("--seed", type=int, default=7)
    p_crit.set_defaults(func=_cmd_criticality)

    p_mit = sub.add_parser("mitigate", help="clamp bound and mitigation costs")
    p_mit.add_argument("--scrub-interval", type=float, default=60.0, help="seconds")
    p_mit.add_argument("--reload-time", type=float, default=0.01, help="seconds")
    p_mit.set_defaults(func=_cmd_mitigate)

    p_onnx = sub.add_parser("onnx", help="export the reference model as ONNX")
    p_onnx.add_argument("--out", default=None, help="path to write the .onnx file")
    p_onnx.set_defaults(func=_cmd_onnx)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
