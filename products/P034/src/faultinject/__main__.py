"""Command-line interface: ``python -m faultinject <subcommand>``.

Printing happens here and nowhere else in the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from . import search as search_mod
from .benchmark import run_benchmark
from .campaign import FaultCase, build_pool, evaluate_pool, execute_case, replay_case
from .coverage import CoverageTracker, all_cells
from .faults import Injection
from .harness import Trace
from .severity import SEVERE_THRESHOLD
from .target import N_STEPS
from .taxonomy import FaultKind, kinds, spec, total_cells


def _parse_params(items: Sequence[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        name, _, value = item.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--param {name!r}: {value!r} is not a number") from None
    return out


def cmd_taxonomy(args: argparse.Namespace) -> int:
    print(f"faultinject taxonomy: {len(kinds())} kinds, {total_cells()} coverage cells")
    print()
    for kind in kinds():
        sp = spec(kind)
        print(f"{sp.fault_class.value:>9} | {kind.value}")
        print(f"            channels: {', '.join(sp.channels)}   cells: {sp.n_cells}")
        if sp.params:
            for p in sp.params:
                flag = " (integer)" if p.integer else ""
                print(
                    f"            param {p.name}: [{p.lo:g}, {p.hi:g}] {p.unit}, "
                    f"{p.n_bins} {p.scale.value} bins{flag}"
                )
        else:
            print("            param (none)")
        print(f"            {sp.description}")
        print(f"            ref: {sp.reference}")
        print()
    return 0


def cmd_cells(args: argparse.Namespace) -> int:
    subset = [FaultKind(args.kind)] if args.kind else None
    cells = all_cells(subset)
    print(f"cells: {len(cells)}")
    if args.list:
        for c in cells:
            print(c.label())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = _parse_params(args.param)
    inj = Injection.create(args.kind, args.channel, params, args.start, args.duration)
    case = FaultCase(inj, args.seed, args.n_steps)
    result = execute_case(case)
    out = result.to_dict()
    out["cell"] = case.cell().label()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _bit_identical(a: Trace, b: Trace) -> bool:
    return a.float_bytes() == b.float_bytes()


def cmd_replay(args: argparse.Namespace) -> int:
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            case = FaultCase.from_json(fh.read())
    elif args.json:
        case = FaultCase.from_json(args.json)
    else:
        raise SystemExit("replay needs --file or --json")
    a, b = replay_case(case)
    same = _bit_identical(a, b)
    print(f"case_id        {case.case_id}")
    print(f"cell           {case.cell().label()}")
    print(f"bytes compared {len(a.float_bytes())}")
    print(f"bit identical  {same}")
    result = execute_case(case)
    print(f"severity       {result.severity.severity:.12f} ({result.severity.label})")
    return 0 if same else 1


def cmd_campaign(args: argparse.Namespace) -> int:
    pool = build_pool(args.pool_seed, args.replicates)
    sev = evaluate_pool(pool)

    def oracle(i: int) -> float:
        return sev[i]

    rng = np.random.default_rng(args.seed)
    if args.strategy == "learned":
        res, model = search_mod.learned(pool, oracle, args.budget, rng, warmup=args.warmup)
        cov, width = model.interval_coverage(list(pool), list(sev))
    else:
        fn = getattr(search_mod, args.strategy)
        res = fn(pool, oracle, args.budget, rng)
        cov, width = float("nan"), float("nan")
    tracker = CoverageTracker(kinds())
    for i in res.order:
        tracker.add(pool[i].injection, pool[i].n_steps)
    print(f"strategy            {res.strategy}")
    print(f"pool size           {len(pool)} cases over {tracker.total} cells")
    print(f"budget              {args.budget}")
    print(f"severe found        {res.n_severe} (threshold {SEVERE_THRESHOLD})")
    print(f"mean severity       {res.mean_severity:.6f}")
    print(f"coverage reached    {tracker.fraction:.6f} ({tracker.covered}/{tracker.total})")
    if args.strategy == "learned":
        print(f"interval coverage   {cov:.6f} (nominal 0.95), mean width {width:.6f}")
    if args.worst:
        pairs = sorted(zip(res.order, res.severities, strict=True), key=lambda t: -t[1])
        print()
        print("worst cases found:")
        for i, s in pairs[: args.worst]:
            print(f"  {s:.6f}  {pool[i].case_id}  {pool[i].cell().label()}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    rep = run_benchmark(
        budget=args.budget,
        n_pools=args.pools,
        n_seeds=args.seeds,
        warmup=args.warmup,
        progress=args.progress,
    )
    print(json.dumps(rep.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Top-level argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m faultinject",
        description=(
            "Fault-injection campaigns for GNC and communications software. "
            "Research-grade; not flight-qualified, not certified."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("taxonomy", help="print the fault taxonomy with units and ranges")
    sp.set_defaults(func=cmd_taxonomy)

    sp = sub.add_parser("cells", help="count or list coverage cells")
    sp.add_argument("--kind", help="restrict to one fault kind")
    sp.add_argument("--list", action="store_true", help="print every cell label")
    sp.set_defaults(func=cmd_cells)

    sp = sub.add_parser("run", help="execute one injected case and score it")
    sp.add_argument("--kind", required=True)
    sp.add_argument("--channel", required=True)
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--start", type=int, default=50, help="first step of injection")
    sp.add_argument("--duration", type=int, default=50, help="steps the fault is active")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--n-steps", dest="n_steps", type=int, default=N_STEPS)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("replay", help="replay a serialised case and check bit identity")
    sp.add_argument("--file", help="path to a JSON case file")
    sp.add_argument("--json", help="JSON case as a string")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("campaign", help="run one search strategy against a case pool")
    sp.add_argument(
        "--strategy", choices=search_mod.STRATEGIES, default="uniform_random"
    )
    sp.add_argument("--budget", type=int, default=60)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--pool-seed", dest="pool_seed", type=int, default=1)
    sp.add_argument("--replicates", type=int, default=2)
    sp.add_argument("--warmup", type=int, default=32)
    sp.add_argument("--worst", type=int, default=5, help="print the N worst cases found")
    sp.set_defaults(func=cmd_campaign)

    sp = sub.add_parser("benchmark", help="compare all strategies with bootstrap intervals")
    sp.add_argument("--budget", type=int, default=60)
    sp.add_argument("--pools", type=int, default=3)
    sp.add_argument("--seeds", type=int, default=8)
    sp.add_argument("--warmup", type=int, default=32)
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_benchmark)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover - only when stdout is a closed pipe
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
