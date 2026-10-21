"""The worked example quoted in README.md, run so that its output is real.

Part one is the declared-value path on the reference case: select, compare
against the greedy baseline, price the next minute of contact, and read the
left-behind report. Part two is the learned path on the case with missing and
stale declarations, where the declared-value passthrough has nothing to say.

Everything printed below is produced by the code printed below. The README
quotes both from this file's committed output, which is regenerated as the last
step of every build.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from downlinkpick import (  # noqa: E402
    CompressibilityModel,
    SelectionProblem,
    ValueModel,
    best_window_to_extend,
    declared_values,
    exact_select,
    format_full_report,
    generate_training_table,
    greedy_select,
    hybrid_values,
    left_behind_report,
    load_case,
    marginal_value_of_contact,
    sizes_from_ratios,
    value_left_behind,
)
from downlinkpick.dataset import value_feature_row  # noqa: E402

PART_ONE = '''\
from downlinkpick import (
    SelectionProblem, best_window_to_extend, exact_select, greedy_select,
    left_behind_report, load_case, marginal_value_of_contact, value_left_behind,
)

case = load_case("examples/cases/three_pass_declared.yaml")
problem = SelectionProblem.from_declared(case.inventory, case.plan)

greedy = greedy_select(problem)
exact = exact_select(problem)
print(f"greedy {greedy.total_value:.4f}  exact {exact.total_value:.4f} ({exact.status})")
print(f"greedy gap {(exact.total_value - greedy.total_value) / exact.total_value:.6f}")
print(f"queue {problem.inventory.total_downlink_bits() / 1e9:.2f} Gbit, "
      f"contact {problem.plan.total_capacity_bits() / 1e9:.2f} Gbit over "
      f"{problem.plan.total_contact_s():.0f} s")

marginals = marginal_value_of_contact(problem, base=exact, delta_s=60.0)
best = best_window_to_extend(marginals)
print(f"extend {best.window}: {best.value_per_second:.6f} value/s, "
      f"{best.delta_bits / 1e9:.3f} Gbit, admits {', '.join(best.entering) or 'nothing'}")

rows = left_behind_report(problem, exact)
print(f"left behind {len(rows)} products worth {sum(r.value for r in rows):.4f}")
for row in rows[:3]:
    print(f"  {row.product:>8s} value {row.value:6.3f}  {row.reason:>9s}  "
          f"{row.extra_seconds_needed:7.1f} s short on {row.best_window}")
print("by reason:", {k: round(v, 3) for k, v in value_left_behind(rows).items()})  # noqa
'''

PART_TWO = '''\
from downlinkpick import (
    CompressibilityModel, SelectionProblem, ValueModel, declared_values,
    exact_select, generate_training_table, hybrid_values, load_case,
    sizes_from_ratios,
)
from downlinkpick.dataset import value_feature_row
import numpy as np

case = load_case("examples/cases/missing_declarations.yaml")
inventory = case.inventory
print(f"{len(inventory.missing_declarations())} of {len(inventory)} values undeclared, "
      f"{len(inventory.missing_compressibility())} compressibilities unmeasured")

table = generate_training_table(6000, 1001)
values_model = ValueModel().fit(table.value_features, table.value_target)
ratio_model = CompressibilityModel().fit(
    table.compression_features, table.compression_target
)

features = np.array([value_feature_row(p, inventory) for p in inventory])
value, dispersion = values_model.predict_with_dispersion(features)
learned = dict(zip(inventory.names, (float(v) for v in value)))
ratios = ratio_model.predict_inventory(inventory, case.autocorrelation)

problem = SelectionProblem(
    inventory, case.plan,
    values=hybrid_values(inventory, learned),
    sizes_bits=sizes_from_ratios(inventory, ratios.conservative_mapping()),
    value_source="declared where present, learned where absent",
)
selection = exact_select(problem)
passthrough = SelectionProblem(
    inventory, case.plan,
    values=declared_values(inventory, fallback="zero"),
    sizes_bits=problem.sizes_bits, value_source="declared passthrough",
)
print(f"hybrid values  : {selection.total_value:.4f} over "
      f"{len(selection.assignments)} products")
print(f"passthrough    : {exact_select(passthrough).total_value:.4f} over "
      f"{len(exact_select(passthrough).assignments)} products")
widest = max(zip(inventory.names, dispersion), key=lambda item: item[1])
print(f"widest ensemble spread: {widest[0]} at +-{widest[1]:.3f} value units")
'''


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("downlinkpick worked example")
    for line in environment_lines():
        out(line)

    out.rule("part one: declared values on the reference case")
    out("the code, exactly as README.md quotes it")
    out("")
    for line in PART_ONE.rstrip("\n").splitlines():
        out(f"    {line}")
    out("")
    out("its output")
    out("")

    case = load_case("examples/cases/three_pass_declared.yaml")
    problem = SelectionProblem.from_declared(case.inventory, case.plan)
    greedy = greedy_select(problem)
    exact = exact_select(problem)
    out(f"greedy {greedy.total_value:.4f}  exact {exact.total_value:.4f} ({exact.status})")
    out(f"greedy gap {(exact.total_value - greedy.total_value) / exact.total_value:.6f}")
    out(
        f"queue {problem.inventory.total_downlink_bits() / 1e9:.2f} Gbit, "
        f"contact {problem.plan.total_capacity_bits() / 1e9:.2f} Gbit over "
        f"{problem.plan.total_contact_s():.0f} s"
    )
    marginals = marginal_value_of_contact(problem, base=exact, delta_s=60.0)
    best = best_window_to_extend(marginals)
    out(
        f"extend {best.window}: {best.value_per_second:.6f} value/s, "
        f"{best.delta_bits / 1e9:.3f} Gbit, admits "
        f"{', '.join(best.entering) or 'nothing'}"
    )
    rows = left_behind_report(problem, exact)
    out(f"left behind {len(rows)} products worth {sum(r.value for r in rows):.4f}")
    for row in rows[:3]:
        out(
            f"  {row.product:>8s} value {row.value:6.3f}  {row.reason:>9s}  "
            f"{row.extra_seconds_needed:7.1f} s short on {row.best_window}"
        )
    out(
        "by reason: {"
        + ", ".join(
            f"{key!r}: {value:.3f}" for key, value in value_left_behind(rows).items()
        )
        + "}"
    )

    out.rule("part one, continued: the full report the CLI prints")
    out(format_full_report(problem, exact, marginals, rows))

    out.rule("part two: the learned path on the case with missing declarations")
    out("the code, exactly as README.md quotes it")
    out("")
    for line in PART_TWO.rstrip("\n").splitlines():
        out(f"    {line}")
    out("")
    out("its output")
    out("")

    case = load_case("examples/cases/missing_declarations.yaml")
    inventory = case.inventory
    out(
        f"{len(inventory.missing_declarations())} of {len(inventory)} values undeclared, "
        f"{len(inventory.missing_compressibility())} compressibilities unmeasured"
    )
    table = generate_training_table(6000, 1001)
    values_model = ValueModel().fit(table.value_features, table.value_target)
    ratio_model = CompressibilityModel().fit(
        table.compression_features, table.compression_target
    )
    features = np.array([value_feature_row(p, inventory) for p in inventory])
    value, dispersion = values_model.predict_with_dispersion(features)
    learned = dict(zip(inventory.names, (float(v) for v in value), strict=True))
    ratios = ratio_model.predict_inventory(inventory, case.autocorrelation)
    problem = SelectionProblem(
        inventory,
        case.plan,
        values=hybrid_values(inventory, learned),
        sizes_bits=sizes_from_ratios(inventory, ratios.conservative_mapping()),
        value_source="declared where present, learned where absent",
    )
    selection = exact_select(problem)
    passthrough = SelectionProblem(
        inventory,
        case.plan,
        values=declared_values(inventory, fallback="zero"),
        sizes_bits=problem.sizes_bits,
        value_source="declared passthrough",
    )
    passthrough_selection = exact_select(passthrough)
    out(
        f"hybrid values  : {selection.total_value:.4f} over "
        f"{len(selection.assignments)} products"
    )
    out(
        f"passthrough    : {passthrough_selection.total_value:.4f} over "
        f"{len(passthrough_selection.assignments)} products"
    )
    widest = max(
        zip(inventory.names, dispersion, strict=True), key=lambda item: item[1]
    )
    out(f"widest ensemble spread: {widest[0]} at +-{widest[1]:.3f} value units")
    out("")
    out("the two objectives above are NOT comparable as numbers: they are measured")
    out("against different value tables. What is comparable is the count of products")
    out("selected and the fact that the passthrough cannot see 40 per cent of the queue")
    out("at all. The scored comparison, on ground-truth values through the same solver,")
    out("is in validation/validate_valuemodel_output.txt.")

    out.rule("part two, continued: the ten widest ensemble spreads")
    out("the products an operator should look at by hand before spending a pass on them")
    out("")
    out(f"{'product':>12s} {'declared':>10s} {'learned':>10s} {'spread':>9s} "
        f"{'ratio':>8s} {'ratio +2s':>10s}")
    order = np.argsort(-dispersion)[:10]
    ratio_mapping = ratios.as_mapping()
    conservative_mapping = ratios.conservative_mapping()
    for index in order:
        name = inventory.names[index]
        declared = inventory[name].declared_value
        out(
            f"{name:>12s} "
            f"{'none' if declared is None else f'{declared:10.4f}':>10s} "
            f"{value[index]:10.4f} {dispersion[index]:9.4f} "
            f"{ratio_mapping[name]:8.4f} {conservative_mapping[name]:10.4f}"
        )

    out.rule("summary")
    out("no assertions in this script: it is the worked example, and its purpose is to")
    out("produce the output README.md quotes")
    out(f"output  : {out.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
