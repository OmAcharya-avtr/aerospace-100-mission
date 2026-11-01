"""The worked example quoted in README.md, run so that the quote is the output.

The body of :func:`example` is the code block in README.md with ``out`` spelled
``print``. Its printed output below is what README.md quotes, copied from
``validation/worked_example_output.txt``, which this script writes.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()


def example(out: Tee) -> None:
    """The README's worked example, verbatim apart from ``out`` for ``print``."""
    out.rule("the worked example from README.md")
    out("")

    # --- begin README code block ---
    import pointmodel
    from pointmodel import synthetic

    # A pointing run confined to a narrow elevation band, which is what a
    # mission tracking one orbital regime produces. The generating
    # coefficients are known, so the fit can be scored against them.
    run = synthetic.generate("narrow_el", 300, sigma_arcsec=2.0, seed=20260101)

    budget = pointmodel.decompose(run.design, run.observed, sigma_arcsec=2.0)

    out(f"verdict        : {budget.identifiability.verdict}")
    out(f"rank           : {budget.identifiability.numerical_rank} of 9")
    out(f"condition      : {budget.identifiability.condition_number:.4g}")
    out(f"terms refused  : {', '.join(budget.refused_terms)}")
    out("")
    out("what the geometry cannot determine:")
    for direction in budget.identifiability.null_directions:
        out(f"  {direction.statement()}")
    out("")
    out("what it did determine, best first:")
    for combination in budget.identifiability.combinations[:3]:
        out(f"  {combination.statement()}")
    out("")
    out("the per-term budget, with the refusal marked:")
    for row in budget.terms:
        mark = "ok " if row.separable else "REF"
        truth = run.truth_mapping()[row.name]
        out(
            f"  {mark} {row.name:<4s} {row.coefficient:12.4f} "
            f"+/- {row.standard_error:9.4f}  truth {truth:8.4f}  "
            f"error {row.coefficient - truth:+12.4f}"
        )
    out("")
    separable = [row for row in budget.terms if row.separable]
    worst = max(abs(row.coefficient - run.truth_mapping()[row.name]) for row in separable)
    out(f"worst error on a SEPARABLE term  : {worst:.4f} arcsec")
    refused = [row for row in budget.terms if not row.separable]
    worst_refused = max(
        abs(row.coefficient - run.truth_mapping()[row.name]) for row in refused
    )
    out(f"worst error on a REFUSED term    : {worst_refused:.4f} arcsec")
    out(f"post-fit residual rms            : {budget.fit_rms_arcsec:.4f} arcsec")
    assert budget.huber_disagreement is not None
    out(
        f"robust disagreement              : "
        f"{budget.huber_disagreement.prediction_rms_arcsec:.4f} arcsec, "
        f"flagged {budget.huber_disagreement.flagged}"
    )
    # --- end README code block ---

    out("")
    out("The refusal is the point. The post-fit residual is a fraction of the")
    out("declared noise and the robust diagnostic is clean, so by every ordinary")
    out("measure the fit is good. Seven of the nine coefficients are refused, and")
    out("four of the seven are wrong by thousands of arcseconds: the elevation index")
    out("by 55,786. Nothing but a rank analysis says so.")
    out("")
    out.check(
        "the narrow band is refused",
        not budget.identifiability.separable,
        f"rank {budget.identifiability.numerical_rank} of 9",
    )
    out.check(
        "every separable term is recovered to better than 2 arcsec",
        worst < 2.0,
        f"worst {worst:.4f} arcsec",
    )
    out.check(
        "at least one refused term is wrong by more than 100 arcsec",
        worst_refused > 100.0,
        f"worst {worst_refused:.4f} arcsec",
    )
    out.check(
        "the fit looks good by its residual alone",
        budget.fit_rms_arcsec < 2.0 * run.sigma_arcsec,
        f"post-fit rms {budget.fit_rms_arcsec:.4f} arcsec against a declared sigma "
        f"of {run.sigma_arcsec:g}",
    )


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("pointmodel -- the worked example from README.md")
    for line in environment_lines():
        out(line)
    example(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
