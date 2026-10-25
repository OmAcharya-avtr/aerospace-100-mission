"""Write the built-in cases and missions to ``examples/diagrams/`` and check they round-trip.

The declared diagram is the product's contribution, so the declaration format
has to be a stable artifact: a committed YAML file that a reviewer can diff,
and that the loader reads back to the same object it was written from.

This script writes every built-in case and mission profile to
``examples/diagrams/`` and then re-loads each one and compares the re-serialised
text with what was written. A mismatch is a **blocking** failure, because it
means the declaration cannot be trusted as an artifact.

The CI workflow runs this script and then ``git diff --exit-code
examples/diagrams/``, so a change to a built-in case that was not regenerated
fails the build rather than leaving the committed YAML stale.
"""

from __future__ import annotations

from _bootstrap import ROOT, Tee, bootstrap, header, relative

bootstrap()

from reliablock.cases import CASES, MISSIONS  # noqa: E402
from reliablock.diagram import load_diagram  # noqa: E402
from reliablock.mission import load_mission  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402

TARGET = ROOT / "examples" / "diagrams"


def main() -> int:
    tee = Tee("regenerate_declarations_output.txt")
    header(tee, "declared diagrams and missions: written out, read back, compared")
    tee("")
    tee(f"  target directory                   : {relative(TARGET)}")
    TARGET.mkdir(parents=True, exist_ok=True)

    tee.rule("diagrams")
    tee("")
    tee(f"  {'case':<24s}{'file':<42s}{'blocks':>8s}{'bytes':>8s}{'round-trip':>12s}")
    tee(f"  {'-' * 94}")
    for name in sorted(CASES):
        diagram = CASES[name]
        path = TARGET / f"{name}.yaml"
        text = diagram.to_yaml()
        path.write_text(text)
        reloaded = load_diagram(path)
        same = reloaded.to_yaml() == text
        tee(
            f"  {name:<24s}{relative(path):<42s}{len(diagram.blocks()):>8d}"
            f"{len(text):>8d}{str(same):>12s}"
        )
        tee.require(f"{name} round-trips through YAML", same)
        # The reloaded diagram must also quantify to the same number, which is
        # the property that actually matters: a declaration that round-trips
        # textually but quantifies differently would be worse than one that
        # fails to load.
        original = quantify(diagram, 8766.0)
        restored = quantify(reloaded, 8766.0)
        tee.require(
            f"{name} quantifies to the same reliability after a round trip",
            original.reliability == restored.reliability,
            f"{original.reliability!r} against {restored.reliability!r}",
        )

    tee.rule("mission profiles")
    tee("")
    tee(f"  {'mission':<24s}{'file':<42s}{'phases':>8s}{'hours':>12s}{'round-trip':>12s}")
    tee(f"  {'-' * 98}")
    for name in sorted(MISSIONS):
        profile = MISSIONS[name]
        path = TARGET / f"{name}.yaml"
        text = profile.to_yaml()
        path.write_text(text)
        reloaded = load_mission(path)
        same = reloaded.to_yaml() == text
        tee(
            f"  {name:<24s}{relative(path):<42s}{len(profile.phases):>8d}"
            f"{profile.total_hours:>12.6g}{str(same):>12s}"
        )
        tee.require(f"{name} round-trips through YAML", same)

    tee.rule("what a declaration looks like")
    tee("")
    tee("  The smallest built-in case, in full, so that a reader can see the")
    tee("  format without opening a file:")
    tee("")
    for line in CASES["redundant_pair"].to_yaml().splitlines():
        tee(f"    {line}")

    tee.rule("the loader rejects what it does not understand")
    tee("")
    tee("  A declaration is an artifact a reviewer has to be able to check, so an")
    tee("  unknown key is an error and not a silently ignored field. These are")
    tee("  the rejections, exercised here so that the messages are on record.")
    tee("")
    import tempfile
    from pathlib import Path

    bad_declarations = {
        "unknown top-level key": "name: x\nroot: {kind: block, name: a, "
        "distribution: {kind: exponential, rate_per_hour: 1.0e-5}}\nextra: 1\n",
        "misspelled distribution parameter": "name: x\nroot: {kind: block, name: a, "
        "distribution: {kind: exponential, rate_per_hr: 1.0e-5}}\n",
        "unknown node kind": "name: x\nroot: {kind: majority_voter, items: []}\n",
        "parallel group of one": "name: x\nroot: {kind: parallel, items: "
        "[{kind: block, name: a, distribution: {kind: exponential, "
        "rate_per_hour: 1.0e-5}}]}\n",
        "standby without a switch reliability": "name: x\nroot: {kind: standby, "
        "primary: {kind: block, name: p, distribution: {kind: exponential, "
        "rate_per_hour: 1.0e-5}}, standbys: []}\n",
        "k outside 1..n": "name: x\nroot: {kind: k_of_n, k: 4, items: "
        "[{kind: block, name: a, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}, "
        "{kind: block, name: b, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}]}\n",
        "duplicate block names": "name: x\nroot: {kind: parallel, items: "
        "[{kind: block, name: a, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}, "
        "{kind: block, name: a, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}]}\n",
        "beta factor on a k = n group": "name: x\nroot: {kind: k_of_n, k: 2, "
        "beta_factor: 0.1, items: "
        "[{kind: block, name: a, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}, "
        "{kind: block, name: b, distribution: {kind: exponential, rate_per_hour: 1.0e-5}}]}\n",
    }
    with tempfile.TemporaryDirectory() as scratch:
        for label, body in bad_declarations.items():
            path = Path(scratch) / "bad.yaml"
            path.write_text(body)
            try:
                load_diagram(path)
            except (ValueError, TypeError) as error:
                message = " ".join(str(error).split())
                # The loader quotes back the path it was given, and that path
                # is in a temporary directory outside the repository. No
                # committed file in this repository may carry an absolute
                # path, so the scratch prefix is normalised out HERE,
                # deliberately, and this comment records that it was.
                message = message.replace(str(path), "<temporary directory>/bad.yaml")
                message = message.replace(scratch, "<temporary directory>")
                tee(f"  {label}:")
                tee(f"    {message[:300]}")
                tee.check(f"{label} is rejected", True)
            else:
                tee.check(f"{label} is rejected", False, "it was accepted")
            tee("")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
