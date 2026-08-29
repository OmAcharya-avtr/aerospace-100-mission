#!/usr/bin/env python3
"""Derive mission quota totals from products.yaml.

products.yaml is the single machine-readable source of truth (ADR-009).
MISSION_STATUS.md totals are derived from this script, never hand-maintained.

Counts cover products BUILT to the completion gate. A product counts as built
when its status is one of BUILT_STATES below; PLANNED / RESEARCHING /
SPECIFYING / DEVELOPING products are registered but not yet built.

Usage:  python3 scripts/quota_report.py [--check]
        --check exits non-zero if any quota is over-subscribed.

No third-party dependencies: this must run in the publication environment,
which does not carry the build toolchain (ADR-004, Environment B).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGETS = {
    "total": 100,
    "flagship": 20,
    "medium": 30,
    "compact": 50,
    "ai": 70,          # minimum, not a cap
    "L1": 10,
    "L2": 60,
    "L3": 25,
    "L4": 5,
}

BUILT_STATES = {
    "TESTING", "VALIDATING", "SECURITY REVIEW", "DOCUMENTING",
    "REVIEW REQUIRED", "READY FOR APPROVAL", "APPROVED", "PUBLISHED",
    "NEEDS HARDENING",
}

FIELD = re.compile(r"^    (\w+): *(.*?)\s*(?:#.*)?$", re.M)


def parse(path: Path) -> list[dict]:
    text = path.read_text()
    products = []
    for block in text.split("\n  - id: ")[1:]:
        rec = {"id": block.split("\n", 1)[0].strip()}
        for key, value in FIELD.findall(block):
            rec[key] = value
        products.append(rec)
    return products


def tally(products: list[dict]) -> tuple[dict, dict]:
    built = [p for p in products if p.get("status") in BUILT_STATES]
    counts = {
        "total": len(built),
        "flagship": sum(p.get("class") == "flagship" for p in built),
        "medium": sum(p.get("class") == "medium" for p in built),
        "compact": sum(p.get("class") == "compact" for p in built),
        "ai": sum(p.get("ai_enabled") == "true" for p in built),
    }
    for level in (1, 2, 3, 4):
        counts[f"L{level}"] = sum(
            p.get("validation_level") == str(level) for p in built
        )
    published = sum(p.get("published") == "true" for p in built)
    meta = {"registered": len(products), "published": published}
    return counts, meta


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    products = parse(root / "products.yaml")
    counts, meta = tally(products)

    print(f"Registered: {meta['registered']} / {TARGETS['total']}")
    print(f"Built:      {counts['total']} / {TARGETS['total']}")
    print(f"Published:  {meta['published']} / {TARGETS['total']}")
    print()
    print(f"{'Quota':<22}{'Target':>8}{'Built':>8}{'Remaining':>11}")
    over = []
    for key in ("flagship", "medium", "compact", "ai", "L1", "L2", "L3", "L4"):
        target, got = TARGETS[key], counts[key]
        label = key if key != "ai" else "ai (minimum)"
        print(f"{label:<22}{target:>8}{got:>8}{target - got:>11}")
        if key != "ai" and got > target:
            over.append(f"{key}: {got} built exceeds target {target}")

    if over:
        print("\nQUOTA OVER-SUBSCRIBED:")
        for line in over:
            print("  " + line)
        return 1 if "--check" in sys.argv else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
