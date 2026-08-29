# Master Roadmap

## Batch Plan

| Batch | Theme (indicative) | Flagship | Medium | Compact | Status |
|---|---|---:|---:|---:|---|
| 01 | FSO link engineering + PAT foundations | 2 | 3 | 5 | **APPROVED · PUBLISHED** |
| 02 | Atmospheric propagation + adaptive optics | 2 | 3 | 5 | **IN PROGRESS — 5 of 10 built** |
| 03 | Aerospace GNC core (estimation, control) | 2 | 3 | 5 | PLANNED |
| 04 | Space operations (orbits, contacts, scheduling) | 2 | 3 | 5 | PLANNED |
| 05 | Optical modulation, coding, networking | 2 | 3 | 5 | PLANNED |
| 06 | Autonomy and AI assurance | 2 | 3 | 5 | PLANNED |
| 07 | Airborne/UAV optical links | 2 | 3 | 5 | PLANNED |
| 08 | Digital twins + HIL preparation | 2 | 3 | 5 | PLANNED |
| 09 | V&V, safety, reliability tooling | 2 | 3 | 5 | PLANNED |
| 10 | Integration, Jetson progression, portfolio hardening | 2 | 3 | 5 | PLANNED |

Batch themes after Batch 1 are indicative and re-planned at each batch specification against the validation distribution (10 L1 / 60 L2 / 25 L3 / 5 L4) and the ≥70 AI requirement.

## Fixed Gates

1. Each batch stops at READY FOR APPROVAL before any publication.
2. ≤5 public pushed commits per session.
3. Secret scan + dependency scan before every push.
4. IP gate (§28) answered per product before release.
5. No unsupported flight-safety or certification claims, ever.

## Schedule Reality

The original plan assumed a persistent 9-hour nightly local agent. Actual execution is discrete cloud sessions (manually started or fired by scheduled tasks). Velocity is measured, not assumed; the roadmap is re-baselined after Batch 1.

Re-baselined 2026-08-29. Measured unit of planning is one batch per productive session. Eighty-five products remain,
roughly nine batches, therefore roughly nine to twelve productive sessions. Between 2026-08-07 and 2026-08-29 the
repository received zero commits because the nightly automation stalled — the binding constraint has been
orchestration reliability, not engineering throughput. See `system/AUTOMATION.md` and R-11.

## Quota Discipline

Batch themes after Batch 1 are re-planned at each batch specification against the remaining quota gap, which is
computed by `scripts/quota_report.py` rather than asserted (ADR-009). Current gap: 18 flagship, 27 medium, 40 compact,
≥61 AI, and 6 / 51 / 23 / 5 across validation Levels 1–4. Level 4 has not started and must begin no later than
Batch 05. Products stay within the mission's declared domains; no expansion into unrelated generic aerospace apps.
