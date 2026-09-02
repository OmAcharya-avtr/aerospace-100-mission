# Session Report — YYYY-MM-DD

**Session window:** (start–end, ET)
**Batch in progress:**

## Products worked on
| ID | Name | Status before | Status after |
|---|---|---|---|

## Completed tasks
-

## Failed tasks
-

## Test status
(passed / failed / skipped per product)

## Validation status
-

## Blockers
-

## Local commit hashes
-

## Files changed
-

## Resource usage
(container CPU/RAM/disk notes; training time budgets)

## Agent and token accounting — REQUIRED, no estimates

Every build agent's tool result reports `subagent_tokens`, `tool_uses` and
`duration_ms`. Copy those numbers; never estimate them, never carry a figure
across from another session, and never leave the section out because a run
was short. A row with an unknown value says `not recorded` and explains why.

| Agent | Product | Model | Tokens | Tool calls | Wall time | Outcome |
|---|---|---|---:|---:|---:|---|
| | | | | | | |
| **Total (agents)** | | | | | | |

Coordinating session: tokens `<value>`, tool calls `<value>`.
**Session total: `<agents + coordinator>` tokens.**

Cost drivers worth naming when they apply: agents terminated by a usage limit
and resumed (resumption re-reads the transcript and is not free); products
rebuilt rather than resumed; validation campaigns re-run by the coordinator,
which is required by ADR-010 and is the single largest coordinator cost.

## Security findings
-

## Approval needs
-

## Next actions
-
