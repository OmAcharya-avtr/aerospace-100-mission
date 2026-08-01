# Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R-01 | 100 products in 4 weeks exceeds real session throughput | High | High | Re-baseline after Batch 1 velocity measurement; protect quality gates over quantity | OPEN |
| R-02 | GitHub PAT was pasted in chat (violates mission §8) | Certain (occurred) | Medium | Token now stored at local `secrets/github_token.txt`; owner should rotate the token on GitHub and update that file; all future reads come from the file | OPEN — ACTION: OWNER |
| R-03 | Automated environment blocked from creating GitHub repos | Occurred | Medium | Owner creates empty repos in browser; automation pushes content; retry API periodically | OPEN |
| R-04 | Nightly scheduled sessions cannot reach token if Mac is asleep / desktop app closed | Medium | Medium | Keep Mac awake during window, or run sessions manually | OPEN |
| R-05 | Cloud container is ephemeral; unpushed work is lost when container is reclaimed | Medium | High | Push state to GitHub at end of every session; trackers synced to local folder | OPEN |
| R-06 | Overstated aerospace claims damage credibility | Low | High | Status vocabulary enforced; validation evidence required; no certification language | OPEN |
| R-07 | Name conflicts with existing packages/products | Medium | Low | Conflict check (PyPI/npm/GitHub search) per product before publication; names provisional until then | OPEN |
| R-08 | AI products lacking meaningful baselines fail §11 | Medium | Medium | Baseline-first development: analytic/classical baseline implemented before ML model | OPEN |
| R-09 | Large artifacts (datasets, checkpoints) bloat repos | Medium | Low | .gitignore excludes; regeneration scripts committed instead of artifacts | OPEN |
| R-10 | Build environment (2-core cloud container) limits training scale | High | Medium | Small models, synthetic datasets, budgeted training runs; document compute limits in model cards | OPEN |
