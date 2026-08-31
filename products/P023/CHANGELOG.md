# Changelog

All notable changes to AllocLab are documented in this file.

## [0.1.0] - 2026-08-31

Initial release.

### Added

- `alloclab.effectors`: `EffectorSet` around a general `(3, m)` control
  effectiveness matrix with per-effector command bounds; builders for thruster
  clusters (`thruster_cluster`, torque column `r x F_hat`), reaction-wheel
  arrays (`reaction_wheel_array`, column `-a_hat`), the standard skewed
  four-wheel pyramid (`pyramid_reaction_wheels`, isotropic at
  `arctan(sqrt(2))` = 54.7356 deg), and an orthogonal triad; failure modelling
  through `with_failures`, which pins a failed effector's bounds rather than
  deleting its column so a stuck-open thruster keeps its torque bias.
- `alloclab.allocation`: five allocators sharing one `AllocationResult` —
  `pseudo_inverse_allocate`, `weighted_pseudo_inverse_allocate`,
  `redistributed_pseudo_inverse_allocate` (Bodson 2002 sec. V.A),
  `lp_allocate` (min-1-norm-error and min-control objectives, HiGHS or
  PuLP/CBC), and `qp_allocate` (Haerkegaard 2002 mixed optimisation, solved
  exactly by BVLS bounded least squares). `is_attainable` gives an exact LP
  feasibility certificate.
- `alloclab.ams`: exact attainable-moment-set geometry as a zonotope, by
  Durham's pairwise facet construction or by brute-force box enumeration;
  closed-form `zonotope_volume` and `expected_vertex_count`; hull membership
  and direct-allocation `boundary_scale`.
- `alloclab.failure`: `reallocate_after_failure`, which decides feasibility
  with the LP certificate before looking at the allocator's output, so "the
  allocator failed" and "no command could have worked" are distinguished;
  `failure_margin`; `InfeasibleAllocationError` for callers that want a raise.
- `alloclab.dataset`: deterministic seeded generation of (torque, health) to
  QP-command examples, and the eight-thruster reference cluster.
- `alloclab.ml.LearnedAllocator`: an ensemble of five scikit-learn MLPs trained
  to imitate the QP, with per-effector ensemble spread and a scalar confidence.
- `python -m alloclab` CLI: `config`, `ams`, `allocate` (with `--failed`),
  exiting 1 on an infeasible command.
- Validation suite (`validation/`), five scripts with saved raw stdout.
- 185 tests: unit, input-validation, hand-calculated known-answer, edge-case,
  Hypothesis property, integration, CLI-subprocess, and pinned
  benchmark/regression tests.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `README.md`, CI workflow.

### Fixed during development

- The pairwise attainable-moment-set construction originally enumerated only
  the four corners of each facet parallelogram, which is correct only when
  exactly two generators lie in the facet plane. Hypothesis, cross-checking it
  against brute-force enumeration, found a configuration with three coplanar
  generators where a vertex 0.5 N*m away was lost and the volume was
  under-reported by 3.6%. Facets are now enumerated exactly as 2-D zonotopes
  by an angular walk
  (`tests/test_ams.py::test_coplanar_generators_do_not_lose_facet_vertices`).

### Known limitations (see README "Limitations")

- The learned allocator violates actuator bounds on 95.15% of held-out samples,
  by up to 0.3294 N of a 1 N thrust limit, where the exact QP never does; and
  called one command at a time it is 2x slower than the QP, so its measured
  speed advantage exists only for large batches.
- The QP's torque residual on an attainable command scales as `1/gamma` rather
  than being exact, and degrades further when the active set leaves open a
  direction of very small torque effectiveness.
- LP allocation inherits the HiGHS primal feasibility tolerance (1e-7), giving
  a residual floor of about `1e-7 * ||B||` N*m.
- The redistributed pseudo-inverse missed 23 of 642 attainable commands over
  random configurations, which is a documented property of the heuristic.
- No actuator dynamics, no rate limits, no wheel momentum saturation, no
  closed-loop evaluation, and no comparison against hardware.
