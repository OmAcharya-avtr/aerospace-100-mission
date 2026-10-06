# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `arqlonghaul.link` — link geometry reduced to the four numbers ARQ
  throughput needs, with the slot count `N = 1 + RTT/T_f`, the
  bandwidth-delay product in bits, bytes and frames, and four illustrative
  link presets whose nominal geometry is labelled as such.
- `arqlonghaul.closedform` — the classical stop-and-wait, go-back-N and
  selective-repeat throughput expressions with their derivations and validity
  conditions in the docstrings, a dispatcher, and the window knee. The
  window-limited selective-repeat branch is documented as an upper bound that
  is exact only at a window of one, with the measured shortfall quoted.
- `arqlonghaul.protocols` — slotted simulators for the three protocol state
  machines, with a feedback pipeline of `N` slots, in-order window-base
  advancement, go-back storms and head-of-line blocking. All three consume a
  pre-generated per-slot error array so that protocol comparisons use common
  random numbers, and all three report a batch-means standard error.
- `arqlonghaul.window` — window sizing against the bandwidth-delay product in
  frames and bytes, the marginal throughput gained per extra frame of window,
  and a sweep that marks go-back-N as undefined below its continuity threshold
  rather than returning a wrong number there.
- `arqlonghaul.channel` — memoryless and Gilbert-Elliott frame-error channels
  behind one interface, with a constructor that fixes the marginal frame error
  rate and the mean burst length independently so that correlated and
  memoryless channels can be compared at the same marginal; stationary
  marginal, sojourn and autocorrelation expressions; coherent BPSK bit error
  rate and the independent-bit bit-to-frame error mapping.
- `arqlonghaul.crc` — five catalogue CRCs (CRC-32, CRC-32C, CRC-16/CCITT-FALSE,
  CRC-16/ARC, CRC-8) implemented in-package in both a table-driven and a
  bit-at-a-time form, verified against the `crcmod` 1.7 catalogue check values
  and, for CRC-32, against `zlib.crc32`, plus frame check sequence append and
  verify.
- `arqlonghaul.harq` — type-I chase combining and type-II incremental
  redundancy over a bounded-distance decoder with the correcting power taken
  from the Singleton bound and an explicit code-family efficiency; goodput and
  residual frame error rate by an independent-round approximation, by an exact
  nested dynamic program, and by Monte Carlo; the throughput-optimal
  first-transmission rate; and the retransmit-versus-redundancy crossover with
  its sensitivity to every assumption that fixes it.
- `arqlonghaul.datasets` — the HARQ-over-fade environment, the long-burst and
  short-burst regimes that share a marginal error rate but not a correlation
  time, and the three disjoint fit / tune / report seed sets with a
  construction that refuses to build overlapping ones.
- `arqlonghaul.policy` — the analytic fixed schedule computed from the closed
  form with no data, two baselines tuned on the tune seeds, and a
  random-forest cost-to-go learned policy with an across-tree standard
  deviation and a decision margin as its uncertainty output.
- `python -m arqlonghaul` — `link`, `closedform`, `window`, `simulate`, `harq`
  and `crc` subcommands.
- 229 tests; nine validation scripts with their raw output; four examples
  producing four figures.

### Published results

- The simulator agrees with the classical closed forms to a worst **0.616 per
  cent** relative over 78 cases in the regime where they are exact, with a
  worst `|z|` of **2.36** against the batch-means standard error.
- At a 5 per cent marginal frame error rate on an `N = 60` link, the
  independent-error go-back-N expression **understates** measured goodput by
  **74.1 per cent** once errors arrive in bursts of 100 frames. Stop-and-wait
  moves by at most **0.52 per cent**, as predicted.
- The textbook window of one bandwidth-delay product plus one frame delivers
  **56.5 per cent** of the ideal selective-repeat goodput at a 5 per cent
  frame error rate; the window needed to come within 1 per cent is three to
  six times that.
- The retransmit-versus-redundancy crossover sits at a feedback latency of
  **85.9 channel symbol times** at the stated operating point, four to six
  orders of magnitude below every link preset in the repository.
- **The tuned fixed schedule beats the learned policy** on the reporting
  seeds, by 1.66 per cent of cost per frame in the long-burst regime
  (`|z| = 2.27`) and 3.03 per cent in the short-burst regime (`|z| = 23.50`).
  The learned policy does beat the analytic fixed schedule, by 24.1 and 36.4
  per cent of cost. Both results are published as measured.

### Known limitations

See the Limitations section of `README.md`. The reverse channel is error free
and its delay is inside `N`; frames are of constant length; type-I chase
combining rounds are modelled as independent decoding trials; the code family
is a Singleton-bound model rather than a named standardised code; the HARQ
throughput accounting assumes one process with no pipelining.
