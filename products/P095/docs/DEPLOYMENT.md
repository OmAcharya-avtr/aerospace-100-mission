# Deployment — boardprobe on an NVIDIA Jetson Orin Nano

A procedure to execute, not prose about a procedure. Every step has a command,
an expected result and what to do when the result differs.

**NONE OF THIS HAS BEEN EXECUTED ON A JETSON ORIN NANO.** No board was
reachable from the environment this package was built in. Steps 1 to 3 were
executed on an x86_64 Linux host and behave as written there; steps 4 onward
are written from the documentation cited in `src/boardprobe/orin_nano.py` and
are **UNVERIFIED**. Expect to correct them on first contact with the board,
and record the corrections.

---

## 0. Prerequisites

- A Jetson Orin Nano Developer Kit running NVIDIA Jetson Linux (L4T), booted,
  with a shell.
- Python 3.11 or later. `python3 --version`.
- Read access to `/sys` as the user that will run the harness, or `sudo`.
- No GPU libraries, no CUDA, no vendor Python packages. This package depends on
  `numpy` and nothing else.

## 1. Install

```bash
git clone https://github.com/OmAcharya-avtr/boardprobe.git
cd boardprobe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: a successful install of `numpy`, `pytest`, `hypothesis`, `ruff` and
`matplotlib`.

If `pip` cannot build `numpy` from source on the board, install the distribution
package first (`sudo apt install python3-numpy`) and create the virtual
environment with `--system-site-packages`.

## 2. Verify the install without touching the board

```bash
python -m pytest tests/ -q
python -m boardprobe dry-run
```

Expected: the suite passes, and the dry run prints

```
device accesses against the real root : 0
side-effect free                      : yes
```

The dry run reads no device node. If `side-effect free` reads `NO`, stop: a
file appeared under the repository while the walk ran and the cause must be
found before any board work. See `docs/RECOVERY.md` §4.

## 3. Identify the board

```bash
python -m boardprobe attest
```

Expected on the target:

```
declared target : NVIDIA Jetson Orin Nano
observed model  : NVIDIA Jetson Orin Nano Developer Kit
source          : proc/device-tree/model
matches target  : yes
```

If `matches target` reads `NO`:

- `observed model` names another board — you are not on the target. Nothing
  this package labels will be a measurement of a Jetson Orin Nano, and every
  certificate will say so. That is correct behaviour, not a fault.
- `source` reads `platform.machine() fallback` — `proc/device-tree/model` does
  not exist. On a device-tree platform it should. Check that `/proc` is
  mounted.
- `observed model` is a Jetson string that does not start with the declared
  prefix — a carrier board or L4T release this package has not seen. Change
  `TARGET_MODEL_PREFIX` in `src/boardprobe/orin_nano.py` deliberately, in a
  commit, with the observed string in the message. Do not pass a different
  prefix on the command line as a workaround.

## 4. Discover the thermal zone indices — UNVERIFIED

The zone index for a named zone is **not fixed by any document** and is the
single most likely thing in the channel table to be wrong on a given L4T
release. Discover it:

```bash
for zone in /sys/devices/virtual/thermal/thermal_zone*/; do
  printf '%s %s %s\n' "$zone" "$(cat "$zone/type" 2>/dev/null)" \
    "$(cat "$zone/temp" 2>/dev/null)"
done
```

Expected: one line per zone, with names such as `cpu-thermal`, `gpu-thermal`,
`cv0-thermal`, `soc0-thermal` and `tj-thermal`, and a temperature in
**millidegrees Celsius** — a plausible idle reading is around `40000`, meaning
40 °C.

Then reconcile against the declared table:

```bash
python -m boardprobe channels --references
```

For every die-temperature channel whose declared `thermal_zone<N>` index does
not match the zone of that name above, edit
`src/boardprobe/orin_nano.py`. Set `verified_on_hardware=True` on each channel
you have now read, and **only** on those. A channel flagged verified that was
never read is the one thing in this package that would make its own labels
worthless.

If a `temp` file reads a value near `4e4` rather than `4e7`, the node is in
degrees rather than millidegrees on this kernel; change the channel's `scale`
and `raw_unit` and say so in the commit.

## 5. Find the INA3221 power monitor — UNVERIFIED

```bash
ls -d /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/ 2>/dev/null
for node in /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/; do
  for index in 1 2 3; do
    printf '%s rail %s: %s %s mV %s mA\n' "$node" "$index" \
      "$(cat "$node/in${index}_label" 2>/dev/null)" \
      "$(cat "$node/in${index}_input" 2>/dev/null)" \
      "$(cat "$node/curr${index}_input" 2>/dev/null)"
  done
done
```

Expected: one `hwmon<N>` directory under `1-0040`, with rails labelled
`VDD_IN`, `VDD_CPU_GPU_CV` and `VDD_SOC`, bus voltages near `4800`–`5100` mV
and currents of a few hundred to a few thousand mA at idle.

- Nothing listed: the driver is not bound. `lsmod | grep ina3221`;
  `dmesg | grep -i ina3221`. On some L4T releases the driver is
  `ina3221_driver` or is built in.
- A different I2C address or bus: the declared locator is
  `sys/bus/i2c/drivers/ina3221/1-0040/...`. Change
  `POWER_DIRECTORY_GLOB` in `src/boardprobe/orin_nano.py` to the address you
  found.
- The rail LABELS in a different order than the channel indices assume: the
  channel table pairs `in1`/`curr1` as `VDD_IN` and `in2`/`curr2` as
  `VDD_CPU_GPU_CV`. If the labels say otherwise, rename the channels to match
  the labels. A rail-power figure attributed to the wrong rail is a wrong
  number with a plausible magnitude, which is the worst kind.

Note that the driver exposes **voltage and current and not power**, so this
package derives power as their product. `RAIL_POWER_PAIRS` in
`src/boardprobe/orin_nano.py` declares which channels pair up.

## 6. Enable the performance counters — UNVERIFIED

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

Expected: `1` or lower for unprivileged per-process counting. If it is `2` or
`3`:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
# to persist:
echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/99-perf.conf
```

Then check that the PMU driver is bound:

```bash
dmesg | grep -i pmuv3
ls /sys/bus/event_source/devices/
```

Expected: an `armv8_pmuv3` or `armv8_cortex_a78` entry. The Orin Nano's CPU
complex is six Cortex-A78AE cores and this package uses the kernel's
GENERALISED hardware events (`PERF_COUNT_HW_CPU_CYCLES` and so on) rather than
raw PMUv3 event numbers, precisely so that the mapping is the kernel's problem
and not this package's.

Lowering `perf_event_paranoid` widens what any unprivileged process on the
board can observe. It is a real change to the board's security posture and
should be made deliberately and reverted when the campaign ends.

## 7. First live sample

```bash
python -m boardprobe sample --backend live
```

Expected: every channel served, every reading stamped `live`, and a
certificate reading `MEASURED on NVIDIA Jetson Orin Nano Developer Kit`
**without** `OFF-TARGET`.

- A channel in the `channels refused` table: go back to the step for its kind.
  The refusal message names the locator and what to check.
- The certificate still reads `OFF-TARGET`: step 3 did not pass. The platform
  attestation, not the readings, is what decides that word.
- A reading with an implausible magnitude — a die temperature of 48123 rather
  than 48.1, a rail power of 23668 rather than 23.7 — is a scaling error of
  exactly 1000, which means the channel's `scale` does not match the raw unit
  the kernel is reporting. Fix the declaration; do not scale the output.

## 8. Record a trace, so the board is not needed again for development

```bash
python - <<'PY'
from pathlib import Path
from boardprobe import ORIN_NANO, LiveSource, Provenance, Trace, TraceSample, read_available

source = LiveSource(ORIN_NANO)
samples = []
for step in range(120):
    readings, refusals = read_available(source)
    for reading in readings:
        samples.append(TraceSample(reading.channel, reading.value, reading.unit,
                                   reading.timestamp_ns))
trace = Trace(
    board_model=ORIN_NANO.model,
    origin=(
        "RECORDED ON AN NVIDIA JETSON ORIN NANO DEVELOPER KIT on <date> by "
        "docs/DEPLOYMENT.md step 8, L4T <version>, idle. These are measurements."
    ),
    content_provenance=Provenance.RECORDED,
    channels=tuple(dict.fromkeys(sample.channel for sample in samples)),
    samples=tuple(samples),
)
Path("fixtures/orin_nano_recorded_trace.json").write_text(trace.to_json())
PY
```

Fill in the date and the L4T version in `origin` **by hand**. It is printed in
every report built from the trace and is the only place a reader learns where
the numbers came from. A trace recorded from hardware replays as `RECORDED`,
so figures derived from it are labelled `REPLAYED` and never `MEASURED`, which
is correct: a replay is not a measurement taken now.

Then replace the two synthetic fixtures in `fixtures/` with this one, and
update `tests/test_fixtures.py`, which currently re-synthesises both committed
fixtures from their declared seeds and compares. A recorded trace cannot be
re-synthesised, so that test becomes a hash or a checked-in sample count
instead. Do not leave a test asserting that a recorded trace equals a
generator's output.

## 9. Run the benchmark

```bash
python -m boardprobe benchmark --repeats 500 --environment live
```

Expected: a `MEASURED on NVIDIA Jetson Orin Nano Developer Kit` headline and
timing figures with resource figures beside them.

Before quoting any figure, read the overhead section of the same output. The
harness's own per-repeat cost on the board will differ from the figures in this
repository, which are x86_64 numbers, and a payload whose cost is within a few
times the overhead is being perturbed by the instrument. The perturbation
curve in `screenshots/harness_overhead.png` shows the shape of that on the
build host; re-make it on the board with
`python examples/harness_overhead.py`.

## 10. Re-label the product

Only after steps 4 to 9 have all been done, and only for the channels actually
read:

- `verified_on_hardware=True` on those channels in
  `src/boardprobe/orin_nano.py`;
- the validation level raised from `Level 3, hardware-pending`, in
  `README.md`, `validation/VALIDATION.md`, the README badge and
  `boardprobe.VALIDATION_LEVEL`;
- every validation script re-run on the board and its output re-committed,
  because the figures in `validation/*_output.txt` are build-host figures and
  must not be left standing beside a Level 4 claim.

`docs/REQUIREMENTS.md` lists the four requirements that are NOT MET without
the board. All four have to move before the level does.
