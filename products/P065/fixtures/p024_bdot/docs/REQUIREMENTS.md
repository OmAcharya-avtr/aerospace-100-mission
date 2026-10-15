# Requirements for the B-dot detumble loop of P024 DetumbleSim

**What this document is.** A list of statements about the B-dot detumble loop
implemented in P024 DetumbleSim, written **by hand for the hazardtrace worked
example** in the declaration format `hazardtrace` parses by default
(`REQ-NNN` at the start of a line).

**What this document is not.** It is not a document P024 DetumbleSim ships.
P024 has no requirements document, which is itself one of the findings the
worked example reports: a system-level constraint there can be traced to source
code and to tests, but not to a requirement, because no requirement was ever
written down. Nothing here is agreed with anybody, carries a verification
method, a review state, a baseline or an approval.

Each statement below was read out of P024's own source and docstrings. Two
identifiers the worked example cites deliberately do **not** appear here, and
that is the point: `hazardtrace` reports them as unresolved rather than letting
the analysis claim a requirement that does not exist.

## Control law

### REQ-001 B-dot command

The detumble controller shall command a magnetic dipole
`m = -k dB/dt` [A m^2], with `k` in A m^2 s T^-1, opposing the measured rate of
change of the body-frame magnetic field.

### REQ-002 Derivative estimation

The controller shall estimate `dB/dt` by backward difference of successive
magnetometer samples, `dB/dt[i] = (B[i] - B[i-1]) / dt`, with `dt` the control
interval [s].

### REQ-003 First sample

On the first control step, where no previous magnetometer sample exists, the
controller shall command a zero dipole.

### REQ-004 Zero-order hold

The commanded dipole shall be held constant over one control interval and the
attitude dynamics advanced within it.

## Actuation

### REQ-005 Per-axis dipole limit

The commanded dipole shall be clipped per axis to the coil limit
`max_dipole_am2` [A m^2]; the achievable set is a box, not a ball.

### REQ-006 Applied torque

The applied torque shall be `L = m x B` [N m], which is identically
perpendicular to the instantaneous field direction.

## Sensing

### REQ-007 Magnetometer sampling

The magnetometer shall be sampled once per control interval, at the same
instant the control law is evaluated.

### REQ-008 Magnetometer noise model

Magnetometer noise shall be modelled as zero-mean Gaussian per axis with a
declared standard deviation [T], drawn from a seeded stream so that a run is
reproducible.

## Mode management

### REQ-009 Detumble completion threshold

Detumble shall be declared complete at the first time the body-rate magnitude
`|omega|` falls to the declared target rate [rad/s], default 0.1 deg/s.

### REQ-010 No-crossing reporting

Where the target rate is not reached inside the simulated span, the detumble
time shall be reported as not-a-number rather than as the span length.

## Stated model limits

### REQ-011 Field model

The geomagnetic field shall be the tilted centred dipole, the degree-1
truncation of IGRF-14, and the documentation shall state its measured error
against IGRF-14.

### REQ-012 Energy monotonicity

Under the ideal unsaturated B-dot law the rotational kinetic energy shall not
increase on any step, and the documentation shall state that the momentum
magnitude is **not** monotone for an asymmetric inertia.

### REQ-013 Quaternion normalisation

The attitude quaternion shall be renormalised once per control interval and the
largest pre-normalisation norm error reported.
