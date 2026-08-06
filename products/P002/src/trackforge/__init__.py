"""TrackForge: pointing-acquisition-tracking (PAT) simulation suite for optical links.

Modules
-------
- ``trackforge.scan``     : acquisition scan-pattern generators and acquisition statistics
- ``trackforge.dynamics`` : 2-axis gimbal dynamics, jitter synthesis, sensor model
- ``trackforge.control``  : PID / LQR controllers and benchmark harness
- ``trackforge.reacq``    : reacquisition policies (scripted baselines + tabular Q-learning)
- ``trackforge.sim``      : end-to-end episode simulator, scenario config, metrics

All angles are in radians, times in seconds, torques in N m unless stated otherwise.

This software is research-grade. It is not flight-qualified, not certified, and not
approved for operational aerospace use.
"""

__version__ = "0.1.0"

from trackforge import control, dynamics, reacq, scan, sim

__all__ = ["scan", "dynamics", "control", "reacq", "sim", "__version__"]
