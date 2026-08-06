"""centroidnet: optical spot centroid estimation for pointing/tracking sensors.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use.
"""

from centroidnet.baselines import cog_centroid, quadcell_centroid
from centroidnet.generator import generate_spots, snr_estimate, spot_image
from centroidnet.ml import MLCentroider

__all__ = [
    "generate_spots",
    "spot_image",
    "snr_estimate",
    "cog_centroid",
    "quadcell_centroid",
    "MLCentroider",
]

__version__ = "0.1.0"
