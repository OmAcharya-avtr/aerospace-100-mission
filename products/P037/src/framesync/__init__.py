"""framesync -- a frame-level link performance harness for CCSDS telemetry.

What this is: a harness that measures how a CCSDS telemetry *link* behaves at
the frame level -- attached sync marker correlation detection, the sync
acquisition / flywheel / loss state machine, slip and false-sync behaviour,
and frame-error rate against Eb/N0 for an uncoded link, an RS(255,223) link
and a convolutionally coded link.

What this is not: it is not a codec and it is not a packet parser.
``ccsdspy`` and ``spacepackets`` already do CCSDS packet and framing work
well; ``reedsolo`` and ``commpy`` are the mature codec libraries. This
package sits above them and measures frame-level performance.

Conventions are stated once, in :mod:`framesync.channel`: ``ebn0_db`` is
always the energy per **information** bit over the noise density, in dB, and
a code of rate R puts Es/N0 = R * Eb/N0 on the channel.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use.
"""

from __future__ import annotations

from .asm import (
    ASM_32_BITS,
    ASM_32_HEX,
    asm_autocorrelation,
    detect_asm,
    expected_false_syncs,
    false_sync_probability,
    hamming_distances,
)
from .channel import awgn_bpsk_samples, bpsk_ber, bsc_flip, ebn0_to_esn0_db, qfunc
from .conv import CCSDS_G1, CCSDS_G2, ConvCode, free_distance
from .crc import crc16, crc16_batch
from .fer import (
    FerPoint,
    binomial_stderr,
    coding_gain_db,
    ebn0_for_target,
    measure_conv_fer,
    measure_rs_fer,
    measure_uncoded_fer,
    n_frames_for_target,
    uncoded_fer,
)
from .frames import FrameGeometry, build_stream
from .rs import (
    RS_E,
    RS_K,
    RS_N,
    RS_RATE,
    ReedSolomonLink,
    codeword_failure_probability,
    rs_frame_error_rate,
    rs_output_bit_error_rate,
    symbol_error_probability,
)
from .sync import (
    FrameSynchroniser,
    SlipReport,
    SyncConfig,
    SyncEvent,
    SyncState,
    acquisition_offsets,
    analyse_slip,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # asm
    "ASM_32_HEX",
    "ASM_32_BITS",
    "hamming_distances",
    "detect_asm",
    "false_sync_probability",
    "expected_false_syncs",
    "asm_autocorrelation",
    # channel
    "qfunc",
    "bpsk_ber",
    "ebn0_to_esn0_db",
    "bsc_flip",
    "awgn_bpsk_samples",
    # crc
    "crc16",
    "crc16_batch",
    # frames
    "FrameGeometry",
    "build_stream",
    # rs
    "RS_N",
    "RS_K",
    "RS_E",
    "RS_RATE",
    "symbol_error_probability",
    "codeword_failure_probability",
    "rs_frame_error_rate",
    "rs_output_bit_error_rate",
    "ReedSolomonLink",
    # conv
    "CCSDS_G1",
    "CCSDS_G2",
    "ConvCode",
    "free_distance",
    # sync
    "SyncState",
    "SyncConfig",
    "SyncEvent",
    "FrameSynchroniser",
    "analyse_slip",
    "SlipReport",
    "acquisition_offsets",
    # fer
    "uncoded_fer",
    "binomial_stderr",
    "n_frames_for_target",
    "FerPoint",
    "measure_uncoded_fer",
    "measure_rs_fer",
    "measure_conv_fer",
    "ebn0_for_target",
    "coding_gain_db",
]
