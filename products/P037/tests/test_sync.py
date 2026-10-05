"""Acquisition state machine: the hand trace, and slip behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from framesync.asm import ASM_32_BITS
from framesync.frames import FrameGeometry, build_stream
from framesync.sync import (
    FrameSynchroniser,
    SyncConfig,
    SyncState,
    acquisition_offsets,
    analyse_slip,
)

# ---------------------------------------------------------------------------
# THE HAND TRACE
#
# Configuration: search_tolerance = 0, lock_tolerance = 3,
#                check_required = 2, flywheel_max = 3.
#
# Rules applied by hand, from the module docstring of framesync.sync:
#   SEARCH   + hit  -> CHECK  (check_count := 1), because check_required > 1
#   SEARCH   + miss -> SEARCH
#   CHECK    + hit  -> check_count += 1; LOCK once check_count >= 2
#   CHECK    + miss -> SEARCH (check_count := 0)
#   LOCK     + hit  -> LOCK (miss_count := 0)
#   LOCK     + miss -> FLYWHEEL (miss_count := 1)
#   FLYWHEEL + hit  -> LOCK (miss_count := 0)
#   FLYWHEEL + miss -> miss_count += 1; SEARCH once miss_count >= 3
#
# Observation sequence (18 observations):
#   miss hit miss hit hit hit miss hit miss miss hit miss miss miss hit hit hit miss
#
#  i  obs    state before  counters before      state after   counters after
#  -- -----  ------------  -------------------  ------------  --------------------
#  0  miss   SEARCH        chk=0 miss=0         SEARCH        chk=0 miss=0
#  1  hit    SEARCH        chk=0 miss=0         CHECK         chk=1 miss=0
#  2  miss   CHECK         chk=1                SEARCH        chk=0 miss=0
#  3  hit    SEARCH        chk=0                CHECK         chk=1 miss=0
#  4  hit    CHECK         chk=1 -> 2 >= 2      LOCK          chk=2 miss=0
#  5  hit    LOCK          miss=0               LOCK          chk=2 miss=0
#  6  miss   LOCK          miss=0               FLYWHEEL      chk=2 miss=1
#  7  hit    FLYWHEEL      miss=1               LOCK          chk=2 miss=0
#  8  miss   LOCK          miss=0               FLYWHEEL      chk=2 miss=1
#  9  miss   FLYWHEEL      miss=1 -> 2 < 3      FLYWHEEL      chk=2 miss=2
# 10  hit    FLYWHEEL      miss=2               LOCK          chk=2 miss=0
# 11  miss   LOCK          miss=0               FLYWHEEL      chk=2 miss=1
# 12  miss   FLYWHEEL      miss=1 -> 2 < 3      FLYWHEEL      chk=2 miss=2
# 13  miss   FLYWHEEL      miss=2 -> 3 >= 3     SEARCH        chk=0 miss=0
# 14  hit    SEARCH        chk=0                CHECK         chk=1 miss=0
# 15  hit    CHECK         chk=1 -> 2 >= 2      LOCK          chk=2 miss=0
# 16  hit    LOCK          miss=0               LOCK          chk=2 miss=0
# 17  miss   LOCK          miss=0               FLYWHEEL      chk=2 miss=1
#
# Frames are delivered in LOCK and FLYWHEEL, so by hand the delivering
# observations are i = 4,5,6,7,8,9,10,11,12 and 15,16,17 -> 12 of 18.
# ---------------------------------------------------------------------------

HAND_TRACE_CONFIG = SyncConfig(
    search_tolerance=0, lock_tolerance=3, check_required=2, flywheel_max=3
)

HAND_TRACE_OBSERVATIONS = [
    "miss", "hit", "miss", "hit", "hit", "hit", "miss", "hit", "miss",
    "miss", "hit", "miss", "miss", "miss", "hit", "hit", "hit", "miss",
]

HAND_TRACE_STATES = [
    "SEARCH", "CHECK", "SEARCH", "CHECK", "LOCK", "LOCK", "FLYWHEEL", "LOCK",
    "FLYWHEEL", "FLYWHEEL", "LOCK", "FLYWHEEL", "FLYWHEEL", "SEARCH", "CHECK",
    "LOCK", "LOCK", "FLYWHEEL",
]

HAND_TRACE_MISS_COUNTS = [0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1]


def test_hand_traced_sequence_reproduced_exactly():
    sm = FrameSynchroniser(HAND_TRACE_CONFIG)
    events = sm.run(HAND_TRACE_OBSERVATIONS)
    assert [e.state_after.value for e in events] == HAND_TRACE_STATES
    assert [e.miss_count for e in events] == HAND_TRACE_MISS_COUNTS
    assert sm.state is SyncState.FLYWHEEL


def test_hand_traced_delivering_count():
    sm = FrameSynchroniser(HAND_TRACE_CONFIG)
    delivering = 0
    for obs in HAND_TRACE_OBSERVATIONS:
        sm.step(obs)
        delivering += int(sm.delivering)
    assert delivering == 12


def test_default_config_single_hit_locks_immediately():
    """check_required = 1 means SEARCH goes straight to LOCK, no CHECK state."""
    sm = FrameSynchroniser()
    assert sm.step("hit").state_after is SyncState.LOCK


def test_flywheel_max_one_declares_loss_on_the_first_miss():
    sm = FrameSynchroniser(SyncConfig(check_required=1, flywheel_max=1))
    sm.step("hit")
    assert sm.step("miss").state_after is SyncState.SEARCH


def test_reset_clears_state_and_history():
    sm = FrameSynchroniser()
    sm.run(["hit", "hit", "miss"])
    sm.reset()
    assert sm.state is SyncState.SEARCH and sm.history == [] and sm.miss_count == 0


def test_config_validation():
    with pytest.raises(ValueError, match="tighter test"):
        SyncConfig(search_tolerance=5, lock_tolerance=2)
    with pytest.raises(ValueError, match="check_required"):
        SyncConfig(check_required=0)
    with pytest.raises(ValueError, match="flywheel_max"):
        SyncConfig(flywheel_max=0)
    with pytest.raises(ValueError, match="non-negative"):
        SyncConfig(search_tolerance=-1)


def test_bad_observation_rejected():
    with pytest.raises(ValueError, match="'hit' or 'miss'"):
        FrameSynchroniser().step("maybe")


def test_clean_stream_locks_and_stays_locked():
    geom = FrameGeometry(data_octets=120, fecf=True)
    rng = np.random.default_rng(2)
    stream, offsets = build_stream(geom, 12, rng)
    found = acquisition_offsets(stream, geom.frame_bits, 0)
    assert np.array_equal(found[: offsets.size], offsets)
    sm = FrameSynchroniser()
    sm.run(["hit"] * 12)
    assert sm.state is SyncState.LOCK


def test_slip_forces_loss_of_sync_and_reacquisition():
    rep = analyse_slip(frame_bits=256, n_frames=14, slip_at_frame=6, slip_bits=3)
    assert rep.frames_before_slip == 6
    # flywheel_max = 4 by default: the first miss delivers in FLYWHEEL and
    # three more misses expire it, so exactly 4 frames go out at the wrong
    # phase before SEARCH is re-entered.
    assert rep.frames_delivered_at_wrong_phase == 4
    assert rep.reacquisition_frames == 4
    assert rep.final_state is SyncState.SEARCH


def test_slip_of_a_deleted_bit_behaves_the_same_way():
    rep = analyse_slip(frame_bits=256, n_frames=14, slip_at_frame=6, slip_bits=-2)
    assert rep.frames_delivered_at_wrong_phase == 4
    assert rep.final_state is SyncState.SEARCH


def test_slip_validation():
    with pytest.raises(ValueError, match="slip_bits must be non-zero"):
        analyse_slip(frame_bits=64, n_frames=5, slip_at_frame=2, slip_bits=0)
    with pytest.raises(ValueError, match="slip_at_frame"):
        analyse_slip(frame_bits=64, n_frames=5, slip_at_frame=9, slip_bits=1)
    with pytest.raises(ValueError, match="n_frames must be at least 3"):
        analyse_slip(frame_bits=64, n_frames=2, slip_at_frame=1, slip_bits=1)
    with pytest.raises(ValueError, match="frame_bits must be positive"):
        analyse_slip(frame_bits=0, n_frames=5, slip_at_frame=1, slip_bits=1)


def test_acquisition_offsets_drops_wrongly_spaced_hits():
    geom = FrameGeometry(data_octets=60, fecf=False, asm_bits=32)
    rng = np.random.default_rng(8)
    stream, _ = build_stream(geom, 6, rng)
    # Plant a spurious exact marker at an off-period offset; the spacing
    # filter must discard it.
    stream[100:132] = ASM_32_BITS
    kept = acquisition_offsets(stream, geom.frame_bits, 0)
    assert 100 not in kept.tolist()
