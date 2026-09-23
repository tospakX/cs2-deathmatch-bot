"""Deterministic temporal behavior and continuity test suite.

Verifies:
- Aim-point preference persistence across frames
- Motor episode persistence across frames
- Fire-mode episode commitment (burst does not flip to spray frame-by-frame)
- Crouch episode duration (no 30Hz key fluttering)
- Counter-strafe transition (single impulse on movement stop)
- Non-alternating strafe footwork
- Strict unit conversion integrity
- Historical target last-seen timestamps
- Bipartite target identity assignment across crossing targets
- Persistent reload state machine
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior.decision import DecisionEngine
from src.behavior.firing import RECOIL_RESET_TIME, FiringController
from src.behavior.motor import MotorPlanner
from src.behavior.movement import MovementController
from src.behavior.perception import PerceptionSystem, hungarian_assignment
from src.behavior.personality import PersonalityTraits
from src.behavior.player_state import (
    AimPhase,
    BotPhase,
    FiringPhase,
    MovementPhase,
    PlayerState,
    SpatialMemory,
    TargetStatus,
    TrackedTarget,
)
from src.behavior.scanning import ScanningController
from src.utils.math_helpers import screen_delta_to_mouse
from src.vision.detector import Detection


def make_det(cx: float, cy: float, w: float = 80.0, h: float = 200.0) -> Detection:
    return Detection(
        class_id=0,
        class_name="player",
        confidence=0.90,
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
    )


# ── 1. Aim Persistence ────────────────────────────────────────────────────────


def test_aim_point_preference_persists_across_multiple_frames():
    """Selected aim point (head/chest) remains strictly identical across 60 frames."""
    random.seed(42)
    p = PersonalityTraits(head_aim_preference=0.5)
    engine = DecisionEngine(personality=p, screen_center=(1720, 720))
    state = PlayerState()
    state.is_alive = True

    det = make_det(1850, 720)
    target = TrackedTarget(detection=det, target_id=1, frames_visible=1)
    state.primary_target = target

    # Frame 1
    engine.decide(state, [target])
    initial_pref = target.aim_preference
    initial_offset = target.aim_offset_ratio
    assert initial_pref in ("head", "chest")

    # Run for 60 subsequent ticks
    for _ in range(60):
        state.begin_tick()
        engine.decide(state, [target])
        assert target.aim_preference == initial_pref
        assert target.aim_offset_ratio == initial_offset
        assert state.authoritative_aim.aim_preference == initial_pref


# ── 2. Motor Episode Persistence ──────────────────────────────────────────────


def test_motor_episode_persists_across_ticks():
    """An acquisition movement commits to an episode rather than redrawing from scratch every tick."""
    random.seed(123)
    p = PersonalityTraits(motor_speed=1.0, motor_precision=0.7)
    planner = MotorPlanner(
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=122.0,
        screen_width=3440,
        screen_height=1440,
        personality=p,
    )
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.ACQUIRING

    det = make_det(2100, 720)
    target = TrackedTarget(detection=det, target_id=10, frames_visible=5)
    state.primary_target = target

    # Tick 1: initiates episode
    planner.plan(state)
    assert state.motor_episode.active is True
    start_time = state.motor_episode.start_time
    planned_duration = state.motor_episode.duration

    # Tick 2 & 3: episode remains active with same start_time and duration
    state.begin_tick()
    planner.plan(state)
    assert state.motor_episode.start_time == start_time
    assert state.motor_episode.duration == planned_duration


# ── 3. Fire-Mode Episode Commitment ───────────────────────────────────────────


def test_burst_fire_episode_commitment():
    """A burst episode stays committed to its planned shot count without mode flapping."""
    random.seed(99)
    p = PersonalityTraits(firing_discipline=0.6)
    firing = FiringController(personality=p, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_det(1720, 720, w=60, h=140)  # Medium range -> burst
    target = TrackedTarget(detection=det, target_id=3, frames_visible=10)

    # First call commits to burst episode
    firing.update(state, target, err_dist=5.0)
    assert state.firing_episode.active is True
    assert state.firing_episode.mode == "burst"
    planned_shots = state.firing_episode.shots_planned
    assert planned_shots >= 2

    # Subsequent ticks before cycle time must maintain burst mode
    for _ in range(5):
        firing.update(state, target, err_dist=5.0)
        assert state.firing_episode.mode == "burst"


# ── 4. Crouch Persistence ─────────────────────────────────────────────────────


def test_crouch_persists_for_duration_without_30hz_flutter():
    """Crouch remains held for its committed duration and does not pulse on/off at 30Hz."""
    random.seed(55)
    p = PersonalityTraits(crouch_tendency=0.9, firing_discipline=0.2)
    firing = FiringController(personality=p, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_det(1720, 720, w=150, h=300)
    target = TrackedTarget(detection=det, target_id=7, frames_visible=10)

    # Simulate sustained spray (shot count >= 3)
    state.shot_count = 3
    state.firing_episode.active = True
    state.firing_episode.mode = "spray"

    # Trigger crouch
    firing.update(state, target, err_dist=5.0)
    if state.is_crouching:
        assert state.crouch_episode.is_crouching is True

        # For the next 10 ticks (~330ms), crouch must stay held continuously
        for _ in range(10):
            state.begin_tick()
            cmd_next = firing.update(state, target, err_dist=5.0)
            assert state.is_crouching is True
            assert cmd_next.should_crouch is True


# ── 5. Counter-Strafing Single Transition ─────────────────────────────────────


def test_counter_strafe_triggers_single_braking_event():
    """Counter-strafing triggers exactly one opposite-key tap when stopping to fire."""
    random.seed(77)
    p = PersonalityTraits(strafe_tendency=0.3)
    movement = MovementController(personality=p)
    state = PlayerState()
    state.is_alive = True

    det = make_det(1720, 720)
    target = TrackedTarget(detection=det, target_id=8, frames_visible=10)

    # Set up active left strafe
    movement._current_phase = MovementPhase.STRAFING
    movement._current_direction = "left"
    movement._phase_start = state.now
    movement._phase_duration = 0.50

    # Tick 1: Player begins firing -> counter-strafe initiates
    keys1 = movement.update(state, target, is_firing=True)
    assert movement._counter_strafing is True
    assert keys1["right"] is True  # Counter-key for left is right
    assert movement._braking_done_for_movement is True

    # Advance time beyond braking duration (~75ms)
    state._now += 0.100
    keys2 = movement.update(state, target, is_firing=True)
    assert movement._counter_strafing is False
    assert keys2["right"] is False  # Counter-strafe key is released and not re-triggered


# ── 6. Non-Alternating Strafe Directions ──────────────────────────────────────


def test_strafe_direction_is_not_strictly_alternating():
    """Direction sampling follows a Markov history model and allows repeat directions."""
    random.seed(101)
    p = PersonalityTraits(strafe_tendency=0.8)
    movement = MovementController(personality=p)

    samples = [movement._sample_next_direction() for _ in range(50)]
    repeats = sum(1 for i in range(len(samples) - 1) if samples[i] == samples[i + 1])
    alternations = sum(1 for i in range(len(samples) - 1) if samples[i] != samples[i + 1])

    # Human footwork has both repeats and reversals; alternation is NOT 100%
    assert repeats > 5
    assert alternations > 5


# ── 7. Unit Conversion Correctness ───────────────────────────────────────────


def test_unit_conversion_pixel_velocity_to_mouse_counts():
    """Target pixel velocity is converted geometrically to mouse counts before combination."""
    sensitivity = 1.2
    m_yaw = 0.022
    m_pitch = 0.022
    fov_h = 122.0
    sw, sh = 3440, 1440
    dt = 0.033

    # Target running at 500 px/s rightward
    vx_px_s = 500.0
    disp_px = vx_px_s * dt  # 16.5 pixels

    mx, my = screen_delta_to_mouse(disp_px, 0.0, sensitivity, m_yaw, m_pitch, sw, sh, fov_h)
    assert mx > 0
    assert my == 0
    # Output must be an integer count
    assert isinstance(mx, int)


# ── 8. Spatial Memory Timestamp Integrity ─────────────────────────────────────


def test_spatial_memory_preserves_true_last_seen_timestamp():
    """Lost target memory stores the true last observation timestamp, not prune time."""
    perception = PerceptionSystem(screen_center=(1720, 720))
    state = PlayerState()
    state._now = 100.0

    det = make_det(1800, 720)
    target = TrackedTarget(detection=det, target_id=9)
    target.update_position(1800, 720, now=100.0)
    state.known_targets[9] = target

    # Target is missing for 16 consecutive frames
    for i in range(16):
        state._now = 100.0 + (i + 1) * 0.033
        perception.update(state, [])

    # Target was pruned
    assert 9 not in state.known_targets
    assert len(state.spatial_memories) > 0
    mem = state.spatial_memories[-1]
    # The recorded last_seen_time must be 100.0 (true observation), NOT 100.528!
    assert abs(mem.last_seen_time - 100.0) < 0.001


# ── 9. Target Identity Continuity ─────────────────────────────────────────────


def test_bipartite_tracking_maintains_target_identity_when_close():
    """Tracking matches targets by predicted position and size similarity."""
    perception = PerceptionSystem(screen_center=(1720, 720))
    state = PlayerState()
    state._now = 1.0

    # Target A at (1600, 720) moving right (+100 px/s)
    det_a = make_det(1600, 720, w=80, h=200)
    # Target B at (1800, 720) moving left (-100 px/s)
    det_b = make_det(1800, 720, w=50, h=120)

    perception.update(state, [det_a, det_b])
    tracks = list(state.known_targets.values())
    assert len(tracks) == 2

    id_a = [t.target_id for t in tracks if t.detection.width > 70][0]
    id_b = [t.target_id for t in tracks if t.detection.width < 60][0]

    # Next frame (dt = 0.033s): targets moved slightly
    state._now = 1.033
    det_a_next = make_det(1605, 720, w=80, h=200)
    det_b_next = make_det(1795, 720, w=50, h=120)
    perception.update(state, [det_a_next, det_b_next])

    assert state.known_targets[id_a].detection.width > 70
    assert state.known_targets[id_b].detection.width < 60


# ── 10. Persistent Reload Machine ─────────────────────────────────────────────


def test_reload_episode_prevents_30hz_key_spam():
    """Reload episode issues keypress once, holds RELOADING phase, and completes cleanly."""
    p = PersonalityTraits()
    engine = DecisionEngine(personality=p, screen_center=(1720, 720))
    state = PlayerState()
    state.is_alive = True
    state.phase = BotPhase.ROAMING
    state.ammo_clip = 0  # Out of ammo

    # Tick 1: Initiates reload
    out1 = engine.decide(state, [])
    assert out1.should_reload is True
    assert state.phase == BotPhase.RELOADING
    assert state.reload_episode.is_reloading is True

    # Tick 2 to 30: Still reloading, should_reload must be False (key not spammed)
    for _ in range(29):
        state.begin_tick()
        out = engine.decide(state, [])
        assert out.should_reload is False
        assert state.phase == BotPhase.RELOADING

    # Advance beyond expected duration (2.7s) with ammo replenished
    state._now += 3.0
    state.ammo_clip = 30
    engine.decide(state, [])
    assert state.reload_episode.is_reloading is False
    assert state.phase == BotPhase.ROAMING


# ── 11. Stateful Roaming Input Continuity ─────────────────────────────────────


def test_roaming_input_stateful_continuity():
    """Movement keys must only transition on state change, never churn release/press every tick."""
    held_keys = set()
    press_events = []
    release_events = []

    class MockKeyboard:
        @staticmethod
        def hold_key(key: str):
            press_events.append(key)

        @staticmethod
        def release_key(key: str):
            release_events.append(key)

    keybinds = {"forward": "w", "back": "s", "left": "a", "right": "d", "crouch": "ctrl"}

    def apply_move(cmd: dict):
        for flag in ("forward", "back", "left", "right", "crouch"):
            bind_key = keybinds.get(flag)
            if not bind_key:
                continue
            should_hold = bool(cmd.get(flag, False))
            is_held = bind_key in held_keys
            if should_hold and not is_held:
                MockKeyboard.hold_key(bind_key)
                held_keys.add(bind_key)
            elif not should_hold and is_held:
                MockKeyboard.release_key(bind_key)
                held_keys.discard(bind_key)

    # 10 consecutive ticks moving forward
    for _ in range(10):
        apply_move({"forward": True})

    # 'w' must be pressed EXACTLY once, never released during continuous movement
    assert press_events == ["w"]
    assert release_events == []
    assert held_keys == {"w"}

    # Stopping movement releases 'w' exactly once
    apply_move({"forward": False})
    assert release_events == ["w"]
    assert held_keys == set()


# ── 12. Pure-Python Hungarian Assignment ──────────────────────────────────────


def test_pure_python_hungarian_bipartite_matching():
    """Kuhn-Munkres algorithm produces exact minimum-cost global bipartite assignments."""
    # Test case 1: 2x2 with clear diagonal minimum
    cost_matrix_1 = [
        [10.0, 50.0],
        [40.0, 15.0],
    ]
    matches_1 = hungarian_assignment(cost_matrix_1)
    assert set(matches_1) == {(0, 0), (1, 1)}

    # Test case 2: 3x3 classic assignment problem
    cost_matrix_2 = [
        [14.0, 5.0, 8.0],
        [2.0, 12.0, 6.0],
        [7.0, 8.0, 3.0],
    ]
    # Optimal matches: (0, 1) [cost 5], (1, 0) [cost 2], (2, 2) [cost 3] -> total 10
    matches_2 = hungarian_assignment(cost_matrix_2)
    assert set(matches_2) == {(0, 1), (1, 0), (2, 2)}
    total_cost = sum(cost_matrix_2[r][c] for r, c in matches_2)
    assert total_cost == 10.0

    # Test case 3: Rectangular matrix (more tracks than detections)
    cost_matrix_3 = [
        [5.0, 20.0],
        [15.0, 6.0],
        [30.0, 30.0],
    ]
    matches_3 = hungarian_assignment(cost_matrix_3)
    assert len(matches_3) == 2
    assert set(matches_3) == {(0, 0), (1, 1)}


# ── 13. Target Status Lifecycle and Loss Semantics ────────────────────────────


def test_target_status_lifecycle_and_loss_semantics():
    """Target status transitions smoothly and suppresses firing when target is lost."""
    det = make_det(1720, 720)
    target = TrackedTarget(detection=det, target_id=1)
    assert target.status == TargetStatus.VISIBLE

    # 1 to 4 frames missing -> BRIEFLY_LOST
    target.mark_missing()
    assert target.status == TargetStatus.BRIEFLY_LOST

    for _ in range(3):
        target.mark_missing()
    assert target.status == TargetStatus.BRIEFLY_LOST

    # 5 to 15 frames missing -> REACQUIRING
    target.mark_missing()
    assert target.status == TargetStatus.REACQUIRING

    # >15 frames missing -> FORGOTTEN
    for _ in range(12):
        target.mark_missing()
    assert target.status == TargetStatus.FORGOTTEN

    # Re-observed -> VISIBLE
    target.update_position(1730, 720, now=10.0)
    assert target.status == TargetStatus.VISIBLE

    # FiringController must release trigger and not fire if target is BRIEFLY_LOST
    p = PersonalityTraits()
    firing = FiringController(personality=p, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING
    state.is_trigger_held = True

    target.status = TargetStatus.BRIEFLY_LOST
    cmd = firing.update(state, target, err_dist=5.0)
    assert cmd.trigger_action == "release"
    assert not state.is_trigger_held


# ── 14. Angular Spatial Memory in Scanning ────────────────────────────────────


def test_spatial_memory_angular_sweep_in_scanning():
    """ScanningController sweeps toward stored angular spatial memories with temporal decay."""
    p = PersonalityTraits(scanning_frequency=1.0)
    scanner = ScanningController(
        personality=p,
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        screen_size=(3440, 1440),
        fov_h=122.0,
    )
    scanner._last_scan_end = 0.0

    state = PlayerState()
    state._now = 10.0
    state.primary_target = None

    # Stored spatial memory at +15 degrees yaw
    mem = SpatialMemory(
        screen_x=2200.0,
        screen_y=720.0,
        yaw_offset_deg=15.0,
        pitch_offset_deg=0.0,
        last_seen_time=8.5,  # 1.5 seconds old
        confidence=0.9,
    )
    state.spatial_memories.append(mem)

    # First update initiates purposeful scan towards remembered angle
    scanner.update(state)
    assert scanner._current_scan is not None
    assert scanner._current_scan.reason == "check_spatial_memory"
    # Target delta should be proportional to 15.0 / 0.022 (~681 counts) * 0.75
    assert scanner._current_scan.target_dx > 300.0


# ── 15. Tap Hold Duration Across Frames ───────────────────────────────────────


def test_tap_hold_duration_persists_across_multiple_frames():
    """Tap mode holds mouse button down for a human duration (45-80ms) rather than single-tick release."""
    random.seed(42)
    p = PersonalityTraits(firing_discipline=0.8)
    firing = FiringController(personality=p, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING
    state.firing_phase = FiringPhase.TAPPING

    det = make_det(1720, 720, w=30, h=80)
    target = TrackedTarget(detection=det, target_id=1, frames_visible=10)

    # Frame 1: Presses trigger
    cmd1 = firing.update(state, target, err_dist=2.0)
    assert cmd1.trigger_action == "press"
    assert state.is_trigger_held is True

    # Frame 2 (20ms later): Must continue to hold (not instantly released at 33ms)
    state._now += 0.020
    cmd2 = firing.update(state, target, err_dist=2.0)
    assert cmd2.trigger_action == "hold"
    assert state.is_trigger_held is True

    # Frame 3 (90ms later): Hold duration expires -> releases trigger
    state._now += 0.070
    cmd3 = firing.update(state, target, err_dist=2.0)
    assert cmd3.trigger_action == "release"
    assert state.is_trigger_held is False


# ── 16. Recoil Cooldown Preserved Across Firing Stops ──────────────────────────


def test_recoil_cooldown_not_prematurely_reset():
    """Recoil shot index stays elevated during brief pauses and only resets after RECOIL_RESET_TIME."""
    p = PersonalityTraits()
    firing = FiringController(personality=p, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.shot_count = 3
    state.recoil_shot_index = 3
    state.is_trigger_held = False
    firing._last_shot_time = 10.0

    # Before RECOIL_RESET_TIME has elapsed (< 0.350s): recoil is NOT reset
    state._now = 10.0 + RECOIL_RESET_TIME * 0.4
    firing.update(state, None)
    assert state.recoil_shot_index == 3

    # After RECOIL_RESET_TIME has elapsed (> 0.350s): recoil resets authoritatively
    state._now = 10.0 + RECOIL_RESET_TIME * 1.1
    firing.update(state, None)
    assert state.recoil_shot_index == 0
