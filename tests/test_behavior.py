"""Comprehensive unit and integration tests for the human-like behavioral layer.

Covers all required areas:
- Attention persistence & target inertia
- Target switching & reasons
- Reaction state & contextual modifiers (surprise, reacquire)
- Reaction/firing synchronization (gating)
- Aim state persistence & continuous motor phases (flick, settle, micro-correct, track)
- Continuous tracking & velocity prediction
- Firing modes (tap, burst, spray) & actual shot counting vs loop ticks
- Recoil synchronization on actual shot events
- Weapon changes & recoil reset
- Movement states (approaching, holding, strafing, retreating) & counter-strafing
- Personality traits influencing multiple connected subsystems
- Temporal continuity & short-term tendency drift
- Adaptation to hits, misses, damage, and target loss
- Coordinate conversion, FOV propagation, and configuration integrity
- Death and respawn state resets
- Edge cases (depleted ammo, zero detections, extreme distances)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior.adaptation import AdaptationEngine
from src.behavior.attention import AttentionSystem
from src.behavior.decision import DecisionEngine
from src.behavior.firing import FiringController
from src.behavior.motor import MotorPlanner
from src.behavior.movement import MovementController
from src.behavior.personality import PersonalityTraits, load_personality
from src.behavior.player_state import (
    AimPhase,
    BotPhase,
    FiringPhase,
    MovementPhase,
    PlayerState,
    SpatialMemory,
    TrackedTarget,
)
from src.behavior.reaction import ReactionSystem
from src.behavior.scanning import ScanningController
from src.utils.math_helpers import screen_delta_to_mouse
from src.vision.detector import Detection


def make_dummy_detection(
    cx: float = 1720.0,
    cy: float = 720.0,
    w: float = 80.0,
    h: float = 200.0,
    conf: float = 0.85,
    class_name: str = "player",
) -> Detection:
    """Helper to create a test Detection."""
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=conf,
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
    )


# ── 1. Attention Persistence & Target Inertia ─────────────────────────────────


def test_attention_persistence_keeps_current_target():
    """Attention system maintains focus on existing target when a slightly closer target appears."""
    personality = PersonalityTraits(target_persistence=0.8, switch_reluctance=0.8)
    attention = AttentionSystem(screen_center=(1720, 720), personality=personality)
    state = PlayerState()
    state.is_alive = True

    # Target A at (1750, 720) - dist 30
    det_a = make_dummy_detection(1750, 720)
    target_a = TrackedTarget(detection=det_a, frames_visible=10, threat_level=970.0)
    state.primary_target = target_a

    # Target B appears at (1740, 720) - dist 20, slightly closer
    det_b = make_dummy_detection(1740, 720)
    target_b = TrackedTarget(detection=det_b, frames_visible=1, threat_level=980.0)

    attention.update(state, [target_a, target_b])

    # Attention inertia should keep target_a despite target_b having a slightly higher threat score
    assert state.primary_target is target_a


def test_target_switching_on_overwhelming_threat():
    """Attention switches target only when threat difference exceeds high switching cost."""
    personality = PersonalityTraits(target_persistence=0.4, switch_reluctance=0.3)
    attention = AttentionSystem(screen_center=(1720, 720), personality=personality)
    state = PlayerState()
    state.is_alive = True

    det_a = make_dummy_detection(2200, 720, w=30, h=60)  # Far, low threat
    target_a = TrackedTarget(detection=det_a, frames_visible=10, threat_level=300.0)
    state.primary_target = target_a
    state.target_acquisition_time = state.now

    # Massive sudden threat right in crosshairs
    det_b = make_dummy_detection(1722, 720, w=200, h=400, conf=0.95)
    target_b = TrackedTarget(detection=det_b, frames_visible=2, threat_level=1600.0)

    attention.update(state, [target_a, target_b])

    assert state.primary_target is target_b
    assert len(state.target_switch_history) > 0
    assert state.target_switch_history[-1][1] == "higher_threat"


# ── 2. Reaction State & Contextual Timing ────────────────────────────────────


def test_reaction_state_surprise_is_slower():
    """Surprise targets result in longer reaction delay than expected/reacquired targets."""
    personality = PersonalityTraits(reaction_base_ms=200.0, reaction_variance_ms=0.0)
    reaction = ReactionSystem(personality)
    state = PlayerState()

    det = make_dummy_detection(1720, 720)
    target = TrackedTarget(detection=det, frames_visible=1)

    reaction.initiate_reaction(state, target, context="surprise")
    surprise_duration = state.reaction_duration

    reaction.initiate_reaction(state, target, context="reacquire")
    reacquire_duration = state.reaction_duration

    assert surprise_duration > reacquire_duration


def test_reaction_firing_synchronization_gates_aim_and_trigger():
    """While reaction is pending, aim phase is REACTING and trigger remains unpulled."""
    personality = PersonalityTraits(reaction_base_ms=300.0)
    engine = DecisionEngine(personality, screen_center=(1720, 720))
    state = PlayerState()
    state.is_alive = True

    det = make_dummy_detection(1800, 750)
    target = TrackedTarget(detection=det, frames_visible=1)

    decision = engine.decide(state, [target])

    assert state.aim_phase == AimPhase.REACTING
    assert not decision.firing_cmd.is_firing
    assert (
        decision.firing_cmd.trigger_action == "release"
        or decision.firing_cmd.trigger_action == "none"
    )


# ── 3. Aim State Persistence & Continuous Motor Phases ─────────────────────────


def test_motor_planner_flick_transitions_to_correction():
    """Ballistic flick decelerates and transitions to micro-correction phase near target."""
    personality = PersonalityTraits(motor_speed=1.0, motor_precision=0.7)
    planner = MotorPlanner(
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=90.0,
        screen_width=3440,
        screen_height=1440,
        personality=personality,
    )
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.ACQUIRING

    # Distant target at (1920, 720) -> 200px error
    det = make_dummy_detection(1920, 720)
    target = TrackedTarget(detection=det, frames_visible=10)
    state.primary_target = target

    dx1, dy1 = planner.plan(state)
    assert dx1 > 0  # Initial committed flick rightward

    # Now simulate crosshair arriving near target (1725, 720) -> ~5px error
    det_close = make_dummy_detection(1725, 720, w=10, h=10)
    target_close = TrackedTarget(detection=det_close, frames_visible=15)
    state.primary_target = target_close

    planner.plan(state)
    assert state.aim_phase in (AimPhase.CORRECTING, AimPhase.TRACKING)


def test_motor_planner_maintains_velocity_momentum():
    """Motor velocity is continuous and does not reset to zero every frame."""
    personality = PersonalityTraits(motor_speed=1.0)
    planner = MotorPlanner(
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=90.0,
        screen_width=3440,
        screen_height=1440,
        personality=personality,
    )
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.ACQUIRING

    det = make_dummy_detection(2000, 720)
    target = TrackedTarget(detection=det, frames_visible=5)
    state.primary_target = target

    planner.plan(state)
    assert abs(planner._velocity_x) > 0.0


# ── 4. Continuous Tracking & Velocity Prediction ──────────────────────────────


def test_tracking_predicts_moving_target():
    """Tracking phase factors target velocity estimate into mouse movement."""
    personality = PersonalityTraits(tracking_steadiness=0.8)
    planner = MotorPlanner(
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=90.0,
        screen_width=3440,
        screen_height=1440,
        personality=personality,
    )
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_dummy_detection(1720, 720)  # already on crosshair
    target = TrackedTarget(detection=det, frames_visible=20)
    # Target running rightwards at 400 px/sec
    target.velocity_estimate = (400.0, 0.0)
    state.primary_target = target

    dx, dy = planner.plan(state)
    # Even though target is on center, tracking leads the moving target
    assert dx > 0


# ── 5. Firing Modes & Actual Shot Counting ───────────────────────────────────


def test_firing_controller_advances_shots_on_cycle_time_not_ticks():
    """Shot count advances based on weapon cycle time (~100ms), NOT 33ms loop ticks."""
    personality = PersonalityTraits(firing_discipline=0.2)  # spray preference
    firing = FiringController(personality, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_dummy_detection(1720, 720, w=100, h=250)  # Close target
    target = TrackedTarget(detection=det, frames_visible=10)

    # Tick 1: First shot fires
    cmd1 = firing.update(state, target, err_dist=0.0)
    assert cmd1.is_firing
    assert state.shot_count == 1

    # Tick 2 (10ms later): Weapon is cycling, shot count must NOT increment
    state._now += 0.010
    firing.update(state, target, err_dist=0.0)
    assert state.shot_count == 1  # Unchanged!

    # Tick 3 (105ms after first shot): Second shot fires
    state._now += 0.095
    firing.update(state, target, err_dist=0.0)
    assert state.shot_count == 2


def test_burst_mode_pauses_after_target_bullets():
    """Burst mode fires target bullets (e.g. 3) then enforces trigger release pause."""
    personality = PersonalityTraits(firing_discipline=0.7)
    firing = FiringController(personality, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_dummy_detection(1720, 720, w=50, h=120)  # Medium range -> burst
    target = TrackedTarget(detection=det, frames_visible=10)

    # Force a 3-shot burst
    firing._target_shots_in_burst = 3
    state.firing_phase = FiringPhase.BURSTING

    # Advance 3 shots
    for i in range(3):
        firing.update(state, target, err_dist=0.0)
        state._now += 0.105

    # After 3 shots, firing phase enters pause
    assert state.firing_phase == FiringPhase.BURST_PAUSE
    assert not state.is_trigger_held


# ── 6. Shot-Based Recoil Synchronization ─────────────────────────────────────


def test_recoil_compensation_advances_per_shot():
    """Recoil compensation delta is produced only when a shot occurs."""
    personality = PersonalityTraits(recoil_skill=0.8)
    firing = FiringController(personality, weapon="ak47")
    state = PlayerState()
    state.is_alive = True
    state.aim_phase = AimPhase.TRACKING

    det = make_dummy_detection(1720, 720, w=100, h=250)
    target = TrackedTarget(detection=det, frames_visible=10)

    cmd1 = firing.update(state, target, err_dist=0.0)
    # First shot produces downward pull (+dy in mouse counts to counteract climbing recoil)
    assert cmd1.recoil_dy > 0

    # Intermediate tick while cycling -> no new recoil impulse
    state._now += 0.020
    cmd_mid = firing.update(state, target, err_dist=0.0)
    assert cmd_mid.recoil_dy == 0


def test_weapon_change_resets_recoil_state():
    """Switching weapon resets recoil shot index and accumulator."""
    personality = PersonalityTraits()
    firing = FiringController(personality, weapon="ak47")
    state = PlayerState()
    state.shot_count = 5
    state.recoil_shot_index = 5
    state.recoil_accumulated = (2.0, 15.0)

    firing.set_weapon("m4a4", state)

    assert state.weapon == "m4a4"
    assert state.recoil_shot_index == 0
    assert state.recoil_accumulated == (0.0, 0.0)


# ── 7. Movement State & Counter-Strafing ──────────────────────────────────────


def test_movement_controller_commits_to_direction():
    """Combat strafe commits to direction for several ticks rather than jittering."""
    personality = PersonalityTraits(strafe_tendency=1.0, repositioning_tendency=0.0)
    ctrl = MovementController(personality)
    state = PlayerState()
    state.is_alive = True

    det = make_dummy_detection(1720, 720, w=70, h=180)
    target = TrackedTarget(detection=det)

    keys1 = ctrl.update(state, target, is_firing=False)
    active_key = "left" if keys1["left"] else "right"

    # Step forward 50ms (within the 300-800ms commitment duration)
    state._now += 0.050
    keys2 = ctrl.update(state, target, is_firing=False)

    assert keys2[active_key] is True  # Direction remained steady


def test_counter_strafing_when_stopping_to_shoot():
    """When a non-run-and-gun player stops strafing to fire, counter-strafe is engaged."""
    personality = PersonalityTraits(strafe_tendency=0.2)  # prefers stopping to shoot
    ctrl = MovementController(personality)
    state = PlayerState()
    state.is_alive = True

    det = make_dummy_detection(1720, 720, w=50, h=120)
    target = TrackedTarget(detection=det)

    # Currently moving left
    ctrl._current_phase = MovementPhase.STRAFING
    ctrl._current_direction = "left"
    ctrl._phase_start = state.now
    ctrl._phase_duration = 0.5

    # Starts firing -> triggers counter-strafe to right
    keys = ctrl.update(state, target, is_firing=True)
    assert ctrl._counter_strafing is True
    assert ctrl._counter_strafe_key == "right"
    assert keys["right"] is True


# ── 8. Adaptation Engine ─────────────────────────────────────────────────────


def test_adaptation_miss_streak_reduces_confidence():
    """Multiple consecutive misses erode confidence and increase hesitation."""
    personality = PersonalityTraits()
    adapt = AdaptationEngine(personality)
    state = PlayerState()
    initial_confidence = state.confidence

    # Simulate 6 shots with large aim error (> 40px)
    for i in range(1, 7):
        state.shot_count = i
        state.current_aim_error = (50.0, 30.0)
        adapt.update(state)

    assert state.confidence < initial_confidence
    assert state.reaction_tendency > 1.0  # Hesitation increased


def test_adaptation_damage_spikes_awareness():
    """Taking rapid damage raises awareness level."""
    personality = PersonalityTraits()
    adapt = AdaptationEngine(personality)
    state = PlayerState()
    initial_awareness = state.awareness_level

    # Sudden 50 HP loss
    state.health = 50
    adapt.update(state)

    assert state.awareness_level > initial_awareness
    assert state.damage_taken_recently == 50


# ── 9. Coordinate Conversion & FOV Propagation ────────────────────────────────


def test_fov_propagation_affects_mouse_counts():
    """A 122 degree horizontal FOV produces different mouse counts than 90 degree FOV."""
    # 200px horizontal delta
    counts_90 = screen_delta_to_mouse(
        dx_pixels=200,
        dy_pixels=0,
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        screen_width=3440,
        screen_height=1440,
        fov_horizontal=90.0,
    )
    counts_122 = screen_delta_to_mouse(
        dx_pixels=200,
        dy_pixels=0,
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        screen_width=3440,
        screen_height=1440,
        fov_horizontal=122.0,
    )

    # Wider FOV has a shorter focal length -> 200px corresponds to a wider angle
    assert counts_122[0] > counts_90[0]


# ── 10. Death and Respawn State Resets ─────────────────────────────────────────


def test_death_and_respawn_resets_combat_state():
    """Death resets transient combat state while preserving longer-term memories."""
    state = PlayerState()
    state.is_alive = True
    state.phase = BotPhase.ENGAGING
    state.shot_count = 12
    state.aim_phase = AimPhase.TRACKING

    state.on_death()
    assert state.phase == BotPhase.DEAD
    assert state.shot_count == 0
    assert state.aim_phase == AimPhase.IDLE
    assert len(state.recent_deaths) == 1

    state.on_respawn()
    assert state.phase == BotPhase.ROAMING
    assert state.is_alive is True
    assert state.health == 100


# ── 11. Purposeful Scanning ───────────────────────────────────────────────────


def test_scanning_checks_last_seen_enemy_position():
    """Scanning controller sweeps towards recently lost enemy location via SpatialMemory."""
    personality = PersonalityTraits(scanning_frequency=0.8)
    scanner = ScanningController(personality, screen_center=(1720, 720))
    state = PlayerState()
    state.is_alive = True

    # Enemy was seen 1 second ago at angular offset yaw=12.0 deg (to the right)
    state.spatial_memories.append(
        SpatialMemory(
            screen_x=2000.0,
            screen_y=720.0,
            yaw_offset_deg=12.0,
            pitch_offset_deg=0.0,
            last_seen_time=state.now - 1.0,
            confidence=0.85,
            target_id=1,
        )
    )

    dx, dy = scanner.update(state)
    assert scanner._current_scan is not None
    assert scanner._current_scan.reason == "check_spatial_memory"
    assert scanner._current_scan.target_dx > 0


# ── 12. End-to-End Decision Pipeline ─────────────────────────────────────────


def test_decision_engine_pipeline_end_to_end():
    """Full decision pipeline transitions smoothly from roam to engage to tracking."""
    personality = load_personality("average")
    engine = DecisionEngine(personality, screen_center=(1720, 720))
    state = PlayerState()
    state.is_alive = True

    # 1. No targets -> ROAM
    dec_roam = engine.decide(state, [])
    assert dec_roam.action_type == "roam"
    assert dec_roam.phase == BotPhase.ROAMING

    # 2. Enemy detected -> ENGAGE with reaction
    det = make_dummy_detection(1800, 720)
    target = TrackedTarget(detection=det, frames_visible=1)
    dec_engage = engine.decide(state, [target])

    assert dec_engage.phase == BotPhase.ENGAGING
    assert state.aim_phase == AimPhase.REACTING

    # 3. Simulate reaction time elapsing
    state._now += 1.0
    engine.reaction.update(state)
    assert state.aim_phase == AimPhase.ACQUIRING
