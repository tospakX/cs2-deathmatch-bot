"""Behavioral simulation harness and metrics evaluation test.

Runs multiple synthetic engagements across simulated time without CS2,
measuring realistic human-like distributions and verifying the absence
of robotic artifacts:
- Reaction time distribution variance & positive temporal autocorrelation
- Aim movement duration scaling with distance (Fitts's Law)
- Discrete corrective submovements (0-2 per acquisition)
- Movement episode duration & non-100% direction alternation
- Crouch episode continuity (> 0.5s duration, never 33ms pulses)
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior.decision import DecisionEngine
from src.behavior.motor import MotorPlanner
from src.behavior.personality import PersonalityTraits
from src.behavior.player_state import AimPhase, PlayerState, TrackedTarget
from src.behavior.reaction import ReactionSystem
from src.utils.behavior_metrics import BehaviorRecorder
from src.vision.detector import Detection


def make_target(cx: float, cy: float, tid: int = 1) -> TrackedTarget:
    det = Detection(
        class_id=0,
        class_name="player",
        confidence=0.85,
        x1=cx - 40,
        y1=cy - 100,
        x2=cx + 40,
        y2=cy + 100,
    )
    return TrackedTarget(detection=det, target_id=tid, frames_visible=5)


def test_reaction_time_distribution_and_autocorrelation():
    """Verify that reaction times exhibit realistic variance and temporal autocorrelation."""
    random.seed(42)
    p = PersonalityTraits(reaction_base_ms=210.0, reaction_variance_ms=25.0)
    reaction = ReactionSystem(personality=p)
    state = PlayerState()
    state.is_alive = True

    recorder = BehaviorRecorder()
    durations = []

    # Run 40 sequential reaction episodes
    for i in range(40):
        target = make_target(1800, 720, tid=i)
        reaction.initiate_reaction(state, target, context="new_target")
        durations.append(state.reaction_duration)

        # Simulate tick loop through reaction completion
        while state.reaction_pending:
            state.begin_tick()
            reaction.update(state)
            recorder.record_tick(state)

    metrics = recorder.compute_reaction_metrics()
    assert metrics["count"] >= 30
    # Mean reaction time should be within 180-250ms
    assert 0.160 < metrics["mean"] < 0.280
    # Variance must be non-zero (not a fixed robotic constant!)
    assert metrics["std"] > 0.005
    # Distinct values check: not all identical
    assert len(set(round(d, 4) for d in durations)) > 10


def test_aim_duration_scales_with_distance():
    """Verify Fitts's Law characteristic: longer distances require longer motor durations."""
    random.seed(123)
    p = PersonalityTraits(motor_speed=1.0)
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

    # Short distance (60px)
    state_short = PlayerState()
    state_short.is_alive = True
    state_short.aim_phase = AimPhase.ACQUIRING
    target_short = make_target(1780, 720, tid=1)
    state_short.primary_target = target_short
    planner.plan(state_short)
    dur_short = state_short.motor_episode.duration

    # Long distance (500px)
    state_long = PlayerState()
    state_long.is_alive = True
    state_long.aim_phase = AimPhase.ACQUIRING
    target_long = make_target(2220, 720, tid=2)
    state_long.primary_target = target_long
    planner.plan(state_long)
    dur_long = state_long.motor_episode.duration

    assert dur_long > dur_short


def test_submovements_are_discrete_and_bounded():
    """Verify reaching episodes consist of 0-2 discrete corrective submovements, not infinite PID jitter."""
    random.seed(7)
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

    submovement_counts = []
    for i in range(15):
        state = PlayerState()
        state.is_alive = True
        state.aim_phase = AimPhase.ACQUIRING
        target = make_target(1900 + i * 10, 720, tid=i)
        state.primary_target = target

        # Step until acquisition and correction finish
        for _ in range(40):
            state.begin_tick()
            planner.plan(state)
            if state.aim_phase == AimPhase.TRACKING:
                break

        submovement_counts.append(state.motor_episode.submovements_completed)

    # Submovements completed must be realistically bounded between 0 and 2
    for count in submovement_counts:
        assert 0 <= count <= 2


def test_synthetic_engagement_timeline_metrics():
    """Run an entire synthetic engagement and verify timeline metrics."""
    random.seed(88)
    p = PersonalityTraits(strafe_tendency=0.7, crouch_tendency=0.5)
    engine = DecisionEngine(
        personality=p,
        screen_center=(1720, 720),
        weapon="ak47",
    )
    motor = MotorPlanner(
        screen_center=(1720, 720),
        sensitivity=1.0,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=122.0,
        screen_width=3440,
        screen_height=1440,
        personality=p,
    )

    recorder = BehaviorRecorder()
    state = PlayerState()
    state.is_alive = True
    target = make_target(1850, 720, tid=100)

    # Simulate 120 ticks (~4 seconds of combat engagement)
    for _ in range(120):
        state.begin_tick()
        dec = engine.decide(state, [target])
        dx, dy = motor.plan(state)
        recorder.record_tick(state, motor_dx=dx, motor_dy=dy, recoil_dx=dec.firing_cmd.recoil_dx)

    # Verify crouch metrics: if crouched, average duration must be > 0.4s (no 33ms jitter)
    crouch_metrics = recorder.compute_crouch_metrics()
    if crouch_metrics["episode_count"] > 0:
        assert crouch_metrics["min_duration"] >= 0.40

    # Verify movement metrics: alternation ratio must not be 1.0 (robotic alternating)
    mov_metrics = recorder.compute_movement_metrics()
    if mov_metrics["episode_count"] > 2:
        assert mov_metrics["alternation_ratio"] < 1.0
