"""Deterministic long-run behavioral simulation and metrics evaluation tool.

Runs multi-scenario behavioral simulations using FakeClock and simulated game
events without requiring a live CS2 client, display, or GPU. Evaluates statistical
distributions and human-likeness metrics across different player personalities.

Usage:
    python tools/run_behavioral_simulation.py
    python tools/run_behavioral_simulation.py --personality all --ticks 2000
    python tools/run_behavioral_simulation.py --personality tryhard --output-json reports/sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior.decision import DecisionEngine
from src.behavior.executor import MotorExecutor
from src.behavior.motor import MotorPlanner
from src.behavior.perception import PerceptionSystem
from src.behavior.personality import PersonalityTraits, load_personality
from src.behavior.player_state import (
    BotPhase,
    PlayerState,
)
from src.utils.behavior_metrics import BehaviorRecorder
from src.utils.clock import FakeClock
from src.vision.detector import Detection


class DummyMouseBackend:
    """Mock mouse backend recording relative movements and clicks."""

    def __init__(self):
        self.dx_total: int = 0
        self.dy_total: int = 0
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[str] = []
        self.mouse_downs: list[str] = []
        self.mouse_ups: list[str] = []

    def move_relative(self, dx: int, dy: int) -> None:
        self.dx_total += dx
        self.dy_total += dy
        self.moves.append((dx, dy))

    def mouse_down(self, button: str = "left") -> None:
        self.mouse_downs.append(button)

    def mouse_up(self, button: str = "left") -> None:
        self.mouse_ups.append(button)


class DummyKeyboardBackend:
    """Mock keyboard backend tracking held and released keys."""

    def __init__(self):
        self.held_keys: set[str] = set()
        self.key_presses: list[str] = []

    def key_down(self, key: str) -> None:
        self.held_keys.add(key)

    def key_up(self, key: str) -> None:
        self.held_keys.discard(key)

    def hold_key(self, key: str) -> None:
        self.held_keys.add(key)

    def release_key(self, key: str) -> None:
        self.held_keys.discard(key)

    def release_all(self) -> None:
        self.held_keys.clear()

    def key_press(self, key: str) -> None:
        self.key_presses.append(key)


def make_sim_detection(
    cx: float,
    cy: float,
    distance_m: float = 15.0,
    confidence: float = 0.92,
) -> Detection:
    """Generate a realistic bounding box for a target at distance_m meters."""
    # Approximate player height in screen pixels as a function of distance
    # At 5m: ~380px, at 15m: ~180px, at 30m: ~85px
    h = max(35.0, min(500.0, 2700.0 / max(1.0, distance_m)))
    w = h * 0.38
    return Detection(
        class_id=0,
        class_name="player",
        confidence=confidence,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
    )


class SimulationHarness:
    """Simulates multi-scenario gameplay using FakeClock and stateful behavioral components."""

    def __init__(
        self,
        personality: PersonalityTraits,
        screen_w: int = 1920,
        screen_h: int = 1080,
    ):
        self.personality = personality
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cx = screen_w // 2
        self.cy = screen_h // 2
        self.clock = FakeClock(initial_time=100.0, default_dt=0.033)

        self.player_state = PlayerState(clock=self.clock)
        self.player_state.is_alive = True
        self.player_state.transition_phase(BotPhase.ROAMING)

        self.perception = PerceptionSystem(
            screen_center=(self.cx, self.cy),
            screen_size=(self.screen_w, self.screen_h),
            fov_h=90.0,
        )
        self.decision = DecisionEngine(
            personality=self.personality,
            screen_center=(self.cx, self.cy),
            screen_width=self.screen_w,
            screen_height=self.screen_h,
            fov_h=90.0,
        )
        self.motor = MotorPlanner(
            screen_center=(self.cx, self.cy),
            sensitivity=1.0,
            m_yaw=0.022,
            m_pitch=0.022,
            fov_h=90.0,
            screen_width=self.screen_w,
            screen_height=self.screen_h,
            personality=self.personality,
        )

        self.camera_yaw_deg: float = 0.0
        self.camera_pitch_deg: float = 0.0
        aspect = self.screen_w / max(1.0, float(self.screen_h))
        self.fov_v = 2.0 * math.degrees(math.atan(math.tan(math.radians(90.0 / 2.0)) / aspect))

        self.dummy_mouse = DummyMouseBackend()
        self.dummy_keyboard = DummyKeyboardBackend()
        self.keybinds = {
            "forward": "w",
            "back": "s",
            "left": "a",
            "right": "d",
            "crouch": "ctrl",
            "walk": "shift",
            "reload": "r",
            "inspect": "f",
        }

        self.executor = MotorExecutor(
            player_state=self.player_state,
            motor_planner=self.motor,
            firing_controller=self.decision.firing,
            movement_controller=self.decision.movement,
            mouse_backend=self.dummy_mouse,
            keyboard_backend=self.dummy_keyboard,
            keybinds=self.keybinds,
            clock=self.clock,
        )
        self.recorder = BehaviorRecorder()

    def project_world_target(
        self,
        world_yaw: float,
        world_pitch: float,
        distance_m: float,
        confidence: float = 0.92,
    ) -> Detection | None:
        """Project a 3D world target into 2D camera viewport detection."""
        rel_yaw = (world_yaw - self.camera_yaw_deg + 180.0) % 360.0 - 180.0
        rel_pitch = world_pitch - self.camera_pitch_deg

        # If outside camera frustum, not visible on screen
        if abs(rel_yaw) > (90.0 / 2.0) or abs(rel_pitch) > (self.fov_v / 2.0):
            return None

        screen_x = self.cx + (rel_yaw / 90.0) * self.screen_w
        screen_y = self.cy + (rel_pitch / self.fov_v) * self.screen_h
        return make_sim_detection(screen_x, screen_y, distance_m=distance_m, confidence=confidence)

    def step_frame(self, detections: list[Detection]) -> None:
        """Step one 30Hz frame and high-cadence substeps."""
        tick_start = self.clock.now()
        dt = 0.033
        tick_end = tick_start + dt

        # 1. Player state begin tick
        self.player_state.begin_tick(tick_start)

        # 2. Perception update
        self.perception.update(self.player_state, detections)
        targets = self.perception.get_visible_targets(self.player_state)

        # 3. Decision update
        decision = self.decision.decide(self.player_state, targets)

        # 4. Motor planning and recoil compensation
        motor_dx, motor_dy = 0, 0
        if decision.target is not None:
            motor_dx, motor_dy = self.motor.plan(self.player_state)
        elif decision.scan_delta != (0.0, 0.0):
            motor_dx, motor_dy = self.motor.plan_scan(
                decision.scan_delta[0], decision.scan_delta[1], self.player_state
            )

        recoil_dx = decision.firing_cmd.recoil_dx
        recoil_dy = decision.firing_cmd.recoil_dy
        if recoil_dx != 0 or recoil_dy != 0:
            motor_dx += recoil_dx
            motor_dy += recoil_dy

        if motor_dx != 0 or motor_dy != 0:
            self.dummy_mouse.move_relative(motor_dx, motor_dy)
            yaw_deg = motor_dx * self.motor.m_yaw * self.motor.sensitivity
            pitch_deg = motor_dy * self.motor.m_pitch * self.motor.sensitivity
            self.camera_yaw_deg = (self.camera_yaw_deg + yaw_deg + 180.0) % 360.0 - 180.0
            self.camera_pitch_deg = max(-89.0, min(89.0, self.camera_pitch_deg + pitch_deg))
            self.player_state.on_camera_rotated(yaw_deg, pitch_deg)

        # 5. Apply firing trigger action
        if decision.firing_cmd.trigger_action in ("press", "hold"):
            self.dummy_mouse.mouse_down("left")
            self.player_state.is_trigger_held = True
        elif decision.firing_cmd.trigger_action == "release":
            self.dummy_mouse.mouse_up("left")
            self.player_state.is_trigger_held = False

        # 6. Apply movement keys
        for action, should_hold in decision.movement_keys.items():
            key = self.keybinds.get(action)
            if key:
                if should_hold:
                    self.dummy_keyboard.hold_key(key)
                    self.player_state.held_keys.add(action)
                else:
                    self.dummy_keyboard.release_key(key)
                    self.player_state.held_keys.discard(action)

        # 7. Record tick metrics
        self.recorder.record_tick(
            self.player_state,
            motor_dx=motor_dx,
            motor_dy=motor_dy,
            recoil_dx=recoil_dx,
            recoil_dy=recoil_dy,
        )

        # 8. High-cadence motor executor substeps until next tick begins
        self.executor.run_substeps_until(tick_end, min_sleep=0.002)

        # Advance clock to tick_end if not already there
        if self.clock.now() < tick_end:
            self.clock.advance(tick_end - self.clock.now())

    def run_scenario_roam(self, ticks: int = 150) -> None:
        """Scenario: Roaming without enemies."""
        for _ in range(ticks):
            self.step_frame([])

    def run_scenario_encounter(
        self,
        distance_m: float = 12.0,
        initial_yaw_deg: float = 12.0,
        initial_pitch_deg: float = -2.0,
        ticks: int = 120,
    ) -> None:
        """Scenario: Sudden enemy pop-up requiring reaction, aim, and firing."""
        world_yaw = (self.camera_yaw_deg + initial_yaw_deg + 180.0) % 360.0 - 180.0
        world_pitch = self.camera_pitch_deg + initial_pitch_deg

        for i in range(ticks):
            drift_yaw = math.sin(i * 0.1) * 0.2
            det = self.project_world_target(
                world_yaw + drift_yaw, world_pitch, distance_m=distance_m
            )
            dets = [det] if det is not None else []
            self.step_frame(dets)

    def run_scenario_lost_reacquire(self, ticks_visible: int = 40, ticks_lost: int = 50) -> None:
        """Scenario: Target observed, breaks line of sight, and is reacquired via spatial memory."""
        world_yaw = (self.camera_yaw_deg + 14.0 + 180.0) % 360.0 - 180.0
        world_pitch = self.camera_pitch_deg - 1.0

        # 1. Target visible
        for _ in range(ticks_visible):
            det = self.project_world_target(world_yaw, world_pitch, distance_m=14.0)
            dets = [det] if det is not None else []
            self.step_frame(dets)

        # 2. Target breaks line of sight (behind smoke/wall)
        for _ in range(ticks_lost):
            self.step_frame([])

    def run_scenario_crossing_targets(self, ticks: int = 100) -> None:
        """Scenario: Two targets crossing paths requiring stable target assignment."""
        center_yaw = self.camera_yaw_deg
        for i in range(ticks):
            # Target 1 moving left to right
            y1 = center_yaw - 10.0 + (i * 0.20)
            p1 = self.camera_pitch_deg - 0.5
            det1 = self.project_world_target(y1, p1, distance_m=12.0, confidence=0.90)

            # Target 2 moving right to left
            y2 = center_yaw + 10.0 - (i * 0.20)
            p2 = self.camera_pitch_deg - 0.5
            det2 = self.project_world_target(y2, p2, distance_m=15.0, confidence=0.88)

            dets = [d for d in (det1, det2) if d is not None]
            self.step_frame(dets)


def run_full_simulation_suite(
    personality_name: str,
    total_ticks: int = 2000,
) -> dict[str, Any]:
    """Execute complete simulation across all scenarios for the specified personality."""
    personality = load_personality(personality_name)
    harness = SimulationHarness(personality=personality)

    # Allocate tick budget across scenarios
    roam_ticks = int(total_ticks * 0.20)
    enc_long_ticks = int(total_ticks * 0.20)
    enc_mid_ticks = int(total_ticks * 0.25)
    enc_close_ticks = int(total_ticks * 0.15)
    crossing_ticks = int(total_ticks * 0.10)
    lost_ticks = int(total_ticks * 0.10)

    # 1. Roam & corner checking
    harness.run_scenario_roam(roam_ticks)

    # 2. Long range encounter (tap preference)
    harness.run_scenario_encounter(
        distance_m=24.0, initial_yaw_deg=8.0, initial_pitch_deg=-1.5, ticks=enc_long_ticks
    )

    # 3. Intermission roam
    harness.run_scenario_roam(30)

    # 4. Mid range encounter (burst preference)
    harness.run_scenario_encounter(
        distance_m=12.0, initial_yaw_deg=-10.0, initial_pitch_deg=1.0, ticks=enc_mid_ticks
    )

    # 5. Intermission roam
    harness.run_scenario_roam(30)

    # 6. Close range encounter (spray preference)
    harness.run_scenario_encounter(
        distance_m=6.0, initial_yaw_deg=5.0, initial_pitch_deg=0.5, ticks=enc_close_ticks
    )

    # 7. Crossing targets
    harness.run_scenario_crossing_targets(crossing_ticks)

    # 8. Target lost and reacquired
    harness.run_scenario_lost_reacquire(ticks_visible=40, ticks_lost=lost_ticks)

    # Compute comprehensive metrics
    rec = harness.recorder
    rx_metrics = rec.compute_reaction_metrics()
    mov_metrics = rec.compute_movement_metrics()
    vel_metrics = rec.compute_mouse_velocity_metrics()
    tap_metrics = rec.compute_tap_interval_metrics()
    crouch_metrics = rec.compute_crouch_metrics()

    return {
        "personality": personality_name,
        "total_ticks_recorded": len(rec.records),
        "simulated_time_s": len(rec.records) * 0.033,
        "reaction": rx_metrics,
        "movement": mov_metrics,
        "mouse_velocity": vel_metrics,
        "tapping": tap_metrics,
        "crouch": crouch_metrics,
    }


def print_simulation_report(results: list[dict[str, Any]]) -> None:
    """Print structured comparison report."""
    print("=" * 82)
    print("         CS2 DEATHMATCH BOT - BEHAVIORAL SIMULATION EVALUATION REPORT         ")
    print("=" * 82)

    for res in results:
        p_name = res["personality"].upper()
        ticks = res["total_ticks_recorded"]
        t_sim = res["simulated_time_s"]
        rx = res["reaction"]
        mv = res["movement"]
        vel = res["mouse_velocity"]
        tap = res["tapping"]
        cr = res["crouch"]

        print(f"\n[PERSONALITY PROFILE: {p_name}] ({ticks} ticks | {t_sim:.1f}s simulated)")
        print("-" * 82)

        # 1. Reaction Metrics
        print("  1. REACTION & ATTENTION TEMPORAL CONTINUITY:")
        print(f"     * Reaction Events:        {int(rx['count'])}")
        rx_mean_ms = rx["mean"] * 1000.0
        rx_std_ms = rx["std"] * 1000.0
        print(f"     * Mean Reaction Latency:  {rx_mean_ms:.1f} ms (std: {rx_std_ms:.1f} ms)")
        print(f"     * Lag-1 Autocorrelation:  {rx['autocorr']:+.3f}")

        # 2. Movement & Footwork Metrics
        print("  2. COMBAT FOOTWORK & COUNTER-STRAFING:")
        print(f"     * Strafe Episodes:        {int(mv['episode_count'])}")
        same_t = int(mv["same_transitions"])
        opp_t = int(mv["opposite_transitions"])
        print(f"     * Same vs Opposite Turns: {same_t} same / {opp_t} opposite")
        print(f"     * Alternation Ratio:      {mv['alternation_ratio']:.3f}")
        print(f"     * Tick Persistence:       {mv['tick_persistence']:.3f}")

        # 3. Motor Kinematics
        print("  3. MOTOR KINEMATICS & REACHING TRAJECTORY:")
        print(f"     * Mean Mouse Speed:       {vel['mean_speed']:.2f} counts/tick")
        print(f"     * Peak Mouse Speed:       {vel['max_speed']:.2f} counts/tick")
        print(f"     * Velocity Autocorr:      {vel['velocity_autocorr']:+.3f}")

        # 4. Firing & Recoil
        print("  4. FIRING DISCIPLINE & PACING:")
        print(f"     * Total Shots Fired:      {int(tap['shot_count'])}")
        if tap["interval_count"] > 0:
            mean_int_ms = tap["mean_interval"] * 1000.0
            std_int_ms = tap["std_interval"] * 1000.0
            min_int_ms = tap["min_interval"] * 1000.0
            print(f"     * Mean Shot Interval:     {mean_int_ms:.1f} ms (std: {std_int_ms:.1f} ms)")
            print(f"     * Min Shot Interval:      {min_int_ms:.1f} ms (respects cycle rate)")

        # 5. Crouch Footwork
        print("  5. CROUCH COMMITMENT & FLUTTER PREVENTION:")
        print(f"     * Crouch Episodes:        {int(cr['episode_count'])}")
        if cr["episode_count"] > 0:
            print(f"     * Min Crouch Duration:    {cr['min_duration']:.2f} s")
            print(f"     * Avg Crouch Duration:    {cr['avg_duration']:.2f} s")

        # Human-likeness verdict
        is_human = (
            0.35 <= mv["alternation_ratio"] <= 0.85
            and vel["velocity_autocorr"] > 0.15
            and (cr["episode_count"] == 0 or cr["min_duration"] >= 0.45)
        )
        verdict = (
            "PASSED (Human-like behavior verified)" if is_human else "FAILED (Non-human profile)"
        )
        print(f"  --> VERDICT: {verdict}")

    print("\n" + "=" * 82)


def main() -> None:
    parser = argparse.ArgumentParser(description="CS2 Bot Behavioral Simulation Harness")
    parser.add_argument(
        "--personality",
        choices=["average", "noob", "tryhard", "all"],
        default="average",
        help="Personality profile to simulate (default: average)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=2000,
        help="Number of simulated ticks per personality (default: 2000 ticks = ~66s)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to output metrics JSON file",
    )
    args = parser.parse_args()

    personalities = (
        ["average", "noob", "tryhard"] if args.personality == "all" else [args.personality]
    )

    results: list[dict[str, Any]] = []
    for p_name in personalities:
        res = run_full_simulation_suite(p_name, total_ticks=args.ticks)
        results.append(res)

    print_simulation_report(results)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved simulation report JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
