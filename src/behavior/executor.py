"""High-frequency non-blocking motor and input episode executor.

Decouples the 30Hz perception/decision cadence from high-frequency (100-250Hz)
motor trajectory generation, trigger release timing, and counter-strafe braking.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.behavior.firing import FiringController
    from src.behavior.motor import MotorPlanner
    from src.behavior.movement import MovementController
    from src.behavior.player_state import PlayerState


class MotorExecutor:
    """High-frequency input executor stepping between 30Hz perception frames."""

    def __init__(
        self,
        player_state: PlayerState,
        motor_planner: MotorPlanner,
        firing_controller: FiringController,
        movement_controller: MovementController,
        mouse_backend: Any,
        keyboard_backend: Any,
        keybinds: dict[str, Any],
    ):
        self.player_state = player_state
        self.motor_planner = motor_planner
        self.firing_controller = firing_controller
        self.movement_controller = movement_controller
        self.mouse = mouse_backend
        self.keyboard = keyboard_backend
        self.keybinds = keybinds
        self._last_step_time: float = time.perf_counter()

    def run_substeps_until(self, end_time: float, min_sleep: float = 0.002) -> None:
        """Execute high-cadence sub-steps until the next 30Hz frame begins.

        Yields CPU gracefully while providing ~200-250Hz temporal precision for
        trigger release, counter-strafe impulse completion, and reaching trajectory.
        """
        self._last_step_time = time.perf_counter()

        while True:
            now = time.perf_counter()
            remaining = end_time - now
            if remaining <= 0.0005:
                break

            dt = max(0.001, min(0.050, now - self._last_step_time))
            self._last_step_time = now

            # 1. High-cadence trigger release check (e.g. 45-80ms finger tap hold)
            if self.firing_controller.check_trigger_release(self.player_state, now):
                if hasattr(self.mouse, "mouse_up"):
                    self.mouse.mouse_up("left")

            # 2. High-cadence counter-strafe braking lifecycle check (e.g. 60-80ms braking tap)
            key_changes, changed = self.movement_controller.check_lifecycle(now)
            if changed:
                for action_name, should_press in key_changes.items():
                    key_code = self.keybinds.get(action_name)
                    if key_code and hasattr(self.keyboard, "key_up"):
                        if should_press:
                            self.keyboard.key_down(key_code)
                        else:
                            self.keyboard.key_up(key_code)

            # 3. High-cadence motor plan evaluation for active reaching episodes
            if self.player_state.motor_episode.active:
                dx, dy = self.motor_planner.plan(self.player_state, dt=dt, now=now)
                if dx != 0 or dy != 0:
                    if hasattr(self.mouse, "move_relative"):
                        self.mouse.move_relative(dx, dy)
                    yaw_deg = dx * self.motor_planner.m_yaw * self.motor_planner.sensitivity
                    pitch_deg = dy * self.motor_planner.m_pitch * self.motor_planner.sensitivity
                    self.player_state.on_camera_rotated(yaw_deg, pitch_deg)

            # Small non-blocking sleep yielding to OS scheduler
            rem_after = end_time - time.perf_counter()
            if rem_after > min_sleep:
                time.sleep(min(min_sleep, rem_after - 0.0005))
            else:
                break
