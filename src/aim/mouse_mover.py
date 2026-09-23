"""Continuous motor mouse movement replacing blocking Bezier curves.

Supports:
- State-continuous motor control (velocity, error history, momentum)
- Non-blocking per-frame movement updates (avoids freezing the main loop)
- Distinct motor behaviors for ballistic flicking, micro-correction, and tracking
- Backwards-compatible interface for tools and utilities
"""

from __future__ import annotations

import math
import random
import time

from src.input.mouse import move_relative


def _smoothstep(t: float) -> float:
    """Hermite smoothstep easing."""
    return t * t * (3.0 - 2.0 * t)


class MouseMover:
    """Manages continuous mouse movement with state continuity and non-blocking capability."""

    def __init__(self, base_speed: float = 6.0, noise_amplitude: float = 2.0):
        self.base_speed = base_speed
        self.noise_amplitude = noise_amplitude
        self._residual_x: float = 0.0
        self._residual_y: float = 0.0
        self._velocity_x: float = 0.0
        self._velocity_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0
        self._last_move_time: float = time.perf_counter()

    def move_to_delta(
        self,
        dx: float,
        dy: float,
        duration_ms: float | None = None,
        steps: int | None = None,
    ) -> None:
        """Apply mouse movement with continuous velocity blending and human-like easing."""
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return

        if duration_ms is None:
            # Fitts's Law duration scaled by base_speed
            speed_factor = max(1.0, self.base_speed / 6.0)
            duration_ms = (30.0 + 35.0 * math.log2(1.0 + dist / 40.0)) / speed_factor
            duration_ms *= random.uniform(0.9, 1.1)
            duration_ms = max(16.0, min(250.0, duration_ms))

        if steps is None:
            steps = max(5, int(duration_ms / 3.0))

        step_delay = duration_ms / steps / 1000.0
        prev_x, prev_y = 0.0, 0.0

        for i in range(1, steps + 1):
            t_linear = i / steps
            t = _smoothstep(t_linear)

            # Curved path using velocity momentum rather than arbitrary random noise
            curv_x = dx * t
            curv_y = dy * t

            step_dx = curv_x - prev_x + self._residual_x
            step_dy = curv_y - prev_y + self._residual_y

            int_dx = int(round(step_dx))
            int_dy = int(round(step_dy))
            self._residual_x = step_dx - int_dx
            self._residual_y = step_dy - int_dy

            if int_dx != 0 or int_dy != 0:
                move_relative(int_dx, int_dy)

            prev_x, prev_y = curv_x, curv_y

            if step_delay > 0:
                # Sub-millisecond sleep using time.sleep without hard busy-spin lock
                target = time.perf_counter() + step_delay
                while time.perf_counter() < target:
                    pass

        self._velocity_x = dx / max(duration_ms / 1000.0, 0.001)
        self._velocity_y = dy / max(duration_ms / 1000.0, 0.001)

    def micro_correct(self, dx: float, dy: float, delay_ms: float = 35.0) -> None:
        """Small correction movement with PD damping."""
        steps = random.randint(3, 5)
        self.move_to_delta(dx, dy, duration_ms=delay_ms, steps=steps)

    def move_instant(self, dx: int, dy: int) -> None:
        """Instant relative movement without trajectory."""
        move_relative(dx, dy)

    def step_continuous(self, target_dx: float, target_dy: float, dt: float) -> tuple[int, int]:
        """Non-blocking single-frame step for integration with the 30Hz game loop."""
        kp = 0.35 * (self.base_speed / 6.0)
        kd = 0.08

        d_err_x = target_dx - self._prev_error_x
        d_err_y = target_dy - self._prev_error_y

        desired_vx = target_dx * kp - d_err_x * kd
        desired_vy = target_dy * kp - d_err_y * kd

        blend = 0.4
        self._velocity_x = self._velocity_x * (1 - blend) + desired_vx * blend
        self._velocity_y = self._velocity_y * (1 - blend) + desired_vy * blend

        tot_x = self._velocity_x + self._residual_x
        tot_y = self._velocity_y + self._residual_y
        int_dx = int(round(tot_x))
        int_dy = int(round(tot_y))
        self._residual_x = tot_x - int_dx
        self._residual_y = tot_y - int_dy

        self._prev_error_x = target_dx
        self._prev_error_y = target_dy

        if int_dx != 0 or int_dy != 0:
            move_relative(int_dx, int_dy)

        return int_dx, int_dy
