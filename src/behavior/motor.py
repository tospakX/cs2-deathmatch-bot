"""Continuous motor planner — replaces blocking Bezier aim paths.

Models distinct motor behaviors:
- Large target acquisition (ballistic flick)
- Small correction (micro-adjust)
- Continuous tracking (smooth pursuit)
- Camera scanning (deliberate look)
- Recoil compensation (counter-pull)

Each behavior maintains continuous state.  The next movement depends on
the previous movement, creating temporal continuity rather than
independently generated trajectories every frame.

All output is non-blocking: the motor system produces a (dx, dy) delta
to apply THIS TICK, not a blocking sequence of movements.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from src.behavior.player_state import AimPhase, MotorPhase
from src.utils.math_helpers import bbox_to_aim_point, screen_delta_to_mouse

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class MotorPlanner:
    """Non-blocking continuous motor system for aiming.

    Every tick, call ``plan()`` to get the mouse delta to apply.
    The planner maintains internal velocity, error, and phase state
    that persists across frames.
    """

    def __init__(
        self,
        screen_center: tuple[int, int],
        sensitivity: float,
        m_yaw: float,
        m_pitch: float,
        fov_h: float,
        screen_width: int,
        screen_height: int,
        personality: PersonalityTraits,
    ):
        self.cx, self.cy = screen_center
        self.sensitivity = sensitivity
        self.m_yaw = m_yaw
        self.m_pitch = m_pitch
        self.fov_h = fov_h
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.personality = personality

        # ── Persistent motor state ───────────────────────────────────────
        self._velocity_x: float = 0.0
        self._velocity_y: float = 0.0
        self._residual_x: float = 0.0
        self._residual_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0
        self._overshoot_active: bool = False
        self._overshoot_correction_pending: bool = False
        self._settle_ticks: int = 0
        # Short-term motor tendency (drift): slight consistent bias in aim.
        self._bias_x: float = random.gauss(0, 0.3)
        self._bias_y: float = random.gauss(0, 0.2)

    def plan(self, state: PlayerState) -> tuple[int, int]:
        """Compute this tick's mouse delta based on current aim/motor state.

        Returns (dx, dy) in raw mouse counts to apply via SendInput.
        """
        aim = state.aim_phase

        if aim == AimPhase.IDLE or aim == AimPhase.REACTING:
            # Not moving the mouse toward a target.
            self._decay_velocity()
            return self._idle_micro_jitter(state)

        target = state.primary_target
        if target is None:
            self._decay_velocity()
            return (0, 0)

        # Compute screen-space error to target.
        err_x, err_y, err_dist, mouse_dx, mouse_dy = self._compute_error(target, state)

        # Store error for tracking derivative.
        state.current_aim_error = (err_x, err_y)

        if aim == AimPhase.ACQUIRING:
            return self._acquisition_move(state, mouse_dx, mouse_dy, err_dist)
        elif aim == AimPhase.CORRECTING:
            return self._correction_move(state, mouse_dx, mouse_dy, err_dist)
        elif aim == AimPhase.TRACKING:
            return self._tracking_move(state, target, mouse_dx, mouse_dy, err_dist)
        elif aim == AimPhase.REACQUIRING:
            return self._reacquisition_move(state, mouse_dx, mouse_dy, err_dist)

        return (0, 0)

    def plan_scan(self, dx: float, dy: float, state: PlayerState) -> tuple[int, int]:
        """Plan a scanning/camera movement."""
        state.motor_phase = MotorPhase.SCANNING_TURN
        # Smooth the scan with velocity blending.
        alpha = 0.3
        self._velocity_x = self._velocity_x * (1 - alpha) + dx * alpha
        self._velocity_y = self._velocity_y * (1 - alpha) + dy * alpha
        return self._emit(self._velocity_x, self._velocity_y)

    # ── Motor behaviors ──────────────────────────────────────────────────

    def _acquisition_move(
        self, state: PlayerState, mouse_dx: float, mouse_dy: float, err_dist: float
    ) -> tuple[int, int]:
        """Large ballistic flick toward target.

        Uses velocity-based approach: ramp up, coast, decelerate.
        The flick doesn't try to land perfectly — it gets close and
        transitions to correction phase.
        """
        p = self.personality
        state.motor_phase = MotorPhase.FLICKING

        # Target velocity: cover the remaining distance over a personality-
        # dependent duration.
        duration = p.sample_motor_duration(err_dist)
        if duration < 0.001:
            duration = 0.033  # at least one tick

        ticks_remaining = max(1.0, duration / max(state.tick_dt, 0.001))

        # How much to move this tick: proportional control with momentum.
        gain = min(0.6 + p.motor_speed * 0.3, 0.95)
        target_vx = mouse_dx * gain / ticks_remaining
        target_vy = mouse_dy * gain / ticks_remaining

        # Blend with current velocity for smooth acceleration.
        blend = 0.4 + p.motor_speed * 0.2
        self._velocity_x = self._velocity_x * (1 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1 - blend) + target_vy * blend

        # Apply imprecision based on personality.
        imprecision = (1.0 - p.motor_precision) * 2.0
        self._velocity_x += random.gauss(0, imprecision) + self._bias_x
        self._velocity_y += random.gauss(0, imprecision * 0.7) + self._bias_y

        # Overshoot decision (once per acquisition, not every frame).
        if (
            not self._overshoot_active
            and not self._overshoot_correction_pending
            and err_dist > 50
            and random.random() < p.overshoot_tendency
        ):
            self._overshoot_active = True
            overshoot_factor = 1.0 + random.uniform(0.05, 0.25) * (1.0 - p.motor_precision)
            self._velocity_x *= overshoot_factor
            self._velocity_y *= overshoot_factor

        # Transition to correction when close enough.
        threshold = 30 + (1.0 - p.motor_precision) * 40
        if err_dist < threshold:
            state.aim_phase = AimPhase.CORRECTING
            state.motor_phase = MotorPhase.SETTLING
            self._settle_ticks = 0
            if self._overshoot_active:
                self._overshoot_active = False
                self._overshoot_correction_pending = True

        result = self._emit(self._velocity_x, self._velocity_y)
        self._prev_error_x, self._prev_error_y = mouse_dx, mouse_dy
        return result

    def _correction_move(
        self, state: PlayerState, mouse_dx: float, mouse_dy: float, err_dist: float
    ) -> tuple[int, int]:
        """Small corrections after initial acquisition.

        Slower, more precise movements.  Uses error derivative for
        damping (PD-like behavior).
        """
        p = self.personality
        self._settle_ticks += 1
        state.motor_phase = MotorPhase.MICRO_CORRECTING

        if err_dist < 5:
            # Close enough — transition to tracking.
            state.aim_phase = AimPhase.TRACKING
            self._overshoot_correction_pending = False
            self._velocity_x *= 0.3
            self._velocity_y *= 0.3
            return self._emit(self._velocity_x, self._velocity_y)

        # Proportional term.
        kp = 0.15 + p.correction_tendency * 0.25
        # Derivative term (damping based on error change).
        kd = 0.05 + p.motor_precision * 0.1
        d_err_x = mouse_dx - self._prev_error_x
        d_err_y = mouse_dy - self._prev_error_y

        target_vx = mouse_dx * kp - d_err_x * kd
        target_vy = mouse_dy * kp - d_err_y * kd

        # Smooth blend.
        blend = 0.5
        self._velocity_x = self._velocity_x * (1 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1 - blend) + target_vy * blend

        # Less noise during correction.
        noise_scale = (1.0 - p.motor_precision) * 0.5
        self._velocity_x += random.gauss(0, noise_scale)
        self._velocity_y += random.gauss(0, noise_scale * 0.7)

        # If we've been correcting too long, accept imperfect aim.
        if self._settle_ticks > 15:
            state.aim_phase = AimPhase.TRACKING

        result = self._emit(self._velocity_x, self._velocity_y)
        self._prev_error_x, self._prev_error_y = mouse_dx, mouse_dy
        return result

    def _tracking_move(
        self,
        state: PlayerState,
        target: TrackedTarget,
        mouse_dx: float,
        mouse_dy: float,
        err_dist: float,
    ) -> tuple[int, int]:
        """Continuous smooth pursuit tracking.

        Maintains aim on a moving target by predicting target movement
        and applying smooth corrections.  Error slowly drifts and
        is periodically corrected.
        """
        p = self.personality
        state.motor_phase = MotorPhase.SMOOTH_TRACKING

        # Predict target movement from velocity.
        vx, vy = target.velocity_estimate
        pred_factor = state.tick_dt * p.tracking_steadiness * 0.5

        # Tracking gain: how aggressively to follow.
        tracking_gain = 0.1 + p.tracking_steadiness * 0.15

        target_vx = mouse_dx * tracking_gain + vx * pred_factor
        target_vy = mouse_dy * tracking_gain + vy * pred_factor

        # Smooth blend (more than acquisition for stability).
        blend = 0.3 + p.tracking_steadiness * 0.2
        self._velocity_x = self._velocity_x * (1 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1 - blend) + target_vy * blend

        # Tracking jitter — represents hand tremor during tracking.
        tremor = (1.0 - p.tracking_steadiness) * 1.5
        self._velocity_x += random.gauss(0, tremor) + self._bias_x * 0.3
        self._velocity_y += random.gauss(0, tremor * 0.7) + self._bias_y * 0.3

        # If error grows too large, switch back to correction.
        if err_dist > 60 + (1.0 - p.motor_precision) * 40:
            state.aim_phase = AimPhase.CORRECTING
            self._settle_ticks = 0

        result = self._emit(self._velocity_x, self._velocity_y)
        self._prev_error_x, self._prev_error_y = mouse_dx, mouse_dy
        return result

    def _reacquisition_move(
        self, state: PlayerState, mouse_dx: float, mouse_dy: float, err_dist: float
    ) -> tuple[int, int]:
        """Re-acquiring a briefly lost target.

        Faster than initial acquisition since we have recent memory.
        """
        p = self.personality
        state.motor_phase = MotorPhase.FLICKING

        gain = 0.4 + p.motor_speed * 0.3
        target_vx = mouse_dx * gain
        target_vy = mouse_dy * gain

        blend = 0.5
        self._velocity_x = self._velocity_x * (1 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1 - blend) + target_vy * blend

        if err_dist < 30:
            state.aim_phase = AimPhase.TRACKING

        result = self._emit(self._velocity_x, self._velocity_y)
        self._prev_error_x, self._prev_error_y = mouse_dx, mouse_dy
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_error(
        self, target: TrackedTarget, state: PlayerState
    ) -> tuple[float, float, float, float, float]:
        """Compute screen-space error and mouse-count delta to target.

        Returns (err_x_px, err_y_px, err_dist_px, mouse_dx, mouse_dy).
        """
        head_aim = random.random() < self.personality.head_aim_preference
        d = target.detection
        aim_x, aim_y = bbox_to_aim_point(d.x1, d.y1, d.x2, d.y2, head_aim=head_aim)

        err_x = aim_x - self.cx
        err_y = aim_y - self.cy
        err_dist = math.sqrt(err_x * err_x + err_y * err_y)

        mouse_dx, mouse_dy = screen_delta_to_mouse(
            err_x,
            err_y,
            self.sensitivity,
            self.m_yaw,
            self.m_pitch,
            self.screen_w,
            self.screen_h,
            self.fov_h,
        )

        return err_x, err_y, err_dist, float(mouse_dx), float(mouse_dy)

    def _emit(self, vx: float, vy: float) -> tuple[int, int]:
        """Convert float velocity to integer mouse counts with residual tracking."""
        total_x = vx + self._residual_x
        total_y = vy + self._residual_y
        int_dx = int(round(total_x))
        int_dy = int(round(total_y))
        self._residual_x = total_x - int_dx
        self._residual_y = total_y - int_dy
        return (int_dx, int_dy)

    def _decay_velocity(self) -> None:
        """Gradually decay velocity when not aiming."""
        self._velocity_x *= 0.7
        self._velocity_y *= 0.7

    def _idle_micro_jitter(self, state: PlayerState) -> tuple[int, int]:
        """Very slight jitter when idle (hand tremor)."""
        p = self.personality
        if random.random() < 0.3:
            jx = random.gauss(0, 0.3 * (1.0 - p.motor_precision))
            jy = random.gauss(0, 0.2 * (1.0 - p.motor_precision))
            return self._emit(jx, jy)
        return (0, 0)

    def drift_bias(self) -> None:
        """Slowly drift the motor bias (called periodically)."""
        self._bias_x += random.gauss(0, 0.05)
        self._bias_y += random.gauss(0, 0.03)
        # Mean-revert.
        self._bias_x *= 0.98
        self._bias_y *= 0.98
