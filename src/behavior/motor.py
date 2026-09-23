"""Continuous motor planner — models human motor control & reaching submovements.

Models distinct motor phases according to human motor research:
1. Primary movement (ballistic flick with bell-shaped velocity profile)
2. Deceleration and visual assessment
3. Discrete corrective submovements (0, 1, or 2 based on personality and distance)
4. Smooth pursuit tracking with velocity lead and realistic sensory lag
5. Camera scanning and deliberate reorientation

Units are strictly defined and converted:
- Screen error: screen pixels
- Target velocity: screen pixels per second
- Motor command: raw mouse counts (integer SendInput relative movements)
- Time: seconds (dt)
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from src.behavior.player_state import AimPhase, MotorPhase
from src.utils.math_helpers import screen_delta_to_mouse

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class MotorPlanner:
    """Non-blocking continuous motor system executing structured motor episodes.

    Every tick, call ``plan()`` to obtain this tick's integer (dx, dy) mouse counts.
    The planner executes an active MotorEpisode across multiple frames rather than
    resampling trajectories on each 30Hz tick.
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
        self._velocity_x: float = 0.0  # mouse counts / tick
        self._velocity_y: float = 0.0  # mouse counts / tick
        self._residual_x: float = 0.0  # sub-pixel mouse remainder
        self._residual_y: float = 0.0  # sub-pixel mouse remainder

        # Persistent motor bias (drifts slowly over seconds, models hand posture)
        self._bias_x: float = random.gauss(0, 0.2)
        self._bias_y: float = random.gauss(0, 0.15)
        self._prev_error_counts: tuple[float, float] = (0.0, 0.0)

    def plan(self, state: PlayerState) -> tuple[int, int]:
        """Compute this tick's mouse counts delta based on current aim state.

        Returns (dx, dy) in raw integer mouse counts.
        """
        aim = state.aim_phase

        if aim in (AimPhase.IDLE, AimPhase.REACTING):
            self._decay_velocity()
            return self._idle_micro_steadiness(state)

        target = state.primary_target
        if target is None:
            self._decay_velocity()
            return (0, 0)

        # Ensure authoritative aim matches the current primary target
        d = target.detection
        w = max(1.0, d.x2 - d.x1)
        h = max(1.0, d.y2 - d.y1)
        if not target.aim_preference:
            head_choice = random.random() < self.personality.head_aim_preference
            target.aim_preference = "head" if head_choice else "chest"
            target.aim_offset_ratio = (0.5, 0.18) if head_choice else (0.5, 0.45)
        rx, ry = target.aim_offset_ratio
        aim_x = d.x1 + w * rx
        aim_y = d.y1 + h * ry

        if (
            not state.authoritative_aim.is_valid
            or state.authoritative_aim.target_id != target.target_id
            or abs(state.authoritative_aim.aim_point_screen[0] - aim_x) > 1.0
            or abs(state.authoritative_aim.aim_point_screen[1] - aim_y) > 1.0
        ):
            err_x = aim_x - self.cx
            err_y = aim_y - self.cy
            err_dist = math.hypot(err_x, err_y)
            mdx, mdy = screen_delta_to_mouse(
                err_x,
                err_y,
                self.sensitivity,
                self.m_yaw,
                self.m_pitch,
                self.screen_w,
                self.screen_h,
                self.fov_h,
            )
            state.authoritative_aim.target_id = target.target_id
            state.authoritative_aim.aim_point_screen = (aim_x, aim_y)
            state.authoritative_aim.screen_error = (err_x, err_y)
            state.authoritative_aim.screen_error_dist = err_dist
            state.authoritative_aim.motor_error = (float(mdx), float(mdy))
            state.authoritative_aim.aim_preference = target.aim_preference
            state.authoritative_aim.is_valid = True

        # Authoritative error already computed once for the tick
        auth_aim = state.authoritative_aim
        mouse_dx, mouse_dy = auth_aim.motor_error
        err_dist_px = auth_aim.screen_error_dist

        if aim == AimPhase.ACQUIRING:
            return self._execute_acquisition_episode(state, mouse_dx, mouse_dy, err_dist_px)
        elif aim == AimPhase.CORRECTING:
            return self._execute_correction_submovement(state, mouse_dx, mouse_dy, err_dist_px)
        elif aim == AimPhase.TRACKING:
            return self._execute_tracking_pursuit(state, target, mouse_dx, mouse_dy, err_dist_px)
        elif aim == AimPhase.REACQUIRING:
            return self._execute_reacquisition(state, mouse_dx, mouse_dy, err_dist_px)

        return (0, 0)

    def plan_scan(self, dx_counts: float, dy_counts: float, state: PlayerState) -> tuple[int, int]:
        """Plan a purposeful camera movement."""
        state.motor_phase = MotorPhase.SCANNING_TURN
        # Smooth camera blend using actual dt
        alpha = min(1.0, max(0.1, state.tick_dt * 12.0))
        self._velocity_x = self._velocity_x * (1.0 - alpha) + dx_counts * alpha
        self._velocity_y = self._velocity_y * (1.0 - alpha) + dy_counts * alpha
        return self._emit(self._velocity_x, self._velocity_y)

    # ── Episode-driven Motor Behaviors ───────────────────────────────────

    def _execute_acquisition_episode(
        self,
        state: PlayerState,
        mouse_dx: float,
        mouse_dy: float,
        err_dist_px: float,
    ) -> tuple[int, int]:
        """Execute a multi-tick reaching movement with bell-shaped velocity profile."""
        p = self.personality
        episode = state.motor_episode
        now = state.now

        # 1. Initialize episode if not active
        if not episode.active:
            episode.active = True
            episode.start_time = now
            # Plan duration based on Fitts's law / personality motor speed
            duration = p.sample_motor_duration(err_dist_px)
            episode.duration = max(0.066, duration)
            episode.start_error_counts = (mouse_dx, mouse_dy)
            episode.target_displacement_counts = (mouse_dx, mouse_dy)
            episode.phase = "primary"
            episode.submovements_planned = (
                1 if (p.motor_precision > 0.65 and err_dist_px < 80) else 2
            )
            episode.submovements_completed = 0
            episode.accepted_error_px = 7.0 + (1.0 - p.motor_precision) * 15.0

        # 2. Progress through primary movement
        elapsed = now - episode.start_time
        progress = min(1.0, elapsed / episode.duration)
        state.motor_progress = progress
        state.motor_phase = MotorPhase.FLICKING

        # Bell-shaped velocity weighting (minimum jerk trajectory profile)
        # Velocity curve: 30 * t^2 * (1-t)^2 normalized
        vel_weight = 30.0 * (progress**2) * ((1.0 - progress) ** 2)
        # Fraction of remaining distance to traverse this tick
        fraction = max(0.15, min(0.95, vel_weight * state.tick_dt * 8.0 + 0.2))

        # Primary movement towards target
        target_vx = mouse_dx * fraction
        target_vy = mouse_dy * fraction

        # Signal-dependent motor noise (proportional to movement amplitude)
        speed = math.hypot(target_vx, target_vy)
        sd_noise = speed * (1.0 - p.motor_precision) * 0.08
        target_vx += random.gauss(0, sd_noise) + self._bias_x * 0.3
        target_vy += random.gauss(0, sd_noise) + self._bias_y * 0.3

        # Blend with current momentum for physical continuity
        blend = min(0.85, 0.4 + p.motor_speed * 0.3)
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        # Check primary movement completion / transition to evaluation
        if progress >= 0.85 or err_dist_px <= max(25.0, episode.accepted_error_px * 2.0):
            episode.phase = "evaluating"
            state.aim_phase = AimPhase.CORRECTING
            state.motor_phase = MotorPhase.SETTLING
            episode.correction_start_time = now
            episode.correction_duration = random.uniform(0.08, 0.14)
            episode.correction_vector = (mouse_dx, mouse_dy)

        return self._emit(self._velocity_x, self._velocity_y)

    def _execute_correction_submovement(
        self,
        state: PlayerState,
        mouse_dx: float,
        mouse_dy: float,
        err_dist_px: float,
    ) -> tuple[int, int]:
        """Execute a discrete corrective submovement (not continuous random jitter)."""
        p = self.personality
        episode = state.motor_episode
        now = state.now
        state.motor_phase = MotorPhase.MICRO_CORRECTING

        # Check if error is within human acceptance threshold
        if err_dist_px <= episode.accepted_error_px:
            # Reached acceptable accuracy -> transition to tracking
            episode.submovements_completed += 1
            episode.active = False
            state.aim_phase = AimPhase.TRACKING
            self._velocity_x *= 0.3
            self._velocity_y *= 0.3
            return self._emit(self._velocity_x, self._velocity_y)

        # Elapsed time in current corrective submovement
        corr_elapsed = now - episode.correction_start_time
        if corr_elapsed >= episode.correction_duration:
            episode.submovements_completed += 1
            if episode.submovements_completed >= episode.submovements_planned or err_dist_px < 25.0:
                # Finished planned submovements -> accept current aim and track
                episode.active = False
                state.aim_phase = AimPhase.TRACKING
                return self._emit(self._velocity_x * 0.4, self._velocity_y * 0.4)
            else:
                # Start second corrective submovement
                episode.correction_start_time = now
                episode.correction_duration = random.uniform(0.07, 0.12)
                episode.correction_vector = (mouse_dx, mouse_dy)

        # Smooth, damped submovement toward target
        kp = 0.20 + p.correction_tendency * 0.25
        target_vx = mouse_dx * kp
        target_vy = mouse_dy * kp

        blend = 0.45
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        return self._emit(self._velocity_x, self._velocity_y)

    def _execute_tracking_pursuit(
        self,
        state: PlayerState,
        target: TrackedTarget,
        mouse_dx: float,
        mouse_dy: float,
        err_dist_px: float,
    ) -> tuple[int, int]:
        """Continuous smooth pursuit tracking with proper velocity unit conversion."""
        p = self.personality
        state.motor_phase = MotorPhase.SMOOTH_TRACKING

        # ── Unit Conversion: Convert target velocity (px/sec) to mouse counts / tick ──
        # Target velocity is estimated in screen pixels/second
        vx_px_s, vy_px_s = target.velocity_estimate
        # Screen displacement in pixels over one tick
        disp_x_px = vx_px_s * state.tick_dt
        disp_y_px = vy_px_s * state.tick_dt

        # Convert screen displacement pixels into mouse counts
        lead_mouse_x, lead_mouse_y = screen_delta_to_mouse(
            disp_x_px,
            disp_y_px,
            self.sensitivity,
            self.m_yaw,
            self.m_pitch,
            self.screen_w,
            self.screen_h,
            self.fov_h,
        )

        # Human tracking lead factor (skilled players predict ahead; casual players lag)
        lead_gain = 0.4 + p.tracking_steadiness * 0.5
        tracking_gain = 0.12 + p.tracking_steadiness * 0.16

        # Velocity in mouse counts
        target_vx = mouse_dx * tracking_gain + float(lead_mouse_x) * lead_gain
        target_vy = mouse_dy * tracking_gain + float(lead_mouse_y) * lead_gain

        # Inertial smoothing
        blend = 0.35 + p.tracking_steadiness * 0.20
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        # Small physiological tremor (correlated, not white noise)
        tremor_scale = (1.0 - p.tracking_steadiness) * 0.4
        self._velocity_x += self._bias_x * tremor_scale
        self._velocity_y += self._bias_y * tremor_scale

        # If tracking error exceeds threshold, trigger a corrective submovement
        if err_dist_px > (45.0 + (1.0 - p.motor_precision) * 35.0):
            state.aim_phase = AimPhase.CORRECTING
            state.motor_episode.active = True
            state.motor_episode.correction_start_time = state.now
            state.motor_episode.correction_duration = 0.10
            state.motor_episode.submovements_planned = 1
            state.motor_episode.submovements_completed = 0

        return self._emit(self._velocity_x, self._velocity_y)

    def _execute_reacquisition(
        self,
        state: PlayerState,
        mouse_dx: float,
        mouse_dy: float,
        err_dist_px: float,
    ) -> tuple[int, int]:
        """Fast re-acquisition of a briefly lost target."""
        p = self.personality
        state.motor_phase = MotorPhase.FLICKING

        gain = 0.35 + p.motor_speed * 0.3
        target_vx = mouse_dx * gain
        target_vy = mouse_dy * gain

        blend = 0.5
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        if err_dist_px < 25.0:
            state.aim_phase = AimPhase.TRACKING

        return self._emit(self._velocity_x, self._velocity_y)

    # ── Helpers ───────────────────────────────────────────────────────────

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
        self._velocity_x *= 0.6
        self._velocity_y *= 0.6

    def _idle_micro_steadiness(self, state: PlayerState) -> tuple[int, int]:
        """Hand steadiness during idle / reaction (no twitching)."""
        p = self.personality
        # Drift motor bias slowly
        self._bias_x += random.gauss(0, 0.02)
        self._bias_y += random.gauss(0, 0.015)
        self._bias_x *= 0.95
        self._bias_y *= 0.95

        if random.random() < 0.15:
            scale = 0.2 * (1.0 - p.motor_precision)
            return self._emit(self._bias_x * scale, self._bias_y * scale)
        return (0, 0)
