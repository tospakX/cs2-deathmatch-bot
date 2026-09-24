"""Continuous motor planner — models human motor control & reaching submovements.

Models distinct motor phases according to human motor research:
1. Primary movement (ballistic flick with bell-shaped minimum-jerk velocity profile)
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

from src.behavior.player_state import AimPhase, MotorPhase, TargetStatus
from src.utils.math_helpers import screen_delta_to_mouse

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class MotorPlanner:
    """Non-blocking continuous motor system executing structured motor episodes.

    Every tick or sub-step, call ``plan()`` to obtain this interval's integer (dx, dy)
    mouse counts. The planner executes an active MotorEpisode across multiple frames
    rather than resampling trajectories on each 30Hz tick.
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

    def plan(
        self,
        state: PlayerState,
        dt: float | None = None,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Compute this step's mouse counts delta based on current aim state.

        Supports injectable monotonic time `now` and step interval `dt` for
        high-cadence motor execution decoupling.
        """
        t_now = now if now is not None else state.now
        t_dt = dt if dt is not None else max(0.005, min(0.100, state.tick_dt))

        aim = state.aim_phase
        if aim in (AimPhase.IDLE, AimPhase.REACTING):
            self._decay_velocity()
            return self._idle_micro_steadiness(state)

        target = state.primary_target
        if target is None or target.status == TargetStatus.FORGOTTEN:
            self._decay_velocity()
            return (0, 0)

        # If target is briefly lost or reacquiring, execute reacquisition via spatial memory
        if (
            target.status in (TargetStatus.BRIEFLY_LOST, TargetStatus.REACQUIRING)
            or aim == AimPhase.REACQUIRING
        ):
            return self._execute_reacquisition(state, t_now, t_dt)

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
            return self._execute_acquisition_episode(
                state, target, mouse_dx, mouse_dy, err_dist_px, t_now, t_dt
            )
        elif aim == AimPhase.CORRECTING:
            return self._execute_correction_submovement(
                state, mouse_dx, mouse_dy, err_dist_px, t_now
            )
        elif aim == AimPhase.TRACKING:
            return self._execute_tracking_pursuit(
                state, target, mouse_dx, mouse_dy, err_dist_px, t_dt, t_now
            )

        return (0, 0)

    def plan_scan(
        self,
        dx_counts: float,
        dy_counts: float,
        state: PlayerState,
        dt: float | None = None,
    ) -> tuple[int, int]:
        """Plan a purposeful camera movement."""
        step_dt = dt if dt is not None else state.tick_dt
        state.motor_phase = MotorPhase.SCANNING_TURN
        # Smooth camera blend using step dt
        alpha = min(1.0, max(0.1, step_dt * 12.0))
        self._velocity_x = self._velocity_x * (1.0 - alpha) + dx_counts * alpha
        self._velocity_y = self._velocity_y * (1.0 - alpha) + dy_counts * alpha
        return self._emit(self._velocity_x, self._velocity_y)

    # ── Episode-driven Motor Behaviors ───────────────────────────────────

    def _execute_acquisition_episode(
        self,
        state: PlayerState,
        target: TrackedTarget,
        mouse_dx: float,
        mouse_dy: float,
        err_dist_px: float,
        now: float,
        dt: float,
    ) -> tuple[int, int]:
        """Execute a plan-first reaching movement with minimum jerk velocity profile."""
        p = self.personality
        episode = state.motor_episode

        # 1. Initialize episode if not active or target changed
        if not episode.active or episode.target_id != target.target_id:
            episode.active = True
            episode.target_id = target.target_id
            episode.start_time = now
            duration = p.sample_motor_duration(err_dist_px)
            episode.duration = max(0.066, duration)
            episode.start_error_counts = (mouse_dx, mouse_dy)
            episode.target_displacement_counts = (mouse_dx, mouse_dy)
            episode.traversed_fraction = 0.0
            episode.phase = "primary"
            episode.trajectory_shape = "minimum_jerk"
            dist_counts = math.hypot(mouse_dx, mouse_dy)
            episode.peak_velocity = (dist_counts / episode.duration) * 1.875
            episode.deceleration_point = 0.50
            episode.submovements_planned = (
                1 if (p.motor_precision > 0.65 and err_dist_px < 80) else 2
            )
            episode.submovements_completed = 0
            episode.accepted_error_px = 7.0 + (1.0 - p.motor_precision) * 15.0
            episode.abort_threshold_px = max(180.0, err_dist_px * 2.5)
            episode.completed = False
            episode.aborted = False

        # 2. Check abort condition
        if err_dist_px > episode.abort_threshold_px:
            episode.aborted = True
            episode.active = False
            state.aim_phase = AimPhase.IDLE
            self._decay_velocity()
            return (0, 0)

        # 3. Progress through primary movement strictly as a function of elapsed episode time
        elapsed = (now - episode.start_time) + dt
        tau = min(1.0, max(0.0, elapsed / episode.duration))
        state.motor_progress = tau

        if tau < episode.deceleration_point:
            episode.phase = "primary"
            state.motor_phase = MotorPhase.FLICKING
        else:
            episode.phase = "decelerating"
            state.motor_phase = MotorPhase.SETTLING

        # Minimum jerk normalized position: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s_pos = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
        # Fractional step for this execution interval
        delta_s = max(0.0, s_pos - episode.traversed_fraction)
        episode.traversed_fraction = s_pos

        # Base displacement generated strictly from the plan
        disp_x, disp_y = episode.target_displacement_counts
        target_vx = disp_x * delta_s
        target_vy = disp_y * delta_s

        # Signal-dependent neuromotor noise (proportional to planned movement amplitude)
        speed = math.hypot(target_vx, target_vy)
        sd_noise = speed * (1.0 - p.motor_precision) * 0.05
        target_vx += random.gauss(0, sd_noise) + self._bias_x * 0.20
        target_vy += random.gauss(0, sd_noise) + self._bias_y * 0.20

        # Blend with current momentum for physical continuity
        blend = min(0.85, 0.40 + p.motor_speed * 0.30)
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        # 4. Check primary movement completion / transition to evaluation
        reached_endpoint = (
            tau >= 0.90
            or episode.traversed_fraction >= 0.98
            or err_dist_px <= max(25.0, episode.accepted_error_px * 2.0)
        )
        if reached_endpoint:
            episode.phase = "evaluating"
            if err_dist_px <= episode.accepted_error_px:
                # Within human acceptance threshold -> transition to tracking directly
                episode.completed = True
                episode.active = False
                state.aim_phase = AimPhase.TRACKING
                self._velocity_x *= 0.4
                self._velocity_y *= 0.4
            else:
                # Outside acceptable threshold -> plan discrete corrective submovement
                episode.phase = "corrective"
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
        now: float,
    ) -> tuple[int, int]:
        """Execute a discrete corrective submovement (not continuous random jitter)."""
        p = self.personality
        episode = state.motor_episode
        state.motor_phase = MotorPhase.MICRO_CORRECTING

        # Check if error is within human acceptance threshold
        if err_dist_px <= episode.accepted_error_px:
            # Reached acceptable accuracy -> transition to tracking
            episode.submovements_completed += 1
            episode.completed = True
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
                episode.completed = True
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
        dt: float,
        now: float,
    ) -> tuple[int, int]:
        """Continuous smooth pursuit tracking with proper velocity unit conversion."""
        p = self.personality
        state.motor_phase = MotorPhase.SMOOTH_TRACKING

        # ── Unit Conversion: Convert target velocity (px/sec) to mouse counts / step ──
        vx_px_s, vy_px_s = target.velocity_estimate
        disp_x_px = vx_px_s * dt
        disp_y_px = vy_px_s * dt

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

        lead_gain = 0.4 + p.tracking_steadiness * 0.5
        tracking_gain = 0.12 + p.tracking_steadiness * 0.16

        target_vx = mouse_dx * tracking_gain + float(lead_mouse_x) * lead_gain
        target_vy = mouse_dy * tracking_gain + float(lead_mouse_y) * lead_gain

        # Inertial smoothing
        blend = 0.35 + p.tracking_steadiness * 0.20
        self._velocity_x = self._velocity_x * (1.0 - blend) + target_vx * blend
        self._velocity_y = self._velocity_y * (1.0 - blend) + target_vy * blend

        # Small physiological tremor (correlated)
        tremor_scale = (1.0 - p.tracking_steadiness) * 0.4
        self._velocity_x += self._bias_x * tremor_scale
        self._velocity_y += self._bias_y * tremor_scale

        # If tracking error exceeds threshold, trigger a corrective submovement
        if err_dist_px > (45.0 + (1.0 - p.motor_precision) * 35.0):
            state.aim_phase = AimPhase.CORRECTING
            state.motor_episode.active = True
            state.motor_episode.correction_start_time = now
            state.motor_episode.correction_duration = 0.10
            state.motor_episode.submovements_planned = 1
            state.motor_episode.submovements_completed = 0

        return self._emit(self._velocity_x, self._velocity_y)

    def _execute_reacquisition(
        self,
        state: PlayerState,
        now: float,
        dt: float,
    ) -> tuple[int, int]:
        """Fast re-acquisition of a briefly lost target using spatial memory."""
        p = self.personality
        state.motor_phase = MotorPhase.FLICKING

        # Check if we have an active spatial memory for where the enemy was / went
        valid_memories = [m for m in state.spatial_memories if (now - m.last_seen_time) < 5.0]
        if not valid_memories:
            self._decay_velocity()
            return (0, 0)

        # Pick highest confidence memory
        mem = max(
            valid_memories,
            key=lambda m: (m.confidence * m.decay_factor) / max(0.5, m.uncertainty_deg),
        )

        deg_per_count_x = self.m_yaw * self.sensitivity
        deg_per_count_y = self.m_pitch * self.sensitivity
        target_dx = mem.yaw_offset_deg / max(1e-5, deg_per_count_x)
        target_dy = mem.pitch_offset_deg / max(1e-5, deg_per_count_y)

        # Smooth camera reorientation towards remembered location with realistic human speed cap
        gain = min(0.40, (0.25 + p.motor_speed * 0.20) * (dt / 0.033))
        step_vx = target_dx * gain
        step_vy = target_dy * gain
        max_speed = 350.0 + p.motor_speed * 450.0  # realistic human turn speed cap
        speed = math.hypot(step_vx, step_vy)
        if speed > max_speed:
            scale = max_speed / speed
            step_vx *= scale
            step_vy *= scale

        self._velocity_x = self._velocity_x * 0.6 + step_vx
        self._velocity_y = self._velocity_y * 0.6 + step_vy

        # If aligned with remembered location, settle
        if math.hypot(target_dx, target_dy) < 30.0:
            state.aim_phase = AimPhase.IDLE

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
