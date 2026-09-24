"""Central persistent player-behavior state.

Maintains meaningful state across frames and actions. Every subsystem
reads from and writes to this shared state, ensuring that decisions,
motor actions, firing, and movement are all coherent.

Nothing here is reset every frame. Resets happen only on death/respawn
or explicit state transitions.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from src.utils.clock import Clock, RealClock
from src.vision.detector import Detection

# ── Enumerations ─────────────────────────────────────────────────────────────


class BotPhase(Enum):
    """High-level behavioral phase."""

    DEAD = auto()
    SPAWNING = auto()
    ROAMING = auto()
    SCANNING = auto()
    ENGAGING = auto()
    TRACKING = auto()
    RETREATING = auto()
    RELOADING = auto()


class AimPhase(Enum):
    """Motor phase of the aiming system."""

    IDLE = auto()
    REACTING = auto()  # noticed target, waiting for reaction
    ACQUIRING = auto()  # large flick toward target
    CORRECTING = auto()  # small corrections after acquisition
    TRACKING = auto()  # continuous tracking of moving target
    REACQUIRING = auto()  # lost target briefly, reacquiring


class MotorPhase(Enum):
    """Fine-grained motor execution phase."""

    IDLE = auto()
    FLICKING = auto()  # large ballistic movement
    SETTLING = auto()  # post-flick settle
    MICRO_CORRECTING = auto()  # small corrections
    SMOOTH_TRACKING = auto()  # continuous pursuit
    SCANNING_TURN = auto()  # deliberate camera turn


class FiringPhase(Enum):
    """Firing state machine."""

    NOT_FIRING = auto()
    TAPPING = auto()
    BURSTING = auto()
    SPRAYING = auto()
    BURST_PAUSE = auto()  # brief pause between bursts
    COOLDOWN = auto()  # post-spray cooldown


class MovementPhase(Enum):
    """Movement behavioral state."""

    STATIONARY = auto()
    APPROACHING = auto()
    HOLDING = auto()
    STRAFING = auto()
    RETREATING = auto()
    REPOSITIONING = auto()
    SEARCHING = auto()
    RECOVERING = auto()
    PAUSING = auto()


class TargetStatus(Enum):
    """Visibility and tracking status of an observed target."""

    VISIBLE = auto()
    BRIEFLY_LOST = auto()
    REACQUIRING = auto()
    FORGOTTEN = auto()


# ── Action / Episode records ──────────────────────────────────────────────────


@dataclass
class AuthoritativeAim:
    """Single authoritative aim state consumed by motor planning, firing, and logging."""

    target_id: int = -1
    aim_point_screen: tuple[float, float] = (0.0, 0.0)
    screen_error: tuple[float, float] = (0.0, 0.0)
    screen_error_dist: float = 999.0
    motor_error: tuple[float, float] = (0.0, 0.0)
    aim_preference: str = "chest"  # "head", "chest", "center"
    is_valid: bool = False


@dataclass
class MotorEpisode:
    """Persistent motor plan for an acquisition / correction episode."""

    active: bool = False
    target_id: int = -1
    start_time: float = 0.0
    duration: float = 0.0
    start_error_counts: tuple[float, float] = (0.0, 0.0)
    target_displacement_counts: tuple[float, float] = (0.0, 0.0)
    traversed_fraction: float = 0.0
    phase: str = (
        "idle"  # "primary", "decelerating", "evaluating", "corrective", "settling", "pursuit"
    )
    trajectory_shape: str = "minimum_jerk"
    peak_velocity: float = 0.0  # planned peak velocity in mouse counts/s
    deceleration_point: float = 0.50  # normalized tau where deceleration begins
    submovements_planned: int = 1
    submovements_completed: int = 0
    correction_start_time: float = 0.0
    correction_duration: float = 0.0
    correction_vector: tuple[float, float] = (0.0, 0.0)
    accepted_error_px: float = 12.0
    abort_threshold_px: float = 180.0
    completed: bool = False
    aborted: bool = False


@dataclass
class FiringEpisode:
    """Committed firing episode preventing tick-by-tick mode flapping."""

    active: bool = False
    mode: str = "none"  # "tap", "burst", "spray"
    start_time: float = 0.0
    shots_planned: int = 0
    shots_fired: int = 0
    pause_until: float = 0.0
    cooldown_until: float = 0.0
    trigger_hold_duration: float = 0.0
    press_time: float = 0.0


@dataclass
class CrouchEpisode:
    """Persistent crouch episode avoiding 30Hz key fluttering."""

    is_crouching: bool = False
    start_time: float = 0.0
    duration: float = 0.0
    cooldown_until: float = 0.0


@dataclass
class ReloadEpisode:
    """Persistent reload episode machine."""

    is_reloading: bool = False
    start_time: float = 0.0
    expected_duration: float = 2.6
    key_pressed: bool = False
    grace_until: float = 0.0


@dataclass
class SpatialMemory:
    """Persistent directional memory of an enemy with true last_seen timestamp."""

    screen_x: float  # Historical screen pixel reference (diagnostic/reference only)
    screen_y: float  # Historical screen pixel reference (diagnostic/reference only)
    yaw_offset_deg: float  # Current angular yaw offset relative to camera view
    pitch_offset_deg: float  # Current angular pitch offset relative to camera view
    last_seen_time: float
    confidence: float
    target_id: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)  # Screen px/s estimated velocity
    uncertainty_deg: float = 1.0  # Grows with time elapsed since observation
    decay_factor: float = 1.0  # Decays exponentially with age

    def update_camera_rotation(self, yaw_delta_deg: float, pitch_delta_deg: float) -> None:
        """Update relative angular offset when camera rotates."""
        self.yaw_offset_deg = (self.yaw_offset_deg - yaw_delta_deg + 180.0) % 360.0 - 180.0
        self.pitch_offset_deg = max(-89.0, min(89.0, self.pitch_offset_deg - pitch_delta_deg))

    def advance_time(
        self,
        dt: float,
        current_time: float,
        fov_h: float = 122.0,
        screen_w: int = 3440,
        screen_h: int = 1440,
    ) -> None:
        """Advance time: extrapolate target velocity, grow uncertainty, and decay confidence."""
        age = current_time - self.last_seen_time
        aspect = screen_w / max(1.0, float(screen_h))
        fov_v = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov_h / 2.0)) / aspect))
        vx_px_s, vy_px_s = self.velocity
        yaw_vel = (vx_px_s / max(1.0, float(screen_w))) * fov_h
        pitch_vel = (vy_px_s / max(1.0, float(screen_h))) * fov_v
        self.yaw_offset_deg += yaw_vel * dt
        self.pitch_offset_deg += pitch_vel * dt
        self.uncertainty_deg = 1.0 + age * 2.0
        self.decay_factor = math.exp(-age / 3.0)


# ── Target record ────────────────────────────────────────────────────────────


@dataclass
class TrackedTarget:
    """Everything the player knows/remembers about a specific target."""

    detection: Detection  # most recent detection
    target_id: int = 0  # unique ID for target identity
    first_seen: float = 0.0  # time first noticed
    last_seen: float = 0.0  # time last detected
    frames_visible: int = 0  # consecutive frames visible
    frames_missing: int = 0  # consecutive frames NOT visible
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_estimate: tuple[float, float] = (0.0, 0.0)  # in screen pixels per second
    is_primary: bool = False  # currently the focused target
    engagement_time: float = 0.0  # how long we've been fighting this one
    shots_at: int = 0  # shots fired at this target
    hits_estimated: int = 0  # estimated hits
    threat_level: float = 0.0  # current threat score
    last_switch_reason: str = ""  # why we switched to/from this target

    # Persistent aim preference: decided upon acquisition, never rerolled every tick!
    aim_preference: str = ""  # "head" or "chest"
    aim_offset_ratio: tuple[float, float] = (
        0.5,
        0.2,
    )  # Relative (x, y) offset within detection bbox
    initial_error_dist: float = 0.0
    status: TargetStatus = TargetStatus.VISIBLE

    def update_position(self, cx: float, cy: float, now: float) -> None:
        """Record a new position observation."""
        self.position_history.append((cx, cy, now))
        self.last_seen = now
        self.frames_visible += 1
        self.frames_missing = 0
        self.status = TargetStatus.VISIBLE

        # Estimate velocity from recent positions (screen pixels / second)
        if len(self.position_history) >= 3:
            p_old = self.position_history[-3]
            dt = now - p_old[2]
            if dt > 0.001:
                # Exponential smoothing of velocity
                new_vx = (cx - p_old[0]) / dt
                new_vy = (cy - p_old[1]) / dt
                old_vx, old_vy = self.velocity_estimate
                alpha = 0.4
                self.velocity_estimate = (
                    old_vx * (1 - alpha) + new_vx * alpha,
                    old_vy * (1 - alpha) + new_vy * alpha,
                )

    def mark_missing(self) -> None:
        """Target not detected this frame."""
        self.frames_missing += 1
        self.frames_visible = 0
        if self.frames_missing <= 4:
            self.status = TargetStatus.BRIEFLY_LOST
        elif self.frames_missing <= 15:
            self.status = TargetStatus.REACQUIRING
        else:
            self.status = TargetStatus.FORGOTTEN


# ── Core player state ────────────────────────────────────────────────────────


class PlayerState:
    """Central persistent state for the simulated player.

    This object is the single source of truth for the player's current
    situation. All behavioral subsystems read and write through it.
    """

    def __init__(self, clock: Clock | None = None):
        self.clock: Clock = clock if clock is not None else RealClock()
        self._now: float = self.clock.now()

        # ── Phase state ──────────────────────────────────────────────────
        self.phase: BotPhase = BotPhase.DEAD
        self.phase_start: float = self._now
        self.previous_phase: BotPhase = BotPhase.DEAD

        # ── Target state ─────────────────────────────────────────────────
        self.primary_target: TrackedTarget | None = None
        self.previous_target: TrackedTarget | None = None
        self.known_targets: dict[int, TrackedTarget] = {}  # keyed by track_id
        self.target_switch_history: deque = deque(maxlen=10)
        self.target_acquisition_time: float = 0.0
        self._next_track_id: int = 0

        # ── Attention state ──────────────────────────────────────────────
        self.attention_focus: tuple[float, float] | None = None  # screen coords
        self.attention_confidence: float = 0.0  # 0=unfocused, 1=locked
        self.peripheral_alerts: deque = deque(maxlen=5)
        self.last_enemy_positions: deque = deque(maxlen=10)
        self.spatial_memories: deque[SpatialMemory] = deque(maxlen=10)
        self.awareness_level: float = 0.5  # 0=oblivious, 1=hyper-alert

        # ── Authoritative Aim state (shared across all subsystems) ───────
        self.authoritative_aim: AuthoritativeAim = AuthoritativeAim()

        # ── Aim / motor state & episodes ─────────────────────────────────
        self.aim_phase: AimPhase = AimPhase.IDLE
        self.motor_phase: MotorPhase = MotorPhase.IDLE
        self.motor_episode: MotorEpisode = MotorEpisode()
        self.current_aim_error: tuple[float, float] = (0.0, 0.0)
        self.previous_aim_error: tuple[float, float] = (0.0, 0.0)
        self.motor_velocity: tuple[float, float] = (0.0, 0.0)
        self.aim_target_screen: tuple[float, float] = (0.0, 0.0)
        self.motor_residual: tuple[float, float] = (0.0, 0.0)
        self.motor_phase_start: float = 0.0
        self.motor_phase_duration: float = 0.0
        self.motor_progress: float = 0.0  # 0..1 through current phase

        # ── Firing state & episodes ──────────────────────────────────────
        self.firing_phase: FiringPhase = FiringPhase.NOT_FIRING
        self.firing_episode: FiringEpisode = FiringEpisode()
        self.crouch_episode: CrouchEpisode = CrouchEpisode()
        self.reload_episode: ReloadEpisode = ReloadEpisode()
        self.is_trigger_held: bool = False
        self.shot_count: int = 0  # actual shots this spray
        self.burst_shot_target: int = 0  # target shots for current burst
        self.last_shot_time: float = 0.0
        self.shots_since_reset: int = 0
        self.recent_shots: deque = deque(maxlen=30)  # timestamps
        self.firing_start_time: float = 0.0
        self.burst_pause_until: float = 0.0

        # ── Recoil state ─────────────────────────────────────────────────
        self.recoil_shot_index: int = 0
        self.recoil_accumulated: tuple[float, float] = (0.0, 0.0)
        self.weapon: str = "default"

        # ── Movement state ───────────────────────────────────────────────
        self.movement_phase: MovementPhase = MovementPhase.STATIONARY
        self.movement_phase_start: float = 0.0
        self.movement_direction: str = ""  # current key direction
        self.movement_duration: float = 0.0  # planned duration
        self.held_keys: set[str] = set()
        self.is_crouching: bool = False
        self.recent_movement_tendencies: deque = deque(maxlen=10)

        # ── Health / HUD ─────────────────────────────────────────────────
        self.health: int = 100
        self.ammo_clip: int = 30
        self.is_alive: bool = False
        self.last_damage_time: float = 0.0
        self.damage_taken_recently: int = 0

        # ── Reaction state ───────────────────────────────────────────────
        self.reaction_pending: bool = False
        self.reaction_start: float = 0.0
        self.reaction_duration: float = 0.0
        self.reaction_context: str = ""
        self.recent_reaction_times: deque = deque(maxlen=10)
        self.reaction_tendency: float = 1.0  # unified short-term multiplier (drifts slowly)

        # ── Confidence / adaptation ──────────────────────────────────────
        self.confidence: float = 0.5
        self.recent_hits: deque = deque(maxlen=20)
        self.recent_misses: deque = deque(maxlen=20)
        self.recent_kills: deque = deque(maxlen=10)
        self.recent_deaths: deque = deque(maxlen=10)
        self.engagement_outcomes: deque = deque(maxlen=20)

        # ── Temporal continuity ──────────────────────────────────────────
        self.frame_count: int = 0
        self.session_start: float = self._now
        self.last_tick_time: float = self._now
        self.tick_dt: float = 0.033

    # ── Time management ──────────────────────────────────────────────────

    @property
    def now(self) -> float:
        return self._now

    def begin_tick(self, now: float | None = None) -> None:
        """Call at the start of every main-loop tick."""
        current_time = now if now is not None else self.clock.now()
        self.tick_dt = max(0.001, min(0.2, current_time - self._now))
        self._now = current_time
        self.frame_count += 1
        self.last_tick_time = current_time

    @property
    def time_in_phase(self) -> float:
        return self._now - self.phase_start

    # ── Phase transitions ────────────────────────────────────────────────

    def transition_phase(self, new_phase: BotPhase) -> None:
        """Transition to a new behavioral phase."""
        if new_phase == self.phase:
            return
        self.previous_phase = self.phase
        self.phase = new_phase
        self.phase_start = self._now

    # ── Death / respawn ──────────────────────────────────────────────────

    def on_death(self) -> None:
        """Reset combat state on death. Keep personality-level memories."""
        self.transition_phase(BotPhase.DEAD)
        self.primary_target = None
        self.known_targets.clear()
        self.authoritative_aim = AuthoritativeAim()
        self.motor_episode = MotorEpisode()
        self.firing_episode = FiringEpisode()
        self.crouch_episode = CrouchEpisode()
        self.reload_episode = ReloadEpisode()
        self.aim_phase = AimPhase.IDLE
        self.motor_phase = MotorPhase.IDLE
        self.firing_phase = FiringPhase.NOT_FIRING
        self.is_trigger_held = False
        self.shot_count = 0
        self.recoil_shot_index = 0
        self.recoil_accumulated = (0.0, 0.0)
        self.movement_phase = MovementPhase.STATIONARY
        self.held_keys.clear()
        self.is_crouching = False
        self.reaction_pending = False
        self.attention_confidence = 0.0
        self.motor_velocity = (0.0, 0.0)
        self.recent_deaths.append(self._now)

    def on_respawn(self) -> None:
        """Transition from dead to alive."""
        self.transition_phase(BotPhase.ROAMING)
        self.health = 100
        self.ammo_clip = 30
        self.is_alive = True
        self.awareness_level = 0.5
        self.confidence = max(0.3, self.confidence * 0.9)  # slight reset, not full
        self.authoritative_aim = AuthoritativeAim()
        self.motor_episode = MotorEpisode()
        self.firing_episode = FiringEpisode()
        self.crouch_episode = CrouchEpisode()
        self.reload_episode = ReloadEpisode()

    # ── Firing events ────────────────────────────────────────────────────

    def record_shot(self) -> None:
        """Call when an actual shot is fired (trigger event)."""
        self.shot_count += 1
        self.shots_since_reset += 1
        self.recoil_shot_index += 1
        self.last_shot_time = self._now
        self.recent_shots.append(self._now)
        if self.primary_target is not None:
            self.primary_target.shots_at += 1

    def reset_spray(self) -> None:
        """Reset spray/recoil tracking."""
        self.shot_count = 0
        self.recoil_shot_index = 0
        self.recoil_accumulated = (0.0, 0.0)

    # ── Target management ────────────────────────────────────────────────

    def assign_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def switch_target(self, new_target: TrackedTarget | None, reason: str) -> None:
        """Switch primary target with history tracking."""
        if self.primary_target is not None:
            self.primary_target.is_primary = False
        self.previous_target = self.primary_target
        self.primary_target = new_target
        if new_target is not None:
            new_target.is_primary = True
            new_target.last_switch_reason = reason
        self.target_switch_history.append((self._now, reason))
        self.target_acquisition_time = self._now

        # Reset episode states on target switch
        self.authoritative_aim = AuthoritativeAim()
        self.motor_episode = MotorEpisode()
        self.firing_episode = FiringEpisode()
        self.reset_spray()

    # ── Adaptation helpers ───────────────────────────────────────────────

    def update_confidence(self, hit: bool) -> None:
        """Adjust confidence based on recent combat performance."""
        if hit:
            self.recent_hits.append(self._now)
            self.confidence = min(1.0, self.confidence + 0.02)
        else:
            self.recent_misses.append(self._now)
            self.confidence = max(0.1, self.confidence - 0.01)

    def record_engagement_outcome(self, won: bool) -> None:
        """Record whether we won or lost an engagement."""
        self.engagement_outcomes.append((self._now, won))
        if won:
            self.confidence = min(1.0, self.confidence + 0.05)
            self.recent_kills.append(self._now)
        else:
            self.confidence = max(0.1, self.confidence - 0.03)

    # ── Camera & Spatial Memory Tracking ─────────────────────────────────

    def on_camera_rotated(self, yaw_delta_deg: float, pitch_delta_deg: float) -> None:
        """Update angular offsets of stored spatial memories when camera rotates."""
        for mem in self.spatial_memories:
            mem.update_camera_rotation(yaw_delta_deg, pitch_delta_deg)

    def advance_spatial_memories(
        self,
        dt: float,
        fov_h: float = 122.0,
        screen_w: int = 3440,
        screen_h: int = 1440,
    ) -> None:
        """Advance time for all active spatial memories."""
        to_prune = []
        for mem in self.spatial_memories:
            mem.advance_time(dt, self._now, fov_h, screen_w, screen_h)
            # Prune if too old or too decayed
            if (self._now - mem.last_seen_time > 8.0) or (mem.confidence * mem.decay_factor < 0.12):
                to_prune.append(mem)
        for m in to_prune:
            try:
                self.spatial_memories.remove(m)
            except ValueError:
                pass
