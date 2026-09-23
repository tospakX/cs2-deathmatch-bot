"""Central persistent player-behavior state.

Maintains meaningful state across frames and actions.  Every subsystem
reads from and writes to this shared state, ensuring that decisions,
motor actions, firing, and movement are all coherent.

Nothing here is reset every frame.  Resets happen only on death/respawn
or explicit state transitions.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

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


# ── Target record ────────────────────────────────────────────────────────────


@dataclass
class TrackedTarget:
    """Everything the player knows/remembers about a specific target."""

    detection: Detection  # most recent detection
    first_seen: float = 0.0  # time first noticed
    last_seen: float = 0.0  # time last detected
    frames_visible: int = 0  # consecutive frames visible
    frames_missing: int = 0  # consecutive frames NOT visible
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_estimate: tuple[float, float] = (0.0, 0.0)
    is_primary: bool = False  # currently the focused target
    engagement_time: float = 0.0  # how long we've been fighting this one
    shots_at: int = 0  # shots fired at this target
    hits_estimated: int = 0  # estimated hits
    threat_level: float = 0.0  # current threat score
    last_switch_reason: str = ""  # why we switched to/from this target

    def update_position(self, cx: float, cy: float, now: float) -> None:
        """Record a new position observation."""
        self.position_history.append((cx, cy, now))
        self.last_seen = now
        self.frames_visible += 1
        self.frames_missing = 0

        # Estimate velocity from recent positions
        if len(self.position_history) >= 3:
            p_old = self.position_history[-3]
            dt = now - p_old[2]
            if dt > 0.001:
                self.velocity_estimate = (
                    (cx - p_old[0]) / dt,
                    (cy - p_old[1]) / dt,
                )

    def mark_missing(self) -> None:
        """Target not detected this frame."""
        self.frames_missing += 1
        self.frames_visible = 0


# ── Core player state ────────────────────────────────────────────────────────


class PlayerState:
    """Central persistent state for the simulated player.

    This object is the single source of truth for the player's current
    situation.  All behavioral subsystems read and write through it.
    """

    def __init__(self):
        self._now: float = time.perf_counter()

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
        self.awareness_level: float = 0.5  # 0=oblivious, 1=hyper-alert

        # ── Aim / motor state ────────────────────────────────────────────
        self.aim_phase: AimPhase = AimPhase.IDLE
        self.motor_phase: MotorPhase = MotorPhase.IDLE
        self.current_aim_error: tuple[float, float] = (0.0, 0.0)
        self.previous_aim_error: tuple[float, float] = (0.0, 0.0)
        self.motor_velocity: tuple[float, float] = (0.0, 0.0)
        self.aim_target_screen: tuple[float, float] = (0.0, 0.0)
        self.motor_residual: tuple[float, float] = (0.0, 0.0)
        self.motor_phase_start: float = 0.0
        self.motor_phase_duration: float = 0.0
        self.motor_progress: float = 0.0  # 0..1 through current phase

        # ── Firing state ─────────────────────────────────────────────────
        self.firing_phase: FiringPhase = FiringPhase.NOT_FIRING
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
        self.reaction_tendency: float = 1.0  # short-term multiplier

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

    def begin_tick(self) -> None:
        """Call at the start of every main-loop tick."""
        now = time.perf_counter()
        self.tick_dt = now - self._now
        self._now = now
        self.frame_count += 1
        self.last_tick_time = now

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
        """Reset combat state on death.  Keep personality-level memories."""
        self.transition_phase(BotPhase.DEAD)
        self.primary_target = None
        self.known_targets.clear()
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
        self.is_alive = True
        self.awareness_level = 0.5
        self.confidence = max(0.3, self.confidence * 0.9)  # slight reset, not full

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
        # Reset firing state on target switch
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
