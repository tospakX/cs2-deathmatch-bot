"""Stateful combat movement controller.

Models human-like combat footwork:
- Direction commitment with momentum and history (no mechanical left-right-left-right flipping)
- Explicit CounterStrafeState lifecycle machine: INACTIVE -> BRAKING -> SETTLED -> INACTIVE
- Contextual combat movement:
    * Approaching: closing distance when weapon or situation favors it
    * Holding: holding position when holding an angle or burst-firing
    * Strafing: rhythmic combat strafes with varied duration and pauses
    * Retreating: low health, seeking cover, moving backwards while breaking sight
    * Repositioning: relocating between bursts or after missing shots
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from src.behavior.player_state import MovementPhase

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class CounterStrafeState(Enum):
    """Lifecycle state of a counter-strafe maneuver."""

    INACTIVE = auto()
    BRAKING = auto()  # Counter-strafe key pressed to kill velocity (~60-80ms)
    SETTLED = auto()  # Lateral velocity dead, settling pause before next move (~25-45ms)


@dataclass
class MovementEpisode:
    """Persistent movement footwork episode."""

    active: bool = False
    phase: MovementPhase = MovementPhase.STATIONARY
    direction: str = ""  # "left", "right", "forward", "back"
    start_time: float = 0.0
    duration: float = 0.0
    same_dir_streak: int = 0
    braking_done: bool = False


class MovementController:
    """Manages continuous stateful movement during combat and roaming."""

    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self.episode = MovementEpisode()
        self._current_phase: MovementPhase = MovementPhase.STATIONARY
        self._phase_start: float = 0.0
        self._phase_duration: float = 0.0
        self._current_direction: str = ""  # "left", "right", "forward", "back"

        # Direction history and streak tracking
        self._direction_history: deque[str] = deque(maxlen=10)
        self._same_dir_streak: int = 0

        # Counter-strafing explicit lifecycle state machine
        self._cs_state: CounterStrafeState = CounterStrafeState.INACTIVE
        self._counter_strafe_end: float = 0.0
        self._counter_strafe_settle_end: float = 0.0
        self._counter_strafe_key: str = ""
        self._braking_done_for_movement: bool = False

    @property
    def _counter_strafing(self) -> bool:
        return self._cs_state == CounterStrafeState.BRAKING

    @_counter_strafing.setter
    def _counter_strafing(self, value: bool) -> None:
        if value:
            self._cs_state = CounterStrafeState.BRAKING
        else:
            self._cs_state = CounterStrafeState.INACTIVE

    @property
    def is_settled(self) -> bool:
        """True if player is counter-strafed and settled with zero lateral velocity."""
        return self._cs_state in (
            CounterStrafeState.SETTLED,
            CounterStrafeState.INACTIVE,
        ) and self._current_phase in (
            MovementPhase.HOLDING,
            MovementPhase.STATIONARY,
        )

    def check_lifecycle(self, now: float) -> tuple[dict[str, bool], bool]:
        """High-cadence check for counter-strafe impulse & settle transitions.

        Called between 30Hz frames to release braking key at the exact planned millisecond.
        Returns:
            (key_changes, state_changed)
        """
        if self._cs_state == CounterStrafeState.BRAKING:
            if now >= self._counter_strafe_end:
                self._cs_state = CounterStrafeState.SETTLED
                # Return instruction to release braking key immediately
                return {self._counter_strafe_key: False}, True
        elif self._cs_state == CounterStrafeState.SETTLED:
            if now >= self._counter_strafe_settle_end:
                self._cs_state = CounterStrafeState.INACTIVE
                return {}, True
        return {}, False

    def update(
        self,
        state: PlayerState,
        target: TrackedTarget | None,
        is_firing: bool,
    ) -> dict[str, bool]:
        """Compute movement key states for this tick."""
        now = state.now
        p = self.personality
        keys = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "crouch": False,
            "walk": False,
        }

        # ── 1. Counter-strafing State Machine Execution ──────────────────────
        if self._cs_state == CounterStrafeState.BRAKING:
            if now < self._counter_strafe_end:
                if self._counter_strafe_key in keys:
                    keys[self._counter_strafe_key] = True
                return keys
            else:
                self._cs_state = CounterStrafeState.SETTLED

        if self._cs_state == CounterStrafeState.SETTLED:
            if now < self._counter_strafe_settle_end:
                # Settling window: stationary, dead stop for 100% first-shot accuracy
                return keys
            else:
                self._cs_state = CounterStrafeState.INACTIVE

        # ── 2. Determine / transition movement phase ─────────────────────────
        phase_elapsed = now - self._phase_start
        should_rethink = (
            phase_elapsed >= self._phase_duration or self._current_phase == MovementPhase.STATIONARY
        )

        if state.health <= p.disengage_health and target is not None:
            # Low health: prioritize retreating
            if self._current_phase != MovementPhase.RETREATING:
                self._transition_to(MovementPhase.RETREATING, now, random.uniform(1.2, 2.5))
        elif target is not None:
            # In combat with an enemy
            if should_rethink:
                self._select_combat_phase(state, target, is_firing, now)
        else:
            # Roaming / idle
            if should_rethink:
                self._select_roam_phase(state, now)

        # ── 3. Execute Current Phase ─────────────────────────────────────────
        if self._current_phase == MovementPhase.STRAFING:
            # Low strafe tendency players (or players firing precision taps) brake to shoot
            if is_firing and p.strafe_tendency <= 0.6:
                if not self._braking_done_for_movement and self._current_direction in (
                    "left",
                    "right",
                ):
                    self._trigger_counter_strafe(self._current_direction, now)
                    if self._counter_strafe_key in keys:
                        keys[self._counter_strafe_key] = True
            else:
                if self._current_direction in keys:
                    keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.APPROACHING:
            keys["forward"] = True
            if self._current_direction in ("left", "right"):
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.RETREATING:
            keys["back"] = True
            if self._current_direction in ("left", "right"):
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.REPOSITIONING:
            if self._current_direction in keys:
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.HOLDING:
            # Stationary, holding angle / settling recoil
            pass

        # ── 4. Crouch state from player_state / crouch_episode ───────────────
        if state.is_crouching:
            keys["crouch"] = True

        state.movement_phase = self._current_phase
        state.movement_direction = self._current_direction
        state.movement_phase_start = self._phase_start
        state.movement_duration = self._phase_duration
        self.episode.phase = self._current_phase
        self.episode.direction = self._current_direction
        return keys

    def _select_combat_phase(
        self,
        state: PlayerState,
        target: TrackedTarget,
        is_firing: bool,
        now: float,
    ) -> None:
        """Choose combat movement phase based on distance, history, and personality."""
        p = self.personality
        est_distance = 1000.0 / max(target.detection.area**0.5, 1.0)

        # Close range (< 8m): rapid dynamic strafes or crouch commitments
        if est_distance < 8.0:
            direction = self._sample_next_direction()
            duration = random.uniform(0.35, 0.70) * (0.8 + (1.0 - p.motor_precision) * 0.4)
            self._current_direction = direction
            self._transition_to(MovementPhase.STRAFING, now, duration)

        # Medium range (8-18m): rhythmic combat strafes with counter-strafe stops
        elif est_distance < 18.0:
            roll = random.random()
            if roll < p.strafe_tendency:
                direction = self._sample_next_direction()
                duration = random.uniform(0.40, 0.85)
                self._current_direction = direction
                self._transition_to(MovementPhase.STRAFING, now, duration)
            elif roll < p.strafe_tendency + p.repositioning_tendency:
                self._current_direction = random.choice(["left", "right", "back"])
                duration = random.uniform(0.30, 0.60)
                self._transition_to(MovementPhase.REPOSITIONING, now, duration)
            else:
                self._transition_to(MovementPhase.HOLDING, now, random.uniform(0.3, 0.7))

        # Long range (> 18m): holding angle or micro-repositioning
        else:
            if random.random() < 0.6:
                self._transition_to(MovementPhase.HOLDING, now, random.uniform(0.4, 0.9))
            else:
                direction = self._sample_next_direction()
                self._current_direction = direction
                self._transition_to(MovementPhase.STRAFING, now, random.uniform(0.30, 0.55))

    def _sample_next_direction(self) -> str:
        """Sample next strafe direction based on streak momentum and history."""
        last_dir = self._direction_history[-1] if self._direction_history else ""

        if not last_dir or last_dir not in ("left", "right"):
            chosen = random.choice(["left", "right"])
            self._same_dir_streak = 1
        else:
            opposite = "right" if last_dir == "left" else "left"
            roll = random.random()
            # Streak-dependent momentum:
            # streak 1: ~35% repeat (double strafe), ~65% reverse
            # streak 2: ~15% repeat, ~85% reverse
            # streak >= 3: ~5% repeat, ~95% reverse
            if self._same_dir_streak <= 1:
                repeat_prob = 0.35
            elif self._same_dir_streak == 2:
                repeat_prob = 0.15
            else:
                repeat_prob = 0.05

            if roll < repeat_prob:
                chosen = last_dir
                self._same_dir_streak += 1
            else:
                chosen = opposite
                self._same_dir_streak = 1

        self._direction_history.append(chosen)
        self.episode.same_dir_streak = self._same_dir_streak
        return chosen

    def _select_roam_phase(self, state: PlayerState, now: float) -> None:
        """Choose roaming movement state."""
        self._transition_to(MovementPhase.APPROACHING, now, random.uniform(1.2, 3.0))

    def _trigger_counter_strafe(self, moving_direction: str, now: float) -> None:
        """Initiate single counter-strafe braking event."""
        opposite = {"left": "right", "right": "left", "forward": "back", "back": "forward"}
        opp_key = opposite.get(moving_direction, "")
        if opp_key and self._cs_state == CounterStrafeState.INACTIVE:
            self._cs_state = CounterStrafeState.BRAKING
            # CS2 counter-strafe impulse is ~60-80ms
            self._counter_strafe_end = now + random.uniform(0.060, 0.080)
            self._counter_strafe_settle_end = self._counter_strafe_end + random.uniform(
                0.025, 0.045
            )
            self._counter_strafe_key = opp_key
            self._braking_done_for_movement = True
            self.episode.braking_done = True

    def _transition_to(self, new_phase: MovementPhase, now: float, duration: float) -> None:
        """Transition movement state and reset braking state for the new movement."""
        self._current_phase = new_phase
        self._phase_start = now
        self._phase_duration = duration
        self._braking_done_for_movement = False
        self.episode.phase = new_phase
        self.episode.start_time = now
        self.episode.duration = duration
        self.episode.braking_done = False
