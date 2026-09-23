"""Stateful combat movement controller.

Replaces frame-by-frame random left/right/crouch decisions with continuous,
context-driven movement:
- Direction commitment: commits to a strafe direction for a realistic duration (300-800ms)
- Counter-strafing: brief opposite key press before firing to kill velocity and gain
  first-shot accuracy
- Contextual engagement movement:
    * Approaching: closing distance when weapon or situation favors it
    * Holding: holding position when holding an angle or burst-firing
    * Strafing: rhythmic combat strafes with counter-strafe stops
    * Retreating: low health, seeking cover, moving backwards while breaking sight
    * Repositioning: relocating between bursts or after missing shots
    * Searching: cautious corner checks and deliberate search paths
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.behavior.player_state import MovementPhase

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class MovementController:
    """Manages continuous stateful movement during combat and roaming."""

    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self._current_phase: MovementPhase = MovementPhase.STATIONARY
        self._phase_start: float = 0.0
        self._phase_duration: float = 0.0
        self._current_direction: str = ""  # "left", "right", "forward", "back"
        self._counter_strafing: bool = False
        self._counter_strafe_end: float = 0.0
        self._counter_strafe_key: str = ""
        self._strafe_alternation: int = 1  # 1 or -1
        self._crouch_committed: bool = False

    def update(
        self,
        state: PlayerState,
        target: TrackedTarget | None,
        is_firing: bool,
    ) -> dict[str, bool]:
        """Compute movement key states for this tick.

        Returns:
            dict with boolean flags for "forward", "back", "left", "right", "crouch", "walk".
        """
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

        # ── 1. Counter-strafing sub-phase ────────────────────────────────────
        # When stopping to shoot, press opposite key for ~60ms to cancel velocity.
        if self._counter_strafing:
            if now < self._counter_strafe_end:
                if self._counter_strafe_key in keys:
                    keys[self._counter_strafe_key] = True
                return keys
            else:
                self._counter_strafing = False

        # ── 2. Determine / transition movement phase ─────────────────────────
        phase_elapsed = now - self._phase_start
        should_rethink = (
            phase_elapsed >= self._phase_duration or self._current_phase == MovementPhase.STATIONARY
        )

        if state.health <= p.disengage_health and target is not None:
            # Low health: prioritize retreating
            if self._current_phase != MovementPhase.RETREATING:
                self._transition_to(MovementPhase.RETREATING, now, random.uniform(1.0, 2.5))
        elif target is not None:
            # In combat with an enemy
            if should_rethink:
                self._select_combat_phase(state, target, is_firing, now)
        else:
            # Roaming / idle
            if should_rethink:
                self._select_roam_phase(state, now)

        # ── 3. Execute current phase ─────────────────────────────────────────
        if self._current_phase == MovementPhase.STRAFING:
            if is_firing and p.strafe_tendency <= 0.6:
                # Low strafe-tendency players stop while firing (better recoil control)
                if self._current_direction in ("left", "right"):
                    self._trigger_counter_strafe(self._current_direction, now)
                    if self._counter_strafe_key in keys:
                        keys[self._counter_strafe_key] = True
            else:
                if self._current_direction in keys:
                    keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.APPROACHING:
            keys["forward"] = True
            if self._current_direction in ("left", "right") and random.random() < 0.3:
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.RETREATING:
            keys["back"] = True
            # Add evasive side strafe while running backwards
            if self._current_direction in ("left", "right"):
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.REPOSITIONING:
            if self._current_direction in keys:
                keys[self._current_direction] = True

        elif self._current_phase == MovementPhase.HOLDING:
            # Stationary; holding angle
            pass

        # ── 4. Crouch management ─────────────────────────────────────────────
        if state.is_crouching:
            keys["crouch"] = True

        state.movement_phase = self._current_phase
        return keys

    def _select_combat_phase(
        self,
        state: PlayerState,
        target: TrackedTarget,
        is_firing: bool,
        now: float,
    ) -> None:
        """Choose combat movement phase based on distance, weapon, and personality."""
        p = self.personality
        est_distance = 1000.0 / max(target.detection.area**0.5, 1.0)

        # Close range: strafe or committed crouch spray
        if est_distance < 8.0:
            if is_firing and random.random() < p.crouch_tendency:
                self._transition_to(MovementPhase.HOLDING, now, random.uniform(0.5, 1.2))
                state.is_crouching = True
                return
            else:
                state.is_crouching = False

            # Alternate strafe direction
            self._strafe_alternation *= -1
            direction = "left" if self._strafe_alternation > 0 else "right"
            duration = random.uniform(0.35, 0.70) * (0.8 + (1.0 - p.motor_precision) * 0.4)
            self._current_direction = direction
            self._transition_to(MovementPhase.STRAFING, now, duration)

        # Medium range: rhythm of strafe -> counter-strafe stop -> burst
        elif est_distance < 18.0:
            state.is_crouching = False
            roll = random.random()
            if roll < p.strafe_tendency:
                self._strafe_alternation *= -1
                self._current_direction = "left" if self._strafe_alternation > 0 else "right"
                duration = random.uniform(0.40, 0.85)
                self._transition_to(MovementPhase.STRAFING, now, duration)
            elif roll < p.strafe_tendency + p.repositioning_tendency:
                self._current_direction = random.choice(["left", "right", "back"])
                duration = random.uniform(0.30, 0.60)
                self._transition_to(MovementPhase.REPOSITIONING, now, duration)
            else:
                self._transition_to(MovementPhase.HOLDING, now, random.uniform(0.3, 0.7))

        # Long range: holding angle or micro-repositioning
        else:
            state.is_crouching = False
            if random.random() < 0.6:
                self._transition_to(MovementPhase.HOLDING, now, random.uniform(0.4, 0.9))
            else:
                self._strafe_alternation *= -1
                self._current_direction = "left" if self._strafe_alternation > 0 else "right"
                self._transition_to(MovementPhase.STRAFING, now, random.uniform(0.25, 0.50))

    def _select_roam_phase(self, state: PlayerState, now: float) -> None:
        """Choose roaming movement state."""
        state.is_crouching = False
        # When roaming, default to APPROACHING (which maps to forward roaming in navigator)
        self._transition_to(MovementPhase.APPROACHING, now, random.uniform(1.0, 3.0))

    def _trigger_counter_strafe(self, moving_direction: str, now: float) -> None:
        """Initiate counter-strafe to stop player velocity."""
        opposite = {"left": "right", "right": "left", "forward": "back", "back": "forward"}
        opp_key = opposite.get(moving_direction, "")
        if opp_key:
            self._counter_strafing = True
            # Counter-strafe key duration is ~60-80ms in CS2
            self._counter_strafe_end = now + random.uniform(0.060, 0.080)
            self._counter_strafe_key = opp_key

    def _transition_to(self, phase: MovementPhase, now: float, duration: float) -> None:
        """Transition movement phase."""
        self._current_phase = phase
        self._phase_start = now
        self._phase_duration = duration
