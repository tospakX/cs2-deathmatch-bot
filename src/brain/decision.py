"""High-level decision making with state persistence and contextual awareness."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.behavior.personality import PersonalityTraits
from src.brain.state_machine import BotState
from src.vision.detector import Detection

if TYPE_CHECKING:
    from src.humanizer.personality import Personality


class Action:
    """A decision output that the main loop executes."""

    def __init__(self, action_type: str, **kwargs):
        self.type = action_type
        self.params = kwargs

    def __repr__(self) -> str:
        return f"Action({self.type}, {self.params})"


class DecisionMaker:
    """Makes coherent high-level decisions based on persistent state and context."""

    def __init__(self, personality: Personality | PersonalityTraits):
        self.personality = personality
        self._spray_count = 0
        self._current_target: Detection | None = None
        self._target_engaged_time: float = 0.0
        self._combat_move_direction: str = "none"
        self._combat_move_ticks: int = 0
        self._last_fire_mode: str = "tap"

    def decide(
        self,
        state: BotState,
        enemies: list[Detection],
        health: int,
        ammo_clip: int,
        time_in_state: float,
    ) -> Action:
        """Make a decision based on current state with temporal continuity."""
        if state == BotState.DEAD:
            self._current_target = None
            self._spray_count = 0
            return self._decide_dead(time_in_state)
        elif state == BotState.FIGHTING:
            return self._decide_fighting(enemies, health, ammo_clip)
        elif state == BotState.SEARCHING:
            self._current_target = None
            return self._decide_searching(time_in_state)
        elif state == BotState.RETREATING:
            return self._decide_retreating(enemies)
        elif state == BotState.STUCK:
            return self._decide_stuck(time_in_state)
        else:  # ROAMING
            self._current_target = None
            return self._decide_roaming()

    def _decide_dead(self, time_in_state: float) -> Action:
        """Wait for respawn in Deathmatch."""
        if time_in_state > 1.0 and random.random() < 0.1:
            return Action("click")
        return Action("wait")

    def _decide_fighting(self, enemies: list[Detection], health: int, ammo_clip: int) -> Action:
        """Engage enemies with target persistence and continuous firing/movement."""
        if not enemies:
            return Action("search")

        if ammo_clip <= 0:
            self._spray_count = 0
            return Action("reload")

        # Target persistence: maintain current target if still present among detections
        target = None
        if self._current_target is not None:
            # Check if previous target still roughly matches any enemy detection
            tcx, tcy = self._current_target.center
            for e in enemies:
                ecx, ecy = e.center
                if ((ecx - tcx) ** 2 + (ecy - tcy) ** 2) < (150.0**2):
                    target = e
                    break

        if target is None:
            # Acquire top priority target
            target = enemies[0]
            self._current_target = target
            self._spray_count = 0

        # Decide fire mode based on distance and personality discipline
        fire_mode = self._choose_fire_mode(target)

        # Stateful combat movement with directional commitment
        move = self._update_combat_movement()

        return Action("engage", target=target, fire_mode=fire_mode, combat_move=move)

    def _choose_fire_mode(self, target: Detection) -> str:
        """Choose fire mode contextually based on distance and personality."""
        p = self.personality
        est_distance = 1000.0 / max(target.area**0.5, 1.0)

        # Long range: tap fire
        if est_distance > 18.0:
            self._spray_count = 0
            return "tap"

        # Tapping preference
        if random.random() < getattr(p, "tap_chance", 0.3):
            self._spray_count = 0
            return "tap"

        burst_lo, burst_hi = getattr(p, "burst_length", [3, 7])
        if self._spray_count >= random.randint(burst_lo, burst_hi):
            self._spray_count = 0
            return "burst_end"

        self._spray_count += 1
        return "spray"

    def _update_combat_movement(self) -> str | None:
        """Maintain strafe direction for several ticks instead of jittering."""
        p = self.personality
        if not getattr(p, "strafe_while_shooting", True):
            return None

        self._combat_move_ticks -= 1
        if self._combat_move_ticks <= 0:
            # Commit to a new direction for ~10 to 25 ticks (300-800ms at 30Hz)
            self._combat_move_ticks = random.randint(10, 25)
            if random.random() < getattr(p, "crouch_spray_chance", 0.4):
                self._combat_move_direction = "crouch"
            else:
                self._combat_move_direction = random.choice(["strafe_left", "strafe_right"])

        return self._combat_move_direction

    def _decide_searching(self, time_in_state: float) -> Action:
        """Search for enemies after losing sight."""
        if time_in_state < 1.0:
            return Action("check_corner", direction="left")
        elif time_in_state < 2.0:
            return Action("check_corner", direction="right")
        return Action("roam")

    def _decide_retreating(self, enemies: list[Detection]) -> Action:
        """Disengage and flee from threat."""
        if enemies:
            return Action("flee", enemy=enemies[0])
        return Action("roam")

    def _decide_stuck(self, time_in_state: float) -> Action:
        """Recover from stuck position."""
        if time_in_state < 1.0:
            return Action("unstick", phase="backup")
        elif time_in_state < 2.0:
            return Action("unstick", phase="turn")
        return Action("unstick", phase="forward")

    def _decide_roaming(self) -> Action:
        """Roaming behavior with occasional idle actions."""
        p = self.personality
        roll = random.random()
        inspect_chance = getattr(p, "inspect_chance", 0.01)
        look_around_chance = getattr(p, "look_around_chance", 0.03)
        random_jump_chance = getattr(p, "random_jump_chance", 0.01)

        if roll < inspect_chance:
            return Action("inspect_weapon")
        elif roll < inspect_chance + look_around_chance:
            return Action("look_around")
        elif roll < inspect_chance + look_around_chance + random_jump_chance:
            return Action("jump")

        return Action("roam")
