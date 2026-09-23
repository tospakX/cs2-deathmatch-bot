"""Central decision engine coordinating perception, attention, motor planning, and actions.

Replaces the old stateless DecisionMaker.  Maintains temporal continuity
and processes the complete human-like behavioral pipeline:
  Perception -> Attention -> PlayerState -> DecisionEngine -> Motor/Firing/Movement Plans
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.behavior.adaptation import AdaptationEngine
from src.behavior.attention import AttentionSystem
from src.behavior.firing import FiringCommand, FiringController
from src.behavior.movement import MovementController
from src.behavior.player_state import AimPhase, BotPhase
from src.behavior.reaction import ReactionSystem
from src.behavior.scanning import ScanningController

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


@dataclass
class DecisionOutput:
    """The aggregate plan produced by the decision engine for execution this tick."""

    phase: BotPhase
    action_type: str  # "engage", "roam", "search", "retreat", "reload", "wait"
    target: TrackedTarget | None
    firing_cmd: FiringCommand
    movement_keys: dict[str, bool]
    scan_delta: tuple[float, float]  # (dx, dy) purposeful scan adjustment
    should_reload: bool = False
    idle_action: str | None = None  # "inspect", "jump", None


class DecisionEngine:
    """Orchestrates behavioral subsystems into cohesive simulated human action."""

    def __init__(
        self,
        personality: PersonalityTraits,
        screen_center: tuple[int, int],
        weapon: str = "default",
    ):
        self.personality = personality
        self.cx, self.cy = screen_center
        self.attention = AttentionSystem(screen_center, personality)
        self.reaction = ReactionSystem(personality)
        self.firing = FiringController(personality, weapon)
        self.movement = MovementController(personality)
        self.scanning = ScanningController(personality, screen_center)
        self.adaptation = AdaptationEngine(personality)
        self._prev_primary_target: TrackedTarget | None = None

    def decide(
        self,
        state: PlayerState,
        visible_targets: list[TrackedTarget],
    ) -> DecisionOutput:
        """Run one tick of the decision pipeline.

        Args:
            state: Persistent player state.
            visible_targets: Currently visible tracked targets from perception layer.
        """
        now = state.now
        p = self.personality

        # ── 1. Death / Respawn handling ──────────────────────────────────────
        if not state.is_alive:
            if state.phase != BotPhase.DEAD:
                state.on_death()
                self.adaptation.on_engagement_lost(state)
            return DecisionOutput(
                phase=BotPhase.DEAD,
                action_type="wait",
                target=None,
                firing_cmd=FiringCommand(trigger_action="release"),
                movement_keys={
                    k: False for k in ("forward", "back", "left", "right", "crouch", "walk")
                },
                scan_delta=(0.0, 0.0),
            )
        elif state.phase == BotPhase.DEAD:
            state.on_respawn()

        # ── 2. Attention: select / maintain focus ────────────────────────────
        self.attention.update(state, visible_targets)
        target = state.primary_target

        # Detect new target acquisition -> trigger contextual reaction
        if target is not None and target is not self._prev_primary_target:
            context = "new_target"
            if target is state.previous_target:
                context = "reacquire"
            elif self._prev_primary_target is not None:
                context = "switch"
            elif target.frames_visible <= 2:
                context = "surprise"

            self.reaction.initiate_reaction(state, target, context)
            self._prev_primary_target = target
        elif target is None and self._prev_primary_target is not None:
            self._prev_primary_target = None
            if state.aim_phase != AimPhase.IDLE:
                state.aim_phase = AimPhase.IDLE

        # ── 3. Reaction update ───────────────────────────────────────────────
        self.reaction.update(state)

        # ── 4. Adaptation update ─────────────────────────────────────────────
        self.adaptation.update(state)

        # ── 5. High-level Phase determination ────────────────────────────────
        action_type = "roam"
        should_reload = False

        # Need reload check: ammo depleted or safe reload when low and out of combat
        if state.ammo_clip <= 0:
            should_reload = True
            action_type = "reload"
        elif (
            target is None
            and state.ammo_clip < 15
            and random.random() < (1.0 - p.firing_discipline) * 0.02
        ):
            should_reload = True
            action_type = "reload"

        if target is not None:
            if state.health <= p.disengage_health:
                state.transition_phase(BotPhase.RETREATING)
                action_type = "retreat"
            else:
                state.transition_phase(BotPhase.ENGAGING)
                action_type = "engage"
        elif state.last_enemy_positions and (now - state.last_enemy_positions[-1][2] < 3.0):
            state.transition_phase(BotPhase.SCANNING)
            action_type = "search"
        else:
            state.transition_phase(BotPhase.ROAMING)
            action_type = "roam"

        # ── 6. Aim error calculation ─────────────────────────────────────────
        err_dist = 999.0
        if target is not None:
            tcx, tcy = target.detection.center
            err_dist = math.sqrt((tcx - self.cx) ** 2 + (tcy - self.cy) ** 2)

        # ── 7. Firing plan ───────────────────────────────────────────────────
        firing_cmd = self.firing.update(state, target, err_dist)

        # ── 8. Movement plan ─────────────────────────────────────────────────
        move_keys = self.movement.update(state, target, firing_cmd.is_firing)
        if firing_cmd.should_crouch:
            move_keys["crouch"] = True

        # ── 9. Scanning / looking plan ───────────────────────────────────────
        scan_delta = (0.0, 0.0)
        if target is None:
            scan_delta = self.scanning.update(state)

        # ── 10. Idle quirks (inspect weapon / random fidget) ──────────────────
        idle_action = None
        if state.phase == BotPhase.ROAMING and not firing_cmd.is_firing:
            roll = random.random()
            if roll < p.inspect_tendency * 0.05:
                idle_action = "inspect"
            elif roll < (p.inspect_tendency + p.fidget_tendency) * 0.05:
                idle_action = "fidget"

        return DecisionOutput(
            phase=state.phase,
            action_type=action_type,
            target=target,
            firing_cmd=firing_cmd,
            movement_keys=move_keys,
            scan_delta=scan_delta,
            should_reload=should_reload,
            idle_action=idle_action,
        )
