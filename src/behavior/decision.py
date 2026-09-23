"""Central decision engine coordinating perception, attention, and motor planning.

Maintains temporal continuity across the human-like behavioral pipeline:
  Perception -> Attention -> Authoritative Aim -> PlayerState -> Motor/Firing Plans
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
from src.behavior.player_state import AimPhase, AuthoritativeAim, BotPhase, TargetStatus
from src.behavior.reaction import ReactionSystem
from src.behavior.scanning import ScanningController
from src.utils.math_helpers import screen_delta_to_mouse

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
        sensitivity: float = 1.0,
        m_yaw: float = 0.022,
        m_pitch: float = 0.022,
        fov_h: float = 122.0,
        screen_width: int = 3440,
        screen_height: int = 1440,
    ):
        self.personality = personality
        self.cx, self.cy = screen_center
        self.sensitivity = sensitivity
        self.m_yaw = m_yaw
        self.m_pitch = m_pitch
        self.fov_h = fov_h
        self.screen_w = screen_width
        self.screen_h = screen_height

        self.attention = AttentionSystem(screen_center, personality)
        self.reaction = ReactionSystem(personality)
        self.firing = FiringController(personality, weapon)
        self.movement = MovementController(personality)
        self.scanning = ScanningController(
            personality,
            screen_center,
            sensitivity=sensitivity,
            m_yaw=m_yaw,
            m_pitch=m_pitch,
            screen_size=(screen_width, screen_height),
            fov_h=fov_h,
        )
        self.adaptation = AdaptationEngine(personality)
        self._prev_primary_target: TrackedTarget | None = None

    def decide(
        self,
        state: PlayerState,
        visible_targets: list[TrackedTarget],
    ) -> DecisionOutput:
        """Run one tick of the decision pipeline."""
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

        # Detect target acquisition / loss events
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
            # If previous target was shot multiple times and vanished, treat as potential kill
            if self._prev_primary_target.shots_at >= 3:
                self.adaptation.on_engagement_won(state)
            self._prev_primary_target = None
            if state.aim_phase != AimPhase.IDLE:
                state.aim_phase = AimPhase.IDLE

        # ── 3. Reaction update ───────────────────────────────────────────────
        self.reaction.update(state)

        # ── 4. Authoritative Aim State (Calculated ONCE per tick) ────────────
        if target is not None and target.status == TargetStatus.VISIBLE:
            # Select aim preference once upon acquisition (never reroll per frame!)
            if not target.aim_preference:
                head_choice = random.random() < p.head_aim_preference
                target.aim_preference = "head" if head_choice else "chest"
                target.aim_offset_ratio = (0.5, 0.18) if head_choice else (0.5, 0.45)

            # Compute screen coordinates of the persistent aim point
            d = target.detection
            w = max(1.0, d.x2 - d.x1)
            h = max(1.0, d.y2 - d.y1)
            rx, ry = target.aim_offset_ratio
            aim_x = d.x1 + w * rx
            aim_y = d.y1 + h * ry

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

            state.authoritative_aim = AuthoritativeAim(
                target_id=target.target_id,
                aim_point_screen=(aim_x, aim_y),
                screen_error=(err_x, err_y),
                screen_error_dist=err_dist,
                motor_error=(float(mouse_dx), float(mouse_dy)),
                aim_preference=target.aim_preference,
                is_valid=True,
            )
            state.current_aim_error = (err_x, err_y)
        else:
            state.authoritative_aim = AuthoritativeAim(is_valid=False)
            state.current_aim_error = (0.0, 0.0)
            if target is not None:
                # Target is not currently visible (BRIEFLY_LOST / REACQUIRING)
                if state.aim_phase in (AimPhase.ACQUIRING, AimPhase.CORRECTING, AimPhase.TRACKING):
                    state.aim_phase = AimPhase.REACQUIRING

        # ── 5. Adaptation update ─────────────────────────────────────────────
        self.adaptation.update(state)

        # ── 6. Persistent Reload Machine ─────────────────────────────────────
        should_press_reload_key = False
        action_type = "roam"
        reload_ep = state.reload_episode

        if reload_ep.is_reloading:
            # Check if reload finished (expected duration elapsed or HUD confirmed ammo restored)
            if (now - reload_ep.start_time >= reload_ep.expected_duration) or (
                state.ammo_clip > 15
            ):
                reload_ep.is_reloading = False
                reload_ep.key_pressed = False
                state.transition_phase(BotPhase.ROAMING)
            else:
                state.transition_phase(BotPhase.RELOADING)
                action_type = "reload"
        else:
            # Check if reload should be initiated
            need_reload = state.ammo_clip <= 0
            tactical_reload = (
                target is None
                and state.ammo_clip < 15
                and random.random() < ((1.0 - p.firing_discipline) * 0.02)
            )
            if need_reload or tactical_reload:
                reload_ep.is_reloading = True
                reload_ep.start_time = now
                reload_ep.expected_duration = 2.7
                reload_ep.key_pressed = True
                should_press_reload_key = True
                state.transition_phase(BotPhase.RELOADING)
                action_type = "reload"

        # ── 7. High-level Phase determination ────────────────────────────────
        if not reload_ep.is_reloading:
            if target is not None and target.status == TargetStatus.VISIBLE:
                if state.health <= p.disengage_health:
                    state.transition_phase(BotPhase.RETREATING)
                    action_type = "retreat"
                else:
                    state.transition_phase(BotPhase.ENGAGING)
                    action_type = "engage"
            elif target is not None and target.status in (
                TargetStatus.BRIEFLY_LOST,
                TargetStatus.REACQUIRING,
            ):
                state.transition_phase(BotPhase.SCANNING)
                action_type = "search"
            elif state.spatial_memories or (
                state.last_enemy_positions and (now - state.last_enemy_positions[-1][2] < 3.0)
            ):
                state.transition_phase(BotPhase.SCANNING)
                action_type = "search"
            else:
                state.transition_phase(BotPhase.ROAMING)
                action_type = "roam"

        # ── 8. Firing plan (Consumes Authoritative Aim State) ─────────────────
        firing_cmd = FiringCommand()
        if not reload_ep.is_reloading and state.authoritative_aim.is_valid:
            err_dist = state.authoritative_aim.screen_error_dist
            firing_cmd = self.firing.update(state, target, err_dist)
        else:
            if state.firing_episode.active:
                state.firing_episode.active = False
            state.is_trigger_held = False

        # ── 9. Movement plan ─────────────────────────────────────────────────
        move_keys = self.movement.update(state, target, firing_cmd.is_firing)
        if firing_cmd.should_crouch:
            move_keys["crouch"] = True

        # ── 10. Scanning / looking plan ──────────────────────────────────────
        scan_delta = (0.0, 0.0)
        if (target is None or target.status != TargetStatus.VISIBLE) and not reload_ep.is_reloading:
            scan_delta = self.scanning.update(state)

        # ── 11. Idle quirks ──────────────────────────────────────────────────
        idle_action = None
        if (
            state.phase == BotPhase.ROAMING
            and not firing_cmd.is_firing
            and not reload_ep.is_reloading
        ):
            roll = random.random()
            if roll < p.inspect_tendency * 0.03:
                idle_action = "inspect"

        return DecisionOutput(
            phase=state.phase,
            action_type=action_type,
            target=target,
            firing_cmd=firing_cmd,
            movement_keys=move_keys,
            scan_delta=scan_delta,
            should_reload=should_press_reload_key,
            idle_action=idle_action,
        )
