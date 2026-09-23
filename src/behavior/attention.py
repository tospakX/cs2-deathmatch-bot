"""Attention model — determines what the player is actually focusing on.

A human player does not process every visible target identically.
This layer manages:
- Focus continuity (staying on a target)
- Peripheral detection (noticing new threats slowly)
- Attention switching (actual decision to change focus)
- Search behavior (looking for lost targets)
- Attention inertia (resisting constant switching)
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from src.utils.math_helpers import distance

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import (
        PlayerState,
        TrackedTarget,
    )


class AttentionSystem:
    """Models human-like selective attention with continuity and inertia."""

    def __init__(self, screen_center: tuple[int, int], personality: PersonalityTraits):
        self.cx, self.cy = screen_center
        self.personality = personality
        # Attention has temporal continuity — these persist across frames.
        self._switch_cooldown_until: float = 0.0
        self._last_switch_time: float = 0.0

    def update(self, state: PlayerState, visible_targets: list[TrackedTarget]) -> None:
        """Update attention focus based on visible targets and current state.

        This is called every tick.  It decides:
        1. Should we keep our current target?
        2. Should we switch to a new target?
        3. Should we enter search mode?
        """
        now = state.now
        # No targets visible.
        if not visible_targets:
            if state.primary_target is not None:
                lost_target = state.primary_target
                # Don't immediately drop — give a brief memory window.
                if lost_target.frames_missing > 5:
                    state.switch_target(None, "target_lost")
                    state.awareness_level = min(1.0, state.awareness_level + 0.1)
            return

        # Current target still visible?
        current_still_visible = (
            state.primary_target is not None and state.primary_target.frames_missing == 0
        )

        if current_still_visible:
            # ── Target persistence: stick with current target ────────────
            self._update_focus_on_current(state, visible_targets, now)
        else:
            # ── Need a new target ────────────────────────────────────────
            self._acquire_new_target(state, visible_targets, now)

    def _update_focus_on_current(
        self,
        state: PlayerState,
        visible_targets: list[TrackedTarget],
        now: float,
    ) -> None:
        """Decide whether to stay on current target or switch."""
        current = state.primary_target
        assert current is not None
        p = self.personality

        # How long have we been on this target (capped at 5 seconds)
        engagement_duration = max(0.0, min(5.0, now - state.target_acquisition_time))

        # Check if there's a much more threatening target.
        for target in visible_targets:
            if target is current:
                continue

            # Switching cost increases with engagement duration and personality.
            switch_cost = (
                100.0  # base cost
                + p.target_persistence * 300.0  # personality
                + engagement_duration * 50.0  # time investment
                + p.switch_reluctance * 200.0  # reluctance trait
            )

            # Check if on switch cooldown.
            if now < self._switch_cooldown_until:
                switch_cost += 500.0

            # Benefit of switching.
            threat_advantage = target.threat_level - current.threat_level

            # Position factor: target in peripheral vision is less noticeable.
            cx, cy = target.detection.center
            angular_offset = distance((self.cx, self.cy), (cx, cy))
            # Targets far from center are harder to notice.
            peripheral_penalty = angular_offset * (1.0 - p.attention_breadth) * 0.5

            switch_benefit = threat_advantage - peripheral_penalty

            # Surprise factor: suddenly appearing target gets attention bonus.
            if target.frames_visible <= 3 and target.threat_level > 500:
                switch_benefit += 200.0

            if switch_benefit > switch_cost:
                state.switch_target(target, "higher_threat")
                self._switch_cooldown_until = now + 0.3 + p.switch_reluctance * 0.5
                self._last_switch_time = now
                return

        # Stay on current target — update attention confidence.
        state.attention_confidence = min(1.0, state.attention_confidence + 0.05)

    def _acquire_new_target(
        self,
        state: PlayerState,
        visible_targets: list[TrackedTarget],
        now: float,
    ) -> None:
        """Acquire a new primary target from visible targets."""
        p = self.personality

        if not visible_targets:
            return

        # Score targets considering attention factors.
        best_target = None
        best_score = -999999.0

        for target in visible_targets:
            score = target.threat_level

            # Targets that have been visible longer are easier to notice.
            visibility_bonus = min(target.frames_visible * 30.0, 200.0)
            score += visibility_bonus

            # Central targets are noticed faster.
            cx, cy = target.detection.center
            ang_offset = distance((self.cx, self.cy), (cx, cy))
            centrality_bonus = max(0.0, 500.0 - ang_offset)
            score += centrality_bonus * p.attention_breadth

            # Previously tracked targets get recognition bonus.
            if target is state.previous_target:
                score += 150.0  # familiarity

            # Suddenly appearing large targets get surprise bonus.
            if target.frames_visible <= 2 and target.detection.area > 5000:
                score += 300.0

            if score > best_score:
                best_score = score
                best_target = target

        if best_target is not None:
            state.switch_target(best_target, "new_acquisition")
            state.attention_confidence = 0.2  # low initial confidence
            self._last_switch_time = now

    def get_scan_direction(self, state: PlayerState) -> tuple[float, float] | None:
        """When no targets are visible, suggest a direction to look.

        Returns (dx, dy) relative mouse movement suggestion, or None.
        Based on recent enemy positions, awareness, and personality.
        """
        p = self.personality

        # Check recent enemy positions.
        if state.last_enemy_positions:
            # Look toward most recent enemy position.
            last = state.last_enemy_positions[-1]
            dx = last[0] - self.cx
            dy = last[1] - self.cy
            d = math.sqrt(dx * dx + dy * dy)
            if d > 10:
                # Scale to a reasonable scan amount, with personality influence.
                scale = 15.0 + p.scanning_amplitude * 30.0
                return (dx / d * scale, dy / d * scale * 0.5)

        # Random scanning with personality influence.
        if random.random() < p.scanning_frequency * 0.1:
            amp = 20.0 + p.scanning_amplitude * 40.0
            dx = random.gauss(0, amp)
            dy = random.gauss(0, amp * 0.3)  # less vertical
            return (dx, dy)

        return None
