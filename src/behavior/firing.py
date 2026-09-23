"""Contextual firing behavior and shot-based recoil synchronization.

Firing decisions arise from:
- Target distance (pixel distance & estimated range from bounding box area)
- Target size and movement velocity
- Current aim error (gating: do not spray if crosshair is far off target)
- Weapon characteristics (cycle time, recoil curve, reset time)
- Confidence and personality firing discipline
- Engagement phase and burst state

Recoil is strictly SHOT-BASED:
- Advances only when an actual shot is fired based on weapon fire-rate / cycle time
- Counter-pull values are integrated into motor control / aim correction
- Resets when trigger is released and recoil recovery time elapses
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.behavior.player_state import AimPhase, FiringPhase

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


# ── Weapon recoil patterns (per-shot pixel/angle offsets) ─────────────────────
# Negative dy = crosshair climbs up -> mouse must pull down (+dy) to compensate.
# Each entry is (dx, dy) compensation in mouse counts for that shot index.
WEAPON_RECOIL_PATTERNS: dict[str, list[tuple[int, int]]] = {
    "ak47": [
        (0, -3),
        (0, -6),
        (0, -9),
        (-1, -11),
        (-1, -12),
        (-2, -12),
        (-3, -11),
        (-4, -9),
        (-3, -7),
        (-1, -6),
        (2, -5),
        (3, -5),
        (4, -4),
        (3, -4),
        (2, -3),
        (0, -3),
        (-2, -3),
        (-3, -2),
        (-3, -2),
        (-2, -1),
        (0, -1),
        (1, -1),
        (2, -1),
        (2, -1),
        (1, -1),
        (0, -1),
        (0, -1),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    "m4a4": [
        (0, -2),
        (0, -5),
        (0, -7),
        (-1, -9),
        (-1, -9),
        (-2, -8),
        (-2, -7),
        (-3, -6),
        (-2, -5),
        (-1, -5),
        (1, -4),
        (2, -4),
        (3, -3),
        (2, -3),
        (1, -2),
        (0, -2),
        (-1, -2),
        (-2, -2),
        (-1, -1),
        (0, -1),
        (0, -1),
        (0, -1),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    "default": [
        (0, -3),
        (0, -5),
        (0, -7),
        (-1, -8),
        (-1, -9),
        (-2, -9),
        (-2, -8),
        (-2, -7),
        (-1, -6),
        (0, -5),
        (1, -4),
        (2, -4),
        (2, -3),
        (1, -3),
        (0, -2),
        (-1, -2),
        (-1, -2),
        (0, -1),
        (0, -1),
        (0, -1),
    ],
}

# Weapon cycle times in seconds (e.g. 600 RPM = 0.100s)
WEAPON_CYCLE_TIMES: dict[str, float] = {
    "ak47": 0.100,
    "m4a4": 0.090,
    "m4a1-s": 0.100,
    "pistol": 0.150,
    "default": 0.100,
}

# Time in seconds for recoil to fully reset after trigger release
RECOIL_RESET_TIME = 0.350


@dataclass
class FiringCommand:
    """Action produced by the firing controller for input execution."""

    trigger_action: str = "none"  # "press", "hold", "release", "none"
    recoil_dx: int = 0
    recoil_dy: int = 0
    should_crouch: bool = False
    is_firing: bool = False
    mode: str = "none"


class FiringController:
    """Manages contextual firing decisions and shot-synchronized recoil."""

    def __init__(self, personality: PersonalityTraits, weapon: str = "default"):
        self.personality = personality
        self.weapon = weapon
        self._last_shot_time: float = 0.0
        self._trigger_pressed_time: float = 0.0
        self._target_shots_in_burst: int = 0
        self._burst_pause_until: float = 0.0
        self._cooldown_until: float = 0.0

    @property
    def cycle_time(self) -> float:
        return WEAPON_CYCLE_TIMES.get(self.weapon, WEAPON_CYCLE_TIMES["default"])

    def set_weapon(self, weapon: str, state: PlayerState) -> None:
        """Switch weapon profile and reset recoil state."""
        self.weapon = weapon
        state.weapon = weapon
        state.reset_spray()
        self._target_shots_in_burst = 0

    def update(
        self,
        state: PlayerState,
        target: TrackedTarget | None,
        err_dist: float,
    ) -> FiringCommand:
        """Update firing state and produce firing/recoil commands for this tick.

        Args:
            state: Central player state.
            target: Currently engaged target (if any).
            err_dist: Pixel distance from crosshair to aim point.
        """
        now = state.now
        cmd = FiringCommand()
        p = self.personality

        # Check if recoil has reset during trigger release
        if not state.is_trigger_held:
            if state.recoil_shot_index > 0 and (now - self._last_shot_time) > RECOIL_RESET_TIME:
                state.reset_spray()

        # Cannot fire if dead or no ammo or waiting for burst pause / cooldown
        if not state.is_alive or state.ammo_clip <= 0:
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            return cmd

        if now < self._burst_pause_until:
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.BURST_PAUSE
            return cmd

        # If no target or still reacting to target, release trigger
        if (
            target is None
            or state.aim_phase == AimPhase.REACTING
            or state.aim_phase == AimPhase.IDLE
        ):
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            return cmd

        # ── Aim error gating ─────────────────────────────────────────────────
        # Human players do not hold trigger when aim is way off target (> 50-70px).
        # Precision players have tighter gating.
        max_firing_err = 35.0 + (1.0 - p.firing_discipline) * 45.0
        # If target is very close (large bbox area), we can tolerate wider aim error.
        if target.detection.area > 15000:
            max_firing_err *= 1.4

        if err_dist > max_firing_err:
            # Crosshair not on target yet
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            return cmd

        # ── Choose / evaluate firing mode ────────────────────────────────────
        # Contextual decision:
        # Long distance (small bbox, large screen dist) -> Tap
        # Medium distance -> Burst (2-5 shots)
        # Close distance -> Spray (controlled pull-down)
        est_distance = 1000.0 / max(math.sqrt(target.detection.area), 1.0)
        mode = self._select_fire_mode(state, target, err_dist, est_distance)
        cmd.mode = mode

        # ── Execute mode state machine ───────────────────────────────────────
        if mode == "tap":
            return self._handle_tap(state, now, cmd)
        elif mode == "burst":
            return self._handle_burst(state, now, cmd)
        else:  # "spray"
            return self._handle_spray(state, now, cmd)

    def _select_fire_mode(
        self,
        state: PlayerState,
        target: TrackedTarget,
        err_dist: float,
        est_distance: float,
    ) -> str:
        """Select appropriate fire mode based on combat context."""
        p = self.personality

        # If already committed to a burst, stay in burst until target shots reached
        if state.firing_phase == FiringPhase.BURSTING:
            return "burst"

        # If already spraying and target remains in range, stay spraying
        if state.firing_phase == FiringPhase.SPRAYING and state.is_trigger_held:
            max_spray = int(8 + (1.0 - p.firing_discipline) * 12)
            if state.shot_count < max_spray:
                return "spray"
            # Spray limit reached -> force burst pause
            return "tap"

        # Far distance -> tap
        if est_distance > 15.0 or err_dist > 25.0:
            if random.random() < (0.3 + p.firing_discipline * 0.5):
                return "tap"
            return "burst"

        # Medium distance -> burst
        if est_distance > 8.0:
            if random.random() < p.firing_discipline * 0.7:
                return "burst"
            return "spray"

        # Close quarters -> spray
        if random.random() < (0.7 + (1.0 - p.firing_discipline) * 0.3):
            return "spray"
        return "burst"

    def _handle_tap(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Single tap fire with spread recovery delay."""
        p = self.personality
        tap_interval = self.cycle_time + (0.100 + (1.0 - p.firing_discipline) * 0.150)

        if now - self._last_shot_time >= tap_interval:
            # Fire single shot
            cmd.trigger_action = "press"
            state.is_trigger_held = True
            state.firing_phase = FiringPhase.TAPPING
            self._record_shot_fired(state, now)
            cmd.is_firing = True
            self._apply_shot_recoil(state, cmd)
            # Schedule release on next tick
        elif state.is_trigger_held:
            cmd.trigger_action = "release"
            state.is_trigger_held = False

        return cmd

    def _handle_burst(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Burst fire of N actual shots followed by recovery pause."""
        p = self.personality

        if state.firing_phase != FiringPhase.BURSTING:
            # Initialize new burst
            state.firing_phase = FiringPhase.BURSTING
            # Burst length in actual bullets: 2 to 5 based on discipline
            lo = max(2, int(3 - p.firing_discipline * 2))
            hi = max(lo + 1, int(5 - p.firing_discipline * 2))
            self._target_shots_in_burst = random.randint(lo, hi)

        # Check if enough time has elapsed to advance a shot
        if now - self._last_shot_time >= self.cycle_time:
            if not state.is_trigger_held:
                cmd.trigger_action = "hold"
                state.is_trigger_held = True

            self._record_shot_fired(state, now)
            cmd.is_firing = True
            self._apply_shot_recoil(state, cmd)

            # Check if burst target reached
            if state.shot_count >= self._target_shots_in_burst:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.BURST_PAUSE
                pause_time = 0.200 + p.firing_discipline * 0.250
                self._burst_pause_until = now + pause_time
                state.reset_spray()
        else:
            if state.is_trigger_held:
                cmd.trigger_action = "hold"
                cmd.is_firing = True

        return cmd

    def _handle_spray(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Full spray with recoil compensation and crouch evaluation."""
        p = self.personality
        state.firing_phase = FiringPhase.SPRAYING

        # Contextual crouch: crouch during sustained spray (shot 3+)
        if state.shot_count >= 3 and random.random() < p.crouch_tendency:
            cmd.should_crouch = True

        if now - self._last_shot_time >= self.cycle_time:
            if not state.is_trigger_held:
                cmd.trigger_action = "hold"
                state.is_trigger_held = True

            self._record_shot_fired(state, now)
            cmd.is_firing = True
            self._apply_shot_recoil(state, cmd)
        else:
            if state.is_trigger_held:
                cmd.trigger_action = "hold"
                cmd.is_firing = True

        return cmd

    def _record_shot_fired(self, state: PlayerState, now: float) -> None:
        """Record shot occurrence in state and timers."""
        self._last_shot_time = now
        state.record_shot()

    def _apply_shot_recoil(self, state: PlayerState, cmd: FiringCommand) -> None:
        """Compute recoil compensation for the shot just fired."""
        p = self.personality
        pattern = WEAPON_RECOIL_PATTERNS.get(self.weapon, WEAPON_RECOIL_PATTERNS["default"])
        shot_idx = state.recoil_shot_index - 1

        if shot_idx < 0:
            return

        if shot_idx < len(pattern):
            base_dx, base_dy = pattern[shot_idx]
        else:
            base_dx, base_dy = pattern[-1]

        # Recoil skill scales compensation accuracy (0.4 to 0.95 for human players)
        skill = max(0.2, min(1.0, p.recoil_skill))
        # Imperfect compensation with natural jitter
        jitter = random.uniform(0.85, 1.15)
        # Note: CS2 recoil moves crosshair up (-dy in pattern), so compensation must pull DOWN (+dy)
        comp_dx = int(-base_dx * skill * jitter)
        comp_dy = int(-base_dy * skill * jitter)

        cmd.recoil_dx = comp_dx
        cmd.recoil_dy = comp_dy
        state.recoil_accumulated = (
            state.recoil_accumulated[0] + comp_dx,
            state.recoil_accumulated[1] + comp_dy,
        )
