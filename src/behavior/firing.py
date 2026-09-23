"""Contextual firing behavior, firing episodes, and shot-synchronized recoil.

Firing decisions arise from:
- Target distance (pixel distance & estimated range from bounding box area)
- Authoritative aim error
- Weapon characteristics (cycle time, recoil curve, reset time)
- Confidence, panic, and personality firing discipline
- Committed FiringEpisodes (tap, burst, spray) rather than per-tick mode flapping

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

from src.behavior.player_state import AimPhase, FiringPhase, TargetStatus

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
    """Manages contextual firing episodes and shot-synchronized recoil."""

    def __init__(self, personality: PersonalityTraits, weapon: str = "default"):
        self.personality = personality
        self.weapon = weapon
        self._last_shot_time: float = 0.0
        self._recoil_bias_x: float = random.gauss(0, 0.2)
        self._recoil_bias_y: float = random.gauss(0, 0.15)

    @property
    def cycle_time(self) -> float:
        return WEAPON_CYCLE_TIMES.get(self.weapon, WEAPON_CYCLE_TIMES["default"])

    def set_weapon(self, weapon: str, state: PlayerState) -> None:
        """Switch weapon profile and reset recoil state."""
        self.weapon = weapon
        state.weapon = weapon
        state.reset_spray()
        state.firing_episode.active = False

    def update(
        self,
        state: PlayerState,
        target: TrackedTarget | None,
        err_dist: float = 0.0,
    ) -> FiringCommand:
        """Update firing state and produce firing/recoil commands for this tick."""
        now = state.now
        cmd = FiringCommand()
        p = self.personality
        episode = state.firing_episode
        err_dist_px = err_dist

        # Check recoil reset after trigger release
        if not state.is_trigger_held:
            if state.recoil_shot_index > 0 and (now - self._last_shot_time) > RECOIL_RESET_TIME:
                state.reset_spray()

        # Cannot fire if dead or out of ammo
        if not state.is_alive or state.ammo_clip <= 0:
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            episode.active = False
            return cmd

        # Waiting out burst pause or episode cooldown
        if now < episode.pause_until or now < episode.cooldown_until:
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.BURST_PAUSE
            return cmd

        # If no target, target is not visible, or still reacting, release trigger
        if (
            target is None
            or target.status != TargetStatus.VISIBLE
            or state.aim_phase in (AimPhase.REACTING, AimPhase.IDLE)
        ):
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            episode.active = False
            return cmd

        # ── 1. Graded Aim Error Firing Gate (not binary) ─────────────────────
        # Real players shoot with graded readiness based on aim error, panic, and distance
        tight_threshold = 28.0 + (1.0 - p.firing_discipline) * 35.0
        if target.detection.area > 12000:
            tight_threshold *= 1.4

        # Graded probability: if aim is within tight threshold, fire.
        # If slightly outside, probability decays smoothly rather than abrupt cliff.
        can_fire = False
        if err_dist_px <= tight_threshold:
            can_fire = True
        elif episode.active and episode.mode == "spray" and state.is_trigger_held:
            # Committed sprayers tolerate brief tracking overshoot
            can_fire = err_dist_px <= (tight_threshold * 1.6)
        else:
            # Graded chance of premature / hurried shot
            over_err = err_dist_px - tight_threshold
            panic_bonus = (1.0 - state.confidence) * 0.2 if state.health < 40 else 0.0
            premature_prob = max(0.0, 0.25 + panic_bonus - (over_err / 60.0))
            can_fire = random.random() < premature_prob

        if not can_fire:
            if state.is_trigger_held:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.NOT_FIRING
            return cmd

        # ── 2. Firing Episode Commitment ─────────────────────────────────────
        est_distance = 1000.0 / max(math.sqrt(target.detection.area), 1.0)

        if not episode.active:
            # Start new firing episode based on context
            episode.active = True
            episode.start_time = now
            episode.shots_fired = 0
            if state.firing_phase == FiringPhase.BURSTING:
                mode = "burst"
            elif state.firing_phase == FiringPhase.TAPPING:
                mode = "tap"
            elif state.firing_phase == FiringPhase.SPRAYING:
                mode = "spray"
            else:
                mode = self._choose_episode_mode(p, est_distance, err_dist_px)
            episode.mode = mode

            if mode == "tap":
                episode.shots_planned = 1
            elif mode == "burst":
                if getattr(self, "_target_shots_in_burst", 0) > 0:
                    episode.shots_planned = self._target_shots_in_burst
                else:
                    lo = max(2, int(3 - p.firing_discipline * 2))
                    hi = max(lo + 1, int(5 - p.firing_discipline * 2))
                    episode.shots_planned = random.randint(lo, hi)
            else:  # spray
                episode.shots_planned = int(8 + (1.0 - p.firing_discipline) * 10)

        cmd.mode = episode.mode

        # ── 3. Crouch Episode Management ─────────────────────────────────────
        # Update crouch statefully without 30Hz flutter
        crouch = state.crouch_episode
        if not crouch.is_crouching:
            # Can we initiate a crouch during sustained spray?
            if (
                episode.mode == "spray"
                and episode.shots_fired >= 3
                and now >= crouch.cooldown_until
                and random.random() < p.crouch_tendency * 0.3
            ):
                crouch.is_crouching = True
                crouch.start_time = now
                crouch.duration = random.uniform(0.7, 1.4)
                state.is_crouching = True
        else:
            if (now - crouch.start_time) >= crouch.duration or not episode.active:
                crouch.is_crouching = False
                crouch.cooldown_until = now + random.uniform(1.2, 2.5)
                state.is_crouching = False

        cmd.should_crouch = state.is_crouching

        # ── 4. Execute Mode on Actual Weapon Cycle Rate ──────────────────────
        if episode.mode == "tap":
            return self._execute_tap(state, now, cmd)
        elif episode.mode == "burst":
            return self._execute_burst(state, now, cmd)
        else:
            return self._execute_spray(state, now, cmd)

    def _choose_episode_mode(
        self,
        p: PersonalityTraits,
        est_distance: float,
        err_dist_px: float,
    ) -> str:
        """Choose firing mode when initiating a new episode."""
        if est_distance > 16.0 or err_dist_px > 30.0:
            return "tap" if random.random() < (0.35 + p.firing_discipline * 0.45) else "burst"
        elif est_distance > 8.0:
            return "burst" if random.random() < (0.4 + p.firing_discipline * 0.4) else "spray"
        else:
            return (
                "spray" if random.random() < (0.65 + (1.0 - p.firing_discipline) * 0.3) else "burst"
            )

    def _execute_tap(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Single tap with contextual hold duration and recovery delay."""
        p = self.personality
        episode = state.firing_episode
        tap_interval = self.cycle_time + (0.080 + (1.0 - p.firing_discipline) * 0.120)

        # If currently holding down the tap trigger
        if state.is_trigger_held and episode.mode == "tap":
            if (now - episode.press_time) < episode.trigger_hold_duration:
                cmd.trigger_action = "hold"
                cmd.is_firing = True
                return cmd
            else:
                # Tap hold duration expired -> release trigger
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.BURST_PAUSE
                episode.active = False
                episode.pause_until = now + tap_interval
                return cmd

        # Initiating a new tap
        if now - self._last_shot_time >= tap_interval:
            cmd.trigger_action = "press"
            state.is_trigger_held = True
            state.firing_phase = FiringPhase.TAPPING
            episode.press_time = now
            # Realistic physical finger press duration: 45 to 80ms
            episode.trigger_hold_duration = random.uniform(0.045, 0.080)
            self._record_shot(state, now)
            cmd.is_firing = True
            self._apply_recoil(state, cmd)

            episode.shots_fired += 1
            episode.active = True
        elif state.is_trigger_held:
            cmd.trigger_action = "release"
            state.is_trigger_held = False

        return cmd

    def _execute_burst(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Multi-shot burst advancing strictly on cycle time."""
        p = self.personality
        episode = state.firing_episode
        state.firing_phase = FiringPhase.BURSTING

        if now - self._last_shot_time >= self.cycle_time:
            if not state.is_trigger_held:
                cmd.trigger_action = "hold"
                state.is_trigger_held = True

            self._record_shot(state, now)
            cmd.is_firing = True
            self._apply_recoil(state, cmd)
            episode.shots_fired += 1

            if episode.shots_fired >= episode.shots_planned:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.BURST_PAUSE
                # Inter-burst pause with human motor timing jitter
                pause_time = 0.160 + p.firing_discipline * 0.200 + random.uniform(-0.025, 0.035)
                episode.pause_until = now + max(0.080, pause_time)
                episode.active = False
        else:
            if state.is_trigger_held:
                cmd.trigger_action = "hold"
                cmd.is_firing = True

        return cmd

    def _execute_spray(self, state: PlayerState, now: float, cmd: FiringCommand) -> FiringCommand:
        """Sustained spray with recoil compensation."""
        episode = state.firing_episode
        state.firing_phase = FiringPhase.SPRAYING

        if now - self._last_shot_time >= self.cycle_time:
            if not state.is_trigger_held:
                cmd.trigger_action = "hold"
                state.is_trigger_held = True

            self._record_shot(state, now)
            cmd.is_firing = True
            self._apply_recoil(state, cmd)
            episode.shots_fired += 1

            if episode.shots_fired >= episode.shots_planned:
                cmd.trigger_action = "release"
                state.is_trigger_held = False
                state.firing_phase = FiringPhase.COOLDOWN
                episode.cooldown_until = now + 0.350 + random.uniform(-0.030, 0.040)
                episode.active = False
        else:
            if state.is_trigger_held:
                cmd.trigger_action = "hold"
                cmd.is_firing = True

        return cmd

    def _record_shot(self, state: PlayerState, now: float) -> None:
        self._last_shot_time = now
        state.record_shot()

    def _apply_recoil(self, state: PlayerState, cmd: FiringCommand) -> None:
        p = self.personality
        pattern = WEAPON_RECOIL_PATTERNS.get(self.weapon, WEAPON_RECOIL_PATTERNS["default"])
        shot_idx = state.recoil_shot_index - 1
        if shot_idx < 0:
            return

        if shot_idx < len(pattern):
            base_dx, base_dy = pattern[shot_idx]
        else:
            base_dx, base_dy = pattern[-1]

        skill = max(0.2, min(1.0, p.recoil_skill))
        # Recoil drift / bias (random walk with mean reversion)
        self._recoil_bias_x = self._recoil_bias_x * 0.88 + random.gauss(0, 0.05)
        self._recoil_bias_y = self._recoil_bias_y * 0.88 + random.gauss(0, 0.04)
        # Bounded compensation jitter proportional to (1.0 - skill)
        jitter_x = random.gauss(0, (1.0 - skill) * 0.5)
        jitter_y = random.gauss(0, (1.0 - skill) * 0.3)

        comp_dx = int(round(-base_dx * skill + self._recoil_bias_x + jitter_x))
        comp_dy = int(round(-base_dy * skill + self._recoil_bias_y + jitter_y))

        cmd.recoil_dx = comp_dx
        cmd.recoil_dy = comp_dy
        state.recoil_accumulated = (
            state.recoil_accumulated[0] + comp_dx,
            state.recoil_accumulated[1] + comp_dy,
        )
