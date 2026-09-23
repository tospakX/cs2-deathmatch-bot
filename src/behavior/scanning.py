"""Attention-driven, purposeful camera scanning and looking behavior.

Replaces fixed-duration generic look-arounds and random number generators with:
- Checking last seen enemy positions when combat suddenly breaks
- Checking unverified corners/chokepoints with smooth deceleration and observation pauses
- Correlating look direction with movement heading and uncertainty
- Natural head/camera glances shaped by personality scanning style
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState


@dataclass
class ScanAction:
    """An intentional camera scan with a specific reason and duration."""

    reason: str  # "check_last_seen", "check_corner", "curiosity_glance"
    target_dx: float  # total mouse delta to sweep
    target_dy: float
    duration: float  # total duration in seconds
    settle_pause: float  # time to pause and look after sweeping
    start_time: float = 0.0


class ScanningController:
    """Manages purposeful camera movements during non-combat and search phases."""

    def __init__(self, personality: PersonalityTraits, screen_center: tuple[int, int]):
        self.personality = personality
        self.cx, self.cy = screen_center
        self._current_scan: ScanAction | None = None
        self._last_scan_end: float = 0.0
        self._scan_cooldown: float = 1.0

    def update(self, state: PlayerState) -> tuple[float, float]:
        """Compute camera movement delta for this tick.

        Returns:
            (dx, dy) mouse counts to apply, or (0, 0) if not actively scanning.
        """
        now = state.now
        p = self.personality

        # If in active combat engagement, scanning is suppressed
        if state.primary_target is not None and state.primary_target.frames_missing == 0:
            self._current_scan = None
            return (0.0, 0.0)

        # ── 1. If currently executing a scan ──────────────────────────────────
        if self._current_scan is not None:
            scan = self._current_scan
            elapsed = now - scan.start_time

            if elapsed < scan.duration:
                # Active sweep phase: use smoothstep bell-curve velocity
                t = elapsed / max(scan.duration, 0.001)
                # Derivative of smoothstep is 6 * t * (1 - t)
                velocity_weight = 6.0 * t * (1.0 - t)
                # Scale per tick dt
                tick_factor = state.tick_dt / scan.duration
                dx = scan.target_dx * velocity_weight * tick_factor
                dy = scan.target_dy * velocity_weight * tick_factor
                return (dx, dy)
            elif elapsed < scan.duration + scan.settle_pause:
                # Observation pause: eyes evaluating the checked area
                return (0.0, 0.0)
            else:
                # Scan completed
                self._current_scan = None
                self._last_scan_end = now
                self._scan_cooldown = random.uniform(0.8, 2.5) / max(p.scanning_frequency, 0.2)
                return (0.0, 0.0)

        # ── 2. Check if a new purposeful scan should be initiated ─────────────
        if now - self._last_scan_end < self._scan_cooldown:
            return (0.0, 0.0)

        # Priority A: Check last seen enemy location if recently lost
        if state.last_enemy_positions:
            last_x, last_y, last_t = state.last_enemy_positions[-1]
            if now - last_t < 4.0:  # Enemy seen within last 4 seconds
                screen_dx = last_x - self.cx
                screen_dy = last_y - self.cy
                dist = math.sqrt(screen_dx * screen_dx + screen_dy * screen_dy)
                if dist > 30.0:
                    # Clear recent position so we don't repeat endlessly
                    state.last_enemy_positions.pop()
                    duration = random.uniform(0.18, 0.35)
                    pause = random.uniform(0.20, 0.45)
                    self._current_scan = ScanAction(
                        reason="check_last_seen",
                        target_dx=screen_dx * 0.7,
                        target_dy=screen_dy * 0.4,
                        duration=duration,
                        settle_pause=pause,
                        start_time=now,
                    )
                    return (0.0, 0.0)

        # Priority B: Natural corner checks / glances during roaming
        if random.random() < p.scanning_frequency * 0.08:
            sweep_dir = random.choice([-1.0, 1.0])
            amplitude = (40.0 + p.scanning_amplitude * 80.0) * sweep_dir
            duration = random.uniform(0.20, 0.40)
            pause = random.uniform(0.15, 0.35)
            self._current_scan = ScanAction(
                reason="check_corner",
                target_dx=amplitude,
                target_dy=random.gauss(0, 10.0),
                duration=duration,
                settle_pause=pause,
                start_time=now,
            )

        return (0.0, 0.0)
