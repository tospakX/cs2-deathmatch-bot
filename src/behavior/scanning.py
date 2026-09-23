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

from src.behavior.player_state import TargetStatus

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState


@dataclass
class ScanAction:
    """An intentional camera scan with a specific reason and duration."""

    reason: str  # "check_spatial_memory", "check_last_seen", "check_corner", "curiosity_glance"
    target_dx: float  # total mouse delta to sweep
    target_dy: float
    duration: float  # total duration in seconds
    settle_pause: float  # time to pause and look after sweeping
    start_time: float = 0.0


class ScanningController:
    """Manages purposeful camera movements during non-combat and search phases."""

    def __init__(
        self,
        personality: PersonalityTraits,
        screen_center: tuple[int, int],
        sensitivity: float = 1.0,
        m_yaw: float = 0.022,
        m_pitch: float = 0.022,
        screen_size: tuple[int, int] = (3440, 1440),
        fov_h: float = 122.0,
    ):
        self.personality = personality
        self.cx, self.cy = screen_center
        self.sensitivity = sensitivity
        self.m_yaw = m_yaw
        self.m_pitch = m_pitch
        self.screen_w, self.screen_h = screen_size
        self.fov_h = fov_h
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

        # If in active combat engagement with a visible target, scanning is suppressed
        if (
            state.primary_target is not None
            and state.primary_target.frames_missing == 0
            and state.primary_target.status == TargetStatus.VISIBLE
        ):
            self._current_scan = None
            return (0.0, 0.0)

        # ── 1. If currently executing a scan ──────────────────────────────────
        if self._current_scan is not None:
            scan = self._current_scan
            elapsed = now - scan.start_time

            if elapsed < scan.duration:
                # Active sweep phase: use smoothstep bell-curve velocity
                t = elapsed / max(scan.duration, 0.001)
                velocity_weight = 6.0 * t * (1.0 - t)
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

        # Priority A: Check directional angular spatial memory with temporal decay
        valid_memories = [m for m in state.spatial_memories if (now - m.last_seen_time) < 6.0]
        if valid_memories:
            mem = valid_memories[-1]
            state.spatial_memories.remove(mem)
            age = now - mem.last_seen_time

            # Uncertainty jitter proportional to elapsed time (memory decay)
            uncert_yaw = random.gauss(0, age * 1.5)
            uncert_pitch = random.gauss(0, age * 0.8)

            yaw_deg = mem.yaw_offset_deg + uncert_yaw
            pitch_deg = mem.pitch_offset_deg + uncert_pitch

            # Convert angular offsets to mouse counts
            deg_per_count_x = self.m_yaw * self.sensitivity
            deg_per_count_y = self.m_pitch * self.sensitivity
            mouse_dx = yaw_deg / max(1e-5, deg_per_count_x)
            mouse_dy = pitch_deg / max(1e-5, deg_per_count_y)

            dist_counts = math.hypot(mouse_dx, mouse_dy)
            if dist_counts > 25.0:
                duration = random.uniform(0.20, 0.38)
                pause = random.uniform(0.25, 0.50)
                self._current_scan = ScanAction(
                    reason="check_spatial_memory",
                    target_dx=mouse_dx * 0.75,
                    target_dy=mouse_dy * 0.50,
                    duration=duration,
                    settle_pause=pause,
                    start_time=now,
                )
                return (0.0, 0.0)

        # Priority B: Fallback to last seen screen positions
        if state.last_enemy_positions:
            last_x, last_y, last_t = state.last_enemy_positions[-1]
            if now - last_t < 4.0:
                screen_dx = last_x - self.cx
                screen_dy = last_y - self.cy
                dist = math.hypot(screen_dx, screen_dy)
                if dist > 30.0:
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

        # Priority C: Natural corner checks / glances during roaming
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
