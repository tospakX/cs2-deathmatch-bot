"""Contextual mistake injection for human-like imperfection.

Replaces uniform Gaussian error + forced overshoot sequences with:
- Contextual error magnitude depending on movement, distance, and confidence
- Undershoot and overshoot with realistic probabilities
- Delayed correction or hesitation
- Short-term precision consistency rather than frame-by-frame chaos
"""

from __future__ import annotations

import math
import random


class MistakeMaker:
    """Injects contextual human-like mistakes into aim and motor execution."""

    def __init__(
        self,
        overshoot_chance: float = 0.35,
        overshoot_magnitude: float = 1.3,
        tracking_error: float = 8.0,
    ):
        self.overshoot_chance = overshoot_chance
        self.overshoot_magnitude = overshoot_magnitude
        self.tracking_error = tracking_error
        # Short-term steadiness bias (creates streaks of good or bad aim)
        self._steadiness_factor: float = 1.0

    def apply_aim_error(
        self,
        target_x: float,
        target_y: float,
        confidence: float = 0.5,
        is_moving: bool = False,
    ) -> tuple[float, float]:
        """Add contextual error to an aim target point."""
        # Slowly drift steadiness
        self._steadiness_factor += random.gauss(0, 0.05)
        self._steadiness_factor += (1.0 - self._steadiness_factor) * 0.1
        self._steadiness_factor = max(0.6, min(1.5, self._steadiness_factor))

        # Error scale modulates with confidence and movement
        scale = self.tracking_error * self._steadiness_factor
        if is_moving:
            scale *= 1.3
        scale *= 1.4 - confidence * 0.6

        err_x = random.gauss(0, scale)
        err_y = random.gauss(0, scale * 0.8)  # human vertical error is often lower than horizontal
        return target_x + err_x, target_y + err_y

    def should_overshoot(self, distance_pixels: float = 100.0) -> bool:
        """Decide if this aim movement should overshoot based on distance."""
        if distance_pixels < 20.0:
            return False
        # Longer flicks have higher chance of overshoot
        prob = self.overshoot_chance * min(1.3, distance_pixels / 80.0)
        return random.random() < prob

    def should_undershoot(self, distance_pixels: float = 100.0) -> bool:
        """Decide if this aim movement should undershoot (bail out early)."""
        if distance_pixels < 30.0:
            return False
        return random.random() < 0.20

    def overshoot_target(
        self, current_x: float, current_y: float, target_x: float, target_y: float
    ) -> tuple[float, float]:
        """Calculate an overshoot point past the target."""
        dx = target_x - current_x
        dy = target_y - current_y

        factor = self.overshoot_magnitude * random.uniform(0.85, 1.15)
        overshoot_x = current_x + dx * factor
        overshoot_y = current_y + dy * factor

        # Slight angular deviation
        angle_err = random.gauss(0, 2.5)  # degrees
        rad = math.radians(angle_err)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rel_x = overshoot_x - current_x
        rel_y = overshoot_y - current_y
        overshoot_x = current_x + rel_x * cos_a - rel_y * sin_a
        overshoot_y = current_y + rel_x * sin_a + rel_y * cos_a

        return overshoot_x, overshoot_y

    def undershoot_target(
        self, current_x: float, current_y: float, target_x: float, target_y: float
    ) -> tuple[float, float]:
        """Calculate an undershoot point short of the target."""
        factor = random.uniform(0.70, 0.90)
        return (
            current_x + (target_x - current_x) * factor,
            current_y + (target_y - current_y) * factor,
        )

    def micro_correction_count(self, corrections_range: list[int] | None = None) -> int:
        """How many micro-corrections to make after initial flick."""
        if corrections_range is None:
            corrections_range = [1, 3]
        lo, hi = corrections_range
        return random.randint(lo, hi)

    def should_whiff(self, base_chance: float = 0.04) -> bool:
        """Occasionally completely miss under panic."""
        return random.random() < (base_chance * self._steadiness_factor)
