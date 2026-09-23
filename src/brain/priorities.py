"""Target selection and threat assessment with target persistence and attention inertia."""

from __future__ import annotations

from src.utils.math_helpers import distance
from src.vision.detector import Detection


class ThreatAssessor:
    """Evaluate and prioritize detected enemies with attention inertia."""

    def __init__(self, screen_center: tuple[int, int], persistence_bias: float = 120.0):
        self.cx, self.cy = screen_center
        self.persistence_bias = persistence_bias
        self.current_target: Detection | None = None
        self._target_seen_count: int = 0

    def assess_threat(self, detection: Detection, is_current: bool = False) -> float:
        """Score a detection's threat level (higher = more threatening).

        Factors:
        - Distance to crosshair (closer = more threat)
        - Size of bounding box (closer enemies = bigger bbox)
        - Confidence (higher confidence = more certain it's real)
        - Attention persistence bonus if this is already our engaged target
        """
        cx, cy = detection.center
        crosshair_dist = distance((self.cx, self.cy), (cx, cy))

        # Closer to crosshair = higher threat
        proximity_score = max(0.0, 1000.0 - crosshair_dist)

        # Larger bbox = closer enemy = more threat
        size_score = detection.area / 100.0

        # Confidence boost
        conf_score = detection.confidence * 200.0

        # Inertia bonus: stick with current target to avoid frame-by-frame target flutter
        persistence_score = self.persistence_bias if is_current else 0.0

        return proximity_score + size_score + conf_score + persistence_score

    def prioritize_targets(
        self, detections: list[Detection], our_team: str = ""
    ) -> list[Detection]:
        """Sort detections by threat level with target persistence.

        Returns enemies sorted by threat (highest first).
        """
        enemies = [d for d in detections if not d.is_head]

        if not enemies:
            self.current_target = None
            self._target_seen_count = 0
            return []

        # Find if current target matches any new detection
        matched_current = None
        if self.current_target is not None:
            tcx, tcy = self.current_target.center
            for e in enemies:
                ecx, ecy = e.center
                if ((ecx - tcx) ** 2 + (ecy - tcy) ** 2) < (120.0**2):
                    matched_current = e
                    break

        # Sort by threat score (descending), applying persistence bonus to matched current
        enemies.sort(
            key=lambda d: self.assess_threat(d, is_current=(d is matched_current)),
            reverse=True,
        )

        self.current_target = enemies[0]
        self._target_seen_count += 1
        return enemies

    def should_switch_target(
        self,
        current: Detection | None,
        new_targets: list[Detection],
        switch_threshold: float = 200.0,
    ) -> bool:
        """Decide if we should switch to a new target considering switching costs."""
        if current is None:
            return True

        if not new_targets:
            return False

        best = new_targets[0]
        current_threat = self.assess_threat(current, is_current=True)
        best_threat = self.assess_threat(best, is_current=False)

        return best_threat > (current_threat + switch_threshold)
