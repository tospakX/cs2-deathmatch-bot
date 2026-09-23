"""Perception layer — converts raw detections into tracked targets.

Handles temporal tracking of detected enemies, position history,
velocity estimation, and target identity persistence across frames.
This is *what is visible*, not *what the player is focusing on*
(that's the attention layer).
"""

from __future__ import annotations

from src.behavior.player_state import PlayerState, TrackedTarget
from src.utils.math_helpers import distance
from src.vision.detector import Detection

# Maximum pixel distance to consider a detection as the same target.
_MATCH_DISTANCE = 180.0
# Maximum bbox size ratio change to match.
_MATCH_SIZE_RATIO = 0.45
# Frames before a missing target is forgotten.
_FORGET_FRAMES = 15


class PerceptionSystem:
    """Tracks detected enemies across frames, maintaining identity and history.

    Each detected enemy is assigned a track_id that persists as long as
    the enemy keeps appearing in a consistent position.  The system
    estimates velocity from position changes and maintains visibility
    history.
    """

    def __init__(self, screen_center: tuple[int, int]):
        self.cx, self.cy = screen_center

    def update(self, state: PlayerState, detections: list[Detection]) -> None:
        """Process new detections and update tracked targets in player state.

        1. Match new detections to existing tracked targets.
        2. Update matched targets with new positions.
        3. Create new tracked targets for unmatched detections.
        4. Mark and prune targets that have been missing too long.
        """
        now = state.now
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Filter to body detections only (skip head-only boxes).
        body_dets = [d for d in detections if not d.is_head]

        # Match existing targets to new detections.
        for track_id, target in list(state.known_targets.items()):
            best_det_idx = -1
            best_dist = _MATCH_DISTANCE

            old_cx, old_cy = target.detection.center
            old_area = target.detection.area

            for i, det in enumerate(body_dets):
                if i in matched_dets:
                    continue
                new_cx, new_cy = det.center
                d = distance((old_cx, old_cy), (new_cx, new_cy))
                # Also check size similarity.
                if old_area > 0:
                    size_ratio = min(det.area, old_area) / max(det.area, old_area)
                    if size_ratio < _MATCH_SIZE_RATIO:
                        continue
                if d < best_dist:
                    best_dist = d
                    best_det_idx = i

            if best_det_idx >= 0:
                det = body_dets[best_det_idx]
                target.detection = det
                cx, cy = det.center
                target.update_position(cx, cy, now)
                target.threat_level = self._compute_threat(det)
                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)
            else:
                target.mark_missing()

        # Create new tracked targets for unmatched detections.
        for i, det in enumerate(body_dets):
            if i in matched_dets:
                continue

            track_id = state.assign_track_id()
            cx, cy = det.center
            target = TrackedTarget(
                detection=det,
                first_seen=now,
                last_seen=now,
                frames_visible=1,
                threat_level=self._compute_threat(det),
            )
            target.update_position(cx, cy, now)
            state.known_targets[track_id] = target

        # Prune targets that have been missing too long.
        to_remove = []
        for track_id, target in state.known_targets.items():
            if target.frames_missing > _FORGET_FRAMES:
                # Record last known position for scanning.
                if len(target.position_history) > 0:
                    last_pos = target.position_history[-1]
                    state.last_enemy_positions.append((last_pos[0], last_pos[1], now))
                to_remove.append(track_id)

        for track_id in to_remove:
            removed = state.known_targets.pop(track_id, None)
            # If we lost our primary target, clear it.
            if removed is not None and removed is state.primary_target:
                state.primary_target = None

    def _compute_threat(self, detection: Detection) -> float:
        """Score a detection's threat level (higher = more threatening)."""
        cx, cy = detection.center
        crosshair_dist = distance((self.cx, self.cy), (cx, cy))

        # Closer to crosshair = higher threat.
        proximity_score = max(0.0, 1000.0 - crosshair_dist)
        # Larger bbox = closer enemy.
        size_score = detection.area / 100.0
        # Confidence boost.
        conf_score = detection.confidence * 200.0

        return proximity_score + size_score + conf_score

    def get_visible_targets(self, state: PlayerState) -> list[TrackedTarget]:
        """Return currently visible targets sorted by threat."""
        visible = [t for t in state.known_targets.values() if t.frames_missing == 0]
        visible.sort(key=lambda t: t.threat_level, reverse=True)
        return visible
