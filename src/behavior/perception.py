"""Perception layer — converts raw detections into tracked targets.

Handles temporal tracking of detected enemies, position history,
velocity estimation, and target identity persistence across frames.
This is *what is visible*, not *what the player is focusing on*
(that's the attention layer).
"""

from __future__ import annotations

import math

from src.behavior.player_state import PlayerState, SpatialMemory, TargetStatus, TrackedTarget
from src.utils.math_helpers import distance
from src.vision.detector import Detection

# Maximum pixel distance to consider a detection as the same target.
_MATCH_DISTANCE = 220.0
# Maximum bbox size ratio change to match.
_MATCH_SIZE_RATIO = 0.35
# Frames before a missing target is forgotten.
_FORGET_FRAMES = 15


# Maximum match cost threshold beyond which pairs are rejected
_MAX_MATCH_COST = 350.0


def hungarian_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Solve minimum weight bipartite matching problem using Kuhn-Munkres (Hungarian) algorithm.

    Args:
        cost_matrix: N x M matrix where cost_matrix[i][j] is the cost of assigning row i to col j.

    Returns:
        List of (row, col) matches that minimize total cost.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []

    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    dim = max(n_rows, n_cols)
    inf = 1e9

    # Pad matrix to dim x dim
    cost = [[inf] * dim for _ in range(dim)]
    for i in range(n_rows):
        for j in range(n_cols):
            cost[i][j] = cost_matrix[i][j]

    u = [0.0] * (dim + 1)
    v = [0.0] * (dim + 1)
    p = [0] * (dim + 1)
    way = [0] * (dim + 1)

    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (dim + 1)
        used = [False] * (dim + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, dim + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches: list[tuple[int, int]] = []
    for j in range(1, dim + 1):
        if p[j] > 0:
            r = p[j] - 1
            c = j - 1
            if r < n_rows and c < n_cols:
                matches.append((r, c))

    return matches


class PerceptionSystem:
    """Tracks detected enemies across frames, maintaining identity and history.

    Each detected enemy is assigned a track_id that persists as long as
    the enemy keeps appearing in a consistent position. The system
    estimates velocity from position changes and maintains visibility
    history.
    """

    def __init__(
        self,
        screen_center: tuple[int, int],
        screen_size: tuple[int, int] = (3440, 1440),
        fov_h: float = 122.0,
    ):
        self.cx, self.cy = screen_center
        self.screen_w, self.screen_h = screen_size
        self.fov_h = fov_h
        aspect = self.screen_w / max(1.0, float(self.screen_h))
        self.fov_v = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov_h / 2.0)) / aspect))

    def update(self, state: PlayerState, detections: list[Detection]) -> None:
        """Process new detections and update tracked targets in player state.

        Uses global minimum-cost bipartite matching (Hungarian algorithm) to prevent
        target identity flapping when enemies cross paths.
        """
        now = state.now

        # Filter to body detections only (skip head-only boxes).
        body_dets = [d for d in detections if not d.is_head]
        tracks = list(state.known_targets.items())
        n_tracks = len(tracks)
        n_dets = len(body_dets)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if n_tracks > 0 and n_dets > 0:
            # ── 1. Global Cost Matrix Computation ──────────────────────────
            cost_matrix = [[1e6] * n_dets for _ in range(n_tracks)]

            for r_idx, (track_id, target) in enumerate(tracks):
                old_cx, old_cy = target.detection.center
                old_area = target.detection.area
                old_w = target.detection.width
                old_h = max(1.0, target.detection.height)
                old_aspect = old_w / old_h

                # Position prediction using target velocity
                vx, vy = target.velocity_estimate
                dt = max(0.001, min(0.3, now - target.last_seen))
                pred_cx = old_cx + vx * dt
                pred_cy = old_cy + vy * dt

                for c_idx, det in enumerate(body_dets):
                    det_cx, det_cy = det.center
                    det_w = det.width
                    det_h = max(1.0, det.height)
                    det_aspect = det_w / det_h

                    dist = distance((pred_cx, pred_cy), (det_cx, det_cy))
                    if dist > _MATCH_DISTANCE:
                        continue

                    # Size similarity
                    if old_area > 0 and det.area > 0:
                        size_ratio = min(det.area, old_area) / max(det.area, old_area)
                        if size_ratio < _MATCH_SIZE_RATIO:
                            continue
                        size_cost = (1.0 - size_ratio) * 100.0
                    else:
                        size_cost = 50.0

                    # Aspect ratio similarity
                    aspect_diff = abs(det_aspect - old_aspect) / max(det_aspect, old_aspect)
                    aspect_cost = aspect_diff * 50.0

                    cost_matrix[r_idx][c_idx] = dist + size_cost + aspect_cost

            # Solve global minimum-cost assignment
            assignments = hungarian_assignment(cost_matrix)

            for r_idx, c_idx in assignments:
                cost = cost_matrix[r_idx][c_idx]
                if cost < _MAX_MATCH_COST:
                    track_id, target = tracks[r_idx]
                    det = body_dets[c_idx]
                    matched_tracks.add(track_id)
                    matched_dets.add(c_idx)

                    target.detection = det
                    cx, cy = det.center
                    target.update_position(cx, cy, now)
                    target.threat_level = self._compute_threat(det)

        # Mark unmatched tracks as missing
        for track_id, target in tracks:
            if track_id not in matched_tracks:
                target.mark_missing()

        # ── 2. Create new tracked targets for unmatched detections ──────────
        for det_idx, det in enumerate(body_dets):
            if det_idx in matched_dets:
                continue

            track_id = state.assign_track_id()
            cx, cy = det.center
            target = TrackedTarget(
                detection=det,
                target_id=track_id,
                first_seen=now,
                last_seen=now,
                frames_visible=1,
                threat_level=self._compute_threat(det),
                status=TargetStatus.VISIBLE,
            )
            target.update_position(cx, cy, now)
            state.known_targets[track_id] = target

        # ── 3. Prune targets missing too long & record true historical memory ──
        to_remove = []
        for track_id, target in state.known_targets.items():
            if target.frames_missing > _FORGET_FRAMES:
                if len(target.position_history) > 0:
                    last_pos = target.position_history[-1]
                    px, py, true_timestamp = last_pos

                    # Directional representation relative to camera view
                    yaw_deg = ((px - self.cx) / max(1.0, float(self.screen_w))) * self.fov_h
                    pitch_deg = ((py - self.cy) / max(1.0, float(self.screen_h))) * self.fov_v

                    # Store in spatial memory with true observation timestamp
                    spatial = SpatialMemory(
                        screen_x=px,
                        screen_y=py,
                        yaw_offset_deg=yaw_deg,
                        pitch_offset_deg=pitch_deg,
                        last_seen_time=true_timestamp,
                        confidence=target.detection.confidence,
                        velocity=target.velocity_estimate,
                    )
                    state.spatial_memories.append(spatial)
                    state.last_enemy_positions.append((px, py, true_timestamp))

                to_remove.append(track_id)

        for track_id in to_remove:
            removed = state.known_targets.pop(track_id, None)
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
        visible = [
            t
            for t in state.known_targets.values()
            if t.frames_missing == 0 and t.status == TargetStatus.VISIBLE
        ]
        visible.sort(key=lambda t: t.threat_level, reverse=True)
        return visible
