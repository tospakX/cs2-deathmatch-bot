"""Instrumentation and evaluation harness for human-like behavioral metrics.

Records high-resolution behavioral timelines and evaluates:
- Reaction-time distribution & temporal autocorrelation (lag-1 correlation > 0)
- Aim movement episode duration vs distance & submovement counts
- Tracking lag and lead velocity scaling
- Fire-mode episode commitments & burst length distributions
- Directional persistence & non-alternating Markov footwork
- Crouch episode durations (absence of 30Hz flutter)
- State persistence and transitions
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.behavior.player_state import PlayerState


@dataclass
class BehavioralTickRecord:
    """Complete chronological snapshot of a single simulation tick."""

    timestamp: float
    bot_phase: str
    aim_phase: str
    motor_phase: str
    firing_phase: str
    movement_phase: str
    movement_direction: str
    is_crouching: bool
    is_firing: bool
    is_trigger_held: bool
    target_id: int
    target_aim_preference: str
    screen_error_dist: float
    motor_dx: int
    motor_dy: int
    recoil_dx: int
    recoil_dy: int
    shot_count: int
    reaction_pending: bool
    reaction_duration: float
    reaction_tendency: float
    confidence: float
    awareness_level: float


class BehaviorRecorder:
    """Records tick-by-tick simulation snapshots and computes behavioral metrics."""

    def __init__(self):
        self.records: list[BehavioralTickRecord] = []

    def record_tick(
        self,
        state: PlayerState,
        motor_dx: int = 0,
        motor_dy: int = 0,
        recoil_dx: int = 0,
        recoil_dy: int = 0,
    ) -> None:
        """Capture state snapshot for this tick."""
        target_id = state.primary_target.target_id if state.primary_target else -1
        pref = state.primary_target.aim_preference if state.primary_target else ""
        err_dist = (
            state.authoritative_aim.screen_error_dist if state.authoritative_aim.is_valid else 999.0
        )

        rec = BehavioralTickRecord(
            timestamp=state.now,
            bot_phase=state.phase.name,
            aim_phase=state.aim_phase.name,
            motor_phase=state.motor_phase.name,
            firing_phase=state.firing_phase.name,
            movement_phase=state.movement_phase.name,
            movement_direction=state.movement_direction,
            is_crouching=state.is_crouching,
            is_firing=state.is_trigger_held,
            is_trigger_held=state.is_trigger_held,
            target_id=target_id,
            target_aim_preference=pref,
            screen_error_dist=err_dist,
            motor_dx=motor_dx,
            motor_dy=motor_dy,
            recoil_dx=recoil_dx,
            recoil_dy=recoil_dy,
            shot_count=state.shot_count,
            reaction_pending=state.reaction_pending,
            reaction_duration=state.reaction_duration,
            reaction_tendency=state.reaction_tendency,
            confidence=state.confidence,
            awareness_level=state.awareness_level,
        )
        self.records.append(rec)

    def to_dict_list(self) -> list[dict]:
        return [asdict(r) for r in self.records]

    # ── Behavioral Metric Extractors ──────────────────────────────────────

    def compute_reaction_metrics(self) -> dict[str, float]:
        """Compute reaction time mean, std, and autocorrelation."""
        durations = [r.reaction_duration for r in self.records if r.reaction_duration > 0]
        # Remove consecutive duplicate readings from the same reaction event
        unique_reactions: list[float] = []
        for d in durations:
            if not unique_reactions or abs(d - unique_reactions[-1]) > 0.0001:
                unique_reactions.append(d)

        if len(unique_reactions) < 2:
            return {"count": float(len(unique_reactions)), "mean": 0.0, "autocorr": 0.0}

        mean = sum(unique_reactions) / len(unique_reactions)
        var = sum((x - mean) ** 2 for x in unique_reactions) / len(unique_reactions)
        std = math.sqrt(var) if var > 0 else 0.0

        # Lag-1 Autocorrelation
        autocorr = 0.0
        if var > 1e-9 and len(unique_reactions) > 2:
            cov = sum(
                (unique_reactions[i] - mean) * (unique_reactions[i + 1] - mean)
                for i in range(len(unique_reactions) - 1)
            ) / (len(unique_reactions) - 1)
            autocorr = cov / var

        return {
            "count": float(len(unique_reactions)),
            "mean": mean,
            "std": std,
            "autocorr": autocorr,
        }

    def compute_movement_metrics(self) -> dict[str, float]:
        """Verify that movement directions do NOT follow robotic 100% left-right alternation."""
        dirs = [
            r.movement_direction for r in self.records if r.movement_direction in ("left", "right")
        ]
        # Group into direction episodes
        episodes: list[str] = []
        for d in dirs:
            if not episodes or d != episodes[-1]:
                episodes.append(d)

        same_transitions = 0
        opposite_transitions = 0
        for i in range(len(episodes) - 1):
            if episodes[i] == episodes[i + 1]:
                same_transitions += 1
            else:
                opposite_transitions += 1

        total = same_transitions + opposite_transitions
        alt_ratio = (opposite_transitions / total) if total > 0 else 0.0

        return {
            "episode_count": float(len(episodes)),
            "same_transitions": float(same_transitions),
            "opposite_transitions": float(opposite_transitions),
            "alternation_ratio": alt_ratio,  # 1.0 means pure robotic L-R-L-R
        }

    def compute_crouch_metrics(self) -> dict[str, float]:
        """Verify that crouch does not pulse frame-to-frame."""
        crouch_durations: list[float] = []
        current_start: float | None = None

        for r in self.records:
            if r.is_crouching:
                if current_start is None:
                    current_start = r.timestamp
            else:
                if current_start is not None:
                    crouch_durations.append(r.timestamp - current_start)
                    current_start = None

        if current_start is not None and self.records:
            crouch_durations.append(self.records[-1].timestamp - current_start)

        min_dur = min(crouch_durations) if crouch_durations else 0.0
        avg_dur = (sum(crouch_durations) / len(crouch_durations)) if crouch_durations else 0.0

        return {
            "episode_count": float(len(crouch_durations)),
            "min_duration": min_dur,
            "avg_duration": avg_dur,
        }
