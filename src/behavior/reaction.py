"""Contextual reaction pipeline.

Models the full human reaction chain:
  notice → visual confirmation → decision → motor initiation

Reaction time depends on context (surprise, visibility, attention state,
current engagement, target position) and has temporal correlation —
a player maintains a consistent short-term reaction tendency stored in
PlayerState that drifts slowly rather than independently sampling fresh
values for every event.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.behavior.player_state import AimPhase, BotPhase

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState, TrackedTarget


class ReactionSystem:
    """Context-dependent reaction timing with temporal continuity."""

    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self._tendency_drift_rate: float = 0.02

    def initiate_reaction(
        self,
        state: PlayerState,
        target: TrackedTarget,
        context: str = "new_target",
    ) -> None:
        """Start a new reaction timer based on context.

        The reaction blocks aiming (aim_phase stays REACTING) until the
        reaction duration elapses.
        """
        p = self.personality
        modifier = self._compute_context_modifier(state, target, context)

        # Apply short-term tendency directly from state
        modifier *= state.reaction_tendency

        # Sample reaction time using personality distribution
        duration = p.sample_reaction_time(modifier)

        # Record in state
        state.reaction_pending = True
        state.reaction_start = state.now
        state.reaction_duration = duration
        state.reaction_context = context
        state.aim_phase = AimPhase.REACTING

    def update(self, state: PlayerState) -> None:
        """Check if reaction has completed. Called every tick."""
        if not state.reaction_pending:
            return

        elapsed = state.now - state.reaction_start
        if elapsed >= state.reaction_duration:
            # Reaction complete — transition to acquiring
            state.reaction_pending = False
            state.aim_phase = AimPhase.ACQUIRING
            state.recent_reaction_times.append(state.reaction_duration)
            # Drift tendency in state
            self._drift_tendency(state)

    def _compute_context_modifier(
        self,
        state: PlayerState,
        target: TrackedTarget,
        context: str,
    ) -> float:
        """Compute a multiplier for reaction time based on context.

        < 1.0 = faster than base (expected target).
        > 1.0 = slower than base (surprise).
        """
        modifier = 1.0

        # ── Surprise vs expectation ──────────────────────────────────────
        if context == "surprise":
            modifier *= 1.25
        elif context == "reacquire":
            modifier *= 0.72
        elif context == "switch":
            modifier *= 0.85

        # ── Target visibility ────────────────────────────────────────────
        if target.frames_visible > 8:
            modifier *= 0.82
        elif target.frames_visible <= 2:
            modifier *= 1.18

        # ── Attention state ──────────────────────────────────────────────
        if state.attention_confidence > 0.7:
            modifier *= 0.85
        elif state.attention_confidence < 0.3:
            modifier *= 1.15

        # ── Current engagement ───────────────────────────────────────────
        if state.phase == BotPhase.ENGAGING:
            modifier *= 0.90

        # ── Awareness level ──────────────────────────────────────────────
        modifier *= 1.25 - state.awareness_level * 0.45

        # ── Confidence ───────────────────────────────────────────────────
        modifier *= 1.10 - state.confidence * 0.20

        return max(0.55, min(1.85, modifier))

    def _drift_tendency(self, state: PlayerState) -> None:
        """Slowly drift the unified reaction tendency in PlayerState."""
        drift = random.gauss(0, self._tendency_drift_rate)
        state.reaction_tendency += drift
        # Mean-revert toward 1.0
        state.reaction_tendency += (1.0 - state.reaction_tendency) * 0.08
        state.reaction_tendency = max(0.75, min(1.30, state.reaction_tendency))
