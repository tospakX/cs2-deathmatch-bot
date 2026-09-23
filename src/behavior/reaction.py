"""Contextual reaction pipeline.

Models the full human reaction chain:
  notice → visual confirmation → decision → motor initiation

Reaction time depends on context (surprise, visibility, attention state,
current engagement, target position) and has temporal correlation — a
player maintains a reasonably consistent short-term reaction tendency
rather than independently sampling fresh values for every event.
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
        # Short-term reaction tendency: drifts slowly, not reset per event.
        self._tendency: float = 1.0
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

        # Apply short-term tendency (temporal correlation).
        modifier *= self._tendency

        # Sample reaction time.
        duration = p.sample_reaction_time(modifier)

        # Record.
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
            # Reaction complete — transition to acquiring.
            state.reaction_pending = False
            state.aim_phase = AimPhase.ACQUIRING
            state.recent_reaction_times.append(state.reaction_duration)
            # Slowly drift tendency.
            self._drift_tendency()

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
            # Target appeared unexpectedly.
            modifier *= 1.3
        elif context == "reacquire":
            # Target was recently tracked — faster to react.
            modifier *= 0.7
        elif context == "switch":
            # Switching to already-visible target.
            modifier *= 0.85

        # ── Target visibility ────────────────────────────────────────────
        if target.frames_visible > 10:
            # Target has been visible for a while — faster.
            modifier *= 0.8
        elif target.frames_visible <= 2:
            # Just appeared — slower.
            modifier *= 1.15

        # ── Attention state ──────────────────────────────────────────────
        if state.attention_confidence > 0.7:
            # Already focused — faster.
            modifier *= 0.85
        elif state.attention_confidence < 0.3:
            # Unfocused — slower.
            modifier *= 1.2

        # ── Current engagement ───────────────────────────────────────────
        if state.phase == BotPhase.ENGAGING:
            # Already in combat — faster reactions to new threats.
            modifier *= 0.9

        # ── Awareness level ──────────────────────────────────────────────
        # High awareness (just lost a target, heard something) = faster.
        modifier *= 1.3 - state.awareness_level * 0.5

        # ── Confidence ───────────────────────────────────────────────────
        # High confidence = slightly faster reactions.
        modifier *= 1.1 - state.confidence * 0.2

        return max(0.5, min(2.0, modifier))

    def _drift_tendency(self) -> None:
        """Slowly drift the short-term reaction tendency.

        This creates temporal correlation: a player who was reacting
        fast will tend to keep reacting fast, and vice versa.
        """
        # Small random walk.
        drift = random.gauss(0, self._tendency_drift_rate)
        self._tendency += drift
        # Mean-revert toward 1.0.
        self._tendency += (1.0 - self._tendency) * 0.1
        self._tendency = max(0.7, min(1.3, self._tendency))
