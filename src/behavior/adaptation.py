"""State-driven adaptation responding to recent combat events and performance.

Adapts the simulated player's short-term tendencies based on:
- Hit vs miss evidence (accumulated over streaks with realistic uncertainty)
- Taking sudden damage (spikes awareness, increases evasive movement, triggers retreat)
- Successful engagements (boosts confidence and aggression)
- Target loss (elevates awareness and initiates focused search)
- Target-switching fatigue (increases persistence to avoid rapid target thrashing)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.behavior.personality import PersonalityTraits
    from src.behavior.player_state import PlayerState


class AdaptationEngine:
    """Modulates player behavior dynamically in response to recent game events."""

    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self._consecutive_misses: int = 0
        self._consecutive_hits: int = 0
        self._last_evaluated_shot_count: int = 0
        self._last_evaluated_health: int = 100
        self._frustration: float = 0.0  # 0..1 scale
        self._hit_evidence_accumulator: float = 0.0
        self._miss_evidence_accumulator: float = 0.0

    def update(self, state: PlayerState) -> None:
        """Evaluate recent combat history and update adaptive behavioral parameters."""
        now = state.now

        # ── 1. Damage-driven adaptation ──────────────────────────────────────
        if state.health < self._last_evaluated_health:
            damage_taken = self._last_evaluated_health - state.health
            state.damage_taken_recently += damage_taken
            state.last_damage_time = now

            # Sudden damage spikes awareness and lowers immediate confidence
            state.awareness_level = min(1.0, state.awareness_level + 0.25)
            state.confidence = max(0.1, state.confidence - 0.04 * (damage_taken / 20.0))
            self._frustration = min(1.0, self._frustration + 0.1)

        self._last_evaluated_health = state.health

        # Decay damage taken recently
        if now - state.last_damage_time > 3.0:
            state.damage_taken_recently = 0
            state.awareness_level = max(0.4, state.awareness_level - state.tick_dt * 0.05)

        # ── 2. Shot accuracy with uncertainty accumulation ───────────────────
        if state.shot_count > self._last_evaluated_shot_count:
            new_shots = state.shot_count - self._last_evaluated_shot_count
            self._last_evaluated_shot_count = state.shot_count

            # Evaluate aim error at time of shot
            if state.authoritative_aim.is_valid:
                err_dist = state.authoritative_aim.screen_error_dist
            elif state.current_aim_error != (0.0, 0.0):
                err_dist = (
                    state.current_aim_error[0] ** 2 + state.current_aim_error[1] ** 2
                ) ** 0.5
            else:
                err_dist = state.authoritative_aim.screen_error_dist

            # Probabilistic hit estimation (maintains realistic uncertainty)
            hit_prob = max(0.0, min(1.0, 1.0 - (err_dist / 40.0)))
            self._hit_evidence_accumulator += hit_prob * new_shots
            self._miss_evidence_accumulator += (1.0 - hit_prob) * new_shots

            # Require accumulated evidence before altering confidence
            if self._hit_evidence_accumulator >= 1.0:
                self._consecutive_hits += 1
                self._consecutive_misses = 0
                self._hit_evidence_accumulator = 0.0
                state.update_confidence(hit=True)
                self._frustration = max(0.0, self._frustration - 0.04)

            if self._miss_evidence_accumulator >= 1.5:
                self._consecutive_misses += 1
                self._consecutive_hits = 0
                self._miss_evidence_accumulator = 0.0
                state.update_confidence(hit=False)

                if self._consecutive_misses >= 2:
                    self._frustration = min(1.0, self._frustration + 0.10)
                    state.reaction_tendency = min(1.25, state.reaction_tendency + 0.05)

        if state.shot_count == 0:
            self._last_evaluated_shot_count = 0

        # ── 3. Frustration decay ─────────────────────────────────────────────
        if self._frustration > 0:
            self._frustration = max(0.0, self._frustration - state.tick_dt * 0.02)

        # ── 4. Target switch frequency fatigue ────────────────────────────────
        recent_switches = [t for t, _ in state.target_switch_history if now - t < 2.0]
        if len(recent_switches) >= 2:
            state.awareness_level = min(1.0, state.awareness_level + 0.1)

    def on_engagement_won(self, state: PlayerState) -> None:
        """Call when an engaged enemy is eliminated."""
        state.record_engagement_outcome(won=True)
        self._consecutive_misses = 0
        self._hit_evidence_accumulator = 0.0
        self._miss_evidence_accumulator = 0.0
        self._frustration = max(0.0, self._frustration - 0.25)
        # Slight boost in reaction speed following a kill
        state.reaction_tendency = max(0.85, state.reaction_tendency - 0.06)

    def on_engagement_lost(self, state: PlayerState) -> None:
        """Call on death or failed engagement."""
        state.record_engagement_outcome(won=False)
        self._consecutive_hits = 0
        self._consecutive_misses = 0
        self._hit_evidence_accumulator = 0.0
        self._miss_evidence_accumulator = 0.0
        self._frustration = min(1.0, self._frustration + 0.20)
