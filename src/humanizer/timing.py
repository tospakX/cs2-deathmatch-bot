"""Reaction time distributions and timing humanization with temporal continuity."""

from __future__ import annotations

import random
import time


class ReactionTimer:
    """Generates human-like reaction times with temporal correlation."""

    def __init__(
        self,
        mean_ms: float = 250.0,
        std_ms: float = 50.0,
        min_ms: float = 150.0,
        max_ms: float = 500.0,
    ):
        self.mean_ms = mean_ms
        self.std_ms = std_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self._pending_reaction: float | None = None
        self._reaction_start: float = 0.0
        # Short-term reaction tendency drifts slowly rather than jumping wildly
        self._tendency: float = 1.0

    def start_reaction(self, context_multiplier: float = 1.0) -> float:
        """Start a new reaction timer with temporally correlated variation."""
        # Slowly drift short-term tendency (random walk with mean reversion)
        self._tendency += random.gauss(0, 0.03)
        self._tendency += (1.0 - self._tendency) * 0.1
        self._tendency = max(0.7, min(1.3, self._tendency))

        base = random.gauss(self.mean_ms, self.std_ms) * self._tendency * context_multiplier
        delay_ms = max(self.min_ms, min(self.max_ms, base))
        self._pending_reaction = delay_ms / 1000.0
        self._reaction_start = time.perf_counter()
        return self._pending_reaction

    def is_ready(self) -> bool:
        """Check if the current reaction time has elapsed."""
        if self._pending_reaction is None:
            return True
        elapsed = time.perf_counter() - self._reaction_start
        if elapsed >= self._pending_reaction:
            self._pending_reaction = None
            return True
        return False

    def remaining(self) -> float:
        """Seconds remaining on current reaction, or 0 if ready."""
        if self._pending_reaction is None:
            return 0.0
        elapsed = time.perf_counter() - self._reaction_start
        return max(0.0, self._pending_reaction - elapsed)

    def reset(self) -> None:
        """Clear pending reaction."""
        self._pending_reaction = None


class ActionCooldown:
    """Simple cooldown timer for rate-limiting actions."""

    def __init__(self, cooldown_ms: float):
        self.cooldown_s = cooldown_ms / 1000.0
        self._last_action = 0.0

    def try_action(self) -> bool:
        """Returns True if cooldown has elapsed. Resets timer on success."""
        now = time.perf_counter()
        if now - self._last_action >= self.cooldown_s:
            self._last_action = now
            return True
        return False

    def reset(self) -> None:
        """Force reset the cooldown."""
        self._last_action = 0.0
