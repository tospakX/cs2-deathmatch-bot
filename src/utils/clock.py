"""Clock abstraction supporting real-time wall clocks and deterministic fake clocks.

Allows the behavioral simulation to run deterministically in tests and simulations
without depending on OS schedulers, CPU performance, or wall-clock timing.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Protocol for monotonic time providers."""

    def now(self) -> float:
        """Return current monotonic time in seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Pause or advance time by the specified duration in seconds."""
        ...


class RealClock:
    """Production wall-clock provider using time.perf_counter and time.sleep."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic simulated clock for tests and offline behavioral simulations.

    Advances time explicitly via ``advance()`` or automatically when ``sleep()`` is called.
    Never blocks the OS thread or consumes real wall-clock time.
    """

    def __init__(self, initial_time: float = 0.0, default_dt: float = 0.033):
        self._time: float = float(initial_time)
        self.default_dt: float = float(default_dt)

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> float:
        """Advance simulated time by ``seconds`` and return the new timestamp."""
        if seconds > 0:
            self._time += seconds
        return self._time

    def sleep(self, seconds: float) -> None:
        """Simulate sleep by advancing the clock without blocking."""
        self.advance(seconds)

    def step(self) -> float:
        """Advance by the default step interval."""
        return self.advance(self.default_dt)
