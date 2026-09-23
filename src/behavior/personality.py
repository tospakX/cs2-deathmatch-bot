"""Rich personality system that actually influences multiple connected behaviors.

A personality represents persistent tendencies of a simulated player.  Every
trait maps to one or more runtime behaviors through the decision engine,
motor planner, attention model, and firing controller.  Traits are *not*
simple probability overrides — they shape base parameters that the
contextual systems then modulate.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import yaml


@dataclass
class PersonalityTraits:
    """Persistent behavioral tendencies that shape multiple subsystems.

    Every field here MUST be consumed by at least one runtime system.
    Dead fields are bugs.
    """

    name: str = "unnamed"
    description: str = ""

    # ── Reaction tendency ────────────────────────────────────────────────
    # Base reaction speed — the attention and reaction systems add context.
    reaction_base_ms: float = 250.0
    reaction_variance_ms: float = 40.0  # within-session consistency
    reaction_min_ms: float = 140.0
    reaction_max_ms: float = 500.0

    # ── Aim / motor tendencies ───────────────────────────────────────────
    motor_speed: float = 1.0  # 0.5=sluggish, 1.0=normal, 1.5=fast
    motor_precision: float = 0.6  # 0=sloppy, 1=precise
    tracking_steadiness: float = 0.5  # how smoothly tracking is maintained
    correction_tendency: float = 0.5  # tendency to make small corrections
    overshoot_tendency: float = 0.3  # base likelihood of overshooting
    head_aim_preference: float = 0.3  # how often to prefer head aim point

    # ── Aggression / combat style ────────────────────────────────────────
    aggression: float = 0.5  # 0=passive, 1=very aggressive
    confidence_base: float = 0.5  # starting confidence level
    risk_tolerance: float = 0.5  # willingness to stay in dangerous fights
    firing_discipline: float = 0.5  # 0=spray everything, 1=careful taps
    preferred_engagement_distance: float = 400.0  # pixels; shapes fire mode
    recoil_skill: float = 0.5  # 0=no compensation, 1=very good

    # ── Target handling ──────────────────────────────────────────────────
    target_persistence: float = 0.6  # how strongly to stick with current target
    attention_breadth: float = 0.5  # 0=tunnel vision, 1=wide awareness
    switch_reluctance: float = 0.5  # cost of switching targets

    # ── Movement style ───────────────────────────────────────────────────
    movement_aggression: float = 0.5  # push vs hold vs retreat tendency
    strafe_tendency: float = 0.5  # tendency to strafe during combat
    crouch_tendency: float = 0.4  # tendency to crouch spray
    repositioning_tendency: float = 0.3  # tendency to move to better position
    disengage_health: int = 30  # health threshold for retreat

    # ── Scanning / awareness ─────────────────────────────────────────────
    scanning_frequency: float = 0.5  # how often to look around
    scanning_amplitude: float = 0.5  # how far to look when scanning
    hesitation_tendency: float = 0.3  # tendency to hesitate before acting

    # ── Idle quirks ──────────────────────────────────────────────────────
    inspect_tendency: float = 0.01  # per-tick chance of weapon inspect
    fidget_tendency: float = 0.03  # per-tick chance of idle look

    def sample_reaction_time(self, context_modifier: float = 1.0) -> float:
        """Sample a base reaction time in seconds, modified by context.

        context_modifier < 1.0 = faster (expected target),
        context_modifier > 1.0 = slower (surprise).
        """
        base = random.gauss(self.reaction_base_ms, self.reaction_variance_ms)
        base *= context_modifier
        clamped = max(self.reaction_min_ms, min(self.reaction_max_ms, base))
        return clamped / 1000.0

    def sample_motor_duration(self, pixel_distance: float) -> float:
        """Fitts's-law-inspired movement duration in seconds."""
        if pixel_distance < 2:
            return 0.0
        # Base: 30ms + 40ms * log2(1 + dist/50)
        base_ms = 30.0 + 40.0 * math.log2(1.0 + pixel_distance / 50.0)
        # Speed modifier
        base_ms /= self.motor_speed
        # Human variance (±15%)
        base_ms *= random.uniform(0.85, 1.15)
        return max(0.016, min(0.300, base_ms / 1000.0))

    # ── Legacy compatibility properties ──────────────────────────────────
    @property
    def reaction_mean_ms(self) -> float:
        return self.reaction_base_ms

    @property
    def reaction_std_ms(self) -> float:
        return self.reaction_variance_ms

    @property
    def aim_speed(self) -> float:
        return self.motor_speed * 6.0

    @property
    def overshoot_chance(self) -> float:
        return self.overshoot_tendency

    @property
    def overshoot_magnitude(self) -> float:
        return 1.0 + (1.0 - self.motor_precision) * 0.8

    @property
    def head_aim_chance(self) -> float:
        return self.head_aim_preference

    @property
    def tracking_error(self) -> float:
        return max(1.0, (1.0 - self.motor_precision) * 25.0)

    @property
    def recoil_compensation(self) -> float:
        return self.recoil_skill

    @property
    def tap_chance(self) -> float:
        return self.firing_discipline

    @property
    def burst_length(self) -> list[int]:
        return [2, 5]

    @property
    def strafe_while_shooting(self) -> bool:
        return self.strafe_tendency > 0.3

    @property
    def crouch_spray_chance(self) -> float:
        return self.crouch_tendency

    @property
    def inspect_chance(self) -> float:
        return self.inspect_tendency

    @property
    def look_around_chance(self) -> float:
        return self.fidget_tendency

    @property
    def engage_distance(self) -> float:
        return self.preferred_engagement_distance


Personality = PersonalityTraits


def load_personality(name: str, config_dir: str = "config/personalities") -> PersonalityTraits:
    """Load a personality from YAML, mapping old and new formats."""
    path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Personality profile not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Build traits from YAML, supporting both old and new format
    traits = PersonalityTraits()
    traits.name = data.get("name", name)
    traits.description = data.get("description", "")

    # Reaction
    r = data.get("reaction", {})
    traits.reaction_base_ms = r.get("mean_ms", r.get("base_ms", 250.0))
    traits.reaction_variance_ms = r.get("std_ms", r.get("variance_ms", 40.0))
    traits.reaction_min_ms = r.get("min_ms", 140.0)
    traits.reaction_max_ms = r.get("max_ms", 500.0)

    # Aim / motor
    a = data.get("aim", {})
    # Map old base_speed (3-10 scale) to motor_speed (0.5-1.5 scale)
    old_speed = a.get("base_speed", None)
    if old_speed is not None:
        traits.motor_speed = max(0.3, min(2.0, old_speed / 6.0))
    else:
        traits.motor_speed = a.get("motor_speed", 1.0)

    traits.motor_precision = a.get("motor_precision", 1.0 - a.get("tracking_error", 8.0) / 30.0)
    traits.tracking_steadiness = a.get("tracking_steadiness", traits.motor_precision * 0.8)
    traits.correction_tendency = a.get("correction_tendency", 0.5)
    traits.overshoot_tendency = a.get("overshoot_chance", a.get("overshoot_tendency", 0.3))
    traits.head_aim_preference = a.get("head_aim_chance", a.get("head_aim_preference", 0.3))

    # Spray / firing
    s = data.get("spray", {})
    traits.firing_discipline = s.get("firing_discipline", s.get("tap_chance", 0.3))
    traits.recoil_skill = s.get("recoil_compensation", s.get("recoil_skill", 0.5))

    # Combat
    c = data.get("combat", {})
    traits.aggression = c.get("aggression", 0.5)
    traits.confidence_base = c.get("confidence_base", 0.5)
    traits.risk_tolerance = c.get("risk_tolerance", 0.5)
    traits.preferred_engagement_distance = c.get(
        "engage_distance", c.get("preferred_engagement_distance", 400.0)
    )
    traits.target_persistence = c.get("target_persistence", 0.6)
    traits.attention_breadth = c.get("attention_breadth", 0.5)
    traits.switch_reluctance = c.get("switch_reluctance", 0.5)
    traits.disengage_health = c.get("disengage_health", 30)

    # Movement
    m = data.get("movement", {})
    traits.movement_aggression = m.get("movement_aggression", 0.5)
    traits.strafe_tendency = m.get(
        "strafe_tendency", 0.5 if m.get("strafe_while_shooting", True) else 0.1
    )
    traits.crouch_tendency = m.get("crouch_spray_chance", m.get("crouch_tendency", 0.4))
    traits.repositioning_tendency = m.get("repositioning_tendency", 0.3)

    # Scanning
    traits.scanning_frequency = data.get("scanning", {}).get("frequency", 0.5)
    traits.scanning_amplitude = data.get("scanning", {}).get("amplitude", 0.5)
    traits.hesitation_tendency = data.get("scanning", {}).get("hesitation_tendency", 0.3)

    # Idle
    i = data.get("idle", {})
    traits.inspect_tendency = i.get("inspect_chance", i.get("inspect_tendency", 0.01))
    traits.fidget_tendency = i.get("look_around_chance", i.get("fidget_tendency", 0.03))

    return traits
