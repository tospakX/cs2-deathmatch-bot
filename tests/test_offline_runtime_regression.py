"""Regression tests for 1080p coordinates, screen center, angular wrapping, clock, and map handling."""

import math
import os
import random

import yaml

from src.behavior.executor import MotorExecutor
from src.behavior.motor import MotorPlanner
from src.behavior.personality import load_personality
from src.behavior.player_state import AimPhase, BotPhase, PlayerState, SpatialMemory
from src.movement.navigator import NavigationController, WaypointGraph
from src.utils.clock import FakeClock
from src.utils.math_helpers import screen_delta_to_mouse
from src.utils.validator import scale_regions, validate_environment


def test_1080p_settings_and_coordinates():
    """Verify settings.yaml assumptions are calibrated for 1920x1080 and within bounds."""
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Display resolution
    disp = cfg["display"]
    assert disp["width"] == 1920, f"Expected 1920 width, got {disp['width']}"
    assert disp["height"] == 1080, f"Expected 1080 height, got {disp['height']}"

    # 2. Screen center & FOV
    game = cfg["game"]
    assert game["crosshair_x"] == 960, f"Expected 960 crosshair_x, got {game['crosshair_x']}"
    assert game["crosshair_y"] == 540, f"Expected 540 crosshair_y, got {game['crosshair_y']}"
    assert game["fov_horizontal"] in (106, 106.0, 106.26)

    # 3. All HUD regions within 1920x1080 bounds
    regions = cfg["regions"]
    for name, (x, y, w, h) in regions.items():
        assert x >= 0, f"Region {name} x < 0"
        assert y >= 0, f"Region {name} y < 0"
        assert w > 0, f"Region {name} width <= 0"
        assert h > 0, f"Region {name} height <= 0"
        assert x + w <= 1920, f"Region {name} exceeds screen width: {x + w} > 1920"
        assert y + h <= 1080, f"Region {name} exceeds screen height: {y + h} > 1080"

    # 4. Minimap region within 1920x1080 bounds
    mm = cfg["minimap"]
    assert mm["x"] >= 0 and mm["y"] >= 0
    assert mm["x"] + mm["size"] <= 1920
    assert mm["y"] + mm["size"] <= 1080


def test_screen_center_calculation():
    """Verify screen-center calculations, zero delta, and symmetry at 1920x1080."""
    # Center pixel delta should result in exactly zero mouse counts
    dx, dy = screen_delta_to_mouse(0, 0, 0.85, 0.022, 0.022, 1920, 1080, 106.0)
    assert dx == 0 and dy == 0

    # Symmetric offsets produce exact opposite mouse counts
    dx_pos, dy_pos = screen_delta_to_mouse(150, 75, 0.85, 0.022, 0.022, 1920, 1080, 106.0)
    dx_neg, dy_neg = screen_delta_to_mouse(-150, -75, 0.85, 0.022, 0.022, 1920, 1080, 106.0)
    assert dx_pos == -dx_neg
    assert dy_pos == -dy_neg

    # Scale regions helper preserves relative alignment
    base_regions = {"box": [100, 200, 50, 40]}
    scaled = scale_regions(base_regions, (1920, 1080), (3840, 2160))
    assert scaled["box"] == [200, 400, 100, 80]


def test_yaw_wrapping_regression():
    """Verify SpatialMemory wraps yaw within [-180, 180] degrees across multi-turn rotations."""
    mem = SpatialMemory(
        yaw_offset_deg=10.0,
        pitch_offset_deg=5.0,
        last_seen_time=1.0,
        confidence=0.9,
    )

    # 1. Simple rotation
    mem.update_camera_rotation(30.0, 0.0)
    assert abs(mem.yaw_offset_deg - (-20.0)) < 1e-6

    # 2. Large rotation crossing 180 degrees
    mem.update_camera_rotation(180.0, 0.0)
    assert -180.0 <= mem.yaw_offset_deg <= 180.0
    assert abs(mem.yaw_offset_deg - 160.0) < 1e-6

    # 3. Full 360 degree spin
    prev_yaw = mem.yaw_offset_deg
    mem.update_camera_rotation(360.0, 0.0)
    assert abs(mem.yaw_offset_deg - prev_yaw) < 1e-6

    # 4. Multi-turn sweep test
    rng = random.Random(42)
    for _ in range(500):
        delta = rng.uniform(-1080.0, 1080.0)
        mem.update_camera_rotation(delta, 0.0)
        assert -180.0 <= mem.yaw_offset_deg <= 180.0, f"Yaw overflow: {mem.yaw_offset_deg}"


def test_pitch_clamping_regression():
    """Verify SpatialMemory clamps pitch strictly within [-89, 89] degrees."""
    mem = SpatialMemory(
        yaw_offset_deg=0.0,
        pitch_offset_deg=0.0,
        last_seen_time=1.0,
        confidence=0.9,
    )

    # Clamping at top
    mem.update_camera_rotation(0.0, 120.0)
    assert mem.pitch_offset_deg == -89.0

    # Clamping at bottom
    mem.update_camera_rotation(0.0, -300.0)
    assert mem.pitch_offset_deg == 89.0

    # Multi-turn pitch test
    rng = random.Random(1337)
    for _ in range(500):
        delta = rng.uniform(-500.0, 500.0)
        mem.update_camera_rotation(0.0, delta)
        assert -89.0 <= mem.pitch_offset_deg <= 89.0, f"Pitch out of bounds: {mem.pitch_offset_deg}"


def test_deterministic_clock_behavior():
    """Verify FakeClock advances deterministically and decouples behavior from wall-clock sleep."""
    clock = FakeClock(initial_time=50.0, default_dt=0.033)
    assert clock.now() == 50.0

    clock.advance(1.5)
    assert clock.now() == 51.5

    clock.sleep(0.5)
    assert clock.now() == 52.0

    state = PlayerState(clock=clock)
    assert state.now == 52.0
    state.begin_tick(now=52.033)
    assert abs(state.now - 52.033) < 1e-6
    assert abs(state.tick_dt - 0.033) < 1e-6

    # Clamping test: dt > 0.2s is clamped to 0.2s
    state.begin_tick(now=53.5)
    assert state.tick_dt == 0.2


def test_missing_map_and_config_handling():
    """Verify missing maps and configs fail safely without crashing."""
    graph = WaypointGraph()
    # Loading non-existent map should not crash
    graph.load("config/maps/definitely_missing_map_12345.json")
    assert len(graph.waypoints) == 0

    nav = NavigationController(graph)
    assert not nav.has_waypoints()

    # Empty graph update falls back safely
    cmd = nav.update((100, 100))
    assert cmd["has_route"] is False
    assert cmd["forward"] is False

    # Validator with missing config reports failure cleanly
    val = validate_environment(config_path="config/non_existent_config_xyz.yaml", quiet=True)
    assert not val.is_valid
    assert any("not found" in err.lower() for err in val.errors)


def test_long_simulation_stability():
    """Run 1000 simulated ticks with FakeClock to detect stuck states or numerical instability."""
    clock = FakeClock(initial_time=0.0)
    state = PlayerState(clock=clock)
    state.transition_phase(BotPhase.ROAMING)
    personality = load_personality("average")

    planner = MotorPlanner(
        screen_center=(960, 540),
        sensitivity=0.85,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=106.0,
        screen_width=1920,
        screen_height=1080,
        personality=personality,
    )

    class MockMouse:
        def __init__(self):
            self.dx, self.dy = 0, 0

        def move_relative(self, dx, dy):
            self.dx += dx
            self.dy += dy

        def mouse_down(self, btn):
            pass

        def mouse_up(self, btn):
            pass

    class MockKeyboard:
        def __init__(self):
            self.keys = set()

        def key_down(self, k):
            self.keys.add(k)

        def key_up(self, k):
            self.keys.discard(k)

        def hold_key(self, k):
            self.keys.add(k)

        def release_key(self, k):
            self.keys.discard(k)

        def release_all(self):
            self.keys.clear()

    from src.behavior.decision import DecisionEngine

    engine = DecisionEngine(
        personality=personality,
        screen_center=(960, 540),
        weapon="default",
        sensitivity=0.85,
        m_yaw=0.022,
        m_pitch=0.022,
        fov_h=106.0,
        screen_width=1920,
        screen_height=1080,
    )

    executor = MotorExecutor(
        player_state=state,
        motor_planner=planner,
        firing_controller=engine.firing,
        movement_controller=engine.movement,
        mouse_backend=MockMouse(),
        keyboard_backend=MockKeyboard(),
        keybinds={"forward": "w", "back": "s", "left": "a", "right": "d", "crouch": "ctrl"},
        clock=clock,
    )

    for i in range(1000):
        clock.advance(0.033)
        state.begin_tick()

        # Alternate phases
        if i % 250 == 0 and i > 0:
            state.transition_phase(BotPhase.ROAMING)
            state.aim_phase = AimPhase.IDLE

        dec = engine.decide(state, [])
        assert dec is not None
        assert not math.isnan(dec.scan_delta[0])
        assert not math.isnan(dec.scan_delta[1])

        # Step executor
        executor.run_substeps_until(clock.now() + 0.010)

    assert state.frame_count == 1000
