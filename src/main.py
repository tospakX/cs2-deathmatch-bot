"""Main loop orchestrator for the CS2 Deathmatch Bot.

Runs the authoritative behavioral architecture:
  Screen Capture -> YOLO / Confirmation Filter -> HUD Reader
  -> PlayerState -> PerceptionSystem -> DecisionEngine
  -> MotorPlanner -> FiringController -> MovementController
  -> SendInput (Mouse & Keyboard)
"""

import ctypes
import os
import random
import sys
import threading
import time
from collections import deque

import yaml

# Hotkeys read globally via GetAsyncKeyState, so they work while CS2 has focus.
# END  -> kill the bot instantly and release all input.
# HOME -> toggle pause: the bot lets go of mouse/keyboard so you can take over,
#         press again to hand control back. The process keeps running.
PANIC_VK = 0x23
PANIC_KEY_NAME = "END"
PAUSE_VK = 0x24
PAUSE_KEY_NAME = "HOME"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.behavior.decision import DecisionEngine
from src.behavior.motor import MotorPlanner
from src.behavior.perception import PerceptionSystem
from src.behavior.personality import load_personality
from src.behavior.player_state import BotPhase, PlayerState
from src.brain.state_machine import BotState, StateMachine
from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.movement.bc_policy import BCMovementPolicy
from src.movement.explorer import WallFollower
from src.movement.navigator import NavigationController, WaypointGraph
from src.movement.stuck_detector import StuckDetector
from src.utils.debug_overlay import DebugOverlay
from src.utils.session_logger import SessionLogger
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector
from src.vision.hud_reader import HUDReader
from src.vision.minimap import MinimapReader


def _round(value, ndigits: int = 1):
    """Round floats for compact logging; pass through None/other types."""
    return round(value, ndigits) if isinstance(value, (int, float)) else value


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load main configuration."""
    config_path = os.path.join(PROJECT_ROOT, path)
    with open(config_path) as f:
        return yaml.safe_load(f)


class Bot:
    """Main bot orchestrator running the authoritative behavioral pipeline."""

    def __init__(self, personality_name: str | None = None):
        self.config = load_config()
        self.running = False

        # Personality
        pname = personality_name or self.config["bot"]["default_personality"]
        self.personality = load_personality(
            pname, os.path.join(PROJECT_ROOT, "config", "personalities")
        )
        print(f"[Bot] Loaded personality: {self.personality}")

        # Screen capture
        display = self.config["display"]
        self.capture = ScreenCapture(
            monitor=display["monitor"],
            target_fps=display["capture_fps"],
        )

        # Vision
        det_cfg = self.config["detection"]
        self.detector = YOLODetector(
            model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
            input_size=det_cfg["input_size"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            classes=det_cfg["classes"],
        )
        conf_cfg = self.config.get("confirmation", {})
        self.confirmation_filter = ConfirmationFilter(
            min_confirm_frames=conf_cfg.get("min_confirm_frames", 3),
            max_missing_frames=conf_cfg.get("max_missing_frames", 2),
            match_distance=conf_cfg.get("match_distance", 150.0),
            match_size_ratio=conf_cfg.get("match_size_ratio", 0.4),
        )
        self.hud_reader = HUDReader(self.config["regions"])
        mm = self.config["minimap"]
        self.minimap_reader = MinimapReader(
            mm["x"],
            mm["y"],
            mm["size"],
            sat_min=mm.get("sat_min", 150),
            val_min=mm.get("val_min", 190),
        )

        # Single Authoritative Behavioral Core
        game = self.config["game"]
        self.player_state = PlayerState()
        self.perception = PerceptionSystem(
            screen_center=(game["crosshair_x"], game["crosshair_y"]),
            screen_size=(display["width"], display["height"]),
            fov_h=game.get("fov_horizontal", 122.0),
        )
        self.decision_engine = DecisionEngine(
            personality=self.personality,
            screen_center=(game["crosshair_x"], game["crosshair_y"]),
            weapon=game.get("weapon", "default"),
            sensitivity=game["sensitivity"],
            m_yaw=game["m_yaw"],
            m_pitch=game["m_pitch"],
            fov_h=game.get("fov_horizontal", 122.0),
            screen_width=display["width"],
            screen_height=display["height"],
        )
        self.motor_planner = MotorPlanner(
            screen_center=(game["crosshair_x"], game["crosshair_y"]),
            sensitivity=game["sensitivity"],
            m_yaw=game["m_yaw"],
            m_pitch=game["m_pitch"],
            fov_h=game.get("fov_horizontal", 122.0),
            screen_width=display["width"],
            screen_height=display["height"],
            personality=self.personality,
        )

        # State machine bridge for debug overlay & metrics
        self.state_machine = StateMachine()

        # Movement navigation
        nav_cfg = self.config["navigation"]
        self.explorer = WallFollower(
            wall_threshold=nav_cfg["wall_avoid_threshold"],
        )
        self.nav_graph = WaypointGraph()
        self.nav_controller = NavigationController(
            self.nav_graph,
            reach_dist=nav_cfg["waypoint_reach_dist"],
            turn_gain=nav_cfg.get("nav_turn_gain", 1.6),
            max_turn=nav_cfg.get("nav_max_turn", 22),
            invert_turn=nav_cfg.get("nav_invert_turn", False),
        )

        # Movement mode: "waypoint" (hand-written nav) or "bc" (learned policy).
        self.movement_mode = self.config["bot"].get("movement_mode", "waypoint")
        self.bc_policy = None
        if self.movement_mode == "bc":
            bc = self.config.get("bc_movement", {})
            self.bc_policy = BCMovementPolicy(
                os.path.join(PROJECT_ROOT, bc.get("model_path", "models/movement.onnx")),
                mild_turn=bc.get("mild_turn", 150),
                hard_turn=bc.get("hard_turn", 400),
                invert_turn=bc.get("invert_turn", False),
            )
        self.stuck_detector = StuckDetector(
            timeout=nav_cfg["stuck_timeout"],
        )

        # Debug overlay
        self.debug = None
        if self.config["bot"]["debug_overlay"]:
            self.debug = DebugOverlay(scale=0.5)

        # Session logging for after-the-fact diagnosis of a run.
        self.logger = SessionLogger(
            PROJECT_ROOT,
            enabled=self.config["bot"].get("debug_log", False),
        )
        self._last_nav_cmd: dict = {}

        # Runtime State
        self._is_firing = False
        self._movement_keys_held: set[str] = set()
        self._tick_count = 0
        self.paused = False
        self._is_stuck = False
        self._nudge_ticks = 0  # unstick: commit to a turn sweep when jammed
        self._nudge_dir = -1
        self._bc_pos_hist: deque = deque(maxlen=30)  # ~1.5s of minimap positions
        self._turn_state = 0.0  # low-pass state for smooth camera turning

    def start(self) -> None:
        """Initialize all systems and start the main loop."""
        print("[Bot] Starting...")

        # Start screen capture
        backend = self.capture.start()
        print(f"[Bot] Screen capture: {backend}")

        # Load YOLO model
        try:
            self.detector.load()
        except Exception as e:
            print(f"[Bot] WARNING: Detector failed to load: {e}")
            print("[Bot] Running without detection (debug/movement only)")

        # Load the learned movement policy, or fall back to waypoint nav.
        if self.bc_policy is not None:
            try:
                self.bc_policy.load()
                print("[Bot] Movement: behavioural-cloning policy")
            except Exception as e:
                print(f"[Bot] WARNING: BC policy failed to load ({e}); using waypoint nav")
                self.bc_policy = None

        # Load waypoints if available
        map_name = "dust2"  # TODO: auto-detect map
        wp_path = os.path.join(PROJECT_ROOT, "config", "maps", f"{map_name}.json")
        if os.path.exists(wp_path):
            self.nav_graph.load(wp_path)
            print(f"[Bot] Loaded waypoints for {map_name}")

        self.running = True
        self._loop_start = time.perf_counter()
        self._max_run_seconds = self.config["bot"].get("max_run_seconds", 120)

        # Dedicated watchdog thread polls the panic key every 20ms
        threading.Thread(target=self._panic_watchdog, daemon=True).start()
        print("[Bot] Ready.")
        print(
            f"[Bot] *** {PANIC_KEY_NAME} = STOP instantly | {PAUSE_KEY_NAME} = "
            f"pause/resume (take control) -- both work while CS2 is focused. ***"
        )
        if self._max_run_seconds:
            print(f"[Bot] Auto-stops after {self._max_run_seconds}s as a failsafe.")
        print(f"[Bot] Tick rate: {self.config['bot']['tick_rate']} Hz")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[Bot] Stopped by user.")
        finally:
            self.stop()

    def _release_input(self) -> None:
        """Let go of mouse and keyboard (used by pause and panic)."""
        for cleanup in (self._stop_firing, self._release_all_movement, keyboard.release_all):
            try:
                cleanup()
            except Exception:
                pass

    def _panic_watchdog(self) -> None:
        """Handle the global hotkeys: END = kill, HOME = pause/resume toggle."""
        user32 = ctypes.windll.user32
        prev_pause_down = False
        while self.running:
            if user32.GetAsyncKeyState(PANIC_VK) & 0x8000:
                print(f"\n[Bot] PANIC ({PANIC_KEY_NAME}) -- force stopping NOW.")
                self.running = False
                self._release_input()
                try:
                    self.logger.close()
                except Exception:
                    pass
                os._exit(0)

            # Edge-detect HOME so a held key toggles once, not every poll.
            pause_down = bool(user32.GetAsyncKeyState(PAUSE_VK) & 0x8000)
            if pause_down and not prev_pause_down:
                self.paused = not self.paused
                if self.paused:
                    self._release_input()
                    print(
                        f"\n[Bot] PAUSED -- input released. {PAUSE_KEY_NAME} "
                        f"to resume, {PANIC_KEY_NAME} to quit."
                    )
                else:
                    print("\n[Bot] RESUMED.")
            prev_pause_down = pause_down

            time.sleep(0.02)

    def _should_abort(self) -> bool:
        """True if the panic key is down or the run time limit was hit."""
        if ctypes.windll.user32.GetAsyncKeyState(PANIC_VK) & 0x8000:
            print(f"\n[Bot] PANIC key ({PANIC_KEY_NAME}) pressed -- stopping.")
            return True
        if self._max_run_seconds and (
            time.perf_counter() - self._loop_start > self._max_run_seconds
        ):
            print(f"\n[Bot] Run time limit ({self._max_run_seconds}s) reached -- stopping.")
            return True
        return False

    def _main_loop(self) -> None:
        """Main 30 Hz game loop."""
        tick_interval = 1.0 / self.config["bot"]["tick_rate"]

        while self.running:
            tick_start = time.perf_counter()
            self._tick_count += 1

            # 0. Panic / time-limit check BEFORE doing anything else this tick.
            if self._should_abort():
                self.running = False
                break

            # 0b. Paused (HOME): hold off all input/decisions, keep the loop alive.
            if self.paused:
                time.sleep(0.05)
                continue

            # 1. Capture frame
            frame = self.capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            # 2. Run detection + confirmation filter
            raw_detections = self.detector.detect(frame)
            detections = self.confirmation_filter.update(raw_detections)

            # 3. Read HUD
            hud = self.hud_reader.read(frame)

            # 4. Update persistent player state
            self.player_state.begin_tick()
            self.player_state.health = hud.health
            self.player_state.ammo_clip = hud.ammo_clip
            self.player_state.is_alive = hud.is_alive

            # Check stuck
            is_moving = len(self._movement_keys_held) > 0
            self._is_stuck = self.stuck_detector.update(frame, is_moving)

            # 5. Perception: track detected enemies across frames with persistent identity
            self.perception.update(self.player_state, detections)
            visible_targets = self.perception.get_visible_targets(self.player_state)
            enemies = (
                [t.detection for t in visible_targets]
                if visible_targets
                else [d for d in detections if not d.is_head]
            )

            # 6. Authoritative Decision Engine: attention, reaction, firing, movement, adaptation
            decision = self.decision_engine.decide(self.player_state, visible_targets)

            # Synchronize state_machine for debug overlay & session logging
            phase_to_state = {
                BotPhase.DEAD: BotState.DEAD,
                BotPhase.ROAMING: BotState.ROAMING,
                BotPhase.SCANNING: BotState.SEARCHING,
                BotPhase.ENGAGING: BotState.FIGHTING,
                BotPhase.TRACKING: BotState.FIGHTING,
                BotPhase.RETREATING: BotState.RETREATING,
                BotPhase.SPAWNING: BotState.ROAMING,
                BotPhase.RELOADING: BotState.ROAMING,
            }
            mapped_state = phase_to_state.get(self.player_state.phase, BotState.ROAMING)
            self.state_machine.state = mapped_state
            action_name = decision.action_type

            # 7. Execute planned behavior (non-blocking motor + shot-based recoil + movement)
            self._execute_behavior(decision, frame)

            # 8. Session logging (cheap; throttled frame saves)
            if self.logger.enabled:
                nav = self._last_nav_cmd
                self.logger.log_tick(
                    state=self.state_machine.state.name,
                    action=action_name,
                    enemies=len(enemies),
                    health=hud.health,
                    ammo=hud.ammo_clip,
                    alive=hud.is_alive,
                    stuck=self._is_stuck,
                    nav_heading=_round(nav.get("heading_deg")),
                    nav_yaw_err=_round(nav.get("yaw_error_deg")),
                    nav_turn=nav.get("turn_x"),
                    nav_route=nav.get("has_route"),
                    bc_turn=nav.get("turn_class"),
                    bc_keys=nav.get("key_probs"),
                    bc_stuck=nav.get("bc_stuck"),
                    bc_move=[nav.get("forward"), nav.get("left"), nav.get("back"), nav.get("right")]
                    if "key_probs" in nav
                    else None,
                    infer_ms=round(self.detector.inference_ms, 1),
                )
                self.logger.maybe_save_frame(
                    frame,
                    label=f"{self.state_machine.state.name} | {action_name} | "
                    f"en={len(enemies)} hp={hud.health}",
                )

            # 9. Debug overlay
            if self.debug:
                vis = self.debug.draw(
                    frame,
                    detections,
                    self.state_machine.state,
                    hud_info=str(hud),
                    inference_ms=self.detector.inference_ms,
                    extra_lines=[
                        f"Action: {action_name}",
                        f"Raw: {len(raw_detections)} | Confirmed: {len(detections)}",
                        f"Stuck recoveries: {self.stuck_detector.recovery_count}",
                    ],
                )
                if not self.debug.show(vis):
                    self.running = False

            # 10. Sleep remainder of tick
            elapsed = time.perf_counter() - tick_start
            sleep_time = tick_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _execute_behavior(self, decision, frame) -> None:
        """Execute the planned actions from the authoritative behavioral engine."""
        keybinds = self.config["keybinds"]

        # 1. Non-blocking continuous motor planning (mouse aiming)
        mouse_dx, mouse_dy = (0, 0)
        if decision.target is not None:
            mouse_dx, mouse_dy = self.motor_planner.plan(self.player_state)
        elif decision.scan_delta != (0.0, 0.0):
            mouse_dx, mouse_dy = self.motor_planner.plan_scan(
                decision.scan_delta[0], decision.scan_delta[1], self.player_state
            )

        # 2. Shot-based recoil compensation
        if decision.firing_cmd.recoil_dx != 0 or decision.firing_cmd.recoil_dy != 0:
            mouse_dx += decision.firing_cmd.recoil_dx
            mouse_dy += decision.firing_cmd.recoil_dy

        # Apply continuous mouse movement (SendInput, non-blocking)
        if mouse_dx != 0 or mouse_dy != 0:
            mouse.move_relative(mouse_dx, mouse_dy)

        # 3. Firing execution (synchronized with trigger events)
        if decision.firing_cmd.trigger_action == "press":
            mouse.mouse_down("left")
            self._is_firing = True
        elif decision.firing_cmd.trigger_action == "release":
            self._stop_firing()
        elif decision.firing_cmd.trigger_action == "hold" and not self._is_firing:
            mouse.mouse_down("left")
            self._is_firing = True

        # 4. Movement & Reload execution
        if decision.should_reload:
            self._stop_firing()
            keyboard.key_press(keybinds["reload"])
        elif decision.phase == BotPhase.RELOADING:
            self._stop_firing()
        elif decision.phase in (BotPhase.ENGAGING, BotPhase.RETREATING):
            self._apply_combat_movement(decision.movement_keys, keybinds)
        elif decision.phase in (BotPhase.ROAMING, BotPhase.SCANNING):
            self._stop_firing()
            self._roam(frame, keybinds)

        # 5. Idle quirks
        if decision.idle_action == "inspect":
            keyboard.key_press(keybinds["inspect"])

    def _apply_combat_movement(self, move_keys: dict[str, bool], keybinds: dict) -> None:
        """Apply combat movement keys based on stateful decision output."""
        for action_name in ("forward", "back", "left", "right", "crouch", "walk"):
            bind_key = keybinds.get(action_name)
            if not bind_key:
                continue
            should_hold = move_keys.get(action_name, False)
            is_held = bind_key in self._movement_keys_held
            if should_hold and not is_held:
                keyboard.hold_key(bind_key)
                self._movement_keys_held.add(bind_key)
            elif not should_hold and is_held:
                keyboard.release_key(bind_key)
                self._movement_keys_held.discard(bind_key)

    def _stop_firing(self) -> None:
        """Stop firing if currently firing."""
        if self._is_firing:
            mouse.mouse_up("left")
            self._is_firing = False

    def _roam(self, frame, keybinds: dict) -> None:
        """Roaming movement with a unified minimap anti-stick for any driver."""
        pos, _ = self.minimap_reader.read(frame)
        self._bc_pos_hist.append(pos)

        # Position-stuck: minimap dot frozen while we believe we're moving.
        pos_stuck = False
        if len(self._bc_pos_hist) == self._bc_pos_hist.maxlen:
            ox, oy = self._bc_pos_hist[0]
            pos_stuck = ((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2) ** 0.5 < 3.0
        if self._nudge_ticks <= 0 and pos_stuck:
            self._nudge_ticks = 20
            self._nudge_dir = random.choice([-1, 1])

        # Pick the driver: unstick sweep > BC policy > waypoint nav > wall-follow.
        if self._nudge_ticks > 0:
            self._nudge_ticks -= 1
            hard = self.config.get("bc_movement", {}).get("hard_turn", 400)
            cmd = {"forward": True, "turn_x": self._nudge_dir * hard}
        elif self.bc_policy is not None:
            cmd = self.bc_policy.act(frame)
        elif self.nav_controller.has_waypoints():
            cmd = self.nav_controller.update(pos)
            if not cmd.get("has_route"):
                cmd = self.explorer.get_movement(frame)
        else:
            cmd = self.explorer.get_movement(frame)

        cmd["bc_stuck"] = pos_stuck
        self._last_nav_cmd = cmd
        self._apply_move(cmd, keybinds)

    def _apply_move(self, cmd: dict, keybinds: dict) -> None:
        """Execute a movement command with stateful key holding and smoothed view turn."""
        for flag in ("forward", "back", "left", "right", "crouch"):
            bind_key = keybinds.get(flag)
            if not bind_key:
                continue
            should_hold = bool(cmd.get(flag, False))
            is_held = bind_key in self._movement_keys_held
            if should_hold and not is_held:
                keyboard.hold_key(bind_key)
                self._movement_keys_held.add(bind_key)
            elif not should_hold and is_held:
                keyboard.release_key(bind_key)
                self._movement_keys_held.discard(bind_key)

        turn = self._smooth_turn(int(cmd.get("turn_x", 0)))
        if turn != 0:
            mouse.move_relative(turn, 0)

    def _smooth_turn(self, target: int) -> int:
        """Low-pass the view turn so the camera eases instead of jerking each tick."""
        self._turn_state += (target - self._turn_state) * 0.4
        return int(round(self._turn_state))

    def _release_all_movement(self) -> None:
        """Release all held movement keys."""
        for key in list(self._movement_keys_held):
            keyboard.release_key(key)
        self._movement_keys_held.clear()

    def stop(self) -> None:
        """Clean shutdown."""
        print("[Bot] Shutting down...")
        self._stop_firing()
        self.player_state.reset_spray()
        self._release_all_movement()
        keyboard.release_all()
        self.capture.stop()
        if self.debug:
            self.debug.cleanup()
        self.logger.close()
        print("[Bot] Stopped.")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CS2 Deathmatch Bot")
    parser.add_argument(
        "--personality",
        "-p",
        type=str,
        default=None,
        help="Personality profile (noob, average, tryhard)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlay")
    args = parser.parse_args()

    bot = Bot(personality_name=args.personality)

    if args.no_debug:
        bot.debug = None

    bot.start()


if __name__ == "__main__":
    main()
