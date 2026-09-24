"""Environment and configuration validation for CS2 Deathmatch Bot.

Validates dependencies, screen resolution, models, waypoints, and HUD configurations
to ensure smooth offline match execution and provide actionable diagnostics.
"""

from __future__ import annotations

import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import detect_screen_resolution


class ValidationResult:
    """Stores status and messages of pre-flight environment checks."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.detected_resolution: tuple[int, int] = (1920, 1080)
        self.config: dict[str, Any] = {}

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def scale_regions(
    regions: dict[str, list[int]],
    src_res: tuple[int, int],
    dst_res: tuple[int, int],
) -> dict[str, list[int]]:
    """Scale HUD bounding boxes proportionally between resolutions."""
    src_w, src_h = src_res
    dst_w, dst_h = dst_res
    if (src_w, src_h) == (dst_w, dst_h) or src_w <= 0 or src_h <= 0:
        return regions

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    scaled: dict[str, list[int]] = {}
    for name, bbox in regions.items():
        if len(bbox) == 4:
            x, y, w, h = bbox
            scaled[name] = [
                int(round(x * sx)),
                int(round(y * sy)),
                int(round(w * sx)),
                int(round(h * sy)),
            ]
        else:
            scaled[name] = bbox
    return scaled


def validate_environment(
    config_path: str = "config/settings.yaml",
    auto_adapt_resolution: bool = True,
    quiet: bool = False,
) -> ValidationResult:
    """Run comprehensive checks on dependencies, configs, models, and screen parameters."""
    result = ValidationResult()

    # 1. Check Python version
    py_major, py_minor = sys.version_info[:2]
    if py_major < 3 or (py_major == 3 and py_minor < 10):
        result.errors.append(f"Python 3.10+ required. Current version is {sys.version.split()[0]}")
    else:
        result.info.append(f"Python version: {sys.version.split()[0]} (OK)")

    # 2. Check dependencies
    required_packages = [
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("onnxruntime", "onnxruntime"),
    ]
    for mod_name, pkg_name in required_packages:
        try:
            __import__(mod_name)
            result.info.append(f"Dependency '{pkg_name}': installed")
        except ImportError:
            result.errors.append(
                f"Missing required package '{pkg_name}'. Run: pip install {pkg_name}"
            )

    # Check screen capture backend
    has_dxcam = False
    has_mss = False
    try:
        import dxcam  # noqa: F401

        has_dxcam = True
    except ImportError:
        pass
    try:
        import mss  # noqa: F401

        has_mss = True
    except ImportError:
        pass

    if has_dxcam:
        result.info.append("Screen capture backend: DXcam (primary) + mss (fallback)")
    elif has_mss:
        result.info.append("Screen capture backend: mss (fallback, DXcam not installed)")
    else:
        result.errors.append(
            "No screen capture backend found. Install DXcam (pip install dxcam) "
            "or mss (pip install mss)"
        )

    # 3. Load & validate configuration file
    full_cfg_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(full_cfg_path):
        result.errors.append(f"Configuration file not found: {full_cfg_path}")
        return result

    try:
        import yaml

        with open(full_cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        result.config = cfg
    except Exception as e:
        result.errors.append(f"Failed to parse {config_path}: {e}")
        return result

    # 4. Check display & resolution
    disp = cfg.get("display", {})
    cfg_w = disp.get("width", 1920)
    cfg_h = disp.get("height", 1080)
    monitor_idx = disp.get("monitor", 0)

    # Detect active monitor resolution
    det_w, det_h = detect_screen_resolution(monitor_idx)
    result.detected_resolution = (det_w, det_h)

    if (det_w, det_h) != (cfg_w, cfg_h):
        result.warnings.append(
            f"Configured resolution ({cfg_w}x{cfg_h}) differs from "
            f"detected display ({det_w}x{det_h})."
        )
        if auto_adapt_resolution:
            result.info.append(
                f"Auto-adapting resolution to match detected display: {det_w}x{det_h}"
            )
            # Update in-memory config for runtime
            cfg["display"]["width"] = det_w
            cfg["display"]["height"] = det_h
            cfg["game"]["crosshair_x"] = det_w // 2
            cfg["game"]["crosshair_y"] = det_h // 2
            # Scale regions
            if "regions" in cfg:
                cfg["regions"] = scale_regions(cfg["regions"], (cfg_w, cfg_h), (det_w, det_h))
            # Scale minimap
            if "minimap" in cfg:
                mm = cfg["minimap"]
                sx = det_w / float(cfg_w)
                sy = det_h / float(cfg_h)
                s = min(sx, sy)
                mm["x"] = int(round(mm["x"] * sx))
                mm["y"] = int(round(mm["y"] * sy))
                mm["size"] = int(round(mm["size"] * s))
            cfg_w, cfg_h = det_w, det_h
    else:
        result.info.append(f"Display resolution: {cfg_w}x{cfg_h} (matches detected display)")

    # Validate crosshair
    cx = cfg.get("game", {}).get("crosshair_x", cfg_w // 2)
    cy = cfg.get("game", {}).get("crosshair_y", cfg_h // 2)
    if not (0 <= cx <= cfg_w and 0 <= cy <= cfg_h):
        result.warnings.append(
            f"Crosshair center ({cx}, {cy}) is outside screen bounds ({cfg_w}x{cfg_h}). "
            f"Resetting to center ({cfg_w // 2}, {cfg_h // 2})."
        )
        cfg["game"]["crosshair_x"] = cfg_w // 2
        cfg["game"]["crosshair_y"] = cfg_h // 2

    # Validate regions bounds
    regions = cfg.get("regions", {})
    for rname, rbox in regions.items():
        if len(rbox) == 4:
            rx, ry, rw, rh = rbox
            if rx < 0 or ry < 0 or (rx + rw) > cfg_w or (ry + rh) > cfg_h:
                result.warnings.append(
                    f"HUD region '{rname}' [{rx}, {ry}, {rw}, {rh}] "
                    f"extends outside screen bounds ({cfg_w}x{cfg_h})."
                )

    # Validate minimap bounds
    mm = cfg.get("minimap", {})
    mx = mm.get("x", 16)
    my = mm.get("y", 16)
    msize = mm.get("size", 256)
    if mx < 0 or my < 0 or (mx + msize) > cfg_w or (my + msize) > cfg_h:
        result.warnings.append(
            f"Minimap region [{mx}, {my}, size={msize}] "
            f"extends outside screen bounds ({cfg_w}x{cfg_h})."
        )

    # 5. Check model files
    det_model_rel = cfg.get("detection", {}).get("model_path", "models/cs2_yolov8n.onnx")
    det_model_abs = os.path.join(PROJECT_ROOT, det_model_rel)
    if not os.path.exists(det_model_abs):
        result.errors.append(f"Required YOLO model missing: {det_model_abs}")
    else:
        file_size_mb = os.path.getsize(det_model_abs) / (1024 * 1024)
        if file_size_mb < 0.5:
            result.errors.append(
                f"YOLO model file appears truncated/corrupted ({file_size_mb:.2f} MB): "
                f"{det_model_abs}"
            )
        else:
            result.info.append(f"Detection model: {det_model_rel} ({file_size_mb:.1f} MB, OK)")

    # Check movement policy model if enabled
    mov_mode = cfg.get("bot", {}).get("movement_mode", "waypoint")
    if mov_mode == "bc":
        bc_model_rel = cfg.get("bc_movement", {}).get("model_path", "models/movement.onnx")
        bc_model_abs = os.path.join(PROJECT_ROOT, bc_model_rel)
        if not os.path.exists(bc_model_abs):
            result.warnings.append(
                f"BC policy model not found: {bc_model_abs}. Will fall back to waypoint navigation."
            )

    # 6. Check map and waypoint files
    map_name = cfg.get("bot", {}).get("map", "dust2")
    wp_path = os.path.join(PROJECT_ROOT, "config", "maps", f"{map_name}.json")
    if os.path.exists(wp_path):
        try:
            import json

            with open(wp_path, encoding="utf-8") as f:
                wp_data = json.load(f)
            result.info.append(f"Waypoint map: {map_name} ({len(wp_data)} waypoints, OK)")
        except Exception as e:
            result.warnings.append(f"Waypoint file {wp_path} failed to parse: {e}")
    else:
        # Check available maps
        available_maps = []
        maps_dir = os.path.join(PROJECT_ROOT, "config", "maps")
        if os.path.exists(maps_dir):
            for fname in os.listdir(maps_dir):
                if fname.endswith(".json") and not fname.endswith("_areas.json"):
                    available_maps.append(fname[:-5])
        result.warnings.append(
            f"Waypoint file not found for map '{map_name}' ({wp_path}). "
            f"Available maps: {available_maps if available_maps else 'none'}. "
            f"Bot will safely use visual obstacle avoidance (WallFollower)."
        )

    # 7. Check if CS2 is active
    if check_cs2_running():
        result.info.append("Counter-Strike 2: window/process detected and active")
    else:
        result.warnings.append(
            "Counter-Strike 2 is not currently running. Launch CS2 and join an offline match."
        )

    if not quiet:
        print_diagnostics_report(result)

    return result


def check_cs2_running() -> bool:
    """Check if Counter-Strike 2 window or process is detected."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, "Counter-Strike 2")
        return bool(hwnd)
    except Exception:
        return False


def print_diagnostics_report(res: ValidationResult) -> None:
    """Print an attractive and clear pre-flight validation report to stdout."""
    print("=" * 72)
    print("           CS2 DEATHMATCH BOT - PRE-FLIGHT SYSTEM DOCTOR           ")
    print("=" * 72)

    for item in res.info:
        print(f"  [OK]   {item}")

    for item in res.warnings:
        print(f"  [WARN] {item}")

    for item in res.errors:
        print(f"  [FAIL] {item}")

    print("-" * 72)
    if res.is_valid:
        print("  STATUS: SYSTEM READY FOR LOCAL OFFLINE CS2 TEST")
    else:
        print("  STATUS: SETUP ISSUES FOUND - PLEASE RESOLVE [FAIL] ITEMS ABOVE")
    print("=" * 72)
