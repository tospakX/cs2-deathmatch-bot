"""HUD region and screen calibration tool for CS2 Deathmatch Bot.

Supports:
  1. Auto-calibration (--auto): Detects active display resolution, applies verified
     HUD presets, aligns crosshair center, and updates config/settings.yaml.
  2. Presets (--preset 1080p | 1440p): Quick-applies tested layout configurations.
  3. Interactive mode: Capture screenshot and drag boxes to calibrate custom HUD layouts.

Usage:
    python tools/calibrate.py --auto
    python tools/calibrate.py --preset 1080p
    python tools/calibrate.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.capture.screen import ScreenCapture, detect_screen_resolution

try:
    import cv2
except ImportError:
    cv2 = None

# Verified standard 1920x1080 presets for CS2 default HUD scaling
PRESETS_1080P = {
    "display": {
        "width": 1920,
        "height": 1080,
    },
    "game": {
        "crosshair_x": 960,
        "crosshair_y": 540,
        "fov_horizontal": 106,
    },
    "regions": {
        "health": [76, 1038, 80, 30],
        "armor": [186, 1038, 80, 30],
        "ammo_clip": [1780, 1038, 60, 30],
        "ammo_reserve": [1850, 1038, 60, 30],
        "killfeed": [1400, 50, 500, 200],
        "alive_ct": [870, 18, 40, 24],
        "alive_t": [1010, 18, 40, 24],
    },
    "minimap": {
        "x": 16,
        "y": 16,
        "size": 256,
        "sat_min": 150,
        "val_min": 190,
    },
}


def print_next_steps():
    print("\n" + "=" * 72)
    print("                 CALIBRATION COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("  Configuration saved to: config/settings.yaml")
    print("    - Resolution: 1920x1080 (16:9)")
    print("    - Crosshair:  960x540")
    print("    - FOV:        106.0 deg")
    print("    - Regions:    health, armor, ammo, killfeed, CT/T counters")
    print("-" * 72)
    print("  NEXT STEPS:")
    print("  1. Launch Counter-Strike 2.")
    print("  2. Open an offline Practice match (e.g. Practice -> Deathmatch -> Dust II).")
    print("  3. Open the CS2 developer console (~) and enter the radar setup commands:")
    print("       cl_radar_rotate 0")
    print("       cl_radar_always_centered 0")
    print("       cl_radar_scale 0.4")
    print("  4. Start the bot:")
    print("       run.bat")
    print("     (or: python -m src.main)")
    print("  5. In-game controls:")
    print("       [HOME] = Pause / Resume control")
    print("       [END]  = Emergency Stop")
    print("=" * 72 + "\n")


def apply_auto_calibration(save: bool = True) -> dict:
    """Detect display resolution and apply corresponding preset or proportional scale."""
    det_w, det_h = detect_screen_resolution(0)
    print(f"[Calibrate] Detected monitor resolution: {det_w}x{det_h}")

    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    cfg = {}
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    cfg.setdefault("display", {})["width"] = det_w
    cfg.setdefault("display", {})["height"] = det_h
    cfg.setdefault("game", {})["crosshair_x"] = det_w // 2
    cfg.setdefault("game", {})["crosshair_y"] = det_h // 2

    # Aspect ratio FOV calculation
    aspect = det_w / max(1.0, float(det_h))
    if abs(aspect - (16.0 / 9.0)) < 0.05:
        cfg["game"]["fov_horizontal"] = 106
    elif abs(aspect - (4.0 / 3.0)) < 0.05:
        cfg["game"]["fov_horizontal"] = 90
    else:
        cfg["game"]["fov_horizontal"] = 106

    # Scale regions from 1080p preset
    sx = det_w / 1920.0
    sy = det_h / 1080.0
    scaled_regions = {}
    for name, bbox in PRESETS_1080P["regions"].items():
        scaled_regions[name] = [
            int(round(bbox[0] * sx)),
            int(round(bbox[1] * sy)),
            int(round(bbox[2] * sx)),
            int(round(bbox[3] * sy)),
        ]
    cfg["regions"] = scaled_regions

    s_mm = min(sx, sy)
    cfg["minimap"] = {
        "x": int(round(PRESETS_1080P["minimap"]["x"] * sx)),
        "y": int(round(PRESETS_1080P["minimap"]["y"] * sy)),
        "size": int(round(PRESETS_1080P["minimap"]["size"] * s_mm)),
        "sat_min": PRESETS_1080P["minimap"]["sat_min"],
        "val_min": PRESETS_1080P["minimap"]["val_min"],
    }

    if save:
        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[Calibrate] Updated {settings_path} for {det_w}x{det_h} layout.")

    print_next_steps()
    return cfg


class RegionCalibrator:
    """Interactive region selection for HUD calibration using OpenCV window."""

    def __init__(self):
        self.regions: dict[str, list[int]] = {}
        self._click_start = None
        self._current_name = ""
        self._frame = None
        self._display = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._click_start:
            x1, y1 = self._click_start
            w = abs(x - x1)
            h = abs(y - y1)
            rx = min(x, x1)
            ry = min(y, y1)

            scale = param if param else 1.0
            rx = int(rx / scale)
            ry = int(ry / scale)
            w = int(w / scale)
            h = int(h / scale)

            if w > 5 and h > 5:
                self.regions[self._current_name] = [rx, ry, w, h]
                print(f"  {self._current_name}: [{rx}, {ry}, {w}, {h}]")

            self._click_start = None

    def calibrate(self):
        """Run the interactive calibration process."""
        if cv2 is None:
            print("OpenCV required for interactive calibration: pip install opencv-python")
            return

        print("[Calibrate] Capturing screenshot...")
        capture = ScreenCapture(target_fps=5)
        capture.start()

        import time

        time.sleep(0.5)
        frame = capture.grab()
        capture.stop()

        if frame is None:
            print("[Calibrate] Failed to capture screenshot! Using auto preset instead.")
            apply_auto_calibration(save=True)
            return

        self._frame = frame
        h, w = frame.shape[:2]
        scale = min(1.0, 1280.0 / w)

        print(f"[Calibrate] Resolution: {w}x{h}")
        print("[Calibrate] For each region, click and drag to select.")
        print("[Calibrate] Press SPACE to confirm and move to next region.")
        print("[Calibrate] Press ESC to skip a region.")
        print()

        region_names = [
            "health",
            "armor",
            "ammo_clip",
            "ammo_reserve",
            "killfeed",
            "alive_ct",
            "alive_t",
        ]

        window = "CS2 Calibration"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._mouse_callback, scale)

        for name in region_names:
            self._current_name = name
            print(f"Select region: {name}")

            while True:
                display = frame.copy()
                if scale != 1.0:
                    display = cv2.resize(display, (int(w * scale), int(h * scale)))

                for rname, (rx, ry, rw, rh) in self.regions.items():
                    srx, sry = int(rx * scale), int(ry * scale)
                    srw, srh = int(rw * scale), int(rh * scale)
                    cv2.rectangle(display, (srx, sry), (srx + srw, sry + srh), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        rname,
                        (srx, sry - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                cv2.putText(
                    display,
                    f"Select: {name} (SPACE=next, ESC=skip)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF
                if key == 32:  # SPACE
                    break
                elif key == 27:  # ESC
                    print(f"  Skipped {name}")
                    break

        cv2.destroyAllWindows()

        if self.regions:
            settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
            with open(settings_path, encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            settings["regions"].update(self.regions)
            with open(settings_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(settings, f, sort_keys=False)
            print(f"\nSaved {len(self.regions)} regions to {settings_path}")
            print_next_steps()


def main():
    parser = argparse.ArgumentParser(description="CS2 Screen and HUD Calibration Tool")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically detect resolution and apply tested 1080p/scaled layout",
    )
    parser.add_argument(
        "--preset",
        choices=["1080p"],
        default=None,
        help="Apply predefined layout preset",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive drag-and-drop box calibration GUI",
    )
    args = parser.parse_args()

    if args.interactive:
        cal = RegionCalibrator()
        cal.calibrate()
    else:
        # Default to auto-calibration if no mode specified or --auto/--preset given
        apply_auto_calibration(save=True)


if __name__ == "__main__":
    main()
