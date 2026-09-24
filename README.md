# CS2 Deathmatch Bot

A computer-vision behavioral simulation that plays Counter-Strike 2 deathmatch in private/offline matches from raw screen pixels alone.

The bot sees the game via YOLO object detection, tracks player location on the minimap, and generates continuous, human-like mouse and keyboard inputs via Win32 `SendInput`. No game memory is read or written; all inputs and perceptions are closed-loop vision in, synthetic input out.

> [!NOTE]
> This project is designed strictly for offline, private server testing and behavioral simulation research. It does not contain anti-cheat evasion, bypasses, or stealth mechanisms.

---

## Quick Start (Fresh Checkout to Offline Match)

### 1. Requirements & Installation

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10, 3.11, or 3.12
- **Resolution**: 1920x1080 (standard 16:9 fullscreen or borderless window)

Open PowerShell or Command Prompt in the repository folder:

```cmd
:: Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

:: Install runtime and test dependencies
pip install -r requirements.txt
pip install ruff pytest
```

### 2. Verify Your Setup

Run the pre-flight doctor to ensure dependencies, models, and screen resolution are valid:

```cmd
run.bat --check
```
*(Alternatively: `python -m src.main --check` or `python tools/doctor.py`)*

If any check fails, the doctor will display exact remediation steps.

---

## Running in a Local Offline CS2 Match

### Step 1: Start CS2 in Practice Mode

1. Launch Counter-Strike 2.
2. Go to **Play** &rarr; **Practice** &rarr; **Deathmatch**.
3. Select **Dust II** (or another supported map) and click **Go**.

### Step 2: Configure Radar Console Commands

The bot tracks its position hue-agnostically from a fixed minimap. Open the in-game developer console (`~`) and enter:

```text
cl_radar_rotate 0
cl_radar_always_centered 0
cl_radar_scale 0.4
```

*(Tip: You can add these lines to your `autoexec.cfg` so they apply automatically.)*

### Step 3: Start the Bot

Run the launcher from PowerShell or Command Prompt:

```cmd
run.bat
```
*(Alternatively: `python -m src.main`)*

Command-line options:
- Select personality: `run.bat -p tryhard` (`noob`, `average`, or `tryhard`)
- Select map: `run.bat -m dust2`
- Disable debug overlay window: `run.bat --no-debug`
- Custom timeout failsafe: `run.bat --max-run-seconds 60`

### Step 4: In-Game Hotkeys

| Hotkey | Action | Description |
|---|---|---|
| `HOME` | **Pause / Resume** | Instantly releases keyboard and mouse buttons so you can take control. Press again to hand control back to the bot. |
| `END` | **Emergency Stop** | Immediately kills the bot process and safely releases all inputs. |

A dedicated 20ms watchdog thread polls these keys globally so they function even while CS2 is focused.

---

## Calibration (Optional / First-Time Setup)

The bot comes pre-calibrated with verified coordinates for **1920x1080**. You only need calibration if you use non-standard resolutions or customized HUD scales.

### 1. Screen & HUD Auto-Calibration
To auto-detect your monitor resolution and apply the calibrated HUD bounding boxes:
```cmd
python tools/calibrate.py --auto
```
*(To manually verify or drag custom bounding boxes in a GUI window, run `python tools/calibrate.py --interactive`)*

### 2. Navigation Turn Calibration
To calibrate the heading error turn gain and turn direction against your in-game sensitivity:
1. Join an offline match, stand in an open area facing a long sightline, and run:
   ```cmd
   python tools/calibrate_nav.py --write
   ```
2. Switch to CS2 before the countdown ends. The tool will walk forward, turn the camera, and save `nav_turn_gain` and `nav_invert_turn` to `config/settings.yaml`.

### 3. Aim Scale Calibration
To verify the relationship between mouse counts and in-game pixel displacement:
1. Stand still in CS2 looking at a textured wall, then run:
   ```cmd
   python tools/calibrate_aim.py --write
   ```

---

## Architecture Overview

```text
Perception:
  Screen Capture (DXcam / mss fallback)
    └── YOLOv8n Target Detector + Confirmation Filter
    └── HUD Reader (Health, Armor, Ammo, Alive Status)
    └── Minimap Reader (Hue-agnostic position & heading)

Player State & Cognition:
  PlayerState (Monotonic Clock abstraction, temporal state, health, spray)
    └── PerceptionSystem (Hungarian assignment target tracking & SpatialMemory)
    └── DecisionEngine (Attention, reaction latency, firing mode, movement)

Motor Execution & Physical Input:
  MotorPlanner (Minimum-jerk reaching trajectories, sub-step easing)
    └── FiringController (Cycle timing, tap/burst/spray commitment, recoil compensation)
    └── MovementController (Persistent strafing, counter-strafing, wall avoidance)
    └── MotorExecutor -> Win32 SendInput (Mouse & Keyboard)
```

---

## Development & Testing

Run the automated test suite, linter, and behavioral simulation:

```cmd
:: 1. Run all pytest unit & integration tests
python -m pytest

:: 2. Check code style and formatting
ruff check src tests tools run.py
ruff format --check src tests tools run.py

:: 3. Run long-run offline behavioral simulation
python tools/run_behavioral_simulation.py --personality all
```
