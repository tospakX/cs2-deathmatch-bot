"""Easy launcher and pre-flight runner for CS2 Deathmatch Bot.

Validates environment, configuration, models, and screen resolution,
then starts the bot in your offline CS2 lobby.

Usage:
    python run.py
    python run.py --personality tryhard --map dust2
    python run.py --check
    run.bat
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.main import Bot
from src.utils.validator import validate_environment


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Deathmatch Bot - Easy Launcher & Pre-Flight Check"
    )
    parser.add_argument(
        "--personality",
        "-p",
        type=str,
        default=None,
        help="Personality profile (noob, average, tryhard)",
    )
    parser.add_argument(
        "--map",
        "-m",
        type=str,
        default=None,
        help="Map name (dust2, etc.) for waypoint navigation",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug overlay window",
    )
    parser.add_argument(
        "--max-run-seconds",
        type=int,
        default=None,
        help="Auto-stop failsafe duration in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Countdown delay in seconds before activating inputs (default 5s)",
    )
    parser.add_argument(
        "--check",
        "--doctor",
        action="store_true",
        help="Run pre-flight checks and diagnostics only, without starting the bot",
    )
    args = parser.parse_args()

    # 1. Pre-flight validation
    val_res = validate_environment(
        config_path="config/settings.yaml",
        auto_adapt_resolution=True,
        quiet=False,
    )

    if not val_res.is_valid:
        print("\n[Launcher] ERROR: Pre-flight checks failed! Please fix the errors listed above.")
        sys.exit(1)

    if args.check:
        print("\n[Launcher] Pre-flight check successful. Exiting (--check mode).")
        sys.exit(0)

    # 2. Launch bot
    print("\n[Launcher] Pre-flight checks passed! Starting bot...")
    print("[Launcher] Press HOME in CS2 to Pause/Resume, or END to Emergency Stop.")
    print("=" * 72 + "\n")

    bot = Bot(
        personality_name=args.personality,
        map_name=args.map,
        validated_config=val_res.config,
        start_delay=args.delay,
    )

    if args.no_debug:
        bot.debug = None

    if args.max_run_seconds is not None:
        bot._max_run_seconds = args.max_run_seconds

    bot.start()


if __name__ == "__main__":
    main()
