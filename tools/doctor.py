"""Standalone system diagnostics tool for CS2 Deathmatch Bot.

Validates dependencies, configs, models, resolution, and waypoints.

Usage:
    python tools/doctor.py
    python tools/doctor.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.validator import validate_environment


def main():
    parser = argparse.ArgumentParser(description="CS2 Deathmatch Bot System Doctor")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--no-adapt",
        action="store_true",
        help="Do not auto-adapt resolution to active monitor",
    )
    args = parser.parse_args()

    result = validate_environment(
        config_path=args.config,
        auto_adapt_resolution=not args.no_adapt,
        quiet=False,
    )

    if not result.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
