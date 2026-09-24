"""Tests for CS2 process detection, safe startup gating, countdown timing, and zero auto-launch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.main import Bot
from src.utils.validator import check_cs2_process_running, validate_environment


def test_process_detection_real_and_absent():
    """Verify check_cs2_process_running accurately detects running processes and absent ones."""
    # When checking for python.exe (which is running this pytest session), it must return True
    assert check_cs2_process_running("python.exe") is True
    # Case insensitivity test
    assert check_cs2_process_running("PYTHON.EXE") is True

    # When checking for a definitely nonexistent process, it must return False
    assert check_cs2_process_running("nonexistent_process_123456789.exe") is False


def test_negative_delay_rejection_in_bot_init():
    """Verify Bot.__init__ rejects negative start_delay with ValueError."""
    with pytest.raises(ValueError, match="start_delay must be non-negative"):
        Bot(start_delay=-1, require_cs2=False)

    with pytest.raises(ValueError, match="start_delay must be non-negative"):
        Bot(start_delay=-5, require_cs2=False)


def test_bot_start_aborts_safely_when_cs2_not_running(capsys):
    """Verify Bot.start() immediately and safely exits without entering main loop if CS2 is absent."""
    with patch("src.utils.validator.check_cs2_process_running", return_value=False):
        bot = Bot(start_delay=5, require_cs2=True)
        bot.start()

        assert bot.running is False
        captured = capsys.readouterr()
        assert "Counter-Strike 2 ('cs2.exe') is not currently running" in captured.out


def test_countdown_does_not_consume_max_run_seconds():
    """Verify that _loop_start is recorded AFTER countdown so countdown does not consume max_run_seconds."""
    fake_now = [100.0]

    def mock_perf_counter():
        return fake_now[0]

    def mock_sleep(seconds):
        fake_now[0] += seconds

    with (
        patch("src.utils.validator.check_cs2_process_running", return_value=True),
        patch("time.perf_counter", side_effect=mock_perf_counter),
        patch("time.sleep", side_effect=mock_sleep),
        patch.object(Bot, "_main_loop", return_value=None) as mock_main_loop,
        patch("threading.Thread"),
    ):
        bot = Bot(start_delay=5, require_cs2=True)
        bot._max_run_seconds = 10
        bot.start()

        # Countdown should have run for 5 seconds (from 100.0 to 105.0)
        assert fake_now[0] == 105.0
        # _loop_start must be set AFTER countdown (at 105.0, not 100.0)
        assert bot._loop_start == 105.0
        # Main loop must have been called
        mock_main_loop.assert_called_once()


def test_panic_aborts_during_countdown():
    """Verify that triggering stop during countdown aborts immediately before main loop."""

    def mock_sleep_panic(seconds):
        # Simulate user pressing panic key during the first countdown second
        bot.running = False

    with (
        patch("src.utils.validator.check_cs2_process_running", return_value=True),
        patch("time.sleep", side_effect=mock_sleep_panic),
        patch.object(Bot, "_main_loop") as mock_main_loop,
        patch("threading.Thread"),
    ):
        bot = Bot(start_delay=5, require_cs2=True)
        bot.start()

        # Main loop must NOT have been called because countdown was aborted
        mock_main_loop.assert_not_called()


def test_validator_reports_cs2_process_status():
    """Verify validate_environment reports CS2 process status accurately."""
    # When CS2 is not running
    with patch("src.utils.validator.check_cs2_process_running", return_value=False):
        res = validate_environment(quiet=True)
        assert any("cs2.exe" in w and "not running" in w for w in res.warnings)

    # When CS2 is running
    with patch("src.utils.validator.check_cs2_process_running", return_value=True):
        res = validate_environment(quiet=True)
        assert any("cs2.exe" in item and "running" in item for item in res.info)


def test_no_cs2_auto_launch_in_codebase():
    """Verify no code path launches CS2 via Steam URLs, subprocess, or shell execution."""
    import re

    forbidden_patterns = [
        re.compile(r"steam://run/730", re.IGNORECASE),
        re.compile(r"subprocess\.(Popen|run|call)\(\s*\[.*cs2\.exe", re.IGNORECASE),
        re.compile(r"os\.system\(.*cs2\.exe", re.IGNORECASE),
        re.compile(r"os\.startfile\(.*cs2", re.IGNORECASE),
    ]

    import glob
    import os

    root_dir = os.path.dirname(os.path.dirname(__file__))
    source_files = glob.glob(os.path.join(root_dir, "src", "**", "*.py"), recursive=True)
    source_files.append(os.path.join(root_dir, "run.py"))

    for filepath in source_files:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for pattern in forbidden_patterns:
                assert not pattern.search(content), (
                    f"Forbidden auto-launch pattern found in {filepath}: {pattern.pattern}"
                )
