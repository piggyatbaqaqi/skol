"""Tests for export_golden_ids.py."""

import subprocess
import sys


def test_help() -> None:
    """Verify the script runs with --help."""
    result = subprocess.run(
        [sys.executable, "bin/export_golden_ids.py", "--help"],
        capture_output=True,
        text=True,
        cwd="/data/piggy/src/github.com/piggyatbaqaqi/skol",
    )
    assert result.returncode == 0
    assert "golden" in result.stdout.lower()
