"""Hygiene tests for the bin/ argparse-strict refactor.

Every CLI entry point in ``bin/`` must reject unknown flags (exit code 2)
instead of silently swallowing them.  See ``docs/bin-argparse-strict.md``
for the design (the ``parents=[common_parser()]`` parent-parser pattern).

These are regression guards over already-converted scripts, not
test-first specs for new code:

* :func:`test_no_unpragmaed_parse_known_args` -- static: ``parse_known_args``
  is only permitted behind a ``# pragma: argparse-passthrough`` comment.
* :func:`test_cli_rejects_unknown_flag` -- behavioral: running each strict
  CLI with a bogus flag exits 2.
"""
import ast
import concurrent.futures
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

BIN = Path(__file__).resolve().parent
PRAGMA = "pragma: argparse-passthrough"
BOGUS_FLAG = "--zzz-not-a-real-flag-qwxz"
RUN_TIMEOUT = 180  # heavy ML imports run before argparse is reached


def _bin_modules() -> List[Path]:
    return [
        p for p in sorted(BIN.glob("*.py"))
        if not p.name.endswith("_test.py")
    ]


def _is_cli(text: str) -> bool:
    """A runnable CLI builds a parser and has a __main__ entry point."""
    return "argparse.ArgumentParser(" in text and "__main__" in text


def _cli_scripts() -> List[Path]:
    return [p for p in _bin_modules() if _is_cli(p.read_text(errors="replace"))]


def _calls_parse_known_args(text: str) -> bool:
    """True if the source *calls* parse_known_args (ignoring comments)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - shouldn't happen on tracked code
        return "parse_known_args(" in text
    return any(
        isinstance(node, ast.Attribute) and node.attr == "parse_known_args"
        for node in ast.walk(tree)
    )


BIN_MODULES = _bin_modules()
CLI_SCRIPTS = _cli_scripts()


@pytest.mark.parametrize("module", BIN_MODULES, ids=lambda p: p.name)
def test_no_unpragmaed_parse_known_args(module: Path) -> None:
    """parse_known_args is only allowed behind the passthrough pragma."""
    text = module.read_text(errors="replace")
    if _calls_parse_known_args(text):
        assert PRAGMA in text, (
            f"{module.name} calls parse_known_args() without "
            f"'# {PRAGMA}'.  Adopt the parents=[common_parser()] pattern "
            f"with parse_args(), or justify deliberate passthrough with the "
            f"pragma comment."
        )


def _run_bogus(script: Path) -> Tuple[Optional[int], str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script), BOGUS_FLAG],
            capture_output=True, text=True, timeout=RUN_TIMEOUT, cwd=str(BIN),
        )
        return proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired:
        return None, "timeout"


def test_cli_rejects_unknown_flag() -> None:
    """Every strict CLI exits 2 on an unknown flag.

    Scripts are launched concurrently (heavy ML imports dominate the wall
    time).  A timeout or an unmet import dependency is skipped, not failed
    -- the strict-parsing contract is what's under test, not the runtime
    environment.
    """
    strict = [
        p for p in CLI_SCRIPTS
        if PRAGMA not in p.read_text(errors="replace")
    ]
    assert strict, "no strict CLI scripts discovered -- test misconfigured"

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(zip(strict, pool.map(_run_bogus, strict)))

    failures: List[str] = []
    skipped: List[str] = []
    for script, (code, err) in results.items():
        if code == 2:
            continue
        if code is None:
            skipped.append(f"{script.name}: timed out >{RUN_TIMEOUT}s")
        elif "ModuleNotFoundError" in err or "ImportError" in err:
            skipped.append(f"{script.name}: import deps unavailable")
        else:
            failures.append(f"{script.name}: exit {code}\n{err[-400:].strip()}")

    assert not failures, (
        "Scripts that did not reject an unknown flag with exit 2:\n\n"
        + "\n\n".join(failures)
        + (f"\n\n(skipped: {'; '.join(skipped)})" if skipped else "")
    )
