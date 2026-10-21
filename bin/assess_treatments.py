#!/usr/bin/env python3
"""Record a quality verdict for every treatment in an experiment.

Runs the detector suite that ``tests/pathologies_test.py`` holds to
the fixture ground truth (``triage_signals.treatment_signals`` plus
``merge_metric.treatment_merge_metric``, rolled up by
``predicted_issues``) over the experiment's prose treatments, and
writes one verdict per treatment to the experiment's
``databases.treatment_quality`` database::

    {"_id": "<treatment id>", "clean": true, "flags": [],
     "usable_text": true, "gn_evaluated": true,
     "checks_version": "<hash>", "checked_at": "<ISO time>"}

A treatment is ``clean`` only when no detector fires, its text is
usable (``treatments_json_translator.is_translatable``: an empty
treatment fires no flags), and the gnfinder authored-binomial check
actually ran.  Clean means no known pathology, not perfect.

``checks_version`` hashes the detector sources, so adding or changing
a detector makes every verdict visibly stale; the next run reassesses
those and skips current ones (``--force`` reassesses everything).

Usage::

    bin/assess_treatments --experiment production_v4_1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
DETECTOR_SOURCES: Sequence[Path] = (
    REPO / 'treatments_to_structured' / 'triage_signals.py',
    REPO / 'treatments_to_structured' / 'merge_metric.py',
)

GnCheck = Callable[[str], Optional[bool]]


def checks_version(sources: Sequence[Path] = DETECTOR_SOURCES) -> str:
    """Hash of the detector source files, in order."""
    raise NotImplementedError


def assess(
    treatment: Dict[str, Any],
    authored_binomial: Optional[bool],
) -> Dict[str, Any]:
    """Run the detectors on one treatment.

    ``authored_binomial`` is the gnfinder result, or None when the
    check did not run.

    Returns ``{clean, flags, usable_text, gn_evaluated}``; ``flags`` is
    ``predicted_issues`` split on ``|`` (values kept, e.g.
    ``§6:merge_metric=14``).
    """
    raise NotImplementedError


def verdict_doc(
    treatment_id: str,
    verdict: Dict[str, Any],
    version: str,
    checked_at: str,
) -> Dict[str, Any]:
    """The quality-DB document for one verdict."""
    raise NotImplementedError


def needs_assessment(
    existing: Optional[Dict[str, Any]],
    version: str,
    force: bool = False,
) -> bool:
    """True when there is no verdict, it is stale, or ``force``."""
    raise NotImplementedError


def quality_db_name(config: Dict[str, Any]) -> str:
    """The experiment's treatment_quality database.

    Raises:
        ValueError: If the experiment record has none.
    """
    raise NotImplementedError


def assess_all(
    treatments: Iterable[Dict[str, Any]],
    quality_db: Any,
    gn_check: GnCheck,
    version: str,
    checked_at: str,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Assess each treatment that needs it and save its verdict.

    Returns ``{scanned, assessed, clean, skipped}``.
    """
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    from env_config import common_parser

    return argparse.ArgumentParser(
        parents=[common_parser()],
        description='Record a detector-suite quality verdict for every '
                    'treatment in an experiment.',
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
