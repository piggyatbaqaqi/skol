#!/usr/bin/env python3
"""Skeleton — implementation lands in the next commit (TDD)."""

from typing import Any, Dict


class AttachmentCache:
    def __init__(self, server: Any, max_entries: int = 8) -> None:
        raise NotImplementedError

    def text(self, db: str, doc_id: str, attachment: str) -> str:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


def backfill_treatment(treatment: Dict[str, Any],
                       cache: AttachmentCache) -> int:
    raise NotImplementedError
