"""Skeleton — implementation lands in the next commit (TDD)."""

from typing import Any, Dict, Optional


class SpanResolutionError(RuntimeError):
    pass


class CoordinateSpace:
    pass


def coordinate_space(treatment: Dict[str, Any]) -> CoordinateSpace:
    raise NotImplementedError


def span_head(text: str) -> str:
    raise NotImplementedError


def verify_head(stored: Optional[str], actual: str) -> None:
    raise NotImplementedError


def resolve_span(treatment: Dict[str, Any], span: Dict[str, Any],
                 server: Any) -> str:
    raise NotImplementedError
