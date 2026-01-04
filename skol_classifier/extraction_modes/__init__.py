"""
Extraction mode classes for different text granularities.

This module provides an object-oriented approach to handling different
extraction granularities (line, paragraph, section) for text classification.
"""

from .base import AnnotatedTextParser
from .line_mode import LineAnnotatedTextParser
from .paragraph_mode import ParagraphAnnotatedTextParser
from .section_mode import SectionAnnotatedTextParser
from .mode import ExtractionMode
from .line import LineExtractionMode
from .paragraph import ParagraphExtractionMode
from .section import SectionExtractionMode


def get_parser(mode: str, collapse_labels: bool = True) -> AnnotatedTextParser:
    """
    Factory function to create the appropriate annotated text parser.

    Args:
        mode: String identifier ('line', 'paragraph', or 'section')
        collapse_labels: Whether to collapse labels to 3 main categories

    Returns:
        AnnotatedTextParser subclass instance

    Raises:
        ValueError: If mode is not recognized
    """
    parsers = {
        'line': LineAnnotatedTextParser,
        'paragraph': ParagraphAnnotatedTextParser,
        'section': SectionAnnotatedTextParser,
    }

    if mode not in parsers:
        raise ValueError(
            f"Unknown extraction_mode: {mode}. "
            f"Must be 'line', 'paragraph', or 'section'"
        )

    return parsers[mode](collapse_labels=collapse_labels)


def get_mode(mode: str) -> ExtractionMode:
    """
    Factory function to create the appropriate extraction mode object.

    Args:
        mode: String identifier ('line', 'paragraph', or 'section')

    Returns:
        ExtractionMode subclass instance

    Raises:
        ValueError: If mode is not recognized
    """
    modes = {
        'line': LineExtractionMode,
        'paragraph': ParagraphExtractionMode,
        'section': SectionExtractionMode,
    }

    if mode not in modes:
        raise ValueError(
            f"Unknown extraction_mode: {mode}. "
            f"Must be 'line', 'paragraph', or 'section'"
        )

    return modes[mode]()


__all__ = [
    # Parser classes
    'AnnotatedTextParser',
    'LineAnnotatedTextParser',
    'ParagraphAnnotatedTextParser',
    'SectionAnnotatedTextParser',
    'get_parser',
    # Mode classes
    'ExtractionMode',
    'LineExtractionMode',
    'ParagraphExtractionMode',
    'SectionExtractionMode',
    'get_mode',
]
