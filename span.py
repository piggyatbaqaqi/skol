"""Represent a contiguous region of source text.

A Span encapsulates position information for a paragraph within
the source article.txt, enabling precise highlighting and gap detection
for the Source Context Viewer.
"""
from typing import Any, Dict, Optional


class Span:
    """Represents a contiguous region of source text."""

    def __init__(
        self,
        paragraph_number: int,
        start_line: int,
        end_line: int,
        start_char: int,
        end_char: int,
        pdf_page: int,
        pdf_label: Optional[str],
        empirical_page: Optional[str],
    ):
        """
        Initialize a Span with position information.

        Args:
            paragraph_number: Sequential paragraph number in the document
            start_line: First line number of the span
            end_line: Last line number of the span
            start_char: Character offset of span start in article.txt
            end_char: Character offset of span end in article.txt
            pdf_page: PDF page number (from PDF page markers)
            pdf_label: PDF page label (may differ from page number)
            empirical_page: Journal page number for bibliographic references
        """
        self.paragraph_number = paragraph_number
        self.start_line = start_line
        self.end_line = end_line
        self.start_char = start_char
        self.end_char = end_char
        self.pdf_page = pdf_page
        self.pdf_label = pdf_label
        self.empirical_page = empirical_page

    def has_gap_before(self, other: 'Span') -> bool:
        """
        Check if there's a gap between other span and this one.

        Uses character offset comparison which is more reliable than
        paragraph_number because paragraph gaps could result from
        blank paragraphs, figures, or tables that were intentionally skipped.

        Args:
            other: The preceding span to compare against

        Returns:
            True if this span starts after other span ends (indicating a gap)
        """
        return self.start_char > other.end_char

    def gap_size(self, other: 'Span') -> int:
        """
        Return the character count of the gap between other and this span.

        Args:
            other: The preceding span to compare against

        Returns:
            Number of characters between end of other and start of this span,
            or 0 if there is no gap (spans are adjacent or overlapping)
        """
        return max(0, self.start_char - other.end_char)

    def as_dict(self) -> Dict[str, Any]:
        """
        Export as JSON-serializable dict for CouchDB storage.

        Returns:
            Dictionary with all span attributes
        """
        return {
            'paragraph_number': self.paragraph_number,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'pdf_page': self.pdf_page,
            'pdf_label': self.pdf_label,
            'empirical_page': self.empirical_page,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Span':
        """
        Reconstruct a Span from a CouchDB document dict.

        Args:
            d: Dictionary with span attributes

        Returns:
            New Span instance
        """
        return cls(
            paragraph_number=d['paragraph_number'],
            start_line=d['start_line'],
            end_line=d['end_line'],
            start_char=d['start_char'],
            end_char=d['end_char'],
            pdf_page=d['pdf_page'],
            pdf_label=d.get('pdf_label'),
            empirical_page=d.get('empirical_page'),
        )

    def __repr__(self) -> str:
        return (
            f"Span(para={self.paragraph_number}, "
            f"lines={self.start_line}-{self.end_line}, "
            f"chars={self.start_char}-{self.end_char}, "
            f"pdf={self.pdf_page}, empirical={self.empirical_page})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Span):
            return NotImplemented
        return (
            self.paragraph_number == other.paragraph_number
            and self.start_line == other.start_line
            and self.end_line == other.end_line
            and self.start_char == other.start_char
            and self.end_char == other.end_char
            and self.pdf_page == other.pdf_page
            and self.pdf_label == other.pdf_label
            and self.empirical_page == other.empirical_page
        )
