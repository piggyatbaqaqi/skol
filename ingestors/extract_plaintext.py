"""Centralized plaintext extraction from multiple sources.

Provides functions to extract article plaintext from:
- PDF bytes (via PDFSectionExtractor)
- JATS/TaxPub XML strings
- NCBI E-utilities efetch API
- YEDDA annotation strings (tag stripping)

This module is the single point of entry for obtaining article.txt
content. bin/extract_plaintext.py provides the CLI; predict_classifier
requires article.txt to already exist.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

from ingestors.jats_to_yedda import extract_text, strip_ns
from ingestors.rate_limited_client import RateLimitedHttpClient
from pdf_section_extractor import PDFSectionExtractor

# NCBI E-utilities base URL
_EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Default skip tags for JATS text extraction (same as jats_to_yedda)
_JATS_SKIP_TAGS = frozenset({"object-id"})

# NCBI E-utilities user agent
_NCBI_USER_AGENT = "skol/1.0 (taxonomy extraction; mailto:piggy@baqaqi.com)"


def plaintext_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plaintext from PDF bytes.

    Wraps PDFSectionExtractor.pdf_to_text(). Preserves page markers
    (``--- PDF Page N Label L ---``) for downstream section parsing.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Extracted text with page markers.

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed.
    """
    extractor = PDFSectionExtractor(verbosity=0)
    return extractor.pdf_to_text(pdf_bytes)


def plaintext_from_jats(xml_string: str) -> str:
    """Extract plaintext from a JATS/TaxPub XML string.

    Extracts text from the ``<body>`` element only (excludes abstract,
    front matter, and back matter). Preserves section structure with
    blank-line separators. Inline markup is stripped; ``<object-id>``
    elements are skipped.

    Args:
        xml_string: Complete JATS XML document as a string.

    Returns:
        Body text with sections separated by blank lines.

    Raises:
        ValueError: If the XML has no ``<body>`` element.
    """
    root = ET.fromstring(xml_string)
    strip_ns(root)
    body = root.find(".//body")
    if body is None:
        raise ValueError("JATS XML has no <body> element")

    # Extract text from each top-level child of body, separated by
    # blank lines to preserve section boundaries.
    section_texts: List[str] = []
    for child in body:
        text = extract_text(child, _JATS_SKIP_TAGS).strip()
        if text:
            section_texts.append(text)

    return "\n\n".join(section_texts)


def plaintext_from_efetch(
    pmcid: str,
    api_key: Optional[str] = None,
) -> str:
    """Download JATS XML from NCBI E-utilities efetch and extract plaintext.

    PMC efetch always returns JATS XML regardless of retmode, so this
    function downloads the XML and parses it into plaintext via
    ``plaintext_from_jats()``.

    Respects NCBI rate limits: 3 requests/second without an API key,
    10 requests/second with one.

    Args:
        pmcid: PMC article ID (e.g., "PMC10858444" or "10858444").
        api_key: Optional NCBI API key for higher rate limits.

    Returns:
        Article plaintext extracted from JATS XML body.

    Raises:
        ValueError: If the HTTP response is non-200, empty, or lacks
            a ``<body>`` element.
    """
    # Strip leading "PMC" if present for the id parameter.
    numeric_id = pmcid.lstrip("PMC")

    params: Dict[str, str] = {
        "db": "pmc",
        "id": f"PMC{numeric_id}",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EFETCH_BASE}?{urlencode(params)}"

    # NCBI rate limits: 3 rps without key, 10 rps with key.
    if api_key:
        rate_min_ms = 100
        rate_max_ms = 150
    else:
        rate_min_ms = 334
        rate_max_ms = 500

    client = RateLimitedHttpClient(
        user_agent=_NCBI_USER_AGENT,
        verbosity=0,
        rate_limit_min_ms=rate_min_ms,
        rate_limit_max_ms=rate_max_ms,
    )

    response = client.get(url)

    if response.status_code != 200:
        raise ValueError(
            f"efetch returned HTTP {response.status_code} for {pmcid}"
        )

    xml_text = response.text
    if not xml_text or not xml_text.strip():
        raise ValueError(f"efetch returned empty response for {pmcid}")

    return plaintext_from_jats(xml_text)


# YEDDA tag pattern: [@text#Tag*]
_YEDDA_BLOCK_RE = re.compile(r"\[@\s*(.*?)\s*#([^*]+)\*\]", re.DOTALL)


def plaintext_from_yedda(yedda_text: str) -> str:
    """Extract plaintext from a YEDDA annotation string by stripping tags.

    Recovers the original article text from ``[@text#Tag*]`` blocks.
    Blocks are joined by blank lines, preserving paragraph structure.

    Args:
        yedda_text: YEDDA-annotated string.

    Returns:
        Plain text with tags stripped, blocks separated by blank lines.
    """
    parts: List[str] = []
    for match in _YEDDA_BLOCK_RE.finditer(yedda_text):
        text = match.group(1).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def count_yedda_tags(yedda_text: str) -> Tuple[Set[str], int]:
    """Count distinct tag types in a YEDDA annotation string.

    Args:
        yedda_text: YEDDA-annotated string.

    Returns:
        Tuple of (set of tag names, count of blocks).
    """
    tags: Set[str] = set()
    block_count = 0
    for match in _YEDDA_BLOCK_RE.finditer(yedda_text):
        tag = match.group(2).strip()
        tags.add(tag)
        block_count += 1
    return tags, block_count
