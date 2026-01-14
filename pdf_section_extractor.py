"""
PDF Section Extractor for CouchDB

This module provides a class to extract section headers and paragraphs from
PDF attachments stored in CouchDB documents.

Usage:
    extractor = PDFSectionExtractor(
        couchdb_url='http://localhost:5984',
        username='admin',
        password='secret'
    )

    # Extract from specific document
    sections = extractor.extract_from_document(
        database='skol_dev',
        doc_id='00df9554e9834283b5e844c7a994ba5f',
        attachment_name='article.pdf'
    )

    # Or auto-detect PDF attachment
    sections = extractor.extract_from_document(
        database='skol_dev',
        doc_id='00df9554e9834283b5e844c7a994ba5f'
    )
"""

import os
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urlunparse
from io import BytesIO
import couchdb

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# PySpark availability check
PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import DataFrame
    PYSPARK_AVAILABLE = True
except ImportError:
    pass

from constants import pdf_page_pattern

class PDFSectionExtractor:
    """
    Extract section headers and paragraphs from PDF attachments in CouchDB.

    This class handles:
    - CouchDB authentication and connection
    - PDF attachment retrieval (directly in memory, no temp files)
    - Text extraction using PyMuPDF (fitz)
    - Intelligent section/paragraph parsing
    """

    def __init__(
        self,
        couchdb_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verbosity: int = 1,
        spark: Optional[Any] = None,
        read_text: bool = False,
        save_text: Optional[str] = None,
        union_batch_size: int = 1000
    ):
        """
        Initialize the PDF section extractor.

        Args:
            couchdb_url: CouchDB server URL (default: from COUCHDB_URL env var)
            username: CouchDB username (default: from COUCHDB_USER env var)
            password: CouchDB password (default: from COUCHDB_PASSWORD env var)
            verbosity: Logging verbosity (0=silent, 1=info, 2=debug)
            spark: SparkSession instance (required for DataFrame output)
            read_text: If True, read from .txt attachment instead of converting PDF
            save_text: Text attachment saving strategy:
                      'eager': Always convert PDF and save/replace .txt attachment
                      'lazy': Save .txt attachment only if one doesn't exist
                      None: Do not save .txt attachment
            union_batch_size: Number of DataFrames to union at once (default: 1000)
                            Larger values = fewer intermediate unions but deeper plans
                            Smaller values = more intermediate unions but shallower plans
        """
        self.verbosity = verbosity
        self.spark = spark
        self.read_text = read_text
        self.save_text = save_text
        self.union_batch_size = union_batch_size

        # Storage for figure captions (populated during parsing)
        self.figure_captions = []

        # Get credentials from environment if not provided
        self.couchdb_url = couchdb_url or os.environ.get('COUCHDB_URL', 'http://localhost:5984')
        self.username = username or os.environ.get('COUCHDB_USER', 'admin')
        self.password = password or os.environ.get('COUCHDB_PASSWORD', '')

        # Build authenticated URL
        self.auth_url = self._build_auth_url()

        # Connect to CouchDB server
        self.couch = couchdb.Server(self.auth_url)

        if self.verbosity >= 1:
            print(f"Connected to CouchDB at {self.couchdb_url}")

    def _build_auth_url(self) -> str:
        """Build CouchDB URL with embedded credentials."""
        if not self.username or not self.password:
            return self.couchdb_url

        parsed = urlparse(self.couchdb_url)
        netloc_with_auth = f"{self.username}:{self.password}@{parsed.hostname}"
        if parsed.port:
            netloc_with_auth += f":{parsed.port}"

        return urlunparse((
            parsed.scheme,
            netloc_with_auth,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

    def get_document(self, database: str, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document from CouchDB.

        Args:
            database: Database name
            doc_id: Document ID

        Returns:
            Document dictionary

        Raises:
            KeyError: If database doesn't exist
            couchdb.ResourceNotFound: If document doesn't exist
        """
        db = self.couch[database]
        return db[doc_id]

    def list_attachments(self, database: str, doc_id: str) -> Dict[str, Dict[str, Any]]:
        """
        List all attachments for a document.

        Args:
            database: Database name
            doc_id: Document ID

        Returns:
            Dictionary mapping attachment names to their metadata
        """
        doc = self.get_document(database, doc_id)
        return doc.get('_attachments', {})

    def find_pdf_attachment(self, database: str, doc_id: str) -> Optional[str]:
        """
        Find the first PDF or text attachment in a document.

        Searches for PDF files first, then falls back to .txt files.

        Args:
            database: Database name
            doc_id: Document ID

        Returns:
            Attachment name of first PDF or txt found, or None
        """
        attachments = self.list_attachments(database, doc_id)

        # First pass: look for PDFs
        for name, info in attachments.items():
            content_type = info.get('content_type', '')
            if 'pdf' in content_type.lower() or name.lower().endswith('.pdf'):
                if self.verbosity >= 2:
                    print(f"Found PDF attachment: {name} ({content_type})")
                return name

        # Second pass: look for .txt files if no PDF found
        for name, info in attachments.items():
            content_type = info.get('content_type', '')
            if 'text' in content_type.lower() or name.lower().endswith('.txt'):
                if self.verbosity >= 2:
                    print(f"Found text attachment: {name} ({content_type})")
                return name

        return None

    def pdf_to_text(
        self,
        pdf_data: bytes,
        use_layout: bool = True
    ) -> str:
        """
        Convert PDF to text using PyMuPDF (fitz).

        This uses the same approach as jupyter/ist769_skol.ipynb:pdf_to_text,
        working directly with PDF bytes without creating temporary files.

        Args:
            pdf_data: PDF file contents as bytes
            use_layout: Whether to preserve layout (default: True, ignored)

        Returns:
            Extracted text

        Raises:
            ImportError: If PyMuPDF is not installed
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF extraction. "
                "Install with: pip install PyMuPDF"
            )

        # Open PDF document from bytes
        doc = fitz.open(stream=BytesIO(pdf_data), filetype="pdf")

        full_text = ''
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text with whitespace preservation and dehyphenation
            text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_DEHYPHENATE
            )
            full_text += f"\n--- PDF Page {page_num+1} Label {page.get_label()} ---\n"
            full_text += text

        if self.verbosity >= 2:
            print(f"Extracted {len(full_text)} characters from PDF")

        return full_text

    def txt_to_text_with_pages(self, txt_data: bytes) -> str:
        """
        Process text attachment, replacing form feeds with page markers.

        Form feed characters (^L, ASCII 12) are replaced with page number
        annotations in the format "--- PDF Page N Label L ---" to match PDF output.

        Args:
            txt_data: Text file contents as bytes

        Returns:
            Processed text with page markers

        Example:
            Input: "Page 1 text\\fPage 2 text\\fPage 3 text"
            Output: "--- PDF Page 1 Label i ---\\nPage 1 text\\n--- PDF Page 2 Label ii ---\\nPage 2 text\\n--- PDF Page 3 Label iii ---\\nPage 3 text"
        """
        # Decode bytes to string
        try:
            text = txt_data.decode('utf-8')
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            text = txt_data.decode('latin-1', errors='replace')

        # Split on form feed characters
        pages = text.split('\f')

        # Add page markers with proper spacing
        full_text = ''
        for page_num, page_content in enumerate(pages, start=1):
            # Add page marker
            if page_num > 1:
                # Ensure clean separation between pages
                if not full_text.endswith('\n'):
                    full_text += '\n'
                full_text += '\n'
            pdf_label = "i"  # Default label; could be improved to match actual labels
            full_text += f"--- PDF Page {page_num} Label {pdf_label} ---\n"
            full_text += page_content

        if self.verbosity >= 2:
            print(f"Extracted {len(full_text)} characters from text file ({len(pages)} pages)")

        return full_text

    def _is_likely_header(
        self,
        line: str,
        next_line: Optional[str] = None
    ) -> bool:
        """
        Determine if a line is likely a section header.

        Args:
            line: Line to check
            next_line: Next line (for context)

        Returns:
            True if line appears to be a header
        """
        text = line.strip()

        # Empty lines are not headers
        if not text:
            return False

        # Common header keywords
        header_keywords = [
            'introduction', 'abstract', 'key words', 'keywords', 'taxonomy',
            'materials and methods', 'methods', 'results', 'discussion',
            'acknowledgments', 'acknowledgements', 'references', 'conclusion',
            'description', 'etymology', 'specimen', 'holotype', 'paratype',
            'literature cited', 'background', 'objectives', 'summary',
            'figures', 'tables', 'appendix', 'supplementary'
        ]

        text_lower = text.lower()

        # Standalone keyword headers
        if any(text_lower == kw or text_lower.startswith(kw) for kw in header_keywords):
            return True

        # Short lines that are title case or all caps (but not too short)
        if 3 < len(text) < 100:
            # Check if it's centered (lots of leading spaces)
            if len(line) - len(line.lstrip()) > 10:
                return True

            # All caps or title case
            if text.isupper():
                # Not a sentence (doesn't end with period or has very few words)
                if not text.endswith('.') or text.count(' ') <= 5:
                    return True

            # Title case (first letter of most words capitalized)
            words = text.split()
            if len(words) <= 10 and sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.7:
                # Doesn't end with sentence-ending punctuation
                if not text.endswith(('.', ',', ';', ':')):
                    return True

        return False

    def _is_blank_or_whitespace(self, line: str) -> bool:
        """Check if line is blank or only whitespace."""
        return not line.strip()

    def _get_pdf_page_marker(self, line: str) -> Optional[re.Match]:
        """
        Check if line is a PDF page marker and extract page number.

        Args:
            line: Line to check

        Returns:
            Match object with page number in group(1), or None
        """
        return re.match(pdf_page_pattern, line.strip())

    def _extract_empirical_page_number(self, lines_buffer: List[str]) -> Optional[int]:
        """
        Extract empirical page number from first/last lines of a page.

        Looks for patterns like:
        - "485"
        - "486 ... Wang"
        - "485–489"

        Args:
            lines_buffer: List of lines from start or end of page

        Returns:
            Page number as integer, or None if not found
        """
        for line in lines_buffer:
            text = line.strip()

            # Pattern 1: Just a number (possibly with surrounding text)
            # e.g., "485" or "485 ... Wang" or "Page 485"
            match = re.search(r'\b(\d{1,4})\b', text)
            if match:
                page_num = int(match.group(1))
                # Reasonable page number range
                if 1 <= page_num <= 9999:
                    return page_num

        return None

    def _is_page_number_line(self, line: str) -> bool:
        """
        Check if a line appears to be a page number line.

        Page number lines typically:
        - Are very short (< 50 chars)
        - Contain mostly numbers
        - Match patterns like "485", "486 ... Author", "485–489"

        Args:
            line: Line to check

        Returns:
            True if line appears to be a page number
        """
        text = line.strip()

        # Empty lines are not page numbers
        if not text:
            return False

        # Too long to be a page number line
        if len(text) > 50:
            return False

        # Pattern: mostly numbers with optional separators/ellipsis
        # e.g., "485", "486 ... Wang", "485–489"
        if re.match(r'^\d+(\s*[–—-]\s*\d+)?(\s+\.{2,}\s+\S+)?$', text):
            return True

        # Pattern: Page number with optional text
        # e.g., "Page 485", "p. 485"
        if re.match(r'^(Page|p\.?|pp\.?)\s+\d+', text, re.IGNORECASE):
            return True

        return False

    def _get_section_name(self, header_text: str) -> Optional[str]:
        """
        Extract standardized section name from header text.

        Args:
            header_text: Header text to analyze

        Returns:
            Standardized section name, or None if not a known section
        """
        text_lower = header_text.strip().lower()

        # Map of section keywords to standardized names
        section_map = {
            'introduction': 'Introduction',
            'abstract': 'Abstract',
            'key words': 'Keywords',
            'keywords': 'Keywords',
            'taxonomy': 'Taxonomy',
            'materials and methods': 'Materials and Methods',
            'methods': 'Methods',
            'results': 'Results',
            'discussion': 'Discussion',
            'acknowledgments': 'Acknowledgments',
            'acknowledgements': 'Acknowledgments',
            'references': 'References',
            'conclusion': 'Conclusion',
            'description': 'Description',
            'etymology': 'Etymology',
            'specimen': 'Specimen',
            'holotype': 'Holotype',
            'paratype': 'Paratype',
            'literature cited': 'Literature Cited',
            'background': 'Background',
            'objectives': 'Objectives',
            'summary': 'Summary',
            'figures': 'Figures',
            'tables': 'Tables',
            'appendix': 'Appendix',
            'supplementary': 'Supplementary'
        }

        # Check for exact matches or starts with
        for keyword, standard_name in section_map.items():
            if text_lower == keyword or text_lower.startswith(keyword):
                return standard_name

        return None

    def _is_figure_caption(self, text: str) -> bool:
        """
        Check if text appears to be a figure caption.

        Detects patterns like:
        - "Fig. 1."
        - "Figure 1:"
        - "Fig 1A."
        - "FIG. 1."

        Args:
            text: Text to check

        Returns:
            True if text appears to be a figure caption
        """
        text_stripped = text.strip()

        # Pattern: Fig/Figure followed by number/letter
        # Matches: "Fig. 1", "Figure 1:", "Fig 1A.", "FIG. 1."
        pattern = r'^(Fig\.?|Figure|FIG\.?)\s*\d+[A-Za-z]?[\.:,\s]'

        return bool(re.match(pattern, text_stripped))

    def _extract_figure_number(self, caption_text: str) -> Optional[str]:
        """
        Extract figure number from caption text.

        Args:
            caption_text: Figure caption text

        Returns:
            Figure number/identifier (e.g., "1", "2A", "3B") or None
        """
        # Pattern to extract figure number
        pattern = r'^(?:Fig\.?|Figure|FIG\.?)\s*(\d+[A-Za-z]?)'
        match = re.match(pattern, caption_text.strip())

        if match:
            return match.group(1)

        return None

    def _parse_yedda_annotations(self, text: str) -> Dict[int, str]:
        """
        Parse YEDDA annotations from text and create line-to-label mapping.

        YEDDA format: [@ text content #Label*]
        Annotations can nest, in which case the innermost label takes precedence.
        Annotations can span multiple lines.

        Args:
            text: Full text with potential YEDDA annotations

        Returns:
            Dictionary mapping line numbers (1-indexed) to labels

        Example:
            >>> text = "[@ Line 1\\nLine 2\\n#Label1*]\\nLine 3\\n[@ Line 4\\n#Label2*]"
            >>> mapping = extractor._parse_yedda_annotations(text)
            >>> # Returns: {1: 'Label1', 2: 'Label1', 4: 'Label2'}
        """
        line_to_label = {}
        lines = text.split('\n')

        # Track annotation boundaries with labels
        # Stack of (start_line, label) tuples for nested annotations
        annotation_stack = []

        # Pattern to match YEDDA annotation start: [@
        start_pattern = r'\[@'
        # Pattern to match YEDDA annotation end: #Label*]
        end_pattern = r'#([^*]+)\*\]'

        for line_idx, line in enumerate(lines):
            line_number = line_idx + 1  # 1-indexed

            # Check for annotation starts
            start_matches = list(re.finditer(start_pattern, line))
            for match in start_matches:
                # Push onto stack (we don't know the label yet)
                annotation_stack.append({'start_line': line_number, 'label': None})

            # Check for annotation ends
            end_matches = list(re.finditer(end_pattern, line))
            for match in end_matches:
                label = match.group(1).strip()

                # Pop from stack and assign label
                if annotation_stack:
                    annotation = annotation_stack.pop()
                    annotation['label'] = label
                    annotation['end_line'] = line_number

                    # Now assign this label to all lines in the range
                    for ln in range(annotation['start_line'], annotation['end_line'] + 1):
                        # Innermost label wins - only set if not already set by nested annotation
                        if ln not in line_to_label:
                            line_to_label[ln] = label

        return line_to_label

    def parse_text_to_sections(
        self,
        text: str,
        doc_id: str,
        attachment_name: str,
        min_paragraph_length: int = 10
    ):
        """
        Parse extracted text into section headers and paragraphs.

        Returns a PySpark DataFrame with columns:
        - value: Section/paragraph text (with YEDDA annotations preserved)
        - doc_id: Document ID
        - attachment_name: Name of the PDF attachment
        - paragraph_number: Sequential paragraph number within the attachment
        - line_number: Line number of the first line of the section
        - pdf_page: PDF page number from page markers
        - empirical_page_number: Page number extracted from document itself
        - section_name: Standardized section name (e.g., "Introduction", "Methods")

        Figure captions (e.g., "Fig. 1. Description") are automatically detected
        and excluded from the DataFrame. Access them via get_figure_captions().

        YEDDA annotations in format [@ text #Label*] are PRESERVED in the value field.
        Label extraction is delegated to AnnotatedTextParser for consistency with
        other extraction modes.

        Args:
            text: Extracted text from PDF
            doc_id: Document ID
            attachment_name: Attachment name
            min_paragraph_length: Minimum characters for a paragraph

        Returns:
            PySpark DataFrame with sections and metadata (figure captions excluded)

        Raises:
            ImportError: If PySpark is not available
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError(
                "PySpark is required for DataFrame output. "
                "Install with: pip install pyspark"
            )

        if self.spark is None:
            raise ValueError(
                "SparkSession is required for DataFrame output. "
                "Pass spark parameter to PDFSectionExtractor.__init__()"
            )

        lines = text.split('\n')

        # YEDDA label extraction is delegated to AnnotatedTextParser for consistency.
        # PDFSectionExtractor focuses on structural extraction (sections, pages, layout).

        # First pass: identify page boundaries and extract empirical page numbers
        page_boundaries = []  # List of (pdf_page_num, start_line_idx, end_line_idx)
        current_pdf_page = 1
        current_pdf_label: Optional[str] = None
        page_start_idx = 0

        for i, line in enumerate(lines):
            page_marker = self._get_pdf_page_marker(line)
            if page_marker:
                # End of previous page
                if i > page_start_idx:
                    page_boundaries.append((current_pdf_page, page_start_idx, i - 1))
                # Start of new page
                current_pdf_page = int(page_marker.group(1))
                current_pdf_label = page_marker.group(3)  # May be None
                page_start_idx = i + 1

        # Add last page
        if page_start_idx < len(lines):
            page_boundaries.append((current_pdf_page, page_start_idx, len(lines) - 1))

        # Extract empirical page numbers for each page
        empirical_page_map = {}  # pdf_page_num -> empirical_page_num
        for pdf_page, start_idx, end_idx in page_boundaries:
            # Check first 5 and last 5 lines of page
            first_lines = [lines[i] for i in range(start_idx, min(start_idx + 5, end_idx + 1))]
            last_lines = [lines[i] for i in range(max(end_idx - 4, start_idx), end_idx + 1)]

            empirical_num = self._extract_empirical_page_number(first_lines + last_lines)
            if empirical_num:
                empirical_page_map[pdf_page] = empirical_num

        # Second pass: parse sections with metadata
        # Clear figure captions from previous extraction
        self.figure_captions = []

        records = []
        current_paragraph = []
        current_paragraph_start_line = None
        current_pdf_page = 1
        current_pdf_label = None
        current_section_name = None
        paragraph_number = 0

        for i, line in enumerate(lines):
            line_number = i + 1  # 1-indexed
            next_line = lines[i+1] if i+1 < len(lines) else None

            # Check if this is a PDF page marker
            page_marker = self._get_pdf_page_marker(line)
            if page_marker:
                current_pdf_page = int(page_marker.group(1))
                current_pdf_label = page_marker.group(3)  # May be None
                continue

            # Skip page number lines
            if self._is_page_number_line(line):
                continue

            # Skip completely blank lines at the start (before any content)
            if not records and not current_paragraph and self._is_blank_or_whitespace(line):
                continue

            # Check if this is a header
            if self._is_likely_header(line, next_line):
                # Save current paragraph if exists
                if current_paragraph:
                    para_text = ' '.join(current_paragraph).strip()
                    if len(para_text) >= min_paragraph_length:
                        # Check if this is a figure caption
                        if self._is_figure_caption(para_text):
                            # Store as figure caption instead of regular paragraph
                            figure_num = self._extract_figure_number(para_text)
                            self.figure_captions.append({
                                'figure_number': figure_num,
                                'caption': para_text,
                                'doc_id': doc_id,
                                'attachment_name': attachment_name,
                                'line_number': current_paragraph_start_line,
                                'pdf_page': current_pdf_page,
                                'pdf_label': current_pdf_label,
                                'empirical_page_number': empirical_page_map.get(current_pdf_page),
                                'section_name': current_section_name
                            })
                        else:
                            # Regular paragraph
                            paragraph_number += 1
                            records.append({
                                'value': para_text,
                                'doc_id': doc_id,
                                'attachment_name': attachment_name,
                                'paragraph_number': paragraph_number,
                                'line_number': current_paragraph_start_line,
                                'pdf_page': current_pdf_page,
                                'pdf_label': current_pdf_label,
                                'section_name': current_section_name,
                            })
                    current_paragraph = []
                    current_paragraph_start_line = None

                # Add header
                header_text = line.strip()
                if header_text:
                    # Update current section name
                    section_name = self._get_section_name(header_text)
                    if section_name:
                        current_section_name = section_name

                    paragraph_number += 1
                    records.append({
                        'value': header_text,
                        'doc_id': doc_id,
                        'attachment_name': attachment_name,
                        'paragraph_number': paragraph_number,
                        'line_number': line_number,
                        'pdf_page': current_pdf_page,
                        'pdf_label': current_pdf_label,
                        'empirical_page_number': empirical_page_map.get(current_pdf_page),
                        'section_name': section_name,  # For headers, use the section name if detected
                    })

            # Blank line indicates paragraph break
            elif self._is_blank_or_whitespace(line):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph).strip()
                    if len(para_text) >= min_paragraph_length:
                        # Check if this is a figure caption
                        if self._is_figure_caption(para_text):
                            # Store as figure caption instead of regular paragraph
                            figure_num = self._extract_figure_number(para_text)
                            self.figure_captions.append({
                                'figure_number': figure_num,
                                'caption': para_text,
                                'doc_id': doc_id,
                                'attachment_name': attachment_name,
                                'line_number': current_paragraph_start_line,
                                'pdf_page': current_pdf_page,
                                'pdf_label': current_pdf_label,
                                'empirical_page_number': empirical_page_map.get(current_pdf_page),
                                'section_name': current_section_name
                            })
                        else:
                            # Regular paragraph
                            paragraph_number += 1
                            records.append({
                                'value': para_text,
                                'doc_id': doc_id,
                                'attachment_name': attachment_name,
                                'paragraph_number': paragraph_number,
                                'line_number': current_paragraph_start_line,
                                'pdf_page': current_pdf_page,
                                'pdf_label': current_pdf_label,
                                'empirical_page_number': empirical_page_map.get(current_pdf_page),
                                'section_name': current_section_name,
                            })
                    current_paragraph = []
                    current_paragraph_start_line = None

            # Regular content line
            else:
                text_content = line.strip()
                if text_content:
                    if current_paragraph_start_line is None:
                        current_paragraph_start_line = line_number
                    current_paragraph.append(text_content)

        # Don't forget the last paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph).strip()
            if len(para_text) >= min_paragraph_length:
                # Check if this is a figure caption
                if self._is_figure_caption(para_text):
                    # Store as figure caption instead of regular paragraph
                    figure_num = self._extract_figure_number(para_text)
                    self.figure_captions.append({
                        'figure_number': figure_num,
                        'caption': para_text,
                        'doc_id': doc_id,
                        'attachment_name': attachment_name,
                        'line_number': current_paragraph_start_line,
                        'pdf_label': current_pdf_label,
                        'pdf_page': current_pdf_page,
                        'section_name': current_section_name
                    })
                else:
                    # Regular paragraph
                    paragraph_number += 1
                    records.append({
                        'value': para_text,
                        'doc_id': doc_id,
                        'attachment_name': attachment_name,
                        'paragraph_number': paragraph_number,
                        'line_number': current_paragraph_start_line,
                        'empirical_page_number': empirical_page_map.get(current_pdf_page),
                        'pdf_page': current_pdf_page,
                        'pdf_label': current_pdf_label,
                        'section_name': current_section_name,
                    })

        if self.verbosity >= 1:
            print(f"Parsed {len(records)} sections/paragraphs")
            if self.figure_captions:
                print(f"Extracted {len(self.figure_captions)} figure captions")
            if empirical_page_map:
                print(f"Extracted empirical page numbers: {empirical_page_map}")

        # Create DataFrame with explicit schema
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType

        schema = StructType([
            StructField("value", StringType(), False),
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("paragraph_number", IntegerType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("pdf_page", IntegerType(), False),
            StructField("empirical_page_number", IntegerType(), True), # Nullable
            StructField("pdf_page", IntegerType(), True),  # Nullable
            StructField("pdf_label", StringType(), True),  # Nullable
            StructField("section_name", StringType(), True),  # Nullable
        ])

        df = self.spark.createDataFrame(records, schema=schema)
        return df

    def extract_from_document(
        self,
        database: str,
        doc_id: str,
        attachment_name: Optional[str] = None,
        cleanup: bool = True
    ) -> DataFrame:
        """
        Extract sections from a PDF or text attachment in a CouchDB document.

        This is the main convenience method that handles the entire pipeline:
        1. Find PDF or text attachment
        2. Get attachment data directly from CouchDB (no temp file)
        3. Extract/process text (PDFs via PyMuPDF, txt files with form feed replacement)
        4. Parse into sections DataFrame

        For .txt files, form feed characters (^L) are replaced with page markers
        to maintain compatibility with PDF processing.

        Text attachment behavior (controlled by read_text and save_text):
        - save_text='eager' + read_text=True: Always convert PDF and replace .txt
        - save_text='eager' + read_text=False: Always convert PDF and save/replace .txt
        - save_text='lazy' + read_text=True: Read .txt if exists, else convert PDF and save .txt
        - save_text='lazy' + read_text=False: Convert PDF and save .txt only if it doesn't exist
        - save_text=None + read_text=True: Read .txt if exists, else convert PDF (don't save)
        - save_text=None + read_text=False: Convert PDF without saving (original behavior)

        Args:
            database: Database name
            doc_id: Document ID
            attachment_name: PDF or txt attachment name (auto-detected if None)
            cleanup: Deprecated (kept for API compatibility, has no effect)

        Returns:
            PySpark DataFrame with columns:
            - value: Section/paragraph text
            - doc_id: Document ID
            - attachment_name: Attachment name
            - paragraph_number: Sequential paragraph number
            - line_number: Line number of first line
            - pdf_page: PDF page number (from page markers)
            - empirical_page_number: Page number extracted from document (nullable)
            - pdf_label: Human-readable label from PDF page marker (nullable)
            - section_name: Standardized section name like "Introduction" (nullable)
            - label: YEDDA annotation label active at first line (nullable)

        Raises:
            ValueError: If no PDF or text attachment found
            ImportError: If PyMuPDF is not installed (for PDFs) or PySpark is not installed
        """
        if self.verbosity >= 1:
            print(f"\nExtracting sections from document {doc_id} in {database}")

        # Find PDF or text attachment if not specified
        if attachment_name is None:
            attachment_name = self.find_pdf_attachment(database, doc_id)
            if attachment_name is None:
                attachments = self.list_attachments(database, doc_id)
                raise ValueError(
                    f"No PDF or text attachment found in document {doc_id}. "
                    f"Available attachments: {list(attachments.keys())}"
                )

        # Determine text attachment name for read_text/save_text functionality
        txt_attachment_name = self._get_text_attachment_name(attachment_name)
        txt_exists = self._text_attachment_exists(database, doc_id, txt_attachment_name)

        text = None

        # Decision logic for read_text and save_text
        if self.save_text == 'eager':
            # Always convert PDF and save/replace .txt
            if self.verbosity >= 1:
                print(f"save_text='eager': Converting PDF and will save/replace text attachment")
            file_data = self._attachment_from_couch(
                database, doc_id, attachment_name
            )
            text = self.pdf_to_text(file_data)
            # Save the text attachment
            self._save_text_attachment(database, doc_id, txt_attachment_name, text)

        elif self.save_text == 'lazy':
            # Save .txt only if it doesn't exist
            if txt_exists and self.read_text:
                # .txt exists and we want to read it
                text = self._read_text_attachment(database, doc_id, txt_attachment_name)
            else:
                # .txt doesn't exist, or we're not reading - convert PDF
                if self.verbosity >= 1:
                    if txt_exists:
                        print(f"save_text='lazy', read_text=False: Converting PDF (ignoring existing .txt)")
                    else:
                        print(f"save_text='lazy': Text attachment doesn't exist, converting PDF and saving")
                file_data = self._attachment_from_couch(
                    database, doc_id, attachment_name,
                )
                text = self.pdf_to_text(file_data)
                # Save only if doesn't exist
                if not txt_exists:
                    self._save_text_attachment(database, doc_id, txt_attachment_name, text)

        elif self.save_text is None:
            # Don't save .txt attachment
            if self.read_text and txt_exists:
                # Read from .txt if exists
                text = self._read_text_attachment(database, doc_id, txt_attachment_name)
            else:
                # Convert PDF without saving
                if self.read_text and not txt_exists:
                    if self.verbosity >= 1:
                        print(f"Text attachment {txt_attachment_name} not found, converting PDF")
                file_data = self._attachment_from_couch(
                    database, doc_id, attachment_name
                )
                # Extract text from bytes (method depends on file type)
                if attachment_name.lower().endswith('.pdf'):
                    text = self.pdf_to_text(file_data)
                elif attachment_name.lower().endswith('.txt'):
                    text = self.txt_to_text_with_pages(file_data)
                else:
                    # Try to detect based on content
                    # If it looks like PDF magic bytes, treat as PDF
                    if file_data[:4] == b'%PDF':
                        text = self.pdf_to_text(file_data)
                    else:
                        # Default to text processing
                        text = self.txt_to_text_with_pages(file_data)

        # Parse into sections DataFrame
        sections_df = self.parse_text_to_sections(text, doc_id, attachment_name)

        return sections_df

    def extract_from_multiple_documents(
        self,
        database: str,
        doc_ids: List[str],
        attachment_name: Optional[str] = None
    ):
        """
        Extract sections from multiple documents.

        Args:
            database: Database name
            doc_ids: List of document IDs
            attachment_name: PDF attachment name (auto-detected if None)

        Returns:
            Combined PySpark DataFrame with all sections from all documents

        Raises:
            ImportError: If PySpark is not available
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError(
                "PySpark is required for DataFrame output. "
                "Install with: pip install pyspark"
            )

        dfs = []
        for doc_id in doc_ids:
            try:
                sections_df = self.extract_from_document(
                    database, doc_id, attachment_name
                )
                dfs.append(sections_df)
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"Error extracting from {doc_id}: {e}")

        if not dfs:
            # Return empty DataFrame with correct schema
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType

            schema = StructType([
                StructField("value", StringType(), False),
                StructField("doc_id", StringType(), False),
                StructField("attachment_name", StringType(), False),
                StructField("paragraph_number", IntegerType(), False),
                StructField("line_number", IntegerType(), False),
                StructField("pdf_page", IntegerType(), False),
                StructField("empirical_page_number", IntegerType(), True),
                StructField("section_name", StringType(), True)
            ])
            return self.spark.createDataFrame([], schema=schema)

        # Union all DataFrames using batched unions to avoid deep query plans
        from functools import reduce
        from pyspark.sql import DataFrame

        # Batch the unions to create a balanced tree instead of a skewed one
        # This limits plan depth to ~log(N) instead of N
        if self.verbosity >= 2:
            print(f"[PDFSectionExtractor] Unioning {len(dfs)} DataFrames in batches of {self.union_batch_size}")

        batch_size = self.union_batch_size
        batched_dfs = []

        for i in range(0, len(dfs), batch_size):
            batch = dfs[i:i+batch_size]
            if batch:
                if len(batch) == 1:
                    batch_df = batch[0]
                else:
                    batch_df = reduce(DataFrame.unionAll, batch)

                # Checkpoint each batch to completely break lineage
                # Unlike cache(), checkpoint() severs the logical plan connection
                # This prevents query planner OOM when building complex trees

                # Materialize first to avoid checkpointing lazy operations
                row_count = batch_df.count()

                # Use localCheckpoint (in-memory) instead of checkpoint (disk)
                # This is faster but still breaks lineage for the query planner
                batch_df = batch_df.localCheckpoint(eager=True)

                batched_dfs.append(batch_df)

                if self.verbosity >= 2:
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(dfs) + batch_size - 1) // batch_size
                    print(f"[PDFSectionExtractor] Checkpointed batch {batch_num}/{total_batches}: {len(batch)} docs, {row_count} rows")

        # Union the batches
        if len(batched_dfs) == 1:
            combined_df = batched_dfs[0]
        else:
            if self.verbosity >= 2:
                print(f"[PDFSectionExtractor] Final union of {len(batched_dfs)} batches")
            combined_df = reduce(DataFrame.unionAll, batched_dfs)

            # Checkpoint the final union to break lineage
            combined_df = combined_df.localCheckpoint(eager=True)

        if self.verbosity >= 2:
            print(f"[PDFSectionExtractor] Union complete, checkpointed result")

        return combined_df

    def _get_text_attachment_name(self, pdf_attachment_name: str) -> str:
        """
        Convert PDF attachment name to text attachment name.

        Args:
            pdf_attachment_name: PDF attachment name (e.g., "foo.pdf")

        Returns:
            Text attachment name (e.g., "foo.txt")

        Example:
            >>> extractor._get_text_attachment_name("article.pdf")
            'article.txt'
            >>> extractor._get_text_attachment_name("foo.PDF")
            'foo.txt'
        """
        # Replace .pdf extension with .txt (case-insensitive)
        if pdf_attachment_name.lower().endswith('.pdf'):
            return pdf_attachment_name[:-4] + '.txt'
        # If no .pdf extension, just append .txt
        return pdf_attachment_name + '.txt'

    def _text_attachment_exists(self, database: str, doc_id: str, txt_attachment_name: str) -> bool:
        """
        Check if a text attachment exists in the document.

        Args:
            database: Database name
            doc_id: Document ID
            txt_attachment_name: Text attachment name to check

        Returns:
            True if attachment exists, False otherwise
        """
        attachments = self.list_attachments(database, doc_id)
        return txt_attachment_name in attachments

    def _read_text_attachment(self, database: str, doc_id: str, txt_attachment_name: str) -> str:
        """
        Read text from a text attachment.

        Args:
            database: Database name
            doc_id: Document ID
            txt_attachment_name: Text attachment name

        Returns:
            Text content with page markers
        """
        file_data = self._attachment_from_couch(
            database, doc_id, txt_attachment_name
        )
        # Use the existing method to process text with form feeds
        return self.txt_to_text_with_pages(file_data)

    def _attachment_from_couch(self, database: str, doc_id: str, attachment_name: str) -> bytes:
        db = self.couch[database]
        result: bytes = db.get_attachment(doc_id, attachment_name).read()
        if self.verbosity >= 1:
            print(f"Retrieved attachment: {attachment_name} ({len(result):,} bytes)")
        return result

    def _save_text_attachment(
        self,
        database: str,
        doc_id: str,
        txt_attachment_name: str,
        text_content: str
    ) -> None:
        """
        Save text as a text attachment to the document.

        If the attachment already exists, it will be replaced.

        Args:
            database: Database name
            doc_id: Document ID
            txt_attachment_name: Text attachment name
            text_content: Text content to save
        """
        db = self.couch[database]

        # Get current document to access _rev
        doc = db[doc_id]

        # Save as text attachment
        text_bytes = text_content.encode('utf-8')
        db.put_attachment(
            doc,
            text_bytes,
            filename=txt_attachment_name,
            content_type='text/plain'
        )

        if self.verbosity >= 1:
            print(f"Saved text attachment: {txt_attachment_name} ({len(text_bytes):,} bytes)")

    def get_figure_captions(self) -> List[Dict[str, Any]]:
        """
        Get all figure captions extracted from the last document processing.

        Figure captions are automatically detected and excluded from the main
        DataFrame during parsing. This method provides access to them.

        Returns:
            List of dictionaries, each containing:
            - figure_number: Extracted figure number (e.g., "1", "2A", "3B")
            - caption: Full caption text
            - doc_id: Document ID
            - attachment_name: PDF attachment name
            - line_number: Line number of the caption
            - pdf_page: PDF page number
            - empirical_page_number: Document page number (nullable)
            - section_name: Section where the caption appears (nullable)

        Example:
            >>> extractor = PDFSectionExtractor(spark=spark)
            >>> sections_df = extractor.extract_from_document('db', 'doc_id')
            >>> captions = extractor.get_figure_captions()
            >>> for caption in captions:
            ...     print(f"Figure {caption['figure_number']}: {caption['caption'][:50]}...")
        """
        return self.figure_captions

    def get_section_by_keyword(
        self,
        sections: List[str],
        keyword: str,
        case_sensitive: bool = False
    ) -> List[str]:
        """
        Get all sections that contain a keyword.

        Args:
            sections: List of sections
            keyword: Keyword to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matching sections
        """
        if case_sensitive:
            return [
                s for s in sections
                if keyword in s
            ]
        keyword = keyword.lower()
        return [
            s for s in sections
            if keyword in s.lower()
        ]

    def extract_metadata(
        self,
        sections: List[str]
    ) -> Dict[str, Any]:
        """
        Extract common metadata from sections.

        Args:
            sections: List of sections

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': [],
            'sections_found': []
        }

        # Common section headers to track
        section_headers = [
            'Introduction', 'Methods', 'Results', 'Discussion',
            'Conclusion', 'References', 'Acknowledgments', 'Abstract'
        ]

        for i, section in enumerate(sections):
            text = section.strip()
            text_lower = text.lower()

            # Potential title (usually one of first few lines, title case)
            if i < 5 and metadata['title'] is None:
                if len(text) < 200 and text[0].isupper() and not text_lower.startswith('abstract'):
                    metadata['title'] = text

            # Abstract
            if text_lower.startswith('abstract'):
                # Get following paragraphs until next section
                abstract_parts = [text]
                for j in range(i+1, min(i+5, len(sections))):
                    next_text = sections[j].strip()
                    if any(next_text.lower().startswith(h.lower()) for h in section_headers):
                        break
                    abstract_parts.append(next_text)
                metadata['abstract'] = ' '.join(abstract_parts)

            # Keywords
            if 'key words' in text_lower or 'keywords' in text_lower:
                # Extract keywords after the header
                kw_text = re.sub(r'^.*?key\s*words?\s*[—:-]\s*', '', text, flags=re.IGNORECASE)
                keywords = [k.strip() for k in re.split(r'[,;]', kw_text)]
                metadata['keywords'] = [k for k in keywords if k]

            # Track section headers found
            for header in section_headers:
                if text.lower() == header.lower():
                    metadata['sections_found'].append(header)

        return metadata


# Example usage
if __name__ == '__main__':
    # Initialize Spark
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("PDFSectionExtractor") \
        .getOrCreate()

    # Initialize extractor with SparkSession
    extractor = PDFSectionExtractor(verbosity=2, spark=spark)

    # Example: Extract from the Arachnopeziza paper
    try:
        sections_df = extractor.extract_from_document(
            database='skol_dev',
            doc_id='00df9554e9834283b5e844c7a994ba5f',
            attachment_name='article.pdf'
        )

        print("\n" + "="*70)
        print("EXTRACTION RESULTS")
        print("="*70)
        print(f"Total sections: {sections_df.count()}\n")

        # Show schema
        print("DataFrame schema:")
        sections_df.printSchema()

        # Show first 5 sections
        print("\nFirst 5 sections:")
        sections_df.select("paragraph_number", "value", "pdf_page", "line_number") \
            .show(5, truncate=80, vertical=False)

        # Example queries
        print("\nSections on page 1:")
        sections_df.filter(sections_df.pdf_page == 1).select("paragraph_number", "value").show(5, truncate=60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
