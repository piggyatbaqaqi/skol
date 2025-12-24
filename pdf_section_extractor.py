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
import tempfile
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
        spark: Optional[Any] = None
    ):
        """
        Initialize the PDF section extractor.

        Args:
            couchdb_url: CouchDB server URL (default: from COUCHDB_URL env var)
            username: CouchDB username (default: from COUCHDB_USER env var)
            password: CouchDB password (default: from COUCHDB_PASSWORD env var)
            verbosity: Logging verbosity (0=silent, 1=info, 2=debug)
            spark: SparkSession instance (required for DataFrame output)
        """
        self.verbosity = verbosity
        self.spark = spark

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
            full_text += f"\n--- PDF Page {page_num+1} ---\n"
            full_text += text

        if self.verbosity >= 2:
            print(f"Extracted {len(full_text)} characters from PDF")

        return full_text

    def txt_to_text_with_pages(self, txt_data: bytes) -> str:
        """
        Process text attachment, replacing form feeds with page markers.

        Form feed characters (^L, ASCII 12) are replaced with page number
        annotations in the format "--- PDF Page N ---" to match PDF output.

        Args:
            txt_data: Text file contents as bytes

        Returns:
            Processed text with page markers

        Example:
            Input: "Page 1 text\\fPage 2 text\\fPage 3 text"
            Output: "--- PDF Page 1 ---\\nPage 1 text\\n--- PDF Page 2 ---\\nPage 2 text\\n--- PDF Page 3 ---\\nPage 3 text"
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
            full_text += f"--- PDF Page {page_num} ---\n"
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

    def _get_pdf_page_marker(self, line: str):
        """
        Check if line is a PDF page marker and extract page number.

        Args:
            line: Line to check

        Returns:
            Match object with page number in group(1), or None
        """
        pattern = r'^---\s*PDF\s+Page\s+(\d+)\s*---\s*$'
        return re.match(pattern, line.strip())

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
        - page_number: PDF page number from page markers
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
        page_start_idx = 0

        for i, line in enumerate(lines):
            page_marker = self._get_pdf_page_marker(line)
            if page_marker:
                # End of previous page
                if i > page_start_idx:
                    page_boundaries.append((current_pdf_page, page_start_idx, i - 1))
                # Start of new page
                current_pdf_page = int(page_marker.group(1))
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
        current_page_number = 1
        current_section_name = None
        paragraph_number = 0

        for i, line in enumerate(lines):
            line_number = i + 1  # 1-indexed
            next_line = lines[i+1] if i+1 < len(lines) else None

            # Check if this is a PDF page marker
            page_marker = self._get_pdf_page_marker(line)
            if page_marker:
                current_page_number = int(page_marker.group(1))
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
                                'page_number': current_page_number,
                                'empirical_page_number': empirical_page_map.get(current_page_number),
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
                                'page_number': current_page_number,
                                'empirical_page_number': empirical_page_map.get(current_page_number),
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
                        'page_number': current_page_number,
                        'empirical_page_number': empirical_page_map.get(current_page_number),
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
                                'page_number': current_page_number,
                                'empirical_page_number': empirical_page_map.get(current_page_number),
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
                                'page_number': current_page_number,
                                'empirical_page_number': empirical_page_map.get(current_page_number),
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
                        'page_number': current_page_number,
                        'empirical_page_number': empirical_page_map.get(current_page_number),
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
                        'page_number': current_page_number,
                        'empirical_page_number': empirical_page_map.get(current_page_number),
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
            StructField("page_number", IntegerType(), False),
            StructField("empirical_page_number", IntegerType(), True),  # Nullable
            StructField("section_name", StringType(), True)  # Nullable
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
            - page_number: PDF page number (from page markers)
            - empirical_page_number: Page number extracted from document (nullable)
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

        # Get attachment data directly from CouchDB (no temp file)
        db = self.couch[database]
        file_data = db.get_attachment(doc_id, attachment_name).read()

        if self.verbosity >= 1:
            print(f"Retrieved attachment: {attachment_name} ({len(file_data):,} bytes)")

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
                StructField("page_number", IntegerType(), False),
                StructField("empirical_page_number", IntegerType(), True),
                StructField("section_name", StringType(), True)
            ])
            return self.spark.createDataFrame([], schema=schema)

        # Union all DataFrames
        from functools import reduce
        from pyspark.sql import DataFrame

        combined_df = reduce(DataFrame.unionAll, dfs)
        return combined_df

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
            - page_number: PDF page number
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
        if not case_sensitive:
            keyword = keyword.lower()
            return [
                s for s in sections
                if keyword in s.lower()
            ]
        else:
            return [
                s for s in sections
                if keyword in s
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
        sections_df.select("paragraph_number", "value", "page_number", "line_number") \
            .show(5, truncate=80, vertical=False)

        # Example queries
        print("\nSections on page 1:")
        sections_df.filter(sections_df.page_number == 1).select("paragraph_number", "value").show(5, truncate=60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
