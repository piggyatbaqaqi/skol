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
        Find the first PDF attachment in a document.

        Args:
            database: Database name
            doc_id: Document ID

        Returns:
            Attachment name of first PDF found, or None
        """
        attachments = self.list_attachments(database, doc_id)

        for name, info in attachments.items():
            content_type = info.get('content_type', '')
            if 'pdf' in content_type.lower() or name.lower().endswith('.pdf'):
                if self.verbosity >= 2:
                    print(f"Found PDF attachment: {name} ({content_type})")
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
        - value: Section/paragraph text
        - doc_id: Document ID
        - attachment_name: Name of the PDF attachment
        - paragraph_number: Sequential paragraph number within the attachment
        - line_number: Line number of the first line of the section
        - page_number: PDF page number from page markers

        Args:
            text: Extracted text from PDF
            doc_id: Document ID
            attachment_name: Attachment name
            min_paragraph_length: Minimum characters for a paragraph

        Returns:
            PySpark DataFrame with sections and metadata

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
        records = []
        current_paragraph = []
        current_paragraph_start_line = None
        current_page_number = 1
        paragraph_number = 0

        for i, line in enumerate(lines):
            line_number = i + 1  # 1-indexed
            next_line = lines[i+1] if i+1 < len(lines) else None

            # Check if this is a PDF page marker
            page_marker = self._get_pdf_page_marker(line)
            if page_marker:
                current_page_number = int(page_marker.group(1))
                continue

            # Skip completely blank lines at the start
            if not records and self._is_blank_or_whitespace(line):
                continue

            # Check if this is a header
            if self._is_likely_header(line, next_line):
                # Save current paragraph if exists
                if current_paragraph:
                    para_text = ' '.join(current_paragraph).strip()
                    if len(para_text) >= min_paragraph_length:
                        paragraph_number += 1
                        records.append({
                            'value': para_text,
                            'doc_id': doc_id,
                            'attachment_name': attachment_name,
                            'paragraph_number': paragraph_number,
                            'line_number': current_paragraph_start_line,
                            'page_number': current_page_number
                        })
                    current_paragraph = []
                    current_paragraph_start_line = None

                # Add header
                header_text = line.strip()
                if header_text:
                    paragraph_number += 1
                    records.append({
                        'value': header_text,
                        'doc_id': doc_id,
                        'attachment_name': attachment_name,
                        'paragraph_number': paragraph_number,
                        'line_number': line_number,
                        'page_number': current_page_number
                    })

            # Blank line indicates paragraph break
            elif self._is_blank_or_whitespace(line):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph).strip()
                    if len(para_text) >= min_paragraph_length:
                        paragraph_number += 1
                        records.append({
                            'value': para_text,
                            'doc_id': doc_id,
                            'attachment_name': attachment_name,
                            'paragraph_number': paragraph_number,
                            'line_number': current_paragraph_start_line,
                            'page_number': current_page_number
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
                paragraph_number += 1
                records.append({
                    'value': para_text,
                    'doc_id': doc_id,
                    'attachment_name': attachment_name,
                    'paragraph_number': paragraph_number,
                    'line_number': current_paragraph_start_line,
                    'page_number': current_page_number
                })

        if self.verbosity >= 1:
            print(f"Parsed {len(records)} sections/paragraphs")

        # Create DataFrame with explicit schema
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType

        schema = StructType([
            StructField("value", StringType(), False),
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("paragraph_number", IntegerType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("page_number", IntegerType(), False)
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
        Extract sections from a PDF attachment in a CouchDB document.

        This is the main convenience method that handles the entire pipeline:
        1. Find PDF attachment
        2. Get PDF data directly from CouchDB (no temp file)
        3. Extract text from bytes
        4. Parse into sections DataFrame

        Args:
            database: Database name
            doc_id: Document ID
            attachment_name: PDF attachment name (auto-detected if None)
            cleanup: Deprecated (kept for API compatibility, has no effect)

        Returns:
            PySpark DataFrame with columns:
            - value: Section/paragraph text
            - doc_id: Document ID
            - attachment_name: Attachment name
            - paragraph_number: Sequential paragraph number
            - line_number: Line number of first line
            - page_number: PDF page number

        Raises:
            ValueError: If no PDF attachment found
            ImportError: If PyMuPDF or PySpark is not installed
        """
        if self.verbosity >= 1:
            print(f"\nExtracting sections from document {doc_id} in {database}")

        # Find PDF attachment if not specified
        if attachment_name is None:
            attachment_name = self.find_pdf_attachment(database, doc_id)
            if attachment_name is None:
                attachments = self.list_attachments(database, doc_id)
                raise ValueError(
                    f"No PDF attachment found in document {doc_id}. "
                    f"Available attachments: {list(attachments.keys())}"
                )

        # Get PDF data directly from CouchDB (no temp file)
        db = self.couch[database]
        pdf_data = db.get_attachment(doc_id, attachment_name).read()

        if self.verbosity >= 1:
            print(f"Retrieved PDF: {attachment_name} ({len(pdf_data):,} bytes)")

        # Extract text from bytes
        text = self.pdf_to_text(pdf_data)

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
                StructField("page_number", IntegerType(), False)
            ])
            return self.spark.createDataFrame([], schema=schema)

        # Union all DataFrames
        from functools import reduce
        from pyspark.sql import DataFrame

        combined_df = reduce(DataFrame.unionAll, dfs)
        return combined_df

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
