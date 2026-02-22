#!/usr/bin/env python3
"""
Pipeline to extract Taxon objects from CouchDB annotated files and save back to CouchDB.

This module provides a UDF-based PySpark pipeline that:
1. Reads annotated files from an ingest CouchDB database
2. Extracts Taxon objects using the SKOL pipeline
3. Saves Taxa as JSON documents to a taxon CouchDB database
4. Ensures idempotent operations using composite keys: (doc_id, url, line_number)
"""

import hashlib
import os
import sys
import logging
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

import couchdb
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, MapType, IntegerType, ArrayType

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_classifier.couchdb_io import CouchDBConnection
from env_config import get_env_config
from ingestors.timestamps import set_timestamps

from couchdb_file import read_couchdb_partition
from finder import parse_annotated, remove_interstitials
from taxon import group_paragraphs, Taxon, get_ingest_field

# Set up logging
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG_TRACE = False
DEBUG_DOC_ID = None


def row_to_dict_recursive(obj: Any) -> Any:
    """
    Recursively convert PySpark Row objects to dictionaries.

    PySpark's Row.asDict() doesn't recursively convert nested Row objects,
    leaving them to serialize as arrays/tuples instead of dictionaries.
    This function ensures all nested structures are proper Python dicts/lists.

    Args:
        obj: Any object - Row, list, dict, or primitive

    Returns:
        The object with all Row instances converted to dicts
    """
    if isinstance(obj, Row):
        return {key: row_to_dict_recursive(value) for key, value in obj.asDict().items()}
    elif isinstance(obj, list):
        return [row_to_dict_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: row_to_dict_recursive(value) for key, value in obj.items()}
    else:
        return obj


def restore_span_types(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restore proper types for span fields after MapType string conversion.

    MapType(StringType(), StringType()) converts all values to strings.
    This restores integer fields to int and handles None values.

    Args:
        span: Span dictionary with string values

    Returns:
        Span dictionary with proper types
    """
    int_fields = ['paragraph_number', 'start_line', 'end_line', 'start_char', 'end_char', 'pdf_page']
    str_fields = ['pdf_label', 'empirical_page']

    result = {}
    for field in int_fields:
        value = span.get(field)
        if value is not None and value != 'None':
            try:
                result[field] = int(value)
            except (ValueError, TypeError):
                result[field] = None
        else:
            result[field] = None

    for field in str_fields:
        value = span.get(field)
        if value is not None and value != 'None':
            result[field] = str(value)
        else:
            result[field] = None

    return result


def generate_taxon_doc_id(taxon_text: str, description_text: str) -> str:
    """
    Generate a content-based, deterministic document ID for a taxon.

    Identical taxon+description content always produces the same ID,
    regardless of which ingest path produced it.

    Args:
        taxon_text: The nomenclature/taxon text
        description_text: The description text

    Returns:
        Deterministic document ID as 'taxon_<sha256_hex>'
    """
    content = (taxon_text or "").strip() + ":" + (description_text or "").strip()
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return f"taxon_{hash_obj.hexdigest()}"


def extract_taxa_from_partition(
    partition: Iterator[Row],
    ingest_db_name: str
) -> Iterator[Taxon]:
    """
    Extract Taxa from a partition of CouchDB rows.

    This function processes annotated files from CouchDB and yields
    Taxon objects for further processing.

    Args:
        partition: Iterator of Rows with columns:
            - doc_id: CouchDB document ID
            - attachment_name: Attachment filename
            - value: Text content
            - human_url: Optional URL
        ingest_db_name: Database name for metadata tracking

    Yields:
        Taxon objects with nomenclature and description paragraphs
    """
    # Convert to list to enable tracing
    partition_list = list(partition)

    if DEBUG_TRACE:
        for row in partition_list:
            if DEBUG_DOC_ID is None or row.doc_id == DEBUG_DOC_ID:
                logger.info(f"[TRACE] Row from CouchDB: doc_id={row.doc_id}, "
                           f"human_url={getattr(row, 'human_url', 'NOT_PRESENT')}, "
                           f"pdf_url={getattr(row, 'pdf_url', 'NOT_PRESENT')}")

    # Read lines from partition
    lines = read_couchdb_partition(iter(partition_list), ingest_db_name)

    # Trace first few lines
    lines_list = []
    for i, line in enumerate(lines):
        lines_list.append(line)
        if DEBUG_TRACE and i < 3:
            if DEBUG_DOC_ID is None or (hasattr(line, 'doc_id') and line.doc_id == DEBUG_DOC_ID):
                logger.info(f"[TRACE] Line {i}: doc_id={getattr(line, 'doc_id', 'N/A')}, "
                           f"human_url={getattr(line, 'human_url', 'N/A')}, "
                           f"pdf_url={getattr(line, 'pdf_url', 'N/A')}")

    # Parse annotated content
    paragraphs = parse_annotated(iter(lines_list))

    # Remove interstitial paragraphs
    filtered = remove_interstitials(paragraphs)

    # Convert to list to preserve paragraph objects for metadata extraction
    filtered_list = list(filtered)

    if DEBUG_TRACE and filtered_list:
        first_para = filtered_list[0]
        if DEBUG_DOC_ID is None or (hasattr(first_para.first_line, 'doc_id') and first_para.first_line.doc_id == DEBUG_DOC_ID):
            logger.info(f"[TRACE] First paragraph: "
                       f"human_url={getattr(first_para, 'human_url', 'N/A')}, "
                       f"pdf_url={getattr(first_para, 'pdf_url', 'N/A')}")

    # Group into taxa (returns Taxon objects with references to paragraphs)
    taxa = group_paragraphs(iter(filtered_list))

    # Yield Taxon objects directly
    for taxon in taxa:
        # Only yield taxa that have nomenclature
        if taxon.has_nomenclature():
            if DEBUG_TRACE:
                taxon_row = taxon.as_row()
                ingest = taxon_row.get('ingest') or {}
                doc_id = ingest.get('_id')
                if DEBUG_DOC_ID is None or doc_id == DEBUG_DOC_ID:
                    logger.info(f"[TRACE] Taxon extracted: doc_id={doc_id}, "
                               f"human_url={ingest.get('url')}, "
                               f"pdf_url={ingest.get('pdf_url')}")
            yield taxon


def convert_taxa_to_rows(partition: Iterator[Taxon]) -> Iterator[Row]:
    """
    Convert Taxon objects to PySpark Rows suitable for DataFrame creation.

    Args:
        partition: Iterator of Taxon objects

    Yields:
        PySpark Row objects with fields:
            - taxon: String of concatenated nomenclature paragraphs
            - description: String of concatenated description paragraphs
            - ingest: Full ingest document (contains _id, url, pdf_url, etc.)
            - line_number: Line number of first nomenclature paragraph
            - paragraph_number: Paragraph number of first nomenclature paragraph
            - pdf_page: PDF page number
            - pdf_label: Human-readable PDF page label.
            - empirical_page_number: Empirical page number of first nomenclature paragraph
    """
    for taxon in partition:
        taxon_dict = taxon.as_row()

        if DEBUG_TRACE:
            # Use ingest field names for consistency
            source_doc_id = get_ingest_field(taxon_dict, '_id')
            if DEBUG_DOC_ID is None or source_doc_id == DEBUG_DOC_ID:
                logger.info(f"[TRACE] convert_taxa_to_rows: doc_id={source_doc_id}, "
                           f"human_url={get_ingest_field(taxon_dict, 'url')}, "
                           f"pdf_url={get_ingest_field(taxon_dict, 'pdf_url')}")

        if '_id' not in taxon_dict:
            taxon_dict['_id'] = generate_taxon_doc_id(
                taxon_dict.get('taxon', ''),
                taxon_dict.get('description', '')
            )
        if 'json_annotated' not in taxon_dict:
            taxon_dict['json_annotated'] = None
        # Convert dict to Row
        row = Row(**taxon_dict)

        if DEBUG_TRACE:
            doc_id = get_ingest_field(taxon_dict, '_id')
            if DEBUG_DOC_ID is None or doc_id == DEBUG_DOC_ID:
                logger.info(f"[TRACE] Row created: ingest={row.ingest}")

        yield row


class TaxonExtractor:
    """
    Extract and save Taxa from CouchDB annotated files.

    This class encapsulates the complete pipeline for:
    1. Loading annotated documents from a CouchDB ingest database
    2. Extracting Taxon objects using the SKOL pipeline
    3. Saving Taxa to a CouchDB taxon database with idempotent keys

    Args:
        spark: SparkSession for distributed processing
        ingest_couchdb_url: URL of ingest CouchDB server
        ingest_db_name: Name of ingest database
        taxon_db_name: Name of taxon database
        taxon_couchdb_url: URL of taxon CouchDB server (defaults to ingest_couchdb_url)
        ingest_username: Optional username for ingest database
        ingest_password: Optional password for ingest database
        taxon_username: Optional username for taxon database (defaults to ingest_username)
        taxon_password: Optional password for taxon database (defaults to ingest_password)

    Example:
        >>> spark = SparkSession.builder.appName("TaxonExtractor").getOrCreate()
        >>>
        >>> extractor = TaxonExtractor(
        ...     spark=spark,
        ...     ingest_couchdb_url="http://localhost:5984",
        ...     ingest_db_name="mycobank_annotations",
        ...     taxon_db_name="mycobank_taxa",
        ...     ingest_username="admin",
        ...     ingest_password="secret"
        ... )
        >>>
        >>> # Step-by-step debugging
        >>> annotated_df = extractor.load_annotated_documents()
        >>> print(f"Loaded {annotated_df.count()} documents")
        >>>
        >>> taxa_df = extractor.extract_taxa(annotated_df)
        >>> print(f"Extracted {taxa_df.count()} taxa")
        >>> taxa_df.show(5)
        >>>
        >>> results = extractor.save_taxa(taxa_df)
        >>> print(f"Saved: {results.filter('success = true').count()}")
        >>>
        >>> # Or run the complete pipeline
        >>> results = extractor.run_pipeline()
        >>> results.filter("success = false").show()
    """

    def __init__(
        self,
        spark: SparkSession,
        ingest_couchdb_url: str,
        ingest_db_name: str,
        taxon_db_name: str,
        taxon_couchdb_url: Optional[str] = None,
        ingest_username: Optional[str] = None,
        ingest_password: Optional[str] = None,
        taxon_username: Optional[str] = None,
        taxon_password: Optional[str] = None,
        verbosity: int = 1
    ):
        self.spark = spark
        self.ingest_couchdb_url = ingest_couchdb_url
        self.ingest_db_name = ingest_db_name
        self.ingest_username = ingest_username
        self.ingest_password = ingest_password

        self.taxon_couchdb_url = taxon_couchdb_url or ingest_couchdb_url
        self.taxon_db_name = taxon_db_name
        self.taxon_username = taxon_username or ingest_username
        self.taxon_password = taxon_password or ingest_password
        self.verbosity = verbosity

        # Schema for extracted taxa
        # All metadata is in the ingest field
        # Span schema uses MapType to preserve dictionary structure in CouchDB
        # (StructType converts dicts to Row objects which serialize as arrays)
        span_map_schema = MapType(StringType(), StringType(), valueContainsNull=True)

        self._extract_schema = StructType([
            StructField("taxon", StringType(), False),
            StructField("description", StringType(), False),
            StructField("ingest", MapType(StringType(), StringType(),
                                          valueContainsNull=True), True),
            StructField("line_number", IntegerType(), True),
            StructField("paragraph_number", IntegerType(), True),
            StructField("pdf_page", IntegerType(), True),
            StructField("pdf_label", StringType(), True),
            StructField("empirical_page_number", StringType(), True),
            StructField("nomenclature_spans", ArrayType(span_map_schema), True),
            StructField("description_spans", ArrayType(span_map_schema), True),
            StructField("attachment_name", StringType(), True),
            StructField("_id", StringType(), True),
            StructField("json_annotated", StringType(), True)
        ])

        # Schema for save results
        self._save_schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("success", BooleanType(), False),
            StructField("error_message", StringType(), False),
        ])

    def load_annotated_documents(self, pattern: str = "*.ann") -> DataFrame:
        """
        Load annotated documents from CouchDB ingest database.

        Args:
            pattern: Pattern for attachment names (default: "*.ann")
                    Matches both article.txt.ann and article.pdf.ann

        Returns:
            DataFrame with columns: doc_id, attachment_name, value
        """
        ingest_conn = CouchDBConnection(
            self.ingest_couchdb_url,
            self.ingest_db_name,
            self.ingest_username,
            self.ingest_password
        )
        return ingest_conn.load_distributed(self.spark, pattern)

    def extract_taxa(self, annotated_df: DataFrame) -> DataFrame:
        """
        Extract taxa from annotated documents DataFrame.

        Args:
            annotated_df: DataFrame with columns: doc_id, attachment_name, value
                         (or prediction DataFrame with doc_id, pos, value, prediction, ...)

        Returns:
            DataFrame with taxa information (taxon, description, source, line numbers, etc.)
        """
        # Select only the columns we need for extraction
        # All metadata (doc_id, url, pdf_url) comes from the ingest column
        required_cols = ["value", "ingest"]

        # Add attachment_name if it exists (from CouchDB)
        if "attachment_name" in annotated_df.columns:
            required_cols.append("attachment_name")

        # Select only required columns to avoid schema mismatch
        annotated_df_filtered = annotated_df.select(*required_cols)

        # Debug: Print schema to verify
        if self.verbosity >= 2:
            print(f"[TaxonExtractor] Input DataFrame columns: {annotated_df.columns}")
            print(f"[TaxonExtractor] Filtered DataFrame columns: {annotated_df_filtered.columns}")
            print(f"[TaxonExtractor] Filtered DataFrame schema:")
            annotated_df_filtered.printSchema()

        # Extract to local variable to avoid serializing self
        db_name = self.ingest_db_name

        def extract_partition(partition):  # type: ignore[reportUnknownParameterType]
            # Extract Taxon objects
            taxa = extract_taxa_from_partition(iter(partition), db_name)  # type: ignore[reportUnknownArgumentType]
            # Convert to Rows for DataFrame
            return convert_taxa_to_rows(taxa)

        taxa_rdd = annotated_df_filtered.rdd.mapPartitions(extract_partition)  # type: ignore[reportUnknownArgumentType]
        taxa_df = self.spark.createDataFrame(taxa_rdd, self._extract_schema)

        return taxa_df

    def load_taxa(self, pattern: str = "taxon_*") -> DataFrame:
        """
        Load taxa from CouchDB taxon database.

        This method performs the inverse operation of save_taxa(), loading
        taxa documents from CouchDB and converting them back to a DataFrame.

        Args:
            pattern: Pattern for document IDs to load (default: "taxon_*")
                    Use "*" to load all documents
                    Use "taxon_abc*" to load specific subset

        Returns:
            DataFrame with taxa information matching the extract_taxa() schema:
            - taxon: String of concatenated nomenclature paragraphs
            - description: String of concatenated description paragraphs
            - source: Dict with keys doc_id, human_url, pdf_url, db_name
            - line_number: Line number of first nomenclature paragraph
            - paragraph_number: Paragraph number of first nomenclature paragraph
            - page_number: Page number of first nomenclature paragraph
            - pdf_page: PDF page number (same as page_number)
            - empirical_page_number: Empirical page number of first nomenclature paragraph

        Example:
            >>> # Load all taxa
            >>> taxa_df = extractor.load_taxa()
            >>> print(f"Loaded {taxa_df.count()} taxa")
            >>>
            >>> # Load specific subset
            >>> subset_df = extractor.load_taxa(pattern="taxon_abc*")
        """
        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.taxon_db_name
        username = self.taxon_username
        password = self.taxon_password
        extract_schema = self._extract_schema
        verbosity = self.verbosity

        def load_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Load taxa from CouchDB for an entire partition."""
            # Connect to CouchDB once per partition
            try:
                server = couchdb.Server(couchdb_url)
                if username and password:
                    server.resource.credentials = (username, password)

                # Check if database exists
                if db_name not in server:
                    if verbosity >= 1:
                        print(f"Database {db_name} does not exist")
                    return

                db = server[db_name]

                # Process each row (which contains doc_id)
                for row in partition:
                    doc_id = "unknown"
                    try:
                        doc_id = row.doc_id if hasattr(row, 'doc_id') else str(row[0])

                        # Load document from CouchDB
                        if doc_id in db:
                            doc = db[doc_id]

                            # Convert CouchDB document to Row
                            taxon_data = {
                                'taxon': doc.get('taxon', ''),
                                'description': doc.get('description', ''),
                                'ingest': doc.get('ingest'),
                                'line_number': doc.get('line_number'),
                                'paragraph_number': doc.get('paragraph_number'),
                                'pdf_page': doc.get('pdf_page'),
                                'pdf_label': doc.get('pdf_label'),
                                'empirical_page_number': doc.get('empirical_page_number'),
                                'nomenclature_spans': doc.get('nomenclature_spans'),
                                'description_spans': doc.get('description_spans'),
                                'attachment_name': doc.get('attachment_name'),
                                '_id': doc.get('_id'),
                                'json_annotated': doc.get('json_annotated'),
                            }

                            yield Row(**taxon_data)
                        else:
                            if verbosity >= 1:
                                print(f"Document {doc_id} not found in database")

                    except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                        if verbosity >= 1:
                            print(f"Error loading taxon {doc_id}: {e}")

            except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                if verbosity >= 1:
                    print(f"Error connecting to CouchDB: {e}")

        # First, get list of document IDs matching pattern from CouchDB
        # We need to create a DataFrame with doc_ids to process
        doc_ids = self._get_matching_doc_ids(pattern)

        if not doc_ids:
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], extract_schema)

        # Create DataFrame with doc_ids for parallel processing
        doc_ids_rdd = self.spark.sparkContext.parallelize(doc_ids)
        doc_ids_df = doc_ids_rdd.map(lambda x: Row(doc_id=x)).toDF()

        # Load taxa using mapPartitions
        taxa_rdd = doc_ids_df.rdd.mapPartitions(load_partition)
        taxa_df = self.spark.createDataFrame(taxa_rdd, extract_schema)

        return taxa_df

    def get_existing_ingest_doc_ids(self) -> set:
        """
        Get set of ingest document IDs that already have taxa in the taxon database.

        This queries the taxon database and collects all unique ingest._id values,
        which represent ingest documents that have already been processed.

        Returns:
            Set of ingest document IDs that have existing taxa
        """
        try:
            server = couchdb.Server(self.taxon_couchdb_url)
            if self.taxon_username and self.taxon_password:
                server.resource.credentials = (self.taxon_username, self.taxon_password)

            # Check if database exists
            if self.taxon_db_name not in server:
                if self.verbosity >= 1:
                    print(f"Taxon database {self.taxon_db_name} does not exist yet")
                return set()

            db = server[self.taxon_db_name]

            # Collect unique ingest._id values from all taxa documents
            existing_ids: set = set()
            for doc_id in db:
                if doc_id.startswith('_design/'):
                    continue
                try:
                    doc = db[doc_id]
                    ingest = doc.get('ingest')
                    if ingest and isinstance(ingest, dict):
                        ingest_id = ingest.get('_id')
                        if ingest_id:
                            existing_ids.add(ingest_id)
                except Exception:
                    pass  # Skip documents we can't read

            if self.verbosity >= 1:
                print(f"Found {len(existing_ids)} ingest documents with existing taxa")

            return existing_ids

        except Exception as e:
            if self.verbosity >= 1:
                print(f"Error querying existing taxa: {e}")
            return set()

    def _get_matching_doc_ids(self, pattern: str) -> list:
        """
        Get list of document IDs matching the pattern from CouchDB.

        Args:
            pattern: Pattern for document IDs (e.g., "taxon_*", "*")

        Returns:
            List of matching document IDs
        """
        try:
            server = couchdb.Server(self.taxon_couchdb_url)
            if self.taxon_username and self.taxon_password:
                server.resource.credentials = (self.taxon_username, self.taxon_password)

            # Check if database exists
            if self.taxon_db_name not in server:
                if self.verbosity >= 1:
                    print(f"Database {self.taxon_db_name} does not exist")
                return []

            db = server[self.taxon_db_name]

            # Get all document IDs
            all_doc_ids = [doc_id for doc_id in db if not doc_id.startswith('_design/')]

            # Filter by pattern
            if pattern == "*":
                # Return all non-design documents
                return all_doc_ids
            else:
                # Simple pattern matching (prefix matching for now)
                # Convert glob pattern to prefix
                if pattern.endswith('*'):
                    prefix = pattern[:-1]
                    return [doc_id for doc_id in all_doc_ids if doc_id.startswith(prefix)]
                else:
                    # Exact match
                    return [doc_id for doc_id in all_doc_ids if doc_id == pattern]

        except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
            if self.verbosity >= 1:
                print(f"Error getting document IDs from CouchDB: {e}")
            return []

    def save_taxa(self, taxa_df: DataFrame, deduplicate: bool = True) -> DataFrame:
        """
        Save taxa DataFrame to CouchDB taxon database.

        Args:
            taxa_df: DataFrame with taxa information
            deduplicate: If True, deduplicate by _id before saving (default: True)

        Returns:
            DataFrame with save results (doc_id, success, error_message)
        """
        from pyspark.sql.functions import col, row_number
        from pyspark.sql.window import Window

        # Deduplicate by _id to prevent conflicts from duplicate taxa
        if deduplicate and "_id" in taxa_df.columns:
            original_count = taxa_df.count()
            # Keep only the first occurrence of each _id
            window = Window.partitionBy("_id").orderBy(col("line_number"))
            taxa_df = taxa_df.withColumn("_row_num", row_number().over(window)) \
                            .filter(col("_row_num") == 1) \
                            .drop("_row_num")
            deduped_count = taxa_df.count()
            if self.verbosity >= 1 and original_count != deduped_count:
                print(f"  Deduplicated: {original_count} -> {deduped_count} taxa ({original_count - deduped_count} duplicates removed)")

        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.taxon_db_name
        username = self.taxon_username
        password = self.taxon_password
        verbosity = self.verbosity

        def save_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Save taxa to CouchDB for an entire partition (idempotent)."""
            MAX_RETRIES = 3

            # Connect to CouchDB once per partition
            try:
                server = couchdb.Server(couchdb_url)
                if username and password:
                    server.resource.credentials = (username, password)

                # Get or create database
                if db_name in server:
                    db = server[db_name]
                else:
                    db = server.create(db_name)  # pyright: ignore[reportUnknownMemberType]

                # Process each taxon in the partition
                for row in partition:
                    success = False
                    error_msg = ""
                    doc_id = "unknown"

                    try:
                        # Convert row to dict, recursively handling nested Row objects
                        row_dict = row_to_dict_recursive(row)

                        # Restore proper types for span fields (MapType stores as strings)
                        for span_field in ['nomenclature_spans', 'description_spans']:
                            if span_field in row_dict and row_dict[span_field]:
                                row_dict[span_field] = [
                                    restore_span_types(span) for span in row_dict[span_field]
                                ]

                        # Use ingest field names via get_ingest_field()
                        source_doc_id: str = str(get_ingest_field(row_dict, '_id', default='unknown'))
                        source_url: Optional[str] = get_ingest_field(row_dict, 'url')
                        line_number: Any = row.line_number if hasattr(row, 'line_number') else 0  # type: ignore[reportUnknownMemberType]

                        if DEBUG_TRACE:
                            if DEBUG_DOC_ID is None or source_doc_id == DEBUG_DOC_ID:
                                logger.info(f"[TRACE] save_partition: doc_id={source_doc_id}, "
                                           f"url={source_url}, "
                                           f"pdf_url={get_ingest_field(row_dict, 'pdf_url')}")

                        # Generate deterministic document ID
                        doc_id = generate_taxon_doc_id(
                            row_dict.get('taxon', ''),
                            row_dict.get('description', '')
                        )

                        # Use row_dict (already converted above) for CouchDB storage
                        taxon_doc = row_dict

                        if DEBUG_TRACE:
                            if DEBUG_DOC_ID is None or source_doc_id == DEBUG_DOC_ID:
                                logger.info(f"[TRACE] taxon_doc before save: _id={doc_id}, "
                                           f"ingest={taxon_doc.get('ingest')}")

                        # Retry loop to handle concurrent update conflicts
                        for attempt in range(MAX_RETRIES):
                            try:
                                # Check if document already exists (idempotent)
                                is_new_doc = doc_id not in db
                                if not is_new_doc:
                                    # Document exists - update it with latest _rev
                                    existing_doc = db[doc_id]
                                    taxon_doc['_id'] = doc_id
                                    taxon_doc['_rev'] = existing_doc['_rev']
                                else:
                                    # New document - create it
                                    taxon_doc['_id'] = doc_id
                                    # Remove _rev if present from previous attempt
                                    taxon_doc.pop('_rev', None)

                                set_timestamps(taxon_doc, is_new=is_new_doc)
                                db.save(taxon_doc)  # pyright: ignore[reportUnknownMemberType]
                                success = True

                                if DEBUG_TRACE:
                                    if DEBUG_DOC_ID is None or source_doc_id == DEBUG_DOC_ID:
                                        logger.info(f"[TRACE] Successfully saved taxon: {doc_id}")

                                break  # Success, exit retry loop

                            except couchdb.ResourceConflict:
                                # Conflict - another process updated the document
                                if attempt < MAX_RETRIES - 1:
                                    if verbosity >= 2:
                                        print(f"  Conflict on {doc_id}, retrying ({attempt + 1}/{MAX_RETRIES})...")
                                    continue  # Retry with fresh _rev
                                else:
                                    raise  # Max retries exceeded, propagate error

                    except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                        error_msg = str(e)
                        if verbosity >= 1:
                            print(f"Error saving taxon {doc_id}: {e}")

                    yield Row(
                        doc_id=doc_id,
                        success=success,
                        error_message=error_msg
                    )

            except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                if verbosity >= 1:
                    print(f"Error connecting to CouchDB: {e}")
                # Yield failures for all rows
                for row in partition:
                    yield Row(
                        doc_id="unknown_connection_error",
                        success=False,
                        error_message=str(e)
                    )

        results_df = taxa_df.rdd.mapPartitions(save_partition).toDF(self._save_schema)
        return results_df

    def run_pipeline(
        self,
        pattern: str = "*.ann",
        doc_ids: Optional[list] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
        incremental: bool = False,
        incremental_batch_size: int = 50,
        skip_existing: bool = False,
    ) -> DataFrame:
        """
        Run the complete pipeline: load, extract, and save taxa.

        This method:
        1. Loads annotated files from ingest CouchDB database
        2. Extracts Taxon objects in parallel using mapPartitions
        3. Saves Taxa to taxon CouchDB database with idempotent keys
        4. Returns a DataFrame with success/failure results

        When incremental=True, processes documents in batches and saves after each
        batch completes. This prevents losing progress on crashes.

        Args:
            pattern: Pattern for attachment names (default: "*.ann")
                    Matches both article.txt.ann and article.pdf.ann
            doc_ids: If specified, only process these ingest document IDs
            dry_run: If True, extract taxa but don't save to CouchDB
            limit: If specified, process at most this many documents
            incremental: If True, process in batches and save after each batch
            incremental_batch_size: Number of documents per batch when incremental=True
            skip_existing: If True, skip ingest documents that already have taxa

        Returns:
            DataFrame with columns: doc_id, success, error_message

        Example:
            >>> results = extractor.run_pipeline()
            >>> results.filter("success = true").count()
            >>> results.filter("success = false").show()
            >>> # Process single document
            >>> results = extractor.run_pipeline(doc_ids=["my_document_id"])
            >>> # Dry run
            >>> results = extractor.run_pipeline(dry_run=True)
            >>> # Incremental (crash-resistant)
            >>> results = extractor.run_pipeline(incremental=True, incremental_batch_size=25)
            >>> # Skip documents that already have taxa
            >>> results = extractor.run_pipeline(skip_existing=True)
        """
        from pyspark.sql.functions import col
        from pyspark.sql import Row

        # Step 1: Load annotated documents from CouchDB
        annotated_df = self.load_annotated_documents(pattern)

        # Filter to specific documents if doc_ids specified
        if doc_ids:
            annotated_df = annotated_df.filter(col("doc_id").isin(doc_ids))
            if self.verbosity >= 1:
                count = annotated_df.count()
                print(f"Filtered to {len(doc_ids)} doc_id(s): {count} attachment(s)")

        # Skip documents that already have taxa
        if skip_existing:
            existing_ids = self.get_existing_ingest_doc_ids()
            if existing_ids:
                before_count = annotated_df.select("doc_id").distinct().count()
                annotated_df = annotated_df.filter(~col("doc_id").isin(existing_ids))
                after_count = annotated_df.select("doc_id").distinct().count()
                skipped = before_count - after_count
                if self.verbosity >= 1:
                    print(f"Skipped {skipped} documents with existing taxa ({after_count} remaining)")

        # Apply limit if specified
        if limit is not None:
            # Get distinct doc_ids and limit them
            distinct_doc_ids = annotated_df.select("doc_id").distinct().limit(limit)
            annotated_df = annotated_df.join(distinct_doc_ids, "doc_id")
            if self.verbosity >= 1:
                count = annotated_df.count()
                print(f"Limited to {limit} documents: {count} attachment(s)")

        # Incremental mode: process in batches, saving after each
        if incremental:
            # Get list of distinct doc_ids to process
            all_doc_ids = [row.doc_id for row in annotated_df.select("doc_id").distinct().collect()]
            total_docs = len(all_doc_ids)

            if self.verbosity >= 1:
                print(f"\n{'='*70}")
                print(f"INCREMENTAL MODE: Processing {total_docs} documents in batches of {incremental_batch_size}")
                print(f"{'='*70}")

            # Accumulate results across batches
            all_results = []
            total_taxa = 0
            total_saved = 0
            total_errors = 0
            batch_num = 0

            # Process in batches
            for batch_start in range(0, total_docs, incremental_batch_size):
                batch_num += 1
                batch_doc_ids = all_doc_ids[batch_start:batch_start + incremental_batch_size]
                batch_size_actual = len(batch_doc_ids)

                if self.verbosity >= 1:
                    print(f"\n--- Batch {batch_num}: documents {batch_start + 1}-{batch_start + batch_size_actual} of {total_docs} ---")

                try:
                    # Filter to this batch's documents
                    batch_df = annotated_df.filter(col("doc_id").isin(batch_doc_ids))

                    # Extract taxa for this batch
                    taxa_df = self.extract_taxa(batch_df)
                    batch_taxa_count = taxa_df.count()
                    total_taxa += batch_taxa_count

                    if self.verbosity >= 1:
                        print(f"  Extracted {batch_taxa_count} taxa")

                    if dry_run:
                        if self.verbosity >= 1:
                            print(f"  [DRY RUN] Would save {batch_taxa_count} taxa")
                    else:
                        # Save this batch
                        batch_results = self.save_taxa(taxa_df)
                        batch_successes = batch_results.filter("success = true").count()
                        batch_failures = batch_results.filter("success = false").count()

                        total_saved += batch_successes
                        total_errors += batch_failures

                        if self.verbosity >= 1:
                            print(f"  Saved: {batch_successes}, Errors: {batch_failures}")

                        # Collect results for final DataFrame
                        all_results.extend(batch_results.collect())

                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  ERROR in batch {batch_num}: {e}")
                    total_errors += batch_size_actual
                    # Add error results for this batch
                    for doc_id in batch_doc_ids:
                        all_results.append(Row(doc_id=doc_id, success=False, error_message=str(e)))

            # Final summary
            if self.verbosity >= 1:
                print(f"\n{'='*70}")
                print("Incremental Processing Complete!" + (" (DRY RUN)" if dry_run else ""))
                print(f"{'='*70}")
                print(f"  Total batches: {batch_num}")
                print(f"  Documents processed: {total_docs}")
                print(f"  Taxa extracted: {total_taxa}")
                if not dry_run:
                    print(f"  Taxa saved: {total_saved}")
                    print(f"  Errors: {total_errors}")

            # Return combined results DataFrame
            if dry_run or not all_results:
                return self.spark.createDataFrame([], self._save_schema)
            return self.spark.createDataFrame(all_results, self._save_schema)

        # Non-incremental mode: process all at once (original behavior)
        if self.verbosity >= 1:
            total_docs = annotated_df.select("doc_id").distinct().count()
            print(f"\nProcessing {total_docs} documents (non-incremental mode)")
            print("  TIP: Use --incremental for crash-resistant batch processing")

        # Step 2: Extract taxa from annotated documents
        taxa_df = self.extract_taxa(annotated_df)

        # Step 3: Handle dry run or save taxa to CouchDB
        if dry_run:
            # Dry run - just show what would be saved
            if self.verbosity >= 1:
                taxa_count = taxa_df.count()
                print(f"\n[DRY RUN] Would save {taxa_count} taxa to {self.taxon_db_name}")
                if self.verbosity >= 2:
                    print("\n[DRY RUN] Sample taxa:")
                    taxa_df.select("_id", "taxon", "ingest").show(5, truncate=50)
            # Return empty results DataFrame for dry run
            return self.spark.createDataFrame([], self._save_schema)

        results_df = self.save_taxa(taxa_df)

        return results_df


# Command-line interface
if __name__ == "__main__":
    import argparse

    # Get environment configuration
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description="Extract Taxa from CouchDB annotated files and save to CouchDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration (via environment variables or command-line arguments):
  --ingest-url            CouchDB server URL for ingest database
  --ingest-database       Name of ingest database
  --ingest-username       Username for ingest database
  --ingest-password       Password for ingest database
  --taxon-url             CouchDB server URL for taxon database
  --taxon-database        Name of taxon database
  --taxon-username        Username for taxon database
  --taxon-password        Password for taxon database
  --pattern               Pattern for attachment names (default: *.ann, matches both .txt.ann and .pdf.ann)

Work Control Options (from env_config):
  --dry-run               Preview what would be extracted without saving
  --incremental           Process in batches, saving after each (crash-resistant)
  --incremental-batch-size N
                          Documents per batch when --incremental is set (default: 50)
  --limit N               Process at most N documents
  --doc-id ID1,ID2,...    Process only specific document IDs (comma-separated)
  --skip-existing         Skip ingest documents that already have taxa extracted

Environment Variables:
  DRY_RUN=1               Same as --dry-run
  INCREMENTAL=1           Same as --incremental
  INCREMENTAL_BATCH_SIZE=N  Same as --incremental-batch-size
  LIMIT=N                 Same as --limit N
  DOC_IDS=id1,id2,...     Same as --doc-id
  SKIP_EXISTING=1         Same as --skip-existing

Note: All database configuration can be set via command-line arguments to env_config.
      Example: python extract_taxa_to_couchdb.py --ingest-database mydb --taxon-database mytaxa

Script-specific Options:
"""
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Enable debug tracing of URL propagation through the pipeline"
    )
    parser.add_argument(
        "--debug-doc-id",
        type=str,
        default=None,
        help="Only trace this specific document ID (optional, for focused debugging)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=None,
        help="Skip ingest documents that already have taxa extracted"
    )

    args, _ = parser.parse_known_args()

    # Set up debug tracing (modify module-level variables)
    import sys
    current_module = sys.modules[__name__]
    current_module.DEBUG_TRACE = args.debug_trace
    current_module.DEBUG_DOC_ID = args.debug_doc_id

    if DEBUG_TRACE:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        if DEBUG_DOC_ID:
            logger.info(f"[TRACE] Debug tracing enabled for doc_id: {DEBUG_DOC_ID}")
        else:
            logger.info("[TRACE] Debug tracing enabled for all documents")

    # Validate required arguments
    if not config['ingest_database']:
        parser.error("--ingest-database is required (or set $INGEST_DATABASE)")
    if not config['taxon_database']:
        parser.error("--taxon-database is required (or set $TAXON_DATABASE)")

    # Default taxon credentials to ingest credentials
    taxon_url = config['taxon_url'] or config['ingest_url']
    taxon_username = config['taxon_username'] or config['ingest_username']
    taxon_password = config['taxon_password'] or config['ingest_password']

    # Create Spark session
    spark = SparkSession.builder \
        .appName("SKOL Taxon Extractor") \
        .getOrCreate()

    if config['verbosity'] >= 1:
        print(f"Extracting taxa from {config['ingest_database']} to {config['taxon_database']}...")

    # Create extractor instance
    extractor = TaxonExtractor(
        spark=spark,
        ingest_couchdb_url=config['ingest_url'],
        ingest_db_name=config['ingest_database'],
        taxon_db_name=config['taxon_database'],
        taxon_couchdb_url=taxon_url,
        ingest_username=config['ingest_username'],
        ingest_password=config['ingest_password'],
        taxon_username=taxon_username,
        taxon_password=taxon_password,
        verbosity=config['verbosity']
    )

    # Run pipeline with standard options from env_config
    # Handle --skip-existing (command line or environment variable)
    skip_existing = args.skip_existing
    if skip_existing is None:
        skip_existing = os.environ.get('SKIP_EXISTING', '').lower() in ('1', 'true', 'yes')

    if config['verbosity'] >= 1:
        if config.get('dry_run'):
            print("[DRY RUN MODE]")
        if config.get('incremental'):
            print(f"[INCREMENTAL MODE] Batch size: {config.get('incremental_batch_size', 50)}")
        if skip_existing:
            print("[SKIP EXISTING] Skipping documents with existing taxa")
        if config.get('doc_ids'):
            print(f"Processing specific doc_ids: {config['doc_ids']}")
        if config.get('limit'):
            print(f"Limiting to {config['limit']} documents")

    results = extractor.run_pipeline(
        pattern=config['pattern'],
        doc_ids=config.get('doc_ids'),
        dry_run=config.get('dry_run', False),
        limit=config.get('limit'),
        incremental=config.get('incremental', False),
        incremental_batch_size=config.get('incremental_batch_size', 50),
        skip_existing=skip_existing,
    )

    # Show results
    if config['verbosity'] >= 1:
        total = results.count()
        successes = results.filter("success = true").count()
        failures = results.filter("success = false").count()

        print(f"\nResults:")
        print(f"  Total taxa: {total}")
        print(f"  Successful saves: {successes}")
        print(f"  Failed saves: {failures}")

        if failures > 0 and config['verbosity'] >= 2:
            print("\nFailed documents:")
            results.filter("success = false").show(truncate=False)

    spark.stop()
