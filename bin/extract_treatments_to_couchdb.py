#!/usr/bin/env python3
"""
Pipeline to extract Treatment objects from CouchDB annotated files and save back to CouchDB.

This module provides a UDF-based PySpark pipeline that:
1. Reads annotated files from an ingest CouchDB database
2. Extracts Treatment objects using the SKOL pipeline
3. Saves Treatments as JSON documents to a treatment CouchDB database
4. Ensures idempotent operations using composite keys: (doc_id, url, line_number)
"""

import hashlib
import os
import sys
import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict, Any, Set, Tuple

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
from treatment import group_paragraphs, Treatment, get_ingest_field

# Extraction-pipeline dispatcher (extraction_pipeline.md Commit 1 /
# docs/v3_buildout.md Phase A).  The per-partition function below
# now wraps the dispatcher rather than hand-rolling the parse +
# group flow.
from skol_classifier.extraction.dispatcher import Dispatcher

# Set up logging
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG_TRACE = False
DEBUG_DOC_ID = None


# Schema for extracted treatments.  Hoisted to module level so the
# extract_treatments_to_couchdb_test.py regression test can import it
# without booting a SparkSession.  The field set must match exactly
# what ``Treatment.as_row()`` produces plus ``_id`` and
# ``json_annotated`` (added by ``convert_taxa_to_rows`` below).
#
# Span schema uses MapType (not StructType) to preserve dict structure
# in CouchDB — StructType converts dicts to Row objects which then
# serialize as arrays.
_SPAN_MAP_SCHEMA = MapType(StringType(), StringType(), valueContainsNull=True)

EXTRACT_SCHEMA = StructType([
    StructField("treatment", StringType(), False),
    # Flat section text fields (None when that section is absent).
    StructField("description", StringType(), True),
    StructField("diagnosis", StringType(), True),
    StructField("etymology", StringType(), True),
    StructField("distribution", StringType(), True),
    StructField("materials_examined", StringType(), True),
    StructField("type_designation", StringType(), True),
    StructField("biology", StringType(), True),
    StructField("notes", StringType(), True),
    StructField("key", StringType(), True),
    StructField("figure_captions", StringType(), True),
    StructField("ingest", MapType(StringType(), StringType(),
                                  valueContainsNull=True), True),
    StructField("line_number", IntegerType(), True),
    StructField("paragraph_number", IntegerType(), True),
    StructField("pdf_page", IntegerType(), True),
    StructField("pdf_label", StringType(), True),
    StructField("empirical_page_number", StringType(), True),
    # Span fields (empty list when that section is absent).
    StructField("nomenclature_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("description_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("diagnosis_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("etymology_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("distribution_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("materials_examined_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("type_designation_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("biology_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("notes_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("figure_caption_spans", ArrayType(_SPAN_MAP_SCHEMA), True),
    StructField("attachment_name", StringType(), True),
    # Phase G.2: True for stub-Nomenclature treatments (orphan
    # Description/Diagnosis blocks).  Recognised by the Django UI to
    # render distinctly from name-headed treatments.
    StructField("synthetic_nomenclature", BooleanType(), False),
    StructField("_id", StringType(), True),
    StructField("json_annotated", StringType(), True)
])


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


def generate_taxon_doc_id(taxon_dict: Dict[str, Any]) -> str:
    """
    Generate a content-based, deterministic document ID for a taxon.

    Identical section content always produces the same ID, regardless of
    which ingest path produced it.  All 10 treatment section fields are
    included in the hash in a fixed canonical order so that adding a new
    section to a previously section-less treatment changes its ID.

    Canonical field order (mirrors ``taxon._LABEL_TO_FIELD``):
        taxon, description, diagnosis, etymology, distribution,
        materials_examined, type_designation, biology, notes,
        key, figure_captions

    None and empty string are treated identically; whitespace is stripped.

    Args:
        taxon_dict: Dict as returned by ``Treatment.as_row()``.

    Returns:
        Deterministic document ID as 'taxon_<sha256_hex>'
    """
    _CANONICAL_FIELDS = (
        'treatment', 'description', 'diagnosis', 'etymology', 'distribution',
        'materials_examined', 'type_designation', 'biology', 'notes',
        'key', 'figure_captions',
    )
    parts = [
        (taxon_dict.get(field) or '').strip()
        for field in _CANONICAL_FIELDS
    ]
    content = ':'.join(parts)
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return f"taxon_{hash_obj.hexdigest()}"


def _row_to_dispatcher_doc(row: Row) -> Dict[str, Any]:
    """Build the per-doc dict the dispatcher consumes from a Spark row.

    Rows carry the .ann attachment content in ``row.value`` plus the
    full ingest doc in ``row.ingest``.  The dispatcher's inspectors
    + components read attachments via ``state.get_attachment(name)``,
    so we pre-seed ``_attachments`` with the .ann bytes (the only
    attachment available in a partition row).

    Other CouchDB-resident attachments — ``article.xml``,
    ``article.txt`` — are *not* available in the partition; if they
    were the dispatcher would route through ``taxpub_treatment_extractor``
    instead.  That's a future commit (per docs/extraction_pipeline.md
    migration sequence); today's flow stays on the
    ``classifier_logistic_v3`` path.
    """
    ingest = getattr(row, "ingest", None) or {}
    # Merge ingest fields into the synthetic doc so inspectors see
    # things like xml_format / is_taxpub / url / pdf_url naturally.
    doc: Dict[str, Any] = dict(ingest)
    # Always overwrite _attachments with what's actually in-hand.
    doc["_attachments"] = {row.attachment_name: row.value}
    if "_id" not in doc:
        doc["_id"] = ingest.get("_id", "unknown")
    return doc


def iter_taxpub_treatments(
    docs: Iterable[Tuple[Dict[str, Any], bytes]],
    ingest_db_name: str = "",
) -> Iterator[Treatment]:
    """Yield Treatment objects from is_taxpub docs via the dispatcher's
    ``taxpub_treatment_extractor`` fork — the non-Spark sweep that
    Phase G.1 of v3_buildout adds to close the taxpub coverage gap.

    Each input pair is ``(doc, xml_bytes)``.  ``doc`` is the ingest
    doc dict (read-only — caller's dict is not mutated); ``xml_bytes``
    is the raw ``article.xml`` attachment content.  The bytes are
    injected into ``_attachments['article.xml']`` on a copy of the
    doc so the dispatcher's ``TaxpubMarkupInspector`` sees them; the
    ``TaxpubTreatmentExtractor`` component then fires at priority 10
    and contributes TaggedBlocks that the treatment_assembler turns
    into Treatments.

    Only Treatments with a Nomenclature are yielded — same filter
    the Spark-partition classifier path applies.
    """
    dispatcher = Dispatcher.from_default_catalogs(
        config={"ingest_db_name": ingest_db_name},
    )
    for doc, xml_bytes in docs:
        atts = dict(doc.get("_attachments") or {})
        atts["article.xml"] = xml_bytes
        doc_with_xml = dict(doc)
        doc_with_xml["_attachments"] = atts
        for treatment in dispatcher.extract(doc_with_xml):
            if treatment.has_nomenclature():
                yield treatment


def extract_taxa_from_partition(
    partition: Iterator[Row],
    ingest_db_name: str
) -> Iterator[Treatment]:
    """
    Extract Treatments from a partition of CouchDB rows via the dispatcher.

    Each row is wrapped in a per-doc dict and run through
    :class:`Dispatcher` (see ``skol_classifier/extraction/``).  The
    dispatcher routes to ``classifier_logistic_v3`` (today's only
    selectable labeler given the .ann-only attachment seed), which
    in turn feeds the ``treatment_assembler`` — preserving
    field-equality with the pre-dispatcher pipeline.

    The dispatcher itself is constructed once per partition (per the
    Spark worker model); the catalogs are loaded once and reused
    across rows.

    Args:
        partition: Iterator of Rows with columns:
            - doc_id: CouchDB document ID
            - attachment_name: Attachment filename (e.g. article.txt.ann)
            - value: Text content (the .ann YEDDA string)
            - ingest: Full ingest doc dict with metadata
        ingest_db_name: Database name for metadata tracking.

    Yields:
        Treatment objects with nomenclature and section paragraphs.
    """
    partition_list = list(partition)

    if DEBUG_TRACE:
        for row in partition_list:
            if DEBUG_DOC_ID is None or row.doc_id == DEBUG_DOC_ID:
                logger.info(
                    f"[TRACE] Row from CouchDB: doc_id={row.doc_id}, "
                    f"human_url={getattr(row, 'human_url', 'NOT_PRESENT')}, "
                    f"pdf_url={getattr(row, 'pdf_url', 'NOT_PRESENT')}"
                )

    dispatcher = Dispatcher.from_default_catalogs(
        config={"ingest_db_name": ingest_db_name},
    )

    for row in partition_list:
        doc = _row_to_dispatcher_doc(row)
        for taxon in dispatcher.extract(doc):
            if not taxon.has_nomenclature():
                continue
            if DEBUG_TRACE:
                taxon_row = taxon.as_row()
                ingest = taxon_row.get("ingest") or {}
                taxon_doc_id = ingest.get("_id")
                if DEBUG_DOC_ID is None or taxon_doc_id == DEBUG_DOC_ID:
                    logger.info(
                        f"[TRACE] Treatment extracted: "
                        f"doc_id={taxon_doc_id}, "
                        f"human_url={ingest.get('url')}, "
                        f"pdf_url={ingest.get('pdf_url')}"
                    )
            yield taxon


def convert_taxa_to_rows(partition: Iterator[Treatment]) -> Iterator[Row]:
    """
    Convert Treatment objects to PySpark Rows suitable for DataFrame creation.

    Args:
        partition: Iterator of Treatment objects

    Yields:
        PySpark Row objects with fields:
            - taxon: String of concatenated nomenclature paragraphs
            - description: String of concatenated description paragraphs (None if absent)
            - diagnosis: String of concatenated diagnosis paragraphs (None if absent)
            - etymology: String of concatenated etymology paragraphs (None if absent)
            - distribution: String of concatenated distribution paragraphs (None if absent)
            - materials_examined: String of concatenated materials examined paragraphs (None if absent)
            - type_designation: String of concatenated type designation paragraphs (None if absent)
            - biology: String of concatenated biology paragraphs (None if absent)
            - notes: String of concatenated notes paragraphs (None if absent)
            - key: String of concatenated key paragraphs (None if absent)
            - figure_captions: String of concatenated figure caption paragraphs (None if absent)
            - ingest: Full ingest document (contains _id, url, pdf_url, etc.)
            - line_number: Line number of first nomenclature paragraph
            - paragraph_number: Paragraph number of first nomenclature paragraph
            - pdf_page: PDF page number
            - pdf_label: Human-readable PDF page label
            - empirical_page_number: Empirical page number of first nomenclature paragraph
            - nomenclature_spans: List of span dicts for nomenclature section
            - description_spans: List of span dicts for description section
            - diagnosis_spans: List of span dicts for diagnosis section
            - etymology_spans: List of span dicts for etymology section
            - distribution_spans: List of span dicts for distribution section
            - materials_examined_spans: List of span dicts for materials examined section
            - type_designation_spans: List of span dicts for type designation section
            - biology_spans: List of span dicts for biology section
            - notes_spans: List of span dicts for notes section
            - attachment_name: Name of the source attachment
            - _id: Content-addressable document ID (taxon_<sha256 of all section fields>)
            - json_annotated: JSON string of annotated paragraphs
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
            taxon_dict['_id'] = generate_taxon_doc_id(taxon_dict)
        if 'json_annotated' not in taxon_dict:
            taxon_dict['json_annotated'] = None
        # Convert dict to Row.  Reorder keys to match EXTRACT_SCHEMA
        # because ``createDataFrame`` matches Row fields positionally,
        # not by name — and ``Treatment.as_row()``'s insertion order
        # does not match the schema's field order.  Without this
        # reordering, e.g. the 15th field of the Row (``biology`` in
        # as_row order) would land in the schema's 15th slot
        # (``pdf_page``), failing type validation downstream.
        row = Row(**{name: taxon_dict[name]
                     for name in EXTRACT_SCHEMA.fieldNames()})

        if DEBUG_TRACE:
            doc_id = get_ingest_field(taxon_dict, '_id')
            if DEBUG_DOC_ID is None or doc_id == DEBUG_DOC_ID:
                logger.info(f"[TRACE] Row created: ingest={row.ingest}")

        yield row


class TreatmentExtractor:
    """
    Extract and save treatment from CouchDB annotated files.

    This class encapsulates the complete pipeline for:
    1. Loading annotated documents from a CouchDB annotations database
    2. Extracting Treatment objects using the SKOL pipeline
    3. Saving treatment to a CouchDB treatment database with idempotent keys

    Args:
        spark: SparkSession for distributed processing
        ingest_couchdb_url: URL of ingest CouchDB server
        ingest_db_name: Name of ingest database (where PDFs live, e.g. skol_dev).
            Stored in each treatment record as ``ingest.db_name`` so that views can
            retrieve the PDF directly without guessing.
        treatments_db_name: Name of taxon database
        annotations_db_name: Name of annotations database (where .ann files live,
            e.g. skol_exp_NAME_ann).  Defaults to ``ingest_db_name`` for
            backward-compatibility with setups where annotations live in the
            ingest DB.  Stored in each treatment record as ``annotations_db``.
        taxon_couchdb_url: URL of taxon CouchDB server (defaults to ingest_couchdb_url)
        ingest_username: Optional username for ingest database
        ingest_password: Optional password for ingest database
        taxon_username: Optional username for taxon database (defaults to ingest_username)
        taxon_password: Optional password for taxon database (defaults to ingest_password)

    Example:
        >>> spark = SparkSession.builder.appName("TreatmentExtractor").getOrCreate()
        >>>
        >>> extractor = TreatmentExtractor(
        ...     spark=spark,
        ...     ingest_couchdb_url="http://localhost:5984",
        ...     ingest_db_name="skol_dev",
        ...     annotations_db_name="skol_exp_myexp_02_00_ann_prose",
        ...     treatments_db_name="skol_exp_myexp_03_00_ann_structured",
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
        treatments_db_name: str,
        annotations_db_name: Optional[str] = None,
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
        # annotations_db_name is where .ann files live; falls back to ingest_db_name
        # for setups that have not separated annotations into their own database.
        self.annotations_db_name: str = annotations_db_name or ingest_db_name
        self.ingest_username = ingest_username
        self.ingest_password = ingest_password

        self.taxon_couchdb_url = taxon_couchdb_url or ingest_couchdb_url
        self.treatments_db_name = treatments_db_name
        self.taxon_username = taxon_username or ingest_username
        self.taxon_password = taxon_password or ingest_password
        self.verbosity = verbosity

        # Schema for extracted treatments (defined at module level so
        # the bin/extract_treatments_to_couchdb_test.py field-set
        # regression test can import it without booting Spark).
        self._extract_schema = EXTRACT_SCHEMA

        # Schema for save results
        self._save_schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("success", BooleanType(), False),
            StructField("error_message", StringType(), False),
        ])

    def load_annotated_documents(self, pattern: str = "*.ann") -> DataFrame:
        """
        Load annotated documents from the annotations database.

        Args:
            pattern: Pattern for attachment names (default: "*.ann")
                    Matches both article.txt.ann and article.pdf.ann

        Returns:
            DataFrame with columns: doc_id, attachment_name, value
        """
        annotations_conn = CouchDBConnection(
            self.ingest_couchdb_url,
            self.annotations_db_name,
            self.ingest_username,
            self.ingest_password
        )
        return annotations_conn.load_distributed(self.spark, pattern)

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
            print(f"[TreatmentExtractor] Input DataFrame columns: {annotated_df.columns}")
            print(f"[TreatmentExtractor] Filtered DataFrame columns: {annotated_df_filtered.columns}")
            print(f"[TreatmentExtractor] Filtered DataFrame schema:")
            annotated_df_filtered.printSchema()

        # Extract to local variables to avoid serializing self in the closure
        db_name = self.ingest_db_name
        annotations_db_name = self.annotations_db_name

        def extract_partition(partition):  # type: ignore[reportUnknownParameterType]
            # Extract Treatment objects
            taxa = extract_taxa_from_partition(iter(partition), db_name)  # type: ignore[reportUnknownArgumentType]
            # Convert to Rows for DataFrame
            return convert_taxa_to_rows(taxa)

        taxa_rdd = annotated_df_filtered.rdd.mapPartitions(extract_partition)  # type: ignore[reportUnknownArgumentType]
        taxa_df = self.spark.createDataFrame(taxa_rdd, self._extract_schema)

        return taxa_df

    def load_treatments(self, pattern: str = "taxon_*") -> DataFrame:
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
            >>> taxa_df = extractor.load_treatments()
            >>> print(f"Loaded {taxa_df.count()} taxa")
            >>>
            >>> # Load specific subset
            >>> subset_df = extractor.load_treatments(pattern="taxon_abc*")
        """
        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.treatments_db_name
        username = self.taxon_username
        password = self.taxon_password
        extract_schema = self._extract_schema
        verbosity = self.verbosity

        def load_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Load treatments from CouchDB for an entire partition."""
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
                                'treatment': doc.get('treatment', ''),
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

    def _extract_taxpub_treatments(
        self,
        skip_doc_ids: Optional[Set[str]] = None,
    ) -> Iterator[Treatment]:
        """Iterate ``is_taxpub=True`` docs in the ingest DB and yield
        Treatment objects via the dispatcher's ``taxpub_treatment_extractor``
        fork.

        Phase G.1 of v3_buildout: closes the coverage gap where the
        Spark partition flow never sees ``article.xml`` bytes, so
        the taxpub fork never fires.  Runs as a non-Spark sweep
        (~1,784 docs on dev) — one CouchDB round-trip per doc, both
        for the doc dict and the article.xml attachment.

        ``skip_doc_ids`` filters out ingest doc IDs that already
        produced Treatments (passed through from ``run_pipeline``'s
        ``--skip-existing`` flow).
        """
        server = couchdb.Server(self.ingest_couchdb_url)
        if self.ingest_username and self.ingest_password:
            server.resource.credentials = (
                self.ingest_username, self.ingest_password,
            )
        db = server[self.ingest_db_name]

        def _yield_docs() -> Iterator[Tuple[Dict[str, Any], bytes]]:
            count = 0
            for doc_id in db:
                if skip_doc_ids is not None and doc_id in skip_doc_ids:
                    continue
                try:
                    doc = db[doc_id]
                except Exception:
                    continue
                if not doc.get("is_taxpub"):
                    continue
                atts = doc.get("_attachments") or {}
                if "article.xml" not in atts:
                    continue
                try:
                    xml_bytes = db.get_attachment(doc_id, "article.xml").read()
                except Exception:
                    continue
                count += 1
                yield (dict(doc), xml_bytes)
            if self.verbosity >= 1:
                print(
                    f"[G.1] Scanned {self.ingest_db_name}: "
                    f"{count} is_taxpub docs with article.xml"
                )

        yield from iter_taxpub_treatments(
            _yield_docs(), ingest_db_name=self.ingest_db_name,
        )

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
            if self.treatments_db_name not in server:
                if self.verbosity >= 1:
                    print(f"Taxon database {self.treatments_db_name} does not exist yet")
                return set()

            db = server[self.treatments_db_name]

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
                print(f"Found {len(existing_ids)} ingest documents with existing treatments")

            return existing_ids

        except Exception as e:
            if self.verbosity >= 1:
                print(f"Error querying existing treatments: {e}")
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
            if self.treatments_db_name not in server:
                if self.verbosity >= 1:
                    print(f"Database {self.treatments_db_name} does not exist")
                return []

            db = server[self.treatments_db_name]

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
                print(f"  Deduplicated: {original_count} -> {deduped_count} treatments ({original_count - deduped_count} duplicates removed)")

        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.treatments_db_name
        username = self.taxon_username
        password = self.taxon_password
        verbosity = self.verbosity
        # Stamped into each saved taxon doc so views can locate the
        # source .ann attachment without guessing which database holds
        # it.  ``save_partition`` is a closure executed on Spark workers
        # without ``self`` in scope; without this local extract every
        # save raises ``NameError: name 'annotations_db_name' is not
        # defined`` and the whole partition's writes silently fail.
        annotations_db_name = self.annotations_db_name
        # ``ingest.db_name`` on each saved Treatment must point at the
        # SOURCE ingest DB (where ``article.pdf`` lives) so the Django
        # PDF view ([django/search/views.py: ServeTaxaPdfView]) can
        # fetch the right attachment.  Previously this was set to
        # ``self.treatments_db_name`` (where the Treatment itself
        # lives), which meant the PDF lookup always 404'd.
        ingest_db_name = self.ingest_db_name

        def save_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Save treatments to CouchDB for an entire partition (idempotent)."""
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
                        doc_id = generate_taxon_doc_id(row_dict)

                        # Use row_dict (already converted above) for CouchDB storage
                        taxon_doc = row_dict

                        # Stamp provenance: where to find the PDF and the .ann file.
                        # Both fields are stored alongside the attachment_name so that
                        # views can retrieve attachments without guessing DB names.
                        ingest_data = taxon_doc.get('ingest')
                        if isinstance(ingest_data, dict):
                            ingest_data['db_name'] = ingest_db_name
                        taxon_doc['annotations_db'] = annotations_db_name

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
        2. Extracts Treatment objects in parallel using mapPartitions
        3. Saves treatments to treatment CouchDB database with idempotent keys
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
                    print(f"Skipped {skipped} documents with existing treatments ({after_count} remaining)")

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
                        print(f"  Extracted {batch_taxa_count} treatments")

                    if dry_run:
                        if self.verbosity >= 1:
                            print(f"  [DRY RUN] Would save {batch_taxa_count} treatments")
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
                print(f"  Treatments extracted: {total_taxa}")
                if not dry_run:
                    print(f"  Treatments saved: {total_saved}")
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

        # Step 2: Extract taxa from annotated documents (classifier path)
        taxa_df = self.extract_taxa(annotated_df)

        # Step 2b — Phase G.1: also extract Treatments from is_taxpub
        # docs in the ingest DB.  predict_classifier skips is_taxpub
        # docs by design, so they never reach the annotations DB and
        # the Spark partition flow above never sees them — they
        # require the dispatcher's taxpub_treatment_extractor fork
        # which reads article.xml directly.  See v3_buildout.md §G.1.
        skip_ids = (
            self.get_existing_ingest_doc_ids() if skip_existing else None
        )
        taxpub_treatments = list(
            self._extract_taxpub_treatments(skip_doc_ids=skip_ids)
        )
        if taxpub_treatments:
            taxpub_rows = list(convert_taxa_to_rows(iter(taxpub_treatments)))
            if taxpub_rows:
                taxpub_df = self.spark.createDataFrame(
                    taxpub_rows, self._extract_schema,
                )
                taxa_df = taxa_df.unionByName(taxpub_df)
                if self.verbosity >= 1:
                    print(
                        f"[G.1] Added {len(taxpub_rows)} taxpub treatments "
                        f"from {self.ingest_db_name}"
                    )
        elif self.verbosity >= 1:
            print(
                f"[G.1] No taxpub treatments to add from "
                f"{self.ingest_db_name}"
            )

        # Step 3: Handle dry run or save taxa to CouchDB
        if dry_run:
            # Dry run - just show what would be saved
            if self.verbosity >= 1:
                taxa_count = taxa_df.count()
                print(f"\n[DRY RUN] Would save {taxa_count} treatments to {self.treatments_db_name}")
                if self.verbosity >= 2:
                    print("\n[DRY RUN] Sample treatments:")
                    taxa_df.select("_id", "treatment", "ingest").show(5, truncate=50)
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
        description="Extract Treatments from CouchDB annotated files and save to CouchDB",
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
  --skip-existing         Skip ingest documents that already have treatments extracted

Environment Variables:
  DRY_RUN=1               Same as --dry-run
  INCREMENTAL=1           Same as --incremental
  INCREMENTAL_BATCH_SIZE=N  Same as --incremental-batch-size
  LIMIT=N                 Same as --limit N
  DOC_IDS=id1,id2,...     Same as --doc-id
  SKIP_EXISTING=1         Same as --skip-existing

Note: All database configuration can be set via command-line arguments to env_config.
      Example: python extract_treatments_to_couchdb.py --ingest-database mydb --taxon-database mytaxa

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
        help="Skip ingest documents that already have treatments extracted"
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
    if not config['ingest_db_name']:
        parser.error("--ingest-db-name is required (or set $INGEST_DB_NAME)")
    if not config['treatments_db_name']:
        parser.error("--treatments-db-name is required (or set $TREATMENTS_DB_NAME)")

    # Default taxon credentials to ingest credentials
    taxon_url = config['taxon_url'] or config['ingest_url']
    taxon_username = config['taxon_username'] or config['ingest_username']
    taxon_password = config['taxon_password'] or config['ingest_password']

    # Create Spark session sized from env_config (dev defaults are
    # 24-core-box-sized; prod overrides via SPARK_CORES /
    # SPARK_DRIVER_MEMORY / SPARK_EXECUTOR_MEMORY in /home/skol/.skol_env).
    # Without these overrides the JVM heap defaults to ~1 GiB and a
    # full-skol_dev extract OOMs at the dedup window-function shuffle.
    cores = config.get('cores', 16)
    driver_memory = config.get('spark_driver_memory', '32g')
    executor_memory = config.get('spark_executor_memory', '16g')
    spark = (
        SparkSession.builder
        .appName("SKOL Treatment Extractor")
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.maxResultSize", "0")
        .getOrCreate()
    )

    if config['verbosity'] >= 1:
        print(f"Extracting treatments from {config['ingest_db_name']} to {config['treatments_db_name']}...")

    # Create extractor instance
    # annotations_db_name is where .ann files live; falls back to ingest_db_name
    # when annotations have not been separated into their own database.
    extractor = TreatmentExtractor(
        spark=spark,
        ingest_couchdb_url=config['ingest_url'],
        ingest_db_name=config['ingest_db_name'],
        annotations_db_name=config.get('annotations_db_name') or config['ingest_db_name'],
        treatments_db_name=config['treatments_db_name'],
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
            print("[SKIP EXISTING] Skipping documents with existing treatments")
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
    total = results.count()
    successes = results.filter("success = true").count()
    failures = results.filter("success = false").count()

    if config['verbosity'] >= 1:
        print(f"\nResults:")
        print(f"  Total treatments: {total}")
        print(f"  Successful saves: {successes}")
        print(f"  Failed saves: {failures}")

        if failures > 0 and config['verbosity'] >= 2:
            print("\nFailed documents:")
            results.filter("success = false").show(truncate=False)

    spark.stop()

    # Exit non-zero on total save failure so manage_experiment's
    # runstep treats this as a real failure rather than marking the
    # pipeline step "completed".  We only hard-fail when *every* save
    # failed (e.g. the NameError-in-closure regression that masked
    # itself as exit-0) — partial transient failures stay non-fatal so
    # the daily cron doesn't alarm on a few CouchDB write conflicts.
    if total > 0 and successes == 0:
        print(
            f"\nERROR: 0/{total} saves succeeded — exiting with error.",
            file=sys.stderr,
        )
        sys.exit(1)
