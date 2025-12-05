"""
Taxa JSON Translator

This module provides a class for translating taxa descriptions into structured JSON
using a fine-tuned Mistral model. Designed to work with PySpark DataFrames from
TaxonExtractor.load_taxa().

The TaxaJSONTranslator class encapsulates model loading, inference, and DataFrame
processing, optimized for batch processing of taxa descriptions.
"""

import json
import io
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType


class TaxaJSONTranslator:
    """
    Translates taxa descriptions to structured JSON using a fine-tuned Mistral model.

    This class is optimized for processing PySpark DataFrames created by
    TaxonExtractor.load_taxa(), adding a new column with JSON-formatted features.

    Example:
        >>> translator = TaxaJSONTranslator(
        ...     spark=spark,
        ...     checkpoint_path="./mistral_checkpoints/checkpoint-100"
        ... )
        >>>
        >>> # Load taxa from CouchDB
        >>> taxa_df = extractor.load_taxa()
        >>>
        >>> # Add JSON column
        >>> enriched_df = translator.translate_descriptions(taxa_df)
        >>>
        >>> # Show results
        >>> enriched_df.select("taxon", "description", "features_json").show()
    """

    # Default configuration
    BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    DEFAULT_MAX_LENGTH = 2048
    DEFAULT_MAX_NEW_TOKENS = 1024

    # Default prompt for feature extraction
    DEFAULT_PROMPT = '''\
    # Task: Extract Taxonomic Features from Species Description

## Objective
Extract features, subfeatures, optional subsubfeatures, and their values from the provided species description and format them as structured JSON.

## Pre-processing Steps
1. If any paragraph is in Latin, translate it to English first
2. Then proceed with feature extraction

## Output Format Requirements

### JSON Structure
- **Level 1 (Top level)**: Feature names (keys)
- **Level 2**: Subfeature names (keys)
- **Level 3 (Optional)**: Subsubfeature names (keys)
- **Add additional levels as needed for deeper nesting**
- **Innermost level**: Arrays of string values

### Critical Rules
1. Values are ALWAYS stored in arrays (lists), even if there's only one value
2. Arrays only appear at the deepest/innermost level of nesting
3. All intermediate levels are objects (dictionaries), not arrays
4. When you encounter comma-separated values, split them into separate array elements

## Instructions
1. Read the entire species description carefully
2. Identify all taxonomic features mentioned
3. Organize them hierarchically (feature → subfeature → subsubfeature if needed)
4. Extract all values and place them in arrays at the innermost level
5. Split any comma-separated values into separate array elements
6. Return valid JSON only, with no additional commentary

'''

    def __init__(
        self,
        spark: SparkSession,
        couchdb_url: str,
        checkpoint_path: Optional[str] = None,
        base_model_id: str = BASE_MODEL_ID,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_PROMPT,
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_auth_token: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize the TaxaJSONTranslator.

        Args:
            spark: SparkSession instance
            couchdb_url: URL of CouchDB server (e.g., "http://localhost:5984")
            checkpoint_path: Path to fine-tuned model checkpoint (if None, uses base model)
            base_model_id: Hugging Face model ID for base model
            max_length: Maximum sequence length for tokenization
            max_new_tokens: Maximum new tokens to generate
            prompt: Instruction prompt for the model
            device: Device to run inference on ("cuda" or "cpu")
            load_in_4bit: Whether to load model in 4-bit quantization
            use_auth_token: Whether to use Hugging Face authentication
            username: Optional username for couchdb authentication
            password: Optional password for couchdb authentication
        """
        self.spark = spark
        self.couchdb_url = couchdb_url
        self.checkpoint_path = checkpoint_path
        self.base_model_id = base_model_id
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.use_auth_token = use_auth_token
        self.username = username
        self.password = password

        # Model and tokenizer (lazy loaded)
        self._model = None
        self._tokenizer = None

        print(f"TaxaJSONTranslator initialized")
        print(f"  CouchDB URL: {couchdb_url}")
        print(f"  Base model: {base_model_id}")
        print(f"  Checkpoint: {checkpoint_path or 'None (using base model)'}")
        print(f"  Device: {device}")
        print(f"  4-bit quantization: {load_in_4bit}")

    def load_taxa(
        self,
        db_name: str,
        pattern: str = "*"
    ) -> DataFrame:
        """
        Load taxa from CouchDB taxon database.

        This method loads taxa documents saved by TaxonExtractor.save_taxa()
        and returns them as a DataFrame compatible with translate_descriptions().

        Args:
            db_name: Name of taxon database
            pattern: Pattern for document IDs to load (default: "*")
                    Use "*" to load all documents
                    Use "taxon_abc*" to load specific subset

        Returns:
            DataFrame with columns:
                - _id: CouchDB document ID (for joining results)
                - taxon: String of concatenated nomenclature paragraphs
                - description: String of concatenated description paragraphs
                - source: Dict with keys doc_id, url, db_name
                - line_number: Line number of first nomenclature paragraph
                - paragraph_number: Paragraph number of first nomenclature paragraph
                - page_number: Page number of first nomenclature paragraph
                - empirical_page_number: Empirical page number of first nomenclature paragraph

        Example:
            >>> translator = TaxaJSONTranslator(
            ...     spark=spark,
            ...     couchdb_url="http://localhost:5984",
            ...     username="admin",
            ...     password="secret",
            ...     checkpoint_path="..."
            ... )
            >>> taxa_df = translator.load_taxa(db_name="mycobank_taxa")
            >>> print(f"Loaded {taxa_df.count()} taxa")
        """
        from skol_classifier.couchdb_io import CouchDBConnection
        from pyspark.sql.types import StructType, StructField, StringType, MapType, IntegerType

        # Define schema with _id for joining results
        schema = StructType([
            StructField("_id", StringType(), False),
            StructField("taxon", StringType(), False),
            StructField("description", StringType(), False),
            StructField("source", MapType(StringType(), StringType(), valueContainsNull=True), False),
            StructField("line_number", IntegerType(), True),
            StructField("paragraph_number", IntegerType(), True),
            StructField("page_number", IntegerType(), True),
            StructField("empirical_page_number", StringType(), True)
        ])

        # Use CouchDBConnection to load data
        conn = CouchDBConnection(self.couchdb_url, db_name, username=self.username, password=self.password)

        # Get matching document IDs
        doc_ids = conn.get_all_doc_ids(pattern)

        if not doc_ids:
            print(f"No documents found matching pattern '{pattern}'")
            return self.spark.createDataFrame([], schema)

        print(f"Loading {len(doc_ids)} taxa from {db_name}...")

        # Create DataFrame with doc_ids for parallel processing
        doc_ids_rdd = self.spark.sparkContext.parallelize(doc_ids)
        doc_ids_df = doc_ids_rdd.map(lambda x: (x,)).toDF(["doc_id"])

        # Prepare connection parameters for workers
        couchdb_url = self.couchdb_url
        username = self.username
        password = self.password

        # Load taxa using mapPartitions
        def load_partition(partition):
            """Load taxa from CouchDB for an entire partition."""
            from skol_classifier.couchdb_io import CouchDBConnection
            from pyspark.sql import Row

            # Create connection using CouchDBConnection API
            conn = CouchDBConnection(couchdb_url, db_name, username, password)

            try:
                db = conn.db

                # Process each row (which contains doc_id)
                for row in partition:
                    try:
                        doc_id = row.doc_id if hasattr(row, 'doc_id') else str(row[0])

                        # Load document from CouchDB
                        if doc_id in db:
                            doc = db[doc_id]

                            # Convert CouchDB document to Row (include _id for joining)
                            taxon_data = {
                                '_id': doc.get('_id', doc_id),
                                'taxon': doc.get('taxon', ''),
                                'description': doc.get('description', ''),
                                'source': doc.get('source', {}),
                                'line_number': doc.get('line_number'),
                                'paragraph_number': doc.get('paragraph_number'),
                                'page_number': doc.get('page_number'),
                                'empirical_page_number': doc.get('empirical_page_number')
                            }

                            yield Row(**taxon_data)
                        else:
                            print(f"Document {doc_id} not found in database")

                    except Exception as e:
                        print(f"Error loading taxon {doc_id}: {e}")

            except Exception as e:
                print(f"Error connecting to CouchDB: {e}")

        taxa_rdd = doc_ids_df.rdd.mapPartitions(load_partition)
        taxa_df = self.spark.createDataFrame(taxa_rdd, schema)

        count = taxa_df.count()
        print(f"✓ Loaded {count} taxa")

        return taxa_df

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes quantization configuration.

        Returns:
            BitsAndBytesConfig object
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def _load_tokenizer(self):
        """Load and configure the tokenizer."""
        print(f"Loading tokenizer from {self.base_model_id}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            model_max_length=self.max_length,
            padding_side="left",
            add_eos_token=True
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        print("✓ Tokenizer loaded")

    def _load_model(self):
        """Load the base model and optionally the fine-tuned checkpoint."""
        print(f"Loading base model from {self.base_model_id}...")

        # Create quantization config
        bnb_config = self._create_bnb_config() if self.load_in_4bit else None

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=self.use_auth_token
        )

        print("✓ Base model loaded")

        # Load fine-tuned checkpoint if provided
        if self.checkpoint_path:
            print(f"Loading fine-tuned checkpoint from {self.checkpoint_path}...")
            self._model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
            print("✓ Fine-tuned model loaded")
        else:
            self._model = base_model
            print("⚠ Using base model (no checkpoint provided)")

        # Set to evaluation mode
        self._model.eval()

        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"✓ Using {torch.cuda.device_count()} GPUs")
            self._model.is_parallelizable = True
            self._model.model_parallel = True

    def _make_prompt(self, description: str) -> str:
        """
        Create a formatted prompt for the model.

        Args:
            description: Taxa description text

        Returns:
            Formatted prompt string
        """
        return f"""<s>[INST]{self.prompt}

## Species Description
{description}[/INST]

Result:
"""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from model output.

        Args:
            text: Model output text containing JSON

        Returns:
            Parsed JSON object (or empty dict if parsing fails)
        """
        state = "START"
        lines = []

        try:
            with io.StringIO(text) as f:
                for line in f:
                    if line.startswith('```json') or line.startswith("result:") or line.startswith("Result:"):
                        state = "RECORDING"
                    elif line.startswith('```'):
                        state = "END"
                        return json.loads("\n".join(lines))
                    elif line.startswith("}"):
                        lines.append(line)
                        state = "END"
                        return json.loads("\n".join(lines))
                    elif state == "RECORDING":
                        lines.append(line)

            # If we didn't return yet, try parsing accumulated lines
            if lines:
                return json.loads("\n".join(lines))

            # Last resort: try parsing entire text as JSON
            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            return {}

    def generate_json(self, description: str) -> str:
        """
        Generate JSON from a taxon description.

        Args:
            description: Taxon description text

        Returns:
            JSON object (or empty JSON object if generation fails)
        """
        try:
            # Create prompt
            prompt = self._make_prompt(description)

            # Tokenize
            model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **model_input,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=2,
                    do_sample=False,  # Deterministic output
                    temperature=None,
                    top_p=None
                )

            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract JSON from output
            json_obj = self._extract_json(generated_text)

            # Return as JSON string
            return json.dumps(json_obj, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Error generating JSON: {e}")
            return "{}"

    def translate_descriptions(
        self,
        taxa_df: DataFrame,
        description_col: str = "description",
        output_col: str = "features_json"
    ) -> DataFrame:
        """
        Add JSON translation column to taxa DataFrame.

        This method processes the DataFrame in a distributed manner, but note that
        model inference is performed on the driver node. For large datasets, consider
        using translate_descriptions_batch() instead.

        Args:
            taxa_df: Input DataFrame from TaxonExtractor.load_taxa()
            description_col: Name of column containing descriptions
            output_col: Name of output column for JSON

        Returns:
            DataFrame with additional JSON column

        Example:
            >>> taxa_df = extractor.load_taxa()
            >>> enriched_df = translator.translate_descriptions(taxa_df)
            >>> enriched_df.select("taxon", "features_json").show(truncate=50)
        """
        print(f"Translating descriptions to JSON...")
        print(f"  Input column: {description_col}")
        print(f"  Output column: {output_col}")

        # Create UDF for translation
        translate_udf = udf(self.generate_json, StringType())

        # Apply UDF
        result_df = taxa_df.withColumn(output_col, translate_udf(col(description_col)))

        count = result_df.count()
        print(f"✓ Translated {count} descriptions")

        return result_df

    def translate_descriptions_batch(
        self,
        taxa_df: DataFrame,
        description_col: str = "description",
        output_col: str = "features_json",
        batch_size: int = 10
    ) -> DataFrame:
        """
        Add JSON translation column to taxa DataFrame using batched processing.

        This method collects descriptions in batches, processes them on the driver,
        and joins the results back. More efficient for moderate-sized datasets.

        Args:
            taxa_df: Input DataFrame from TaxonExtractor.load_taxa()
            description_col: Name of column containing descriptions
            output_col: Name of output column for JSON
            batch_size: Number of descriptions to process at once

        Returns:
            DataFrame with additional JSON column

        Example:
            >>> taxa_df = extractor.load_taxa()
            >>> enriched_df = translator.translate_descriptions_batch(
            ...     taxa_df, batch_size=20
            ... )
        """
        print(f"Translating descriptions to JSON (batch mode)...")
        print(f"  Input column: {description_col}")
        print(f"  Output column: {output_col}")
        print(f"  Batch size: {batch_size}")

        # Collect descriptions with _id for joining
        descriptions = taxa_df.select("_id", "taxon", description_col).collect()
        total = len(descriptions)

        print(f"Processing {total} descriptions...")

        # Process in batches
        results = []
        for i in range(0, total, batch_size):
            batch = descriptions[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

            for row in batch:
                doc_id = row['_id']
                description = row[description_col]

                # Generate JSON
                json_obj = self.generate_json(description)

                results.append({
                    '_id': doc_id,
                    output_col: json_obj
                })

        # Create DataFrame from results
        results_df = self.spark.createDataFrame(results)

        # Join back to original DataFrame on _id
        enriched_df = taxa_df.join(results_df, on="_id", how="left")

        print(f"✓ Translated {total} descriptions")

        return enriched_df

    def translate_single(self, description: str) -> Dict[str, Any]:
        """
        Translate a single description to JSON (as dict).

        Args:
            description: Taxa description text

        Returns:
            Parsed JSON object

        Example:
            >>> description = "Pileus 5-10 cm broad, convex, white..."
            >>> features = translator.translate_single(description)
            >>> print(features.keys())
        """
        json_str = self.generate_json(description)
        return json.loads(json_str)

    def save_taxa(
        self,
        taxa_df: DataFrame,
        db_name: str,
        json_annotated_col: str = "features_json"
    ) -> DataFrame:
        """
        Save taxa DataFrame to CouchDB, including the json_annotated field.

        This method saves taxa with the translated JSON features back to CouchDB.
        It handles arbitrary JSON in the json_annotated_col by parsing it before storage.
        The save operation is idempotent - documents with the same composite key
        (source.doc_id, source.url, line_number) will be updated rather than duplicated.

        Uses credentials from self.username and self.password.

        Args:
            taxa_df: DataFrame with taxa and translations (must include json_annotated_col)
            db_name: Name of taxon database
            json_annotated_col: Name of column containing JSON features (default: "features_json")

        Returns:
            DataFrame with save results (doc_id, success, error_message)

        Example:
            >>> # Load taxa and translate
            >>> taxa_df = translator.load_taxa(db_name="mycobank_taxa")
            >>> enriched_df = translator.translate_descriptions(taxa_df)
            >>>
            >>> # Save back to CouchDB
            >>> results = translator.save_taxa(enriched_df, db_name="mycobank_taxa")
            >>> print(f"Saved: {results.filter('success = true').count()}")
        """
        from pyspark.sql import Row
        from pyspark.sql.types import StructType, StructField, StringType, BooleanType

        # Get credentials from self
        couchdb_url = self.couchdb_url
        username = self.username
        password = self.password

        # Schema for save results
        save_schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("success", BooleanType(), False),
            StructField("error_message", StringType(), False),
        ])

        def save_partition(partition):
            """Save taxa to CouchDB for an entire partition (idempotent)."""
            from skol_classifier.couchdb_io import CouchDBConnection
            import hashlib

            def generate_taxon_doc_id(doc_id: str, url: Optional[str], line_number: int) -> str:
                """Generate deterministic document ID for idempotent saves."""
                key_parts = [
                    doc_id,
                    url if url else "no_url",
                    str(line_number)
                ]
                composite_key = ":".join(key_parts)
                hash_obj = hashlib.sha256(composite_key.encode('utf-8'))
                doc_hash = hash_obj.hexdigest()
                return f"taxon_{doc_hash}"

            # Create connection using CouchDBConnection API
            conn = CouchDBConnection(couchdb_url, db_name, username, password)

            # Connect to CouchDB once per partition
            try:
                # Try to get database, create if it doesn't exist
                import couchdb
                server = couchdb.Server(couchdb_url)
                if username and password:
                    server.resource.credentials = (username, password)

                if db_name not in server:
                    server.create(db_name)

                db = conn.db

                # Process each taxon in the partition
                for row in partition:
                    success = False
                    error_msg = ""
                    doc_id = "unknown"

                    try:
                        # Extract source metadata from row
                        source_dict = row.source if hasattr(row, 'source') else {}
                        source = dict(source_dict) if isinstance(source_dict, dict) else {}
                        source_doc_id = str(source.get('doc_id', 'unknown'))
                        source_url = source.get('url')
                        line_number = row.line_number if hasattr(row, 'line_number') else 0

                        # Generate deterministic document ID
                        doc_id = generate_taxon_doc_id(
                            source_doc_id,
                            source_url if isinstance(source_url, str) else None,
                            int(line_number) if line_number else 0
                        )

                        # Convert row to dict for CouchDB storage
                        taxon_doc = row.asDict()

                        # Handle json_annotated field: parse JSON string to dict
                        if json_annotated_col in taxon_doc and taxon_doc[json_annotated_col]:
                            json_str = taxon_doc[json_annotated_col]
                            if isinstance(json_str, str):
                                try:
                                    # Parse JSON string to dict for storage
                                    taxon_doc['json_annotated'] = json.loads(json_str)
                                except json.JSONDecodeError:
                                    print(f"Warning: Invalid JSON in {json_annotated_col} for doc {doc_id}")
                                    taxon_doc['json_annotated'] = {}
                            else:
                                # Already a dict, just rename the field
                                taxon_doc['json_annotated'] = json_str
                            # Remove the original column if it has a different name
                            if json_annotated_col != 'json_annotated':
                                del taxon_doc[json_annotated_col]

                        # Check if document already exists (idempotent)
                        if doc_id in db:
                            # Document exists - update it
                            existing_doc = db[doc_id]
                            taxon_doc['_id'] = doc_id
                            taxon_doc['_rev'] = existing_doc['_rev']
                        else:
                            # New document - create it
                            taxon_doc['_id'] = doc_id

                        db.save(taxon_doc)
                        success = True

                    except Exception as e:
                        error_msg = str(e)
                        print(f"Error saving taxon {doc_id}: {e}")

                    yield Row(
                        doc_id=doc_id,
                        success=success,
                        error_message=error_msg
                    )

            except Exception as e:
                print(f"Error connecting to CouchDB: {e}")
                # Yield failures for all rows
                for row in partition:
                    yield Row(
                        doc_id="unknown_connection_error",
                        success=False,
                        error_message=str(e)
                    )

        print(f"Saving taxa to {db_name}...")
        results_df = taxa_df.rdd.mapPartitions(save_partition).toDF(save_schema)

        total = results_df.count()
        successes = results_df.filter("success = true").count()
        failures = total - successes

        print(f"✓ Save complete:")
        print(f"  Total: {total}")
        print(f"  Successful: {successes}")
        print(f"  Failed: {failures}")

        return results_df

    def save_translations(
        self,
        translated_df: DataFrame,
        output_path: str,
        format: str = "parquet"
    ):
        """
        Save translated DataFrame to disk.

        Args:
            translated_df: DataFrame with translations
            output_path: Output path
            format: Output format ("parquet", "json", or "csv")

        Example:
            >>> enriched_df = translator.translate_descriptions(taxa_df)
            >>> translator.save_translations(
            ...     enriched_df,
            ...     "output/taxa_with_features.parquet"
            ... )
        """
        print(f"Saving translations to {output_path}...")

        if format == "parquet":
            translated_df.write.mode("overwrite").parquet(output_path)
        elif format == "json":
            translated_df.write.mode("overwrite").json(output_path)
        elif format == "csv":
            translated_df.write.mode("overwrite").csv(output_path, header=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Saved to {output_path}")

    def validate_json(
        self,
        translated_df: DataFrame,
        json_col: str = "features_json"
    ) -> DataFrame:
        """
        Validate JSON column and add validation status.

        Args:
            translated_df: DataFrame with JSON column
            json_col: Name of JSON column

        Returns:
            DataFrame with additional 'json_valid' boolean column

        Example:
            >>> enriched_df = translator.translate_descriptions(taxa_df)
            >>> validated_df = translator.validate_json(enriched_df)
            >>> validated_df.filter("json_valid = false").show()
        """
        def is_valid_json(json_str: str) -> bool:
            try:
                obj = json.loads(json_str)
                return isinstance(obj, dict) and len(obj) > 0
            except:
                return False

        validate_udf = udf(is_valid_json, StringType())

        validated_df = translated_df.withColumn(
            "json_valid",
            validate_udf(col(json_col))
        )

        total = validated_df.count()
        valid = validated_df.filter("json_valid = true").count()
        invalid = total - valid

        print(f"JSON Validation Results:")
        print(f"  Total: {total}")
        print(f"  Valid: {valid} ({100*valid/total:.1f}%)")
        print(f"  Invalid: {invalid} ({100*invalid/total:.1f}%)")

        return validated_df


def example_usage():
    """
    Example usage of TaxaJSONTranslator.

    This function demonstrates the complete workflow:
    1. Initialize Spark
    2. Load taxa from CouchDB
    3. Translate descriptions to JSON
    4. Validate and save results
    """
    from pyspark.sql import SparkSession
    from extract_taxa_to_couchdb import TaxonExtractor

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Taxa JSON Translation") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # Initialize extractor
        extractor = TaxonExtractor(
            spark=spark,
            ingest_couchdb_url="http://localhost:5984",
            ingest_db_name="mycobank_annotations",
            taxon_db_name="mycobank_taxa",
            ingest_username="admin",
            ingest_password="password"
        )

        # Load taxa
        print("Loading taxa from CouchDB...")
        taxa_df = extractor.load_taxa()
        print(f"Loaded {taxa_df.count()} taxa")

        # Initialize translator
        translator = TaxaJSONTranslator(
            spark=spark,
            couchdb_url="http://localhost:5984",
            username="admin",
            password="password",
            checkpoint_path="./mistral_checkpoints/checkpoint-100"
        )

        # Translate descriptions
        enriched_df = translator.translate_descriptions(taxa_df)

        # Show sample results
        print("\nSample results:")
        enriched_df.select("taxon", "features_json").show(5, truncate=50)

        # Validate JSON
        validated_df = translator.validate_json(enriched_df)

        # Save results
        translator.save_translations(
            validated_df,
            "output/taxa_with_features.parquet"
        )

        print("\n✓ Complete!")

    finally:
        spark.stop()


if __name__ == "__main__":
    example_usage()
