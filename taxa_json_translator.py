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
import multiprocessing as mp
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


def _inference_worker(descriptions, model_config, batch_size, result_queue, streaming=False):
    """
    Worker function that runs model inference in an isolated subprocess.

    Must be defined at module level for pickling with 'spawn' context.

    Args:
        descriptions: List of dicts with '_id', 'taxon', 'description' keys
        model_config: Dict with model configuration
        batch_size: Number of descriptions per batch (for progress reporting)
        result_queue: Queue to send results back
        streaming: If True, send each result immediately; if False, batch all results
    """
    try:
        # Import inside subprocess to avoid Spark context
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import json as json_module
        import io

        # Load tokenizer
        print("    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model_id'],
            model_max_length=model_config['max_length'],
            padding_side="left",
            add_eos_token=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        print("    Loading model...")
        bnb_config = None
        if model_config['load_in_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model_id'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=model_config['use_auth_token']
        )

        if model_config['checkpoint_path']:
            model = PeftModel.from_pretrained(base_model, model_config['checkpoint_path'])
        else:
            model = base_model

        model.eval()
        print("    ✓ Model ready")

        # Helper to make prompt
        def make_prompt(description):
            return f"""<s>[INST]{model_config['prompt']}

## Species Description
{description}[/INST]

Result:
"""

        # Helper to extract JSON
        def extract_json(text):
            state = "START"
            lines = []
            try:
                with io.StringIO(text) as f:
                    for line in f:
                        if line.startswith('```json') or line.startswith("result:") or line.startswith("Result:"):
                            state = "RECORDING"
                        elif line.startswith('```'):
                            state = "END"
                            return json_module.loads("\n".join(lines))
                        elif line.startswith("}"):
                            lines.append(line)
                            state = "END"
                            return json_module.loads("\n".join(lines))
                        elif state == "RECORDING":
                            lines.append(line)
                if lines:
                    return json_module.loads("\n".join(lines))
                return json_module.loads(text)
            except json_module.JSONDecodeError:
                return {}

        # Process descriptions
        results = {}
        total = len(descriptions)
        for i, item in enumerate(descriptions):
            doc_id = item['_id']
            description = item['description']

            # Progress reporting
            if i % batch_size == 0:
                batch_num = i // batch_size + 1
                total_batches = (total + batch_size - 1) // batch_size
                print(f"    Batch {batch_num}/{total_batches}")

            try:
                prompt = make_prompt(description)
                model_input = tokenizer(prompt, return_tensors="pt").to(model_config['device'])

                with torch.no_grad():
                    output = model.generate(
                        **model_input,
                        max_new_tokens=model_config['max_new_tokens'],
                        pad_token_id=2,
                        do_sample=False,
                        temperature=None,
                        top_p=None
                    )

                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                json_obj = extract_json(generated_text)
                json_str = json_module.dumps(json_obj, ensure_ascii=False)

                if streaming:
                    # Send result immediately with full item data for saving
                    result_queue.put(('result', {
                        '_id': doc_id,
                        'taxon': item.get('taxon', ''),
                        'description': description,
                        'source': item.get('source', {}),
                        'line_number': item.get('line_number'),
                        'paragraph_number': item.get('paragraph_number'),
                        'page_number': item.get('page_number'),
                        'empirical_page_number': item.get('empirical_page_number'),
                        'features_json': json_str,
                        'index': i,
                        'total': total
                    }))
                else:
                    results[doc_id] = json_str

            except Exception as e:
                print(f"    Warning: Error processing {doc_id}: {e}")
                if streaming:
                    result_queue.put(('result', {
                        '_id': doc_id,
                        'taxon': item.get('taxon', ''),
                        'description': description,
                        'source': item.get('source', {}),
                        'line_number': item.get('line_number'),
                        'paragraph_number': item.get('paragraph_number'),
                        'page_number': item.get('page_number'),
                        'empirical_page_number': item.get('empirical_page_number'),
                        'features_json': '{}',
                        'index': i,
                        'total': total,
                        'error': str(e)
                    }))
                else:
                    results[doc_id] = "{}"

        if streaming:
            result_queue.put(('done', None))
        else:
            result_queue.put(('success', results))

    except Exception as e:
        import traceback
        result_queue.put(('error', str(e) + "\n" + traceback.format_exc()))


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

        # Track last loaded database for save_taxa default
        self._last_loaded_db_name = None

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

        # Store the database name for use as default in save_taxa
        self._last_loaded_db_name = db_name

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

        # Pre-load model and tokenizer before UDF execution
        print("  Pre-loading model and tokenizer...")
        _ = self.tokenizer  # Force lazy load
        _ = self.model      # Force lazy load
        print("  ✓ Model ready")

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

        # Step 1: Collect all data from Spark to pure Python
        print("  Collecting data from Spark...")
        rows = taxa_df.select("_id", "taxon", description_col).collect()

        # Convert to pure Python list of dicts (no Spark Row objects)
        descriptions = [
            {'_id': row['_id'], 'taxon': row['taxon'], 'description': row[description_col]}
            for row in rows
        ]
        total = len(descriptions)
        print(f"  ✓ Collected {total} descriptions")

        # Step 2: Run model inference in a subprocess to isolate from Spark daemon
        # This is necessary because Spark's Python daemon processes can conflict
        # with CUDA operations. Using a subprocess works in both local and distributed modes.
        print("  Running model inference in isolated subprocess...")

        results = self._run_inference_subprocess(descriptions, batch_size)

        print(f"✓ Generated JSON for {len(results)} descriptions")

        # Step 3: Add results to DataFrame using broadcast variable
        print("  Adding JSON results to DataFrame...")
        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import StringType

        results_broadcast = self.spark.sparkContext.broadcast(results)

        def lookup_json(doc_id):
            return results_broadcast.value.get(doc_id, "{}")

        lookup_udf = udf(lookup_json, StringType())
        enriched_df = taxa_df.withColumn(output_col, lookup_udf(col("_id")))

        print(f"✓ Translated {total} descriptions")

        return enriched_df

    def _run_inference_subprocess(
        self,
        descriptions: list,
        batch_size: int
    ) -> dict:
        """
        Run model inference in a subprocess isolated from Spark.

        This avoids conflicts between Spark's Python daemon and CUDA operations.
        Works in both local and distributed Spark modes.

        Args:
            descriptions: List of dicts with '_id' and 'description' keys
            batch_size: Number of descriptions to process per batch

        Returns:
            Dict mapping doc_id to JSON string
        """
        import queue

        # Capture model config for subprocess
        model_config = {
            'base_model_id': self.base_model_id,
            'checkpoint_path': self.checkpoint_path,
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'prompt': self.prompt,
            'device': self.device,
            'load_in_4bit': self.load_in_4bit,
            'use_auth_token': self.use_auth_token,
        }

        # Use spawn to ensure clean subprocess without Spark daemon threads
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        # Start subprocess using module-level function (required for pickling with spawn)
        proc = ctx.Process(
            target=_inference_worker,
            args=(descriptions, model_config, batch_size, result_queue)
        )
        proc.start()

        # Wait for result
        try:
            status, result = result_queue.get(timeout=3600)  # 1 hour timeout
            proc.join(timeout=60)
        except queue.Empty:
            proc.terminate()
            raise RuntimeError("Model inference subprocess timed out")

        if status == 'error':
            raise RuntimeError(f"Model inference failed: {result}")

        return result

    def translate_and_save_streaming(
        self,
        taxa_df: DataFrame,
        db_name: str,
        description_col: str = "description",
        batch_size: int = 10,
        validate: bool = True,
        verbosity: int = 1
    ) -> dict:
        """
        Translate descriptions and save to CouchDB incrementally as each record completes.

        This method processes taxa one at a time, saving each to the database immediately
        after translation. This provides:
        - Real-time progress visibility in the database
        - Crash resilience (completed records are already saved)
        - Better progress monitoring

        Args:
            taxa_df: Input DataFrame from load_taxa()
            db_name: Destination CouchDB database name
            description_col: Name of column containing descriptions
            batch_size: Batch size for progress reporting (not for saving)
            validate: If True, validate JSON before saving
            verbosity: Verbosity level (0=silent, 1=info, 2=debug)

        Returns:
            Dict with 'success_count', 'failure_count', 'total' keys

        Example:
            >>> taxa_df = translator.load_taxa(db_name="skol_taxa_dev")
            >>> results = translator.translate_and_save_streaming(
            ...     taxa_df,
            ...     db_name="skol_taxa_full",
            ...     verbosity=2
            ... )
            >>> print(f"Saved {results['success_count']} of {results['total']}")
        """
        import queue
        import hashlib
        import couchdb

        print(f"Translating and saving incrementally to {db_name}...")
        print(f"  Input column: {description_col}")
        print(f"  Batch size (for progress): {batch_size}")

        # Step 1: Collect all data from Spark to pure Python
        if verbosity >= 1:
            print("  Collecting data from Spark...")
        rows = taxa_df.select(
            "_id", "taxon", description_col, "source",
            "line_number", "paragraph_number", "page_number", "empirical_page_number"
        ).collect()

        # Convert to pure Python list of dicts
        descriptions = []
        for row in rows:
            item = {
                '_id': row['_id'],
                'taxon': row['taxon'],
                'description': row[description_col],
                'source': dict(row['source']) if row['source'] else {},
                'line_number': row['line_number'],
                'paragraph_number': row['paragraph_number'],
                'page_number': row['page_number'],
                'empirical_page_number': row['empirical_page_number']
            }
            descriptions.append(item)

        total = len(descriptions)
        if verbosity >= 1:
            print(f"  ✓ Collected {total} descriptions")

        # Step 2: Connect to CouchDB
        if verbosity >= 1:
            print(f"  Connecting to CouchDB...")
        server = couchdb.Server(self.couchdb_url)
        if self.username and self.password:
            server.resource.credentials = (self.username, self.password)

        # Create database if it doesn't exist
        if db_name not in server:
            server.create(db_name)
        db = server[db_name]

        if verbosity >= 1:
            print(f"  ✓ Connected to {db_name}")

        # Helper to generate deterministic doc ID
        def generate_taxon_doc_id(doc_id: str, url, line_number) -> str:
            key_parts = [
                doc_id,
                url if url else "no_url",
                str(line_number) if line_number else "0"
            ]
            composite_key = ":".join(key_parts)
            hash_obj = hashlib.sha256(composite_key.encode('utf-8'))
            doc_hash = hash_obj.hexdigest()
            return f"taxon_{doc_hash}"

        # Helper to validate JSON
        def is_valid_json(json_str: str) -> bool:
            try:
                obj = json.loads(json_str)
                return isinstance(obj, dict) and len(obj) > 0
            except:
                return False

        # Step 3: Capture model config for subprocess
        model_config = {
            'base_model_id': self.base_model_id,
            'checkpoint_path': self.checkpoint_path,
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'prompt': self.prompt,
            'device': self.device,
            'load_in_4bit': self.load_in_4bit,
            'use_auth_token': self.use_auth_token,
        }

        # Step 4: Start subprocess with streaming=True
        if verbosity >= 1:
            print("  Starting model inference subprocess...")
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        proc = ctx.Process(
            target=_inference_worker,
            args=(descriptions, model_config, batch_size, result_queue, True)  # streaming=True
        )
        proc.start()

        # Step 5: Process results as they arrive
        success_count = 0
        failure_count = 0
        validation_failures = 0

        try:
            while True:
                try:
                    status, data = result_queue.get(timeout=600)  # 10 min timeout per record
                except queue.Empty:
                    print("  ⚠ Timeout waiting for result")
                    break

                if status == 'done':
                    break
                elif status == 'error':
                    raise RuntimeError(f"Model inference failed: {data}")
                elif status == 'result':
                    # Process and save this record
                    try:
                        source = data.get('source', {})
                        source_doc_id = str(source.get('doc_id', 'unknown'))
                        # Use human_url to match extract_taxa_to_couchdb.py doc ID generation
                        source_url = source.get('human_url')
                        line_number = data.get('line_number')

                        # Generate deterministic document ID
                        doc_id = generate_taxon_doc_id(
                            source_doc_id,
                            source_url if isinstance(source_url, str) else None,
                            line_number
                        )

                        # Validate JSON if requested
                        json_str = data.get('features_json', '{}')
                        json_valid = True
                        if validate and not is_valid_json(json_str):
                            json_valid = False
                            validation_failures += 1

                        # Parse JSON for storage
                        try:
                            json_annotated = json.loads(json_str)
                        except json.JSONDecodeError:
                            json_annotated = {}

                        # Build document
                        taxon_doc = {
                            '_id': doc_id,
                            'taxon': data.get('taxon', ''),
                            'description': data.get('description', ''),
                            'source': source,
                            'line_number': line_number,
                            'paragraph_number': data.get('paragraph_number'),
                            'page_number': data.get('page_number'),
                            'empirical_page_number': data.get('empirical_page_number'),
                            'json_annotated': json_annotated,
                            'json_valid': json_valid
                        }

                        # Check if document exists (for update)
                        if doc_id in db:
                            existing_doc = db[doc_id]
                            taxon_doc['_rev'] = existing_doc['_rev']

                        # Save to CouchDB
                        db.save(taxon_doc)
                        success_count += 1

                        # Progress reporting
                        idx = data.get('index', 0) + 1
                        if verbosity >= 1:
                            print(f"  ✓ [{idx}/{total}] Saved {doc_id[:40]}...")
                        if not json_valid:
                            taxon_preview = data.get('taxon', '')[:60]
                            desc_text = data.get('description', '')
                            desc_preview = desc_text[:150] if len(desc_text) > 150 else desc_text
                            print(f"    ⚠ Invalid JSON for {doc_id}")
                            print(f"      Taxon: {taxon_preview}...")
                            print(f"      Description: {desc_preview}...")
                            # Log the actual generated JSON for debugging
                            json_preview = json_str[:200] if len(json_str) > 200 else json_str
                            print(f"      Generated: {json_preview}")

                    except Exception as e:
                        failure_count += 1
                        if verbosity >= 1:
                            print(f"  ✗ Error saving record: {e}")

        finally:
            # Clean up subprocess
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)

        # Summary
        print(f"\n{'='*70}")
        print("Streaming Translation Complete!")
        print(f"{'='*70}")
        print(f"✓ Successfully saved: {success_count}")
        if validation_failures > 0:
            print(f"⚠ Validation failures: {validation_failures} (still saved)")
        if failure_count > 0:
            print(f"✗ Failed to save: {failure_count}")
        print(f"Total processed: {total}")

        return {
            'success_count': success_count,
            'failure_count': failure_count,
            'validation_failures': validation_failures,
            'total': total
        }

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
        db_name: Optional[str] = None,
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
            db_name: Name of taxon database. If None, uses the database from the most
                    recent load_taxa() call. If no database was loaded and db_name is
                    None, raises ValueError.
            json_annotated_col: Name of column containing JSON features (default: "features_json")

        Returns:
            DataFrame with save results (doc_id, success, error_message)

        Example:
            >>> # Load taxa and translate
            >>> taxa_df = translator.load_taxa(db_name="mycobank_taxa")
            >>> enriched_df = translator.translate_descriptions(taxa_df)
            >>>
            >>> # Save back to same database (default)
            >>> results = translator.save_taxa(enriched_df)
            >>> print(f"Saved: {results.filter('success = true').count()}")
            >>>
            >>> # Or save to a different database
            >>> results = translator.save_taxa(enriched_df, db_name="mycobank_taxa_v2")
        """
        # Resolve db_name: use provided value or fall back to last loaded
        if db_name is None:
            if self._last_loaded_db_name is None:
                raise ValueError(
                    "db_name must be specified when no database has been loaded via load_taxa()"
                )
            db_name = self._last_loaded_db_name
            print(f"Using database from last load_taxa(): {db_name}")
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
