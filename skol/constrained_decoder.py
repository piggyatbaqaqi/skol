"""
Constrained Decoding for Structured Feature Extraction

This module provides constrained JSON generation for extracting structured
features from taxonomic descriptions using small language models (SLMs).

The implementation follows the approach described in docs/slm_optimization_discussion.md:
- Uses Outlines library for constrained decoding
- Enforces variable-depth JSON schema (2-4 levels with arrays at leaves)
- Guarantees valid JSON output conforming to the schema

Classes:
    TaxonomySchema: Generates JSON schemas for variable-depth taxonomy features
    ConstrainedDecoder: Wraps Outlines for constrained JSON generation
    VocabularyNormalizer: Normalizes extracted vocabulary using embeddings

Example:
    >>> schema = TaxonomySchema(max_depth=3)
    >>> decoder = ConstrainedDecoder(model_name="mistralai/Mistral-7B-v0.1")
    >>> decoder.load_model()
    >>> result = decoder.generate(
    ...     "Pileus convex, 3-5 cm, surface dry, brown",
    ...     schema=schema.to_json_schema()
    ... )
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaxonomySchema:
    """
    Generate JSON schemas for variable-depth taxonomy feature extraction.

    The schema allows for 2-4 levels of nesting, with string arrays at leaf nodes.
    At each level, the model can choose to terminate with an array or continue nesting.

    Attributes:
        max_depth: Maximum nesting depth (2-4, default 4)
        min_depth: Minimum nesting depth (default 2)

    Example schema output (max_depth=3):
        {
            "feature": {
                "subfeature": ["value1", "value2"]
            }
        }
    """

    max_depth: int = 4
    min_depth: int = 2

    def __post_init__(self):
        if not 2 <= self.min_depth <= self.max_depth <= 4:
            raise ValueError(
                f"Depth must satisfy 2 <= min_depth <= max_depth <= 4, "
                f"got min_depth={self.min_depth}, max_depth={self.max_depth}"
            )

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate a JSON Schema that enforces the variable-depth structure.

        Returns:
            JSON Schema dictionary with $defs for each level
        """
        schema: Dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {},
            "type": "object",
        }

        # Build definitions from deepest level up
        # Level N (deepest) is always an array of strings
        deepest_level = f"level{self.max_depth}"
        schema["$defs"][deepest_level] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }

        # Build intermediate levels (can be array or object with next level)
        for level in range(self.max_depth - 1, 0, -1):
            level_name = f"level{level}"
            next_level = f"level{level + 1}"

            if level >= self.min_depth:
                # Can terminate with array or continue nesting
                schema["$defs"][level_name] = {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        {
                            "type": "object",
                            "additionalProperties": {"$ref": f"#/$defs/{next_level}"},
                            "minProperties": 1,
                        },
                    ]
                }
            else:
                # Must continue nesting (below min_depth)
                schema["$defs"][level_name] = {
                    "type": "object",
                    "additionalProperties": {"$ref": f"#/$defs/{next_level}"},
                    "minProperties": 1,
                }

        # Root level references level1
        schema["additionalProperties"] = {"$ref": "#/$defs/level1"}
        schema["minProperties"] = 1

        return schema

    def to_json_schema_string(self) -> str:
        """Return the JSON schema as a formatted string."""
        return json.dumps(self.to_json_schema(), indent=2)

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate that data conforms to the schema.

        Args:
            data: Dictionary to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        try:
            import jsonschema
            jsonschema.validate(data, self.to_json_schema())
            return True
        except ImportError:
            # Fall back to basic validation if jsonschema not available
            return self._basic_validate(data, depth=1)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation failed: {e.message}")

    def _basic_validate(self, data: Any, depth: int) -> bool:
        """Basic validation without jsonschema library."""
        if depth > self.max_depth:
            raise ValueError(f"Exceeded max depth {self.max_depth}")

        if isinstance(data, list):
            if depth < self.min_depth:
                raise ValueError(f"Array at depth {depth} below min_depth {self.min_depth}")
            if not all(isinstance(item, str) for item in data):
                raise ValueError("Array items must be strings")
            return True

        if isinstance(data, dict):
            if not data:
                raise ValueError("Empty objects not allowed")
            for key, value in data.items():
                if not isinstance(key, str):
                    raise ValueError("Keys must be strings")
                self._basic_validate(value, depth + 1)
            return True

        raise ValueError(f"Invalid type at depth {depth}: {type(data)}")


class DecoderBackend(ABC):
    """Abstract base class for decoder backends."""

    @abstractmethod
    def load(self, model_name: str, **kwargs) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate constrained JSON output."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class OutlinesBackend(DecoderBackend):
    """Backend using the Outlines library for constrained generation."""

    def __init__(self):
        self._model = None
        self._generator = None

    def is_available(self) -> bool:
        try:
            import outlines
            return True
        except ImportError:
            return False

    def load(self, model_name: str, **kwargs) -> None:
        """
        Load model using Outlines.

        Args:
            model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-v0.1")
            **kwargs: Additional arguments passed to outlines.models.transformers
        """
        import outlines

        device = kwargs.pop('device', None)
        model_kwargs = kwargs.pop('model_kwargs', {})

        if device:
            model_kwargs['device_map'] = device

        self._model = outlines.models.transformers(
            model_name,
            model_kwargs=model_kwargs,
            **kwargs
        )

    def generate(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output constrained to schema.

        Args:
            prompt: Input prompt describing the specimen
            schema: JSON Schema to constrain output
            **kwargs: Additional generation arguments

        Returns:
            Dictionary conforming to the schema
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import outlines

        generator = outlines.generate.json(self._model, schema)
        result = generator(prompt, **kwargs)

        return result


class MockBackend(DecoderBackend):
    """Mock backend for testing without GPU/model dependencies."""

    def __init__(self):
        self._loaded = False
        self._mock_responses: List[Dict[str, Any]] = []
        self._response_index = 0

    def is_available(self) -> bool:
        return True

    def load(self, model_name: str, **kwargs) -> None:
        self._loaded = True

    def set_mock_responses(self, responses: List[Dict[str, Any]]) -> None:
        """Set mock responses to return from generate()."""
        self._mock_responses = responses
        self._response_index = 0

    def generate(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._mock_responses:
            response = self._mock_responses[self._response_index % len(self._mock_responses)]
            self._response_index += 1
            return response

        # Default mock response
        return {
            "pileus": {
                "shape": ["convex", "campanulate"],
                "surface": ["dry", "smooth"]
            }
        }


@dataclass
class ConstrainedDecoder:
    """
    Wrapper for constrained JSON generation from taxonomic descriptions.

    This class provides a high-level interface for generating structured
    feature dictionaries from free-text species descriptions.

    Attributes:
        model_name: HuggingFace model name or path
        backend: Decoder backend to use ('outlines' or 'mock')
        schema: TaxonomySchema instance for validation
        device: Device to run on ('cuda', 'cpu', or None for auto)

    Example:
        >>> decoder = ConstrainedDecoder(
        ...     model_name="mistralai/Mistral-7B-v0.1",
        ...     schema=TaxonomySchema(max_depth=3)
        ... )
        >>> decoder.load_model()
        >>> features = decoder.extract_features(
        ...     "Pileus convex, 3-5 cm diameter, surface dry and brown"
        ... )
    """

    model_name: str = "mistralai/Mistral-7B-v0.1"
    backend: str = "outlines"
    schema: TaxonomySchema = field(default_factory=TaxonomySchema)
    device: Optional[str] = None
    system_prompt: str = field(default="")

    _backend_instance: Optional[DecoderBackend] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are a taxonomist extracting structured features from species descriptions.
Extract morphological features into a nested JSON structure.
Use consistent terminology (e.g., "pileus" not "cap", "stipe" not "stem").
Group related features hierarchically.
Values should be descriptive terms, not measurements."""

    def _get_backend(self) -> DecoderBackend:
        """Get or create the decoder backend."""
        if self._backend_instance is not None:
            return self._backend_instance

        if self.backend == "mock":
            self._backend_instance = MockBackend()
        elif self.backend == "outlines":
            backend = OutlinesBackend()
            if not backend.is_available():
                raise ImportError(
                    "Outlines library not available. Install with: pip install outlines"
                )
            self._backend_instance = backend
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._backend_instance

    def load_model(self, **kwargs) -> None:
        """
        Load the language model.

        Args:
            **kwargs: Additional arguments passed to the backend
        """
        backend = self._get_backend()
        backend.load(self.model_name, device=self.device, **kwargs)

    def extract_features(
        self,
        description: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured features from a taxonomic description.

        Args:
            description: Free-text species description
            few_shot_examples: Optional list of {"input": ..., "output": ...} examples
            **kwargs: Additional generation arguments

        Returns:
            Nested dictionary of extracted features
        """
        prompt = self._build_prompt(description, few_shot_examples)
        json_schema = self.schema.to_json_schema()

        backend = self._get_backend()
        result = backend.generate(prompt, json_schema, **kwargs)

        # Validate result
        self.schema.validate(result)

        return result

    def _build_prompt(
        self,
        description: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the full prompt with system message and examples."""
        parts = [self.system_prompt, ""]

        if few_shot_examples:
            parts.append("Examples:")
            for example in few_shot_examples:
                parts.append(f"\nInput: {example['input']}")
                parts.append(f"Output: {example['output']}")
            parts.append("")

        parts.append(f"Extract features from this description:")
        parts.append(f"\n{description}")
        parts.append("\nOutput:")

        return "\n".join(parts)

    def batch_extract(
        self,
        descriptions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple descriptions.

        Args:
            descriptions: List of description texts
            **kwargs: Additional generation arguments

        Returns:
            List of feature dictionaries
        """
        results = []
        for desc in descriptions:
            try:
                result = self.extract_features(desc, **kwargs)
                results.append(result)
            except Exception as e:
                # Return empty dict on failure, log the error
                results.append({"_error": str(e)})
        return results


@dataclass
class VocabularyNormalizer:
    """
    Normalize extracted vocabulary using semantic embeddings.

    This class implements frequency-based canonical selection as described
    in the optimization document. Terms are clustered by semantic similarity,
    and the most frequent term in each cluster becomes canonical.

    Attributes:
        similarity_threshold: Minimum cosine similarity to consider terms related
        embedding_model: Name of sentence-transformers model to use

    Example:
        >>> normalizer = VocabularyNormalizer()
        >>> normalizer.fit({"cap shape": 500, "pileus shape": 50, "cap form": 30})
        >>> normalizer.normalize("pileus shape")
        "cap shape"
    """

    similarity_threshold: float = 0.85
    embedding_model: str = "all-MiniLM-L6-v2"

    _model: Any = field(default=None, init=False, repr=False)
    _canonical_map: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _load_model(self) -> Any:
        """Load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers library required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def fit(self, terms_with_counts: Dict[str, int]) -> "VocabularyNormalizer":
        """
        Build canonical vocabulary mapping from term frequency data.

        Args:
            terms_with_counts: Dictionary mapping terms to their occurrence counts

        Returns:
            Self for method chaining
        """
        if not terms_with_counts:
            return self

        model = self._load_model()

        terms = list(terms_with_counts.keys())
        embeddings = model.encode(terms)

        # Cluster based on semantic similarity
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        # For each cluster, pick the most frequent term as canonical
        self._canonical_map = {}
        for cluster_id in set(labels):
            cluster_terms = [t for t, l in zip(terms, labels) if l == cluster_id]
            canonical = max(cluster_terms, key=lambda t: terms_with_counts[t])
            for term in cluster_terms:
                self._canonical_map[term] = canonical

        return self

    def normalize(self, term: str) -> str:
        """
        Map a term to its canonical form.

        Args:
            term: Term to normalize

        Returns:
            Canonical form of the term, or original if not in vocabulary
        """
        return self._canonical_map.get(term, term)

    def normalize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively normalize all keys in a nested dictionary.

        Args:
            data: Dictionary with potentially non-canonical keys

        Returns:
            Dictionary with normalized keys
        """
        if isinstance(data, dict):
            return {
                self.normalize(k): self.normalize_dict(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.normalize_dict(item) if isinstance(item, dict) else item
                    for item in data]
        return data

    def get_canonical_terms(self) -> List[str]:
        """Return list of all canonical terms."""
        return list(set(self._canonical_map.values()))

    def save(self, path: str) -> None:
        """Save the canonical mapping to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self._canonical_map, f, indent=2)

    def load(self, path: str) -> "VocabularyNormalizer":
        """Load a canonical mapping from a JSON file."""
        with open(path, 'r') as f:
            self._canonical_map = json.load(f)
        return self
