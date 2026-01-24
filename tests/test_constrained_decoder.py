"""Unit tests for skol.constrained_decoder module.

Tests for TaxonomySchema, ConstrainedDecoder, and VocabularyNormalizer classes.
These tests do not require GPU or ML model dependencies.
"""

import json
import sys
import unittest
from pathlib import Path

# Add skol directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skol.constrained_decoder import (
    TaxonomySchema,
    ConstrainedDecoder,
    MockBackend,
    VocabularyNormalizer,
)


class TestTaxonomySchema(unittest.TestCase):
    """Test TaxonomySchema JSON schema generation and validation."""

    def test_default_schema_creation(self):
        """Test creating schema with default parameters."""
        schema = TaxonomySchema()
        self.assertEqual(schema.max_depth, 4)
        self.assertEqual(schema.min_depth, 2)

    def test_custom_depth_schema(self):
        """Test creating schema with custom depth parameters."""
        schema = TaxonomySchema(max_depth=3, min_depth=2)
        self.assertEqual(schema.max_depth, 3)
        self.assertEqual(schema.min_depth, 2)

    def test_invalid_depth_raises_error(self):
        """Test that invalid depth parameters raise ValueError."""
        with self.assertRaises(ValueError):
            TaxonomySchema(min_depth=5, max_depth=4)

        with self.assertRaises(ValueError):
            TaxonomySchema(min_depth=1, max_depth=4)

        with self.assertRaises(ValueError):
            TaxonomySchema(min_depth=2, max_depth=5)

    def test_json_schema_structure(self):
        """Test that generated JSON schema has correct structure."""
        schema = TaxonomySchema(max_depth=3, min_depth=2)
        json_schema = schema.to_json_schema()

        # Check top-level structure
        self.assertIn('$schema', json_schema)
        self.assertIn('$defs', json_schema)
        self.assertIn('type', json_schema)
        self.assertEqual(json_schema['type'], 'object')

        # Check definitions exist for each level
        self.assertIn('level1', json_schema['$defs'])
        self.assertIn('level2', json_schema['$defs'])
        self.assertIn('level3', json_schema['$defs'])

    def test_json_schema_level_definitions(self):
        """Test that level definitions have correct structure."""
        schema = TaxonomySchema(max_depth=3, min_depth=2)
        json_schema = schema.to_json_schema()

        # Deepest level should be array only
        level3 = json_schema['$defs']['level3']
        self.assertEqual(level3['type'], 'array')
        self.assertEqual(level3['items']['type'], 'string')

        # Intermediate levels at/above min_depth can be array or object
        level2 = json_schema['$defs']['level2']
        self.assertIn('oneOf', level2)
        self.assertEqual(len(level2['oneOf']), 2)

        # Level below min_depth must be object
        level1 = json_schema['$defs']['level1']
        self.assertEqual(level1['type'], 'object')

    def test_json_schema_string_output(self):
        """Test that to_json_schema_string returns valid JSON."""
        schema = TaxonomySchema()
        json_str = schema.to_json_schema_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertIn('$defs', parsed)

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        schema = TaxonomySchema(max_depth=3, min_depth=2)

        # Valid: dict with array at depth 2
        valid_data = {
            "pileus": {
                "shape": ["convex", "campanulate"]
            }
        }
        self.assertTrue(schema.validate(valid_data))

        # Valid: dict with array at depth 3
        valid_data = {
            "pileus": {
                "surface": {
                    "texture": ["dry", "smooth"]
                }
            }
        }
        self.assertTrue(schema.validate(valid_data))

    def test_validate_invalid_data(self):
        """Test validation of invalid data."""
        schema = TaxonomySchema(max_depth=3, min_depth=2)

        # Invalid: array at depth 1 (below min_depth)
        invalid_data = {
            "colors": ["red", "blue"]  # Array at depth 1
        }
        with self.assertRaises(ValueError):
            schema.validate(invalid_data)

        # Invalid: exceeds max depth
        invalid_data = {
            "a": {"b": {"c": {"d": ["value"]}}}  # Depth 4, max is 3
        }
        with self.assertRaises(ValueError):
            schema.validate(invalid_data)

        # Invalid: empty object
        with self.assertRaises(ValueError):
            schema.validate({})

    def test_validate_non_string_array_items(self):
        """Test that non-string array items are rejected."""
        schema = TaxonomySchema(max_depth=2, min_depth=2)

        invalid_data = {
            "numbers": [1, 2, 3]  # Numbers, not strings
        }
        with self.assertRaises(ValueError):
            schema.validate(invalid_data)


class TestMockBackend(unittest.TestCase):
    """Test MockBackend for testing without GPU dependencies."""

    def test_is_available(self):
        """Test that mock backend is always available."""
        backend = MockBackend()
        self.assertTrue(backend.is_available())

    def test_load_and_generate(self):
        """Test loading and generating with mock backend."""
        backend = MockBackend()
        backend.load("test-model")

        result = backend.generate("Test prompt", {})
        self.assertIsInstance(result, dict)
        self.assertIn("pileus", result)

    def test_generate_without_load_raises_error(self):
        """Test that generating without loading raises error."""
        backend = MockBackend()
        with self.assertRaises(RuntimeError):
            backend.generate("Test prompt", {})

    def test_custom_mock_responses(self):
        """Test setting custom mock responses."""
        backend = MockBackend()
        backend.load("test-model")

        custom_responses = [
            {"feature1": {"sub": ["value1"]}},
            {"feature2": {"sub": ["value2"]}},
        ]
        backend.set_mock_responses(custom_responses)

        # First call returns first response
        result1 = backend.generate("prompt1", {})
        self.assertEqual(result1, custom_responses[0])

        # Second call returns second response
        result2 = backend.generate("prompt2", {})
        self.assertEqual(result2, custom_responses[1])

        # Third call cycles back to first response
        result3 = backend.generate("prompt3", {})
        self.assertEqual(result3, custom_responses[0])


class TestConstrainedDecoder(unittest.TestCase):
    """Test ConstrainedDecoder with mock backend."""

    def test_default_initialization(self):
        """Test default initialization."""
        decoder = ConstrainedDecoder()
        self.assertEqual(decoder.model_name, "mistralai/Mistral-7B-v0.1")
        self.assertEqual(decoder.backend, "outlines")
        self.assertIsInstance(decoder.schema, TaxonomySchema)

    def test_mock_backend_initialization(self):
        """Test initialization with mock backend."""
        decoder = ConstrainedDecoder(backend="mock")
        self.assertEqual(decoder.backend, "mock")

    def test_load_model_with_mock(self):
        """Test loading model with mock backend."""
        decoder = ConstrainedDecoder(backend="mock")
        decoder.load_model()
        # Should not raise any errors

    def test_extract_features_with_mock(self):
        """Test feature extraction with mock backend."""
        decoder = ConstrainedDecoder(backend="mock")
        decoder.load_model()

        result = decoder.extract_features("Pileus convex, 3-5 cm diameter")
        self.assertIsInstance(result, dict)
        self.assertIn("pileus", result)

    def test_batch_extract_with_mock(self):
        """Test batch extraction with mock backend."""
        decoder = ConstrainedDecoder(backend="mock")
        decoder.load_model()

        descriptions = [
            "Pileus convex, 3-5 cm",
            "Stipe cylindrical, white",
            "Lamellae free, crowded"
        ]
        results = decoder.batch_extract(descriptions)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)

    def test_custom_system_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "Extract mushroom features."
        decoder = ConstrainedDecoder(
            backend="mock",
            system_prompt=custom_prompt
        )
        self.assertEqual(decoder.system_prompt, custom_prompt)

    def test_build_prompt(self):
        """Test prompt building."""
        decoder = ConstrainedDecoder(backend="mock")
        prompt = decoder._build_prompt("Test description")

        self.assertIn("Test description", prompt)
        self.assertIn(decoder.system_prompt, prompt)

    def test_build_prompt_with_examples(self):
        """Test prompt building with few-shot examples."""
        decoder = ConstrainedDecoder(backend="mock")
        examples = [
            {"input": "Example input 1", "output": "Example output 1"},
            {"input": "Example input 2", "output": "Example output 2"},
        ]
        prompt = decoder._build_prompt("Test description", examples)

        self.assertIn("Example input 1", prompt)
        self.assertIn("Example output 1", prompt)
        self.assertIn("Examples:", prompt)

    def test_unknown_backend_raises_error(self):
        """Test that unknown backend raises error."""
        decoder = ConstrainedDecoder(backend="unknown")
        with self.assertRaises(ValueError):
            decoder.load_model()


class TestVocabularyNormalizer(unittest.TestCase):
    """Test VocabularyNormalizer.

    Note: These tests use the basic functionality that doesn't require
    sentence-transformers or sklearn. Full normalization tests require
    those dependencies.
    """

    def test_default_initialization(self):
        """Test default initialization."""
        normalizer = VocabularyNormalizer()
        self.assertEqual(normalizer.similarity_threshold, 0.85)
        self.assertEqual(normalizer.embedding_model, "all-MiniLM-L6-v2")

    def test_normalize_unknown_term(self):
        """Test normalizing a term not in vocabulary."""
        normalizer = VocabularyNormalizer()
        # Without fitting, should return original term
        result = normalizer.normalize("unknown_term")
        self.assertEqual(result, "unknown_term")

    def test_normalize_dict_structure_preservation(self):
        """Test that normalize_dict preserves structure."""
        normalizer = VocabularyNormalizer()

        data = {
            "feature": {
                "subfeature": ["value1", "value2"]
            }
        }
        result = normalizer.normalize_dict(data)

        # Structure should be preserved (no normalization without fitting)
        self.assertEqual(result["feature"]["subfeature"], ["value1", "value2"])

    def test_normalize_dict_with_list(self):
        """Test normalize_dict handles lists correctly."""
        normalizer = VocabularyNormalizer()

        data = {
            "features": [
                {"name": ["value1"]},
                {"name": ["value2"]}
            ]
        }
        result = normalizer.normalize_dict(data)

        self.assertEqual(len(result["features"]), 2)
        self.assertEqual(result["features"][0]["name"], ["value1"])

    def test_get_canonical_terms_empty(self):
        """Test getting canonical terms from unfitted normalizer."""
        normalizer = VocabularyNormalizer()
        terms = normalizer.get_canonical_terms()
        self.assertEqual(terms, [])

    def test_save_and_load(self):
        """Test saving and loading canonical mapping."""
        import tempfile
        import os

        normalizer = VocabularyNormalizer()
        # Manually set a mapping for testing
        normalizer._canonical_map = {"term1": "canonical1", "term2": "canonical1"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            normalizer.save(temp_path)

            # Load into new normalizer
            normalizer2 = VocabularyNormalizer()
            normalizer2.load(temp_path)

            self.assertEqual(normalizer2.normalize("term1"), "canonical1")
            self.assertEqual(normalizer2.normalize("term2"), "canonical1")
        finally:
            os.unlink(temp_path)


class TestVocabularyNormalizerWithDependencies(unittest.TestCase):
    """Test VocabularyNormalizer with ML dependencies.

    These tests are skipped if sentence-transformers or sklearn are not available.
    """

    @classmethod
    def setUpClass(cls):
        """Check for required dependencies."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import AgglomerativeClustering
            cls.has_dependencies = True
        except ImportError:
            cls.has_dependencies = False

    def test_fit_with_terms(self):
        """Test fitting normalizer with term counts."""
        if not self.has_dependencies:
            self.skipTest("sentence-transformers or sklearn not available")

        normalizer = VocabularyNormalizer(similarity_threshold=0.7)

        terms_with_counts = {
            "cap shape": 100,
            "pileus shape": 10,
            "cap form": 5,
            "stem color": 80,
            "stipe color": 20,
        }

        normalizer.fit(terms_with_counts)

        # Most frequent term in each cluster should be canonical
        canonical_terms = normalizer.get_canonical_terms()
        self.assertTrue(len(canonical_terms) > 0)

        # "cap shape" should be canonical (most frequent in its cluster)
        self.assertIn("cap shape", canonical_terms)


if __name__ == '__main__':
    unittest.main()
