"""Unit tests for skol.ontology module.

Tests for OntologyTerm, OntologyIndex, OntologyRegistry, OntologyContextBuilder,
and OntologyGuidedGenerator classes.

These tests include:
- Basic functionality tests that don't require external dependencies
- Tests with mocked ontology data
- Integration tests that require sentence-transformers (skipped if not available)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Add skol directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skol.ontology import (
    OntologyTerm,
    OntologyIndex,
    OntologyRegistry,
    OntologyContextBuilder,
    OntologyGuidedGenerator,
    load_mock_registry,
    # Phase 4: Vocabulary Normalization
    NormalizationResult,
    CoverageAnalysis,
    ThresholdGatedNormalizer,
    VocabularyAnalyzer,
    normalize_json_output,
    print_coverage_report,
)


class TestOntologyTerm(unittest.TestCase):
    """Test OntologyTerm dataclass."""

    def test_term_creation(self):
        """Test creating an ontology term."""
        embedding = np.array([0.1, 0.2, 0.3])
        term = OntologyTerm(
            id="PATO:0000001",
            name="quality",
            definition="A dependent entity",
            depth=0,
            ancestors=[],
            embedding=embedding
        )

        self.assertEqual(term.id, "PATO:0000001")
        self.assertEqual(term.name, "quality")
        self.assertEqual(term.depth, 0)
        np.testing.assert_array_equal(term.embedding, embedding)

    def test_term_equality(self):
        """Test term equality based on ID."""
        embedding = np.array([0.1, 0.2, 0.3])
        term1 = OntologyTerm(
            id="PATO:0000001", name="quality", definition=None,
            depth=0, ancestors=[], embedding=embedding
        )
        term2 = OntologyTerm(
            id="PATO:0000001", name="quality", definition="Different def",
            depth=1, ancestors=["root"], embedding=embedding
        )
        term3 = OntologyTerm(
            id="PATO:0000002", name="different", definition=None,
            depth=0, ancestors=[], embedding=embedding
        )

        self.assertEqual(term1, term2)  # Same ID
        self.assertNotEqual(term1, term3)  # Different ID

    def test_term_hash(self):
        """Test term hashing for use in sets/dicts."""
        embedding = np.array([0.1, 0.2, 0.3])
        term1 = OntologyTerm(
            id="PATO:0000001", name="quality", definition=None,
            depth=0, ancestors=[], embedding=embedding
        )
        term2 = OntologyTerm(
            id="PATO:0000001", name="quality", definition=None,
            depth=0, ancestors=[], embedding=embedding
        )

        # Should be usable in a set
        term_set = {term1, term2}
        self.assertEqual(len(term_set), 1)


class TestOntologyIndex(unittest.TestCase):
    """Test OntologyIndex class."""

    def test_empty_index_creation(self):
        """Test creating an empty index."""
        index = OntologyIndex(name="test")

        self.assertEqual(index.name, "test")
        self.assertEqual(len(index.terms), 0)
        self.assertEqual(index.category, "general")

    def test_search_empty_index(self):
        """Test searching an empty index returns empty results."""
        index = OntologyIndex(name="test")
        results = index.search("query")

        self.assertEqual(results, [])

    def test_get_terms_at_depth_empty(self):
        """Test getting terms at depth from empty index."""
        index = OntologyIndex(name="test")
        results = index.get_terms_at_depth(0, 5)

        self.assertEqual(results, [])

    def test_get_term_not_found(self):
        """Test getting a term that doesn't exist."""
        index = OntologyIndex(name="test")
        result = index.get_term("NONEXISTENT:0001")

        self.assertIsNone(result)

    def test_len(self):
        """Test __len__ method."""
        index = OntologyIndex(name="test")
        self.assertEqual(len(index), 0)


class TestOntologyIndexWithMockData(unittest.TestCase):
    """Test OntologyIndex with mocked data (no real ontology files)."""

    def setUp(self):
        """Set up mock index with test data."""
        self.index = OntologyIndex(name="mock")
        self.index.category = "base"

        # Create mock terms with embeddings
        self.mock_terms = [
            OntologyTerm(
                id="TEST:0001",
                name="shape",
                definition="The shape of something",
                depth=1,
                ancestors=["quality"],
                embedding=np.array([1.0, 0.0, 0.0])
            ),
            OntologyTerm(
                id="TEST:0002",
                name="round",
                definition="Having a circular shape",
                depth=2,
                ancestors=["quality", "shape"],
                embedding=np.array([0.9, 0.1, 0.0])
            ),
            OntologyTerm(
                id="TEST:0003",
                name="color",
                definition="The color of something",
                depth=1,
                ancestors=["quality"],
                embedding=np.array([0.0, 1.0, 0.0])
            ),
            OntologyTerm(
                id="TEST:0004",
                name="brown",
                definition="A brown color",
                depth=2,
                ancestors=["quality", "color"],
                embedding=np.array([0.0, 0.9, 0.1])
            ),
        ]

        self.index.terms = self.mock_terms
        self.index.term_embeddings = np.stack([t.embedding for t in self.mock_terms])

        # Build lookup
        for term in self.mock_terms:
            self.index._term_lookup[term.id] = term
            self.index._term_lookup[term.name.lower()] = term

    def test_get_term_by_id(self):
        """Test getting a term by ID."""
        term = self.index.get_term("TEST:0001")

        self.assertIsNotNone(term)
        self.assertEqual(term.name, "shape")

    def test_get_term_by_name(self):
        """Test getting a term by name."""
        term = self.index.get_term("round")

        self.assertIsNotNone(term)
        self.assertEqual(term.id, "TEST:0002")

    def test_get_terms_at_depth(self):
        """Test filtering terms by depth."""
        depth_1 = self.index.get_terms_at_depth(1, 1)
        depth_2 = self.index.get_terms_at_depth(2, 2)

        self.assertEqual(len(depth_1), 2)  # shape, color
        self.assertEqual(len(depth_2), 2)  # round, brown

    def test_len_with_terms(self):
        """Test __len__ with terms."""
        self.assertEqual(len(self.index), 4)


class TestOntologyRegistry(unittest.TestCase):
    """Test OntologyRegistry class."""

    def test_empty_registry(self):
        """Test creating an empty registry."""
        registry = OntologyRegistry()

        self.assertEqual(len(registry.ontologies), 0)
        self.assertEqual(registry.list_registered(), [])

    def test_register_index(self):
        """Test registering a pre-built index."""
        registry = OntologyRegistry()
        index = OntologyIndex(name="test")
        index.category = "base"

        registry.register_index("test", index)

        self.assertIn("test", registry)
        self.assertEqual(registry.get("test"), index)

    def test_get_nonexistent_raises_error(self):
        """Test that getting nonexistent ontology raises KeyError."""
        registry = OntologyRegistry()

        with self.assertRaises(KeyError):
            registry.get("nonexistent")

    def test_list_registered(self):
        """Test listing registered ontologies."""
        registry = OntologyRegistry()

        index1 = OntologyIndex(name="ont1")
        index1.category = "base"
        index2 = OntologyIndex(name="ont2")
        index2.category = "specialized"

        registry.register_index("ont1", index1)
        registry.register_index("ont2", index2)

        listing = registry.list_registered()

        self.assertEqual(len(listing), 2)
        names = [l["name"] for l in listing]
        self.assertIn("ont1", names)
        self.assertIn("ont2", names)

    def test_get_base_ontologies(self):
        """Test getting only base ontologies."""
        registry = OntologyRegistry()

        base_index = OntologyIndex(name="base")
        base_index.category = "base"
        spec_index = OntologyIndex(name="spec")
        spec_index.category = "specialized"

        registry.register_index("base", base_index)
        registry.register_index("spec", spec_index)

        base_onts = registry.get_base_ontologies()

        self.assertEqual(len(base_onts), 1)
        self.assertEqual(base_onts[0].name, "base")

    def test_get_specialized_ontologies(self):
        """Test getting only specialized ontologies."""
        registry = OntologyRegistry()

        base_index = OntologyIndex(name="base")
        base_index.category = "base"
        spec_index = OntologyIndex(name="spec")
        spec_index.category = "specialized"

        registry.register_index("base", base_index)
        registry.register_index("spec", spec_index)

        spec_onts = registry.get_specialized_ontologies()

        self.assertEqual(len(spec_onts), 1)
        self.assertEqual(spec_onts[0].name, "spec")

    def test_contains(self):
        """Test __contains__ method."""
        registry = OntologyRegistry()
        index = OntologyIndex(name="test")

        registry.register_index("test", index)

        self.assertIn("test", registry)
        self.assertNotIn("other", registry)


class TestOntologyContextBuilder(unittest.TestCase):
    """Test OntologyContextBuilder class."""

    def setUp(self):
        """Set up registry with mock indices."""
        self.registry = load_mock_registry()

    def test_build_context_empty_registry(self):
        """Test building context with empty indices."""
        builder = OntologyContextBuilder(self.registry)
        context = builder.build_context("Pileus convex, brown")

        # Should return something even with empty indices
        self.assertIsInstance(context, str)

    def test_build_context_missing_ontology(self):
        """Test building context when ontology is not registered."""
        empty_registry = OntologyRegistry()
        builder = OntologyContextBuilder(empty_registry)

        # Should not raise error, just return empty/partial context
        context = builder.build_context(
            "Test description",
            anatomy_ontology="nonexistent",
            quality_ontology="nonexistent"
        )

        self.assertIsInstance(context, str)

    def test_build_minimal_context(self):
        """Test building minimal context."""
        builder = OntologyContextBuilder(self.registry)
        context = builder.build_minimal_context("Test description")

        self.assertIsInstance(context, str)
        self.assertTrue(context.startswith("Preferred vocabulary:"))

    def test_context_truncation(self):
        """Test that context is truncated when too long."""
        builder = OntologyContextBuilder(self.registry)

        # Use very small max_context_chars to force truncation
        context = builder.build_context(
            "Test description",
            max_context_chars=10
        )

        self.assertLessEqual(len(context), 30)  # 10 + "[truncated]"


class TestOntologyContextBuilderWithMockData(unittest.TestCase):
    """Test OntologyContextBuilder with populated mock indices."""

    def setUp(self):
        """Set up registry with mock data."""
        self.registry = OntologyRegistry()

        # Create mock anatomy index
        anatomy = OntologyIndex(name="fao")
        anatomy.category = "base"
        anatomy.terms = [
            OntologyTerm(
                id="FAO:0001", name="pileus", definition="Cap of mushroom",
                depth=1, ancestors=["fungal structure"],
                embedding=np.array([1.0, 0.0, 0.0])
            ),
            OntologyTerm(
                id="FAO:0002", name="stipe", definition="Stem of mushroom",
                depth=1, ancestors=["fungal structure"],
                embedding=np.array([0.8, 0.2, 0.0])
            ),
        ]
        anatomy.term_embeddings = np.stack([t.embedding for t in anatomy.terms])

        # Create mock quality index
        quality = OntologyIndex(name="pato")
        quality.category = "base"
        quality.terms = [
            OntologyTerm(
                id="PATO:0001", name="shape", definition="Shape quality",
                depth=1, ancestors=["quality"],
                embedding=np.array([0.0, 1.0, 0.0])
            ),
            OntologyTerm(
                id="PATO:0002", name="convex", definition="Convex shape",
                depth=2, ancestors=["quality", "shape"],
                embedding=np.array([0.0, 0.9, 0.1])
            ),
        ]
        quality.term_embeddings = np.stack([t.embedding for t in quality.terms])

        self.registry.register_index("fao", anatomy)
        self.registry.register_index("pato", quality)

        # Mock encoder for search
        self.mock_encoder = MagicMock()
        self.mock_encoder.encode.return_value = np.array([0.5, 0.5, 0.0])

        anatomy._encoder = self.mock_encoder
        quality._encoder = self.mock_encoder

    def test_format_hierarchy(self):
        """Test hierarchy formatting."""
        builder = OntologyContextBuilder(self.registry)

        results = [
            (self.registry.get("pato").terms[0], 0.9),  # depth 1
            (self.registry.get("pato").terms[1], 0.8),  # depth 2
        ]

        lines = builder._format_hierarchy(results)

        self.assertTrue(len(lines) > 0)
        self.assertTrue(any("[L" in line for line in lines))

    def test_format_path_examples(self):
        """Test path example formatting."""
        builder = OntologyContextBuilder(self.registry)

        results = [
            (self.registry.get("pato").terms[1], 0.8),  # has ancestors
        ]

        lines = builder._format_path_examples(results)

        self.assertTrue(len(lines) > 0)
        self.assertTrue(any(">" in line for line in lines))


class TestOntologyGuidedGenerator(unittest.TestCase):
    """Test OntologyGuidedGenerator class."""

    def setUp(self):
        """Set up mock decoder and registry."""
        self.registry = load_mock_registry()

        # Mock decoder
        self.mock_decoder = MagicMock()
        self.mock_decoder.extract_features.return_value = {
            "pileus": {
                "shape": ["convex"]
            }
        }

        # Mock schema
        self.mock_schema = MagicMock()

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = OntologyGuidedGenerator(
            self.mock_decoder,
            self.registry,
            self.mock_schema
        )

        self.assertEqual(generator.decoder, self.mock_decoder)
        self.assertEqual(generator.schema, self.mock_schema)

    def test_generate(self):
        """Test basic generation."""
        generator = OntologyGuidedGenerator(
            self.mock_decoder,
            self.registry,
            self.mock_schema
        )

        result = generator.generate("Pileus convex, brown")

        self.assertIsInstance(result, dict)
        self.mock_decoder.extract_features.assert_called_once()

    def test_generate_with_context(self):
        """Test generation that returns context."""
        generator = OntologyGuidedGenerator(
            self.mock_decoder,
            self.registry,
            self.mock_schema
        )

        result, context = generator.generate_with_context("Pileus convex")

        self.assertIsInstance(result, dict)
        self.assertIsInstance(context, str)

    def test_generate_with_context_override(self):
        """Test generation with custom context."""
        generator = OntologyGuidedGenerator(
            self.mock_decoder,
            self.registry,
            self.mock_schema
        )

        custom_context = "Custom ontology context here"
        result, context = generator.generate_with_context(
            "Pileus convex",
            context_override=custom_context
        )

        self.assertEqual(context, custom_context)

    def test_custom_prompt_template(self):
        """Test using a custom prompt template."""
        custom_template = "Context: {ontology_context}\nDesc: {description}\n"

        generator = OntologyGuidedGenerator(
            self.mock_decoder,
            self.registry,
            self.mock_schema,
            prompt_template=custom_template
        )

        self.assertEqual(generator.prompt_template, custom_template)


class TestLoadMockRegistry(unittest.TestCase):
    """Test the load_mock_registry convenience function."""

    def test_creates_registry_with_mock_indices(self):
        """Test that mock registry is created with expected indices."""
        registry = load_mock_registry()

        self.assertIn("pato", registry)
        self.assertIn("fao", registry)

    def test_mock_indices_are_empty(self):
        """Test that mock indices have no terms."""
        registry = load_mock_registry()

        self.assertEqual(len(registry.get("pato").terms), 0)
        self.assertEqual(len(registry.get("fao").terms), 0)

    def test_mock_indices_have_correct_category(self):
        """Test that mock indices are marked as base."""
        registry = load_mock_registry()

        self.assertEqual(registry.get("pato").category, "base")
        self.assertEqual(registry.get("fao").category, "base")


class TestIntegrationWithRealEncoder(unittest.TestCase):
    """Integration tests that require sentence-transformers.

    These tests are skipped if the library is not available.
    """

    @classmethod
    def setUpClass(cls):
        """Check for required dependencies."""
        try:
            from sentence_transformers import SentenceTransformer
            cls.has_encoder = True
        except ImportError:
            cls.has_encoder = False

    def test_index_search_with_real_encoder(self):
        """Test semantic search with real encoder."""
        if not self.has_encoder:
            self.skipTest("sentence-transformers not available")

        # Create index and manually add terms with real embeddings
        index = OntologyIndex(name="test")

        # The encoder property will load the real encoder
        encoder = index.encoder

        # Create test terms with real embeddings
        texts = ["round shape", "brown color", "smooth texture"]
        embeddings = encoder.encode(texts)

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            term = OntologyTerm(
                id=f"TEST:{i}",
                name=text,
                definition=f"Test term {i}",
                depth=1,
                ancestors=["root"],
                embedding=emb
            )
            index.terms.append(term)
            index._term_lookup[term.id] = term

        index.term_embeddings = np.stack([t.embedding for t in index.terms])

        # Search for similar term
        results = index.search("circular", top_k=3)

        self.assertEqual(len(results), 3)
        # "round shape" should be most similar to "circular"
        top_term, top_score = results[0]
        self.assertEqual(top_term.name, "round shape")
        self.assertGreater(top_score, 0)


# =============================================================================
# Phase 4: Vocabulary Normalization Tests
# =============================================================================


class TestNormalizationResult(unittest.TestCase):
    """Test NormalizationResult dataclass."""

    def test_result_creation_normalized(self):
        """Test creating a normalized result."""
        result = NormalizationResult(
            original="convex cap",
            normalized="convex",
            similarity=0.92,
            was_normalized=True,
            ontology_source="pato",
            ontology_term_id="PATO:0001234"
        )

        self.assertEqual(result.original, "convex cap")
        self.assertEqual(result.normalized, "convex")
        self.assertTrue(result.was_normalized)
        self.assertEqual(result.ontology_source, "pato")

    def test_result_creation_not_normalized(self):
        """Test creating a result where term was not normalized."""
        result = NormalizationResult(
            original="subcanaliculate",
            normalized="subcanaliculate",
            similarity=0.45,
            was_normalized=False
        )

        self.assertEqual(result.original, result.normalized)
        self.assertFalse(result.was_normalized)
        self.assertIsNone(result.ontology_source)


class TestCoverageAnalysis(unittest.TestCase):
    """Test CoverageAnalysis dataclass."""

    def test_analysis_creation(self):
        """Test creating a coverage analysis."""
        details = [
            ("convex", "convex", 0.95),
            ("brown", "brown", 0.88),
            ("subcanaliculate", "(no match)", 0.32),
        ]

        analysis = CoverageAnalysis(
            total_terms=3,
            well_covered=2,
            partially_covered=0,
            novel=1,
            coverage_ratio=2/3,
            term_details=details
        )

        self.assertEqual(analysis.total_terms, 3)
        self.assertEqual(analysis.well_covered, 2)
        self.assertEqual(analysis.novel, 1)
        self.assertAlmostEqual(analysis.coverage_ratio, 0.667, places=2)


class TestThresholdGatedNormalizer(unittest.TestCase):
    """Test ThresholdGatedNormalizer class."""

    def setUp(self):
        """Set up registry with mock data for normalization tests."""
        self.registry = OntologyRegistry()

        # Create mock index with embeddings
        index = OntologyIndex(name="test")
        index.category = "base"

        # Create terms with specific embeddings for predictable similarity
        self.mock_terms = [
            OntologyTerm(
                id="TEST:0001", name="convex", definition="Convex shape",
                depth=2, ancestors=["shape"], embedding=np.array([1.0, 0.0, 0.0])
            ),
            OntologyTerm(
                id="TEST:0002", name="brown", definition="Brown color",
                depth=2, ancestors=["color"], embedding=np.array([0.0, 1.0, 0.0])
            ),
            OntologyTerm(
                id="TEST:0003", name="smooth", definition="Smooth texture",
                depth=2, ancestors=["texture"], embedding=np.array([0.0, 0.0, 1.0])
            ),
        ]

        index.terms = self.mock_terms
        index.term_embeddings = np.stack([t.embedding for t in self.mock_terms])

        # Build lookup
        for term in self.mock_terms:
            index._term_lookup[term.id] = term
            index._term_lookup[term.name.lower()] = term

        # Mock encoder to return predictable embeddings
        self.mock_encoder = MagicMock()
        index._encoder = self.mock_encoder

        self.registry.register_index("test", index)
        self.registry._encoder = self.mock_encoder

    def test_normalizer_creation(self):
        """Test creating a normalizer."""
        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)

        self.assertEqual(normalizer.threshold, 0.85)
        self.assertEqual(normalizer.min_similarity, 0.5)

    def test_normalize_high_similarity(self):
        """Test normalizing a term with high similarity (above threshold)."""
        # Make encoder return embedding very similar to "convex"
        self.mock_encoder.encode.return_value = np.array([0.99, 0.01, 0.0])

        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)
        result = normalizer.normalize("convex shape")

        self.assertTrue(result.was_normalized)
        self.assertEqual(result.normalized, "convex")
        self.assertGreater(result.similarity, 0.85)

    def test_normalize_low_similarity(self):
        """Test normalizing a term with low similarity (below threshold)."""
        # Make encoder return embedding not similar to any term
        self.mock_encoder.encode.return_value = np.array([0.33, 0.33, 0.33])

        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)
        result = normalizer.normalize("subcanaliculate")

        self.assertFalse(result.was_normalized)
        self.assertEqual(result.normalized, "subcanaliculate")  # Preserved

    def test_normalize_caching(self):
        """Test that normalization results are cached."""
        self.mock_encoder.encode.return_value = np.array([0.99, 0.01, 0.0])

        normalizer = ThresholdGatedNormalizer(self.registry)

        # First call
        result1 = normalizer.normalize("test term")
        # Second call with same term
        result2 = normalizer.normalize("test term")

        # Encoder should only be called once due to caching
        self.assertEqual(self.mock_encoder.encode.call_count, 1)
        self.assertEqual(result1.normalized, result2.normalized)

    def test_normalize_batch(self):
        """Test batch normalization."""
        self.mock_encoder.encode.return_value = np.array([0.99, 0.01, 0.0])

        normalizer = ThresholdGatedNormalizer(self.registry)
        results = normalizer.normalize_batch(["term1", "term2", "term3"])

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], NormalizationResult)

    def test_analyze_coverage(self):
        """Test coverage analysis."""
        # Set up encoder to return different similarities
        call_count = [0]
        def mock_encode(text):
            call_count[0] += 1
            # First term: high similarity
            if call_count[0] == 1:
                return np.array([0.99, 0.01, 0.0])
            # Second term: medium similarity
            elif call_count[0] == 2:
                return np.array([0.7, 0.3, 0.0])
            # Third term: low similarity
            else:
                return np.array([0.33, 0.33, 0.33])

        self.mock_encoder.encode.side_effect = mock_encode

        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)
        analysis = normalizer.analyze_coverage(["high", "medium", "low"])

        self.assertEqual(analysis.total_terms, 3)
        # Results depend on embedding similarities

    def test_clear_cache(self):
        """Test clearing the normalization cache."""
        self.mock_encoder.encode.return_value = np.array([0.99, 0.01, 0.0])

        normalizer = ThresholdGatedNormalizer(self.registry)
        normalizer.normalize("test")

        self.assertEqual(len(normalizer._cache), 1)

        normalizer.clear_cache()

        self.assertEqual(len(normalizer._cache), 0)


class TestVocabularyAnalyzer(unittest.TestCase):
    """Test VocabularyAnalyzer class."""

    def setUp(self):
        """Set up registry with mock data."""
        self.registry = OntologyRegistry()

        index = OntologyIndex(name="test")
        index.category = "base"
        index.terms = [
            OntologyTerm(
                id="TEST:0001", name="pileus", definition="Cap",
                depth=1, ancestors=[], embedding=np.array([1.0, 0.0])
            ),
        ]
        index.term_embeddings = np.stack([t.embedding for t in index.terms])

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([0.9, 0.1])
        index._encoder = mock_encoder

        self.registry.register_index("test", index)
        self.registry._encoder = mock_encoder

    def test_extract_terms_from_dict(self):
        """Test extracting terms from nested dict."""
        analyzer = VocabularyAnalyzer(self.registry)

        data = {
            "pileus": {
                "shape": ["convex", "flat"],
                "color": ["brown"]
            }
        }

        terms = analyzer.extract_terms(data)

        self.assertIn("pileus", terms)
        self.assertIn("shape", terms)
        self.assertIn("convex", terms)
        self.assertIn("flat", terms)
        self.assertIn("color", terms)
        self.assertIn("brown", terms)

    def test_extract_terms_from_list(self):
        """Test extracting terms from list."""
        analyzer = VocabularyAnalyzer(self.registry)

        data = ["term1", "term2", "term3"]
        terms = analyzer.extract_terms(data)

        self.assertEqual(len(terms), 3)

    def test_extract_terms_from_string(self):
        """Test extracting terms from single string."""
        analyzer = VocabularyAnalyzer(self.registry)

        terms = analyzer.extract_terms("single term")

        self.assertEqual(terms, ["single term"])

    def test_analyze_json(self):
        """Test analyzing JSON features."""
        analyzer = VocabularyAnalyzer(self.registry)

        data = {
            "pileus": {
                "shape": ["convex"]
            }
        }

        analysis = analyzer.analyze_json(data)

        self.assertIsInstance(analysis, CoverageAnalysis)
        self.assertGreater(analysis.total_terms, 0)

    def test_get_novel_terms(self):
        """Test getting novel/poorly-covered terms."""
        # Mock encoder to return low similarity
        self.registry._encoder.encode.return_value = np.array([0.1, 0.1])

        analyzer = VocabularyAnalyzer(self.registry, threshold=0.85)

        data = {"pileus": {"novel_term": ["value"]}}
        novel = analyzer.get_novel_terms(data)

        # Should find some novel terms
        self.assertIsInstance(novel, list)


class TestNormalizeJsonOutput(unittest.TestCase):
    """Test normalize_json_output function."""

    def setUp(self):
        """Set up mock normalizer."""
        self.registry = OntologyRegistry()

        index = OntologyIndex(name="test")
        index.category = "base"
        index.terms = [
            OntologyTerm(
                id="TEST:0001", name="convex", definition="Convex",
                depth=1, ancestors=[], embedding=np.array([1.0, 0.0])
            ),
        ]
        index.term_embeddings = np.stack([t.embedding for t in index.terms])

        mock_encoder = MagicMock()
        # Return high similarity for "convex", low for others
        def mock_encode(text):
            if "convex" in text.lower():
                return np.array([0.99, 0.01])
            return np.array([0.5, 0.5])
        mock_encoder.encode.side_effect = mock_encode
        index._encoder = mock_encoder

        self.registry.register_index("test", index)
        self.registry._encoder = mock_encoder

    def test_normalize_values_only(self):
        """Test normalizing only values."""
        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)

        data = {"shape": ["convex shape"]}

        normalized, log = normalize_json_output(
            data, normalizer,
            normalize_keys=False,
            normalize_values=True
        )

        # Keys should be unchanged
        self.assertIn("shape", normalized)
        # Values should be in the log (the original value, not the key)
        self.assertIn("convex shape", log)

    def test_normalize_keys_only(self):
        """Test normalizing only keys."""
        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)

        data = {"convex_key": ["value"]}

        normalized, log = normalize_json_output(
            data, normalizer,
            normalize_keys=True,
            normalize_values=False
        )

        # Original key should be logged
        self.assertIn("convex_key", log)

    def test_normalize_nested_structure(self):
        """Test normalizing deeply nested structure."""
        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)

        data = {
            "level1": {
                "level2": {
                    "level3": ["value"]
                }
            }
        }

        normalized, log = normalize_json_output(data, normalizer)

        # Should process all levels
        self.assertIn("level1", normalized)
        self.assertIn("level2", normalized["level1"])

    def test_normalize_preserves_structure(self):
        """Test that normalization preserves the structure."""
        normalizer = ThresholdGatedNormalizer(self.registry, threshold=0.85)

        data = {
            "key1": ["val1", "val2"],
            "key2": {"nested": ["val3"]}
        }

        normalized, _ = normalize_json_output(data, normalizer)

        # Structure should be preserved
        self.assertIsInstance(normalized["key1"], list)
        self.assertIsInstance(normalized["key2"], dict)
        self.assertIsInstance(normalized["key2"]["nested"], list)


class TestPrintCoverageReport(unittest.TestCase):
    """Test print_coverage_report function."""

    def test_report_generation(self):
        """Test generating a coverage report."""
        analysis = CoverageAnalysis(
            total_terms=10,
            well_covered=7,
            partially_covered=2,
            novel=1,
            coverage_ratio=0.7,
            term_details=[
                ("convex", "convex", 0.95),
                ("brown", "brown", 0.88),
                ("novel", "(no match)", 0.32),
            ]
        )

        report = print_coverage_report(analysis, show_details=True, max_details=10)

        self.assertIn("VOCABULARY COVERAGE ANALYSIS", report)
        self.assertIn("Total unique terms: 10", report)
        self.assertIn("Coverage ratio: 70.0%", report)

    def test_report_without_details(self):
        """Test generating a report without term details."""
        analysis = CoverageAnalysis(
            total_terms=5,
            well_covered=3,
            partially_covered=1,
            novel=1,
            coverage_ratio=0.6,
            term_details=[]
        )

        report = print_coverage_report(analysis, show_details=False)

        self.assertIn("VOCABULARY COVERAGE ANALYSIS", report)
        self.assertNotIn("TERM DETAILS", report)


if __name__ == '__main__':
    unittest.main()
