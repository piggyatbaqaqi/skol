#!/usr/bin/env python3
"""
Functional tests for constrained decoding integration.

This script tests the full constrained decoding pipeline including:
1. TaxonomySchema JSON schema generation
2. ConstrainedDecoder with mock backend
3. Integration with TaxaJSONTranslator (mock mode)

Usage:
    python tests/test_constrained_decoder_functional.py

These tests do not require GPU or actual ML models - they use mock backends.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skol.constrained_decoder import (
    TaxonomySchema,
    ConstrainedDecoder,
    MockBackend,
    VocabularyNormalizer,
)


def test_schema_generation():
    """Test that TaxonomySchema generates valid JSON schemas."""
    print("Testing TaxonomySchema generation...")

    # Test default schema
    schema = TaxonomySchema()
    json_schema = schema.to_json_schema()

    assert '$schema' in json_schema, "Missing $schema field"
    assert '$defs' in json_schema, "Missing $defs field"
    assert json_schema['type'] == 'object', "Root type should be object"

    print(f"  Generated schema with levels: {list(json_schema['$defs'].keys())}")

    # Test custom depth schema
    schema2 = TaxonomySchema(max_depth=3, min_depth=2)
    json_schema2 = schema2.to_json_schema()

    assert 'level3' in json_schema2['$defs'], "Missing level3 definition"
    assert 'level4' not in json_schema2['$defs'], "Should not have level4"

    # Verify schema is valid JSON
    json_str = schema.to_json_schema_string()
    parsed = json.loads(json_str)
    assert parsed == json_schema, "JSON string should match dict"

    print("  Schema generation tests passed!")
    return True


def test_schema_validation():
    """Test TaxonomySchema validation of data."""
    print("Testing TaxonomySchema validation...")

    schema = TaxonomySchema(max_depth=3, min_depth=2)

    # Valid data: nested dict with arrays at leaf
    valid_data = {
        "pileus": {
            "shape": ["convex", "campanulate"],
            "surface": ["dry", "smooth"]
        },
        "stipe": {
            "color": ["white", "cream"]
        }
    }

    try:
        result = schema.validate(valid_data)
        assert result is True, "Valid data should return True"
        print("  Valid data accepted")
    except ValueError as e:
        print(f"  ERROR: Valid data rejected: {e}")
        return False

    # Invalid data: array at wrong depth
    invalid_data = {
        "colors": ["red", "blue"]  # Array at depth 1, but min_depth is 2
    }

    try:
        schema.validate(invalid_data)
        print("  ERROR: Invalid data should have been rejected")
        return False
    except ValueError:
        print("  Invalid data correctly rejected (array at wrong depth)")

    # Invalid data: exceeds max depth
    invalid_data2 = {
        "a": {"b": {"c": {"d": ["value"]}}}  # Depth 4, max is 3
    }

    try:
        schema.validate(invalid_data2)
        print("  ERROR: Invalid data should have been rejected")
        return False
    except ValueError:
        print("  Invalid data correctly rejected (exceeds max depth)")

    print("  Schema validation tests passed!")
    return True


def test_mock_decoder():
    """Test ConstrainedDecoder with mock backend."""
    print("Testing ConstrainedDecoder with mock backend...")

    decoder = ConstrainedDecoder(
        model_name="test-model",
        backend="mock",
        schema=TaxonomySchema(max_depth=3, min_depth=2)
    )

    # Load model
    decoder.load_model()
    print("  Model loaded (mock)")

    # Test single extraction
    description = "Pileus convex, 3-5 cm diameter, surface dry, brown"
    result = decoder.extract_features(description)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert len(result) > 0, "Result should not be empty"
    print(f"  Single extraction result: {json.dumps(result, indent=2)}")

    # Test batch extraction
    descriptions = [
        "Pileus convex, white to cream",
        "Stipe cylindrical, hollow",
        "Lamellae free, crowded, pink"
    ]
    results = decoder.batch_extract(descriptions)

    assert len(results) == 3, "Should have 3 results"
    print(f"  Batch extraction returned {len(results)} results")

    # Test with custom mock responses
    backend = decoder._get_backend()
    custom_responses = [
        {"cap": {"shape": ["convex"]}},
        {"stem": {"color": ["white"]}},
    ]
    backend.set_mock_responses(custom_responses)

    result1 = decoder.extract_features("test1")
    result2 = decoder.extract_features("test2")

    assert result1 == custom_responses[0], "First response should match"
    assert result2 == custom_responses[1], "Second response should match"
    print("  Custom mock responses work correctly")

    print("  Mock decoder tests passed!")
    return True


def test_prompt_building():
    """Test prompt construction."""
    print("Testing prompt building...")

    decoder = ConstrainedDecoder(backend="mock")

    # Basic prompt
    prompt = decoder._build_prompt("Test description text")
    assert "Test description text" in prompt, "Prompt should contain description"
    assert decoder.system_prompt in prompt, "Prompt should contain system prompt"
    print("  Basic prompt built correctly")

    # Prompt with few-shot examples
    examples = [
        {"input": "Species X has convex cap", "output": '{"cap": {"shape": ["convex"]}}'},
        {"input": "Species Y has white stem", "output": '{"stem": {"color": ["white"]}}'},
    ]
    prompt_with_examples = decoder._build_prompt("Test description", examples)

    assert "Examples:" in prompt_with_examples, "Should have Examples section"
    assert "Species X" in prompt_with_examples, "Should contain example input"
    print("  Prompt with few-shot examples built correctly")

    print("  Prompt building tests passed!")
    return True


def test_vocabulary_normalizer_basic():
    """Test VocabularyNormalizer basic functionality."""
    print("Testing VocabularyNormalizer basic functions...")

    normalizer = VocabularyNormalizer()

    # Test normalize without fitting (should return original)
    result = normalizer.normalize("unknown_term")
    assert result == "unknown_term", "Unfitted normalizer should return original"
    print("  Unfitted normalizer returns original terms")

    # Test normalize_dict
    data = {
        "pileus": {
            "shape": ["convex"],
            "color": ["brown", "tan"]
        }
    }
    normalized = normalizer.normalize_dict(data)
    assert normalized == data, "Unfitted normalizer should not change dict"
    print("  normalize_dict preserves structure")

    # Test manual canonical mapping
    normalizer._canonical_map = {
        "cap": "pileus",
        "stem": "stipe",
        "gills": "lamellae"
    }

    assert normalizer.normalize("cap") == "pileus"
    assert normalizer.normalize("stem") == "stipe"
    assert normalizer.normalize("gills") == "lamellae"
    assert normalizer.normalize("unknown") == "unknown"
    print("  Manual canonical mapping works correctly")

    # Test get_canonical_terms
    canonical = normalizer.get_canonical_terms()
    assert "pileus" in canonical
    assert "stipe" in canonical
    print(f"  Canonical terms: {canonical}")

    print("  VocabularyNormalizer basic tests passed!")
    return True


def test_vocabulary_normalizer_dict():
    """Test VocabularyNormalizer with nested dictionaries."""
    print("Testing VocabularyNormalizer with nested dicts...")

    normalizer = VocabularyNormalizer()
    normalizer._canonical_map = {
        "cap": "pileus",
        "stem": "stipe",
    }

    data = {
        "cap": {
            "shape": ["convex"]
        },
        "stem": {
            "color": ["white"]
        }
    }

    normalized = normalizer.normalize_dict(data)

    assert "pileus" in normalized, "cap should be normalized to pileus"
    assert "stipe" in normalized, "stem should be normalized to stipe"
    assert normalized["pileus"]["shape"] == ["convex"]
    print(f"  Normalized: {json.dumps(normalized, indent=2)}")

    print("  VocabularyNormalizer dict normalization passed!")
    return True


def test_full_pipeline_mock():
    """Test the full constrained decoding pipeline with mocks."""
    print("Testing full pipeline with mock backend...")

    # Create schema
    schema = TaxonomySchema(max_depth=3, min_depth=2)

    # Create decoder
    decoder = ConstrainedDecoder(
        model_name="test-model",
        backend="mock",
        schema=schema
    )

    # Set up realistic mock responses
    mock_responses = [
        {
            "pileus": {
                "shape": ["convex", "campanulate"],
                "surface": ["dry", "smooth"]
            },
            "stipe": {
                "color": ["white", "cream"]
            }
        }
    ]

    decoder.load_model()
    decoder._get_backend().set_mock_responses(mock_responses)

    # Process a description
    description = """
    Pileus 3-5 cm broad, convex to campanulate, surface dry and smooth,
    brown to tan. Stipe 5-8 cm long, cylindrical, white to cream,
    surface fibrillose.
    """

    result = decoder.extract_features(description)

    # Validate result against schema
    is_valid = schema.validate(result)
    assert is_valid, "Result should be valid according to schema"

    print(f"  Extracted features: {json.dumps(result, indent=2)}")
    print("  Result validated against schema")

    # Test with normalizer
    normalizer = VocabularyNormalizer()
    normalizer._canonical_map = {
        "pileus": "cap",
        "stipe": "stem"
    }

    normalized = normalizer.normalize_dict(result)
    print(f"  Normalized features: {json.dumps(normalized, indent=2)}")

    print("  Full pipeline test passed!")
    return True


def main():
    """Run all functional tests."""
    print("=" * 70)
    print("Constrained Decoder Functional Tests")
    print("=" * 70)
    print()

    tests = [
        ("Schema Generation", test_schema_generation),
        ("Schema Validation", test_schema_validation),
        ("Mock Decoder", test_mock_decoder),
        ("Prompt Building", test_prompt_building),
        ("Vocabulary Normalizer Basic", test_vocabulary_normalizer_basic),
        ("Vocabulary Normalizer Dict", test_vocabulary_normalizer_dict),
        ("Full Pipeline Mock", test_full_pipeline_mock),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print()
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"  EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll functional tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
