"""Validation tests for treatments_to_structured/schemas/*.json.

These confirm each hand-authored Phase 1 schema:
  1. is well-formed JSON,
  2. conforms to the JSON Schema 2020-12 metaschema,
  3. accepts the worked-example annotation we designed it from,
  4. rejects obvious malformed input (missing required fields,
     wrong types).

The schemas themselves are non-code artifacts (JSON files), but
treating them as code we test against is the cheapest way to keep
them honest as Phase 1 evolves.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_SCHEMAS_DIR = Path(__file__).resolve().parent / 'schemas'


def _load_schema(name: str) -> Dict[str, Any]:
    """Read a schema JSON file by stem (e.g., 'pileus' → pileus.json)."""
    path = _SCHEMAS_DIR / f'{name}.json'
    with path.open('r') as f:
        return json.load(f)


class TestPileusSchema:
    """The Phase 1 seed schema for Pileus annotations.

    Worked example (from the planning conversation):
    Aureoboletus miniatoaurantiacus,
    taxon_0029f1413f69fd6d270f11e5aa0f806d3ba8687c0c1a5826330988ca90c02c06
    in skol_exp_production_v4_02_00_treatments_prose.

    Source pileus text:
        "Pileus 2.6–8 cm diam, subhemispherical when young, then
        convex to applanate; surface tomentose or pulverous, slightly
        wrinkled, yellowish orange to orange; context white,
        unchanging in colour when injured."
    """

    def test_schema_is_valid_json(self) -> None:
        """Parses without exception."""
        _load_schema('pileus')

    def test_schema_conforms_to_2020_12_metaschema(self) -> None:
        """The schema must itself be a valid JSON Schema 2020-12
        document — caught early via check_schema before we ever try
        to validate data against it."""
        schema = _load_schema('pileus')
        # Raises if the schema is malformed against the metaschema.
        Draft202012Validator.check_schema(schema)

    def test_has_canonical_metadata(self) -> None:
        """$schema, $id, title, description are required JSON Schema
        metadata for any artifact we ship as a contract."""
        schema = _load_schema('pileus')
        assert schema.get('$schema', '').startswith(
            'https://json-schema.org/draft/2020-12/'
        )
        assert schema['$id']
        assert schema['title'] == 'Pileus'
        assert schema['description']

    def test_aureoboletus_example_validates(self) -> None:
        """An annotation derived from the Aureoboletus worked example
        passes validation.  This is the canonical 'happy path' — if
        someone tightens the schema in a way that breaks this, they
        need to update the worked example too."""
        schema = _load_schema('pileus')
        annotation = {
            'size_mm': {'min': 26, 'max': 80},
            'shape': ['subhemispherical', 'convex', 'applanate'],
            'surface_texture': ['tomentose', 'pulverous', 'wrinkled'],
            'color': 'yellowish orange to orange',
            'context_color': 'white',
            'context_change_on_injury': 'unchanging',
        }
        Draft202012Validator(schema).validate(annotation)

    def test_partial_annotation_validates(self) -> None:
        """All Pileus sub-features are optional individually — a
        treatment that records only size + color (omitting texture,
        context, etc.) is still a valid annotation.  This lets the
        LLM emit whatever it actually found in the text without
        having to make things up to fill required slots."""
        schema = _load_schema('pileus')
        partial = {
            'size_mm': {'min': 26, 'max': 80},
            'color': 'orange',
        }
        Draft202012Validator(schema).validate(partial)

    def test_empty_annotation_validates(self) -> None:
        """The trivially empty {} is valid — useful as a degenerate
        case in the LLM pipeline (\"I found no pileus features in
        this treatment\")."""
        schema = _load_schema('pileus')
        Draft202012Validator(schema).validate({})

    def test_size_mm_requires_min_and_max(self) -> None:
        """If you record size at all, both min and max are required.
        A single point measurement should set min == max."""
        schema = _load_schema('pileus')
        bad = {'size_mm': {'min': 26}}  # missing max
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(bad)

    def test_size_mm_with_optional_observed_outliers_validates(
        self,
    ) -> None:
        """The parenthetical-extension form (Pileus 38–75(–120) mm)
        maps to min_observed / max_observed.  Both are optional."""
        schema = _load_schema('pileus')
        annotation = {
            'size_mm': {
                'min': 38, 'max': 75, 'max_observed': 120,
            },
        }
        Draft202012Validator(schema).validate(annotation)

    def test_shape_must_be_non_empty_list(self) -> None:
        """If `shape` is provided, it carries at least one descriptor."""
        schema = _load_schema('pileus')
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate({'shape': []})

    def test_color_must_be_non_empty_string(self) -> None:
        """A blank color field is a data-loss bug, not a deliberate
        omission — omit the property entirely if you have no color."""
        schema = _load_schema('pileus')
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate({'color': ''})

    def test_unknown_property_rejected(self) -> None:
        """additionalProperties: false — typos in field names fail
        loudly rather than silently storing garbage."""
        schema = _load_schema('pileus')
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate({
                'pileus_color': 'orange',  # typo'd field name
            })

    def test_size_mm_negative_rejected(self) -> None:
        """Physical-impossibility guards: a pileus can't have negative
        diameter.  Caught via minimum: 0 on each numeric field."""
        schema = _load_schema('pileus')
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate({
                'size_mm': {'min': -1, 'max': 10},
            })
