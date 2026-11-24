"""Tests for skol_classifier/label_set.py (E6-0, Experiment 6 plan §8.5).

The v4 label set must reproduce today's module constants exactly, so
moving a consumer onto it changes nothing.  The constants stay; these
tests pin them to the file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import treatment  # noqa: E402
from ingestors.jats_to_yedda import (  # noqa: E402
    JATS_EMIT_TAGS,
    sec_type_to_tag,
)
from ingestors.yedda_tags import ACTIVE_TAGS_19, Tag  # noqa: E402
from skol_classifier.label_set import (  # noqa: E402
    LabelSet,
    LabelSetMismatch,
    available_label_sets,
    check_declaration,
    load_label_set,
)
from skol_classifier.v4 import (  # noqa: E402
    crf_layout,
    crf_single,
    crf_treatment,
)
from skol_classifier.v4.labels import (  # noqa: E402
    ACTIVE_YEDDA_TAGS,
    LAYOUT_YEDDA_TAGS,
    TREATMENT_YEDDA_TAGS,
    map_yedda_to_active,
    map_yedda_to_layout,
    map_yedda_to_treatment,
)

# Every tag the enum knows (deprecated ones included), case variants,
# and labels no v4 model has seen.
PROBE_TAGS = (
    [t.value for t in Tag]
    + [t.value.lower() for t in Tag]
    + [t.value.upper() for t in Tag]
    + ['Habitat-distribution', 'MB', 'Unclassified', '']
)

# A spread of real and odd TaxPub sec-type values.
PROBE_SEC_TYPES = [
    'description', 'Description of the new species', 'etymology',
    'diagnosis', 'Diagnosis (Latin)', 'holotype', 'type species',
    'material', 'specimens examined', 'Additional specimen examined',
    'biology', 'habitat-distribution', 'host', 'phylogeny',
    'new-combinations', 'plates', 'notes', 'comments', 'key',
    'Key to species', 'basidiospores', 'colonies', 'references',
    'literature cited', 'table', 'table 1', 'materials_and_methods',
    'methods', '﻿notes', '  Description  ', 'something else',
]

PROJECTIONS = {
    'layout': (LAYOUT_YEDDA_TAGS, map_yedda_to_layout,
               crf_layout.LAYOUT_LABELS),
    'treatment': (TREATMENT_YEDDA_TAGS, map_yedda_to_treatment,
                  crf_treatment.TREATMENT_LABELS),
    'active': (ACTIVE_YEDDA_TAGS, map_yedda_to_active,
               crf_single.ACTIVE_LABELS),
}


@pytest.fixture
def v4() -> LabelSet:
    return load_label_set('v4')


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def test_v4_is_available() -> None:
    assert 'v4' in available_label_sets()


def test_unknown_version_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match='v4'):
        load_label_set('no-such-version')


def test_v4_carries_its_date(v4: LabelSet) -> None:
    assert v4.version == 'v4'
    assert v4.date


# ---------------------------------------------------------------------
# v4 equals today's constants
# ---------------------------------------------------------------------

def test_tags_are_active_tags_19_in_order(v4: LabelSet) -> None:
    assert v4.tags == tuple(t.value for t in ACTIVE_TAGS_19)


def test_deprecated_tags_name_their_replacements(v4: LabelSet) -> None:
    assert dict(v4.deprecated) == {
        'Holotype': 'Type-designation',
        'Distribution': 'Biology',
        'FIX': None,
    }


@pytest.mark.parametrize('name', sorted(PROJECTIONS))
def test_projection_members_match(v4: LabelSet, name: str) -> None:
    members, _, _ = PROJECTIONS[name]
    assert v4.projections[name].members == members


@pytest.mark.parametrize('name', sorted(PROJECTIONS))
def test_projection_label_order_matches_the_crf(v4: LabelSet, name: str) -> None:
    # Index identity: a checkpoint's output i means the same label.
    _, _, crf_labels = PROJECTIONS[name]
    assert v4.projections[name].labels == crf_labels
    assert v4.label_map(name) == {
        label: i for i, label in enumerate(crf_labels)}


@pytest.mark.parametrize('name', sorted(PROJECTIONS))
@pytest.mark.parametrize('tag', PROBE_TAGS)
def test_project_agrees_with_labels_py(v4: LabelSet, name: str, tag: str) -> None:
    _, mapper, _ = PROJECTIONS[name]
    assert v4.project(name, tag) == mapper(tag)


def test_fields_match_treatment_py(v4: LabelSet) -> None:
    assert dict(v4.fields) == treatment._LABEL_TO_FIELD
    assert dict(v4.spans_fields) == treatment._LABEL_TO_SPANS_FIELD


def test_jats_emit_matches(v4: LabelSet) -> None:
    assert set(v4.jats_emit) == {t.value for t in JATS_EMIT_TAGS}


@pytest.mark.parametrize('sec_type', PROBE_SEC_TYPES)
def test_jats_sec_type_matches(v4: LabelSet, sec_type: str) -> None:
    assert v4.sec_type_to_tag()(sec_type) == sec_type_to_tag(sec_type).value


def test_unknown_projection_raises(v4: LabelSet) -> None:
    with pytest.raises(KeyError):
        v4.project('no-such-projection', 'Description')


# ---------------------------------------------------------------------
# Digest: what a model declares, so an edit in place is caught
# ---------------------------------------------------------------------

def _v4_dict() -> Dict[str, Any]:
    import json
    path = (Path(__file__).resolve().parent / 'label_sets' / 'v4.json')
    return dict(json.loads(path.read_text()))


def test_digest_is_stable(v4: LabelSet) -> None:
    assert load_label_set('v4').digest == v4.digest
    assert len(v4.digest) == 64


def test_digest_changes_with_content(v4: LabelSet) -> None:
    data = _v4_dict()
    data['projections']['treatment']['fallback'] = 'Unclassified'
    assert LabelSet.from_dict(data).digest != v4.digest


def test_digest_ignores_key_order(v4: LabelSet) -> None:
    data = _v4_dict()
    reordered = dict(reversed(list(data.items())))
    assert LabelSet.from_dict(reordered).digest == v4.digest


# ---------------------------------------------------------------------
# Declaration and refusal
# ---------------------------------------------------------------------

def test_declaration_names_version_digest_projection(v4: LabelSet) -> None:
    assert v4.declaration('layout') == {
        'label_set': 'v4',
        'label_set_digest': v4.digest,
        'projection': 'layout',
    }


def test_matching_declaration_is_accepted(v4: LabelSet) -> None:
    meta = dict(v4.declaration('active'))
    assert check_declaration(meta, v4, 'active') == 'declared'


@pytest.mark.parametrize('key,value', [
    ('label_set', 'v6'),
    ('label_set_digest', '0' * 64),
    ('projection', 'treatment'),
])
def test_mismatched_declaration_is_refused(v4: LabelSet, key: str, value: str) -> None:
    meta = dict(v4.declaration('active'))
    meta[key] = value
    with pytest.raises(LabelSetMismatch):
        check_declaration(meta, v4, 'active')


def test_undeclared_v4_model_is_inferred(v4: LabelSet) -> None:
    # Models already in Redis were saved before E6-0.
    meta = {'label_map': dict(crf_layout.LABEL_TO_INDEX)}
    assert check_declaration(meta, v4, 'layout') == 'inferred'


def test_undeclared_model_with_another_label_map_is_refused(v4: LabelSet) -> None:
    meta = {'label_map': {'Description': 0, 'Other': 1}}
    with pytest.raises(LabelSetMismatch):
        check_declaration(meta, v4, 'layout')


def test_undeclared_model_without_label_map_is_refused(v4: LabelSet) -> None:
    with pytest.raises(LabelSetMismatch):
        check_declaration({}, v4, 'layout')


def test_undeclared_model_is_not_inferred_for_another_set(v4: LabelSet) -> None:
    data = _v4_dict()
    data['version'] = 'v6-test'
    other = LabelSet.from_dict(data)
    meta = {'label_map': dict(crf_layout.LABEL_TO_INDEX)}
    with pytest.raises(LabelSetMismatch):
        check_declaration(meta, other, 'layout')
