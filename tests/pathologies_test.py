"""Regression tests for detector suite against real treatment content.

Fixture-based: loads ``tests/fixtures/pathologies.json`` and runs
every detector against every entry.  Poster-children (§0.5 in
the data-quality memo) must fire NO flags.  Pathology entries
must fire exactly their ``expected_flags`` set — no more, no less.

The fixture is the machine-readable pathology catalog.  See the
narrative in ``docs/data_quality_production_v4_model.md`` and the
maintenance conventions in ``tests/fixtures/README.md``.

Deliberately hermetic: no CouchDB access, no gnfinder/gnparser
HTTP.  The ``authored_binomial_in_desc`` value stored in each
fixture entry is the labelled ground-truth baseline (captured
from live services at extraction time); this test uses that
labelled value rather than re-calling the network.  Drift
between labelled and live values is handled by the fixture
maintenance script, not by CI tests.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from treatments_to_structured.merge_metric import treatment_merge_metric
from treatments_to_structured.triage_signals import (
    predicted_issues,
    treatment_signals,
)


FIXTURE_PATH = (
    Path(__file__).parent / 'fixtures' / 'pathologies.json'
)


def _load_fixture() -> Dict[str, Any]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


_FIXTURE = _load_fixture()


def _actual_flag_prefixes(entry: Dict[str, Any]) -> Set[str]:
    """Run every detector against the fixture entry and return
    the actual flag PREFIXES that fire.

    Prefix comparison strips the ``§6:merge_metric=<N>`` suffix so
    the test is stable across metric-value fluctuations — the
    fixture stores the prefix ``§6:merge_metric`` regardless of the
    exact value.
    """
    treatment = {
        'description': entry.get('description', ''),
        'diagnosis': entry.get('diagnosis', ''),
        'synthetic_nomenclature':
            entry.get('synthetic_nomenclature', False),
        # Optional detector 1 input — most fixture entries omit
        # ``description_spans`` and read as an empty list, so
        # count_description_span_gaps returns 0 by default.
        # Populated for taxon_adcb2fcc's §12 fragments case.
        'description_spans':
            entry.get('description_spans', []),
    }
    signals = treatment_signals(
        treatment,
        authored_binomial_in_desc=entry.get(
            'authored_binomial_in_desc', False,
        ),
    )
    merge_metric = treatment_merge_metric(treatment)
    flags_str = predicted_issues(signals, merge_metric)
    if not flags_str:
        return set()
    return {flag.split('=')[0] for flag in flags_str.split('|')}


# ---------------------------------------------------------------------------
# Fixture-integrity tests — catch schema drift before real tests run.
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    def test_schema_version(self) -> None:
        assert _FIXTURE.get('_schema_version') == 1

    def test_has_poster_children_and_pathologies(self) -> None:
        assert isinstance(_FIXTURE.get('poster_children'), list)
        assert isinstance(_FIXTURE.get('pathologies'), list)
        # Sanity floor — if either drops below expected, someone
        # deleted content without justification.
        assert len(_FIXTURE['poster_children']) >= 7
        assert len(_FIXTURE['pathologies']) >= 20

    def test_every_entry_has_required_fields(self) -> None:
        required = {
            'id', 'source_experiment', 'source_db',
            'captured_at', 'captured_rev', 'class',
            'description', 'diagnosis',
            'synthetic_nomenclature',
            'authored_binomial_in_desc',
            'expected_flags', 'known_missed_flags', 'notes',
        }
        for entry in (_FIXTURE['poster_children']
                      + _FIXTURE['pathologies']):
            missing = required - set(entry.keys())
            assert not missing, (
                f'{entry.get("id", "<no-id>")}: missing '
                f'required fixture fields: {missing}'
            )

    def test_ids_unique_across_sections(self) -> None:
        ids: List[str] = []
        for section in ('poster_children', 'pathologies'):
            for entry in _FIXTURE[section]:
                ids.append(entry['id'])
        assert len(ids) == len(set(ids)), (
            'Duplicate treatment ID in fixture'
        )


# ---------------------------------------------------------------------------
# Poster-children — must fire ZERO flags.  Any regression here means
# the detector suite has become too aggressive against clean single-
# species content.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'entry',
    _FIXTURE['poster_children'],
    ids=[e['id'][:22] for e in _FIXTURE['poster_children']],
)
def test_poster_child_fires_no_flags(entry: Dict[str, Any]) -> None:
    """§0.5 poster-children exemplify clean single-species treatments
    that MUST NOT fire any flags.  Any regression breaks the
    contract that detector tightening does not push legitimate
    treatments into failure classifications."""
    actual = _actual_flag_prefixes(entry)
    assert actual == set(), (
        f"{entry['id'][:22]}... ({entry['class']}) fired flags: "
        f"{sorted(actual)}\n"
        f"Poster-children must remain unflagged.  See §0.5 in "
        f"docs/data_quality_production_v4_model.md."
    )


# ---------------------------------------------------------------------------
# Pathologies — must fire exactly the labelled ``expected_flags`` set.
# ``known_missed_flags`` names detectors that SHOULD catch the case but
# currently don't (informational; not asserted).  Anything else that
# fires unexpectedly is a regression (or an FP class worth
# investigating).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'entry',
    _FIXTURE['pathologies'],
    ids=[e['id'][:22] for e in _FIXTURE['pathologies']],
)
def test_pathology_expected_flags(entry: Dict[str, Any]) -> None:
    """Each pathology fires exactly ``expected_flags``.  Extra
    flags → potential regression / new signal.  Missing flags →
    detector broke.  Both fail the test to force review."""
    actual = _actual_flag_prefixes(entry)
    expected = set(entry.get('expected_flags', []))
    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"{entry['id'][:22]}... ({entry['class']}) missed "
        f"expected flags: {sorted(missing)}\n"
        f"Detector regression: this treatment used to fire these."
    )
    assert not extra, (
        f"{entry['id'][:22]}... ({entry['class']}) fired "
        f"unexpected flags: {sorted(extra)}\n"
        f"Either a new true positive (update fixture) or a new "
        f"false positive (investigate).  Notes: {entry['notes']}"
    )
